//! Whisper inference: WAV input, word-level timestamp extraction, and progress
//! reporting.
//!
//! Word timing uses whisper.cpp's built-in word alignment
//! (`token_timestamps` + `split_on_word`); whisper.cpp 1.8 (as vendored by
//! `whisper-rs-sys`) no longer ships the experimental DTW model loader, so
//! full DTW is not available at this whisper.cpp revision.
//!
//! Audio is still cut into short windows before inference: the built-in
//! alignment is anchored to the decoded segment and drifts over long windows.
//! Measured on a 25 s music-heavy clip, one window placed the closing line
//! 0.8 s late, while windows capped at [`MAX_WINDOW_MS`] and cut at the
//! quietest frame near the cap matched the speech within ~0.1 s.
//!
//! Inference is blocking C code and runs on `tokio::task::spawn_blocking`; the
//! returned future cannot be cancelled mid-window (whisper.cpp finishes the
//! current window), but dropping it never leaks processes or memory.
//!
//! # Progress channel semantics
//!
//! Callbacks handed to [`sublayer_whisper::FullParams`] are plain borrows, so
//! the `Sender` clone inside the progress callback is dropped as soon as
//! transcription returns and the channel closes normally. An explicit final
//! tick of `1.0` is still sent as a completion marker for callers that batch
//! updates and do not want to distinguish close from completion.

use std::path::Path;

use hound::{SampleFormat, WavReader};
use sublayer_core::WordToken;
use sublayer_whisper::{
    AlignmentHeads, ContextParams, FullParams, SamplingStrategy, WhisperContext, WhisperState,
};
use tokio::sync::mpsc::Sender;

use crate::AiError;
use crate::vad::{VadConfig, detect_speech};

/// Sample rate required by Whisper's acoustic model.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Options for a transcription run.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriberConfig {
    /// Path to a GGML Whisper model, as produced by
    /// [`ModelManager::ensure_cached`](crate::ModelManager::ensure_cached).
    pub model_path: std::path::PathBuf,
    /// Spoken language (`"en"`, `"de"`, ...); `None` lets Whisper detect it.
    pub language: Option<String>,
    /// Run the energy VAD first and transcribe speech windows only.
    pub enable_vad: bool,
    /// Request the GPU backend; falls back to CPU when whisper.cpp was built
    /// without the `vulkan` feature or no GPU backend exists.
    pub use_gpu: bool,
    /// Worker threads for whisper.cpp.
    pub threads: usize,
}

impl Default for TranscriberConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::new(),
            language: None,
            enable_vad: true,
            use_gpu: false,
            threads: std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(4),
        }
    }
}

/// Transcribes `wav_path` (16 kHz mono 16-bit PCM) into word tokens.
///
/// Progress in `0.0..=1.0` is pushed to `progress_tx`; the channel closes when
/// transcription completes and a final `1.0` marker is sent. Dropping the
/// receiver simply stops progress updates.
pub async fn transcribe_audio(
    wav_path: &Path,
    config: &TranscriberConfig,
    progress_tx: Sender<f32>,
) -> Result<Vec<WordToken>, AiError> {
    let wav_path = wav_path.to_owned();
    let config = config.clone();
    let task =
        tokio::task::spawn_blocking(move || transcribe_blocking(&wav_path, &config, &progress_tx));
    task.await
        .map_err(|error| AiError::TaskJoin(error.to_string()))?
}

/// Longest window handed to whisper.cpp in a single call.
///
/// The token-level alignment is anchored to the decoded segment, so its error
/// grows with the window length; cutting long input keeps every word close to
/// the speech that produced it.
const MAX_WINDOW_MS: u64 = 10_000;

/// Trailing zone of a full window searched for a quiet cut point.
const CUT_SEARCH_MS: u64 = 2_000;

/// RMS analysis frame used to pick a cut point.
const CUT_FRAME_MS: u64 = 30;

/// One transcription window: where it starts on the source timeline and which
/// samples it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WhisperWindow {
    /// Milliseconds from the start of the source media.
    offset_ms: u64,
    /// First sample of the window.
    start: usize,
    /// One-past-last sample of the window.
    end: usize,
}

/// Windows whose concatenation covers `[start, end)`.
///
/// Every window is at most [`MAX_WINDOW_MS`] long; full windows are cut at
/// the quietest [`CUT_FRAME_MS`] frame of their trailing
/// [`CUT_SEARCH_MS`], which lands between words instead of inside them.
fn whisper_windows(
    samples: &[f32],
    sample_rate: u32,
    start: usize,
    end: usize,
) -> Vec<WhisperWindow> {
    let rate = u64::from(sample_rate);
    if rate == 0 || start >= end || end > samples.len() {
        return Vec::new();
    }
    let max_samples = (rate * MAX_WINDOW_MS / 1_000) as usize;
    let search = (rate * CUT_SEARCH_MS / 1_000).max(1) as usize;
    let frame = (rate * CUT_FRAME_MS / 1_000).max(1) as usize;

    let mut windows = Vec::new();
    let mut from = start;
    while from < end {
        let limit = from.saturating_add(max_samples).min(end);
        let cut = if limit == end {
            end
        } else {
            quietest_cut(
                samples,
                limit.saturating_sub(search).max(from),
                limit,
                frame,
            )
            .map(|cut| cut.max(from + frame).min(limit))
            .unwrap_or(limit)
        };
        windows.push(WhisperWindow {
            offset_ms: from as u64 * 1_000 / rate,
            start: from,
            end: cut,
        });
        from = cut;
    }
    windows
}

/// End of the quietest `frame`-long analysis window in `[from, limit)`.
///
/// A frame inside a word is louder than the pause before or after it, so the
/// minimum tends to be silence between words.
fn quietest_cut(samples: &[f32], from: usize, limit: usize, frame: usize) -> Option<usize> {
    let mut index = from;
    let mut best: Option<(f64, usize)> = None;
    while index < limit {
        let end = (index + frame).min(limit);
        if end <= index {
            break;
        }
        let energy: f64 = samples[index..end]
            .iter()
            .map(|sample| f64::from(*sample) * f64::from(*sample))
            .sum();
        let energy = energy / (end - index) as f64;
        if best.is_none_or(|(best_energy, _)| energy < best_energy) {
            best = Some((energy, end));
        }
        index += frame;
    }
    best.map(|(_, end)| end)
}

/// Splits the track into the windows that will be fed to whisper.cpp.
///
/// With the VAD enabled the spans it detects are used and capped; otherwise
/// the whole track is chunked.
fn speech_windows(samples: &[f32], enable_vad: bool) -> Vec<WhisperWindow> {
    if !enable_vad {
        return whisper_windows(samples, WHISPER_SAMPLE_RATE, 0, samples.len());
    }
    detect_speech(samples, WHISPER_SAMPLE_RATE, &VadConfig::default())
        .iter()
        .flat_map(|segment| {
            let start = segment.start_sample(WHISPER_SAMPLE_RATE);
            let end = segment.end_sample(WHISPER_SAMPLE_RATE).min(samples.len());
            whisper_windows(samples, WHISPER_SAMPLE_RATE, start, end)
        })
        .collect()
}

fn transcribe_blocking(
    wav_path: &Path,
    config: &TranscriberConfig,
    progress_tx: &Sender<f32>,
) -> Result<Vec<WordToken>, AiError> {
    // Route whisper.cpp/ggml chatter into `tracing` instead of stderr, so the
    // CLI progress bar and the studio status line stay readable.
    sublayer_whisper::install_log_handler();

    let samples = read_16k_mono_wav(wav_path)?;

    let mut context_params = ContextParams::default();
    context_params.use_gpu(config.use_gpu);
    // DTW word alignment: with the pinned models the right alignment-head
    // preset is known from the file name; custom models fall back to the
    // attention-based timestamps.
    context_params.dtw_timestamps(AlignmentHeads::for_model(&config.model_path));
    let context = WhisperContext::new_with_params(&config.model_path, context_params)?;

    let mut words = Vec::new();
    let windows = speech_windows(&samples, config.enable_vad);
    let window_count = windows.len().max(1) as f32;
    for (index, window) in windows.iter().enumerate() {
        let mut params = make_params(config);
        let sender = progress_tx.clone();
        let window_index = index as f32;
        let mut progress = move |percent: i32| {
            let value = (window_index + percent as f32 / 100.0) / window_count;
            let _ = sender.try_send(value);
        };
        params.set_progress_callback(&mut progress);

        let mut state = context.create_state()?;
        state.full(&mut params, &samples[window.start..window.end])?;
        // `params` (and with it the progress callback) is dropped here, so
        // the sender clone dies and the channel closes.
        append_words(&state, &mut words, window.offset_ms);
    }

    normalize_word_times(&mut words);

    let _ = progress_tx.try_send(1.0);
    Ok(words)
}

/// Builds the decoding parameters shared by every window.
fn make_params<'a>(config: &TranscriberConfig) -> FullParams<'a> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(config.threads.clamp(1, 64) as i32);
    params.set_language(config.language.as_deref());
    // Word-level timestamps via whisper.cpp's token alignment.
    params.set_token_timestamps(true);
    params.set_split_on_word(true);
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_no_speech_thold(0.6);
    params.set_suppress_blank(true);
    params
}

/// Segments whose no-speech probability is above this are dropped: whisper.cpp
/// assigns a high no-speech score to silence and music windows it decoded, and
/// their text is usually hallucinated. (`whisper_full_params.no_speech_thold`
/// itself is not implemented in whisper.cpp, so the filtering happens here.)
const NO_SPEECH_FILTER_THRESHOLD: f32 = 0.9;

/// Floor for a word's end time when whisper.cpp left it degenerate.
///
/// DTW stamps token *starts*; on music-heavy audio the attention-based ends
/// collapse onto the starts, which would otherwise shatter the caption cards
/// into invisible zero-duration flashes.
const MIN_WORD_DURATION_MS: u64 = 60;

/// Extends words whose end collapsed onto (or before) their start, and words
/// shorter than the visibility floor.
///
/// Whisper's DTW stamps token *starts*; on music-heavy audio the attention-
/// based ends collapse onto the starts, which would otherwise shatter the
/// caption cards into invisible zero-duration flashes. Such a word is
/// stretched to the next word's start — merging consecutive degenerate stamps
/// back into one caption — and everything shorter than
/// [`MIN_WORD_DURATION_MS`] gets that floor. Real pauses are untouched: words
/// with a sane end still end before the next word starts.
fn normalize_word_times(words: &mut [WordToken]) {
    for index in 0..words.len() {
        let start = words[index].start_ms;
        let next = words.get(index + 1).map(|word| word.start_ms);
        let spoken = words[index].end_ms;
        let end = if spoken > start && spoken < next.unwrap_or(u64::MAX) {
            spoken
        } else {
            next.unwrap_or(start)
        };
        words[index].end_ms = end.max(start.saturating_add(MIN_WORD_DURATION_MS));
    }
}

/// Collects the window's tokens into `words`, shifting timestamps by
/// `offset_ms` (the window's position in the source audio).
fn append_words(state: &WhisperState<'_>, words: &mut Vec<WordToken>, offset_ms: u64) {
    let tokens: Vec<(String, i64, i64)> = state
        .segments()
        .filter(|segment| segment.no_speech_probability() <= NO_SPEECH_FILTER_THRESHOLD)
        .flat_map(|segment| {
            segment
                .tokens()
                .map(|token| (token.text_lossy(), token.start_ms(), token.end_ms()))
        })
        .collect();

    for word in tokens_to_words(&tokens) {
        words.push(WordToken::new(
            word.text,
            word.start_ms.saturating_add(offset_ms),
            word.end_ms.saturating_add(offset_ms),
        ));
    }
}

/// Merges a Whisper token stream into words.
///
/// Whisper's tokenizer splits words at leading spaces; a token beginning with
/// a space starts the next word. Bracket tokens (`[BLANK_AUDIO]`, ...) carry no
/// caption text and are dropped.
pub(crate) fn tokens_to_words(tokens: &[(String, i64, i64)]) -> Vec<WordToken> {
    let mut words: Vec<WordToken> = Vec::new();
    let mut parts: Vec<String> = Vec::new();
    let mut start_ms = 0_i64;
    let mut end_ms = 0_i64;

    for (text, token_start, token_end) in tokens {
        if text.is_empty() || is_special_token(text.trim()) {
            continue;
        }
        if text.starts_with(' ') && !parts.is_empty() {
            flush_word(&mut words, &mut parts, start_ms, end_ms);
        }
        if parts.is_empty() {
            start_ms = *token_start;
            end_ms = *token_end;
        } else {
            start_ms = start_ms.min(*token_start);
            end_ms = end_ms.max(*token_end);
        }
        parts.push(text.clone());
    }
    flush_word(&mut words, &mut parts, start_ms, end_ms);
    words
}

/// Pushes the accumulated word, clearing `parts`.
fn flush_word(words: &mut Vec<WordToken>, parts: &mut Vec<String>, start_ms: i64, end_ms: i64) {
    if parts.is_empty() {
        return;
    }
    let text = parts.join("").trim().to_owned();
    parts.clear();
    // whisper splits bracket tokens like `[BELL]` into several tokenizer
    // pieces (`" ["`, `"Bell"`, `"]"`), so specials must also be filtered at
    // the assembled-word level.
    if text.is_empty() || is_special_token(&text) {
        return;
    }
    words.push(WordToken::new(
        text,
        start_ms.max(0) as u64,
        end_ms.max(0) as u64,
    ));
}

/// Whether `text` is a whisper special token such as `[BLANK_AUDIO]`.
///
/// The tokenizer sometimes prefixes specials with a space (` "[BELL]"`); the
/// trim keeps those from leaking into captions as words like `[BELL]`.
fn is_special_token(text: &str) -> bool {
    text.starts_with('[') && text.ends_with(']')
}

/// Reads a 16 kHz mono 16-bit (or 32-bit float) PCM WAV into normalized
/// `f32` samples.
fn read_16k_mono_wav(path: &Path) -> Result<Vec<f32>, AiError> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.sample_rate != WHISPER_SAMPLE_RATE {
        return Err(AiError::UnsupportedWav(format!(
            "expected {WHISPER_SAMPLE_RATE} Hz, got {} Hz",
            spec.sample_rate
        )));
    }
    if spec.channels != 1 {
        return Err(AiError::UnsupportedWav(format!(
            "expected mono audio, got {} channels",
            spec.channels
        )));
    }
    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| {
                sample
                    .map(|value| f32::from(value) / 32_768.0)
                    .map_err(AiError::from)
            })
            .collect(),
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|sample| sample.map_err(AiError::from))
            .collect(),
        (format, bits) => Err(AiError::UnsupportedWav(format!(
            "expected 16-bit int or 32-bit float PCM, got {format:?} with {bits} bits"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{WavSpec, WavWriter};
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn write_wav(path: &Path, sample_rate: u32, channels: u16, samples: &[f32]) {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        for sample in samples {
            writer
                .write_sample((sample * f32::from(i16::MAX)).round() as i16)
                .unwrap();
        }
        writer.finalize().unwrap();
    }

    // ------------------------------------------------------------ wav reading

    #[test]
    fn reads_16k_mono_pcm_into_normalized_samples() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("speech.wav");
        let samples = vec![0.5_f32, -0.5, 0.25, -0.25];
        write_wav(&path, WHISPER_SAMPLE_RATE, 1, &samples);

        let decoded = read_16k_mono_wav(&path).unwrap();

        assert_eq!(decoded.len(), 4);
        for (got, expected) in decoded.iter().zip(&samples) {
            assert!(
                (got - expected).abs() < 0.001,
                "got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn rejects_wrong_sample_rates_and_channels() {
        let dir = TempDir::new().unwrap();

        let stereo = dir.path().join("stereo.wav");
        write_wav(&stereo, WHISPER_SAMPLE_RATE, 2, &[0.1, 0.1]);
        assert!(matches!(
            read_16k_mono_wav(&stereo),
            Err(AiError::UnsupportedWav(_))
        ));

        let slow = dir.path().join("slow.wav");
        write_wav(&slow, 8_000, 1, &[0.1]);
        assert!(matches!(
            read_16k_mono_wav(&slow),
            Err(AiError::UnsupportedWav(_))
        ));
    }

    #[test]
    fn rejects_exotic_sample_formats() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("exotic.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: WHISPER_SAMPLE_RATE,
            bits_per_sample: 24,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(&path, spec).unwrap();
        writer.write_sample(0_i32).unwrap();
        writer.finalize().unwrap();

        assert!(matches!(
            read_16k_mono_wav(&path),
            Err(AiError::UnsupportedWav(_))
        ));
    }

    // ---------------------------------------------------------- token merging

    #[test]
    fn merges_tokens_into_words_on_leading_spaces() {
        let tokens = vec![
            (" hello".to_owned(), 0, 300),
            (" world".to_owned(), 300, 700),
        ];
        let words = tokens_to_words(&tokens);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[0].start_ms, 0);
        assert_eq!(words[0].end_ms, 300);
        assert_eq!(words[1].text, "world");
        assert_eq!(words[1].start_ms, 300);
        assert_eq!(words[1].end_ms, 700);
    }

    #[test]
    fn attaches_punctuation_to_the_preceding_word() {
        let tokens = vec![
            (" hello".to_owned(), 0, 200),
            (",".to_owned(), 200, 250),
            (" world".to_owned(), 250, 600),
            ("?".to_owned(), 600, 650),
        ];
        let words = tokens_to_words(&tokens);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello,");
        assert_eq!(words[1].text, "world?");
    }

    #[test]
    fn drops_special_and_empty_tokens() {
        let tokens = vec![
            (" hello".to_owned(), 0, 100),
            ("[BLANK_AUDIO]".to_owned(), 100, 100),
            ("".to_owned(), 100, 100),
            (" world".to_owned(), 200, 400),
        ];
        let words = tokens_to_words(&tokens);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
        assert_eq!(words[1].start_ms, 200);
    }

    #[test]
    fn drops_special_tokens_prefixed_with_a_space() {
        // whisper may emit specials as ` [BELL]`; regression: they used to
        // leak through as caption words on tone-only clips.
        let tokens = vec![
            (" hello".to_owned(), 0, 100),
            (" [BELL]".to_owned(), 150, 150),
            (" world".to_owned(), 200, 400),
        ];
        let words = tokens_to_words(&tokens);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
        assert_eq!(words[1].start_ms, 200);
    }

    #[test]
    fn drops_special_tokens_split_across_tokenizer_pieces() {
        // whisper emits `[BELL]` as three pieces (`" ["`, `"Bell"`, `"]"`)
        // that only form the special at the word level; regression: it used
        // to leak through as a caption word on tone-only clips.
        let tokens = vec![
            (" hello".to_owned(), 0, 100),
            (" [".to_owned(), 150, 150),
            ("Bell".to_owned(), 150, 150),
            ("]".to_owned(), 150, 150),
            (" world".to_owned(), 200, 400),
        ];
        let words = tokens_to_words(&tokens);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
        assert_eq!(words[1].start_ms, 200);
    }

    #[test]
    fn degenerate_word_ends_are_stretched_to_the_next_word() {
        let mut words = vec![
            WordToken::new("this", 1_000, 1_000),
            WordToken::new("is", 1_200, 1_200),
            WordToken::new("so", 1_500, 1_500),
            WordToken::new("long", 1_900, 2_400),
        ];
        normalize_word_times(&mut words);

        assert_eq!(words[0].end_ms, 1_200, "merges into the next word");
        assert_eq!(words[1].end_ms, 1_500);
        assert_eq!(words[2].end_ms, 1_900);
        assert_eq!(words[3].end_ms, 2_400, "a sane end is untouched");
    }

    #[test]
    fn the_last_degenerate_word_gets_a_visibility_floor() {
        let mut words = vec![WordToken::new("oh", 5_000, 5_000)];
        normalize_word_times(&mut words);
        assert_eq!(words[0].end_ms, 5_000 + MIN_WORD_DURATION_MS);
    }

    #[test]
    fn words_shorter_than_the_floor_reach_it_without_eating_the_pause() {
        let mut words = vec![
            WordToken::new("a", 1_000, 1_010),
            WordToken::new("pause", 1_400, 1_800),
        ];
        normalize_word_times(&mut words);
        assert_eq!(words[0].end_ms, 1_000 + MIN_WORD_DURATION_MS);
        assert_eq!(words[0].end_ms, 1_060);
        // The 340 ms pause to the next word survives the floor.
        assert_eq!(words[1].start_ms - words[0].end_ms, 340);
        assert_eq!(words[1].end_ms, 1_800);
    }

    #[test]
    fn sane_word_runs_are_untouched() {
        let mut words = vec![
            WordToken::new("one", 0, 300),
            WordToken::new("two", 700, 900),
        ];
        normalize_word_times(&mut words);
        assert_eq!(words[0].end_ms, 300);
        assert_eq!(words[1].end_ms, 900);
    }

    #[test]
    fn merges_word_fragments_across_tokens() {
        let tokens = vec![
            (" hell".to_owned(), 0, 150),
            ("o".to_owned(), 150, 300),
            (" world".to_owned(), 300, 500),
        ];
        let words = tokens_to_words(&tokens);

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[0].start_ms, 0);
        assert_eq!(words[0].end_ms, 300);
    }

    #[test]
    fn empty_stream_yields_no_words() {
        assert!(tokens_to_words(&[]).is_empty());
        assert!(tokens_to_words(&[("[BLANK_AUDIO]".to_owned(), 0, 0)]).is_empty());
    }

    #[test]
    fn clamps_negative_timestamps_to_zero() {
        let tokens = vec![(" hello".to_owned(), -100, -50)];
        let words = tokens_to_words(&tokens);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].start_ms, 0);
    }

    // --------------------------------------------------- end-to-end (ignored)

    /// Full transcription requires a real GGML model and typically takes a few
    /// seconds. Run manually:
    ///
    /// ```text
    /// curl -L -o /tmp/ggml-tiny.en.bin \
    ///   https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
    /// curl -L -o /tmp/jfk.wav https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav
    /// SUBLAYER_TEST_MODEL=/tmp/ggml-tiny.en.bin SUBLAYER_TEST_AUDIO=/tmp/jfk.wav \
    ///   cargo test -p sublayer-ai --release -- --ignored transcribe_end_to_end
    /// ```
    ///
    /// `SUBLAYER_TEST_AUDIO` is optional; a synthetic tone is used when absent,
    /// which only proves the pipeline completes. `SUBLAYER_TEST_VAD=0` skips
    /// the voice activity windows and transcribes the whole clip at once.
    #[ignore]
    #[tokio::test]
    async fn transcribe_end_to_end_with_model() {
        let model = std::env::var("SUBLAYER_TEST_MODEL")
            .unwrap_or_else(|_| "/tmp/ggml-tiny.en.bin".to_owned());
        if !Path::new(&model).is_file() {
            eprintln!("skipping: set SUBLAYER_TEST_MODEL to a whisper ggml model");
            return;
        }

        let dir = TempDir::new().unwrap();
        let (wav, real_speech) = match std::env::var("SUBLAYER_TEST_AUDIO") {
            Ok(audio) if Path::new(&audio).is_file() => (PathBuf::from(audio), true),
            _ => {
                let synthetic = dir.path().join("speech.wav");
                write_wav(&synthetic, WHISPER_SAMPLE_RATE, 1, &tone_samples(32_000));
                (synthetic, false)
            }
        };

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let vad = std::env::var("SUBLAYER_TEST_VAD")
            .map(|value| value == "1")
            .unwrap_or(true);
        let config = TranscriberConfig {
            model_path: model.into(),
            language: Some("en".to_owned()),
            enable_vad: vad,
            use_gpu: false,
            threads: 2,
        };
        let task = {
            let wav = wav.clone();
            let config = config.clone();
            tokio::spawn(async move { transcribe_audio(&wav, &config, tx).await })
        };

        let mut last_progress = 0.0_f32;
        // The channel closes on its own once transcription returns (callbacks
        // are borrowed, not leaked); the `1.0` marker just makes the loop end
        // deterministically before awaiting the task.
        while let Some(progress) = rx.recv().await {
            last_progress = progress.max(last_progress);
            if progress >= 1.0 {
                break;
            }
        }
        let words = task.await.unwrap().unwrap();

        eprintln!("transcribed {} words", words.len());
        assert!(
            (0.0..=1.0).contains(&last_progress),
            "progress out of range: {last_progress}"
        );
        if real_speech {
            assert!(!words.is_empty(), "real speech must produce words");
        }
        for word in &words {
            assert!(!word.text.is_empty());
            assert!(word.end_ms >= word.start_ms);
        }
    }

    // ------------------------------------------------------------ window plan

    #[test]
    fn windows_cover_the_audio_in_bounded_chunks() {
        let samples = tone_samples(40 * WHISPER_SAMPLE_RATE as usize);
        let windows = whisper_windows(&samples, WHISPER_SAMPLE_RATE, 0, samples.len());

        assert!(windows.len() >= 4, "40 s must split, got {windows:?}");
        assert_eq!(windows.first().unwrap().start, 0);
        assert_eq!(windows.last().unwrap().end, samples.len());
        for pair in windows.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "windows must be contiguous");
        }
        let max = MAX_WINDOW_MS as usize * WHISPER_SAMPLE_RATE as usize / 1_000;
        for window in &windows {
            assert!(
                window.end - window.start <= max,
                "{window:?} exceeds the cap"
            );
        }
    }

    #[test]
    fn windows_cut_at_the_quietest_frame_near_the_limit() {
        // A loud tone with a silent notch at 9 s: the first cut belongs there
        // instead of mid-tone at the 10 s cap.
        let mut samples = vec![0.5_f32; 20 * WHISPER_SAMPLE_RATE as usize];
        let quiet = 9 * WHISPER_SAMPLE_RATE as usize;
        samples[quiet..quiet + 200].fill(0.0);

        let windows = whisper_windows(&samples, WHISPER_SAMPLE_RATE, 0, samples.len());
        let first = windows[0];

        assert!(
            first.end >= quiet && first.end <= quiet + 200 + 480,
            "cut {} outside the notch at {quiet}",
            first.end
        );
    }

    #[test]
    fn window_offsets_track_the_source_timeline() {
        let samples = tone_samples(20 * WHISPER_SAMPLE_RATE as usize);
        let windows = whisper_windows(&samples, WHISPER_SAMPLE_RATE, 0, samples.len());
        let second = windows[1];

        let expected = (second.start as u64 * 1_000 / u64::from(WHISPER_SAMPLE_RATE)) as i64;
        assert!((second.offset_ms as i64 - expected).abs() <= 1);
        assert!(second.offset_ms >= 8_000, "offset {}", second.offset_ms);
    }

    #[test]
    fn degenerate_ranges_produce_no_windows() {
        let samples = tone_samples(1_000);
        assert!(whisper_windows(&samples, WHISPER_SAMPLE_RATE, 0, 0).is_empty());
        assert!(whisper_windows(&samples, WHISPER_SAMPLE_RATE, 500, 500).is_empty());
        assert!(whisper_windows(&samples, WHISPER_SAMPLE_RATE, 800, 500).is_empty());
        assert!(whisper_windows(&samples, WHISPER_SAMPLE_RATE, 0, 2_000).is_empty());
        assert!(whisper_windows(&samples, 0, 0, 1_000).is_empty());
    }

    fn tone_samples(count: usize) -> Vec<f32> {
        (0..count)
            .map(|index| 0.05 * (index as f32 / 32.0).sin())
            .collect()
    }
}
