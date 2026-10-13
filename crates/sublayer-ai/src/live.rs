//! Live (streaming) Whisper transcription.
//!
//! Whisper is not an incremental decoder: it transcribes whatever audio it is
//! given in one pass, so "live" here means decoding a *rolling window* over a
//! growing buffer as the audio arrives (the same strategy as whisper.cpp's
//! `stream` example). A few design points:
//!
//! - **Contiguous windows, no overlap.** Each decode consumes exactly one
//!   window of previously unprocessed audio, so words are emitted exactly
//!   once and never re-stamped by a later decode (an overlapping scheme would
//!   both duplicate and jitter boundary words).
//! - **VAD gating.** The energy VAD decides when speech is in flight; silence
//!   is never handed to whisper (it wastes CPU and hallucinates) and is
//!   dropped as it arrives, so quiet stretches do not grow the buffer. Gaps
//!   shorter than the VAD bridge keep the window open; the tail after a real
//!   silence is flushed as one final short window or dropped when it is too
//!   short to be intelligible.
//! - **Media-timeline timestamps.** Word times are absolute positions in the
//!   media (not relative to decode time), so the caller can lay them out
//!   against the playhead. `reset` re-anchors the timeline for seeks.
//! - **Attention-based timestamps only.** DTW alignment is skipped to keep
//!   windows cheap; whisper.cpp's built-in token alignment (`token_timestamps`
//!   + `split_on_word`) is accurate enough for caption-sized windows.
//!
//! Inference is blocking C code behind a plainly non-`Send` whisper context,
//! so [`LiveCaptioner`] runs it on a dedicated OS thread and talks to the
//! caller over channels. The sync [`LiveTranscriber`] is the testable core:
//! it decodes through a [`Decoder`] seam so the window/VAD state machine can
//! be unit-tested without a model.

use std::path::PathBuf;

use sublayer_core::WordToken;
use sublayer_whisper::{ContextParams, WhisperContext};
use tokio::sync::mpsc;

use crate::AiError;
use crate::transcriber::{WHISPER_SAMPLE_RATE, append_words, make_params};
use crate::vad::{VadConfig, detect_speech};

/// Shortest window handed to whisper; anything below is mostly silence inside
/// a sentence and decodes poorly.
const MIN_WINDOW_MS: u64 = 500;
/// Longest window; clips larger than this stay aligned only with DTW, which
/// live mode deliberately skips.
const MAX_WINDOW_MS: u64 = 30_000;
/// A silent run at least this long ends the current speech span (mirrors
/// `VadConfig::min_silence_ms`).
const VAD_SILENCE_MS: u64 = 350;
/// The sample buffer is drained once this many samples lie behind the decode
/// frontier, keeping memory bounded on long recordings.
const COMPACT_THRESHOLD_SAMPLES: usize = 16 * 16_000;

/// Options for a live transcription session.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveTranscriberConfig {
    /// Path to a GGML Whisper model, as produced by
    /// [`ModelManager::ensure_cached`](crate::ModelManager::ensure_cached).
    pub model_path: PathBuf,
    /// Spoken language (`"en"`, `"de"`, ...); `None` lets Whisper detect it.
    pub language: Option<String>,
    /// Request the GPU backend; falls back to CPU when whisper.cpp was built
    /// without the `vulkan` feature or no GPU backend exists.
    pub use_gpu: bool,
    /// Worker threads for whisper.cpp.
    pub threads: usize,
    /// Gate windows on the energy VAD; `false` decodes continuously (latency
    /// drops, CPU cost and hallucination risk rise).
    pub enable_vad: bool,
    /// Rolling window length in milliseconds; clamped to
    /// [`MIN_WINDOW_MS`]..=[`MAX_WINDOW_MS`] (default 4 000).
    pub window_ms: u64,
    /// Floor for the tail flushed when speech ends, in milliseconds
    /// (default 200, capped at `window_ms`).
    pub min_flush_ms: u64,
}

impl Default for LiveTranscriberConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            language: None,
            use_gpu: false,
            threads: std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(4),
            enable_vad: true,
            window_ms: 4_000,
            min_flush_ms: 200,
        }
    }
}

/// Decodes a PCM window into words on the media timeline.
///
/// A seam around whisper so the live state machine (windows, VAD gating, tail
/// flushing) is unit-testable without a model download.
trait Decoder {
    /// Transcribes `samples` (16 kHz mono) whose first sample sits at
    /// `offset_ms` on the media timeline.
    fn decode(&mut self, samples: &[f32], offset_ms: u64) -> Result<Vec<WordToken>, AiError>;

    /// Language whisper detected for the decoded audio, when auto-detection
    /// is on (`language: None`); `None` for pinned-language decoders and for
    /// decoders that never ran. Clears once read.
    fn take_detected_language(&mut self) -> Option<String> {
        None
    }
}

/// [`Decoder`] backed by a whisper.cpp context.
struct WhisperDecoder {
    context: WhisperContext,
    language: Option<String>,
    threads: usize,
    detected: Option<String>,
}

impl WhisperDecoder {
    fn new(config: &LiveTranscriberConfig) -> Result<Self, AiError> {
        let mut context_params = ContextParams::default();
        context_params.use_gpu(config.use_gpu);
        let context = WhisperContext::new_with_params(&config.model_path, context_params)?;
        Ok(Self {
            context,
            language: config.language.clone(),
            threads: config.threads,
            detected: None,
        })
    }
}

impl Decoder for WhisperDecoder {
    fn decode(&mut self, samples: &[f32], offset_ms: u64) -> Result<Vec<WordToken>, AiError> {
        let mut params = make_params(self.language.as_deref(), self.threads);
        let mut state = self.context.create_state()?;
        state.full(&mut params, samples)?;
        if self.language.is_none() && self.detected.is_none() {
            // whisper.cpp runs its language detector when no language was
            // pinned; surface the first settled result.
            self.detected = state.detected_language();
        }
        let mut words = Vec::new();
        append_words(&state, &mut words, offset_ms);
        Ok(words)
    }

    fn take_detected_language(&mut self) -> Option<String> {
        self.detected.take()
    }
}

/// Sync core of the live pipeline: buffers audio, decodes rolling windows,
/// and emits finalized words with media-timeline timestamps.
///
/// Feed it with [`push`](Self::push) from one thread, call
/// [`flush`](Self::flush) at stop or pause, and [`reset`](Self::reset) after a
/// seek. Not `Send` (the whisper context is a raw pointer), so it lives on the
/// worker thread inside [`LiveCaptioner`].
pub struct LiveTranscriber {
    config: LiveTranscriberConfig,
    decoder: Box<dyn Decoder>,
    /// Audio received but not yet handed to whisper.
    buffer: Vec<f32>,
    /// Sample index of the decode frontier inside `buffer`.
    consumed: usize,
    /// Media-timeline position of `buffer[0]`.
    media_offset_ms: u64,
    /// Whether the VAD believes speech is in flight.
    speech_active: bool,
    /// Whether a speech span started since the last flush; pure silence never
    /// reaches the decoder.
    saw_speech: bool,
    /// Media-timeline end of the last VAD speech frame.
    last_speech_end_ms: u64,
}

impl LiveTranscriber {
    /// Loads the model from `config.model_path` and starts a live session.
    pub fn new(config: LiveTranscriberConfig) -> Result<Self, AiError> {
        // Route whisper.cpp/ggml chatter into `tracing` (idempotent).
        sublayer_whisper::install_log_handler();
        let decoder = Box::new(WhisperDecoder::new(&config)?);
        Ok(Self::with_decoder(config, decoder))
    }

    fn with_decoder(config: LiveTranscriberConfig, decoder: Box<dyn Decoder>) -> Self {
        let config = sanitize(config);
        Self {
            config,
            decoder,
            buffer: Vec::new(),
            consumed: 0,
            media_offset_ms: 0,
            speech_active: false,
            saw_speech: false,
            last_speech_end_ms: 0,
        }
    }

    /// Appends `samples` (16 kHz mono `f32`) and decodes whatever full windows
    /// became available, returning the newly finalized words.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<WordToken>, AiError> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        self.buffer.extend_from_slice(samples);

        let mut words = Vec::new();
        if self.config.enable_vad {
            if let Some(onset_ms) = self.track_speech() {
                // Speech starts mid-silence: skip the silent prefix so the
                // first decode begins at the voice onset.
                let drop_to = samples_at(onset_ms.saturating_sub(self.media_offset_ms))
                    .min(self.buffer.len());
                self.consumed = self.consumed.max(drop_to);
            }
            if !self.speech_active {
                if self.saw_speech {
                    // A speech span just ended: flush its tail as one window.
                    self.saw_speech = false;
                    words.extend(self.flush_tail()?);
                } else if self.silent_unprocessed_ms() >= VAD_SILENCE_MS {
                    // Certainly silence: never hand it to whisper; drop it as
                    // it arrives so quiet stretches cannot grow the buffer.
                    self.consumed = self.buffer.len();
                }
                // Shorter silent runs stay buffered: a speech span below the
                // VAD blip floor may still be accumulating (the player feeds
                // audio in ~20 ms pieces).
            }
        } else {
            self.speech_active = true;
        }

        let window = self.window_samples();
        while self.speech_active && self.unprocessed_len() >= window {
            let offset = self.media_offset_ms + samples_ms(self.consumed);
            let end = self.consumed + window;
            words.extend(
                self.decoder
                    .decode(&self.buffer[self.consumed..end], offset)?,
            );
            self.consumed = end;
        }
        self.compact();
        Ok(words)
    }

    /// Decodes whatever buffered speech is still unprocessed (for example at
    /// stop or pause). Pure silence is dropped without decoding.
    pub fn flush(&mut self) -> Result<Vec<WordToken>, AiError> {
        let words = if self.saw_speech {
            self.saw_speech = false;
            self.flush_tail()?
        } else {
            self.consumed = self.buffer.len();
            Vec::new()
        };
        self.speech_active = false;
        Ok(words)
    }

    /// Drops all buffered audio and re-anchors the timeline at `offset_ms`
    /// (typically the new playhead after a seek).
    pub fn reset(&mut self, offset_ms: u64) {
        self.buffer.clear();
        self.consumed = 0;
        self.media_offset_ms = offset_ms;
        self.speech_active = false;
        self.saw_speech = false;
        self.last_speech_end_ms = offset_ms;
    }

    /// Milliseconds of the next media sample.
    pub fn position_ms(&self) -> u64 {
        self.media_offset_ms + samples_ms(self.buffer.len())
    }

    /// Language whisper detected for the decoded audio (auto-detect mode
    /// only); `None` when detection has not settled or the language was
    /// pinned. Clears once read.
    pub fn take_detected_language(&mut self) -> Option<String> {
        self.decoder.take_detected_language()
    }

    fn window_samples(&self) -> usize {
        (u64::from(WHISPER_SAMPLE_RATE) * self.config.window_ms / 1_000) as usize
    }

    fn unprocessed_len(&self) -> usize {
        self.buffer.len() - self.consumed
    }

    /// Milliseconds of unprocessed audio with no speech span in it; used to
    /// decide when a silent run is certainly silence.
    fn silent_unprocessed_ms(&self) -> u64 {
        samples_ms(self.buffer.len().saturating_sub(self.consumed))
    }

    /// Updates the VAD state over the *accumulated* unprocessed audio.
    ///
    /// Classifying each incoming chunk alone would drop every span: the
    /// player feeds audio in ~20 ms pieces, all below the 200 ms blip floor
    /// of [`detect_speech`]. Returns the media-timeline start of a speech
    /// span that just became certain, so the caller can skip the silent
    /// prefix before it.
    fn track_speech(&mut self) -> Option<u64> {
        let base = self.media_offset_ms + samples_ms(self.consumed);
        let spans = detect_speech(
            &self.buffer[self.consumed..],
            WHISPER_SAMPLE_RATE,
            &VadConfig::default(),
        );
        let Some(first) = spans.first() else {
            // No speech in the unprocessed part; a run this long without a
            // span ends the speech that just finished (the caller flushes
            // its tail).
            if self.silent_unprocessed_ms() >= VAD_SILENCE_MS {
                self.speech_active = false;
            }
            return None;
        };
        let onset = if self.saw_speech {
            None
        } else {
            self.saw_speech = true;
            Some(base + first.start_ms)
        };
        self.speech_active = true;
        self.last_speech_end_ms = base + spans.last().expect("spans is not empty").end_ms;
        onset
    }

    /// Decodes (or drops) the audio between the frontier and the buffer end:
    /// the tail of a speech span, which is shorter than a full window by
    /// construction. Tails below the flush floor are discarded.
    fn flush_tail(&mut self) -> Result<Vec<WordToken>, AiError> {
        let remaining = self.unprocessed_len();
        let floor = self.min_flush_samples();
        if remaining <= floor {
            // Too short to be intelligible; drop it like a VAD blip.
            self.consumed = self.buffer.len();
            return Ok(Vec::new());
        }
        let offset = self.media_offset_ms + samples_ms(self.consumed);
        let words = self.decoder.decode(&self.buffer[self.consumed..], offset)?;
        self.consumed = self.buffer.len();
        Ok(words)
    }

    fn min_flush_samples(&self) -> usize {
        (u64::from(WHISPER_SAMPLE_RATE) * self.config.min_flush_ms / 1_000) as usize
    }

    /// Drops the decoded prefix once it is large enough, keeping the buffer
    /// bounded on long recordings.
    fn compact(&mut self) {
        if self.consumed < COMPACT_THRESHOLD_SAMPLES {
            return;
        }
        let dropped = self.consumed;
        self.buffer.drain(..dropped);
        self.consumed = 0;
        self.media_offset_ms += samples_ms(dropped);
    }
}

/// Clamps config fields into their supported ranges.
fn sanitize(mut config: LiveTranscriberConfig) -> LiveTranscriberConfig {
    if config.window_ms < MIN_WINDOW_MS {
        tracing::warn!(
            window_ms = config.window_ms,
            "live window below {MIN_WINDOW_MS} ms, clamping"
        );
        config.window_ms = MIN_WINDOW_MS;
    }
    if config.window_ms > MAX_WINDOW_MS {
        tracing::warn!(
            window_ms = config.window_ms,
            "live window above {MAX_WINDOW_MS} ms, clamping"
        );
        config.window_ms = MAX_WINDOW_MS;
    }
    config.min_flush_ms = config.min_flush_ms.min(config.window_ms);
    config
}

fn samples_ms(samples: usize) -> u64 {
    samples as u64 * 1_000 / u64::from(WHISPER_SAMPLE_RATE)
}

fn samples_at(media_ms: u64) -> usize {
    (media_ms * u64::from(WHISPER_SAMPLE_RATE) / 1_000) as usize
}

/// Commands sent to the live worker thread.
enum Command {
    Push(Vec<f32>),
    Flush,
    Reset(u64),
    Shutdown,
}

/// Events emitted by the live worker thread.
#[derive(Debug)]
pub enum LiveEvent {
    /// A decode finished; the words carry media-timeline timestamps.
    Words(Vec<WordToken>),
    /// Whisper settled on a spoken language (auto-detect mode). Emitted at
    /// most once per session; the code is whisper's (`"ko"`, `"zh"`, ...).
    LanguageDetected(String),
    /// A window failed to decode; the pipeline continues with the next one.
    /// The failed audio is dropped.
    Error(AiError),
    /// A [`LiveCaptioner::flush`] completed: everything pushed before it has
    /// been decoded and its words already delivered. Batches (full-file
    /// pre-transcription) use this as their completion marker.
    Flushed,
}

/// Async-friendly facade over [`LiveTranscriber`].
///
/// Runs the (non-`Send`, blocking) transcriber on a dedicated OS thread and
/// moves commands and events over channels. The worker decodes the windows
/// each [`push`](Self::push) completes, so pushes should be paced close to
/// real time to keep the caption lag bounded by the window length.
///
/// Dropping the facade sends a shutdown command and **blocks until the worker
/// finishes its current decode** (bounded by one window's inference), so it
/// should not be dropped from a UI main loop mid-window.
pub struct LiveCaptioner {
    commands: std::sync::mpsc::Sender<Command>,
    events: mpsc::UnboundedReceiver<LiveEvent>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl LiveCaptioner {
    /// Loads the model and spawns the worker thread.
    ///
    /// The whisper context is created inside the thread (it is not `Send`),
    /// so a bad model path surfaces as a [`LiveEvent::Error`] before any
    /// words.
    pub fn new(config: LiveTranscriberConfig) -> Result<Self, AiError> {
        let (command_tx, command_rx) = std::sync::mpsc::channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let worker = std::thread::Builder::new()
            .name("sublayer-live-asr".to_owned())
            .spawn(move || run_worker(command_rx, event_tx, config))
            .map_err(|source| AiError::TaskJoin(source.to_string()))?;
        Ok(Self {
            commands: command_tx,
            events: event_rx,
            worker: Some(worker),
        })
    }

    /// Feeds audio to the worker for transcription.
    pub fn push(&self, samples: Vec<f32>) -> Result<(), AiError> {
        self.commands
            .send(Command::Push(samples))
            .map_err(|_| AiError::WorkerShutdown)
    }

    /// Asks the worker to decode whatever speech is still buffered.
    pub fn flush(&self) -> Result<(), AiError> {
        self.commands
            .send(Command::Flush)
            .map_err(|_| AiError::WorkerShutdown)
    }

    /// Re-anchors the transcriber timeline (seek).
    pub fn reset(&self, offset_ms: u64) -> Result<(), AiError> {
        self.commands
            .send(Command::Reset(offset_ms))
            .map_err(|_| AiError::WorkerShutdown)
    }

    /// Hands out the event receiver; only one consumer may hold it.
    pub fn take_events(&mut self) -> mpsc::UnboundedReceiver<LiveEvent> {
        std::mem::replace(&mut self.events, mpsc::unbounded_channel().1)
    }

    /// Stops the worker and waits for it to finish the current decode.
    pub fn shutdown(mut self) {
        self.stop_worker();
    }

    fn stop_worker(&mut self) {
        if let Some(worker) = self.worker.take() {
            let _ = self.commands.send(Command::Shutdown);
            let _ = worker.join();
        }
    }
}

impl Drop for LiveCaptioner {
    fn drop(&mut self) {
        self.stop_worker();
    }
}

/// Drives [`LiveTranscriber`] from the command channel; exits on `Shutdown`
/// or when the channel closes.
fn run_worker(
    commands: std::sync::mpsc::Receiver<Command>,
    events: mpsc::UnboundedSender<LiveEvent>,
    config: LiveTranscriberConfig,
) {
    let mut transcriber = match LiveTranscriber::new(config) {
        Ok(transcriber) => transcriber,
        Err(error) => {
            let _ = events.send(LiveEvent::Error(error));
            return;
        }
    };
    let mut reported_language = false;
    while let Ok(command) = commands.recv() {
        if !reported_language {
            if let Some(language) = transcriber.take_detected_language() {
                reported_language = true;
                let _ = events.send(LiveEvent::LanguageDetected(language));
            }
        }
        let result = match command {
            Command::Push(samples) => transcriber.push(&samples),
            Command::Flush => {
                let result = transcriber.flush();
                forward(result, &events);
                // The tail decode (if any) already delivered its words above;
                // mark the batch complete.
                let _ = events.send(LiveEvent::Flushed);
                continue;
            }
            Command::Reset(offset_ms) => {
                transcriber.reset(offset_ms);
                continue;
            }
            Command::Shutdown => break,
        };
        forward(result, &events);
    }
}

fn forward(result: Result<Vec<WordToken>, AiError>, events: &mpsc::UnboundedSender<LiveEvent>) {
    match result {
        Ok(words) if !words.is_empty() => {
            let _ = events.send(LiveEvent::Words(words));
        }
        Ok(_) => {}
        Err(error) => {
            tracing::warn!(%error, "live decode failed; continuing with the next window");
            let _ = events.send(LiveEvent::Error(error));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = WHISPER_SAMPLE_RATE;

    fn tone(amplitude: f32, ms: u64) -> Vec<f32> {
        vec![amplitude; (ms * u64::from(RATE) / 1_000) as usize]
    }

    fn config() -> LiveTranscriberConfig {
        LiveTranscriberConfig {
            window_ms: 1_000,
            ..LiveTranscriberConfig::default()
        }
    }

    /// Records the window it was handed: the word's `start_ms` is the decode
    /// offset and `end_ms` the window length, so tests can verify both the
    /// timeline placement and the window sizing.
    struct RecordingDecoder;

    impl Decoder for RecordingDecoder {
        fn decode(&mut self, samples: &[f32], offset_ms: u64) -> Result<Vec<WordToken>, AiError> {
            Ok(vec![WordToken::new(
                format!("w{}", samples.len()),
                offset_ms,
                offset_ms + samples_ms(samples.len()),
            )])
        }
    }

    fn live(config: LiveTranscriberConfig) -> LiveTranscriber {
        LiveTranscriber::with_decoder(config, Box::new(RecordingDecoder))
    }

    #[test]
    fn decodes_windows_on_the_media_timeline() {
        let mut transcriber = live(config());
        // 1 s chunks; the 1 s window decodes once per second of speech.
        let words = transcriber.push(&tone(0.1, 1_000)).unwrap();
        assert_eq!(words.len(), 1, "one window per full second");
        assert_eq!(words[0].start_ms, 0);
        assert_eq!(words[0].end_ms, 1_000);

        let words = transcriber.push(&tone(0.1, 1_000)).unwrap();
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].start_ms, 1_000, "second window shifts by 1 s");
        assert_eq!(words[0].end_ms, 2_000);
    }

    #[test]
    fn late_short_chunks_build_toward_the_next_window() {
        let mut transcriber = live(config());
        transcriber.push(&tone(0.1, 500)).unwrap(); // half a window
        let words = transcriber.push(&tone(0.1, 300)).unwrap();
        assert!(words.is_empty(), "still below the window");
        let words = transcriber.push(&tone(0.1, 300)).unwrap();
        assert_eq!(words.len(), 1, "crossing 1 s decodes one window");
        assert_eq!(words[0].start_ms, 0);
        assert_eq!(words[0].end_ms, 1_000);
    }

    #[test]
    fn vad_keeps_silence_away_from_the_decoder() {
        let mut transcriber = live(config());
        for _ in 0..4 {
            let words = transcriber.push(&tone(0.0, 1_000)).unwrap();
            assert!(words.is_empty());
        }
        assert_eq!(transcriber.unprocessed_len(), 0, "silence must be dropped");

        // A tone then decodes, starting exactly at the voice onset.
        let words = transcriber.push(&tone(0.1, 1_000)).unwrap();
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].start_ms, 4_000, "silent prefix must be skipped");
        assert_eq!(words[0].end_ms, 5_000);
    }

    #[test]
    fn speech_end_flushes_the_short_tail() {
        let mut transcriber = live(config());
        transcriber.push(&tone(0.1, 1_000)).unwrap(); // full window decoded
        let words = transcriber.push(&tone(0.0, 500)).unwrap();
        // The gap is >= VAD_SILENCE_MS (350) and the tail (500 ms) is above
        // the flush floor, so it is decoded as one final window.
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].start_ms, 1_000);
        assert_eq!(words[0].end_ms, 1_500);

        // Continued silence after the flush must not decode anything.
        let words = transcriber.push(&tone(0.0, 2_000)).unwrap();
        assert!(words.is_empty(), "post-flush silence must stay undecoded");
        assert_eq!(transcriber.unprocessed_len(), 0);
    }

    #[test]
    fn short_tails_are_dropped_like_vad_blips() {
        let mut config = config();
        config.min_flush_ms = 2_000;
        let mut transcriber = live(config);
        transcriber.push(&tone(0.1, 1_000)).unwrap();
        let words = transcriber.push(&tone(0.0, 500)).unwrap();
        assert!(words.is_empty(), "tail below the flush floor is dropped");
        assert_eq!(transcriber.unprocessed_len(), 0);
    }

    #[test]
    fn speech_in_sub_blip_chunks_still_decodes() {
        // Streaming pushes arrive in pieces far below the VAD's 200 ms blip
        // floor; classification must run over the accumulated buffer or no
        // span is ever confirmed (regression: the player feeds ~20 ms
        // chunks and produced no captions).
        let mut transcriber = live(config());
        let mut decoded = 0;
        for _ in 0..10 {
            decoded += transcriber.push(&tone(0.1, 100)).unwrap().len();
        }
        assert_eq!(
            decoded, 1,
            "1 s of speech in 100 ms pieces decodes one window"
        );

        // The span then ends on the certainty rule like a real gap.
        let words = transcriber.push(&tone(0.0, 500)).unwrap();
        assert_eq!(words.len(), 1, "speech end flushes the tail");
        assert_eq!(words[0].start_ms, 1_000);
        assert_eq!(words[0].end_ms, 1_500);
    }

    #[test]
    fn short_silent_runs_keep_buffering_toward_a_span() {
        // Two 100 ms speech pieces are each below the blip floor; the second
        // push must not discard the first as silence.
        let mut transcriber = live(config());
        transcriber.push(&tone(0.1, 100)).unwrap();
        transcriber.push(&tone(0.1, 100)).unwrap();
        let words = transcriber.push(&tone(0.1, 800)).unwrap();
        assert_eq!(words.len(), 1, "the accumulated span decodes");
        assert_eq!(words[0].start_ms, 0);
    }

    #[test]
    fn gaps_shorter_than_the_bridge_keep_the_window_open() {
        let mut transcriber = live(config());
        transcriber.push(&tone(0.1, 1_000)).unwrap(); // speech, window decoded
        transcriber.push(&tone(0.0, 200)).unwrap(); // gap < 350 ms, bridged
        let words = transcriber.push(&tone(0.1, 3_000)).unwrap();
        // The gap is bridged, so decode continues across it: 3.2 s of buffer
        // behind the frontier -> three contiguous 1 s windows.
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].start_ms, 1_000);
        assert_eq!(words[2].start_ms, 3_000);
    }

    #[test]
    fn flush_decodes_everything_still_buffered() {
        let mut transcriber = live(config());
        transcriber.push(&tone(0.1, 700)).unwrap();
        let words = transcriber.flush().unwrap();
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].start_ms, 0, "tail shorter than the window");
        assert_eq!(words[0].end_ms, 700);
    }

    #[test]
    fn flush_on_pure_silence_decodes_nothing() {
        let mut transcriber = live(config());
        transcriber.push(&tone(0.0, 2_000)).unwrap();
        assert!(transcriber.flush().unwrap().is_empty());
        assert_eq!(transcriber.unprocessed_len(), 0);
    }

    #[test]
    fn reset_reanchors_the_timeline_and_drops_the_buffer() {
        let mut transcriber = live(config());
        transcriber.push(&tone(0.1, 2_000)).unwrap();
        transcriber.reset(30_000);
        let words = transcriber.push(&tone(0.1, 2_000)).unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(
            words[0].start_ms, 30_000,
            "words after a reset follow the new anchor"
        );
        assert_eq!(words[1].start_ms, 31_000);
    }

    #[test]
    fn long_recordings_compact_the_buffer() {
        let mut transcriber = live(config());
        for _ in 0..35 {
            transcriber.push(&tone(0.1, 1_000)).unwrap();
        }
        assert_eq!(transcriber.position_ms(), 35_000);
        assert!(
            transcriber.buffer.len() < COMPACT_THRESHOLD_SAMPLES + 16_000,
            "buffer must stay bounded, got {} samples",
            transcriber.buffer.len()
        );
    }

    #[test]
    fn empty_pushes_are_noops() {
        let mut transcriber = live(config());
        assert!(transcriber.push(&[]).unwrap().is_empty());
        assert_eq!(transcriber.position_ms(), 0);
    }

    #[test]
    fn config_is_clamped_into_supported_ranges() {
        let mut clamped_cfg = config();
        clamped_cfg.window_ms = 50;
        clamped_cfg.min_flush_ms = 9_000;
        let clamped = sanitize(clamped_cfg);
        assert_eq!(clamped.window_ms, MIN_WINDOW_MS);
        assert_eq!(
            clamped.min_flush_ms, MIN_WINDOW_MS,
            "flush capped at window"
        );

        let mut big_cfg = config();
        big_cfg.window_ms = 60_000;
        assert_eq!(sanitize(big_cfg).window_ms, MAX_WINDOW_MS);
    }

    #[test]
    fn detected_language_is_reported_once_and_clears() {
        struct DetectingDecoder {
            detected: Option<String>,
        }
        impl Decoder for DetectingDecoder {
            fn decode(
                &mut self,
                _samples: &[f32],
                _offset_ms: u64,
            ) -> Result<Vec<WordToken>, AiError> {
                Ok(Vec::new())
            }
            fn take_detected_language(&mut self) -> Option<String> {
                self.detected.take()
            }
        }
        let mut config = config();
        config.enable_vad = false;
        let mut transcriber = LiveTranscriber::with_decoder(
            config,
            Box::new(DetectingDecoder {
                detected: Some("ko".to_owned()),
            }),
        );
        transcriber.push(&tone(0.1, 1_000)).unwrap();
        assert_eq!(transcriber.take_detected_language().as_deref(), Some("ko"));
        assert_eq!(
            transcriber.take_detected_language(),
            None,
            "clears once read"
        );
    }

    #[test]
    fn pinned_language_decoders_report_no_detection() {
        let mut config = config();
        config.enable_vad = false;
        config.language = Some("ko".to_owned());
        let mut transcriber = live(config);
        transcriber.push(&tone(0.1, 2_000)).unwrap();
        assert_eq!(transcriber.take_detected_language(), None);
    }

    #[test]
    fn decoder_errors_propagate_from_push() {
        struct FailingDecoder;
        impl Decoder for FailingDecoder {
            fn decode(
                &mut self,
                _samples: &[f32],
                _offset_ms: u64,
            ) -> Result<Vec<WordToken>, AiError> {
                Err(AiError::TaskJoin("boom".to_owned()))
            }
        }
        let mut config = config();
        // Keep speech active without the VAD so the decode fires immediately.
        config.enable_vad = false;
        let mut transcriber = LiveTranscriber::with_decoder(config, Box::new(FailingDecoder));
        assert!(transcriber.push(&tone(0.1, 2_000)).is_err());
    }

    // ------------------------------------------------- worker-level (ignored)

    /// End-to-end through the worker thread and channels; needs a real model.
    ///
    /// ```text
    /// curl -L -o /tmp/ggml-tiny.en.bin \
    ///   https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
    /// SUBLAYER_TEST_MODEL=/tmp/ggml-tiny.en.bin cargo test -p sublayer-ai \
    ///   --release -- --ignored live_captioner_end_to_end
    /// ```
    #[ignore]
    #[tokio::test]
    async fn live_captioner_end_to_end_with_model() {
        let model = std::env::var("SUBLAYER_TEST_MODEL")
            .unwrap_or_else(|_| "/tmp/ggml-tiny.en.bin".to_owned());
        if !std::path::Path::new(&model).is_file() {
            eprintln!("skipping: set SUBLAYER_TEST_MODEL to a whisper ggml model");
            return;
        }

        let mut captioner = LiveCaptioner::new(LiveTranscriberConfig {
            model_path: model.into(),
            language: Some("en".to_owned()),
            window_ms: 4_000,
            ..LiveTranscriberConfig::default()
        })
        .unwrap();
        let mut events = captioner.take_events();

        // Silence first: the VAD gate must keep it out of the decoder.
        captioner.push(vec![0.0_f32; 4 * RATE as usize]).unwrap();

        // A loud tone encodes to a recognizable token in tiny models.
        let mut tone_audio = Vec::new();
        for _ in 0..4 {
            tone_audio.extend_from_slice(&tone(0.9, 1_000));
        }
        captioner.push(tone_audio).unwrap();
        captioner.flush().unwrap();

        let mut saw_words = false;
        let mut saw_flushed = false;
        let start = std::time::Instant::now();
        let deadline = std::time::Duration::from_secs(180);
        while !(saw_words && saw_flushed) && start.elapsed() < deadline {
            let Some(event) =
                tokio::time::timeout(deadline.saturating_sub(start.elapsed()), events.recv())
                    .await
                    .expect("live worker must keep producing events")
            else {
                break;
            };
            match event {
                LiveEvent::Words(words) => {
                    saw_words = true;
                    eprintln!("live words: {}", words.len());
                    for word in &words {
                        assert!(word.end_ms >= word.start_ms);
                    }
                }
                LiveEvent::LanguageDetected(language) => {
                    eprintln!("detected language: {language}");
                }
                LiveEvent::Error(error) => panic!("unexpected live error: {error}"),
                LiveEvent::Flushed => saw_flushed = true,
            }
        }
        assert!(saw_words, "tone must transcribe into at least one word");
        assert!(saw_flushed, "flush must mark the batch complete");

        captioner.shutdown();
    }
}
