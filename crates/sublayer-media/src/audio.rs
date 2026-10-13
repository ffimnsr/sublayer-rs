//! Audio extraction: converts any decodable media input into the canonical
//! 16 kHz mono 16-bit PCM WAV that Whisper expects, or streams the same PCM
//! live over a pipe for real-time consumers.

use std::path::{Path, PathBuf};
use std::process::ExitStatus;

use tokio::io::{AsyncReadExt, BufReader};

use crate::{MediaError, ffmpeg};

/// Sample rate required by Whisper's acoustic model.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Smallest possible WAV file: a header without a single sample.
const EMPTY_WAV_LEN: u64 = 44;

/// Stderr is drained into this many bytes (kept as a tail excerpt) so a
/// verbose ffmpeg can never fill its pipe and deadlock a live stream.
const STDERR_CAP_BYTES: usize = 64 * 1024;

/// Byte chunk read from the decoder in one syscall; 2048 s16le samples.
const READ_CHUNK_BYTES: usize = 4_096;

/// Converts the first audio stream of `input` into a 16 kHz mono 16-bit PCM WAV.
///
/// Loudness is left untouched; "normalized" here means the canonical sample
/// rate, channel count, and sample format, which is what the speech model
/// requires. Existing files at `output` are overwritten. Dropping the returned
/// future kills the FFmpeg child process, so cancellation is safe. On failure
/// `output` may be left in a partial state and must not be consumed.
pub async fn extract_audio_16k(input: &Path, output: &Path) -> Result<(), MediaError> {
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    let program = ffmpeg::resolve(ffmpeg::FFMPEG, ffmpeg::FFMPEG_ENV)?;
    let mut command = ffmpeg::command(&program);
    command
        .args(["-hide_banner", "-nostdin", "-loglevel", "error", "-y"])
        .arg("-i")
        .arg(input)
        .args(["-map", "0:a:0", "-vn", "-ac", "1", "-ar"])
        .arg(WHISPER_SAMPLE_RATE.to_string())
        .args(["-c:a", "pcm_s16le", "-f", "wav"])
        .arg(output);

    let result = ffmpeg::run(ffmpeg::FFMPEG, &mut command).await?;
    if !result.status.success() {
        return Err(classify_failure(input, result.status, &result.stderr));
    }

    if std::fs::metadata(output)?.len() <= EMPTY_WAV_LEN {
        return Err(MediaError::EmptyAudioOutput {
            input: input.to_path_buf(),
        });
    }
    Ok(())
}

/// Maps the FFmpeg "no audio stream matched" failures to
/// [`MediaError::NoAudioStream`] and keeps everything else as
/// [`MediaError::CommandFailed`].
fn classify_failure(input: &Path, status: ExitStatus, stderr: &[u8]) -> MediaError {
    let message = String::from_utf8_lossy(stderr);
    if message.contains("matches no streams") || message.contains("does not contain any stream") {
        return MediaError::NoAudioStream(input.to_path_buf());
    }
    ffmpeg::failure(ffmpeg::FFMPEG, status, stderr)
}

/// Live 16 kHz mono PCM samples decoded from an input by a child ffmpeg.
///
/// The decoder writes raw `s16le` to a pipe; [`read_samples`](Self::read_samples)
/// converts it to normalized `f32` in batches. The child decodes as fast as the
/// (bounded) pipe admits, so a consumer that paces itself at real time keeps
/// the decode-ahead to roughly one pipe buffer (~2 s of audio), which is what
/// a live caption pipeline wants. Errors are reported by
/// [`finish`](Self::finish), which must be called after the stream reads EOF.
///
/// Dropping the stream kills the child (`kill_on_drop`); stderr is drained on
/// the Tokio executor so a verbose decoder can never block on a full pipe.
/// [`stream_audio_16k`] and [`finish`](Self::finish) must run on the **same**
/// Tokio runtime: the stderr drain task is spawned there and awaited by
/// `finish`.
pub struct AudioPcmStream {
    input: PathBuf,
    child: tokio::process::Child,
    reader: BufReader<tokio::process::ChildStdout>,
    stderr: tokio::task::JoinHandle<Vec<u8>>,
    /// Whole s16le samples decoded but not yet handed out.
    pending: Vec<f32>,
    /// Next unconsumed index into `pending`.
    pending_pos: usize,
    /// One odd byte held between reads; s16le streams stay aligned.
    carry: Vec<u8>,
}

/// Spawns ffmpeg decoding the first audio stream of `input` to raw
/// little-endian 16-bit PCM on stdout.
///
/// `offset_ms` seeks before the input is opened (fast input seeking; the
/// cut may land a few samples early, which live captions tolerate).
pub async fn stream_pcm_s16le(
    input: &Path,
    offset_ms: Option<u64>,
    sample_rate: u32,
    channels: u16,
) -> Result<AudioPcmStream, MediaError> {
    let program = ffmpeg::resolve(ffmpeg::FFMPEG, ffmpeg::FFMPEG_ENV)?;
    let mut command = ffmpeg::command(&program);
    command.args(["-hide_banner", "-nostdin", "-loglevel", "error"]);
    if let Some(offset_ms) = offset_ms {
        // ffmpeg's `-ss` counts seconds; fractional offsets keep millisecond
        // seeks accurate (100 ms -> "0.100").
        let seconds = format!("{:.3}", offset_ms as f64 / 1_000.0);
        command.arg("-ss").arg(seconds);
    }
    command
        .arg("-i")
        .arg(input)
        .args(["-map", "0:a:0", "-vn", "-ac"])
        .arg(channels.to_string())
        .args(["-ar"])
        .arg(sample_rate.to_string())
        .args(["-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]);

    tracing::debug!(command = %ffmpeg::describe(&command), "streaming media audio");
    let mut child = command.spawn().map_err(|source| MediaError::Spawn {
        binary: ffmpeg::FFMPEG,
        source,
    })?;
    let stdout = child
        .stdout
        .take()
        .expect("ffmpeg::command always pipes stdout");
    let stderr = child
        .stderr
        .take()
        .expect("ffmpeg::command always pipes stderr");
    let stderr_task = tokio::spawn(drain_stderr(stderr));

    Ok(AudioPcmStream {
        input: input.to_path_buf(),
        child,
        reader: BufReader::new(stdout),
        stderr: stderr_task,
        pending: Vec::new(),
        pending_pos: 0,
        carry: Vec::new(),
    })
}

/// Spawns ffmpeg decoding the first audio stream of `input` to raw 16 kHz
/// mono 16-bit little-endian PCM on stdout.
pub async fn stream_audio_16k(input: &Path) -> Result<AudioPcmStream, MediaError> {
    stream_pcm_s16le(input, None, WHISPER_SAMPLE_RATE, 1).await
}

impl AudioPcmStream {
    /// Reads up to `samples.len()` normalized `f32` samples, blocking until at
    /// least one sample (or EOF) is available. Returns `0` at end of stream
    /// (or when `samples` is empty); call [`finish`](Self::finish) afterwards
    /// to surface decoder errors such as a missing audio stream.
    pub async fn read_samples(&mut self, samples: &mut [f32]) -> Result<usize, MediaError> {
        if samples.is_empty() {
            return Ok(0);
        }
        let mut written = 0;
        while written < samples.len() {
            if let Some(sample) = self.next_pending() {
                samples[written] = sample;
                written += 1;
                continue;
            }
            let mut bytes = [0_u8; READ_CHUNK_BYTES];
            let read = self.reader.read(&mut bytes).await?;
            if read == 0 {
                if self.carry.is_empty() {
                    break;
                }
                return Err(MediaError::TruncatedAudioStream);
            }
            self.decode_chunk(&bytes[..read]);
        }
        Ok(written)
    }

    /// Waits for the decoder to exit and reports its status: a non-zero exit
    /// (for example an input without an audio stream) surfaces as
    /// [`MediaError::NoAudioStream`] or [`MediaError::CommandFailed`].
    ///
    /// The child is killed first so a stream that was not fully drained cannot
    /// deadlock: ffmpeg would otherwise sit blocked on a full stdout pipe
    /// while this waits for it to exit.
    pub async fn finish(mut self) -> Result<(), MediaError> {
        // Killing an already-exited child is a harmless no-op error.
        let _ = self.child.kill().await;
        let status = match self.child.wait().await {
            Ok(status) => status,
            Err(source) => {
                return Err(MediaError::Spawn {
                    binary: ffmpeg::FFMPEG,
                    source,
                });
            }
        };
        let stderr = self.stderr.await.unwrap_or_default();
        if !status.success() {
            return Err(classify_failure(&self.input, status, &stderr));
        }
        Ok(())
    }

    /// Converts a byte chunk into samples, keeping one odd byte for the next
    /// call.
    fn decode_chunk(&mut self, bytes: &[u8]) {
        self.carry.extend_from_slice(bytes);
        let buf = std::mem::take(&mut self.carry);
        let whole = buf.len() - buf.len() % 2;
        if whole < buf.len() {
            self.carry = buf[whole..].to_vec();
        }
        self.pending.reserve(whole / 2);
        for pair in buf[..whole].chunks_exact(2) {
            let sample = i16::from_le_bytes([pair[0], pair[1]]);
            self.pending.push(f32::from(sample) / 32_768.0);
        }
    }

    fn next_pending(&mut self) -> Option<f32> {
        if self.pending_pos >= self.pending.len() {
            self.pending.clear();
            self.pending_pos = 0;
            return None;
        }
        let sample = self.pending[self.pending_pos];
        self.pending_pos += 1;
        Some(sample)
    }
}

/// Keeps the tail of ffmpeg's stderr in a bounded buffer, so no amount of
/// decoder chatter can fill the pipe and stall the child.
pub(crate) async fn drain_stderr(mut stderr: tokio::process::ChildStderr) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0_u8; READ_CHUNK_BYTES];
    loop {
        match stderr.read(&mut chunk).await {
            Ok(0) | Err(_) => return buf,
            Ok(read) => {
                buf.extend_from_slice(&chunk[..read]);
                if buf.len() > STDERR_CAP_BYTES {
                    buf.drain(..buf.len() - STDERR_CAP_BYTES);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{SampleFormat, WavSpec, WavWriter};

    fn failed_status() -> ExitStatus {
        std::process::Command::new("sh")
            .args(["-c", "exit 1"])
            .status()
            .unwrap()
    }

    /// Writes `seconds` of a sine tone as an interleaved WAV `channels` wide.
    fn write_tone_wav(path: &Path, sample_rate: u32, channels: u16, seconds: f32) {
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let frames = (sample_rate as f32 * seconds) as u32;
        let mut writer = WavWriter::create(path, spec).unwrap();
        for frame in 0..frames {
            let phase = 2.0 * std::f32::consts::PI * 440.0 * frame as f32 / sample_rate as f32;
            let sample = (phase.sin() * 0.8 * f32::from(i16::MAX)).round() as i16;
            for _ in 0..channels {
                writer.write_sample(sample).unwrap();
            }
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn missing_audio_streams_are_classified_as_such() {
        let stderr = b"[out#0/wav @ 0x55] Output file does not contain any stream";
        let error = classify_failure(Path::new("silent.mp4"), failed_status(), stderr);
        assert!(matches!(error, MediaError::NoAudioStream(_)));

        let stderr = b"[in#0 @ 0x55] Stream map '0:a:0' matches no streams.";
        let error = classify_failure(Path::new("silent.mp4"), failed_status(), stderr);
        assert!(matches!(error, MediaError::NoAudioStream(_)));
    }

    #[test]
    fn unrelated_failures_stay_command_failures() {
        let stderr = b"clip.mp4: No such file or directory";
        let error = classify_failure(Path::new("missing.mp4"), failed_status(), stderr);
        assert!(matches!(error, MediaError::CommandFailed { .. }));
    }

    #[tokio::test]
    async fn extraction_writes_16khz_mono_pcm_and_creates_parent_dirs() {
        if !crate::test_util::ffmpeg_available() {
            eprintln!("skipping: ffmpeg is not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.wav");
        write_tone_wav(&source, 48_000, 2, 0.25);

        let output = directory.path().join("nested").join("audio.wav");
        tokio::time::timeout(
            std::time::Duration::from_secs(60),
            extract_audio_16k(&source, &output),
        )
        .await
        .expect("extraction timed out")
        .expect("extraction failed");

        let reader = hound::WavReader::open(&output).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, WHISPER_SAMPLE_RATE);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, SampleFormat::Int);

        let frames = reader.len() as i64 / i64::from(spec.channels);
        assert!(
            (frames - 4_000).abs() <= 100,
            "expected ~4000 frames of 0.25s at 16 kHz, got {frames}"
        );
    }

    #[tokio::test]
    async fn extraction_reports_inputs_without_audio() {
        if !crate::test_util::ffmpeg_available() {
            eprintln!("skipping: ffmpeg is not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("silent.mp4");
        if !crate::test_util::generate_test_video(&source, false) {
            eprintln!("skipping: ffmpeg could not generate the fixture");
            return;
        }

        let output = directory.path().join("audio.wav");
        let error = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            extract_audio_16k(&source, &output),
        )
        .await
        .expect("extraction timed out")
        .expect_err("silent input should fail");

        assert!(
            matches!(error, MediaError::NoAudioStream(_)),
            "got {error:?}"
        );
    }

    // -------------------------------------------------------------- streaming

    /// Reads every sample of a stream into a vector.
    async fn read_all(stream: &mut AudioPcmStream) -> Vec<f32> {
        let mut samples = Vec::new();
        let mut chunk = [0.0_f32; 512];
        loop {
            let read = stream.read_samples(&mut chunk).await.unwrap();
            if read == 0 {
                return samples;
            }
            samples.extend_from_slice(&chunk[..read]);
        }
    }

    #[tokio::test]
    async fn stream_reads_16khz_mono_pcm() {
        if !crate::test_util::ffmpeg_available() {
            eprintln!("skipping: ffmpeg is not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.wav");
        write_tone_wav(&source, 48_000, 2, 0.25);

        let mut stream = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            stream_audio_16k(&source),
        )
        .await
        .expect("spawn timed out")
        .expect("spawn failed");
        let samples = read_all(&mut stream).await;
        stream.finish().await.expect("decoder exited cleanly");

        // 0.25 s at 48 kHz stereo resamples to ~4000 mono samples at 16 kHz.
        assert!(
            (samples.len() as i64 - 4_000).abs() <= 2,
            "expected ~4000 samples, got {}",
            samples.len()
        );
        assert!(samples.iter().any(|s| s.abs() > 0.7), "tone must be loud");
        assert!(
            samples.iter().all(|s| s.abs() <= 1.0),
            "samples must be normalized"
        );
    }

    #[tokio::test]
    async fn stream_matches_the_extracted_wav_sample_for_sample() {
        if !crate::test_util::ffmpeg_available() {
            eprintln!("skipping: ffmpeg is not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.wav");
        write_tone_wav(&source, 48_000, 2, 0.25);

        let wav = directory.path().join("audio.wav");
        extract_audio_16k(&source, &wav).await.unwrap();
        let expected: Vec<f32> = hound::WavReader::open(&wav)
            .unwrap()
            .samples::<i16>()
            .map(|sample| f32::from(sample.unwrap()) / 32_768.0)
            .collect();

        let mut stream = stream_audio_16k(&source).await.unwrap();
        let actual = read_all(&mut stream).await;
        stream.finish().await.unwrap();

        let common = actual.len().min(expected.len());
        // Both paths run the same swr resampler, so the overlapping prefix
        // must be bit-identical; allow the tail to differ by one sample.
        assert!(
            (actual.len() as i64 - expected.len() as i64).abs() <= 2,
            "sample counts diverged: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (index, (left, right)) in actual[..common].iter().zip(&expected[..common]).enumerate() {
            assert_eq!(left, right, "sample {index} diverged");
        }
    }

    #[tokio::test]
    async fn stream_handles_odd_sized_reads_across_chunk_boundaries() {
        if !crate::test_util::ffmpeg_available() {
            eprintln!("skipping: ffmpeg is not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.wav");
        write_tone_wav(&source, 48_000, 2, 0.25);

        let mut whole = stream_audio_16k(&source).await.unwrap();
        let expected = read_all(&mut whole).await;
        whole.finish().await.unwrap();

        let mut odd = stream_audio_16k(&source).await.unwrap();
        let mut actual = Vec::new();
        let mut chunk = [0.0_f32; 3]; // deliberately crosses s16le pairs
        loop {
            let read = odd.read_samples(&mut chunk).await.unwrap();
            if read == 0 {
                break;
            }
            actual.extend_from_slice(&chunk[..read]);
        }
        odd.finish().await.unwrap();

        assert_eq!(actual, expected, "odd reads must not corrupt the stream");
    }

    #[tokio::test]
    async fn stream_reports_inputs_without_audio() {
        if !crate::test_util::ffmpeg_available() {
            eprintln!("skipping: ffmpeg is not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("silent.mp4");
        if !crate::test_util::generate_test_video(&source, false) {
            eprintln!("skipping: ffmpeg could not generate the fixture");
            return;
        }

        let mut stream = stream_audio_16k(&source).await.unwrap();
        let mut chunk = [0.0_f32; 256];
        assert_eq!(stream.read_samples(&mut chunk).await.unwrap(), 0);

        let error = stream.finish().await.expect_err("no-audio input must fail");
        assert!(
            matches!(error, MediaError::NoAudioStream(_)),
            "got {error:?}"
        );
    }

    #[tokio::test]
    async fn stream_honors_a_start_offset() {
        if !crate::test_util::ffmpeg_available() {
            eprintln!("skipping: ffmpeg is not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.wav");
        write_tone_wav(&source, 48_000, 2, 0.25);

        let mut full_stream = stream_audio_16k(&source).await.unwrap();
        let full = read_all(&mut full_stream).await;
        full_stream.finish().await.unwrap();

        let mut offset_stream = stream_pcm_s16le(&source, Some(100), WHISPER_SAMPLE_RATE, 1)
            .await
            .unwrap();
        let offset = read_all(&mut offset_stream).await;
        offset_stream.finish().await.unwrap();

        // 100 ms at 16 kHz = 1600 samples; input seeking may cut a few
        // samples early, so allow slack on both ends.
        let expected = 4_000 - 1_600;
        assert!(
            (offset.len() as i64 - expected).abs() <= 200,
            "expected ~{expected} samples after 100 ms, got {}",
            offset.len()
        );
        assert!(
            offset.iter().any(|sample| sample.abs() > 0.7),
            "the offset stream must carry the tone"
        );
        // The seek must have jumped: the offset stream starts at a different
        // phase of the sine than the unseeked stream.
        assert_ne!(
            &offset[..400],
            &full[..400],
            "offset stream must not start at the media beginning"
        );
    }

    #[tokio::test]
    async fn stream_round_trips_a_test_video() {
        if !crate::test_util::ffmpeg_available() {
            eprintln!("skipping: ffmpeg is not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("tone.mp4");
        if !crate::test_util::generate_test_video(&source, true) {
            eprintln!("skipping: ffmpeg could not generate the fixture");
            return;
        }

        let mut stream = stream_audio_16k(&source).await.unwrap();
        let samples = read_all(&mut stream).await;
        stream.finish().await.unwrap();

        // The fixture carries 1 s of 440 Hz tone.
        assert!(
            (samples.len() as i64 - 16_000).abs() <= 256,
            "expected ~16000 samples, got {}",
            samples.len()
        );
        assert!(samples.iter().any(|s| s.abs() > 0.5), "tone must be loud");
    }
}
