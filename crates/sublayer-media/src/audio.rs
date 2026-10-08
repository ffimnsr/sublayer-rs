//! Audio extraction: converts any decodable media input into the canonical
//! 16 kHz mono 16-bit PCM WAV that Whisper expects.

use std::path::Path;
use std::process::ExitStatus;

use crate::{MediaError, ffmpeg};

/// Sample rate required by Whisper's acoustic model.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Smallest possible WAV file: a header without a single sample.
const EMPTY_WAV_LEN: u64 = 44;

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
}
