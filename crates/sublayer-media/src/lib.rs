//! Media foundation for Sublayer: FFprobe metadata, Whisper-ready audio
//! extraction, and timeline waveform caches.
//!
//! Every external invocation goes through the private `ffmpeg` helpers, which
//! locate the executables on `PATH` (or via `SUBLAYER_FFMPEG` /
//! `SUBLAYER_FFPROBE`) and kill child processes when their future is dropped.

pub mod audio;
pub mod error;
mod ffmpeg;
pub mod probe;
pub mod waveform;

pub use audio::{WHISPER_SAMPLE_RATE, extract_audio_16k};
pub use error::MediaError;
pub use probe::probe_video;
pub use waveform::{
    DEFAULT_BUCKETS_PER_SEC, WaveformBucket, WaveformCache, generate_waveform_cache,
    waveform_from_wav,
};

#[cfg(test)]
pub(crate) mod test_util {
    use std::path::Path;
    use std::process::{Command, Stdio};

    use crate::ffmpeg;

    /// Whether the FFmpeg CLI can be located; tests skip when it cannot.
    pub(crate) fn ffmpeg_available() -> bool {
        ffmpeg::resolve(ffmpeg::FFMPEG, ffmpeg::FFMPEG_ENV).is_ok()
    }

    /// Generates a one-second 320x240@30fps MPEG-4 clip, optionally with a
    /// 440 Hz sine track.
    ///
    /// Returns `false` when FFmpeg could not produce the fixture.
    pub(crate) fn generate_test_video(path: &Path, with_audio: bool) -> bool {
        let Ok(program) = ffmpeg::resolve(ffmpeg::FFMPEG, ffmpeg::FFMPEG_ENV) else {
            return false;
        };

        let mut command = Command::new(program);
        command
            .args(["-hide_banner", "-nostdin", "-loglevel", "error", "-y"])
            .args(["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30"]);
        if with_audio {
            command.args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]);
        }
        command.args(["-t", "1", "-c:v", "mpeg4", "-q:v", "5"]);
        if with_audio {
            command.args(["-c:a", "aac", "-b:a", "64k"]);
        }
        command.arg(path);
        command.stdout(Stdio::null()).stderr(Stdio::null());

        command.status().is_ok_and(|status| status.success())
    }
}
