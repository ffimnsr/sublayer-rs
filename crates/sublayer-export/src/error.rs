//! Error type of the export pipeline.

use sublayer_media::MediaError;
use sublayer_subtitles::SubtitleError;

/// Errors raised while probing hardware or rendering a video.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// FFmpeg could not be located, spawned, or exited with a failure.
    #[error(transparent)]
    Media(#[from] MediaError),

    /// Caption compilation failed.
    #[error(transparent)]
    Subtitles(#[from] SubtitleError),

    /// The requested encoder is not available on this machine.
    #[error(
        "hardware encoder `{0}` is not available; run with `--encoder cpu` to force software encoding"
    )]
    EncoderUnavailable(&'static str),

    /// `SUBLAYER_ENCODER` holds a value that names no encoder.
    #[error(
        "unknown encoder `{0}` (expected auto, vaapi, nvenc, or cpu; vulkan requires the `encode_vulkan` feature)"
    )]
    UnknownEncoder(String),

    /// The progress receiver was dropped, so the render was stopped.
    #[error("render cancelled")]
    Cancelled,

    /// Filesystem interaction failed.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}
