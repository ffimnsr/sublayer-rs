//! Errors raised by model management, VAD, and transcription.

use std::path::PathBuf;

/// Errors raised by [`sublayer-ai`](crate) operations.
#[derive(Debug, thiserror::Error)]
pub enum AiError {
    /// The requested model is not in the known-model table.
    #[error("unknown model `{0}`; known models: tiny.en, base.en, small.en, large-v3-turbo")]
    UnknownModel(String),

    /// The HTTP client could not be constructed.
    #[error("HTTP client could not be created: {0}")]
    ClientBuild(#[source] reqwest::Error),

    /// A model download request failed at the transport level.
    #[error("failed to download `{name}`: {source}")]
    Download {
        /// Model being downloaded.
        name: String,
        /// Transport-level failure.
        #[source]
        source: reqwest::Error,
    },

    /// The download server answered with an unexpected status code.
    #[error("download of `{name}` returned HTTP {status}")]
    HttpStatus {
        /// Model being downloaded.
        name: String,
        /// HTTP status code received.
        status: u16,
    },

    /// The downloaded file does not match the pinned SHA-256 digest.
    #[error("sha256 mismatch for {path} (expected {expected}, got {actual})")]
    Sha256Mismatch {
        /// File that failed verification.
        path: PathBuf,
        /// Expected digest.
        expected: String,
        /// Actual digest computed from the file.
        actual: String,
    },

    /// The WAV input cannot be transcribed as-is.
    #[error("unsupported WAV input: {0}")]
    UnsupportedWav(String),

    /// The WAV data chunk could not be decoded.
    #[error("WAV decoding failed: {0}")]
    Wav(#[from] hound::Error),

    /// whisper.cpp refused to load the model or run inference.
    #[error("whisper error: {0}")]
    Whisper(#[from] sublayer_whisper::WhisperError),

    /// The blocking transcription worker panicked.
    #[error("background transcription task failed: {0}")]
    TaskJoin(String),

    /// Filesystem interaction failed.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}
