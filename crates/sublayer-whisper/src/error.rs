//! Errors raised by the whisper.cpp bindings.

use std::path::PathBuf;

/// Errors raised by [`sublayer-whisper`](crate) operations.
#[derive(Debug, thiserror::Error)]
pub enum WhisperError {
    /// `whisper_init_from_file_with_params` returned null.
    #[error("whisper failed to load the model")]
    ContextLoadFailed,

    /// `whisper_init_state` returned null.
    #[error("whisper failed to create an inference state")]
    StateInitFailed,

    /// The model path cannot be represented as a C string.
    #[error("model path is not valid UTF-8 (or contains a NUL byte): {0:?}")]
    InvalidPath(PathBuf),

    /// [`WhisperState::full`](crate::state::WhisperState::full) rejects empty
    /// sample buffers (whisper.cpp would segfault on them).
    #[error("cannot transcribe an empty sample buffer")]
    NoSamples,

    /// The sample buffer cannot be represented as a C `int` length (~37 hours
    /// of 16 kHz audio); whisper.cpp would truncate it.
    #[error("sample buffer is too large to transcribe in one pass")]
    SamplesTooLarge,

    /// `whisper_full_with_state` returned `-1`.
    #[error("whisper failed to compute the log-mel spectrogram")]
    UnableToCalculateSpectrogram,

    /// `whisper_full_with_state` returned `7`.
    #[error("whisper encoder failed")]
    FailedToEncode,

    /// `whisper_full_with_state` returned `8`.
    #[error("whisper decoder failed")]
    FailedToDecode,

    /// `whisper_full_with_state` returned an unexpected code.
    #[error("whisper returned an unexpected status code: {0}")]
    Generic(i32),
}
