//! Error type shared by the studio UI and its background bridge.

use sublayer_ai::AiError;
use sublayer_core::CoreError;
use sublayer_media::MediaError;
use sublayer_subtitles::SubtitleError;

/// Errors raised while running the desktop studio.
#[derive(Debug, thiserror::Error)]
pub enum UiError {
    /// Domain or project-file error.
    #[error(transparent)]
    Core(#[from] CoreError),

    /// Media tooling error.
    #[error(transparent)]
    Media(#[from] MediaError),

    /// Speech-pipeline error.
    #[error(transparent)]
    Ai(#[from] AiError),

    /// Caption compilation error.
    #[error(transparent)]
    Subtitles(#[from] SubtitleError),

    /// Filesystem interaction failed.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// The Slint event loop or window adapter failed.
    #[error("window error: {0}")]
    Platform(#[from] slint::PlatformError),
}
