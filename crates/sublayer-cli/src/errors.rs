//! Error type of the CLI binary: a thin fan-in over every crate in the
//! pipeline plus the CLI's own argument/state errors.

use std::path::PathBuf;

use sublayer_ai::AiError;
use sublayer_core::CoreError;
use sublayer_media::MediaError;
use sublayer_subtitles::SubtitleError;

/// Errors raised by [`sublayer`](crate) commands.
#[derive(Debug, thiserror::Error)]
pub enum CliError {
    /// The speech pipeline failed.
    #[error(transparent)]
    Ai(#[from] AiError),

    /// Probing or audio extraction failed.
    #[error(transparent)]
    Media(#[from] MediaError),

    /// Segmentation or caption compilation failed.
    #[error(transparent)]
    Subtitle(#[from] SubtitleError),

    /// XDG path resolution failed.
    #[error(transparent)]
    Core(#[from] CoreError),

    /// Filesystem interaction failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// JSON encoding failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),

    /// The input file does not exist.
    #[error("input file {0} does not exist")]
    MissingInput(PathBuf),

    /// The theme argument matched neither a preset alias nor a JSON file.
    #[error(
        "unknown theme `{0}`; use a preset name (tiktok, hormozi, podcast, cyber, cinematic) or a theme JSON path"
    )]
    UnknownTheme(String),

    /// The output extension maps to no caption format.
    #[error("unsupported output format `{0}` (expected .ass, .srt, .vtt, or .json)")]
    UnsupportedOutput(String),

    /// The render pipeline is not built yet.
    #[error("render is not implemented yet; it lands with sublayer-export (phase 5)")]
    RenderDeferred,

    /// The blocking transcription worker panicked.
    #[error("background transcription task failed: {0}")]
    TaskJoin(String),
}
