//! Errors raised by the caption segmentation and compilation pipeline.

/// Errors raised by [`sublayer-subtitles`](crate) operations.
#[derive(Debug, thiserror::Error)]
pub enum SubtitleError {
    /// A custom theme file could not be read from disk.
    #[error("theme file could not be read: {0}")]
    Io(#[from] std::io::Error),

    /// A custom theme (or theme JSON) failed to parse. Color literals that are
    /// neither `#RRGGBB` nor `#RRGGBBAA` surface through the serde data error.
    #[error("theme JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
}
