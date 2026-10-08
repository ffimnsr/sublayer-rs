//! Error types shared across Sublayer's domain layer.

use std::path::PathBuf;

/// Errors raised by [`sublayer-core`](crate) operations.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// Filesystem interaction failed.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// Project JSON could not be encoded or decoded.
    #[error("project (de)serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),

    /// The project file was written by an unknown, newer schema revision.
    #[error("unsupported project schema version {found} (this build supports up to {supported})")]
    UnsupportedSchemaVersion {
        /// Schema version found in the file.
        found: u32,
        /// Highest schema version understood by this build.
        supported: u32,
    },

    /// The platform could not provide XDG base directories for the user.
    #[error("no base directories available for the current user")]
    BaseDirectoriesUnavailable,

    /// A color literal was neither `#RRGGBB` nor `#RRGGBBAA`.
    #[error("invalid color {0:?} (expected #RRGGBB or #RRGGBBAA)")]
    InvalidColor(String),

    /// A project path had no parent directory to write into.
    #[error("project path {path} has no parent directory")]
    MissingParent {
        /// The offending path.
        path: PathBuf,
    },
}
