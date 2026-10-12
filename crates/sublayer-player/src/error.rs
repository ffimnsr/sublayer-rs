//! Errors surfaced by the player shell.

/// Errors raised by window construction and the UI backend.
#[derive(Debug, thiserror::Error)]
pub enum PlayerError {
    /// The Slint platform rejected a window or event-loop operation.
    #[error("window error: {0}")]
    Slint(#[from] slint::PlatformError),
}
