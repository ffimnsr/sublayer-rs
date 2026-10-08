//! Domain foundation for Sublayer.
//!
//! This crate owns the project document model, XDG directory resolution, and the
//! error type shared by every other crate in the workspace. It deliberately
//! depends on nothing heavier than `serde` so the GUI, the CLI, and headless
//! tests can all link it without FFmpeg, inference runtimes, or UI toolkits.

pub mod error;
pub mod models;
pub mod paths;

pub use error::CoreError;
pub use models::{
    AnimationType, CaptionSegment, PROJECT_SCHEMA_VERSION, Project, Rgba, ThemeStyle,
    VideoMetadata, WordToken,
};
pub use paths::SublayerPaths;
