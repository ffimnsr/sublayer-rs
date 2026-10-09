//! Caption segmentation, ASS/SRT/VTT compilation, and theme presets.
//!
//! This crate turns the word stream produced by `sublayer-ai` into readable
//! caption cards, then compiles them into the formats an FFmpeg burn-in pass
//! (or a player) can consume. It deliberately depends only on `sublayer-core`
//! so the compilation pipeline is testable without FFmpeg or a speech model.

pub mod ass;
pub mod error;
pub mod fonts;
mod metrics;
pub mod segmenter;
pub mod srt_vtt;
pub mod themes;

pub use ass::build_ass_script;
pub use error::SubtitleError;
pub use fonts::{FONTS_DIR_ENV, resolve_fonts_dir};
pub use segmenter::{SegmenterConfig, segment_words};
pub use srt_vtt::{build_srt, build_vtt};
pub use themes::{PRESET_NAMES, from_json, preset, preset_names, to_json};
