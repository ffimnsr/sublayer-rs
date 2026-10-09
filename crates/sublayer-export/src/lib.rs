//! Hardware-accelerated video rendering for Sublayer.
//!
//! The pipeline is: probe the machine ([`probe_hardware`]) → pick an encoder
//! ([`HardwareProbe::select`]) → burn the compiled ASS script into the source
//! video with [`run_export`], streaming [`ExportProgress`] as FFmpeg reports
//! it. [`export_project`] bundles compilation and rendering for callers that
//! hold a [`sublayer_core::Project`].
//!
//! FFmpeg process handling is shared with `sublayer-media`, so the same
//! `SUBLAYER_FFMPEG` override, kill-on-drop, and error classification apply to
//! renders as to probing and audio extraction.

pub mod error;
pub mod hardware;
pub mod progress;
pub mod runner;

use std::path::Path;

use sublayer_core::Project;
use sublayer_subtitles::build_ass_script;
use tokio::sync::mpsc::Sender;

pub use error::ExportError;
pub use hardware::{
    ENCODER_ENV, EncoderPreference, HardwareEncoder, HardwareProbe, probe_hardware,
};
pub use progress::{ProgressParser, ProgressReport, ProgressTracker};
pub use runner::{ExportOptions, ExportProgress, run_export};

/// Renders `project` into `output_video`, burning its captions in.
///
/// The ASS script is compiled to a temporary file first, so the export honours
/// exactly the same theme, fonts directory, and metadata as the caption
/// export. Missing output directories are created; the FFmpeg encoder is
/// cancelled when `progress_tx` is dropped. The returned encoder is the one
/// that actually produced the file, which may be a fallback when
/// [`ExportOptions::fallbacks`] is populated.
pub async fn export_project(
    project: &Project,
    output_video: &Path,
    fonts_dir: &Path,
    options: &ExportOptions,
    progress_tx: Sender<ExportProgress>,
) -> Result<HardwareEncoder, ExportError> {
    let script = build_ass_script(
        &project.segments,
        &project.theme,
        &project.video_metadata,
        fonts_dir,
    )?;
    let temporary = tempfile::Builder::new()
        .prefix("sublayer-export-")
        .suffix(".ass")
        .tempfile()?;
    std::fs::write(temporary.path(), script)?;

    if let Some(parent) = output_video
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    run_export(
        &project.video_path,
        output_video,
        temporary.path(),
        fonts_dir,
        options,
        progress_tx,
    )
    .await
}
