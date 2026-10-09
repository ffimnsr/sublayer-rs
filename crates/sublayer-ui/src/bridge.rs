//! Tokio ↔ Slint event bridge.
//!
//! Background work (probing, audio extraction, waveform decimation,
//! transcription, frame preview, export) runs on a Tokio runtime owned by the
//! [`Bridge`]. Tasks report through a crossbeam channel that the Slint event
//! loop drains on a timer; each send also wakes the event loop, so a finished
//! task is picked up without waiting for the next tick.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, unbounded};
use sublayer_ai::{MODELS, ModelManager, TranscriberConfig, transcribe_audio};
use sublayer_core::{CaptionSegment, Project, SublayerPaths, VideoMetadata};
use sublayer_export::{
    EncoderPreference, ExportOptions, HardwareEncoder, HardwareProbe, export_project,
    probe_hardware,
};
use sublayer_media::{
    DEFAULT_BUCKETS_PER_SEC, PreviewConfig, RgbaFrame, WaveformCache, decode_frame_at,
    extract_audio_16k, probe_video, waveform_from_wav,
};
use sublayer_subtitles::{SegmenterConfig, build_ass_script, segment_words};
use tokio::runtime::{Handle, Runtime};

use crate::error::UiError;

/// Grace period granted to background tasks when the window closes.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

/// File dialog filters for the video picker.
const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mov", "mkv", "webm", "m4v", "avi"];
/// File dialog filters for project documents.
const PROJECT_EXTENSIONS: &[&str] = &["sublayer", "json"];

/// Events flowing from background tasks to the UI thread.
#[derive(Debug)]
pub enum UiEvent {
    /// A video file was chosen in the open dialog.
    VideoFilePicked { path: PathBuf },
    /// A project file was chosen in the open dialog.
    ProjectFilePicked { path: PathBuf },
    /// A destination was chosen for the project save dialog.
    ProjectSaveFilePicked { path: PathBuf },
    /// A destination was chosen for the caption export dialog.
    ExportFilePicked { path: PathBuf },
    /// A destination was chosen for the video render dialog.
    VideoExportFilePicked { path: PathBuf },
    /// FFprobe finished; the UI can create the project document.
    Probed {
        video_path: PathBuf,
        metadata: VideoMetadata,
    },
    /// Waveform decimation finished.
    WaveformReady { cache: Arc<WaveformCache> },
    /// Transcription produced caption cards.
    WordsReady { segments: Vec<CaptionSegment> },
    /// A preview frame finished decoding.
    PreviewReady {
        token: u64,
        timestamp_ms: u64,
        frame: RgbaFrame,
    },
    /// A preview decode failed (for example past the end of the video).
    PreviewFailed { token: u64 },
    /// A project document finished loading.
    ProjectLoaded { project: Project, path: PathBuf },
    /// A project document finished writing.
    ProjectSaved { path: PathBuf },
    /// A caption file finished writing.
    CaptionsExported { path: PathBuf },
    /// A rendered video finished writing.
    VideoExported { path: PathBuf },
    /// The hardware probe finished; these drive the next render.
    HardwareProbed {
        /// Full probe result, including the VA-API device.
        probe: HardwareProbe,
        /// Encoder renders will use.
        encoder: HardwareEncoder,
    },
    /// Progress of the running video render.
    RenderProgress {
        /// Encoded fraction in `0.0..=1.0`.
        percentage: f32,
        /// Instantaneous encoding speed.
        fps: f32,
        /// Estimated seconds left.
        eta_seconds: f64,
    },
    /// A background task started.
    TaskStarted { label: String },
    /// Fractional progress in `0.0..=1.0` of the running task.
    Progress { fraction: f32 },
    /// The running task finished.
    TaskFinished { status: String },
    /// The running task failed; the message is user-facing.
    Failed { message: String },
}

/// Cloneable sender half handed to background tasks.
#[derive(Debug, Clone)]
struct Emitter {
    sender: Sender<UiEvent>,
}

impl Emitter {
    /// Sends an event and wakes the Slint event loop so it drains promptly.
    fn send(&self, event: UiEvent) {
        if self.sender.send(event).is_ok() {
            // The pump timer drains the channel; the empty callback only
            // nudges the event loop awake when the queue is idle.
            let _ = slint::invoke_from_event_loop(|| {});
        }
    }

    /// Starts a task with a user-facing label.
    fn start(&self, label: impl Into<String>) {
        self.send(UiEvent::TaskStarted {
            label: label.into(),
        });
    }

    /// Reports a failure in the running task.
    fn fail(&self, error: &dyn std::fmt::Display) {
        self.send(UiEvent::Failed {
            message: error.to_string(),
        });
    }
}

/// Requests the transcription pipeline.
#[derive(Debug, Clone)]
pub struct TranscribeRequest {
    /// Source video, re-extracted when the cached WAV is missing.
    pub video_path: PathBuf,
    /// Extracted 16 kHz mono WAV (also the waveform source).
    pub wav_path: PathBuf,
    /// Pinned model name or an explicit GGML path.
    pub model: String,
    /// Request the GPU backend.
    pub use_gpu: bool,
    /// Run the energy VAD first.
    pub enable_vad: bool,
}

/// Owns the Tokio runtime and the channel back to the UI thread.
///
/// Clones share the emitter and the runtime handle but not the runtime
/// itself: dropping a clone never shuts the worker threads down. Only the
/// original instance (the one returned by [`Bridge::new`]) owns and stops the
/// runtime.
#[derive(Debug)]
pub struct Bridge {
    runtime: Option<Runtime>,
    handle: Handle,
    emitter: Emitter,
}

impl Clone for Bridge {
    fn clone(&self) -> Self {
        Self {
            runtime: None,
            handle: self.handle.clone(),
            emitter: self.emitter.clone(),
        }
    }
}

impl Bridge {
    /// Starts the background runtime and the event channel.
    pub fn new() -> Result<(Self, Receiver<UiEvent>), UiError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("sublayer-worker")
            .build()?;
        let handle = runtime.handle().clone();
        let (sender, receiver) = unbounded();
        let bridge = Self {
            runtime: Some(runtime),
            handle,
            emitter: Emitter { sender },
        };
        Ok((bridge, receiver))
    }

    /// Spawns a file dialog on the worker runtime.
    ///
    /// rfd's blocking API is executed on a blocking worker: the portal
    /// round-trip never stalls the Slint thread, and the result arrives as an
    /// event like any other task result.
    pub fn pick_video_file(&self) {
        self.pick_file(DialogKind::Video);
    }

    /// Opens the project-file picker.
    pub fn pick_project_file(&self) {
        self.pick_file(DialogKind::Project);
    }

    /// Opens the project save dialog.
    pub fn pick_project_save_file(&self, default_name: String) {
        self.pick_file(DialogKind::SaveProject { default_name });
    }

    /// Opens the caption export dialog.
    pub fn pick_export_file(&self, default_name: String) {
        self.pick_file(DialogKind::ExportAss { default_name });
    }

    /// Opens the rendered-video destination dialog.
    pub fn pick_video_export_file(&self, default_name: String) {
        self.pick_file(DialogKind::ExportVideo { default_name });
    }

    /// Probes the machine and reports the encoder renders should use.
    pub fn probe_render_encoder(&self, preference: EncoderPreference) {
        let emitter = self.emitter.clone();
        self.handle.spawn(async move {
            let probe = match probe_hardware().await {
                Ok(probe) => probe,
                Err(error) => {
                    tracing::warn!(%error, "hardware probe failed; rendering will use the CPU");
                    HardwareProbe::default()
                }
            };
            let encoder = probe.select(preference).unwrap_or(HardwareEncoder::Cpu);
            emitter.send(UiEvent::HardwareProbed { probe, encoder });
        });
    }

    /// Burns the project's captions into a new video.
    pub fn export_video(
        &self,
        project: Project,
        fonts_dir: PathBuf,
        output: PathBuf,
        options: ExportOptions,
    ) {
        let emitter = self.emitter.clone();
        self.handle.spawn(async move {
            emitter.start(format!("Rendering with {}", options.encoder.label()));
            let (progress_tx, mut progress_rx) = tokio::sync::mpsc::channel(64);
            let render = tokio::spawn({
                let output = output.clone();
                async move { export_project(&project, &output, &fonts_dir, &options, progress_tx).await }
            });
            while let Some(progress) = progress_rx.recv().await {
                emitter.send(UiEvent::RenderProgress {
                    percentage: progress.percentage,
                    fps: progress.current_fps,
                    eta_seconds: progress.eta_seconds,
                });
            }
            match render.await {
                Ok(Ok(())) => emitter.send(UiEvent::VideoExported { path: output }),
                Ok(Err(error)) => emitter.fail(&error),
                Err(error) => emitter.send(UiEvent::Failed {
                    message: format!("render task failed: {error}"),
                }),
            }
        });
    }

    /// Probes a video and builds the waveform; both results arrive as events.
    ///
    /// Audio is always re-extracted: a previously cached WAV belongs to the
    /// video that was open before.
    pub fn open_video(&self, video_path: PathBuf, wav_path: PathBuf, cache_path: PathBuf) {
        let emitter = self.emitter.clone();
        self.handle.spawn(async move {
            emitter.start(format!("Probing {}", video_path.display()));
            let metadata = match probe_video(&video_path).await {
                Ok(metadata) => metadata,
                Err(error) => {
                    emitter.fail(&error);
                    return;
                }
            };
            emitter.send(UiEvent::Probed {
                video_path: video_path.clone(),
                metadata,
            });
            prepare_media(&emitter, video_path, wav_path, cache_path, true).await;
        });
    }

    /// Rebuilds the waveform for an already probed video (used after loading a
    /// project from disk); an existing WAV is reused when present.
    pub fn prepare_media(&self, video_path: PathBuf, wav_path: PathBuf, cache_path: PathBuf) {
        let emitter = self.emitter.clone();
        self.handle.spawn(async move {
            prepare_media(&emitter, video_path, wav_path, cache_path, false).await;
        });
    }

    /// Runs the full transcription pipeline.
    pub fn transcribe(&self, request: TranscribeRequest) {
        let emitter = self.emitter.clone();
        self.handle.spawn(async move {
            let TranscribeRequest {
                video_path,
                wav_path,
                model,
                use_gpu,
                enable_vad,
            } = request;

            if !wav_path.is_file() {
                emitter.start("Extracting 16 kHz audio");
                if let Err(error) = extract_audio_16k(&video_path, &wav_path).await {
                    emitter.fail(&error);
                    return;
                }
            }

            emitter.start(format!("Ensuring model `{model}`"));
            let model_path = if Path::new(&model).is_file() {
                PathBuf::from(&model)
            } else {
                let paths = match SublayerPaths::resolve() {
                    Ok(paths) => paths,
                    Err(error) => {
                        emitter.fail(&error);
                        return;
                    }
                };
                let manager = match ModelManager::new(paths) {
                    Ok(manager) => manager,
                    Err(error) => {
                        emitter.fail(&error);
                        return;
                    }
                };
                match manager.ensure_cached(&model).await {
                    Ok(path) => path,
                    Err(error) => {
                        emitter.fail(&error);
                        return;
                    }
                }
            };

            let config = TranscriberConfig {
                model_path,
                language: None,
                enable_vad,
                use_gpu,
                ..TranscriberConfig::default()
            };
            emitter.start("Transcribing");
            let words = match transcribe_with_progress(&emitter, &wav_path, &config).await {
                Ok(words) => words,
                Err(message) => {
                    emitter.send(UiEvent::Failed { message });
                    return;
                }
            };

            let segments = segment_words(&words, &SegmenterConfig::default());
            let status = format!(
                "Transcribed {} words into {} caption cards",
                words.len(),
                segments.len()
            );
            emitter.send(UiEvent::WordsReady { segments });
            emitter.send(UiEvent::TaskFinished { status });
        });
    }

    /// Decodes one preview frame for the viewport.
    pub fn decode_preview(
        &self,
        token: u64,
        video_path: PathBuf,
        timestamp_ms: u64,
        metadata: VideoMetadata,
    ) {
        let emitter = self.emitter.clone();
        self.handle.spawn(async move {
            let config = PreviewConfig::default();
            let result = decode_frame_at(
                &video_path,
                Duration::from_millis(timestamp_ms),
                &metadata,
                &config,
            )
            .await;
            match result {
                Ok(frame) => emitter.send(UiEvent::PreviewReady {
                    token,
                    timestamp_ms,
                    frame,
                }),
                Err(error) => {
                    tracing::debug!(timestamp_ms, %error, "preview frame unavailable");
                    emitter.send(UiEvent::PreviewFailed { token });
                }
            }
        });
    }

    /// Loads a project document from disk.
    pub fn open_project(&self, path: PathBuf) {
        let emitter = self.emitter.clone();
        self.handle
            .spawn_blocking(move || match Project::load(&path) {
                Ok(project) => emitter.send(UiEvent::ProjectLoaded { project, path }),
                Err(error) => emitter.fail(&error),
            });
    }

    /// Atomically writes a project document.
    pub fn save_project(&self, project: Project, path: PathBuf) {
        let emitter = self.emitter.clone();
        self.handle
            .spawn_blocking(move || match project.save(&path) {
                Ok(()) => emitter.send(UiEvent::ProjectSaved { path }),
                Err(error) => emitter.fail(&error),
            });
    }

    /// Compiles the captions into an ASS file.
    pub fn export_ass(&self, project: Project, fonts_dir: PathBuf, output: PathBuf) {
        let emitter = self.emitter.clone();
        self.handle.spawn_blocking(move || {
            let result = (|| -> Result<(), UiError> {
                let script = build_ass_script(
                    &project.segments,
                    &project.theme,
                    &project.video_metadata,
                    &fonts_dir,
                )?;
                if let Some(parent) = output
                    .parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&output, script)?;
                Ok(())
            })();
            match result {
                Ok(()) => emitter.send(UiEvent::CaptionsExported { path: output }),
                Err(error) => {
                    emitter.fail(&error);
                }
            }
        });
    }

    /// Shared implementation of the file-picker helpers.
    fn pick_file(&self, kind: DialogKind) {
        let emitter = self.emitter.clone();
        self.handle.spawn_blocking(move || {
            let mut dialog = rfd::FileDialog::new();
            dialog = match &kind {
                DialogKind::Video => dialog.add_filter("Video", VIDEO_EXTENSIONS),
                DialogKind::Project => dialog.add_filter("Sublayer project", PROJECT_EXTENSIONS),
                DialogKind::SaveProject { default_name } => dialog
                    .add_filter("Sublayer project", PROJECT_EXTENSIONS)
                    .set_file_name(default_name),
                DialogKind::ExportAss { default_name } => dialog
                    .add_filter("ASS subtitles", &["ass"])
                    .set_file_name(default_name),
                DialogKind::ExportVideo { default_name } => dialog
                    .add_filter("Video", &["mp4", "mkv", "mov", "webm"])
                    .set_file_name(default_name),
            };
            let picked = match kind {
                DialogKind::SaveProject { .. }
                | DialogKind::ExportAss { .. }
                | DialogKind::ExportVideo { .. } => dialog.save_file(),
                _ => dialog.pick_file(),
            };
            if let Some(event) = kind.event(picked) {
                emitter.send(event);
            }
        });
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        // Never block window teardown on a download or a long encode: give
        // running tasks a short grace period, then abandon them.
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_timeout(SHUTDOWN_GRACE);
        }
    }
}

/// Runs the extraction + waveform pair shared by "open video" and "open
/// project".
async fn prepare_media(
    emitter: &Emitter,
    video_path: PathBuf,
    wav_path: PathBuf,
    cache_path: PathBuf,
    force_extract: bool,
) {
    if force_extract || !wav_path.is_file() {
        emitter.start("Extracting 16 kHz audio");
        if let Err(error) = extract_audio_16k(&video_path, &wav_path).await {
            emitter.fail(&error);
            return;
        }
    }

    emitter.start("Building waveform");
    let result = tokio::task::spawn_blocking(move || -> Result<WaveformCache, String> {
        let cache =
            waveform_from_wav(&wav_path, DEFAULT_BUCKETS_PER_SEC).map_err(|e| e.to_string())?;
        // Best-effort persistence: a failed cache write must not lose the
        // in-memory waveform.
        if let Err(error) = cache.save(&cache_path) {
            tracing::warn!(%error, "waveform cache was not written");
        }
        Ok(cache)
    })
    .await;

    match result {
        Ok(Ok(cache)) => {
            emitter.send(UiEvent::WaveformReady {
                cache: Arc::new(cache),
            });
            emitter.send(UiEvent::TaskFinished {
                status: "Media ready".to_owned(),
            });
        }
        Ok(Err(message)) => emitter.send(UiEvent::Failed { message }),
        Err(error) => emitter.send(UiEvent::Failed {
            message: format!("waveform task failed: {error}"),
        }),
    }
}

/// Drives transcription while forwarding whisper's progress to the UI.
async fn transcribe_with_progress(
    emitter: &Emitter,
    wav_path: &Path,
    config: &TranscriberConfig,
) -> Result<Vec<sublayer_core::WordToken>, String> {
    let (progress_tx, mut progress_rx) = tokio::sync::mpsc::channel(64);
    let wav_path = wav_path.to_path_buf();
    let config = config.clone();
    let task = tokio::spawn(async move { transcribe_audio(&wav_path, &config, progress_tx).await });

    while let Some(fraction) = progress_rx.recv().await {
        emitter.send(UiEvent::Progress { fraction });
    }

    let words = task
        .await
        .map_err(|error| format!("transcription task failed: {error}"))?
        .map_err(|error| error.to_string())?;
    Ok(words)
}

/// Which file dialog to show.
#[derive(Debug, Clone)]
enum DialogKind {
    Video,
    Project,
    SaveProject { default_name: String },
    ExportAss { default_name: String },
    ExportVideo { default_name: String },
}

impl DialogKind {
    /// Converts a dialog result into the matching event.
    fn event(&self, path: Option<PathBuf>) -> Option<UiEvent> {
        let path = path?;
        Some(match self {
            Self::Video => UiEvent::VideoFilePicked { path },
            Self::Project => UiEvent::ProjectFilePicked { path },
            Self::SaveProject { .. } => UiEvent::ProjectSaveFilePicked { path },
            Self::ExportAss { .. } => UiEvent::ExportFilePicked { path },
            Self::ExportVideo { .. } => UiEvent::VideoExportFilePicked { path },
        })
    }
}

/// Model names offered by the transcription settings combo, in pinned order.
pub fn model_names() -> Vec<String> {
    MODELS.iter().map(|spec| spec.name.to_owned()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_names_follow_the_pinned_catalog() {
        let names = model_names();
        assert_eq!(names.len(), MODELS.len());
        assert!(names.contains(&"base.en".to_owned()));
    }

    #[test]
    fn dialog_results_map_to_typed_events() {
        let video = DialogKind::Video
            .event(Some(PathBuf::from("/tmp/clip.mp4")))
            .unwrap();
        assert!(matches!(video, UiEvent::VideoFilePicked { .. }));
        assert!(DialogKind::Video.event(None).is_none());
        let export = DialogKind::ExportAss {
            default_name: "subs.ass".to_owned(),
        }
        .event(Some(PathBuf::from("/tmp/subs.ass")))
        .unwrap();
        assert!(matches!(export, UiEvent::ExportFilePicked { .. }));
        let video_export = DialogKind::ExportVideo {
            default_name: "clip-captioned.mp4".to_owned(),
        }
        .event(Some(PathBuf::from("/tmp/clip-captioned.mp4")))
        .unwrap();
        assert!(matches!(
            video_export,
            UiEvent::VideoExportFilePicked { .. }
        ));
    }

    #[test]
    fn events_reach_the_ui_receiver() {
        let (bridge, receiver) = Bridge::new().unwrap();
        bridge.emitter.send(UiEvent::TaskFinished {
            status: "done".to_owned(),
        });
        let event = receiver.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(matches!(event, UiEvent::TaskFinished { status } if status == "done"));
    }

    #[test]
    fn missing_video_fails_without_media_probing() {
        let (bridge, receiver) = Bridge::new().unwrap();
        bridge.open_video(
            PathBuf::from("/nonexistent/sublayer-missing.mp4"),
            PathBuf::from("/nonexistent/audio.wav"),
            PathBuf::from("/nonexistent/waveform.cache"),
        );

        let mut saw_started = false;
        while let Ok(event) = receiver.recv_timeout(Duration::from_secs(30)) {
            match event {
                UiEvent::TaskStarted { .. } => saw_started = true,
                UiEvent::Failed { message } => {
                    assert!(saw_started, "failure must follow the probe label");
                    assert!(!message.is_empty());
                    return;
                }
                _ => {}
            }
        }
        panic!("no failure event arrived for a missing video");
    }
}
