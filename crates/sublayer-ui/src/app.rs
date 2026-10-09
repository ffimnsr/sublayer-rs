//! Window wiring: Slint callbacks in, background events out.
//!
//! The app owns the window, the bridge, and the shared [`Session`]; the pump
//! timer drains bridge events on the UI thread and refreshes exactly the
//! properties each event touches.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use crossbeam_channel::Receiver;
use slint::{Color, ComponentHandle, ModelRc, SharedString, VecModel, Weak};
use sublayer_core::{Project, Rgba};
use sublayer_subtitles::PRESET_NAMES;

use crate::adapters;
use crate::bridge::{Bridge, TranscribeRequest, UiEvent, model_names};
use crate::error::UiError;
use crate::session::{DragMode, PreviewOutcome, Session};
use crate::{MainWindow, ThemeData};

/// Poll interval of the bridge pump; 60 Hz keeps scrubbing responsive.
const PUMP_INTERVAL: Duration = Duration::from_millis(16);

/// Zoom step of the toolbar buttons and Ctrl+Wheel.
const ZOOM_STEP: f32 = 1.4;

/// Files owned by one GUI session: extracted audio, waveform cache, fonts.
struct MediaFiles {
    /// The temporary directory keeping the paths below alive.
    _temp_dir: tempfile::TempDir,
    /// 16 kHz mono WAV shared by waveform generation and transcription.
    wav_path: PathBuf,
    /// Persisted waveform cache (best effort).
    cache_path: PathBuf,
    /// Font directory handed to libass during export.
    fonts_dir: PathBuf,
}

/// The running desktop application.
pub struct App {
    ui: MainWindow,
    /// Owns the worker runtime; clones handed to callbacks are cheap handles.
    _bridge: Bridge,
    session: Rc<RefCell<Session>>,
    /// Event channel drained by the pump timer.
    _events: Receiver<UiEvent>,
    /// WAV, cache, and font paths for the session's lifetime.
    _media: Rc<MediaFiles>,
    /// Drives [`pump`]; stopped when the app is dropped.
    _timer: slint::Timer,
}

impl App {
    /// Builds the window, starts the worker runtime, and wires everything up.
    pub fn new() -> Result<Self, UiError> {
        let ui = MainWindow::new()?;
        let (bridge, events) = Bridge::new()?;
        let session = Rc::new(RefCell::new(Session::default()));

        let temp_dir = tempfile::Builder::new().prefix("sublayer-ui-").tempdir()?;
        let media = Rc::new(MediaFiles {
            wav_path: temp_dir.path().join("audio-16k.wav"),
            cache_path: temp_dir.path().join("waveform.cache"),
            fonts_dir: sublayer_subtitles::resolve_fonts_dir(),
            _temp_dir: temp_dir,
        });

        install_static_options(&ui);
        wire_callbacks(&ui, &bridge, &session, &media);

        let timer = slint::Timer::default();
        {
            let weak = ui.as_weak();
            let bridge = bridge.clone();
            let session = Rc::clone(&session);
            let events = events.clone();
            let media = Rc::clone(&media);
            timer.start(slint::TimerMode::Repeated, PUMP_INTERVAL, move || {
                pump(&weak, &bridge, &session, &events, &media)
            });
        }

        let app = Self {
            ui,
            _bridge: bridge,
            session,
            _events: events,
            _media: media,
            _timer: timer,
        };
        app.refresh_initial();
        Ok(app)
    }

    /// Runs the Slint event loop until the window closes.
    pub fn run(&self) -> Result<(), UiError> {
        self.ui.run()?;
        Ok(())
    }

    /// Pushes the initial property values (models, labels, status).
    fn refresh_initial(&self) {
        let session = self.session.borrow();
        refresh_task(&self.ui, &session);
        refresh_document(&self.ui, &session);
        refresh_segments(&self.ui, &session);
        refresh_playhead(&self.ui, &session);
        refresh_timeline(&self.ui, &session);
    }
}

/// Fills the static option lists (models, presets, alignments, animations).
pub(crate) fn install_static_options(ui: &MainWindow) {
    let names: Vec<SharedString> = model_names().into_iter().map(SharedString::from).collect();
    let default_index = names
        .iter()
        .position(|name| name.as_str() == "base.en")
        .unwrap_or(0) as i32;
    ui.set_model_names(ModelRc::new(VecModel::from(names)));
    ui.set_model_index(default_index);

    ui.set_preset_names(strings(PRESET_NAMES));
    ui.set_alignment_names(strings(&[
        "Bottom left",
        "Bottom center",
        "Bottom right",
        "Middle left",
        "Middle center",
        "Middle right",
        "Top left",
        "Top center",
        "Top right",
    ]));
    ui.set_animation_names(strings(&["None", "Word pop", "Karaoke", "Bounce"]));
}

/// Builds a Slint string model from a slice of string literals.
fn strings(values: &[&str]) -> ModelRc<SharedString> {
    let values: Vec<SharedString> = values
        .iter()
        .map(|value| SharedString::from(*value))
        .collect();
    ModelRc::new(VecModel::from(values))
}

/// Registers every UI callback.
fn wire_callbacks(
    ui: &MainWindow,
    bridge: &Bridge,
    session: &Rc<RefCell<Session>>,
    media: &Rc<MediaFiles>,
) {
    // Owned clone: closures below must be `'static`, and `Bridge::clone`
    // deliberately does not carry the runtime ownership.
    let bridge = Bridge::clone(bridge);
    let weak = ui.as_weak();

    // --- Header -----------------------------------------------------------
    {
        let bridge = bridge.clone();
        let media = Rc::clone(media);
        let weak = weak.clone();
        let session = Rc::clone(session);
        ui.on_open_video(move |path| {
            let path = PathBuf::from(path.as_str());
            if !path.is_file() {
                set_status(&weak, &session, format!("Not a file: {}", path.display()));
                return;
            }
            bridge.open_video(path, media.wav_path.clone(), media.cache_path.clone());
        });
    }
    {
        let bridge = bridge.clone();
        ui.on_browse_requested(move || bridge.pick_video_file());
    }
    {
        let bridge = bridge.clone();
        ui.on_open_project_requested(move || bridge.pick_project_file());
    }
    {
        let bridge = bridge.clone();
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_save_requested(move || {
            let (project, path) = {
                let session = session.borrow();
                (session.project.clone(), session.project_path.clone())
            };
            match (project, path) {
                (Some(project), Some(path)) => bridge.save_project(project, path),
                (Some(project), None) => {
                    let default = default_project_name(&project);
                    bridge.pick_project_save_file(default);
                }
                (None, _) => set_status(&weak, &session, "Nothing to save yet".to_owned()),
            }
        });
    }
    {
        let bridge = bridge.clone();
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_save_as_requested(move || {
            let project = session.borrow().project.clone();
            match project {
                Some(project) => bridge.pick_project_save_file(default_project_name(&project)),
                None => set_status(&weak, &session, "Nothing to save yet".to_owned()),
            }
        });
    }
    {
        let bridge = bridge.clone();
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_export_requested(move || {
            let project = session.borrow().project.clone();
            match project {
                Some(project) => {
                    let stem = project
                        .video_path
                        .file_stem()
                        .map(|stem| stem.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "captions".to_owned());
                    bridge.pick_export_file(format!("{stem}.ass"));
                }
                None => set_status(&weak, &session, "Open a video before exporting".to_owned()),
            }
        });
    }
    {
        let bridge = bridge.clone();
        let media = Rc::clone(media);
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_transcribe_requested(move || {
            let request = {
                let session = session.borrow();
                let Some(project) = session.project.as_ref() else {
                    return;
                };
                if !project.video_metadata.has_audio() {
                    None
                } else {
                    Some(TranscribeRequest {
                        video_path: project.video_path.clone(),
                        wav_path: media.wav_path.clone(),
                        model: session.model_name.clone(),
                        use_gpu: session.use_gpu,
                        enable_vad: session.enable_vad,
                    })
                }
            };
            match request {
                Some(request) => bridge.transcribe(request),
                None => set_status(&weak, &session, "The source has no audio track".to_owned()),
            }
        });
    }
    {
        let session = Rc::clone(session);
        ui.on_settings_changed(move |model_index, use_gpu, use_vad| {
            let names = model_names();
            let mut session = session.borrow_mut();
            if let Some(name) = usize::try_from(model_index)
                .ok()
                .and_then(|index| names.get(index))
            {
                session.model_name = name.clone();
            }
            session.use_gpu = use_gpu;
            session.enable_vad = use_vad;
        });
    }

    // --- Timeline ---------------------------------------------------------
    {
        let bridge = bridge.clone();
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_seek(move |ms| {
            {
                let mut guard = session.borrow_mut();
                guard.set_playhead(ms.max(0) as u64);
                if let Some(ui) = weak.upgrade() {
                    refresh_playhead(&ui, &guard);
                }
            }
            request_preview(&bridge, &session);
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_scroll_requested(move |delta_x, delta_y, zoom| {
            let viewport_px = weak.upgrade().map_or(0.0, |ui| ui.get_timeline_pixels());
            let mut session = session.borrow_mut();
            if zoom {
                let step = delta_y + delta_x;
                if step == 0.0 {
                    return;
                }
                let factor = if step > 0.0 {
                    1.0 / ZOOM_STEP
                } else {
                    ZOOM_STEP
                };
                session.zoom_by(factor, viewport_px);
            } else {
                let delta_px = if delta_x != 0.0 { delta_x } else { delta_y };
                let delta_ms =
                    (f64::from(delta_px) / f64::from(session.pixels_per_second) * 1_000.0) as i64;
                session.scroll_by(delta_ms, viewport_px);
            }
            if let Some(ui) = weak.upgrade() {
                refresh_timeline(&ui, &session);
            }
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_zoom_in_requested(move || {
            zoom_step(&weak, &session, ZOOM_STEP);
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_zoom_out_requested(move || {
            zoom_step(&weak, &session, 1.0 / ZOOM_STEP);
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_zoom_fit_requested(move || {
            let Some(ui) = weak.upgrade() else {
                return;
            };
            let viewport_px = ui.get_timeline_pixels();
            {
                let mut session = session.borrow_mut();
                session.zoom_fit(viewport_px);
            }
            refresh_timeline(&ui, &session.borrow());
        });
    }
    {
        let bridge = bridge.clone();
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_segment_selected(move |index| {
            let Some(ui) = weak.upgrade() else {
                return;
            };
            {
                let mut session = session.borrow_mut();
                if !session.select(index.max(0) as usize) {
                    return;
                }
                let start = session
                    .selected_segment()
                    .and_then(|(_, segment)| segment.start_ms())
                    .unwrap_or(0);
                session.set_playhead(start);
            }
            {
                let session = session.borrow();
                refresh_segments(&ui, &session);
                refresh_playhead(&ui, &session);
                refresh_timeline(&ui, &session);
            }
            request_preview(&bridge, &session);
        });
    }
    {
        let session = Rc::clone(session);
        ui.on_segment_drag_begin(move |index, mode| {
            let Some(mode) = DragMode::from_code(mode) else {
                return;
            };
            session.borrow_mut().begin_drag(index.max(0) as usize, mode);
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_segment_drag_move(move |_index, delta_px| {
            let delta_ms = {
                let session = session.borrow();
                (f64::from(delta_px) / f64::from(session.pixels_per_second) * 1_000.0) as i64
            };
            let changed = session.borrow_mut().apply_drag(delta_ms).is_some();
            if changed && let Some(ui) = weak.upgrade() {
                refresh_segments(&ui, &session.borrow());
            }
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_segment_drag_end(move || {
            session.borrow_mut().end_drag();
            if let Some(ui) = weak.upgrade() {
                refresh_segments(&ui, &session.borrow());
            }
        });
    }

    // --- Inspector --------------------------------------------------------
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_theme_edited(move |data: ThemeData| {
            let Some(ui) = weak.upgrade() else {
                return;
            };
            {
                let mut session = session.borrow_mut();
                if data.preset_index != session.preset_index {
                    // A preset replaces the whole style; the other fields of
                    // `data` still describe the previous style.
                    if let Some(theme) = adapters::preset_by_index(data.preset_index) {
                        session.apply_preset(data.preset_index, theme);
                    }
                } else if let Some(project) = session.project.as_mut() {
                    adapters::apply_theme_data(&data, &mut project.theme);
                    project.touch();
                }
            }
            refresh_theme(&ui, &session.borrow());
            refresh_playhead(&ui, &session.borrow());
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_caption_edited(move |text: SharedString| {
            let changed = {
                let mut session = session.borrow_mut();
                let Some(index) = session.selected else {
                    return;
                };
                session.set_segment_text(index, text.as_str())
            };
            if changed && let Some(ui) = weak.upgrade() {
                refresh_segments(&ui, &session.borrow());
            }
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_delete_segment_requested(move || {
            let deleted = {
                let mut session = session.borrow_mut();
                let Some(index) = session.selected else {
                    return;
                };
                session.delete_segment(index)
            };
            if deleted && let Some(ui) = weak.upgrade() {
                refresh_segments(&ui, &session.borrow());
            }
        });
    }
}

/// Drains bridge events on the UI thread.
fn pump(
    weak: &Weak<MainWindow>,
    bridge: &Bridge,
    session: &Rc<RefCell<Session>>,
    events: &Receiver<UiEvent>,
    media: &Rc<MediaFiles>,
) {
    let Some(ui) = weak.upgrade() else {
        return;
    };
    while let Ok(event) = events.try_recv() {
        handle_event(&ui, bridge, session, media, event);
    }
}

/// Applies one background event to the session and the window.
fn handle_event(
    ui: &MainWindow,
    bridge: &Bridge,
    session: &Rc<RefCell<Session>>,
    media: &Rc<MediaFiles>,
    event: UiEvent,
) {
    match event {
        UiEvent::VideoFilePicked { path } => {
            ui.set_video_path(path.to_string_lossy().into_owned().into());
            bridge.open_video(path, media.wav_path.clone(), media.cache_path.clone());
        }
        UiEvent::ProjectFilePicked { path } => bridge.open_project(path),
        UiEvent::ProjectSaveFilePicked { path } => {
            let project = session.borrow().project.clone();
            if let Some(project) = project {
                bridge.save_project(project, path);
            }
        }
        UiEvent::ExportFilePicked { path } => {
            let project = session.borrow().project.clone();
            if let Some(project) = project {
                bridge.export_ass(project, media.fonts_dir.clone(), path);
            }
        }
        UiEvent::Probed {
            video_path,
            metadata,
        } => {
            let name = video_path
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| "Untitled project".to_owned());
            let project = Project::new(name, video_path.clone(), metadata);
            let preset_index = adapters::preset_index_of(&project.theme.name);
            {
                let mut session = session.borrow_mut();
                session.install_video(project, preset_index);
                session.zoom_fit(ui.get_timeline_pixels());
            }
            refresh_document(ui, &session.borrow());
            refresh_theme(ui, &session.borrow());
            refresh_segments(ui, &session.borrow());
            refresh_playhead(ui, &session.borrow());
            refresh_timeline(ui, &session.borrow());
            request_preview(bridge, session);
        }
        UiEvent::WaveformReady { cache } => {
            session.borrow_mut().waveform = Some(cache);
            refresh_timeline(ui, &session.borrow());
        }
        UiEvent::WordsReady { segments } => {
            {
                let mut session = session.borrow_mut();
                if let Some(project) = session.project.as_mut() {
                    project.segments = segments;
                    project.touch();
                }
                session.selected = None;
            }
            refresh_segments(ui, &session.borrow());
            refresh_playhead(ui, &session.borrow());
        }
        UiEvent::PreviewReady {
            token,
            timestamp_ms,
            frame,
        } => {
            let is_current = session.borrow().preview.token == token;
            if !is_current {
                return;
            }
            let outcome = session.borrow_mut().preview_ready(token, timestamp_ms);
            ui.set_preview_frame(adapters::frame_to_image(&frame));
            ui.set_has_preview(true);
            if let PreviewOutcome::Retry(ms) = outcome {
                spawn_preview(bridge, session, ms);
            }
        }
        UiEvent::PreviewFailed { token } => {
            let is_current = session.borrow().preview.token == token;
            if is_current {
                session.borrow_mut().preview_failed(token);
            }
        }
        UiEvent::ProjectLoaded { project, path } => {
            let preset_index = adapters::preset_index_of(&project.theme.name);
            let video_path = project.video_path.clone();
            let project_label = path.display().to_string();
            {
                let mut session = session.borrow_mut();
                session.install_loaded(project, path, preset_index);
                session.zoom_fit(ui.get_timeline_pixels());
            }
            refresh_document(ui, &session.borrow());
            refresh_theme(ui, &session.borrow());
            refresh_segments(ui, &session.borrow());
            refresh_playhead(ui, &session.borrow());
            refresh_timeline(ui, &session.borrow());

            if video_path.is_file() {
                bridge.prepare_media(video_path, media.wav_path.clone(), media.cache_path.clone());
                request_preview(bridge, session);
            } else {
                let mut session = session.borrow_mut();
                session.finish_task(format!("Video not found: {}", video_path.display()));
                refresh_task(ui, &session);
            }
            tracing::info!(project = %project_label, "project loaded");
        }
        UiEvent::ProjectSaved { path } => {
            {
                let mut session = session.borrow_mut();
                session.project_path = Some(path.clone());
                session.finish_task(format!("Saved {}", path.display()));
            }
            refresh_task(ui, &session.borrow());
        }
        UiEvent::CaptionsExported { path } => {
            session
                .borrow_mut()
                .finish_task(format!("Wrote {}", path.display()));
            refresh_task(ui, &session.borrow());
        }
        UiEvent::TaskStarted { label } => {
            session.borrow_mut().begin_task(label);
            refresh_task(ui, &session.borrow());
        }
        UiEvent::Progress { fraction } => {
            session.borrow_mut().set_task_progress(fraction);
            refresh_task(ui, &session.borrow());
        }
        UiEvent::TaskFinished { status } => {
            session.borrow_mut().finish_task(status);
            refresh_task(ui, &session.borrow());
        }
        UiEvent::Failed { message } => {
            tracing::error!(%message, "background task failed");
            session
                .borrow_mut()
                .finish_task(format!("Error: {message}"));
            refresh_task(ui, &session.borrow());
        }
    }
}

/// Requests a decode of the current frame, if one is not already running.
fn request_preview(bridge: &Bridge, session: &Rc<RefCell<Session>>) {
    let playhead_ms = session.borrow().playhead_ms;
    spawn_preview(bridge, session, playhead_ms);
}

/// Requests a decode of `timestamp_ms` and hands it to the worker runtime.
fn spawn_preview(bridge: &Bridge, session: &Rc<RefCell<Session>>, timestamp_ms: u64) {
    let mut session = session.borrow_mut();
    let Some((token, timestamp_ms)) = session.request_preview(timestamp_ms) else {
        return;
    };
    let Some(project) = session.project.as_ref() else {
        return;
    };
    let video_path = project.video_path.clone();
    let metadata = project.video_metadata.clone();
    drop(session);
    bridge.decode_preview(token, video_path, timestamp_ms, metadata);
}

/// Writes a one-off status message.
fn set_status(weak: &Weak<MainWindow>, session: &Rc<RefCell<Session>>, message: String) {
    session.borrow_mut().finish_task(message);
    if let Some(ui) = weak.upgrade() {
        refresh_task(&ui, &session.borrow());
    }
}

/// Zooms by `factor` around the playhead and refreshes the timeline.
fn zoom_step(weak: &Weak<MainWindow>, session: &Rc<RefCell<Session>>, factor: f32) {
    let Some(ui) = weak.upgrade() else {
        return;
    };
    let viewport_px = ui.get_timeline_pixels();
    {
        let mut session = session.borrow_mut();
        session.zoom_by(factor, viewport_px);
    }
    refresh_timeline(&ui, &session.borrow());
}

/// Default file name offered by the save/export dialogs.
fn default_project_name(project: &Project) -> String {
    let stem = project
        .video_path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "project".to_owned());
    format!("{stem}.sublayer")
}

/// Converts a domain color into a Slint color.
fn slint_color(color: Rgba) -> Color {
    Color::from_argb_u8(color.a, color.r, color.g, color.b)
}

/// Converts a millisecond count into the `int` range used by the UI.
fn ui_ms(ms: u64) -> i32 {
    ms.min(i32::MAX as u64) as i32
}

/// Pushes the project identity and duration.
pub(crate) fn refresh_document(ui: &MainWindow, session: &Session) {
    match session.project.as_ref() {
        Some(project) => {
            ui.set_has_project(true);
            ui.set_project_name(project.name.as_str().into());
            ui.set_video_path(project.video_path.to_string_lossy().into_owned().into());
            ui.set_video_width(project.video_metadata.width.min(i32::MAX as u32) as i32);
            ui.set_video_height(project.video_metadata.height.min(i32::MAX as u32) as i32);
            ui.set_duration_ms(ui_ms(project.video_metadata.duration_ms()));
        }
        None => {
            ui.set_has_project(false);
            ui.set_project_name("Untitled project".into());
            ui.set_duration_ms(0);
        }
    }
}

/// Pushes the theme into the inspector and the preview overlay.
pub(crate) fn refresh_theme(ui: &MainWindow, session: &Session) {
    let Some(project) = session.project.as_ref() else {
        return;
    };
    let theme = &project.theme;
    ui.set_theme(adapters::theme_data_from(theme));
    ui.set_primary_color(slint_color(theme.primary_color));
    ui.set_highlight_color(slint_color(theme.highlight_color));
    ui.set_outline_color(slint_color(theme.outline_color));
}

/// Pushes the caption cards and the inspector's text editor.
pub(crate) fn refresh_segments(ui: &MainWindow, session: &Session) {
    let (segments, selected) = match session.project.as_ref() {
        Some(project) => (project.segments.as_slice(), session.selected),
        None => (&[][..] as &[sublayer_core::CaptionSegment], None),
    };
    ui.set_segments(adapters::segments_model(segments, selected));
    ui.set_has_selection(session.selected.is_some());
    match session.selected_segment() {
        Some((_, segment)) => {
            ui.set_caption_text(segment.text().as_str().into());
            ui.set_caption_meta(
                session
                    .selected
                    .map(|index| adapters::segment_meta(segments, index))
                    .unwrap_or_default(),
            );
        }
        None => {
            ui.set_caption_text(SharedString::default());
            ui.set_caption_meta(
                if segments.is_empty() {
                    "Transcribe to create caption cards"
                } else {
                    "Select a card on the timeline"
                }
                .into(),
            );
        }
    }
}

/// Pushes the playhead, timecode, and the caption shown over the frame.
pub(crate) fn refresh_playhead(ui: &MainWindow, session: &Session) {
    ui.set_playhead_ms(ui_ms(session.playhead_ms));
    ui.set_preview_timecode(adapters::format_timecode(session.playhead_ms).into());
    let caption = session
        .project
        .as_ref()
        .and_then(|project| adapters::caption_at(project, session.playhead_ms));
    ui.set_show_caption(caption.is_some());
    ui.set_active_caption(caption.unwrap_or_default());
}

/// Pushes zoom, scroll, and the waveform window.
pub(crate) fn refresh_timeline(ui: &MainWindow, session: &Session) {
    ui.set_scroll_ms(ui_ms(session.scroll_ms));
    ui.set_pixels_per_second(session.pixels_per_second);

    let viewport_px = ui.get_timeline_pixels();
    let (base_ms, column_ms, columns) = session.column_plan(viewport_px);
    let buckets = session
        .waveform
        .as_ref()
        .map(|cache| adapters::waveform_window(cache, base_ms, column_ms, columns))
        .unwrap_or_default();
    ui.set_column_base_ms(ui_ms(base_ms));
    ui.set_column_ms(ui_ms(column_ms));
    ui.set_buckets(ModelRc::new(VecModel::from(buckets)));
}

/// Pushes the busy indicator and the status line.
pub(crate) fn refresh_task(ui: &MainWindow, session: &Session) {
    ui.set_busy(session.task.busy);
    ui.set_task_progress(session.task.progress);
    ui.set_status(session.task.status.as_str().into());
}
