//! Window wiring: Slint callbacks in, background events out.
//!
//! The app owns the window, the bridge, and the shared [`Session`]; the pump
//! timer drains bridge events on the UI thread and refreshes exactly the
//! properties each event touches.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use slint::{ComponentHandle, SharedString, Weak};
use sublayer_core::Project;
use sublayer_export::{EncoderPreference, ExportOptions};

use crate::adapters;
use crate::bridge::{Bridge, TranscribeRequest, UiEvent, model_names};
use crate::error::UiError;
use crate::session::{DragMode, PreviewOutcome, Session};
use crate::views::{
    UiModels, install_static_options, refresh_caption_drawer, refresh_document, refresh_encoder,
    refresh_playback, refresh_playhead, refresh_playhead_position, refresh_segments, refresh_task,
    refresh_theme, refresh_timeline,
};
use crate::{MainWindow, ThemeData};

/// Poll interval of the bridge pump; 60 Hz keeps scrubbing and playback
/// responsive.
const PUMP_INTERVAL: Duration = Duration::from_millis(16);

/// Registers every bundled font with Slint's shared collection, so the
/// preview resolves the theme's family to the same face libass burns into
/// the render (same files, same shaping widths).
fn register_bundled_fonts() {
    let Ok(entries) = std::fs::read_dir(sublayer_subtitles::resolve_fonts_dir()) else {
        return;
    };
    let mut collection = slint::fontique_011::shared_collection();
    for entry in entries.flatten() {
        let path = entry.path();
        let is_font = matches!(
            path.extension()
                .and_then(|extension| extension.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("ttf") | Some("otf") | Some("ttc")
        );
        if !is_font {
            continue;
        }
        let Ok(data) = std::fs::read(&path) else {
            continue;
        };
        let blob = slint::fontique_011::fontique::Blob::new(std::sync::Arc::new(data));
        collection.register_fonts(blob, None);
    }
}

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
    /// Stable model instances shared with the callbacks and refresh helpers.
    models: Rc<UiModels>,
    /// Drives [`pump`]; stopped when the app is dropped.
    _timer: slint::Timer,
}

impl App {
    /// Builds the window, starts the worker runtime, and wires everything up.
    pub fn new() -> Result<Self, UiError> {
        let ui = MainWindow::new()?;
        // The window brings up the text backend; register the bundled fonts
        // into Slint's shared collection so the preview centers words with
        // the same (fontique-visible) family libass burns into the render.
        register_bundled_fonts();
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
        let models = Rc::new(UiModels::new());
        models.attach(&ui);
        wire_callbacks(&ui, &bridge, &session, &models, &media);
        // Resolve the render encoder in the background; `SUBLAYER_ENCODER`
        // pins a backend, otherwise the probe picks the best one.
        let preference = match EncoderPreference::from_env() {
            Ok(Some(preference)) => preference,
            Ok(None) => EncoderPreference::default(),
            Err(error) => {
                tracing::warn!(%error, "ignoring invalid SUBLAYER_ENCODER");
                EncoderPreference::default()
            }
        };
        bridge.probe_render_encoder(preference);

        let last_tick = Rc::new(Cell::new(Instant::now()));
        let timer = slint::Timer::default();
        {
            let weak = ui.as_weak();
            let bridge = bridge.clone();
            let session = Rc::clone(&session);
            let events = events.clone();
            let media = Rc::clone(&media);
            let models = Rc::clone(&models);
            let last_tick = Rc::clone(&last_tick);
            timer.start(slint::TimerMode::Repeated, PUMP_INTERVAL, move || {
                let now = Instant::now();
                let elapsed = now.saturating_duration_since(last_tick.replace(now));
                pump(&weak, &bridge, &session, &models, &events, &media, elapsed)
            });
        }

        let app = Self {
            ui,
            _bridge: bridge,
            session,
            _events: events,
            _media: media,
            models,
            _timer: timer,
        };
        app.refresh();
        Ok(app)
    }

    /// Runs the Slint event loop until the window closes.
    pub fn run(&self) -> Result<(), UiError> {
        self.ui.run()?;
        Ok(())
    }

    /// Window handle, exposed so headless tests can drive real input events.
    #[cfg(test)]
    pub(crate) fn window(&self) -> &MainWindow {
        &self.ui
    }

    /// Shared editor session, exposed so headless tests can inspect state.
    #[cfg(test)]
    pub(crate) fn session(&self) -> &Rc<RefCell<Session>> {
        &self.session
    }

    /// Advances playback by `elapsed`, exactly like one pump tick.
    #[cfg(test)]
    pub(crate) fn tick(&self, elapsed: Duration) {
        tick_playback(
            &self.ui,
            &self._bridge,
            &self.session,
            &self.models,
            elapsed,
        );
    }

    /// Re-renders every panel from the current session state.
    pub(crate) fn refresh(&self) {
        let session = self.session.borrow();
        let models = &self.models;
        refresh_task(&self.ui, &session);
        refresh_document(&self.ui, &session);
        refresh_theme(&self.ui, &session);
        refresh_segments(&self.ui, &session, models);
        refresh_playhead(&self.ui, &session, models);
        refresh_playback(&self.ui, &session);
        refresh_timeline(&self.ui, &session, models);
        refresh_encoder(&self.ui, &session);
    }
}

/// Registers every UI callback.
fn wire_callbacks(
    ui: &MainWindow,
    bridge: &Bridge,
    session: &Rc<RefCell<Session>>,
    models: &Rc<UiModels>,
    media: &Rc<MediaFiles>,
) {
    let models = Rc::clone(models);
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
        let session = Rc::clone(session);
        let weak = weak.clone();
        ui.on_export_video_requested(move || {
            let name = {
                let session = session.borrow();
                let Some(project) = session.project.as_ref() else {
                    return;
                };
                let stem = project
                    .video_path
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "captioned".to_owned());
                format!("{stem}-captioned.mp4")
            };
            bridge.pick_video_export_file(name);
            let _ = &weak;
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
        let models = Rc::clone(&models);
        ui.on_seek(move |ms| {
            {
                let mut guard = session.borrow_mut();
                guard.set_playhead(ms.max(0) as u64);
                if let Some(ui) = weak.upgrade() {
                    refresh_playhead(&ui, &guard, &models);
                }
            }
            request_preview(&bridge, &session);
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
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
                refresh_timeline(&ui, &session, &models);
            }
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
        ui.on_zoom_in_requested(move || {
            zoom_step(&weak, &session, &models, ZOOM_STEP);
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
        ui.on_zoom_out_requested(move || {
            zoom_step(&weak, &session, &models, 1.0 / ZOOM_STEP);
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
        ui.on_zoom_fit_requested(move || {
            let Some(ui) = weak.upgrade() else {
                return;
            };
            let viewport_px = ui.get_timeline_pixels();
            {
                let mut session = session.borrow_mut();
                session.zoom_fit(viewport_px);
            }
            refresh_timeline(&ui, &session.borrow(), &models);
        });
    }
    {
        let bridge = bridge.clone();
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
        ui.on_play_toggled(move || {
            let Some(ui) = weak.upgrade() else {
                return;
            };
            session.borrow_mut().toggle_playback();
            {
                let session = session.borrow();
                refresh_playback(&ui, &session);
                refresh_playhead(&ui, &session, &models);
            }
            // Starting chases the clock; pausing lands the exact frame.
            request_preview(&bridge, &session);
        });
    }
    {
        let bridge = bridge.clone();
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
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
                refresh_segments(&ui, &session, &models);
                refresh_playhead(&ui, &session, &models);
                refresh_timeline(&ui, &session, &models);
            }
            request_preview(&bridge, &session);
        });
    }
    {
        let session = Rc::clone(session);
        ui.on_segment_drag_begin(move |index, mode, pointer_x| {
            let Some(mode) = DragMode::from_code(mode) else {
                return;
            };
            session
                .borrow_mut()
                .begin_drag(index.max(0) as usize, mode, pointer_x);
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
        ui.on_segment_drag_move(move |pointer_x| {
            let changed = session.borrow_mut().drag_to(pointer_x).is_some();
            if changed && let Some(ui) = weak.upgrade() {
                refresh_segments(&ui, &session.borrow(), &models);
            }
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
        ui.on_segment_drag_end(move || {
            session.borrow_mut().end_drag();
            if let Some(ui) = weak.upgrade() {
                refresh_segments(&ui, &session.borrow(), &models);
            }
        });
    }

    // --- Inspector --------------------------------------------------------
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
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
            refresh_playhead(&ui, &session.borrow(), &models);
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
        ui.on_caption_edited(move |text: SharedString| {
            let changed = {
                let mut session = session.borrow_mut();
                let Some(index) = session.selected else {
                    return;
                };
                session.set_segment_text(index, text.as_str())
            };
            if changed && let Some(ui) = weak.upgrade() {
                refresh_segments(&ui, &session.borrow(), &models);
            }
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
        ui.on_delete_segment_requested(move || {
            let deleted = {
                let mut session = session.borrow_mut();
                let Some(index) = session.selected else {
                    return;
                };
                session.delete_segment(index)
            };
            if deleted && let Some(ui) = weak.upgrade() {
                refresh_segments(&ui, &session.borrow(), &models);
            }
        });
    }

    // --- Captions drawer ----------------------------------------------------
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
        ui.on_caption_drawer_toggled(move || {
            let Some(ui) = weak.upgrade() else {
                return;
            };
            {
                let mut session = session.borrow_mut();
                session.caption_drawer_open = !session.caption_drawer_open;
            }
            refresh_caption_drawer(&ui, &session.borrow(), &models);
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
        let bridge = bridge.clone();
        ui.on_caption_row_selected(move |index: i32| {
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
                refresh_segments(&ui, &session, &models);
                refresh_playhead(&ui, &session, &models);
                refresh_timeline(&ui, &session, &models);
            }
            request_preview(&bridge, &session);
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
        ui.on_caption_drawer_toggled(move || {
            let Some(ui) = weak.upgrade() else {
                return;
            };
            {
                let mut session = session.borrow_mut();
                session.caption_drawer_open = !session.caption_drawer_open;
            }
            refresh_caption_drawer(&ui, &session.borrow(), &models);
        });
    }
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
        ui.on_caption_row_edited(move |index: i32, text: SharedString| {
            let changed = session
                .borrow_mut()
                .set_segment_text(index.max(0) as usize, text.as_str());
            if changed && let Some(ui) = weak.upgrade() {
                refresh_segments(&ui, &session.borrow(), &models);
                refresh_playhead(&ui, &session.borrow(), &models);
            }
        });
    }
    // --- Timeline right-click -------------------------------------------------
    {
        let session = Rc::clone(session);
        let weak = weak.clone();
        let models = Rc::clone(&models);
        let bridge = bridge.clone();
        ui.on_add_caption_requested(move |ms: i32| {
            let Some(ui) = weak.upgrade() else {
                return;
            };
            {
                let mut session = session.borrow_mut();
                if !session.add_caption(ms.max(0) as u64) {
                    return;
                }
            }
            {
                let session = session.borrow();
                refresh_segments(&ui, &session, &models);
                refresh_playhead(&ui, &session, &models);
                refresh_timeline(&ui, &session, &models);
            }
            request_preview(&bridge, &session);
        });
    }
}

/// Drains bridge events on the UI thread.
fn pump(
    weak: &Weak<MainWindow>,
    bridge: &Bridge,
    session: &Rc<RefCell<Session>>,
    models: &UiModels,
    events: &Receiver<UiEvent>,
    media: &Rc<MediaFiles>,
    elapsed: Duration,
) {
    let Some(ui) = weak.upgrade() else {
        return;
    };
    tick_playback(&ui, bridge, session, models, elapsed);
    while let Ok(event) = events.try_recv() {
        handle_event(&ui, bridge, session, models, media, event);
    }
}

/// Advances the playhead on the wall clock and keeps the preview chasing it.
///
/// Only one decode is in flight at a time (see [`Session::request_preview`]),
/// so the frame rate falls wherever FFmpeg lands while the playhead never
/// waits for it.
fn tick_playback(
    ui: &MainWindow,
    bridge: &Bridge,
    session: &Rc<RefCell<Session>>,
    models: &UiModels,
    elapsed: Duration,
) {
    let mut guard = session.borrow_mut();
    if !guard.playing {
        return;
    }
    if !guard.advance_playback(elapsed.as_millis() as u64) {
        return;
    }
    let playing = guard.playing;
    let scroll_changed = playing && guard.follow_playhead(ui.get_timeline_pixels());
    let playhead_ms = guard.playhead_ms;
    refresh_playhead_position(ui, &guard);
    if scroll_changed {
        refresh_timeline(ui, &guard, models);
    }
    if !playing {
        // The tick parked on the last frame; flip the transport button back.
        refresh_playback(ui, &guard);
    }
    drop(guard);
    if playing {
        spawn_preview(bridge, session, playhead_ms);
    }
}

/// Applies one background event to the session and the window.
fn handle_event(
    ui: &MainWindow,
    bridge: &Bridge,
    session: &Rc<RefCell<Session>>,
    models: &UiModels,
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
        UiEvent::VideoExportFilePicked { path } => {
            let path = with_video_extension(path);
            let request = {
                let session = session.borrow();
                session.project.clone().map(|project| {
                    let options = ExportOptions {
                        encoder: session.render_encoder,
                        fallbacks: session.render_fallbacks.clone(),
                        duration_ms: project.video_metadata.duration_ms(),
                        quality: session.render_quality,
                        vaapi_device: session.render_probe.vaapi_device.clone(),
                    };
                    (project, options)
                })
            };
            if let Some((project, options)) = request {
                bridge.export_video(project, media.fonts_dir.clone(), path, options);
            }
        }
        UiEvent::VideoExported { path, encoder } => {
            session.borrow_mut().finish_task(format!(
                "Wrote {} ({})",
                path.display(),
                encoder.label()
            ));
            refresh_task(ui, &session.borrow());
        }
        UiEvent::HardwareProbed {
            probe,
            encoder,
            fallbacks,
        } => {
            session
                .borrow_mut()
                .set_render_encoder(probe, encoder, fallbacks);
            let session = session.borrow();
            refresh_encoder(ui, &session);
        }
        UiEvent::RenderProgress {
            percentage,
            fps,
            eta_seconds,
        } => {
            session
                .borrow_mut()
                .set_render_progress(percentage, fps, eta_seconds);
            refresh_task(ui, &session.borrow());
        }
        UiEvent::Probed {
            video_path,
            metadata,
        } => {
            let name = video_path
                .file_stem()
                .and_then(|stem| project_name_from_stem(&stem.to_string_lossy()))
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
            refresh_segments(ui, &session.borrow(), models);
            refresh_playhead(ui, &session.borrow(), models);
            refresh_playback(ui, &session.borrow());
            refresh_timeline(ui, &session.borrow(), models);
            request_preview(bridge, session);
        }
        UiEvent::WaveformReady { cache } => {
            session.borrow_mut().waveform = Some(cache);
            refresh_timeline(ui, &session.borrow(), models);
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
            refresh_segments(ui, &session.borrow(), models);
            refresh_playhead(ui, &session.borrow(), models);
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
            // While playing, the caption overlay follows the decode rate; the
            // playhead itself already moved on the pump clock.
            if session.borrow().playing {
                refresh_playhead(ui, &session.borrow(), models);
            }
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
            refresh_segments(ui, &session.borrow(), models);
            refresh_playhead(ui, &session.borrow(), models);
            refresh_playback(ui, &session.borrow());
            refresh_timeline(ui, &session.borrow(), models);

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
fn zoom_step(
    weak: &Weak<MainWindow>,
    session: &Rc<RefCell<Session>>,
    models: &UiModels,
    factor: f32,
) {
    let Some(ui) = weak.upgrade() else {
        return;
    };
    let viewport_px = ui.get_timeline_pixels();
    {
        let mut session = session.borrow_mut();
        session.zoom_by(factor, viewport_px);
    }
    refresh_timeline(&ui, &session.borrow(), models);
}

/// Display name for a project opened from a video file.
///
/// Stems that are almost certainly playground assets (`test`, `sample`, …)
/// fall back to the untitled placeholder so the header never advertises a
/// test build; real footage keeps its file name. The save dialog still
/// proposes the raw stem via [`default_project_name`].
pub(crate) fn project_name_from_stem(stem: &str) -> Option<String> {
    const FIXTURE_STEMS: &[&str] = &["test", "clip", "sample", "demo", "example", "untitled"];
    let trimmed = stem.trim().to_ascii_lowercase();
    if trimmed.is_empty() || FIXTURE_STEMS.contains(&trimmed.as_str()) {
        return None;
    }
    Some(stem.trim().to_owned())
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

/// Ensures the render destination names a container FFmpeg can infer.
pub(crate) fn with_video_extension(path: PathBuf) -> PathBuf {
    if path
        .extension()
        .is_some_and(|extension| !extension.is_empty())
    {
        return path;
    }
    path.with_extension("mp4")
}
