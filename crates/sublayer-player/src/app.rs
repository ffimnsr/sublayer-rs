//! UI wiring: callback handlers, the pump timer, and event handling.
//!
//! The pump timer advances the playhead on the wall clock, requests the
//! video frame due at the playhead, and drains bridge events; every pass
//! pushes the session properties into the Slint tree.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use slint::{ComponentHandle, Image, ModelRc, SharedString, Timer, TimerMode, VecModel, Weak};

use sublayer_ai::MODELS;

use crate::PlayerWindow;
use crate::bridge::{Bridge, PlayerEvent};
use crate::error::PlayerError;
use crate::session::{PLAYER_LANGUAGES, Session, format_time};

/// Pump cadence; ~30 Hz keeps the playhead and captions smooth.
const PUMP_INTERVAL: Duration = Duration::from_millis(33);
/// Slider seeks closer than this are displayed but not restarted.
const SEEK_DEBOUNCE_MS: u64 = 1_500;

/// The generated window plus all wiring; created once in `main`.
pub struct App {
    pub(crate) ui: PlayerWindow,
    pub(crate) session: Rc<RefCell<Session>>,
    bridge: Bridge,
    events: Receiver<PlayerEvent>,
    _timer: Timer,
}

impl App {
    /// Builds the window, the bridge, and the static option lists.
    pub fn new() -> Result<Self, PlayerError> {
        let ui = PlayerWindow::new()?;
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let bridge = Bridge::new(event_tx);
        let session = Rc::new(RefCell::new(Session::default()));
        session.borrow_mut().model = MODELS[0].name.to_owned();
        // The volume slider is the one source of truth once touched; with a
        // derived default of 0% any interaction would mute playback, so the
        // player boots at full volume matching the driver's own default.
        session.borrow_mut().volume_percent = 100;

        install_static_options(&ui, &session);

        let app = Self {
            ui,
            session,
            bridge,
            events: event_rx,
            _timer: Timer::default(),
        };
        app.wire_callbacks();
        app.start_pump();
        app.refresh();
        Ok(app)
    }

    /// Runs the event loop.
    pub fn run(self) -> Result<(), PlayerError> {
        self.ui.run().map_err(PlayerError::from)?;
        Ok(())
    }

    /// Starts the pump timer; the closure owns its state, so it never
    /// borrows the `App` (the timer outlives nothing after `run`).
    fn start_pump(&self) {
        let weak = self.ui.as_weak();
        let bridge = self.bridge.clone();
        let session = self.session.clone();
        let events = self.events.clone();
        let last_pump = Rc::new(Cell::new(Instant::now()));
        self._timer
            .start(TimerMode::Repeated, PUMP_INTERVAL, move || {
                pump(&weak, &bridge, &session, &events, &last_pump);
            });
    }

    /// Pushes the session state into the window once.
    pub(crate) fn refresh(&self) {
        refresh(&self.ui, &self.session);
    }

    /// Wires every Slint callback to a session mutation + bridge command.
    fn wire_callbacks(&self) {
        let weak = self.ui.as_weak();
        let bridge = self.bridge.clone();
        let session = self.session.clone();
        self.ui.on_open_clicked(move || {
            let Some(ui) = weak.upgrade() else { return };
            let Some(path) = open_video_dialog() else {
                return;
            };
            session.borrow_mut().status = "probing video…".to_owned();
            bridge.open_video(path);
            ui.set_status(session.borrow().status.clone().into());
        });

        let weak = self.ui.as_weak();
        let bridge = self.bridge.clone();
        let session = self.session.clone();
        self.ui.on_pre_transcribe_clicked(move || {
            let Some(ui) = weak.upgrade() else { return };
            let mut guard = session.borrow_mut();
            if guard.video.is_none() {
                guard.status = "open a video first".to_owned();
                drop(guard);
                ui.set_status("open a video first".into());
                return;
            }
            // Batch-transcribe the whole track; playback then shows captions
            // (and translations) from the pre-built timeline instantly.
            guard.prebuilding = true;
            guard.prebuilt_ready = false;
            guard.prebuilt_clear_for_batch();
            guard.status = "pre-transcribing…".to_owned();
            drop(guard);
            bridge.pre_transcribe();
            ui.set_status("pre-transcribing…".into());
        });

        let weak = self.ui.as_weak();
        let bridge = self.bridge.clone();
        let session = self.session.clone();
        self.ui.on_play_toggled(move || {
            let Some(ui) = weak.upgrade() else { return };
            toggle_play(&ui, &bridge, &session);
        });

        let weak = self.ui.as_weak();
        self.ui.on_escape_fullscreen(move || {
            if let Some(ui) = weak.upgrade() {
                let window = ui.window();
                if window.is_fullscreen() {
                    window.set_fullscreen(false);
                }
            }
        });

        let weak = self.ui.as_weak();
        let bridge = self.bridge.clone();
        let session = self.session.clone();
        self.ui.on_stop_clicked(move || {
            let Some(ui) = weak.upgrade() else { return };
            session.borrow_mut().stop_transport();
            bridge.stop();
            refresh_playback(&ui, &session);
            refresh_caption(&ui, &session);
        });

        let weak = self.ui.as_weak();
        let bridge = self.bridge.clone();
        let session = self.session.clone();
        self.ui.on_seek_by(move |delta_ms| {
            let Some(ui) = weak.upgrade() else { return };
            // Bind the target first: the temporary `Ref` from `borrow()`
            // would otherwise live until `apply_seek` returns, and its
            // `borrow_mut()` would panic with a double borrow.
            let target = session.borrow().playhead_ms as i64 + i64::from(delta_ms);
            apply_seek(&ui, &bridge, &session, target);
        });

        let weak = self.ui.as_weak();
        let bridge = self.bridge.clone();
        let session = self.session.clone();
        self.ui.on_seek(move |ms| {
            let Some(ui) = weak.upgrade() else { return };
            apply_seek(&ui, &bridge, &session, ms as i64);
        });

        let weak = self.ui.as_weak();
        let bridge = self.bridge.clone();
        let session = self.session.clone();
        self.ui.on_volume_changed(move |percent| {
            let Some(ui) = weak.upgrade() else { return };
            let percent = percent.clamp(0.0, 100.0) as u8;
            session.borrow_mut().volume_percent = percent;
            bridge.set_volume(percent);
            ui.set_volume(f32::from(percent));
        });

        let session = self.session.clone();
        let bridge = self.bridge.clone();
        self.ui.on_mute_toggled(move || {
            let mut guard = session.borrow_mut();
            guard.muted = !guard.muted;
            bridge.set_mute(guard.muted);
        });

        let weak = self.ui.as_weak();
        self.ui.on_fullscreen_toggled(move || {
            if let Some(ui) = weak.upgrade() {
                let window = ui.window();
                window.set_fullscreen(!window.is_fullscreen());
            }
        });

        let weak = self.ui.as_weak();
        let bridge = self.bridge.clone();
        let session = self.session.clone();
        self.ui.on_translate_toggled(move |enabled| {
            let Some(ui) = weak.upgrade() else { return };
            let mut guard = session.borrow_mut();
            guard.translate_enabled = enabled;
            if enabled {
                match guard.translation_pair() {
                    Some(pair) => {
                        guard.status = format!("translating {} → English", pair.display_name)
                    }
                    None => {
                        guard.status =
                            "no bundled pair for this language; translation off".to_owned()
                    }
                }
            } else {
                guard.status.clear();
            }
            let playhead = guard.playhead_ms;
            guard.refresh_caption(playhead);
            ui.set_status(guard.status.clone().into());
            ui.set_caption_text(guard.caption_text.clone().into());
            bridge.set_translate(enabled);
        });

        let weak = self.ui.as_weak();
        let bridge = self.bridge.clone();
        let session = self.session.clone();
        self.ui.on_model_selected(move |text| {
            let Some(ui) = weak.upgrade() else { return };
            let model = MODELS
                .iter()
                .find(|spec| spec.name == text.as_str())
                .map(|spec| spec.name)
                .unwrap_or(MODELS[0].name);
            let mut guard = session.borrow_mut();
            if guard.model == model {
                return;
            }
            guard.model = model.to_owned();
            bridge.set_model(model.to_owned());
            if guard.playing {
                guard.status = "restarting with the new model…".to_owned();
                guard.clear_captions();
                bridge.restart(guard.playhead_ms);
                ui.set_status(guard.status.clone().into());
            }
        });

        let weak = self.ui.as_weak();
        let bridge = self.bridge.clone();
        let session = self.session.clone();
        self.ui.on_language_selected(move |text| {
            let Some(ui) = weak.upgrade() else { return };
            let language = PLAYER_LANGUAGES
                .iter()
                .find(|(display, _)| *display == text.as_str())
                .and_then(|(_, code)| *code)
                .map(str::to_owned);
            let mut guard = session.borrow_mut();
            if guard.language == language {
                return;
            }
            guard.language = language.clone();
            bridge.set_language(language.clone());
            if guard.playing {
                guard.status = "restarting with the new language…".to_owned();
                guard.clear_captions();
                bridge.restart(guard.playhead_ms);
                ui.set_status(guard.status.clone().into());
            }
        });
    }
}

/// The pump: advance playback, advance the video decoder, drain events,
/// refresh. Runs on the UI thread; `last_pump` lives in a caller-owned cell
/// because the timer closure owns its state.
pub(crate) fn pump(
    weak: &Weak<PlayerWindow>,
    bridge: &Bridge,
    session: &Rc<RefCell<Session>>,
    events: &Receiver<PlayerEvent>,
    last_pump: &Rc<Cell<Instant>>,
) {
    let Some(ui) = weak.upgrade() else { return };
    let now = Instant::now();
    let elapsed = now.saturating_duration_since(last_pump.replace(now));
    let mut guard = session.borrow_mut();
    if guard.playing {
        let duration = guard.duration_ms();
        let playhead = guard.playhead_ms.saturating_add(elapsed.as_millis() as u64);
        guard.playhead_ms = playhead.min(duration);
        if guard.playhead_ms >= duration {
            guard.playing = false;
            guard.status = "finished".to_owned();
        }
    }
    let playhead = guard.playhead_ms;
    guard.refresh_caption(playhead);
    drop(guard);
    refresh_playback(&ui, session);

    // The video decoder reads on demand: advancing every tick only costs a
    // command when no frame is due, and a frame decode when one is.
    if session.borrow().playing {
        bridge.request_frame(playhead);
    }

    while let Ok(event) = events.try_recv() {
        handle_event(&ui, bridge, session, event);
    }
    refresh(&ui, session);
}

pub(crate) fn handle_event(
    ui: &PlayerWindow,
    bridge: &Bridge,
    session: &Rc<RefCell<Session>>,
    event: PlayerEvent,
) {
    let mut guard = session.borrow_mut();
    match &event {
        PlayerEvent::Opened { title, duration_ms } => {
            guard.open_video(title.clone(), *duration_ms);
            ui.set_can_seek(true);
            ui.set_status("".into());
        }
        PlayerEvent::Status(status) => guard.status = status.clone(),
        PlayerEvent::Words(words) => {
            let sentences = guard.feed_words(words.clone());
            let translate_on = guard.translate_enabled && guard.translation_available();
            let playhead = guard.playhead_ms;
            guard.refresh_caption(playhead);
            if translate_on {
                for sentence in sentences {
                    bridge.translate_sentence(sentence);
                }
            }
        }
        PlayerEvent::LanguageDetected(language) => {
            guard.detected_language = Some(language.clone());
            if guard.translate_enabled && !guard.translation_available() {
                guard.status = format!("no bundled pair for `{language}`; translation off");
            }
        }
        PlayerEvent::Translated(out) => {
            if let Some(text) = out.last() {
                if guard.prebuilding || guard.has_untranslated_cards() {
                    // Batch mode: the translation belongs to the pre-built
                    // card the worker just finished (submitted in order).
                    guard.attach_prebuilt_translation(text.clone());
                } else {
                    guard.set_translation(text.clone());
                }
            }
        }
        PlayerEvent::Frame { pts_ms, frame } => {
            // The decoder streams frames as they come due, so a frame whose
            // timestamp drifted more than a pump tick from the playhead is
            // stale (decode lag) or a pre-seek straggler; skipping it keeps
            // the picture tracking the playhead.
            if pts_ms.abs_diff(guard.playhead_ms) <= 200 {
                let mut buffer =
                    slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(frame.width, frame.height);
                buffer.make_mut_bytes().copy_from_slice(&frame.pixels);
                ui.set_frame(Image::from_rgba8(buffer));
            }
        }
        PlayerEvent::PreTranscribed => {
            guard.mark_prebuilt_ready();
            let translating = if guard.translate_enabled && !guard.translations_done() {
                " · translating"
            } else {
                ""
            };
            guard.status = format!(
                "captions ready ({} words){translating}",
                guard.prebuilt_word_count()
            );
            let playhead = guard.playhead_ms;
            guard.refresh_caption(playhead);
        }
        PlayerEvent::PlaybackEnded => {
            guard.playing = false;
            guard.playhead_ms = guard.duration_ms();
            guard.status = "finished".to_owned();
        }
    }
    if !guard.playing || matches!(event, PlayerEvent::Words(_) | PlayerEvent::Translated(_)) {
        ui.set_caption_text(guard.caption_text.clone().into());
    }
    ui.set_detected_label(detected_label(&guard).into());
    ui.set_status(guard.status.clone().into());
    ui.set_translate_enabled(guard.translate_enabled);
}

fn detected_label(session: &Session) -> String {
    if session.language.is_some() {
        return String::new();
    }
    let Some(code) = session.detected_language.as_deref() else {
        return "detecting language…".to_owned();
    };
    let display = session
        .translation_pair()
        .map(|pair| pair.display_name)
        .unwrap_or(code);
    let translating = if session.translate_enabled && session.translation_available() {
        " · translating"
    } else {
        ""
    };
    format!("detected: {display}{translating}")
}

/// Play/pause toggle shared by the transport button and the video click.
fn toggle_play(ui: &PlayerWindow, bridge: &Bridge, session: &Rc<RefCell<Session>>) {
    // The RefMut guard releases its borrow on drop, not on last use: it must
    // be dropped before anything else re-borrows the session.
    let mut guard = session.borrow_mut();
    if guard.video.is_none() {
        guard.status = "open a video first".to_owned();
        drop(guard);
        ui.set_status("open a video first".into());
        return;
    }
    if guard.playing {
        guard.playing = false;
        bridge.pause();
    } else {
        let playhead = guard.playhead_ms;
        guard.playing = true;
        guard.last_bridge_seek_ms = playhead;
        bridge.play_at(playhead);
    }
    drop(guard);
    refresh_playback(ui, session);
}

/// Seeks to `ms` (clamped), restarting the audio only when the movement is
/// large enough to matter while playing.
fn apply_seek(ui: &PlayerWindow, bridge: &Bridge, session: &Rc<RefCell<Session>>, ms: i64) {
    let mut guard = session.borrow_mut();
    let duration = guard.duration_ms();
    let clamped = (ms.max(0) as u64).min(duration);
    guard.playhead_ms = clamped;
    let restart = guard.playing && clamped.abs_diff(guard.last_bridge_seek_ms) >= SEEK_DEBOUNCE_MS;
    if restart {
        guard.last_bridge_seek_ms = clamped;
    }
    drop(guard);
    if restart {
        bridge.seek(clamped);
    }
    refresh_playhead(ui, session);
}

fn refresh(ui: &PlayerWindow, session: &Rc<RefCell<Session>>) {
    // All reads are immutable and scoped; `refresh_caption` needs a mutable
    // borrow afterwards, so the guard must be dropped first.
    {
        let guard = session.borrow();
        ui.set_file_title(guard.file_title.clone().into());
        ui.set_status(guard.status.clone().into());
        ui.set_playhead_ms(guard.playhead_ms as i32);
        ui.set_duration_ms(guard.duration_ms() as i32);
        let label = format!(
            "{} / {}",
            format_time(guard.playhead_ms),
            format_time(guard.duration_ms())
        );
        ui.set_time_label(label.into());

        // Keep the static combo selections in sync (assignments never fire
        // the `selected` callbacks, so this cannot loop).
        let model_index = MODELS
            .iter()
            .position(|spec| spec.name == guard.model)
            .unwrap_or(0) as i32;
        let language_index = PLAYER_LANGUAGES
            .iter()
            .position(|(_, code)| *code == guard.language.as_deref())
            .unwrap_or(0) as i32;
        ui.set_model_index(model_index);
        ui.set_language_index(language_index);
        ui.set_volume(f32::from(guard.volume_percent));
        ui.set_muted(guard.muted);
        ui.set_playing(guard.playing);
        ui.set_translate_enabled(guard.translate_enabled);
        ui.set_detected_label(detected_label(&guard).into());
    }
    refresh_playback(ui, session);
    refresh_caption(ui, session);
}

fn refresh_playback(ui: &PlayerWindow, session: &Rc<RefCell<Session>>) {
    let guard = session.borrow();
    ui.set_playing(guard.playing);
    ui.set_can_seek(guard.video.is_some());
}

fn refresh_playhead(ui: &PlayerWindow, session: &Rc<RefCell<Session>>) {
    let guard = session.borrow();
    ui.set_playhead_ms(guard.playhead_ms as i32);
    ui.set_duration_ms(guard.duration_ms() as i32);
    let label = format!(
        "{} / {}",
        format_time(guard.playhead_ms),
        format_time(guard.duration_ms())
    );
    ui.set_time_label(label.into());
}

fn refresh_caption(ui: &PlayerWindow, session: &Rc<RefCell<Session>>) {
    let mut guard = session.borrow_mut();
    let playhead = guard.playhead_ms;
    guard.refresh_caption(playhead);
    ui.set_caption_text(guard.caption_text.clone().into());
}

/// Fills the model/language combos and the initial transport state.
fn install_static_options(ui: &PlayerWindow, session: &Rc<RefCell<Session>>) {
    let names: Vec<SharedString> = MODELS.iter().map(|spec| spec.name.into()).collect();
    ui.set_models(ModelRc::new(VecModel::from(names)));
    let languages: Vec<SharedString> = PLAYER_LANGUAGES
        .iter()
        .map(|(display, _)| SharedString::from(*display))
        .collect();
    ui.set_languages(ModelRc::new(VecModel::from(languages)));
    ui.set_model_index(0);
    ui.set_language_index(0);
    refresh_playback(ui, session);
    refresh_playhead(ui, session);
}

/// Native file picker for the Open button.
fn open_video_dialog() -> Option<std::path::PathBuf> {
    rfd::FileDialog::new()
        .add_filter("video", &["mp4", "mkv", "webm", "mov", "avi", "m4v", "ts"])
        .pick_file()
}

#[cfg(test)]
impl App {
    /// Drives one pump tick on the real wiring (playhead, video decoder,
    /// event drain, caption refresh) without a running event loop.
    pub(crate) fn pump_for_test(&self, last_pump: &Rc<Cell<Instant>>) {
        let weak = self.ui.as_weak();
        pump(&weak, &self.bridge, &self.session, &self.events, last_pump);
    }

    /// Opens `path` through the real bridge (probe + orchestrator).
    pub(crate) fn open_for_test(&self, path: std::path::PathBuf) {
        self.bridge.open_video(path);
    }
}
