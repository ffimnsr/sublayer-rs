//! Headless window tests: instantiate the Slint tree on the testing backend
//! and check the static options and event-driven state.
//!
//! The backend is installed per test thread (`init_no_event_loop`), so these
//! tests never need a Wayland or X11 session.

use std::cell::Cell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use i_slint_backend_testing as testing;
use slint::Model;

use crate::PlayerWindow;
use crate::app::App;
use crate::session::Session;

thread_local! {
    /// Whether this thread already installed the testing backend.
    static PLATFORM_READY: Cell<bool> = const { Cell::new(false) };
}

/// The heavy end-to-end tests saturate the CPU (two whisper decoders plus
/// several ffmpeg pipes when they run at once); a shared lock makes them run
/// one after the other so they cannot starve each other.
static E2E_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Installs the testing backend once per thread (re-initializing panics).
fn ensure_platform() {
    PLATFORM_READY.with(|ready| {
        if ready.get() {
            return;
        }
        testing::init_no_event_loop();
        ready.set(true);
    });
}

#[test]
fn window_builds_with_the_static_option_lists() {
    ensure_platform();
    let app = App::new().expect("testing backend must build the window");
    let ui = &app.ui_for_test();

    assert!(!ui.get_playing());
    assert!(
        ui.get_models().row_count() > 0,
        "model list must be populated"
    );
    assert!(
        ui.get_languages().row_count() >= 7,
        "language list must cover the bundled pairs"
    );
    assert_eq!(ui.get_model_index(), 0);
    assert_eq!(ui.get_language_index(), 0);
    assert_eq!(ui.get_playhead_ms(), 0);
    assert_eq!(ui.get_caption_text(), "");
    assert!(!ui.get_can_seek());
}

#[test]
fn fresh_app_boots_at_full_volume() {
    ensure_platform();
    let app = App::new().expect("testing backend must build the window");
    // A 0% default made the volume slider look mute and any interaction
    // with it silenced playback (the driver only learns the slider value
    // when it moves).
    assert_eq!(app.session_for_test().borrow().volume_percent, 100);
    assert_eq!(app.ui_for_test().get_volume(), 100.0);
}

#[test]
fn session_state_drives_the_overlay() {
    ensure_platform();
    let app = App::new().expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let session = app.session_for_test();

    // Opened media enables the transport and fills the header.
    session
        .borrow_mut()
        .open_video("clip.mp4", 60_000, std::path::PathBuf::from("clip.mp4"));
    app.refresh_for_test();
    assert!(ui.get_can_seek());
    assert!(ui.get_file_title().contains("clip.mp4"));

    // Recognized words land in the overlay (source-language subs).
    let sentences = session.borrow_mut().feed_words(vec![
        sublayer_core::WordToken::new("Hello", 100, 400),
        sublayer_core::WordToken::new("world.", 500, 900),
    ]);
    assert_eq!(sentences, ["Hello world."]);
    session.borrow_mut().refresh_caption(1_500);
    app.refresh_for_test();
    assert_eq!(ui.get_caption_text().as_str(), "Hello world.");

    // Translation on: the English card replaces the source.
    session.borrow_mut().translate_enabled = true;
    session.borrow_mut().set_translation("¡Hola mundo!");
    session.borrow_mut().refresh_caption(1_500);
    app.refresh_for_test();
    assert_eq!(ui.get_caption_text().as_str(), "¡Hola mundo!");
    assert!(ui.get_translate_enabled());
}

#[test]
fn play_without_a_video_hints_in_the_status_line() {
    ensure_platform();
    let app = App::new().expect("testing backend must build the window");
    app.toggle_play_for_test();
    assert!(
        app.session_for_test()
            .borrow()
            .status
            .contains("open a video first")
    );
}

#[test]
fn play_at_the_end_restarts_from_the_start() {
    ensure_platform();
    let app = App::new().expect("testing backend must build the window");
    let session = app.session_for_test();
    session
        .borrow_mut()
        .open_video("clip.mp4", 60_000, std::path::PathBuf::from("clip.mp4"));
    session.borrow_mut().playhead_ms = 60_000; // the media ended
    session
        .borrow_mut()
        .feed_words(vec![sublayer_core::WordToken::new("tail", 55_000, 55_300)]);
    assert!(!session.borrow().playing);

    app.toggle_play_for_test();

    let guard = session.borrow();
    assert!(guard.playing);
    assert_eq!(guard.playhead_ms, 0, "play at the end restarts from zero");
    assert_eq!(
        guard.caption_text, "",
        "stale captions are cleared on restart"
    );
}

#[test]
fn seek_buttons_do_not_double_borrow_the_session() {
    ensure_platform();
    let app = App::new().expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    app.session_for_test().borrow_mut().open_video(
        "clip.mp4",
        60_000,
        std::path::PathBuf::from("clip.mp4"),
    );
    app.refresh_for_test();

    // Regression: the `seek_by` callback used to pass `session.borrow()`
    // into `apply_seek`, whose `borrow_mut` then panicked with a live
    // immutable borrow (fast-forward/backward buttons and the arrow keys).
    ui.invoke_seek_by(-5_000);
    ui.invoke_seek_by(5_000);
    ui.invoke_seek_by(5_000);
    assert_eq!(ui.get_playhead_ms(), 10_000);
}

#[test]
fn caption_words_reach_the_overlay_while_playing() {
    ensure_platform();
    let app = App::new().expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let session = app.session_for_test();
    session
        .borrow_mut()
        .open_video("clip.mp4", 60_000, std::path::PathBuf::from("clip.mp4"));
    session.borrow_mut().playing = true;

    // The live-ASR path: recognized words arrive as player events and must
    // land in the overlay text while playback is running.
    let bridge = crate::bridge::Bridge::new(crossbeam_channel::unbounded().0);
    crate::app::handle_event(
        ui,
        &bridge,
        session,
        crate::bridge::PlayerEvent::Words(vec![sublayer_core::WordToken::new(
            "Hello", 1_000, 1_300,
        )]),
    );
    assert!(ui.get_caption_text().as_str().contains("Hello"));

    // The pump's own refresh (every tick) keeps the overlay populated;
    // words inside the whisper-latency window survive.
    session.borrow_mut().refresh_caption(8_000);
    app.refresh_for_test();
    assert!(ui.get_caption_text().as_str().contains("Hello"));
}

#[test]
fn prebuilt_captions_show_instantly_on_play() {
    let _e2e = E2E_LOCK.lock().unwrap();
    ensure_platform();
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_assets/player-test.webm");
    if !fixture.is_file() {
        eprintln!("skipping: test_assets/player-test.webm not present");
        return;
    }

    let app = App::new().expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let session = app.session_for_test();
    // Route through the real combos so the orchestrator shares the session's
    // model/language (the cache key depends on it).
    ui.invoke_model_selected("tiny".into());
    ui.invoke_language_selected("English".into());
    app.refresh_for_test();

    app.open_for_test(fixture);
    let last_pump = Rc::new(Cell::new(Instant::now()));
    let open_deadline = Instant::now() + Duration::from_secs(15);
    while !ui.get_can_seek() && Instant::now() < open_deadline {
        app.pump_for_test(&last_pump);
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(ui.get_can_seek(), "video must open");

    // Batch-transcribe the whole track, then play: captions must appear
    // from the pre-built timeline immediately, with no whisper lag.
    ui.invoke_pre_transcribe_clicked();
    let batch_deadline = Instant::now() + Duration::from_secs(150);
    while Instant::now() < batch_deadline {
        app.pump_for_test(&last_pump);
        {
            let guard = session.borrow();
            if guard.prebuilt_ready {
                break;
            }
            if guard.status.contains("failed") && !guard.prebuilding {
                panic!("pre-transcribe failed: {}", guard.status);
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        session.borrow().prebuilt_ready,
        "batch must finish; words: {}, status: {}",
        session.borrow().prebuilt_word_count(),
        ui.get_status()
    );

    app.toggle_play_for_test();
    let mut last_status = String::new();
    let caption_deadline = Instant::now() + Duration::from_secs(8);
    loop {
        app.pump_for_test(&last_pump);
        let status = ui.get_status().to_string();
        if status != last_status && !status.is_empty() {
            last_status = status.clone();
            eprintln!("status: {status}");
        }
        let caption = ui.get_caption_text().to_string();
        let playhead = ui.get_playhead_ms();
        if !caption.trim().is_empty() && playhead < 4_000 {
            eprintln!("prebuilt captions at {} ms: {caption:?}", playhead);
            return;
        }
        assert!(
            Instant::now() < caption_deadline,
            "no prebuilt captions within 8 s; playhead: {playhead}, caption: {caption:?}, status: {status}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn cached_transcript_skips_the_batch_on_reopen() {
    ensure_platform();
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_assets/player-test.webm");
    if !fixture.is_file() {
        eprintln!("skipping: test_assets/player-test.webm not present");
        return;
    }
    let _e2e = E2E_LOCK.lock().unwrap();

    // First run: pre-transcribe (the batch also caches the transcript).
    let app = App::new().expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let session = app.session_for_test();
    ui.invoke_model_selected("tiny".into());
    ui.invoke_language_selected("English".into());
    app.refresh_for_test();
    app.open_for_test(fixture.clone());
    let last_pump = Rc::new(Cell::new(Instant::now()));
    let open_deadline = Instant::now() + Duration::from_secs(15);
    while !ui.get_can_seek() && Instant::now() < open_deadline {
        app.pump_for_test(&last_pump);
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(ui.get_can_seek(), "video must open");
    ui.invoke_pre_transcribe_clicked();
    let batch_deadline = Instant::now() + Duration::from_secs(150);
    while Instant::now() < batch_deadline {
        app.pump_for_test(&last_pump);
        if session.borrow().prebuilt_ready {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        session.borrow().prebuilt_ready,
        "first run must pre-transcribe (status: {})",
        ui.get_status()
    );
    drop(app);

    // Second run: the same file opens with the cached transcript restored;
    // no pre-transcribe click, no batch.
    let app = App::new().expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let session = app.session_for_test();
    ui.invoke_model_selected("tiny".into());
    ui.invoke_language_selected("English".into());
    app.refresh_for_test();
    app.open_for_test(fixture);
    let last_pump = Rc::new(Cell::new(Instant::now()));
    let cache_deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < cache_deadline {
        app.pump_for_test(&last_pump);
        let guard = session.borrow();
        if guard.prebuilt_ready && !guard.prebuilding {
            break;
        }
        drop(guard);
        std::thread::sleep(Duration::from_millis(50));
    }
    let guard = session.borrow();
    assert!(
        guard.prebuilt_ready && !guard.prebuilding,
        "cache must restore the transcript; status: {}",
        ui.get_status()
    );
    assert!(
        ui.get_status().as_str().contains("from cache"),
        "status must say the captions came from cache: {}",
        ui.get_status()
    );
    assert!(
        guard.prebuilt_word_count() > 0,
        "cached transcript carries words"
    );

    // Play: captions are instant, no whisper involved.
    drop(guard);
    app.toggle_play_for_test();
    let caption_deadline = Instant::now() + Duration::from_secs(8);
    loop {
        app.pump_for_test(&last_pump);
        let caption = ui.get_caption_text().to_string();
        if !caption.trim().is_empty() {
            eprintln!(
                "cached captions at {} ms: {caption:?}",
                ui.get_playhead_ms()
            );
            return;
        }
        assert!(
            Instant::now() < caption_deadline,
            "no cached captions; status: {}",
            ui.get_status()
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn live_captions_flow_end_to_end_with_a_real_video() {
    let _e2e = E2E_LOCK.lock().unwrap();
    ensure_platform();
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_assets/player-test.webm");
    if !fixture.is_file() {
        eprintln!("skipping: test_assets/player-test.webm not present");
        return;
    }

    let app = App::new().expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    // Route through the real combos so the orchestrator shares the session's
    // model/language (the cache key depends on it).
    ui.invoke_model_selected("tiny".into());
    ui.invoke_language_selected("English".into());
    app.refresh_for_test();

    app.open_for_test(fixture);
    // Pre-drain: the probe result must arrive like in production.
    let last_pump = Rc::new(Cell::new(Instant::now()));
    let drain_deadline = Instant::now() + Duration::from_secs(15);
    let mut opened = false;
    while Instant::now() < drain_deadline {
        app.pump_for_test(&last_pump);
        if ui.get_can_seek() {
            opened = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    eprintln!("opened: {opened}, status: {}", ui.get_status());
    app.toggle_play_for_test();

    // Drive the real pump loop until whisper delivers words; the caption
    // pipeline is live (ffmpeg audio decode, VAD, whisper on its own
    // thread, event drain into the overlay).
    let last_pump = Rc::new(Cell::new(Instant::now()));
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut last_status = String::new();
    loop {
        app.pump_for_test(&last_pump);
        let caption = ui.get_caption_text().to_string();
        if !caption.trim().is_empty() {
            eprintln!(
                "captions appeared at {} ms: {caption:?}",
                ui.get_playhead_ms()
            );
            return;
        }
        let status = ui.get_status().to_string();
        if status != last_status && !status.is_empty() {
            last_status = status.clone();
            eprintln!("status: {status}");
        }
        if Instant::now() > deadline {
            panic!(
                "no captions within 120 s; status: {status:?}, playhead: {} ms, playing: {}",
                ui.get_playhead_ms(),
                ui.get_playing()
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Test-only accessors; the production API never exposes the window.
impl App {
    pub(crate) fn ui_for_test(&self) -> &PlayerWindow {
        &self.ui
    }

    pub(crate) fn session_for_test(&self) -> &std::rc::Rc<std::cell::RefCell<Session>> {
        &self.session
    }

    pub(crate) fn refresh_for_test(&self) {
        self.refresh();
    }

    pub(crate) fn toggle_play_for_test(&self) {
        self.ui.invoke_play_toggled();
    }
}
