//! Headless window tests: instantiate the Slint tree on the testing backend
//! and check the static options and event-driven state.
//!
//! The backend is installed per test thread (`init_no_event_loop`), so these
//! tests never need a Wayland or X11 session.

use std::cell::Cell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use i_slint_backend_testing as testing;
use slint::ComponentHandle;
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
    let app = App::new(None).expect("testing backend must build the window");
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
    let app = App::new(None).expect("testing backend must build the window");
    // A 0% default made the volume slider look mute and any interaction
    // with it silenced playback (the driver only learns the slider value
    // when it moves).
    assert_eq!(app.session_for_test().borrow().volume_percent, 100);
    assert_eq!(app.ui_for_test().get_volume(), 100.0);
}

#[test]
fn session_state_drives_the_overlay() {
    ensure_platform();
    let app = App::new(None).expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let session = app.session_for_test();

    // Opened media enables the transport and fills the header.
    session
        .borrow_mut()
        .open_video("clip.mp4", 60_000, std::path::PathBuf::from("clip.mp4"));
    app.refresh_for_test();
    assert!(ui.get_can_seek());
    assert!(ui.get_file_title().contains("clip.mp4"));

    // Batch words land in the overlay once the batch is marked ready.
    session.borrow_mut().prebuilding = true;
    session.borrow_mut().feed_words(vec![
        sublayer_core::WordToken::new("Hello", 100, 400),
        sublayer_core::WordToken::new("world.", 500, 900),
    ]);
    session.borrow_mut().mark_prebuilt_ready();
    session.borrow_mut().playhead_ms = 1_500;
    app.refresh_for_test();
    assert_eq!(ui.get_caption_text().as_str(), "Hello world.");

    // Translation on: the English card replaces the source.
    session.borrow_mut().translate_enabled = true;
    session
        .borrow_mut()
        .attach_prebuilt_translation("¡Hola mundo!".to_owned());
    app.refresh_for_test();
    assert_eq!(ui.get_caption_text().as_str(), "¡Hola mundo!");
    assert!(ui.get_translate_enabled());
}

#[test]
fn play_without_a_video_hints_in_the_status_line() {
    ensure_platform();
    let app = App::new(None).expect("testing backend must build the window");
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
    let app = App::new(None).expect("testing backend must build the window");
    let session = app.session_for_test();
    session
        .borrow_mut()
        .open_video("clip.mp4", 60_000, std::path::PathBuf::from("clip.mp4"));
    session.borrow_mut().prebuilding = true;
    session
        .borrow_mut()
        .feed_words(vec![sublayer_core::WordToken::new("tail", 55_000, 55_300)]);
    session.borrow_mut().mark_prebuilt_ready();
    session.borrow_mut().playhead_ms = 60_000; // the media ended
    session.borrow_mut().refresh_caption(60_000);
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
    let app = App::new(None).expect("testing backend must build the window");
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
fn batch_words_reach_the_overlay_only_after_the_batch_ready_event() {
    ensure_platform();
    let app = App::new(None).expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let session = app.session_for_test();
    session
        .borrow_mut()
        .open_video("clip.mp4", 60_000, std::path::PathBuf::from("clip.mp4"));

    // Batch words arrive as player events; there is no live ASR, so they
    // must not land in the overlay while the batch is still running.
    let bridge = crate::bridge::Bridge::new(crossbeam_channel::unbounded().0);
    session.borrow_mut().prebuilding = true;
    crate::app::handle_event(
        ui,
        &bridge,
        session,
        crate::bridge::PlayerEvent::Words(vec![sublayer_core::WordToken::new(
            "Hello", 1_000, 1_300,
        )]),
    );
    assert!(ui.get_caption_text().as_str().is_empty());

    // `PreTranscribed` switches the display to the pre-built timeline.
    session.borrow_mut().playhead_ms = 1_000;
    crate::app::handle_event(
        ui,
        &bridge,
        session,
        crate::bridge::PlayerEvent::PreTranscribed,
    );
    assert!(ui.get_caption_text().as_str().contains("Hello"));
}

#[test]
fn slider_scrub_then_release_keeps_playhead_and_captions_in_sync() {
    ensure_platform();
    let app = App::new(None).expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let session = app.session_for_test();
    session
        .borrow_mut()
        .open_video("clip.mp4", 60_000, std::path::PathBuf::from("clip.mp4"));
    install_batch(
        &mut session.borrow_mut(),
        &[
            sublayer_core::WordToken::new("Hallo", 0, 300),
            sublayer_core::WordToken::new("Welt.", 400, 700),
            sublayer_core::WordToken::new("Wie", 1_500, 1_800),
            sublayer_core::WordToken::new("geht", 1_900, 2_200),
            sublayer_core::WordToken::new("es?", 2_300, 2_600),
        ],
    );
    app.refresh_for_test();

    // Dragging the slider scrub-pins the playhead and the caption; nothing
    // is committed yet.
    ui.invoke_scrub(1_700);
    assert_eq!(ui.get_playhead_ms(), 1_700);
    assert_eq!(ui.get_caption_text().as_str(), "Wie geht es?");
    ui.invoke_scrub(500);
    assert_eq!(ui.get_playhead_ms(), 500);
    assert_eq!(ui.get_caption_text().as_str(), "Hallo Welt.");
    assert!(session.borrow().scrubbing, "the drag is still in flight");

    // Releasing the knob commits the seek end position.
    ui.invoke_seek(500);
    assert!(!session.borrow().scrubbing);
    assert_eq!(ui.get_playhead_ms(), 500);
    assert_eq!(ui.get_caption_text().as_str(), "Hallo Welt.");

    // A committed seek jumps to its target; the caption follows instantly.
    ui.invoke_seek(2_400);
    assert_eq!(ui.get_playhead_ms(), 2_400);
    assert_eq!(ui.get_caption_text().as_str(), "Wie geht es?");
}

#[test]
fn active_playback_holds_the_playhead_during_a_scrub() {
    ensure_platform();
    let app = App::new(None).expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let session = app.session_for_test();
    session
        .borrow_mut()
        .open_video("clip.mp4", 60_000, std::path::PathBuf::from("clip.mp4"));
    install_batch(
        &mut session.borrow_mut(),
        &[sublayer_core::WordToken::new("Hallo", 0, 300)],
    );
    session.borrow_mut().playing = true;
    app.refresh_for_test();

    // A drag pins the playhead to the knob; the wall clock must not yank it
    // forward while the user is aiming at a timestamp.
    ui.invoke_scrub(10_000);
    let last_pump = Rc::new(Cell::new(Instant::now()));
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(120) {
        app.pump_for_test(&last_pump);
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(
        ui.get_playhead_ms(),
        10_000,
        "scrubbing pins the playhead while playing"
    );

    // Release: the seek lands and the clock takes over again.
    ui.invoke_seek(10_000);
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(120) {
        app.pump_for_test(&last_pump);
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        ui.get_playhead_ms() >= 10_000,
        "playback resumes from the committed seek: {}",
        ui.get_playhead_ms()
    );
}

/// Installs a pre-built transcript directly (no whisper) and marks it ready.
fn install_batch(session: &mut Session, words: &[sublayer_core::WordToken]) {
    session.prebuilding = true;
    session.feed_words(words.to_vec());
    session.mark_prebuilt_ready();
}

/// Lays out the window and returns the seek slider's element handle.
fn seek_slider(ui: &PlayerWindow) -> testing::ElementHandle {
    // Without a window size the item tree has no geometry to hit-test, so
    // the click would land nowhere.
    ui.window()
        .set_size(slint::LogicalSize::new(1_100.0, 720.0));
    testing::ElementHandle::find_by_element_id(ui, "PlayerWindow::seek-slider")
        .next()
        .expect("the seek slider must be reachable by id")
}

#[test]
fn clicking_the_seek_slider_commits_a_seek_and_syncs_the_caption() {
    ensure_platform();
    let app = App::new(None).expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let session = app.session_for_test();
    session
        .borrow_mut()
        .open_video("clip.mp4", 60_000, std::path::PathBuf::from("clip.mp4"));
    install_batch(
        &mut session.borrow_mut(),
        &[
            sublayer_core::WordToken::new("First", 0, 300),
            sublayer_core::WordToken::new("part.", 400, 700),
            sublayer_core::WordToken::new("Second", 29_000, 29_300),
            sublayer_core::WordToken::new("part.", 29_400, 29_700),
        ],
    );
    app.refresh_for_test();

    // A real click on the middle of the slider track: the widget must commit
    // a seek (old bug: the click only moved the display, never the decoders).
    let slider = seek_slider(ui);
    slider.mock_single_click(slint::platform::PointerEventButton::Left);
    let playhead = ui.get_playhead_ms();
    assert!(
        (29_500..=30_500).contains(&playhead),
        "a center click seeks to ~50 % of the 60 s media; got {playhead} ms"
    );
    assert!(!session.borrow().scrubbing, "the click released the knob");
    assert_eq!(
        ui.get_caption_text().as_str(),
        "Second part.",
        "the caption follows the committed seek"
    );
    // The widget's assignment reached the model through the two-way binding
    // (a one-way `value:` binding would have been destroyed by the drag).
    assert_eq!(ui.get_seek_value(), playhead as f32);
}

#[test]
fn dragging_the_seek_slider_commits_the_release_position() {
    ensure_platform();
    let app = App::new(None).expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let session = app.session_for_test();
    session
        .borrow_mut()
        .open_video("clip.mp4", 60_000, std::path::PathBuf::from("clip.mp4"));
    install_batch(
        &mut session.borrow_mut(),
        &[
            sublayer_core::WordToken::new("First", 0, 300),
            sublayer_core::WordToken::new("part.", 400, 700),
        ],
    );
    app.refresh_for_test();
    // Start playback so the commit goes through the playing seek path.
    session.borrow_mut().playing = true;

    let slider = seek_slider(ui);
    let position = slider.absolute_position();
    let size = slider.size();
    // Drag from the center to the far left edge of the track.
    slider.mock_drag(
        slint::LogicalPosition::new(position.x + 2.0, position.y + size.height / 2.0),
        slint::platform::PointerEventButton::Left,
    );

    let playhead = ui.get_playhead_ms();
    assert!(
        playhead < 2_000,
        "the drag must commit its release position; got {playhead} ms"
    );
    assert!(!session.borrow().scrubbing);
    assert_eq!(ui.get_seek_value(), playhead as f32);
    assert_eq!(ui.get_caption_text().as_str(), "First part.");
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

    let app = App::new(None).expect("testing backend must build the window");
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
fn launching_with_a_video_path_opens_it() {
    ensure_platform();
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_assets/player-test.webm");
    if !fixture.is_file() {
        eprintln!("skipping: test_assets/player-test.webm not present");
        return;
    }
    // A file passed on the command line (dropped onto the player, or an
    // "Open with" launch) becomes the initial video.
    let app = App::new(Some(fixture)).expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let last_pump = Rc::new(Cell::new(Instant::now()));
    let deadline = Instant::now() + Duration::from_secs(15);
    while !ui.get_can_seek() && Instant::now() < deadline {
        app.pump_for_test(&last_pump);
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(ui.get_can_seek(), "the dropped video must open");
    assert!(ui.get_file_title().contains("player-test.webm"));
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
    let app = App::new(None).expect("testing backend must build the window");
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
    let app = App::new(None).expect("testing backend must build the window");
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
fn playback_without_a_prebuilt_transcript_shows_no_captions() {
    // Regression: there is no live auto-transcription anymore; playing a
    // video before pre-transcribing must not produce captions.
    let _e2e = E2E_LOCK.lock().unwrap();
    ensure_platform();
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_assets/player-test.webm");
    if !fixture.is_file() {
        eprintln!("skipping: test_assets/player-test.webm not present");
        return;
    }

    let app = App::new(None).expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let session = app.session_for_test();
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
    // A cached transcript from an earlier run is fine — that is the feature.
    if session.borrow().prebuilt_ready {
        eprintln!("skipping: a cached transcript was restored");
        return;
    }

    app.toggle_play_for_test();
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        app.pump_for_test(&last_pump);
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        ui.get_caption_text().as_str().is_empty(),
        "no live captions without a pre-transcription"
    );
}

#[test]
fn seek_while_playing_keeps_playback_running_from_the_target() {
    // Regression: a slider seek during playback used to skip the decoder
    // restart for small movements (< 1.5 s), leaving audio and video on
    // different timestamps forever. The seek must always commit.
    let _e2e = E2E_LOCK.lock().unwrap();
    ensure_platform();
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_assets/player-test.webm");
    if !fixture.is_file() {
        eprintln!("skipping: test_assets/player-test.webm not present");
        return;
    }

    let app = App::new(None).expect("testing backend must build the window");
    let ui = &app.ui_for_test();
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

    app.toggle_play_for_test();
    // Let playback reach a stable playhead.
    let run_deadline = Instant::now() + Duration::from_secs(6);
    while Instant::now() < run_deadline {
        app.pump_for_test(&last_pump);
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(ui.get_playing(), "playback must be running");
    let base = ui.get_playhead_ms();
    assert!(base > 1_000, "playhead should have advanced: {base}");

    // A small backward seek (the old < 1.5 s debounce window) while playing.
    ui.invoke_seek((base - 500).max(0));
    let target = ui.get_playhead_ms();
    assert_eq!(target, (base - 500).max(0));
    let start = Instant::now();
    while Instant::now() < start + Duration::from_millis(700) {
        app.pump_for_test(&last_pump);
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        ui.get_playhead_ms() > target,
        "playback resumes from the committed seek target: {} -> {}",
        target,
        ui.get_playhead_ms()
    );
    assert!(ui.get_playing(), "playback keeps running after the seek");
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
