//! Headless window tests: instantiate the Slint tree on the testing backend
//! and check the static options and event-driven state.
//!
//! The backend is installed per test thread (`init_no_event_loop`), so these
//! tests never need a Wayland or X11 session.

use std::cell::Cell;

use i_slint_backend_testing as testing;
use slint::Model;

use crate::PlayerWindow;
use crate::app::App;
use crate::session::Session;

thread_local! {
    /// Whether this thread already installed the testing backend.
    static PLATFORM_READY: Cell<bool> = const { Cell::new(false) };
}

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
fn session_state_drives_the_overlay() {
    ensure_platform();
    let app = App::new().expect("testing backend must build the window");
    let ui = &app.ui_for_test();
    let session = app.session_for_test();

    // Opened media enables the transport and fills the header.
    session.borrow_mut().open_video("clip.mp4", 60_000);
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
