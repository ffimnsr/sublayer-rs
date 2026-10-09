//! Headless window tests: instantiate the Slint tree on the testing backend,
//! check that the refresh helpers push the expected values, and drive real
//! pointer events through the timeline.
//!
//! The backend is installed per test thread (`init_no_event_loop`), so these
//! tests never need a Wayland or X11 session.

use std::cell::Cell;
use std::path::PathBuf;
use std::sync::Arc;

use i_slint_backend_testing::ElementQuery;
use slint::platform::PointerEventButton;
use slint::{ComponentHandle, LogicalPosition, Model, PhysicalSize};
use sublayer_core::{CaptionSegment, Project, VideoMetadata, WordToken};
use sublayer_media::{WaveformBucket, WaveformCache};

use crate::MainWindow;
use crate::app::{App, project_name_from_stem, with_video_extension};
use crate::session::Session;
use crate::views::{
    UiModels, install_static_options, refresh_caption_drawer, refresh_document, refresh_encoder,
    refresh_playhead, refresh_segments, refresh_task, refresh_theme, refresh_timeline,
};

thread_local! {
    /// Whether this thread already installed the testing backend.
    static PLATFORM_READY: Cell<bool> = const { Cell::new(false) };
}

/// Installs the testing backend once per thread.
///
/// Re-initializing panics, which matters when the harness runs every test on
/// the same thread (`--test-threads=1`). The mock clock is required by
/// [`ElementHandle::mock_drag`](i_slint_backend_testing::ElementHandle::mock_drag),
/// which cannot run against a real-time platform.
fn ensure_platform() {
    PLATFORM_READY.with(|ready| {
        if ready.get() {
            return;
        }
        i_slint_backend_testing::init_no_event_loop();
        ready.set(true);
    });
}

/// Instantiates the window with a fixed size, the static option lists, and the
/// stable model instances.
fn window() -> (MainWindow, UiModels) {
    ensure_platform();
    let ui = MainWindow::new().expect("testing backend must build the window");
    ui.window().set_size(PhysicalSize::new(1320, 860));
    install_static_options(&ui);
    let models = UiModels::new();
    models.attach(&ui);
    (ui, models)
}

fn metadata(duration_ms: u64) -> VideoMetadata {
    VideoMetadata {
        duration_seconds: duration_ms as f64 / 1_000.0,
        width: 1_920,
        height: 1_080,
        fps: 30.0,
        video_codec: Some("h264".to_owned()),
        audio_codec: Some("aac".to_owned()),
        audio_channels: Some(2),
        audio_sample_rate: Some(48_000),
    }
}

/// Ten-second project with two caption cards.
fn project_with_cards() -> Project {
    let mut project = Project::new("clip", "/tmp/clip.mp4", metadata(10_000));
    project.segments = vec![
        CaptionSegment::new(vec![WordToken::new("hello", 1_000, 2_000)]),
        CaptionSegment::new(vec![WordToken::new("world", 3_000, 4_000)]),
    ];
    project
}

fn session_with_cards() -> Session {
    let project = project_with_cards();
    let preset = crate::adapters::preset_index_of(&project.theme.name);
    let mut session = Session::default();
    session.install_video(project, preset);
    session
}

#[test]
fn empty_window_shows_the_idle_state() {
    let (ui, models) = window();
    let session = Session::default();
    refresh_task(&ui, &session);
    refresh_document(&ui, &session);
    refresh_segments(&ui, &session, &models);
    refresh_playhead(&ui, &session, &models);
    refresh_timeline(&ui, &session, &models);

    assert!(!ui.get_has_project());
    assert_eq!(ui.get_status().as_str(), "Ready");
    assert!(!ui.get_busy());
    assert_eq!(ui.get_duration_ms(), 0);
    assert_eq!(ui.get_segments().row_count(), 0);
    assert!(ui.get_model_names().row_count() > 0);
    assert_eq!(ui.get_preset_names().row_count(), 5);
    assert_eq!(ui.get_alignment_names().row_count(), 9);
    assert_eq!(ui.get_animation_names().row_count(), 4);
    assert!(ui.get_timeline_pixels() > 0.0, "layout must resolve");
}

#[test]
fn export_destination_gains_a_container_extension() {
    assert_eq!(
        with_video_extension(PathBuf::from("/tmp/clip-captioned")),
        PathBuf::from("/tmp/clip-captioned.mp4")
    );
    assert_eq!(
        with_video_extension(PathBuf::from("/tmp/clip.mkv")),
        PathBuf::from("/tmp/clip.mkv")
    );
}

#[test]
fn encoder_label_reflects_the_probe() {
    let (ui, _models) = window();
    let mut session = Session::default();
    session.set_render_encoder(
        sublayer_export::HardwareProbe {
            vaapi: true,
            nvenc: false,
            #[cfg(feature = "encode_vulkan")]
            vulkan: false,
            vaapi_device: Some(PathBuf::from("/dev/dri/renderD128")),
        },
        sublayer_export::HardwareEncoder::Vaapi,
        Vec::new(),
    );
    refresh_encoder(&ui, &session);
    assert_eq!(ui.get_encoder_label().as_str(), "Encoder: VA-API");

    // An `auto` selection advertises that a runtime fallback is possible.
    session.set_render_encoder(
        sublayer_export::HardwareProbe::default(),
        sublayer_export::HardwareEncoder::Vaapi,
        vec![sublayer_export::HardwareEncoder::Cpu],
    );
    refresh_encoder(&ui, &session);
    assert_eq!(ui.get_encoder_label().as_str(), "Encoder: VA-API (auto)");
}

#[test]
fn loaded_project_reaches_every_panel() {
    let (ui, models) = window();
    let session = session_with_cards();

    refresh_task(&ui, &session);
    refresh_document(&ui, &session);
    refresh_theme(&ui, &session);
    refresh_segments(&ui, &session, &models);
    refresh_playhead(&ui, &session, &models);
    refresh_timeline(&ui, &session, &models);

    assert!(ui.get_has_project());
    assert_eq!(ui.get_project_name().as_str(), "clip");
    assert_eq!(ui.get_duration_ms(), 10_000);
    assert_eq!(ui.get_video_path().as_str(), "/tmp/clip.mp4");
    assert_eq!(ui.get_segments().row_count(), 2);

    // The theme snapshot mirrors the project's default style. Opaque colors
    // are shown as `#RRGGBB`.
    let theme = ui.get_theme();
    assert_eq!(theme.font_size, 64);
    assert_eq!(theme.primary_hex.as_str(), "#FFFFFF");
    assert!(theme.bold);

    // The playhead sits at 0 s: no caption is overlaid.
    assert!(!ui.get_show_caption());
    assert_eq!(ui.get_preview_timecode().as_str(), "00:00.000");
    assert_eq!(ui.get_buckets().row_count(), 0);
}

#[test]
fn playhead_selects_the_matching_caption() {
    let (ui, models) = window();
    let mut session = session_with_cards();
    session.set_playhead(1_500);
    refresh_playhead(&ui, &session, &models);

    assert!(ui.get_show_caption());
    assert_eq!(ui.get_active_caption().as_str(), "hello");
    assert_eq!(ui.get_preview_timecode().as_str(), "00:01.500");

    // Between the two cards the overlay disappears again.
    session.set_playhead(2_500);
    refresh_playhead(&ui, &session, &models);
    assert!(!ui.get_show_caption());
}

#[test]
fn fixture_video_stems_do_not_leak_into_the_project_name() {
    // Opening `test_assets/test.webm` once showed the header as
    // "SUBlayer · test", which read like a test build of the app itself.
    // Fixture-lookalike stems fall back to the untitled placeholder.
    for stem in [
        "test", "Test", "clip", "sample", "demo", "example", "untitled",
    ] {
        assert_eq!(
            project_name_from_stem(stem),
            None,
            "{stem:?} should be filtered"
        );
    }
    // Real footage keeps its file name as the project name.
    assert_eq!(
        project_name_from_stem("vacation_2026").as_deref(),
        Some("vacation_2026")
    );
    assert_eq!(
        project_name_from_stem("my-test-clip").as_deref(),
        Some("my-test-clip")
    );
    assert_eq!(project_name_from_stem("  "), None);
}

#[test]
fn waveform_window_follows_the_zoom() {
    let (ui, models) = window();
    let mut session = session_with_cards();
    session.waveform = Some(Arc::new(WaveformCache::new(
        16_000,
        10,
        160_000,
        vec![
            WaveformBucket {
                min_amplitude: -0.5,
                max_amplitude: 0.5,
                rms: 0.3,
            };
            100
        ],
    )));
    session.zoom_fit(ui.get_timeline_pixels());
    refresh_timeline(&ui, &session, &models);

    assert!(ui.get_column_ms() > 0);
    let buckets = ui.get_buckets();
    assert!(buckets.row_count() > 0, "the fitted window must render");
    let column = buckets.row_data(0).unwrap();
    assert!(column.high > 0.0 && column.rms > 0.0);
}

#[test]
fn dragging_a_caption_card_retimes_it_through_the_ui() {
    // A full `App` so the real callbacks are wired. The testing backend must be
    // installed first: `mock_drag` advances mock time, which panics against a
    // real-time platform.
    ensure_platform();
    let app = App::new().expect("the studio must build headlessly");
    let ui = app.window();
    ui.window().set_size(PhysicalSize::new(1320, 860));

    {
        let mut session = app.session().borrow_mut();
        session.install_video(project_with_cards(), 0);
    }
    app.refresh();

    // Debug-info ids are qualified with their component (`Track::caption-block`),
    // so match on the suffix to stay independent of the component name.
    let blocks = ElementQuery::from_root(ui)
        .match_predicate(|element| {
            element
                .id()
                .is_some_and(|id| id == "caption-block" || id.ends_with("::caption-block"))
        })
        .find_all();
    assert_eq!(blocks.len(), 2, "both caption cards must be addressable");
    let block = &blocks[0];
    let size = block.size();
    let origin = block.absolute_position();
    assert!(
        size.width > 20.0 && size.height > 10.0,
        "the card must be laid out, got {size:?}"
    );

    // Drag the card 30 px to the right; the timeline starts at 60 px/s, so the
    // card must move by ~500 ms.
    let target = LogicalPosition::new(
        origin.x + size.width / 2.0 + 30.0,
        origin.y + size.height / 2.0,
    );
    block.mock_drag(target, PointerEventButton::Left);

    let session = app.session().borrow();
    let (index, segment) = session
        .selected_segment()
        .expect("dragging a card must select it");
    assert_eq!(index, 0);
    let expected = 1_000.0 + 30.0 * 1_000.0 / f64::from(session.pixels_per_second);
    let moved = segment.start_ms().unwrap() as f64;
    assert!(
        (moved - expected).abs() <= 10.0,
        "card moved to {moved} ms, expected ~{expected} ms"
    );
    assert_eq!(segment.end_ms(), Some(segment.start_ms().unwrap() + 1_000));

    // The window model shows the retimed card as well.
    let row = ui.get_segments().row_data(0).unwrap();
    assert_eq!(row.start_ms, segment.start_ms().unwrap() as i32);
    assert!(row.selected);
}

#[test]
fn preview_pops_the_word_under_the_playhead() {
    let (ui, models) = window();
    let mut project = project_with_cards();
    project.segments = vec![CaptionSegment::new(vec![
        WordToken::new("hello", 1_000, 1_400),
        WordToken::new("there", 1_500, 1_900),
    ])];
    let preset = crate::adapters::preset_index_of(&project.theme.name);
    let mut session = Session::default();
    session.install_video(project, preset);

    session.set_playhead(1_100);
    refresh_playhead(&ui, &session, &models);
    let words = ui.get_caption_words();
    assert_eq!(words.row_count(), 2);
    assert_eq!(words.row_data(0).unwrap().text.as_str(), "hello");
    assert!(words.row_data(0).unwrap().active);
    assert!(!words.row_data(1).unwrap().active);

    session.set_playhead(1_700);
    refresh_playhead(&ui, &session, &models);
    assert!(!ui.get_caption_words().row_data(0).unwrap().active);
    assert!(ui.get_caption_words().row_data(1).unwrap().active);

    // Outside every card no overlay words are shown.
    session.set_playhead(2_500);
    refresh_playhead(&ui, &session, &models);
    assert_eq!(ui.get_caption_words().row_count(), 0);
}

#[test]
fn drawer_lists_cards_and_toggles_visibility() {
    let (ui, models) = window();
    let mut session = session_with_cards();
    session.select(0);
    refresh_segments(&ui, &session, &models);

    assert!(!ui.get_caption_drawer_open());
    assert_eq!(ui.get_caption_rows().row_count(), 2);
    let first = ui.get_caption_rows().row_data(0).unwrap();
    assert_eq!(first.label.as_str(), "00:01.000 → 00:02.000");
    assert_eq!(first.text.as_str(), "hello");
    assert!(first.selected);
    assert!(!ui.get_caption_rows().row_data(1).unwrap().selected);

    session.caption_drawer_open = true;
    refresh_caption_drawer(&ui, &session, &models);
    assert!(ui.get_caption_drawer_open());
}

#[test]
fn drawer_edits_update_the_project_and_the_timeline() {
    let (ui, models) = window();
    let mut session = session_with_cards();
    refresh_segments(&ui, &session, &models);

    assert!(session.set_segment_text(1, "buddy"));
    refresh_segments(&ui, &session, &models);

    assert_eq!(
        ui.get_caption_rows().row_data(1).unwrap().text.as_str(),
        "buddy"
    );
    assert_eq!(
        ui.get_segments().row_data(1).unwrap().text.as_str(),
        "buddy"
    );
}

#[test]
fn empty_project_shows_no_drawer_rows() {
    let (ui, models) = window();
    let session = Session::default();
    refresh_segments(&ui, &session, &models);
    assert_eq!(ui.get_caption_rows().row_count(), 0);
}

#[test]
fn right_click_on_the_timeline_inserts_a_caption_at_that_time() {
    // A full `App` so the real callbacks are wired; see the drag test.
    ensure_platform();
    let app = App::new().expect("the studio must build headlessly");
    let ui = app.window();
    ui.window().set_size(PhysicalSize::new(1320, 860));
    {
        let mut session = app.session().borrow_mut();
        session.install_video(project_with_cards(), 0);
    }
    app.refresh();

    // Right-click the empty lane: the menu opens where the press lands. The
    // testing backend starts every gesture at the element's center, so the
    // expected time is the lane center through the session's pixel scale.
    let lane = ElementQuery::from_root(ui)
        .match_predicate(|element| {
            element
                .id()
                .is_some_and(|id| id == "lane-touch" || id.ends_with("::lane-touch"))
        })
        .find_all()
        .pop()
        .expect("the lane touch area must be addressable");
    let expected_ms = (f64::from(lane.size().width) / 2.0 * 1_000.0
        / f64::from(app.session().borrow().pixels_per_second)) as u64;
    lane.mock_drag(
        LogicalPosition::new(
            lane.absolute_position().x + lane.size().width / 2.0,
            lane.absolute_position().y + lane.size().height / 2.0,
        ),
        PointerEventButton::Right,
    );

    // Click the menu's "Add caption" button.
    let button = ElementQuery::from_root(ui)
        .match_predicate(|element| {
            element.id().is_some_and(|id| {
                id == "add-caption-button" || id.ends_with("::add-caption-button")
            })
        })
        .find_all()
        .pop()
        .expect("the add-caption button must be addressable");
    let button_center = LogicalPosition::new(
        button.absolute_position().x + button.size().width / 2.0,
        button.absolute_position().y + button.size().height / 2.0,
    );
    button.mock_drag(button_center, PointerEventButton::Left);

    let session = app.session().borrow();
    let project = session.project.as_ref().expect("project must be open");
    assert_eq!(project.segments.len(), 3, "a card must have been inserted");
    let (selected, segment) = session.selected_segment().expect("inserted card selected");
    assert_eq!(selected, 2);
    assert_eq!(segment.text(), "New caption");
    assert!(
        (segment.start_ms().unwrap() as i64 - expected_ms as i64).abs() <= 10,
        "inserted at {} ms, expected ~{expected_ms}",
        segment.start_ms().unwrap()
    );
    assert_eq!(
        segment.end_ms().unwrap() - segment.start_ms().unwrap(),
        2_000,
        "default duration"
    );
    // Cards stay sorted by time, and the window models show the new card.
    assert!(
        project
            .segments
            .windows(2)
            .all(|pair| { pair[0].start_ms().unwrap() <= pair[1].start_ms().unwrap() })
    );
    assert_eq!(ui.get_segments().row_count(), 3);
    assert_eq!(ui.get_caption_rows().row_count(), 3);
}
