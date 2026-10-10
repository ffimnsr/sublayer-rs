//! Headless window tests: instantiate the Slint tree on the testing backend,
//! check that the refresh helpers push the expected values, and drive real
//! pointer events through the timeline.
//!
//! The backend is installed per test thread (`init_no_event_loop`), so these
//! tests never need a Wayland or X11 session.

use std::cell::Cell;
use std::path::{Path, PathBuf};
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

/// Six-word card with real-word spacing (pauses between words), as produced
/// by whisper: adjacent words abut exactly, later words carry real pauses.
fn session_with_word_card() -> Session {
    let mut project = Project::new("clip", "/tmp/clip.mp4", metadata(10_000));
    project.segments = vec![CaptionSegment::new(vec![
        WordToken::new("whoo", 1_000, 1_200),
        WordToken::new("hoo!", 1_200, 1_800),
        WordToken::new("gold", 1_820, 1_990),
        WordToken::new("number", 2_140, 2_270),
        WordToken::new("one!", 2_380, 2_560),
        WordToken::new("yeah", 2_740, 2_930),
    ])];
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
    assert_eq!(ui.get_animation_names().row_count(), 5);
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
fn word_pop_follows_the_playhead_through_a_real_word_card() {
    let (ui, models) = window();
    let mut session = session_with_word_card();

    // Sweep the playhead through the card; at every word's midpoint exactly
    // that word is active, past words have returned to the primary color.
    let midpoints = [
        (1_100, 0),
        (1_500, 1),
        (1_900, 2),
        (2_200, 3),
        (2_450, 4),
        (2_850, 5),
    ];
    for (playhead, expected_active) in midpoints {
        session.set_playhead(playhead);
        refresh_playhead(&ui, &session, &models);
        let words = ui.get_caption_words();
        assert_eq!(words.row_count(), 6, "all words of the card are shown");
        for index in 0..6 {
            let active = words.row_data(index).unwrap().active;
            assert_eq!(
                active,
                index == expected_active,
                "playhead {playhead} ms, word {index} active"
            );
        }
    }

    // A pause between words leaves the previous word inactive.
    session.set_playhead(2_100);
    refresh_playhead(&ui, &session, &models);
    let words = ui.get_caption_words();
    assert_eq!(words.row_count(), 6);
    for index in 0..6 {
        assert!(!words.row_data(index).unwrap().active, "word {index}");
    }
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

#[test]
fn caption_overlay_renders_for_all_animations() {
    ensure_platform();
    let app = App::new().expect("the studio must build headlessly");
    let ui = app.window();
    ui.window().set_size(PhysicalSize::new(1320, 860));
    {
        let mut session = app.session().borrow_mut();
        session.install_video(project_with_cards(), 0);
    }
    app.refresh();
    ui.set_has_preview(true);
    ui.set_video_width(1080);
    ui.set_video_height(1920);

    // Set playhead inside the first card (1000ms..2000ms)
    app.session().borrow_mut().set_playhead(1500);
    app.refresh();

    // Verify caption words are populated
    assert_eq!(ui.get_caption_words().row_count(), 1);
    assert_eq!(
        ui.get_caption_words().row_data(0).unwrap().text.as_str(),
        "hello"
    );
    assert!(ui.get_show_caption());
    assert!(!ui.get_active_caption().is_empty());

    // Test across ALL animations: None (0), WordPop (1), Karaoke (2), Bounce (3), HighlightBox (4)
    for anim_idx in 0..5 {
        let mut data = ui.get_theme();
        data.animation_index = anim_idx;
        ui.set_theme(data);

        let text_elements = ElementQuery::from_root(ui)
            .match_predicate(|el| {
                el.accessible_label()
                    .is_some_and(|l| l.contains("HELLO") || l.contains("hello"))
            })
            .find_all();
        assert!(
            !text_elements.is_empty(),
            "animation {anim_idx} must render caption text elements"
        );
        for el in &text_elements {
            assert!(
                el.size().width > 0.0,
                "word text must have positive width in animation {anim_idx}"
            );
            assert!(
                el.size().height > 0.0,
                "word text must have positive height in animation {anim_idx}"
            );
            assert!(
                el.size().height < 120.0,
                "word text/pill must not stretch across video in animation {anim_idx}"
            );
        }
    }
}

#[test]
fn long_card_auto_fits_font_size_in_ui_preview() {
    ensure_platform();
    let app = App::new().expect("the studio must build headlessly");
    let ui = app.window();
    ui.window().set_size(PhysicalSize::new(1320, 860));

    let mut project = session_with_word_card().project.unwrap();
    project.video_metadata.width = 1080;
    project.video_metadata.height = 1920;
    let base_scaled_font_size =
        sublayer_subtitles::scaled_font_size_for(&project.theme, &project.video_metadata);
    {
        let mut session = app.session().borrow_mut();
        session.install_video(project, 0);
        session.set_playhead(1500);
    }
    app.refresh();

    let preview_font_size = ui.get_preview_font_size();
    assert!(
        preview_font_size < base_scaled_font_size as i32,
        "preview font size ({preview_font_size}) must be scaled down from base font size ({base_scaled_font_size}) to prevent overflow"
    );
}

#[test]
fn alignment_changes_affect_theme_and_ui() {
    ensure_platform();
    let app = App::new().expect("the studio must build headlessly");
    let ui = app.window();
    ui.window().set_size(PhysicalSize::new(1320, 860));

    {
        let mut session = app.session().borrow_mut();
        session.install_video(project_with_cards(), 0);
    }
    app.refresh();

    assert_eq!(ui.get_theme().alignment_index, 1);
    assert_eq!(
        app.session()
            .borrow()
            .project
            .as_ref()
            .unwrap()
            .theme
            .alignment,
        2
    );

    let mut theme_data = ui.get_theme();
    theme_data.alignment_index = 7;
    ui.invoke_theme_edited(theme_data);

    assert_eq!(ui.get_theme().alignment_index, 7);
    assert_eq!(
        app.session()
            .borrow()
            .project
            .as_ref()
            .unwrap()
            .theme
            .alignment,
        8
    );

    let mut theme_data = ui.get_theme();
    theme_data.alignment_index = 3;
    ui.invoke_theme_edited(theme_data);

    assert_eq!(ui.get_theme().alignment_index, 3);
    assert_eq!(
        app.session()
            .borrow()
            .project
            .as_ref()
            .unwrap()
            .theme
            .alignment,
        4
    );
}

/// The preview caption card must follow the inspector's Alignment setting:
/// picking bottom/middle/top moves the overlay inside the fitted frame
/// (bottom lowest, top highest), while the centered row keeps its x.
#[test]
fn preview_caption_follows_alignment_anchor() {
    ensure_platform();
    let app = App::new().expect("the studio must build headlessly");
    let ui = app.window();
    ui.window().set_size(PhysicalSize::new(1320, 860));

    {
        let mut session = app.session().borrow_mut();
        session.install_video(project_with_cards(), 0);
    }
    app.refresh();
    ui.set_has_preview(true);
    ui.set_video_width(1080);
    ui.set_video_height(1920);
    app.session().borrow_mut().set_playhead(1500);
    app.refresh();

    // Caption overlay words; the timeline block with the same label sits far
    // below the viewport, so only elements inside the preview count.
    let overlay_words = |ui: &MainWindow| -> Vec<(f32, f32)> {
        ElementQuery::from_root(ui)
            .match_predicate(|el| {
                el.accessible_label()
                    .is_some_and(|l| l == "HELLO" || l == "hello")
            })
            .find_all()
            .into_iter()
            .filter(|el| el.absolute_position().y < 500.0)
            .map(|el| {
                let pos = el.absolute_position();
                (pos.x, pos.y)
            })
            .collect()
    };

    let mut measured = Vec::new();
    for (name, index) in [("bottom", 1), ("middle", 4), ("top", 7)] {
        let mut data = ui.get_theme();
        data.alignment_index = index;
        ui.invoke_theme_edited(data);

        let words = overlay_words(ui);
        assert!(
            !words.is_empty(),
            "{name} alignment must keep the caption visible"
        );
        let xs: Vec<f32> = words.iter().map(|(x, _)| *x).collect();
        let ys: Vec<f32> = words.iter().map(|(_, y)| *y).collect();
        let x_center = (xs.iter().copied().fold(0.0, f32::max)
            + xs.iter().copied().fold(f32::MAX, f32::min))
            / 2.0;
        measured.push((name, x_center, ys[0]));
    }

    // Centered rows keep the same x; the y anchor moves bottom -> middle -> top.
    assert!(
        (measured[0].1 - measured[1].1).abs() < 2.0 && (measured[1].1 - measured[2].1).abs() < 2.0,
        "center alignment must not shift the row horizontally: {measured:?}"
    );
    assert!(
        measured[0].2 > measured[1].2 + 20.0 && measured[1].2 > measured[2].2 + 20.0,
        "alignment must move the caption card bottom -> middle -> top: {measured:?}"
    );
    assert!(
        measured[2].2 > 0.0,
        "top alignment must stay inside the viewport: {measured:?}"
    );
}

/// The preview words must sit exactly where the ASS renderer places them:
/// the overlay scales the measured `\an5\pos` anchors (`place_words`) from
/// video pixels onto the fitted frame.
#[test]
fn preview_words_match_ass_placement() {
    ensure_platform();
    let app = App::new().expect("the studio must build headlessly");
    let ui = app.window();
    ui.window().set_size(PhysicalSize::new(1320, 860));

    // Portrait project metadata so the measurement, the fitted frame, and the
    // window all agree on one aspect ratio.
    let mut project = project_with_cards();
    project.video_metadata.width = 1080;
    project.video_metadata.height = 1920;
    {
        let mut session = app.session().borrow_mut();
        session.install_video(project, 0);
    }
    app.refresh();
    ui.set_has_preview(true);
    ui.set_video_width(1080);
    ui.set_video_height(1920);
    app.session().borrow_mut().set_playhead(1500);
    app.refresh();

    // The fitted preview frame; its size gives the video-pixel -> screen-px
    // scale for both axes.
    let image = ElementQuery::from_root(ui)
        .match_predicate(|el| {
            el.type_name().is_some_and(|t| t == "Image") && el.size().width > 200.0
        })
        .find_first()
        .expect("preview image");
    let image_pos = image.absolute_position();
    let image_size = image.size();
    let scale_x = image_size.width / 1080.0;
    let scale_y = image_size.height / 1920.0;

    for (name, index) in [("bottom-left", 0), ("middle-center", 4), ("top-right", 8)] {
        let mut data = ui.get_theme();
        data.alignment_index = index;
        ui.invoke_theme_edited(data);

        let session = app.session().borrow();
        let project = session.project.as_ref().expect("project");
        let placements = sublayer_subtitles::place_words(
            &project.segments[0],
            &project.theme,
            &project.video_metadata,
            &sublayer_subtitles::resolve_fonts_dir(),
        )
        .expect("placement");
        let (px, py) = (placements[0].x, placements[0].y);
        drop(session);

        let word = ElementQuery::from_root(ui)
            .match_predicate(|el| {
                el.accessible_label().is_some_and(|l| l == "HELLO")
                    && el.absolute_position().y < 500.0
            })
            .find_first()
            .expect("overlay word");
        let pos = word.absolute_position();
        let size = word.size();
        let (center_x, center_y) = (pos.x + size.width / 2.0, pos.y + size.height / 2.0);

        assert!(
            (center_x - (image_pos.x + px * scale_x)).abs() < 1.5,
            "{name}: word center x {center_x:.1} must match ASS anchor {:.1}",
            image_pos.x + px * scale_x
        );
        assert!(
            (center_y - (image_pos.y + py * scale_y)).abs() < 1.5,
            "{name}: word center y {center_y:.1} must match ASS anchor {:.1}",
            image_pos.y + py * scale_y
        );
    }
}

/// Pixel-level regression: render the SAME card through ffmpeg/libass (the
/// export pipeline's burn) and compare the caption's bounding box with the
/// preview overlay — same face, weight, size, and pill proportions, so at
/// least 90 % of either box must overlap the other. Covers a short card, a
/// long (auto-fitted) card, and a HighlightBox pill card.
#[test]
fn preview_caption_matches_the_rendered_frame() {
    ensure_platform();
    let Ok(ffmpeg) = sublayer_media::ffmpeg::resolve(
        sublayer_media::ffmpeg::FFMPEG,
        sublayer_media::ffmpeg::FFMPEG_ENV,
    ) else {
        eprintln!("skipping: ffmpeg unavailable");
        return;
    };

    let app = App::new().expect("the studio must build headlessly");
    let ui = app.window();
    ui.window().set_size(PhysicalSize::new(1320, 860));

    let cards = [
        ("short", vec!["hello"], 300, 1),
        (
            "long",
            vec!["whos", "hoo", "gold", "number", "one!"],
            1100,
            1,
        ),
        ("highlight", vec!["hello"], 300, 4),
    ];
    for (label, words, playhead, animation_index) in cards {
        let mut project = project_with_cards();
        project.video_metadata.width = 1080;
        project.video_metadata.height = 1920;
        project.segments = vec![CaptionSegment::new(
            words
                .iter()
                .enumerate()
                .map(|(index, &text)| {
                    WordToken::new(text, index as u64 * 400, index as u64 * 400 + 400)
                })
                .collect(),
        )];
        {
            let mut session = app.session().borrow_mut();
            session.install_video(project, 0);
        }
        app.refresh();
        ui.set_has_preview(true);
        ui.set_video_width(1080);
        ui.set_video_height(1920);
        app.session().borrow_mut().set_playhead(playhead);
        app.refresh();
        if animation_index != 1 {
            let mut theme = ui.get_theme();
            theme.animation_index = animation_index;
            ui.invoke_theme_edited(theme);
        }

        let image = ElementQuery::from_root(ui)
            .match_predicate(|el| {
                el.type_name().is_some_and(|t| t == "Image") && el.size().width > 200.0
            })
            .find_first()
            .expect("preview image");
        let image_pos = image.absolute_position();
        let image_size = image.size();
        let scale_x = image_size.width / 1080.0;
        let scale_y = image_size.height / 1920.0;

        // Union of the preview word boxes, converted back to video pixels;
        // the HighlightBox pill is part of the caption too.
        let display: Vec<String> = words.iter().map(|w| w.to_uppercase()).collect();
        let caption_elements = ElementQuery::from_root(ui)
            .match_predicate(move |el| {
                let word = el
                    .accessible_label()
                    .is_some_and(|l| display.iter().any(|word| word.as_str() == l.as_str()));
                let pill = el.id().is_some_and(|id| id.ends_with("caption-pill"));
                (word || pill) && el.absolute_position().y < 500.0
            })
            .find_all();
        assert!(
            !caption_elements.is_empty(),
            "{label}: preview caption must render"
        );
        let mut preview_rect = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for el in &caption_elements {
            let pos = el.absolute_position();
            let size = el.size();
            let (x1, y1, x2, y2) = (
                (pos.x - image_pos.x) / scale_x,
                (pos.y - image_pos.y) / scale_y,
                (pos.x + size.width - image_pos.x) / scale_x,
                (pos.y + size.height - image_pos.y) / scale_y,
            );
            preview_rect.0 = preview_rect.0.min(x1);
            preview_rect.1 = preview_rect.1.min(y1);
            preview_rect.2 = preview_rect.2.max(x2);
            preview_rect.3 = preview_rect.3.max(y2);
        }

        // The pill box alone, for the HighlightBox proportions check.
        let pill_elements = ElementQuery::from_root(ui)
            .match_predicate(|el| {
                el.id().is_some_and(|id| id.ends_with("caption-pill"))
                    && el.absolute_position().y < 500.0
            })
            .find_all();
        let mut pill_rect = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for el in &pill_elements {
            let pos = el.absolute_position();
            let size = el.size();
            let (x1, y1, x2, y2) = (
                (pos.x - image_pos.x) / scale_x,
                (pos.y - image_pos.y) / scale_y,
                (pos.x + size.width - image_pos.x) / scale_x,
                (pos.y + size.height - image_pos.y) / scale_y,
            );
            pill_rect.0 = pill_rect.0.min(x1);
            pill_rect.1 = pill_rect.1.min(y1);
            pill_rect.2 = pill_rect.2.max(x2);
            pill_rect.3 = pill_rect.3.max(y2);
        }

        // Burn the same card through ffmpeg's libass and measure the caption
        // bbox (white text on black; the `bbox` filter logs min/max of pixels
        // above its threshold).
        let session = app.session().borrow();
        let project = session.project.as_ref().expect("project");
        let fonts_dir = sublayer_subtitles::resolve_fonts_dir();
        let script = sublayer_subtitles::build_ass_script(
            &project.segments,
            &project.theme,
            &project.video_metadata,
            &fonts_dir,
        )
        .unwrap();
        let dir = tempfile::Builder::new()
            .prefix("sublayer-render-")
            .tempdir()
            .unwrap();
        let ass_path = dir.path().join("subs.ass");
        std::fs::write(&ass_path, &script).unwrap();
        drop(session);

        let output = std::process::Command::new(&ffmpeg)
            .args(["-hide_banner", "-loglevel", "info", "-y"])
            .args(["-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=1:r=30"])
            .arg("-vf")
            .arg(format!(
                "trim=start_frame=6:end_frame=7,ass={}:fontsdir={},bbox=min_val={}",
                escape_filter_path(&ass_path),
                escape_filter_path(&fonts_dir),
                // White text shines through at 120; the HighlightBox pill is
                // a mid-luminance color (e.g. #FE2C55 ≈ 111), so lower it.
                if animation_index == 4 { 80 } else { 120 }
            ))
            .args(["-frames:v", "1", "-f", "null", "-"])
            .output()
            .expect("ffmpeg render");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{label}: ffmpeg failed: {stderr}");
        let values: Vec<f32> = ["x1:", "y1:", "x2:", "y2:"]
            .iter()
            .map(|needle| {
                let line = stderr.lines().rev().find(|line| line.contains(needle));
                let value = line
                    .and_then(|line| line.split(needle).nth(1))
                    .and_then(|rest| rest.split_whitespace().next())
                    .and_then(|value| value.parse::<f32>().ok());
                value.unwrap_or_else(|| panic!("{label}: no bbox {needle} in: {stderr}"))
            })
            .collect();
        let render_rect = (values[0], values[1], values[2], values[3]);

        // The render bbox is the glyph INK box; the preview element reports
        // the line box (ascent+descent), which is taller than the ink. Both
        // are centered on the same measured anchor, so the parity checks are:
        // 90 %+ horizontal overlap, a matched width, and a shared y center.
        let (pw, rw) = (
            preview_rect.2 - preview_rect.0,
            render_rect.2 - render_rect.0,
        );
        let x_overlap =
            (preview_rect.2.min(render_rect.2) - preview_rect.0.max(render_rect.0)).max(0.0);
        assert!(
            x_overlap / pw >= 0.90,
            "{label} card: preview x [{:.1},{:.1}] vs render x [{:.1},{:.1}] \
             overlap {:.3} < 0.90",
            preview_rect.0,
            preview_rect.2,
            render_rect.0,
            render_rect.2,
            x_overlap / pw
        );
        assert!(
            (pw - rw).abs() / rw <= 0.15,
            "{label} card: preview width {pw:.1} vs rendered {rw:.1} drift > 15 %"
        );
        // The preview text box must be tall enough to hold the glyph ink:
        // clipping would slice the letters top and bottom (regression guard
        // for the stretched line box).
        let (ph, rh) = (
            preview_rect.3 - preview_rect.1,
            render_rect.3 - render_rect.1,
        );
        assert!(
            ph >= rh * 0.7,
            "{label} card: preview text height {ph:.1} < 70 % of rendered ink {rh:.1} (cropped?)"
        );
        let (pcx, rcx) = (
            (preview_rect.0 + preview_rect.2) / 2.0,
            (render_rect.0 + render_rect.2) / 2.0,
        );
        let (pcy, rcy) = (
            (preview_rect.1 + preview_rect.3) / 2.0,
            (render_rect.1 + render_rect.3) / 2.0,
        );
        assert!(
            (pcx - rcx).abs() <= 6.0 && (pcy - rcy).abs() <= 6.0,
            "{label} card: preview center ({pcx:.1},{pcy:.1}) vs render center ({rcx:.1},{rcy:.1})"
        );

        // HighlightBox: the preview pill must match the rendered pill spans
        // (the render bbox is dominated by the pill, which is taller than the
        // text ink). The old 1.4x-line-box pill fails this by ~60 %.
        if animation_index == 4 {
            let (phh, rhh) = (pill_rect.3 - pill_rect.1, render_rect.3 - render_rect.1);
            assert!(
                !pill_elements.is_empty(),
                "{label}: preview must show the HighlightBox pill"
            );
            assert!(
                (phh - rhh).abs() / rhh <= 0.15,
                "{label}: pill height {phh:.1} vs rendered {rhh:.1} drift > 15 %"
            );
            assert!(
                (pill_rect.2 - pill_rect.0 - rw).abs() / rw <= 0.15,
                "{label}: pill width {:.1} vs rendered {rw:.1} drift > 15 %",
                pill_rect.2 - pill_rect.0
            );
        }
    }
}

/// Escapes an ASS path for the ffmpeg `ass` filter option (mirrors the
/// export runner).
fn escape_filter_path(path: &Path) -> String {
    let text = path.to_string_lossy();
    let mut escaped = String::with_capacity(text.len() + 2);
    escaped.push('\'');
    for character in text.chars() {
        match character {
            '\\' | '\'' | ':' => {
                escaped.push('\\');
                escaped.push(character);
            }
            _ => escaped.push(character),
        }
    }
    escaped.push('\'');
    escaped
}
