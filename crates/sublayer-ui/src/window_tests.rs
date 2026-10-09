//! Headless window tests: instantiate the Slint tree on the testing backend
//! and check that the refresh helpers push the expected values.
//!
//! The backend is installed per test thread (`init_no_event_loop`), so these
//! tests never need a Wayland or X11 session.

use std::path::PathBuf;
use std::sync::Arc;

use slint::{ComponentHandle, Model, PhysicalSize};
use sublayer_core::{CaptionSegment, Project, VideoMetadata, WordToken};
use sublayer_media::{WaveformBucket, WaveformCache};

use crate::MainWindow;
use crate::app::with_video_extension;
use crate::session::Session;
use crate::views::{
    install_static_options, refresh_document, refresh_encoder, refresh_playhead, refresh_segments,
    refresh_task, refresh_theme, refresh_timeline,
};

/// Instantiates the window with a fixed size and the static option lists.
fn window() -> MainWindow {
    i_slint_backend_testing::init_no_event_loop();
    let ui = MainWindow::new().expect("testing backend must build the window");
    ui.window().set_size(PhysicalSize::new(1320, 860));
    install_static_options(&ui);
    ui
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

fn session_with_cards() -> Session {
    let mut project = Project::new("clip", "/tmp/clip.mp4", metadata(10_000));
    project.segments = vec![
        CaptionSegment::new(vec![WordToken::new("hello", 1_000, 2_000)]),
        CaptionSegment::new(vec![WordToken::new("world", 3_000, 4_000)]),
    ];
    let preset = crate::adapters::preset_index_of(&project.theme.name);
    let mut session = Session::default();
    session.install_video(project, preset);
    session
}

#[test]
fn empty_window_shows_the_idle_state() {
    let ui = window();
    let session = Session::default();
    refresh_task(&ui, &session);
    refresh_document(&ui, &session);
    refresh_segments(&ui, &session);
    refresh_playhead(&ui, &session);
    refresh_timeline(&ui, &session);

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
    let ui = window();
    let mut session = Session::default();
    session.set_render_encoder(
        sublayer_export::HardwareProbe {
            vaapi: true,
            nvenc: false,
            vaapi_device: Some(std::path::PathBuf::from("/dev/dri/renderD128")),
        },
        sublayer_export::HardwareEncoder::Vaapi,
    );
    refresh_encoder(&ui, &session);
    assert_eq!(ui.get_encoder_label().as_str(), "Encoder: VA-API");
}

#[test]
fn loaded_project_reaches_every_panel() {
    let ui = window();
    let session = session_with_cards();

    refresh_task(&ui, &session);
    refresh_document(&ui, &session);
    refresh_theme(&ui, &session);
    refresh_segments(&ui, &session);
    refresh_playhead(&ui, &session);
    refresh_timeline(&ui, &session);

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
    let ui = window();
    let mut session = session_with_cards();
    session.set_playhead(1_500);
    refresh_playhead(&ui, &session);

    assert!(ui.get_show_caption());
    assert_eq!(ui.get_active_caption().as_str(), "hello");
    assert_eq!(ui.get_preview_timecode().as_str(), "00:01.500");

    // Between the two cards the overlay disappears again.
    session.set_playhead(2_500);
    refresh_playhead(&ui, &session);
    assert!(!ui.get_show_caption());
}

#[test]
fn waveform_window_follows_the_zoom() {
    let ui = window();
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
    refresh_timeline(&ui, &session);

    assert!(ui.get_column_ms() > 0);
    let buckets = ui.get_buckets();
    assert!(buckets.row_count() > 0, "the fitted window must render");
    let column = buckets.row_data(0).unwrap();
    assert!(column.high > 0.0 && column.rms > 0.0);
}
