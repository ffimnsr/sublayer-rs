//! End-to-end render tests.
//!
//! Everything here is gated on FFmpeg being installed; when it is missing the
//! test prints why it skipped and returns. The CPU backend is used so the
//! suite also runs on GPU-less CI machines.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use sublayer_core::{CaptionSegment, Project, VideoMetadata, WordToken};
use sublayer_export::{
    ExportError, ExportOptions, ExportProgress, HardwareEncoder, export_project, probe_hardware,
    run_export,
};
use sublayer_media::{ffmpeg, probe_video};
use tokio::sync::mpsc;

/// Generates a `seconds`-long 320x240 clip with a sine track.
async fn generate_clip(path: &Path, seconds: u32) -> Result<(), String> {
    let program = ffmpeg::resolve(ffmpeg::FFMPEG, ffmpeg::FFMPEG_ENV).map_err(|e| e.to_string())?;
    let status = tokio::process::Command::new(program)
        .args(["-hide_banner", "-nostdin", "-loglevel", "error", "-y"])
        .args(["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30"])
        .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"])
        .args(["-t", &seconds.to_string()])
        .args(["-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac", "-b:a", "64k"])
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("ffmpeg exited with {status}"))
    }
}

fn test_project(video: &Path, metadata: VideoMetadata) -> Project {
    let mut project = Project::new("render-test", video, metadata);
    project.segments = vec![CaptionSegment::new(vec![WordToken::new(
        "hello world",
        200,
        900,
    )])];
    project
}

/// Resolves the encoder and assets a test render needs.
///
/// The probe also verifies that FFmpeg runs at all; when it fails the test
/// skips, matching the other crates' FFmpeg-gated suites.
async fn render_setup() -> Option<(ExportOptions, PathBuf)> {
    if let Err(error) = probe_hardware().await {
        eprintln!("skipping: ffmpeg unavailable ({error})");
        return None;
    }
    let options = ExportOptions {
        encoder: HardwareEncoder::Cpu,
        quality: 28,
        ..ExportOptions::default()
    };
    let fonts_dir = sublayer_subtitles::resolve_fonts_dir();
    Some((options, fonts_dir))
}

#[tokio::test]
async fn renders_a_project_to_a_playable_file() {
    let Some((mut options, fonts_dir)) = render_setup().await else {
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let video = directory.path().join("clip.mp4");
    if let Err(error) = generate_clip(&video, 1).await {
        eprintln!("skipping: could not create the test clip ({error})");
        return;
    }
    let metadata = probe_video(&video).await.unwrap();
    options.duration_ms = metadata.duration_ms();
    let project = test_project(&video, metadata);
    let output = directory.path().join("out.mp4");

    let (progress_tx, mut progress_rx) = mpsc::channel(64);
    let render = export_project(&project, &output, &fonts_dir, &options, progress_tx);
    let collector = tokio::spawn(async move {
        let mut seen: Vec<ExportProgress> = Vec::new();
        while let Some(progress) = progress_rx.recv().await {
            seen.push(progress);
        }
        seen
    });

    render.await.expect("cpu render must succeed");
    let samples = collector.await.unwrap();

    assert!(!samples.is_empty(), "progress must be reported");
    assert!(
        samples
            .windows(2)
            .all(|pair| pair[1].percentage >= pair[0].percentage),
        "progress must not go backwards: {samples:?}"
    );
    let last = samples.last().unwrap();
    assert_eq!(last.percentage, 1.0);
    assert_eq!(last.eta_seconds, 0.0);

    let bytes = std::fs::metadata(&output).expect("output must exist").len();
    assert!(
        bytes > 1_000,
        "rendered file is suspiciously small: {bytes}"
    );

    // The output must be a decodable video of the same duration.
    let rendered = probe_video(&output).await.unwrap();
    assert!(rendered.has_audio(), "audio track must be carried over");
    let drift = rendered
        .duration_ms()
        .abs_diff(project.video_metadata.duration_ms());
    assert!(drift <= 250, "duration drifted by {drift} ms");
}

#[tokio::test]
async fn a_dropped_receiver_cancels_the_render() {
    let Some((mut options, fonts_dir)) = render_setup().await else {
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    // Long enough that the render is still running when the first report
    // arrives; the test then drops the receiver and expects cancellation.
    let video = directory.path().join("long.mp4");
    if let Err(error) = generate_clip(&video, 30).await {
        eprintln!("skipping: could not create the test clip ({error})");
        return;
    }
    let metadata = probe_video(&video).await.unwrap();
    options.duration_ms = metadata.duration_ms();
    let ass = directory.path().join("subs.ass");
    std::fs::write(
        &ass,
        sublayer_subtitles::build_ass_script(
            &[CaptionSegment::new(vec![WordToken::new("hi", 0, 500)])],
            &sublayer_subtitles::preset("tiktok").expect("built-in preset"),
            &metadata,
            &fonts_dir,
        )
        .unwrap(),
    )
    .unwrap();
    let output = directory.path().join("cancelled.mp4");

    let (progress_tx, mut progress_rx) = mpsc::channel(8);
    let render = tokio::spawn({
        let options = options.clone();
        let fonts_dir = fonts_dir.clone();
        async move { run_export(&video, &output, &ass, &fonts_dir, &options, progress_tx).await }
    });

    let first = tokio::time::timeout(Duration::from_secs(30), progress_rx.recv())
        .await
        .expect("the render must report progress")
        .expect("progress channel must stay open");
    assert!(first.percentage < 1.0, "render finished before cancelling");
    drop(progress_rx);

    let result = tokio::time::timeout(Duration::from_secs(30), render)
        .await
        .expect("cancelled render must not hang")
        .unwrap();
    assert!(
        matches!(result, Err(ExportError::Cancelled)),
        "expected cancellation, got {result:?}"
    );
}

#[tokio::test]
async fn renders_from_paths_with_filter_separators() {
    let Some((mut options, fonts_dir)) = render_setup().await else {
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    // Colons, spaces, and apostrophes terminate or quote FFmpeg filter options;
    // both the script and the fonts directory must survive them.
    let odd = directory.path().join("od d: it's here");
    std::fs::create_dir_all(&odd).unwrap();
    let odd_fonts = odd.join("fonts");
    std::fs::create_dir_all(&odd_fonts).unwrap();
    let bundled = fonts_dir.join("Montserrat-ExtraBold.ttf");
    let fonts_dir = if bundled.is_file() {
        std::fs::copy(&bundled, odd_fonts.join("Montserrat-ExtraBold.ttf")).unwrap();
        odd_fonts.clone()
    } else {
        fonts_dir
    };

    let video = odd.join("clip.mp4");
    if let Err(error) = generate_clip(&video, 1).await {
        eprintln!("skipping: could not create the test clip ({error})");
        return;
    }
    let metadata = probe_video(&video).await.unwrap();
    options.duration_ms = metadata.duration_ms();
    let project = test_project(&video, metadata);
    let output = odd.join("out.mp4");

    let (progress_tx, mut progress_rx) = mpsc::channel(64);
    let collector = tokio::spawn(async move { while progress_rx.recv().await.is_some() {} });
    export_project(&project, &output, &fonts_dir, &options, progress_tx)
        .await
        .expect("a quoted path must render");
    collector.await.unwrap();

    assert!(std::fs::metadata(&output).unwrap().len() > 1_000);
}

#[tokio::test]
async fn a_failing_encoder_falls_back_to_the_next_in_the_chain() {
    let Some((mut options, fonts_dir)) = render_setup().await else {
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let video = directory.path().join("clip.mp4");
    if let Err(error) = generate_clip(&video, 1).await {
        eprintln!("skipping: could not create the test clip ({error})");
        return;
    }
    let metadata = probe_video(&video).await.unwrap();
    options.duration_ms = metadata.duration_ms();
    // A bogus VA-API device makes the first attempt fail at startup on every
    // machine, without depending on driver or build differences.
    options.encoder = HardwareEncoder::Vaapi;
    options.vaapi_device = Some(directory.path().join("no-such-render-node"));
    options.fallbacks = vec![HardwareEncoder::Cpu];
    let project = test_project(&video, metadata);
    let output = directory.path().join("out.mp4");

    let (progress_tx, mut progress_rx) = mpsc::channel(64);
    let render = export_project(&project, &output, &fonts_dir, &options, progress_tx);
    let collector = tokio::spawn(async move { while progress_rx.recv().await.is_some() {} });

    let used = render
        .await
        .expect("the CPU fallback must finish the render");
    collector.await.unwrap();
    assert_eq!(
        used,
        HardwareEncoder::Cpu,
        "the failed VA-API attempt must not be reported"
    );
    assert!(std::fs::metadata(&output).unwrap().len() > 1_000);
}

#[tokio::test]
async fn a_dropped_receiver_cancels_instead_of_falling_back() {
    let Some((mut options, fonts_dir)) = render_setup().await else {
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let video = directory.path().join("clip.mp4");
    if let Err(error) = generate_clip(&video, 1).await {
        eprintln!("skipping: could not create the test clip ({error})");
        return;
    }
    let metadata = probe_video(&video).await.unwrap();
    options.duration_ms = metadata.duration_ms();
    options.encoder = HardwareEncoder::Vaapi;
    options.vaapi_device = Some(directory.path().join("no-such-render-node"));
    options.fallbacks = vec![HardwareEncoder::Cpu];
    let ass = directory.path().join("subs.ass");
    std::fs::write(
        &ass,
        sublayer_subtitles::build_ass_script(
            &[CaptionSegment::new(vec![WordToken::new("hi", 0, 500)])],
            &sublayer_subtitles::preset("tiktok").expect("built-in preset"),
            &metadata,
            &fonts_dir,
        )
        .unwrap(),
    )
    .unwrap();
    let output = directory.path().join("cancelled.mp4");

    // The receiver is gone before the render starts, so the failing VA-API
    // attempt must report cancellation rather than start a CPU fallback.
    let (progress_tx, progress_rx) = mpsc::channel(4);
    drop(progress_rx);
    let result = run_export(&video, &output, &ass, &fonts_dir, &options, progress_tx).await;
    assert!(
        matches!(result, Err(ExportError::Cancelled)),
        "expected cancellation, got {result:?}"
    );
}

#[tokio::test]
async fn missing_input_is_reported_as_a_render_failure() {
    let Some((options, fonts_dir)) = render_setup().await else {
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let ass = directory.path().join("subs.ass");
    std::fs::write(&ass, "[Script Info]\n").unwrap();

    let (progress_tx, _progress_rx) = mpsc::channel(4);
    let result = run_export(
        &directory.path().join("missing.mp4"),
        &directory.path().join("out.mp4"),
        &ass,
        &fonts_dir,
        &options,
        progress_tx,
    )
    .await;

    match result {
        Err(ExportError::Media(error)) => {
            assert!(
                error.to_string().contains("ffmpeg"),
                "unexpected error: {error}"
            );
        }
        other => panic!("expected a media error, got {other:?}"),
    }
}
