//! Real-time video engine: one persistent FFmpeg decoder per play segment.
//!
//! Mirrors the audio driver: a dedicated Tokio task owns a
//! [`VideoFrameStream`] and serves frame requests from the UI pump. Advancing
//! past the playhead decodes only the single frame that becomes due next, so
//! a stalled UI costs nothing and a slow decoder can never build a buffer of
//! undisplayed frames — the picture always tracks the wall-clock playhead.

use std::path::{Path, PathBuf};

use sublayer_core::VideoMetadata;
use sublayer_media::{DecodedFrame, FrameStreamConfig, stream_frames};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

/// A buffered frame this far behind the playhead (a small slider seek, or a
/// decoder that fell behind the clock) is stale: the stream is pulled
/// forward to the playhead instead of presenting it.
const PULL_FORWARD_MS: u64 = 1_000;

/// Commands sent to the video driver.
pub enum VideoCommand {
    /// (Re)starts the decoder at `offset_ms`, killing the running one.
    Start { offset_ms: u64 },
    /// Drops the decoder (pause / stop / model or language changes).
    Stop,
    /// Presents the frame for `playhead_ms`: the newest frame due at or
    /// before the playhead is emitted (at most one per advance, so a
    /// high-fps source cannot flood the UI); the next frame in the future
    /// is held until it comes due.
    Advance { playhead_ms: u64 },
}

/// Events emitted by the video driver.
pub enum VideoEvent {
    /// A frame on the media timeline, in decode order.
    Frame(DecodedFrame),
    /// Recoverable condition worth showing in the status line.
    Log(String),
}

/// Plays the media's frame rate (or the fallback) at the probe dimensions.
fn stream_config(metadata: &VideoMetadata) -> FrameStreamConfig {
    FrameStreamConfig {
        fps: metadata.fps,
        ..FrameStreamConfig::default()
    }
}

/// Runs the driver until `commands` closes or a `Stop` arrives.
pub async fn run_video_driver(
    video: PathBuf,
    metadata: VideoMetadata,
    mut commands: UnboundedReceiver<VideoCommand>,
    events: UnboundedSender<VideoEvent>,
) {
    let mut stream: Option<sublayer_media::VideoFrameStream> = None;
    let mut pending: Option<DecodedFrame> = None;

    while let Some(command) = commands.recv().await {
        match command {
            VideoCommand::Start { offset_ms } => {
                // Dropping the old stream kills its FFmpeg child.
                stream = None;
                pending = None;
                match stream_frames(&video, offset_ms, &metadata, &stream_config(&metadata)).await {
                    Ok(started) => stream = Some(started),
                    Err(error) => {
                        let _ =
                            events.send(VideoEvent::Log(format!("video decode failed: {error}")));
                    }
                }
            }
            VideoCommand::Stop => {
                stream = None;
                pending = None;
            }
            VideoCommand::Advance { playhead_ms } => {
                advance(
                    &mut stream,
                    &mut pending,
                    &video,
                    &metadata,
                    playhead_ms,
                    &events,
                )
                .await;
            }
        }
    }
}

/// Emits the newest frame due at `playhead_ms`, decoding at most one frame
/// ahead of it.
///
/// Intermediately due frames (high-fps sources, or a playhead that jumped)
/// are read and dropped so the UI receives at most one frame per advance;
/// the picture still tracks the wall clock. When the buffered frame falls
/// more than [`PULL_FORWARD_MS`] behind the playhead — a small slider seek,
/// or a decoder that cannot keep up — the stream is restarted at the
/// playhead so the picture follows the slider instead of presenting stale
/// frames. A decoder failure or EOF drops the stream; the audio driver owns
/// `PlaybackEnded`, so video EOF is silent unless the decoder exited badly.
async fn advance(
    stream: &mut Option<sublayer_media::VideoFrameStream>,
    pending: &mut Option<DecodedFrame>,
    video: &Path,
    metadata: &VideoMetadata,
    playhead_ms: u64,
    events: &UnboundedSender<VideoEvent>,
) {
    let mut last_due: Option<DecodedFrame> = None;
    loop {
        if pending.is_none() {
            let next = match stream.as_mut() {
                Some(active) => active.next_frame().await,
                None => break,
            };
            match next {
                Ok(Some(frame)) => *pending = Some(frame),
                Ok(None) => {
                    let ended = match stream.take() {
                        Some(stream) => stream.finish().await,
                        None => Ok(()),
                    };
                    if let Err(error) = ended {
                        let _ =
                            events.send(VideoEvent::Log(format!("video decode ended: {error}")));
                    }
                    break;
                }
                Err(error) => {
                    let _ = events.send(VideoEvent::Log(format!("video decode failed: {error}")));
                    *stream = None;
                    break;
                }
            }
        }
        let Some(decoded) = pending.take() else { break };
        if decoded.timestamp_ms.saturating_add(PULL_FORWARD_MS) < playhead_ms {
            // The frame is stale (small slider seek, or a decoder that fell
            // behind): restart the stream at the playhead instead.
            match stream_frames(video, playhead_ms, metadata, &stream_config(metadata)).await {
                Ok(started) => *stream = Some(started),
                Err(error) => {
                    let _ = events.send(VideoEvent::Log(format!("video decode failed: {error}")));
                    break;
                }
            }
            continue;
        }
        if decoded.timestamp_ms > playhead_ms {
            // Not due yet; hold it for the next advance.
            *pending = Some(decoded);
            break;
        }
        last_due = Some(decoded);
    }
    if let Some(frame) = last_due {
        let _ = events.send(VideoEvent::Frame(frame));
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Stdio;
    use std::time::Duration;

    use sublayer_media::probe_video;
    use tokio::process::Command;
    use tokio::sync::mpsc;

    use super::*;

    /// Generates a 320x240@30fps MPEG-4 fixture of the given length;
    /// skips when ffmpeg is unavailable.
    async fn fixture(path: &Path, seconds: u32) -> bool {
        let Ok(program) = which::which("ffmpeg") else {
            return false;
        };
        Command::new(program)
            .args(["-hide_banner", "-nostdin", "-loglevel", "error", "-y"])
            .args(["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30"])
            .args(["-t", &seconds.to_string(), "-c:v", "mpeg4", "-q:v", "5"])
            .arg(path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_ok_and(|status| status.success())
    }

    /// Collects decoded frames until `expected` arrive or the driver goes
    /// quiet for a while (it holds a future frame without emitting anything).
    async fn collect_frames(
        events: &mut UnboundedReceiver<VideoEvent>,
        expected: usize,
    ) -> Vec<DecodedFrame> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        let mut frames = Vec::new();
        while frames.len() < expected && tokio::time::Instant::now() < deadline {
            let event = tokio::time::timeout(Duration::from_millis(500), events.recv()).await;
            match event {
                Ok(Some(VideoEvent::Frame(frame))) => frames.push(frame),
                Ok(Some(VideoEvent::Log(_))) | Ok(None) => break,
                Err(_) => break, // quiet: the driver holds a future frame
            }
        }
        frames
    }

    #[tokio::test]
    async fn driver_emits_frames_due_at_the_playhead_and_holds_future_ones() {
        let directory = tempfile::tempdir().unwrap();
        let video = directory.path().join("clip.mp4");
        if !fixture(&video, 4).await {
            eprintln!("skipping: ffmpeg could not generate the fixture");
            return;
        }
        let metadata = probe_video(&video).await.unwrap();
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        tokio::spawn(run_video_driver(
            video,
            metadata.clone(),
            command_rx,
            event_tx,
        ));

        command_tx
            .send(VideoCommand::Start { offset_ms: 0 })
            .unwrap();

        // Playhead at 500 ms: the 0..500 ms frames are due, the 533 ms
        // frame must be held back; only the newest due frame is emitted.
        command_tx
            .send(VideoCommand::Advance { playhead_ms: 500 })
            .unwrap();
        let first = collect_frames(&mut event_rx, 1).await;
        assert_eq!(first.len(), 1, "at most one frame per advance");
        assert_eq!(first[0].timestamp_ms, 500);

        // Same playhead again: the held 533 ms frame is still future, so
        // nothing is emitted.
        command_tx
            .send(VideoCommand::Advance { playhead_ms: 500 })
            .unwrap();
        assert!(collect_frames(&mut event_rx, 1).await.is_empty());

        // A playhead jump without a restart command (a small slider seek
        // below the audio restart threshold) pulls the stream forward to
        // the playhead instead of presenting stale frames.
        command_tx
            .send(VideoCommand::Advance { playhead_ms: 2_500 })
            .unwrap();
        let pulled = collect_frames(&mut event_rx, 1).await;
        assert_eq!(pulled.len(), 1);
        assert_eq!(pulled[0].timestamp_ms, 2_500);

        // Stepping the playhead up in pump-sized chunks (as the UI does)
        // reaches the clip end; the stream ends silently (audio owns
        // PlaybackEnded). Each step emits the newest due frame.
        for playhead in (2_600..=4_000).step_by(100) {
            command_tx
                .send(VideoCommand::Advance {
                    playhead_ms: playhead,
                })
                .unwrap();
        }
        let tail = collect_frames(&mut event_rx, 100).await;
        let duration_ms = (metadata.duration_seconds * 1_000.0) as u64;
        // 2600..=3900 step 100 emits 14 frames, plus the last grid frame
        // before the 4 s end (encoder rounding may shave one off).
        assert!(
            (14..=15).contains(&tail.len()),
            "expected ~15 tail frames, got {}",
            tail.len()
        );
        assert_eq!(tail[0].timestamp_ms, 2_600);
        assert!(
            tail.last().unwrap().timestamp_ms < duration_ms,
            "the last frame stays inside the clip: {}",
            duration_ms
        );

        // Stop kills the stream; further advances emit nothing.
        command_tx.send(VideoCommand::Stop).unwrap();
        command_tx
            .send(VideoCommand::Advance { playhead_ms: 0 })
            .unwrap();
        assert!(collect_frames(&mut event_rx, 1).await.is_empty());
    }
}
