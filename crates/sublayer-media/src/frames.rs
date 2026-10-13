//! Streaming video frame decoder for real-time playback.
//!
//! One FFmpeg child decodes from `offset_ms` to the end of the stream,
//! emitting a constant-frame-rate raw RGBA stream on stdout. Each output
//! frame sits on a fixed grid (`offset_ms + index / fps`), so a consumer can
//! pace presentation against its own clock without parsing timestamps.
//!
//! The OS pipe is smaller than one frame, so FFmpeg blocks mid-frame once the
//! pipe fills: at most one frame is ever decoded ahead, decode cost only
//! tracks presented frames, and dropping the stream kills the child
//! (`kill_on_drop`). Compare [`decode_frame_at`], which starts a fresh FFmpeg
//! process per frame and is right for scrubbing previews, not playback.

use std::path::{Path, PathBuf};
use std::time::Duration;

use sublayer_core::VideoMetadata;
use tokio::io::{AsyncReadExt, BufReader};

use crate::MediaError;
use crate::audio::drain_stderr;
use crate::ffmpeg;
use crate::preview::{DEFAULT_MAX_DIMENSION, RgbaFrame, classify_failure, scaled_dimensions};

/// Output frame rate when neither the caller nor the probe report one.
const FALLBACK_FPS: f64 = 30.0;

/// A single frame read that takes longer than this means a dead decoder.
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(5);

/// One decoded frame with its media-timeline position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedFrame {
    /// Presentation timestamp in milliseconds on the media timeline.
    pub timestamp_ms: u64,
    /// The frame itself in row-major RGBA8 order.
    pub frame: RgbaFrame,
}

/// Tuning knobs for [`stream_frames`].
#[derive(Debug, Clone, PartialEq)]
pub struct FrameStreamConfig {
    /// Longest edge of the decoded frames in pixels; `0` keeps the source
    /// resolution. Both dimensions stay even so chroma-subsampled sources
    /// scale cleanly.
    pub max_dimension: u32,
    /// Output frames per second; `0` follows the probed source rate (30 when
    /// the probe reports none). The `fps` filter maps the source onto this
    /// grid, duplicating and dropping frames as needed.
    pub fps: f64,
    /// Upper bound for a single frame read.
    pub read_timeout: Duration,
}

impl Default for FrameStreamConfig {
    fn default() -> Self {
        Self {
            max_dimension: DEFAULT_MAX_DIMENSION,
            fps: 0.0,
            read_timeout: DEFAULT_READ_TIMEOUT,
        }
    }
}

/// A live stream of decoded frames from one FFmpeg child.
///
/// Read with [`next_frame`](Self::next_frame) until it returns `None`, then
/// call [`finish`](Self::finish) to surface decoder errors. Dropping the
/// stream kills the child; stderr drains on the Tokio executor so a verbose
/// decoder can never block on a full pipe. The stream must live and be
/// finished on the same Tokio runtime that spawned the stderr drain.
pub struct VideoFrameStream {
    input: PathBuf,
    child: tokio::process::Child,
    reader: BufReader<tokio::process::ChildStdout>,
    stderr: tokio::task::JoinHandle<Vec<u8>>,
    width: u32,
    height: u32,
    frame_bytes: usize,
    /// Nanoseconds per frame on the output grid (`1e9 / fps`).
    frame_nanos: u64,
    /// Read timeout for a single frame.
    read_timeout: Duration,
    /// Media position of the next output frame, in nanoseconds.
    next_nanos: u64,
    /// Reused frame buffer; avoids an allocation per decoded frame.
    scratch: Vec<u8>,
    exhausted: bool,
}

/// Spawns ffmpeg decoding the first video stream of `input` to raw RGBA8
/// frames on stdout, starting at `offset_ms` and running to the stream end.
pub async fn stream_frames(
    input: &Path,
    offset_ms: u64,
    metadata: &VideoMetadata,
    config: &FrameStreamConfig,
) -> Result<VideoFrameStream, MediaError> {
    if metadata.width == 0 || metadata.height == 0 {
        return Err(MediaError::NoVideoStream(input.to_path_buf()));
    }
    let (width, height) = scaled_dimensions(metadata.width, metadata.height, config.max_dimension);
    let fps = if config.fps > 0.0 {
        config.fps
    } else if metadata.fps > 0.0 {
        metadata.fps
    } else {
        FALLBACK_FPS
    };
    let frame_bytes = width as usize * height as usize * 4;

    let program = ffmpeg::resolve(ffmpeg::FFMPEG, ffmpeg::FFMPEG_ENV)?;
    let mut command = ffmpeg::command(&program);
    command
        .args(["-hide_banner", "-nostdin", "-loglevel", "error"])
        // Fast input seek; ffmpeg decodes from the prior keyframe and drops
        // everything before the target, so the output starts at `offset_ms`
        // (within one frame).
        .args(["-ss", &format!("{:.3}", offset_ms as f64 / 1_000.0)])
        .arg("-i")
        .arg(input)
        .args(["-map", "0:v:0", "-an", "-sn"])
        .args(["-vf", &format!("scale={width}:{height},fps={fps:.6}")])
        .args(["-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]);

    tracing::debug!(command = %ffmpeg::describe(&command), "streaming video frames");
    let mut child = command.spawn().map_err(|source| MediaError::Spawn {
        binary: ffmpeg::FFMPEG,
        source,
    })?;
    let stdout = child
        .stdout
        .take()
        .expect("ffmpeg::command always pipes stdout");
    let stderr = child
        .stderr
        .take()
        .expect("ffmpeg::command always pipes stderr");
    let stderr_task = tokio::spawn(drain_stderr(stderr));

    Ok(VideoFrameStream {
        input: input.to_path_buf(),
        child,
        reader: BufReader::new(stdout),
        stderr: stderr_task,
        width,
        height,
        frame_bytes,
        frame_nanos: (1e9 / fps).round() as u64,
        read_timeout: config.read_timeout,
        next_nanos: offset_ms.saturating_mul(1_000_000),
        scratch: Vec::new(),
        exhausted: false,
    })
}

impl VideoFrameStream {
    /// Reads the next frame, or `Ok(None)` at the end of the stream. A frame
    /// that starts but does not finish (a truncated stream, or a decoder that
    /// died mid-frame) counts as the end; [`finish`](Self::finish) then
    /// surfaces the decoder's exit status.
    pub async fn next_frame(&mut self) -> Result<Option<DecodedFrame>, MediaError> {
        if self.exhausted {
            return Ok(None);
        }
        let read = tokio::time::timeout(self.read_timeout, self.read_raw())
            .await
            .map_err(|_| MediaError::Timeout {
                operation: "video frame read",
                seconds: self.read_timeout.as_secs(),
            })?;
        let bytes = match read {
            Ok(bytes) => bytes,
            Err(error) => {
                self.exhausted = true;
                return Err(error.into());
            }
        };
        if bytes.len() < self.frame_bytes {
            self.exhausted = true;
            return Ok(None);
        }
        let timestamp_ms = self.next_nanos.saturating_add(500_000) / 1_000_000;
        self.next_nanos = self.next_nanos.saturating_add(self.frame_nanos);
        Ok(Some(DecodedFrame {
            timestamp_ms,
            frame: RgbaFrame {
                width: self.width,
                height: self.height,
                pixels: bytes,
            },
        }))
    }

    /// Waits for the decoder to exit and reports its status: a non-zero exit
    /// (for example an input without a video stream) surfaces as
    /// [`MediaError::NoVideoStream`] or [`MediaError::CommandFailed`].
    ///
    /// The child is killed first so a stream that was not fully drained
    /// cannot deadlock: ffmpeg would otherwise sit blocked on a full stdout
    /// pipe while this waits for it to exit.
    pub async fn finish(mut self) -> Result<(), MediaError> {
        // Killing an already-exited child is a harmless no-op error.
        let _ = self.child.kill().await;
        let status = match self.child.wait().await {
            Ok(status) => status,
            Err(source) => {
                return Err(MediaError::Spawn {
                    binary: ffmpeg::FFMPEG,
                    source,
                });
            }
        };
        let stderr = self.stderr.await.unwrap_or_default();
        if !status.success() {
            return Err(classify_failure(&self.input, status, &stderr));
        }
        Ok(())
    }

    /// Reads exactly one frame into the scratch buffer, or as much of it as
    /// the decoder produced before the stream ended.
    async fn read_raw(&mut self) -> std::io::Result<Vec<u8>> {
        self.scratch.clear();
        self.scratch.resize(self.frame_bytes, 0);
        let mut filled = 0;
        while filled < self.frame_bytes {
            match self.reader.read(&mut self.scratch[filled..]).await {
                Ok(0) => break,
                Ok(read) => filled += read,
                Err(error) => return Err(error),
            }
        }
        self.scratch.truncate(filled);
        Ok(std::mem::take(&mut self.scratch))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::preview::{PreviewConfig, decode_frame_at};
    use crate::test_util::{ffmpeg_available, generate_test_video};

    /// Reads `count` frames from the stream, or as many as it produces.
    async fn read_frames(stream: &mut VideoFrameStream, count: usize) -> Vec<DecodedFrame> {
        let mut frames = Vec::new();
        for _ in 0..count {
            let Some(frame) = stream.next_frame().await.expect("decoder must not fail") else {
                break;
            };
            frames.push(frame);
        }
        frames
    }

    /// Drains the rest of the stream and waits for a clean decoder exit.
    /// `finish` kills the child first, so it must only see a fully drained
    /// stream (or an already-exited one, like the audio stream reader).
    async fn drain_and_finish(stream: VideoFrameStream) {
        let mut stream = stream;
        while stream
            .next_frame()
            .await
            .expect("decoder must not fail")
            .is_some()
        {}
        stream.finish().await.expect("clean decoder exit");
    }

    #[tokio::test]
    async fn streams_frames_from_the_offset_on_the_output_grid() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let video = directory.path().join("clip.mp4");
        if !generate_test_video(&video, false) {
            eprintln!("skipping: ffmpeg could not create the test clip");
            return;
        }
        let metadata = crate::probe_video(&video).await.unwrap();

        let mut stream = stream_frames(&video, 500, &metadata, &FrameStreamConfig::default())
            .await
            .unwrap();

        let frames = read_frames(&mut stream, 3).await;
        assert_eq!(frames.len(), 3);
        assert_eq!((frames[0].frame.width, frames[0].frame.height), (320, 240));
        // 30 fps grid from the 500 ms offset: 500, 533, 567 ms.
        assert_eq!(
            frames
                .iter()
                .map(|frame| frame.timestamp_ms)
                .collect::<Vec<_>>(),
            vec![500, 533, 567]
        );
        assert!(
            frames.iter().all(|frame| frame
                .frame
                .pixels
                .chunks_exact(4)
                .all(|pixel| pixel[3] == 0xFF)),
            "decoded frames must be fully opaque"
        );
        drain_and_finish(stream).await;
    }

    #[tokio::test]
    async fn a_custom_output_rate_resizes_the_grid() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let video = directory.path().join("clip.mp4");
        if !generate_test_video(&video, false) {
            eprintln!("skipping: ffmpeg could not create the test clip");
            return;
        }
        let metadata = crate::probe_video(&video).await.unwrap();
        let config = FrameStreamConfig {
            fps: 60.0,
            ..FrameStreamConfig::default()
        };

        let mut stream = stream_frames(&video, 500, &metadata, &config)
            .await
            .unwrap();
        let frames = read_frames(&mut stream, 3).await;
        // 60 fps grid: 500, 517, 533 ms.
        assert_eq!(
            frames
                .iter()
                .map(|frame| frame.timestamp_ms)
                .collect::<Vec<_>>(),
            vec![500, 517, 533]
        );
        drain_and_finish(stream).await;
    }

    #[tokio::test]
    async fn seeking_lands_on_the_same_frame_as_decode_frame_at() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let video = directory.path().join("clip.mp4");
        if !generate_test_video(&video, false) {
            eprintln!("skipping: ffmpeg could not create the test clip");
            return;
        }
        let metadata = crate::probe_video(&video).await.unwrap();

        let mut stream = stream_frames(&video, 500, &metadata, &FrameStreamConfig::default())
            .await
            .unwrap();
        let first = read_frames(&mut stream, 1)
            .await
            .pop()
            .expect("the seek must yield a frame");

        let single = decode_frame_at(
            &video,
            Duration::from_millis(500),
            &metadata,
            &PreviewConfig::default(),
        )
        .await
        .unwrap();

        assert_eq!(first.timestamp_ms, 500);
        assert_eq!(first.frame, single);
        drain_and_finish(stream).await;
    }

    #[tokio::test]
    async fn the_stream_runs_to_eof_and_finishes_cleanly() {
        if !ffmpeg_available() {
            eprintln!("skipping: ffmpeg not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let video = directory.path().join("clip.mp4");
        if !generate_test_video(&video, false) {
            eprintln!("skipping: ffmpeg could not create the test clip");
            return;
        }
        let metadata = crate::probe_video(&video).await.unwrap();

        let mut stream = stream_frames(&video, 0, &metadata, &FrameStreamConfig::default())
            .await
            .unwrap();
        let mut frames = Vec::new();
        while let Some(frame) = stream.next_frame().await.unwrap() {
            frames.push(frame);
        }
        let duration_ms = (metadata.duration_seconds * 1_000.0) as u64;
        assert!(!frames.is_empty());
        assert_eq!(frames[0].timestamp_ms, 0);
        assert!(frames.last().unwrap().timestamp_ms < duration_ms);
        let monotonic = frames
            .windows(2)
            .all(|pair| pair[0].timestamp_ms < pair[1].timestamp_ms);
        assert!(monotonic, "frame timestamps must rise strictly");
        stream.finish().await.expect("clean decoder exit");
    }
}
