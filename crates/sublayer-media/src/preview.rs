//! Video frame preview: decodes a single frame as a raw RGBA8 buffer.
//!
//! The GUI presents frames through a decoupled pixel buffer instead of
//! embedding a video widget, so preview works identically on Wayland and X11.
//! Frames are decoded on demand with an input seek; the FFmpeg child process
//! dies with the future, so scrubbing through the timeline cannot leak
//! decoders.

use std::path::Path;
use std::time::Duration;

use sublayer_core::VideoMetadata;

use crate::MediaError;
use crate::ffmpeg;

/// Longest edge of a decoded preview frame unless the caller overrides it.
pub const DEFAULT_MAX_DIMENSION: u32 = 1_280;

/// Upper bound for a single decode invocation.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(20);

/// One decoded frame in row-major RGBA8 order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// `width * height * 4` bytes of RGBA data.
    pub pixels: Vec<u8>,
}

impl RgbaFrame {
    /// Number of pixel bytes a frame of this size must contain.
    pub fn byte_len(&self) -> usize {
        self.width as usize * self.height as usize * 4
    }
}

/// Tuning knobs for [`decode_frame_at`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreviewConfig {
    /// Longest edge of the decoded frame in pixels; `0` keeps the source
    /// resolution. The frame keeps its aspect ratio, both dimensions stay even
    /// so chroma-subsampled sources scale cleanly.
    pub max_dimension: u32,
    /// Upper bound for a single decode invocation.
    pub timeout: Duration,
}

impl Default for PreviewConfig {
    fn default() -> Self {
        Self {
            max_dimension: DEFAULT_MAX_DIMENSION,
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

/// Decodes the frame at `timestamp` of `input` into RGBA8 pixels.
///
/// `metadata` supplies the source dimensions (as returned by [`probe_video`]);
/// the frame is scaled down to [`PreviewConfig::max_dimension`] before it is
/// read back, so a 4K source never materializes a full-size buffer.
///
/// Dropping the returned future kills the decoder. A timestamp past the end of
/// the stream yields [`MediaError::NoVideoFrame`].
///
/// [`probe_video`]: crate::probe_video
pub async fn decode_frame_at(
    input: &Path,
    timestamp: Duration,
    metadata: &VideoMetadata,
    config: &PreviewConfig,
) -> Result<RgbaFrame, MediaError> {
    if metadata.width == 0 || metadata.height == 0 {
        return Err(MediaError::NoVideoStream(input.to_path_buf()));
    }
    let (width, height) = scaled_dimensions(metadata.width, metadata.height, config.max_dimension);

    let program = ffmpeg::resolve(ffmpeg::FFMPEG, ffmpeg::FFMPEG_ENV)?;
    let mut command = ffmpeg::command(&program);
    command
        .args(["-hide_banner", "-nostdin", "-loglevel", "error"])
        // The seek precedes the input for a fast jump; FFmpeg still decodes
        // from the prior keyframe and discards until the requested position.
        .args(["-ss", &format!("{:.3}", timestamp.as_secs_f64())])
        .arg("-i")
        .arg(input)
        .args(["-map", "0:v:0", "-an", "-sn", "-frames:v", "1"])
        .args(["-vf", &format!("scale={width}:{height}")])
        .args(["-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]);

    let output = tokio::time::timeout(config.timeout, ffmpeg::run(ffmpeg::FFMPEG, &mut command))
        .await
        .map_err(|_| MediaError::Timeout {
            operation: "video frame decode",
            seconds: config.timeout.as_secs(),
        })??;

    if !output.status.success() {
        return Err(classify_failure(input, output.status, &output.stderr));
    }
    if output.stdout.is_empty() {
        return Err(MediaError::NoVideoFrame {
            timestamp_ms: timestamp.as_millis() as u64,
        });
    }

    let expected = width as usize * height as usize * 4;
    let actual = output.stdout.len();
    if actual != expected {
        return Err(MediaError::InvalidFrameData { expected, actual });
    }

    Ok(RgbaFrame {
        width,
        height,
        pixels: output.stdout,
    })
}

/// Maps the FFmpeg "stream not found" trailer onto [`MediaError::NoVideoStream`]
/// and keeps everything else as [`MediaError::CommandFailed`].
fn classify_failure(input: &Path, status: std::process::ExitStatus, stderr: &[u8]) -> MediaError {
    let message = String::from_utf8_lossy(stderr);
    if message.contains("matches no streams") || message.contains("does not contain any stream") {
        return MediaError::NoVideoStream(input.to_path_buf());
    }
    ffmpeg::failure(ffmpeg::FFMPEG, status, stderr)
}

/// Target dimensions for a frame with the longest edge at most `max_dimension`.
///
/// Both edges are forced to even pixel counts (some encoders emit odd
/// dimensions for odd inputs, and RGBA scaling keeps the aspect ratio closest
/// when the result is even) and never drop below 2 pixels.
fn scaled_dimensions(width: u32, height: u32, max_dimension: u32) -> (u32, u32) {
    let longest = width.max(height);
    if max_dimension == 0 || longest <= max_dimension {
        return (even(width), even(height));
    }
    let scale = f64::from(max_dimension) / f64::from(longest);
    let width = even((f64::from(width) * scale).round() as u32);
    let height = even((f64::from(height) * scale).round() as u32);
    (width, height)
}

/// Rounds `value` down to an even number, with a floor of two pixels.
fn even(value: u32) -> u32 {
    (value & !1).max(2)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::test_util::{ffmpeg_available, generate_test_video};

    #[test]
    fn scaled_dimensions_keep_an_undersized_frame_unchanged() {
        assert_eq!(scaled_dimensions(320, 240, 1_280), (320, 240));
    }

    #[test]
    fn scaled_dimensions_round_to_even_edges() {
        // 240 * (100 / 320) = 75, rounded down to 74.
        assert_eq!(scaled_dimensions(320, 240, 100), (100, 74));
        // 1919 * (1280 / 1919) = 1280 exactly; 1080 -> 720.
        assert_eq!(scaled_dimensions(1919, 1080, 1_280), (1_280, 720));
    }

    #[test]
    fn scaled_dimensions_keep_the_source_resolution_when_unbounded() {
        assert_eq!(scaled_dimensions(640, 360, 0), (640, 360));
    }

    #[test]
    fn scaled_dimensions_never_collapse_below_two_pixels() {
        let (width, height) = scaled_dimensions(10_000, 1, 2);
        assert!(width >= 2 && height >= 2, "{width}x{height}");
    }

    #[tokio::test]
    async fn decodes_a_frame_at_native_size() {
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

        let frame = decode_frame_at(
            &video,
            Duration::from_millis(500),
            &metadata,
            &PreviewConfig::default(),
        )
        .await
        .unwrap();

        assert_eq!((frame.width, frame.height), (320, 240));
        assert_eq!(frame.pixels.len(), frame.byte_len());
        assert!(
            frame.pixels.chunks_exact(4).all(|pixel| pixel[3] == 0xFF),
            "decoded frame must be fully opaque"
        );
    }

    #[tokio::test]
    async fn decodes_a_scaled_frame() {
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
        let config = PreviewConfig {
            max_dimension: 100,
            ..PreviewConfig::default()
        };

        let frame = decode_frame_at(&video, Duration::ZERO, &metadata, &config)
            .await
            .unwrap();

        assert_eq!((frame.width, frame.height), (100, 74));
        assert_eq!(frame.pixels.len(), frame.byte_len());
    }

    #[tokio::test]
    async fn past_the_end_reports_a_missing_frame() {
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

        let result = decode_frame_at(
            &video,
            Duration::from_secs(30),
            &metadata,
            &PreviewConfig::default(),
        )
        .await;

        assert!(matches!(result, Err(MediaError::NoVideoFrame { .. })));
    }
}
