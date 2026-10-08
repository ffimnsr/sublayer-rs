//! FFprobe wrapper: turns a media file into [`VideoMetadata`].

use std::path::Path;
use std::time::Duration;

use serde::Deserialize;
use sublayer_core::VideoMetadata;

use crate::{MediaError, ffmpeg};

/// Upper bound for a single probe invocation; probing reads only headers.
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

/// Reads duration, resolution, frame rate, and audio layout of `path`.
///
/// Dropping the returned future kills the FFprobe child process.
pub async fn probe_video(path: &Path) -> Result<VideoMetadata, MediaError> {
    let program = ffmpeg::resolve(ffmpeg::FFPROBE, ffmpeg::FFPROBE_ENV)?;
    let mut command = ffmpeg::command(&program);
    command
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
        ])
        .arg(path);

    let output = tokio::time::timeout(PROBE_TIMEOUT, ffmpeg::run(ffmpeg::FFPROBE, &mut command))
        .await
        .map_err(|_| MediaError::InvalidProbeData("ffprobe timed out after 30s".to_owned()))??;

    if !output.status.success() {
        return Err(ffmpeg::failure(
            ffmpeg::FFPROBE,
            output.status,
            &output.stderr,
        ));
    }

    let json = String::from_utf8_lossy(&output.stdout);
    parse_probe_json(path, &json)
}

/// FFprobe's JSON document, trimmed to the fields Sublayer uses.
#[derive(Debug, Deserialize)]
struct ProbeOutput {
    #[serde(default)]
    streams: Vec<ProbeStream>,
    #[serde(default)]
    format: Option<ProbeFormat>,
}

#[derive(Debug, Deserialize)]
struct ProbeStream {
    #[serde(default)]
    codec_type: Option<String>,
    #[serde(default)]
    codec_name: Option<String>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    avg_frame_rate: Option<String>,
    #[serde(default)]
    r_frame_rate: Option<String>,
    #[serde(default)]
    channels: Option<u32>,
    #[serde(default)]
    sample_rate: Option<String>,
    #[serde(default)]
    duration: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProbeFormat {
    #[serde(default)]
    duration: Option<String>,
}

/// Parses `ffprobe -show_format -show_streams -print_format json` output.
///
/// Duration and frame rate are optional in FFprobe output; missing or
/// unparsable values fall back to `0.0` so a partially tagged file still opens.
fn parse_probe_json(path: &Path, json: &str) -> Result<VideoMetadata, MediaError> {
    let probe: ProbeOutput = serde_json::from_str(json)
        .map_err(|error| MediaError::InvalidProbeData(error.to_string()))?;

    let video = probe
        .streams
        .iter()
        .find(|stream| stream.codec_type.as_deref() == Some("video"))
        .ok_or_else(|| MediaError::NoVideoStream(path.to_path_buf()))?;
    let audio = probe
        .streams
        .iter()
        .find(|stream| stream.codec_type.as_deref() == Some("audio"));

    let duration_seconds = probe
        .format
        .as_ref()
        .and_then(|format| format.duration.as_deref())
        .and_then(parse_number)
        .or_else(|| video.duration.as_deref().and_then(parse_number))
        .unwrap_or(0.0);

    let fps = [
        video.avg_frame_rate.as_deref(),
        video.r_frame_rate.as_deref(),
    ]
    .into_iter()
    .flatten()
    .filter_map(parse_rational)
    .find(|fps| *fps > 0.0)
    .unwrap_or(0.0);

    Ok(VideoMetadata {
        duration_seconds,
        width: video.width.unwrap_or(0),
        height: video.height.unwrap_or(0),
        fps,
        video_codec: video.codec_name.clone(),
        audio_codec: audio.and_then(|stream| stream.codec_name.clone()),
        audio_channels: audio.and_then(|stream| stream.channels),
        audio_sample_rate: audio
            .and_then(|stream| stream.sample_rate.as_deref())
            .and_then(|rate| rate.parse().ok()),
    })
}

/// Parses a plain decimal number, rejecting `N/A` and non-finite values.
fn parse_number(value: &str) -> Option<f64> {
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite())
}

/// Parses FFprobe's `"numerator/denominator"` frame rates.
fn parse_rational(value: &str) -> Option<f64> {
    let (numerator, denominator) = value.split_once('/')?;
    let numerator = parse_number(numerator)?;
    let denominator = parse_number(denominator)?;
    if denominator == 0.0 {
        return None;
    }
    let ratio = numerator / denominator;
    ratio.is_finite().then_some(ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLIP_JSON: &str = r#"{
        "streams": [
            {
                "index": 0,
                "codec_name": "h264",
                "codec_type": "video",
                "width": 1080,
                "height": 1920,
                "r_frame_rate": "30000/1001",
                "avg_frame_rate": "30000/1001",
                "duration": "12.345000"
            },
            {
                "index": 1,
                "codec_name": "aac",
                "codec_type": "audio",
                "sample_rate": "48000",
                "channels": 2,
                "duration": "12.330000"
            }
        ],
        "format": { "duration": "12.345000" }
    }"#;

    const SILENT_JSON: &str = r#"{
        "streams": [
            { "codec_name": "vp9", "codec_type": "video", "width": 640, "height": 360 }
        ],
        "format": null
    }"#;

    #[test]
    fn parses_video_and_audio_streams() {
        let metadata = parse_probe_json(Path::new("clip.mp4"), CLIP_JSON).unwrap();
        assert_eq!(metadata.width, 1080);
        assert_eq!(metadata.height, 1920);
        assert!((metadata.fps - 29.97).abs() < 0.001, "fps {}", metadata.fps);
        assert!((metadata.duration_seconds - 12.345).abs() < 1e-6);
        assert_eq!(metadata.video_codec.as_deref(), Some("h264"));
        assert_eq!(metadata.audio_codec.as_deref(), Some("aac"));
        assert_eq!(metadata.audio_channels, Some(2));
        assert_eq!(metadata.audio_sample_rate, Some(48_000));
        assert!(metadata.has_audio());
    }

    #[test]
    fn falls_back_to_stream_duration_and_frame_rate() {
        let json = r#"{
            "streams": [
                {
                    "codec_type": "video",
                    "width": 640,
                    "height": 360,
                    "avg_frame_rate": "0/0",
                    "r_frame_rate": "25/1",
                    "duration": "3.500000"
                }
            ],
            "format": {}
        }"#;
        let metadata = parse_probe_json(Path::new("clip.webm"), json).unwrap();
        assert_eq!(metadata.fps, 25.0);
        assert_eq!(metadata.duration_seconds, 3.5);
    }

    #[test]
    fn treats_video_only_input_as_silent() {
        let metadata = parse_probe_json(Path::new("silent.webm"), SILENT_JSON).unwrap();
        assert_eq!(metadata.video_codec.as_deref(), Some("vp9"));
        assert_eq!(metadata.audio_codec, None);
        assert_eq!(metadata.audio_channels, None);
        assert_eq!(metadata.audio_sample_rate, None);
        assert_eq!(metadata.duration_seconds, 0.0);
        assert_eq!(metadata.fps, 0.0);
        assert!(!metadata.has_audio());
    }

    #[test]
    fn rejects_inputs_without_a_video_stream() {
        let json = r#"{ "streams": [ { "codec_type": "audio", "codec_name": "mp3" } ] }"#;
        assert!(matches!(
            parse_probe_json(Path::new("song.mp3"), json),
            Err(MediaError::NoVideoStream(path)) if path == Path::new("song.mp3")
        ));
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(matches!(
            parse_probe_json(Path::new("clip.mp4"), "not json"),
            Err(MediaError::InvalidProbeData(_))
        ));
    }

    #[test]
    fn rational_parser_handles_common_ffprobe_encodings() {
        assert_eq!(parse_rational("30000/1001"), Some(30_000.0 / 1_001.0));
        assert_eq!(parse_rational("24/1"), Some(24.0));
        assert_eq!(parse_rational("0/0"), None);
        assert_eq!(parse_rational("25"), None);
        assert_eq!(parse_number("N/A"), None);
        assert_eq!(parse_number("12.5"), Some(12.5));
        assert_eq!(parse_number("inf"), None);
    }

    #[tokio::test]
    async fn probes_a_real_h264_clip() {
        if !crate::test_util::ffmpeg_available() {
            eprintln!("skipping: ffmpeg is not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let video = directory.path().join("clip.mp4");
        if !crate::test_util::generate_test_video(&video, true) {
            eprintln!("skipping: ffmpeg could not generate the fixture");
            return;
        }

        let metadata = tokio::time::timeout(Duration::from_secs(60), probe_video(&video))
            .await
            .expect("probe timed out")
            .expect("probe failed");

        assert_eq!(metadata.width, 320);
        assert_eq!(metadata.height, 240);
        assert!((metadata.fps - 30.0).abs() < 0.01, "fps {}", metadata.fps);
        assert!(
            (metadata.duration_seconds - 1.0).abs() < 0.25,
            "duration {}",
            metadata.duration_seconds
        );
        assert_eq!(metadata.video_codec.as_deref(), Some("mpeg4"));
        assert_eq!(metadata.audio_codec.as_deref(), Some("aac"));
        assert_eq!(metadata.audio_channels, Some(1));
        assert_eq!(metadata.audio_sample_rate, Some(48_000));
    }

    #[tokio::test]
    async fn probes_a_video_without_audio() {
        if !crate::test_util::ffmpeg_available() {
            eprintln!("skipping: ffmpeg is not available");
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let video = directory.path().join("silent.mp4");
        if !crate::test_util::generate_test_video(&video, false) {
            eprintln!("skipping: ffmpeg could not generate the fixture");
            return;
        }

        let metadata = tokio::time::timeout(Duration::from_secs(60), probe_video(&video))
            .await
            .expect("probe timed out")
            .expect("probe failed");

        assert!(!metadata.has_audio());
        assert_eq!(metadata.audio_codec, None);
    }
}
