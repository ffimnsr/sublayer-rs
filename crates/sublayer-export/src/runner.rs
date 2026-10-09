//! FFmpeg render runner: burns an ASS script into a video, streaming progress.

use std::path::{Path, PathBuf};
use std::time::Instant;

use sublayer_media::MediaError;
use sublayer_media::ffmpeg;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::ChildStderr;
use tokio::sync::mpsc::Sender;

use crate::ExportError;
use crate::hardware::HardwareEncoder;
use crate::progress::{ProgressParser, ProgressTracker};

/// Largest stderr excerpt kept for error reporting.
const STDERR_KEEP_BYTES: usize = 8 * 1024;

/// Progress of a running render.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExportProgress {
    /// Encoded fraction of the source, in `0.0..=1.0`.
    pub percentage: f32,
    /// Instantaneous encoding speed in frames per second.
    pub current_fps: f32,
    /// Wall-clock time since the encoder started.
    pub elapsed_seconds: f64,
    /// Estimated seconds left, `0.0` when it cannot be estimated.
    pub eta_seconds: f64,
}

/// Tuning for one render.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportOptions {
    /// Encoder to attempt first; see [`crate::probe_hardware`].
    pub encoder: HardwareEncoder,
    /// Encoders tried, in order, when `encoder` cannot start or fails
    /// mid-render. Probing sees device nodes and FFmpeg's encoder list, but not
    /// whether a driver actually offers the encode profile, so an `auto`
    /// selection supplies the rest of its chain here. Leave empty when the
    /// user explicitly picked an encoder: that choice must not be silently
    /// downgraded.
    pub fallbacks: Vec<HardwareEncoder>,
    /// Source duration in milliseconds, used for percentage and ETA; `0` when
    /// unknown.
    pub duration_ms: u64,
    /// CRF (x264) or constant-quality value (NVENC / VA-API), `0..=51`.
    pub quality: u8,
    /// VA-API render node to upload frames to; probed when `None`.
    pub vaapi_device: Option<PathBuf>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            encoder: HardwareEncoder::Cpu,
            fallbacks: Vec::new(),
            duration_ms: 0,
            quality: 23,
            vaapi_device: None,
        }
    }
}

/// Renders `input_video` into `output_video`, burning in `ass_path`.
///
/// `options.encoder` is tried first and each of `options.fallbacks` after it,
/// so a hardware encoder that turns out to be unusable at runtime degrades to
/// the next backend instead of failing the render. Returns the encoder that
/// produced the output.
///
/// `-progress pipe:1` is parsed line by line and reported on `progress_tx`;
/// dropping the receiver kills the encoder and returns
/// [`ExportError::Cancelled`]. Dropping the future kills the encoder as well.
pub async fn run_export(
    input_video: &Path,
    output_video: &Path,
    ass_path: &Path,
    fonts_dir: &Path,
    options: &ExportOptions,
    progress_tx: Sender<ExportProgress>,
) -> Result<HardwareEncoder, ExportError> {
    let mut encoders = std::iter::once(options.encoder).chain(options.fallbacks.iter().copied());
    // The chain always starts with `options.encoder`.
    let mut current = encoders.next().expect("the encoder chain is never empty");
    loop {
        match render_once(
            input_video,
            output_video,
            ass_path,
            fonts_dir,
            options,
            current,
            progress_tx.clone(),
        )
        .await
        {
            Ok(()) => return Ok(current),
            // A dropped receiver means the caller is gone; do not keep going.
            Err(ExportError::Cancelled) => return Err(ExportError::Cancelled),
            Err(error) => {
                let Some(next) = encoders.next() else {
                    return Err(error);
                };
                // The receiver may have been dropped while this attempt ran;
                // starting another FFmpeg just to notice is pointless work.
                if progress_tx.is_closed() {
                    return Err(ExportError::Cancelled);
                }
                // The full error carries FFmpeg's stderr excerpt; keep the
                // warning to its first line and leave the rest to `debug`.
                let summary = error.to_string();
                let summary = summary.lines().next().unwrap_or_default();
                tracing::warn!(
                    failed = current.encoder_name(),
                    next = next.encoder_name(),
                    error = summary,
                    "encoder failed, retrying"
                );
                tracing::debug!(%error, "encoder failure details");
                current = next;
            }
        }
    }
}

/// One render attempt with a single encoder.
async fn render_once(
    input_video: &Path,
    output_video: &Path,
    ass_path: &Path,
    fonts_dir: &Path,
    options: &ExportOptions,
    encoder: HardwareEncoder,
    progress_tx: Sender<ExportProgress>,
) -> Result<(), ExportError> {
    let program = ffmpeg::resolve(ffmpeg::FFMPEG, ffmpeg::FFMPEG_ENV)?;
    if let Some(parent) = output_video
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    let mut command = ffmpeg::command(&program);
    command.args(["-hide_banner", "-nostdin", "-loglevel", "error", "-y"]);
    if encoder == HardwareEncoder::Vaapi {
        let device = options
            .vaapi_device
            .clone()
            .unwrap_or_else(|| PathBuf::from("/dev/dri/renderD128"));
        command.arg("-vaapi_device").arg(device);
    }
    command
        .arg("-i")
        .arg(input_video)
        .args(["-map", "0:v:0", "-map", "0:a:0?"])
        .arg("-vf")
        .arg(filter_chain(ass_path, fonts_dir, encoder))
        .args(encoder_args(options, encoder));
    command
        .args(["-c:a", "aac", "-b:a", "192k"])
        .args(muxer_args(output_video))
        .args(["-stats_period", "0.5"])
        .args(["-progress", "pipe:1", "-nostats"])
        .arg(output_video);

    tracing::debug!(encoder = encoder.encoder_name(), "starting render");
    let mut child = command.spawn().map_err(|source| MediaError::Spawn {
        binary: ffmpeg::FFMPEG,
        source,
    })?;
    let stdout = child.stdout.take().ok_or_else(|| MediaError::Spawn {
        binary: ffmpeg::FFMPEG,
        source: std::io::Error::other("ffmpeg stdout is not piped"),
    })?;
    let stderr = child.stderr.take();
    let stderr_task = stderr.map(|stderr| tokio::spawn(drain_stderr(stderr, STDERR_KEEP_BYTES)));

    let mut lines = BufReader::new(stdout).lines();
    let mut parser = ProgressParser::default();
    let tracker = ProgressTracker::new(options.duration_ms);
    let started = Instant::now();
    let mut cancelled = false;
    let mut read_error = None;

    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                if let Some(report) = parser.push_line(&line) {
                    let progress = tracker.update(&report, started.elapsed());
                    if progress_tx.send(progress).await.is_err() {
                        cancelled = true;
                        break;
                    }
                }
            }
            Ok(None) => break,
            Err(error) => {
                read_error = Some(error);
                break;
            }
        }
    }

    // Never leave a child or a helper task behind, whatever the outcome.
    let status = if cancelled || read_error.is_some() {
        let _ = child.kill().await;
        let _ = child.wait().await;
        None
    } else {
        Some(child.wait().await.map_err(|source| MediaError::Spawn {
            binary: ffmpeg::FFMPEG,
            source,
        })?)
    };
    let stderr = match stderr_task {
        Some(task) => task.await.unwrap_or_default(),
        None => Vec::new(),
    };

    if let Some(error) = read_error {
        return Err(ExportError::Io(error));
    }
    if cancelled {
        return Err(ExportError::Cancelled);
    }
    let status = status.expect("status is awaited whenever the read loop succeeded");
    if !status.success() {
        return Err(ffmpeg::failure(ffmpeg::FFMPEG, status, &stderr).into());
    }

    // The final `progress=end` block is not always flushed on success; emit a
    // definite completion so callers can finish their progress UI.
    let _ = progress_tx
        .send(ExportProgress {
            percentage: 1.0,
            current_fps: 0.0,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            eta_seconds: 0.0,
        })
        .await;
    Ok(())
}

/// Encoder arguments of the filter chain shared by every backend.
fn filter_chain(ass_path: &Path, fonts_dir: &Path, encoder: HardwareEncoder) -> String {
    let mut chain = format!(
        "ass={}:fontsdir={}",
        escape_filter_path(ass_path),
        escape_filter_path(fonts_dir)
    );
    match encoder {
        // VA-API uploads NV12 surfaces to the GPU before encoding.
        HardwareEncoder::Vaapi => chain.push_str(",format=nv12,hwupload"),
        HardwareEncoder::Nvenc | HardwareEncoder::Cpu => chain.push_str(",format=yuv420p"),
    }
    chain
}

/// Escapes a path for use as an FFmpeg filter option value.
///
/// Backslash, single quote, and colon are the characters that would otherwise
/// terminate the option; the result is wrapped in quotes to protect the
/// separators `,` and `;`.
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

/// Video-codec arguments for the selected backend.
fn encoder_args(options: &ExportOptions, encoder: HardwareEncoder) -> Vec<String> {
    let quality = options.quality.to_string();
    match encoder {
        HardwareEncoder::Vaapi => vec![
            "-c:v".to_owned(),
            "h264_vaapi".to_owned(),
            "-qp".to_owned(),
            quality,
        ],
        HardwareEncoder::Nvenc => vec![
            "-c:v".to_owned(),
            "h264_nvenc".to_owned(),
            "-preset".to_owned(),
            "p5".to_owned(),
            "-cq".to_owned(),
            quality,
        ],
        HardwareEncoder::Cpu => vec![
            "-c:v".to_owned(),
            "libx264".to_owned(),
            "-preset".to_owned(),
            "medium".to_owned(),
            "-crf".to_owned(),
            quality,
        ],
    }
}

/// Container flags; MP4 gets the streaming-friendly index.
fn muxer_args(output: &Path) -> Vec<String> {
    let is_mp4 = output
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "mp4" | "m4v" | "mov"
            )
        });
    if is_mp4 {
        vec!["-movflags".to_owned(), "+faststart".to_owned()]
    } else {
        Vec::new()
    }
}

/// Reads stderr until EOF, keeping only the last `cap` bytes.
async fn drain_stderr(mut stderr: ChildStderr, cap: usize) -> Vec<u8> {
    let mut kept: Vec<u8> = Vec::new();
    let mut buffer = [0_u8; 4_096];
    loop {
        match stderr.read(&mut buffer).await {
            Ok(0) | Err(_) => break,
            Ok(read) => {
                if kept.len() + read > cap {
                    let excess = (kept.len() + read - cap).min(kept.len());
                    kept.drain(..excess);
                }
                kept.extend_from_slice(&buffer[..read]);
            }
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn filter_chain_matches_each_encoder() {
        let chain = filter_chain(
            Path::new("/tmp/subs.ass"),
            Path::new("/tmp/fonts"),
            HardwareEncoder::Cpu,
        );
        assert_eq!(
            chain,
            "ass='/tmp/subs.ass':fontsdir='/tmp/fonts',format=yuv420p"
        );
        let chain = filter_chain(
            Path::new("/tmp/subs.ass"),
            Path::new("/tmp/fonts"),
            HardwareEncoder::Vaapi,
        );
        assert!(chain.ends_with(",format=nv12,hwupload"), "{chain}");
        let chain = filter_chain(
            Path::new("/tmp/subs.ass"),
            Path::new("/tmp/fonts"),
            HardwareEncoder::Nvenc,
        );
        assert!(chain.ends_with(",format=yuv420p"), "{chain}");
    }

    #[test]
    fn filter_paths_escape_separators() {
        assert_eq!(
            escape_filter_path(Path::new("/tmp/my:subs'ass.ass")),
            "'/tmp/my\\:subs\\'ass.ass'"
        );
        assert_eq!(
            escape_filter_path(Path::new(r"/tmp/back\slash")),
            "'/tmp/back\\\\slash'"
        );
    }

    #[test]
    fn encoder_args_use_quality_for_each_backend() {
        for (encoder, expected_flag) in [
            (HardwareEncoder::Cpu, "-crf"),
            (HardwareEncoder::Nvenc, "-cq"),
            (HardwareEncoder::Vaapi, "-qp"),
        ] {
            let options = ExportOptions {
                encoder,
                quality: 18,
                ..ExportOptions::default()
            };
            let args = encoder_args(&options, encoder);
            assert!(args.contains(&expected_flag.to_owned()), "{args:?}");
            assert!(args.contains(&"18".to_owned()), "{args:?}");
            assert!(
                args.contains(&encoder.encoder_name().to_owned()),
                "{args:?}"
            );
        }
    }

    #[test]
    fn only_mp4_like_outputs_get_faststart() {
        assert_eq!(
            muxer_args(&PathBuf::from("out.MP4")),
            vec!["-movflags".to_owned(), "+faststart".to_owned()]
        );
        assert!(muxer_args(&PathBuf::from("out.mkv")).is_empty());
    }
}
