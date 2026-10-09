//! `sublayer` — headless transcription, probing, and (later) rendering.
//!
//! End-to-end pipeline for CI and batch work: extract 16 kHz audio, transcribe
//! with a pinned Whisper model, cluster the words into caption cards, and
//! compile them into ASS (with theme animation), SRT, VTT, or caption JSON.
//!
//! Progress is drawn on stderr with `indicatif`; stdout stays reserved for
//! machine-readable output, so `sublayer probe in.mp4` can be piped into
//! `jq`.

mod errors;

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use sublayer_ai::{ModelManager, TranscriberConfig, transcribe_audio};
use sublayer_core::{SublayerPaths, VideoMetadata};
use sublayer_media::{extract_audio_16k, probe_video};
use sublayer_subtitles::{
    build_ass_script, build_srt, build_vtt, preset, resolve_fonts_dir, segment_words,
};
use tokio::sync::mpsc;

use errors::CliError;

/// Command-line interface of the Sublayer caption studio.
#[derive(Debug, Parser)]
#[command(
    name = "sublayer",
    version,
    about = "Local-first AI video caption studio (headless)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Prints FFprobe metadata of a video as JSON.
    Probe {
        /// Media file to inspect.
        input: PathBuf,
    },
    /// Extracts 16 kHz audio, transcribes it, and compiles caption files.
    Transcribe {
        /// Media file to transcribe.
        input: PathBuf,
        /// Whisper model: a pinned name (`tiny.en`, `base.en`, `small.en`,
        /// `large-v3-turbo`) or a path to an already-downloaded GGML file.
        #[arg(short, long, default_value = "base.en")]
        model: String,
        /// Output file; the extension selects the format
        /// (`.ass`, `.srt`, `.vtt`, `.json`). Defaults to the input stem with `.ass`.
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Theme preset name (`tiktok`, `hormozi`, `podcast`, `cyber`,
        /// `cinematic`) or a path to a custom theme JSON file.
        #[arg(long, default_value = "tiktok-classic")]
        theme: String,
        /// Spoken language (`en`, `de`, ...); auto-detected when omitted.
        #[arg(long)]
        language: Option<String>,
        /// Skips the energy VAD pre-filter.
        #[arg(long)]
        no_vad: bool,
        /// Requests the GPU backend (requires the `vulkan` feature build).
        #[arg(long)]
        gpu: bool,
        /// Whisper worker threads; defaults to the CPU count.
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Renders captions onto a video. Lands with `sublayer-export` (phase 5).
    Render {
        /// Media file to render.
        input: PathBuf,
        /// Output video file.
        #[arg(short, long)]
        output: PathBuf,
        /// Theme preset name or custom theme JSON path.
        #[arg(long, default_value = "tiktok-classic")]
        theme: String,
    },
}

/// Target formats a caption compile can produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Ass,
    Srt,
    Vtt,
    Json,
}

impl OutputFormat {
    /// Selects the format from the file extension.
    fn from_path(path: &Path) -> Result<Self, CliError> {
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        match extension.as_str() {
            "ass" => Ok(Self::Ass),
            "srt" => Ok(Self::Srt),
            "vtt" => Ok(Self::Vtt),
            "json" => Ok(Self::Json),
            other => Err(CliError::UnsupportedOutput(other.to_owned())),
        }
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(error) = run(cli).await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

/// Options forwarded from the `transcribe` subcommand into the pipeline.
struct TranscribeOptions {
    input: PathBuf,
    model: String,
    output: Option<PathBuf>,
    theme: String,
    language: Option<String>,
    no_vad: bool,
    gpu: bool,
    threads: Option<usize>,
}

/// Dispatches the parsed command. Kept separate from `main` so tests can run
/// the pipeline without spawning a process.
async fn run(cli: Cli) -> Result<(), CliError> {
    match cli.command {
        Command::Probe { input } => run_probe(&input).await,
        Command::Transcribe {
            input,
            model,
            output,
            theme,
            language,
            no_vad,
            gpu,
            threads,
        } => {
            run_transcribe(TranscribeOptions {
                input,
                model,
                output,
                theme,
                language,
                no_vad,
                gpu,
                threads,
            })
            .await
        }
        Command::Render {
            input,
            output,
            theme,
        } => {
            let _ = (input, output, theme);
            Err(CliError::RenderDeferred)
        }
    }
}

/// Prints FFprobe metadata as pretty JSON.
async fn run_probe(input: &Path) -> Result<(), CliError> {
    let metadata = probe_video(input).await?;
    println!("{}", serde_json::to_string_pretty(&metadata)?);
    Ok(())
}

/// Full transcribe pipeline: probe → WAV → model → whisper → segments → file.
///
/// Argument errors (bad theme, bad output extension, missing input) are
/// reported before any download or inference work starts.
async fn run_transcribe(options: TranscribeOptions) -> Result<(), CliError> {
    let TranscribeOptions {
        input,
        model,
        output,
        theme,
        language,
        no_vad,
        gpu,
        threads,
    } = options;
    if !input.is_file() {
        return Err(CliError::MissingInput(input.clone()));
    }
    let theme = resolve_theme(&theme)?;
    let output = output
        .map(|output| output.to_path_buf())
        .unwrap_or_else(|| input.with_extension("ass"));
    let format = OutputFormat::from_path(&output)?;

    let metadata = probe_video(&input).await?;
    let audio = tempfile::Builder::new()
        .prefix("sublayer-")
        .suffix(".wav")
        .tempfile()?;
    let wav_path = audio.path().to_path_buf();

    println!("Extracting 16 kHz mono audio from {}", input.display());
    extract_audio_16k(&input, &wav_path).await?;

    println!("Ensuring model `{model}` is available");
    let manager = ModelManager::new(SublayerPaths::resolve()?)?;
    // `--model` accepts either a pinned model name (downloaded on demand) or
    // a path to an existing GGML file, which skips the downloader.
    let model_path = if Path::new(&model).is_file() {
        PathBuf::from(&model)
    } else {
        manager.ensure_cached(&model).await?
    };

    let config = TranscriberConfig {
        model_path,
        language,
        enable_vad: !no_vad,
        use_gpu: gpu,
        threads: threads.unwrap_or_else(|| TranscriberConfig::default().threads),
    };

    println!("Transcribing with model `{model}`");
    let words = transcribe_with_progress(&wav_path, &config).await?;
    println!("Transcribed {} words", words.len());

    let segments = segment_words(&words, &sublayer_subtitles::SegmenterConfig::default());

    let fonts_dir = resolve_fonts_dir();

    let body = compile(&segments, format, &theme, &metadata, &fonts_dir)?;
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, body)?;
    println!("Wrote {}", output.display());
    Ok(())
}

/// Drives transcription while painting an indicatif progress bar on stderr.
async fn transcribe_with_progress(
    wav_path: &Path,
    config: &TranscriberConfig,
) -> Result<Vec<sublayer_core::WordToken>, CliError> {
    let bar = ProgressBar::new(100);
    bar.set_style(
        ProgressStyle::with_template("{msg} [{bar:20}] {percent}%")
            .expect("static progress template is valid"),
    );
    bar.set_message("transcribing");

    let (progress_tx, mut progress_rx) = mpsc::channel(64);
    // `transcribe_audio` runs the blocking whisper call on the thread pool
    // itself; this outer task only couples it to the progress bar loop.
    let wav_path = wav_path.to_path_buf();
    let config = config.clone();
    let task = tokio::spawn(async move { transcribe_audio(&wav_path, &config, progress_tx).await });
    while let Some(percent) = progress_rx.recv().await {
        bar.set_position((percent * 100.0).round() as u64);
    }
    bar.finish_and_clear();

    let words = task
        .await
        .map_err(|error| CliError::TaskJoin(error.to_string()))??;
    Ok(words)
}

/// Resolves a theme preset alias (or JSON path) into a `ThemeStyle`.
fn resolve_theme(theme_arg: &str) -> Result<sublayer_core::ThemeStyle, CliError> {
    if theme_arg.ends_with(".json") {
        return Ok(sublayer_subtitles::themes::load_json_file(Path::new(
            theme_arg,
        ))?);
    }
    preset(theme_arg).ok_or_else(|| CliError::UnknownTheme(theme_arg.to_owned()))
}

/// Compiles caption segments into the requested format.
fn compile(
    segments: &[sublayer_core::CaptionSegment],
    format: OutputFormat,
    theme: &sublayer_core::ThemeStyle,
    metadata: &VideoMetadata,
    fonts_dir: &Path,
) -> Result<Vec<u8>, CliError> {
    match format {
        OutputFormat::Ass => {
            Ok(build_ass_script(segments, theme, metadata, fonts_dir)?.into_bytes())
        }
        OutputFormat::Srt => Ok(build_srt(segments).into_bytes()),
        OutputFormat::Vtt => Ok(build_vtt(segments).into_bytes()),
        OutputFormat::Json => Ok(serde_json::to_vec_pretty(segments)?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn transcribe_defaults_model_and_theme() {
        let cli = Cli::try_parse_from(["sublayer", "transcribe", "in.mp4"]).unwrap();
        let Command::Transcribe {
            model,
            theme,
            output,
            no_vad,
            ..
        } = cli.command
        else {
            panic!("expected transcribe command");
        };
        assert_eq!(model, "base.en");
        assert_eq!(theme, "tiktok-classic");
        assert!(output.is_none());
        assert!(!no_vad);
    }

    #[test]
    fn transcribe_accepts_explicit_flags() {
        let cli = Cli::try_parse_from([
            "sublayer",
            "transcribe",
            "in.mp4",
            "-m",
            "small.en",
            "-o",
            "out.srt",
            "--no-vad",
            "--gpu",
            "--threads",
            "8",
        ])
        .unwrap();
        let Command::Transcribe {
            model,
            output,
            no_vad,
            gpu,
            threads,
            ..
        } = cli.command
        else {
            panic!("expected transcribe command");
        };
        assert_eq!(model, "small.en");
        assert_eq!(output.unwrap(), PathBuf::from("out.srt"));
        assert!(no_vad);
        assert!(gpu);
        assert_eq!(threads, Some(8));
    }

    #[test]
    fn output_format_follows_extension() {
        assert_eq!(
            OutputFormat::from_path(Path::new("subs.ass")).unwrap(),
            OutputFormat::Ass
        );
        assert_eq!(
            OutputFormat::from_path(Path::new("subs.SRT")).unwrap(),
            OutputFormat::Srt
        );
        assert_eq!(
            OutputFormat::from_path(Path::new("subs.vtt")).unwrap(),
            OutputFormat::Vtt
        );
        assert_eq!(
            OutputFormat::from_path(Path::new("subs.json")).unwrap(),
            OutputFormat::Json
        );
        assert!(matches!(
            OutputFormat::from_path(Path::new("subs.txt")),
            Err(CliError::UnsupportedOutput(_))
        ));
    }

    #[test]
    fn render_reports_deferred_phase() {
        let cli = Cli::try_parse_from(["sublayer", "render", "in.mp4", "-o", "out.mp4"]).unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(run(cli));
        assert!(matches!(result, Err(CliError::RenderDeferred)));
    }

    #[tokio::test]
    #[ignore = "requires a downloaded Whisper model and FFmpeg; set SUBLAYER_TEST_MODEL"]
    async fn transcribe_end_to_end() {
        let Some(model) = std::env::var("SUBLAYER_TEST_MODEL").ok() else {
            eprintln!("skipping: SUBLAYER_TEST_MODEL not set");
            return;
        };
        let directory = tempfile::tempdir().unwrap();
        let video = directory.path().join("clip.mp4");
        if !generate_test_video(&video) {
            eprintln!("skipping: ffmpeg could not create the test clip");
            return;
        }

        let output = directory.path().join("subs.ass");
        let cli = Cli::try_parse_from([
            "sublayer",
            "transcribe",
            video.to_str().unwrap(),
            "-m",
            &model,
            "-o",
            output.to_str().unwrap(),
        ])
        .unwrap();
        run(cli).await.unwrap();

        let script = std::fs::read_to_string(&output).unwrap();
        assert!(
            script.contains("[Script Info]"),
            "output is not an ASS script"
        );
    }

    /// Generates a one-second 320x240@30 clip with a sine track.
    fn generate_test_video(path: &Path) -> bool {
        let program = std::env::var("SUBLAYER_FFMPEG").unwrap_or_else(|_| "ffmpeg".to_owned());
        std::process::Command::new(program)
            .args(["-hide_banner", "-nostdin", "-loglevel", "error", "-y"])
            .args(["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30"])
            .args(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000"])
            .args(["-t", "1", "-c:v", "mpeg4", "-q:v", "5", "-c:a", "pcm_s16le"])
            .arg(path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }
}
