//! Real-time audio engine for the player.
//!
//! One ffmpeg child decodes the audio twice: once resampled to the output
//! format (fed to a system audio helper over stdin — `pw-cat` on PipeWire,
//! `paplay` on PulseAudio, `aplay` on ALSA — which plays it at realtime and
//! paces the whole pipeline by backpressure), once as 16 kHz mono for the
//! whisper captioner. A wall clock anchors the caption reads, so captions
//! stay aligned with the playhead even when the helper lags.
//!
//! No audio library is linked at build time: the helper is resolved from
//! `PATH` at runtime, so builds never need host dev headers. When no helper
//! (or no sound server) exists, or the helper dies mid-play, the video and
//! the caption track keep running, paced by the clock alone, and the status
//! line says so. `Ended` is reserved for the media reaching EOF.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use sublayer_ai::LiveCaptioner;
use sublayer_media::{AudioPcmStream, stream_pcm_s16le};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::mpsc::{self, UnboundedSender};

/// How often the pipeline is stepped.
const STEP_MS: u64 = 20;
/// A read that stalls this long is treated as a dead decoder.
const READ_STALL: Duration = Duration::from_millis(1_000);
/// How long a freshly spawned helper gets to open the sound server before we
/// trust it. With no server behind it (`pw-cat`/`paplay`/`aplay` all fail
/// fast), the helper exits within milliseconds and we fall through to the
/// next candidate.
const PROBE_MS: Duration = Duration::from_millis(150);
/// A helper that stops consuming audio for this long is considered wedged
/// and gets dropped (playback continues silently).
const WRITE_STALL: Duration = Duration::from_secs(2);
/// Output format fed to the audio helper: 48 kHz stereo s16le (resampled by
/// ffmpeg; sinks almost universally accept it, and PipeWire resamples anyway).
const OUTPUT_RATE: u32 = 48_000;
const OUTPUT_CHANNELS: u16 = 2;
/// Roughly 40 ms of output per speaker read (natural write block is the
/// real pacing mechanism, this only bounds the per-tick buffer).
const SPEAKER_CHUNK_SAMPLES: usize = OUTPUT_RATE as usize * OUTPUT_CHANNELS as usize / 25;

/// Commands sent to the audio driver.
pub enum AudioCommand {
    /// (Re)starts playback at `offset_ms`: kills the old decoders, resets the
    /// captioner timeline, and spawns fresh ffmpeg pipes.
    Start { offset_ms: u64 },
    /// Stops the decoders and exits the driver (pause / stop / seek).
    Stop,
    /// Volume in percent (0..=100).
    SetVolume(u8),
    /// Mute override.
    SetMute(bool),
}

/// Events emitted by the audio driver.
#[derive(Debug)]
pub enum AudioEvent {
    /// The media hit EOF (or the audio helper died mid-stream).
    Ended,
    /// Recoverable condition worth showing in the status line.
    Log(String),
}

/// Runs the driver until `commands` closes or a `Stop` arrives.
///
/// `captioner` is `None` when playback uses a pre-built transcript: the ASR
/// pipe (and its whisper feed) is skipped entirely, and the media end is
/// reported by the pump clock instead of the ASR decoder.
pub async fn run_audio_driver(
    video: PathBuf,
    captioner: Option<Arc<LiveCaptioner>>,
    mut commands: mpsc::UnboundedReceiver<AudioCommand>,
    events: UnboundedSender<AudioEvent>,
) {
    let volume = Arc::new(AtomicU32::new(1_000));
    let muted = Arc::new(AtomicBool::new(false));

    let mut readers: Option<(Option<AudioPcmStream>, Option<AudioPcmStream>)> = None;
    let mut helper: Option<AudioHelper> = None;
    let mut started_at: Option<Instant> = None;
    let mut consumed_asr = 0_u64;

    let mut ticker = tokio::time::interval(Duration::from_millis(STEP_MS));
    loop {
        tokio::select! {
            command = commands.recv() => {
                match command {
                    Some(AudioCommand::Start { offset_ms }) => {
                        // Restart decoders and the caption timeline. The helper
                        // is probed first: with no sound server behind it the
                        // candidates exit immediately, and the speaker pipe is
                        // only opened when one survives.
                        readers = None;
                        if let Some(captioner) = captioner.as_ref() {
                            let _ = captioner.reset(offset_ms);
                        }
                        consumed_asr = 0;
                        helper = AudioHelper::spawn(&events, &volume, &muted).await;
                        match open_readers(
                            &video,
                            offset_ms,
                            helper.is_some(),
                            captioner.is_some(),
                        )
                        .await
                        {
                            Ok(pair) => {
                                readers = Some(pair);
                                started_at = Some(Instant::now());
                            }
                            Err(error) => {
                                let _ = helper.take();
                                let _ = events.send(AudioEvent::Log(format!(
                                    "audio decode failed: {error}"
                                )));
                            }
                        }
                    }
                    Some(AudioCommand::SetVolume(percent)) => {
                        let scaled = u32::from(percent).saturating_mul(10).min(1_000);
                        volume.store(scaled, Ordering::Relaxed);
                    }
                    Some(AudioCommand::SetMute(mute)) => {
                        muted.store(mute, Ordering::Relaxed);
                    }
                    Some(AudioCommand::Stop) | None => return,
                }
            }
            _ = ticker.tick() => {
                let Some(started) = started_at else { continue };
                let Some((speaker, asr)) = readers.as_mut() else { continue };
                let elapsed_ms = started.elapsed().as_millis() as u64;

                // Speaker pipe: paced purely by the helper's stdin
                // backpressure. A helper loss (or a dead speaker decoder)
                // drops the helper and continues silently; the caption track
                // is the product, sound is not.
                if let Some(speaker) = speaker.as_mut() {
                    feed_speaker(speaker, &mut helper, &events).await;
                }

                // ASR pipe: 16 kHz mono, read up to the realtime position.
                // 16 000 Hz / 1 000 ms = 16 samples per millisecond; a
                // fractional factor would feed whisper at a fraction of
                // realtime and no window would ever fill during a clip.
                let asr_target = elapsed_ms * 16;
                let asr_take = asr_target.saturating_sub(consumed_asr);
                if asr_take > 0 {
                    let Some(asr) = asr.as_mut() else { continue };
                    let mut samples = vec![0.0_f32; asr_take as usize];
                    match read_full(asr, &mut samples).await {
                        Ok(()) => {
                            consumed_asr += samples.len() as u64;
                            if let Some(captioner) = captioner.as_ref() {
                                let _ = captioner.push(samples);
                            }
                        }
                        Err(ReadError::Eof) => {
                            if let Some(captioner) = captioner.as_ref() {
                                let _ = captioner.flush();
                            }
                            let _ = events.send(AudioEvent::Ended);
                            return;
                        }
                        Err(ReadError::Stall) => {
                            let _ = events.send(AudioEvent::Log("audio decoder stalled".to_owned()));
                            return;
                        }
                    }
                }
            }
        }
    }
}

/// Why a read did not fill its buffer.
enum ReadError {
    Eof,
    Stall,
}

/// Reads `samples` completely, or fails on EOF / a stalled decoder.
async fn read_full(stream: &mut AudioPcmStream, samples: &mut [f32]) -> Result<(), ReadError> {
    let mut filled = 0;
    while filled < samples.len() {
        match tokio::time::timeout(READ_STALL, stream.read_samples(&mut samples[filled..])).await {
            Err(_) => return Err(ReadError::Stall),
            Ok(Err(_)) | Ok(Ok(0)) => return Err(ReadError::Eof),
            Ok(Ok(read)) => filled += read,
        }
    }
    Ok(())
}

/// Opens the ASR decoder (16 kHz, when captions are live) and, when a helper
/// is actually running, the speaker decoder at `offset_ms`.
async fn open_readers(
    video: &Path,
    offset_ms: u64,
    with_speaker: bool,
    with_asr: bool,
) -> Result<(Option<AudioPcmStream>, Option<AudioPcmStream>), sublayer_media::MediaError> {
    let asr = if with_asr {
        Some(stream_pcm_s16le(video, Some(offset_ms), 16_000, 1).await?)
    } else {
        None
    };
    let speaker = if with_speaker {
        Some(stream_pcm_s16le(video, Some(offset_ms), OUTPUT_RATE, OUTPUT_CHANNELS).await?)
    } else {
        None
    };
    Ok((speaker, asr))
}

/// Reads one speaker chunk and pushes it to the helper as s16le.
///
/// A failed or wedged helper, or a dead speaker decoder, is dropped so the
/// rest of the playback (video, captions, translation) keeps running
/// silently; `Ended` is left to the ASR pipe, which is the media's own EOF.
async fn feed_speaker(
    speaker: &mut AudioPcmStream,
    helper: &mut Option<AudioHelper>,
    events: &UnboundedSender<AudioEvent>,
) {
    let mut samples = vec![0.0_f32; SPEAKER_CHUNK_SAMPLES];
    match read_full(speaker, &mut samples).await {
        Ok(()) => {
            if samples.is_empty() {
                return;
            }
            let Some(active) = helper.as_mut() else {
                return;
            };
            let written = tokio::time::timeout(WRITE_STALL, active.write(&samples)).await;
            match written {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    *helper = None;
                    let _ = events.send(AudioEvent::Log(format!(
                        "audio helper lost ({error}); continuing silently"
                    )));
                }
                Err(_) => {
                    *helper = None;
                    let _ = events.send(AudioEvent::Log(
                        "audio helper stalled; continuing silently".to_owned(),
                    ));
                }
            }
        }
        Err(ReadError::Eof) => {
            *helper = None;
            let _ = events.send(AudioEvent::Log(
                "audio track ended; continuing silently".to_owned(),
            ));
        }
        Err(ReadError::Stall) => {
            *helper = None;
            let _ = events.send(AudioEvent::Log(
                "audio decoder stalled; continuing silently".to_owned(),
            ));
        }
    }
}

/// System audio helper candidates, in preference order.
struct HelperKind {
    program: &'static str,
    /// `pw-cat` (PipeWire), `paplay` (PulseAudio), `aplay` (ALSA): all read
    /// raw little-endian 16-bit PCM from stdin.
    args: fn(u32, u16) -> Vec<String>,
}

const HELPER_KINDS: [HelperKind; 3] = [
    HelperKind {
        program: "pw-cat",
        args: pw_cat_args,
    },
    HelperKind {
        program: "paplay",
        args: paplay_args,
    },
    HelperKind {
        program: "aplay",
        args: aplay_args,
    },
];

fn pw_cat_args(rate: u32, channels: u16) -> Vec<String> {
    vec![
        "--playback".into(),
        "--format".into(),
        "s16".into(),
        "--rate".into(),
        rate.to_string(),
        "--channels".into(),
        channels.to_string(),
    ]
}

fn paplay_args(rate: u32, channels: u16) -> Vec<String> {
    vec![
        "--raw".into(),
        "--format=s16le".into(),
        format!("--rate={rate}"),
        format!("--channels={channels}"),
    ]
}

fn aplay_args(rate: u32, channels: u16) -> Vec<String> {
    vec![
        "-q".into(),
        "-r".into(),
        rate.to_string(),
        "-c".into(),
        channels.to_string(),
        "-f".into(),
        "S16_LE".into(),
        "-".into(),
    ]
}

/// The running helper: the child plus the volume/mute knobs it applies.
struct AudioHelper {
    /// Kept alive so `kill_on_drop` reaps the helper if the driver dies
    /// early (seek/pause drop the driver, which kills the child).
    #[allow(dead_code)]
    child: tokio::process::Child,
    stdin: tokio::process::ChildStdin,
    volume: Arc<AtomicU32>,
    muted: Arc<AtomicBool>,
}

impl AudioHelper {
    /// Spawns the first helper that survives the probe (a helper that exits
    /// during it never reached a sound server); `None` (with a status note)
    /// when there is no helper binary or no sound server behind any of them.
    async fn spawn(
        events: &UnboundedSender<AudioEvent>,
        volume: &Arc<AtomicU32>,
        muted: &Arc<AtomicBool>,
    ) -> Option<Self> {
        for kind in &HELPER_KINDS {
            let Ok(program) = which::which(kind.program) else {
                continue;
            };
            let mut command = Command::new(&program);
            command
                .args((kind.args)(OUTPUT_RATE, OUTPUT_CHANNELS))
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .kill_on_drop(true);
            let mut child = match command.spawn() {
                Ok(child) => child,
                Err(error) => {
                    tracing::warn!(program = kind.program, %error, "audio helper failed to start");
                    continue;
                }
            };
            let Some(stdin) = child.stdin.take() else {
                let _ = events.send(AudioEvent::Log(format!("{} refused stdin", kind.program)));
                continue;
            };
            match tokio::time::timeout(PROBE_MS, child.wait()).await {
                Ok(status) => {
                    tracing::warn!(
                        program = kind.program,
                        ?status,
                        "audio helper exited during probe (no sound server?)"
                    );
                    continue;
                }
                Err(_) => {
                    tracing::debug!(program = kind.program, "audio helper started");
                    let _ = events.send(AudioEvent::Log(format!(
                        "playing audio via {}",
                        kind.program
                    )));
                    return Some(Self {
                        child,
                        stdin,
                        volume: Arc::clone(volume),
                        muted: Arc::clone(muted),
                    });
                }
            }
        }
        let _ = events.send(AudioEvent::Log(
            "no working audio helper (pw-cat/paplay/aplay); playing silently".to_owned(),
        ));
        None
    }

    /// Converts `samples` to s16le (applying `gain`) and writes the bytes to
    /// the helper's stdin; errors mean the helper (or sound server) died.
    ///
    /// The helper is always fed, even when muted or the volume is zero:
    /// skipping the write starves its stdin, and the helper (or the sound
    /// server behind it) gives up — silence keeps the pipe and the realtime
    /// clock alive.
    async fn write(&mut self, samples: &[f32]) -> std::io::Result<()> {
        let gain = gain_for(&self.volume, &self.muted);
        let bytes = to_s16le(samples, gain);
        self.stdin.write_all(&bytes).await
    }
}

/// Mute takes precedence over the volume knob; returns `0.0..=1.0`.
fn gain_for(volume: &AtomicU32, muted: &AtomicBool) -> f32 {
    if muted.load(Ordering::Relaxed) {
        0.0
    } else {
        volume.load(Ordering::Relaxed) as f32 / 1_000.0
    }
}

/// Converts normalized samples to little-endian s16 after applying `gain`.
fn to_s16le(samples: &[f32], gain: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let scaled = (sample * gain).clamp(-1.0, 1.0);
        bytes.extend_from_slice(&((scaled * 32_767.0) as i16).to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helper_candidates_cover_pipewire_pulse_and_alsa_in_order() {
        let programs: Vec<&str> = HELPER_KINDS.iter().map(|kind| kind.program).collect();
        assert_eq!(programs, ["pw-cat", "paplay", "aplay"]);
    }

    #[test]
    fn helper_arguments_carry_the_output_format() {
        let (pw, pa, aw) = (
            pw_cat_args(48_000, 2),
            paplay_args(48_000, 2),
            aplay_args(48_000, 2),
        );
        assert_eq!(
            pw,
            [
                "--playback",
                "--format",
                "s16",
                "--rate",
                "48000",
                "--channels",
                "2"
            ]
        );
        assert_eq!(
            pa,
            ["--raw", "--format=s16le", "--rate=48000", "--channels=2"]
        );
        assert_eq!(aw, ["-q", "-r", "48000", "-c", "2", "-f", "S16_LE", "-"]);
    }

    #[test]
    fn s16le_conversion_applies_gain_and_clamps() {
        let samples = vec![-1.5, -0.5, 0.0, 0.5, 1.5];
        let bytes = to_s16le(&samples, 1.0);
        let values: Vec<i16> = bytes
            .windows(2)
            .step_by(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        assert_eq!(
            values,
            [-32_767, -16_383, 0, 16_383, 32_767],
            "unit gain, ends clamped to i16 range"
        );

        // Half gain scales in between; nothing clamps.
        let values: Vec<i16> = to_s16le(&samples, 0.5)
            .windows(2)
            .step_by(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        assert_eq!(values, [-24_575, -8_191, 0, 8_191, 24_575]);
    }

    #[test]
    fn mute_overrides_the_volume_knob() {
        let volume = AtomicU32::new(1_000);
        let muted = AtomicBool::new(true);
        assert_eq!(gain_for(&volume, &muted), 0.0);

        muted.store(false, Ordering::Relaxed);
        volume.store(250, Ordering::Relaxed);
        assert_eq!(gain_for(&volume, &muted), 0.25);
    }

    #[test]
    fn muted_or_zero_volume_still_produces_writable_silence() {
        // The helper must be fed silence instead of being starved, or it
        // (or the sound server) gives up and sound is lost for the rest of
        // the playback.
        let samples = [-1.0, 0.0, 1.0];
        let bytes = to_s16le(&samples, 0.0);
        assert_eq!(bytes, vec![0; 6], "zero gain must emit digital silence");
    }
}
