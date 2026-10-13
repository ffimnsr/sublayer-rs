//! Real-time audio engine for the player.
//!
//! One ffmpeg child decodes the audio resampled to the output format and
//! feeds it to a system audio helper over stdin — `pw-cat` on PipeWire,
//! `paplay` on PulseAudio, `aplay` on ALSA — which plays it at realtime and
//! paces the writes by backpressure. Reads are additionally bounded by the
//! wall clock, so silent playback (no helper) still reaches EOF at the
//! media's real length. Captions come from the pre-built transcript, so
//! there is no ASR pipe.
//!
//! No audio library is linked at build time: the helper is resolved from
//! `PATH` at runtime, so builds never need host dev headers. When no helper
//! (or no sound server) exists, or the helper dies mid-play, the video keeps
//! running, paced by the clock alone, and the status line says so. `Ended` is
//! reserved for the media reaching EOF.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

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
    /// (Re)starts playback at `offset_ms` (play / resume): probes an audio
    /// helper and spawns a fresh ffmpeg decode pipe.
    Start { offset_ms: u64 },
    /// Seeks at `offset_ms` (media playing): restarts only the decode pipe;
    /// the running helper is kept, so the audio gap is just the ffmpeg
    /// restart instead of a full helper probe.
    Seek { offset_ms: u64 },
    /// Stops the decoder and exits the driver (pause / stop).
    Stop,
    /// Volume in percent (0..=150, unity at 100).
    SetVolume(u8),
    /// Mute override.
    SetMute(bool),
}

/// Events emitted by the audio driver.
#[derive(Debug)]
pub enum AudioEvent {
    /// The media reached EOF (the end of the speaker decode pipe).
    Ended,
    /// Recoverable condition worth showing in the status line.
    Log(String),
}

/// Runs the driver until `commands` closes or a `Stop` arrives.
///
/// `volume_percent` and `muted_initial` are the playback settings the
/// previous driver session ended with; a fresh driver boots with them (a
/// restart must not jump back to unity volume / unmuted). The media end (EOF
/// on the speaker decode pipe) is reported as `Ended`.
pub async fn run_audio_driver(
    video: PathBuf,
    volume_percent: u8,
    muted_initial: bool,
    mut commands: mpsc::UnboundedReceiver<AudioCommand>,
    events: UnboundedSender<AudioEvent>,
) {
    let volume = Arc::new(AtomicU32::new(volume_scaled(volume_percent)));
    let muted = Arc::new(AtomicBool::new(muted_initial));

    let mut speaker: Option<AudioPcmStream> = None;
    let mut helper: Option<AudioHelper> = None;
    // Output samples already handed to the helper (or dropped while silent);
    // the wall clock bounds how much of the pipe is consumed per tick so
    // silent playback (no helper) cannot run to EOF ahead of realtime.
    let mut consumed = 0_u64;
    let mut started_at: Option<Instant> = None;

    let mut ticker = tokio::time::interval(Duration::from_millis(STEP_MS));
    loop {
        tokio::select! {
            command = commands.recv() => {
                match command {
                    Some(AudioCommand::Start { offset_ms }) => {
                        // Restart the decoder. The helper is probed first:
                        // with no sound server behind it the candidates exit
                        // immediately, and the speaker pipe is only opened
                        // when one survives.
                        speaker = None;
                        consumed = 0;
                        started_at = None;
                        helper = AudioHelper::spawn(&events, &volume, &muted).await;
                        if let Err(error) = stream_pcm_s16le(
                            &video,
                            Some(offset_ms),
                            OUTPUT_RATE,
                            OUTPUT_CHANNELS,
                        )
                        .await
                        {
                            let _ = events.send(AudioEvent::Log(format!(
                                "audio decode failed: {error}"
                            )));
                        } else {
                            started_at = Some(Instant::now());
                        }
                    }
                    Some(AudioCommand::Seek { offset_ms }) => {
                        // Restart only the decode pipe; the helper stays so
                        // the seek does not wait through another probe.
                        speaker = None;
                        consumed = 0;
                        started_at = None;
                        if let Err(error) = stream_pcm_s16le(
                            &video,
                            Some(offset_ms),
                            OUTPUT_RATE,
                            OUTPUT_CHANNELS,
                        )
                        .await
                        {
                            let _ = events.send(AudioEvent::Log(format!(
                                "audio decode failed: {error}"
                            )));
                        } else {
                            started_at = Some(Instant::now());
                        }
                    }
                    Some(AudioCommand::SetVolume(percent)) => {
                        // The slider runs 0..=150 %: unity at 100, boost past
                        // it (peaks clip in `to_s16le`).
                        volume.store(volume_scaled(percent), Ordering::Relaxed);
                    }
                    Some(AudioCommand::SetMute(mute)) => {
                        muted.store(mute, Ordering::Relaxed);
                    }
                    Some(AudioCommand::Stop) | None => return,
                }
            }
            _ = ticker.tick() => {
                let Some(speaker) = speaker.as_mut() else { continue };
                let Some(started) = started_at else { continue };

                // 48 000 Hz × 2 channels / 1 000 ms = 96 samples per ms.
                let due = started.elapsed().as_millis() as u64
                    * u64::from(OUTPUT_RATE)
                    * u64::from(OUTPUT_CHANNELS)
                    / 1_000;
                let take = (due.saturating_sub(consumed)).min(SPEAKER_CHUNK_SAMPLES as u64);
                if take == 0 {
                    continue;
                }

                // The helper's stdin backpressure paces the writes; the
                // clock bounds the reads. A helper loss (or a dead decoder)
                // drops the helper and continues silently; reaching EOF ends
                // playback.
                if !feed_speaker(speaker, &mut helper, &events, take as usize).await {
                    return;
                }
                consumed += take;
            }
        }
    }
}

/// Maps the 0..=150 % slider to the internal gain storage (10× for .1 %
/// resolution, capped at 150 % actually settable by the slider).
fn volume_scaled(percent: u8) -> u32 {
    u32::from(percent).saturating_mul(10).min(1_500)
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

/// Reads `take` output samples and pushes them to the helper as s16le.
///
/// A failed or wedged helper is dropped so video keeps running silently;
/// EOF on the decode pipe ends playback (`Ended`). Returns whether playback
/// should continue.
async fn feed_speaker(
    speaker: &mut AudioPcmStream,
    helper: &mut Option<AudioHelper>,
    events: &UnboundedSender<AudioEvent>,
    take: usize,
) -> bool {
    let mut samples = vec![0.0_f32; take];
    match read_full(speaker, &mut samples).await {
        Ok(()) => {
            let Some(active) = helper.as_mut() else {
                // Clock-paced silent playback; no helper to feed.
                return true;
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
            true
        }
        Err(ReadError::Eof) => {
            let _ = events.send(AudioEvent::Ended);
            false
        }
        Err(ReadError::Stall) => {
            *helper = None;
            let _ = events.send(AudioEvent::Log(
                "audio decoder stalled; continuing silently".to_owned(),
            ));
            true
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

/// Mute takes precedence over the volume knob; returns `0.0..=1.5` (the
/// slider runs to 150 % for boost past unity).
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
    fn volume_boost_scales_past_unity() {
        // The slider runs to 150 % so increasing the volume is always
        // audible; peaks clip in the s16 conversion.
        let volume = AtomicU32::new(1_500);
        let muted = AtomicBool::new(false);
        assert_eq!(gain_for(&volume, &muted), 1.5);

        let peak = to_s16le(&[1.0], 1.5);
        assert_eq!(
            i16::from_le_bytes([peak[0], peak[1]]),
            32_767,
            "boosted peaks clamp to the i16 range"
        );
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
