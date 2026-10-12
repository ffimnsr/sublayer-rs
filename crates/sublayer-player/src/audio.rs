//! Real-time audio engine for the player.
//!
//! One ffmpeg child decodes the audio twice: once resampled to the output
//! device (fed to `cpal` through a bounded channel — backpressure paces the
//! whole pipeline), once as 16 kHz mono for the whisper captioner. A wall
//! clock anchors both reads, so captions stay aligned with the playhead even
//! when the device callback lags.
//!
//! Audio output is an opt-in build feature (`audio`); without it — or when
//! no device exists at runtime — the video and the caption track keep
//! running, paced by the clock alone.

use std::path::PathBuf;
use std::sync::Arc;
#[cfg(feature = "audio")]
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use sublayer_ai::LiveCaptioner;
use sublayer_media::{AudioPcmStream, stream_pcm_s16le};
use tokio::sync::mpsc::{self, UnboundedSender};

/// How often the pipeline is stepped.
const STEP_MS: u64 = 20;
/// A read that stalls this long is treated as a dead decoder.
const READ_STALL: Duration = Duration::from_millis(1_000);

/// Commands sent to the audio driver.
pub enum AudioCommand {
    /// (Re)starts playback at `offset_ms`: kills the old decoders, resets the
    /// captioner timeline, and spawns fresh ffmpeg pipes.
    Start { offset_ms: u64 },
    /// Stops the decoders and exits the driver (pause / stop / seek).
    Stop,
    /// Volume in percent (0..=100).
    #[cfg_attr(not(feature = "audio"), allow(dead_code))]
    SetVolume(u8),
    /// Mute override.
    #[cfg_attr(not(feature = "audio"), allow(dead_code))]
    SetMute(bool),
}

/// Events emitted by the audio driver.
#[derive(Debug)]
pub enum AudioEvent {
    /// Both decoders hit EOF: playback reached the end of the media.
    Ended,
    /// Recoverable condition worth showing in the status line.
    Log(String),
}

/// Runs the driver until `commands` closes or a `Stop` arrives.
pub async fn run_audio_driver(
    video: PathBuf,
    captioner: Arc<LiveCaptioner>,
    mut commands: mpsc::UnboundedReceiver<AudioCommand>,
    events: UnboundedSender<AudioEvent>,
) {
    #[cfg(feature = "audio")]
    let output = AudioOutput::open(&events).await;
    #[cfg(feature = "audio")]
    let mut active = output;

    #[cfg(feature = "audio")]
    let (feed_tx, feed_rx) = crossbeam_channel::bounded::<Vec<f32>>(48);
    #[cfg(not(feature = "audio"))]
    let (_feed_tx, _feed_rx) = crossbeam_channel::unbounded::<Vec<f32>>();

    let mut readers: Option<(Option<AudioPcmStream>, AudioPcmStream)> = None;
    let mut started_at: Option<Instant> = None;
    let mut consumed_asr = 0_u64;
    #[cfg(feature = "audio")]
    let mut consumed_speaker = 0_u64;

    let mut ticker = tokio::time::interval(Duration::from_millis(STEP_MS));
    loop {
        tokio::select! {
            command = commands.recv() => {
                match command {
                    Some(AudioCommand::Start { offset_ms }) => {
                        // Restart decoders and the caption timeline.
                        readers = None;
                        let _ = captioner.reset(offset_ms);
                        consumed_asr = 0;
                        #[cfg(feature = "audio")]
                        {
                            consumed_speaker = 0;
                            active.stream(&feed_rx);
                        }
                        match open_readers(&video, offset_ms).await {
                            Ok(pair) => {
                                readers = Some(pair);
                                started_at = Some(Instant::now());
                            }
                            Err(error) => {
                                let _ = events.send(AudioEvent::Log(format!(
                                    "audio decode failed: {error}"
                                )));
                            }
                        }
                    }
                    #[cfg(feature = "audio")]
                    Some(AudioCommand::SetVolume(percent)) => active.set_volume(percent),
                    #[cfg(not(feature = "audio"))]
                    Some(AudioCommand::SetVolume(_)) => {}
                    #[cfg(feature = "audio")]
                    Some(AudioCommand::SetMute(mute)) => active.set_mute(mute),
                    #[cfg(not(feature = "audio"))]
                    Some(AudioCommand::SetMute(_)) => {}
                    Some(AudioCommand::Stop) | None => return,
                }
            }
            _ = ticker.tick() => {
                let Some(started) = started_at else { continue };
                let Some((speaker, asr)) = readers.as_mut() else { continue };
                #[cfg(not(feature = "audio"))]
                let _ = speaker;
                let elapsed_ms = started.elapsed().as_millis() as u64;

                #[cfg(feature = "audio")]
                {
                    if let Some(speaker) = speaker.as_mut() {
                        feed_speaker(
                            speaker,
                            &feed_tx,
                            &mut consumed_speaker,
                            elapsed_ms,
                            &events,
                        )
                        .await;
                    }
                }

                // ASR pipe: 16 kHz mono, read up to the realtime position.
                let asr_target = elapsed_ms * 16 / 10;
                let asr_take = asr_target.saturating_sub(consumed_asr);
                if asr_take > 0 {
                    let mut samples = vec![0.0_f32; asr_take as usize];
                    match read_full(asr, &mut samples).await {
                        Ok(()) => {
                            consumed_asr += samples.len() as u64;
                            let _ = captioner.push(samples);
                        }
                        Err(ReadError::Eof) => {
                            let _ = captioner.flush();
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

/// Spawns the speaker and ASR decoders at `offset_ms`.
async fn open_readers(
    video: &PathBuf,
    offset_ms: u64,
) -> Result<(Option<AudioPcmStream>, AudioPcmStream), sublayer_media::MediaError> {
    let asr = stream_pcm_s16le(video, Some(offset_ms), 16_000, 1).await?;
    #[cfg(feature = "audio")]
    {
        let (sample_rate, channels) = SpeakerConfig::read();
        let speaker = stream_pcm_s16le(video, Some(offset_ms), sample_rate, channels).await?;
        return Ok((Some(speaker), asr));
    }
    #[cfg(not(feature = "audio"))]
    {
        let _ = video;
        Ok((None, asr))
    }
}

/// Feeds the speaker pipe at the realtime position; a full feed channel drops
/// the chunk (one gap) instead of stalling the async runtime.
#[cfg(feature = "audio")]
async fn feed_speaker(
    speaker: &mut AudioPcmStream,
    feed_tx: &crossbeam_channel::Sender<Vec<f32>>,
    consumed: &mut u64,
    elapsed_ms: u64,
    events: &UnboundedSender<AudioEvent>,
) {
    let (sample_rate, channels) = SpeakerConfig::read();
    let speaker_target = elapsed_ms * u64::from(sample_rate) * u64::from(channels) / 1_000;
    let speaker_take = speaker_target
        .saturating_sub(*consumed)
        .min(u64::from(sample_rate) * u64::from(channels) * 2);
    if speaker_take == 0 {
        return;
    }
    let mut samples = vec![0.0_f32; speaker_take as usize];
    match read_full(speaker, &mut samples).await {
        Ok(()) => {
            *consumed += samples.len() as u64;
            if !samples.is_empty() {
                let _ = feed_tx.try_send(samples);
            }
        }
        Err(ReadError::Eof) => {
            let _ = events.send(AudioEvent::Ended);
        }
        Err(ReadError::Stall) => {
            let _ = events.send(AudioEvent::Log("audio decoder stalled".to_owned()));
        }
    }
}

/// cpal device state: the output stream plus the volume/mute knobs.
#[cfg(feature = "audio")]
struct AudioOutput {
    device: Option<cpal::Device>,
    stream: Option<cpal::Stream>,
    volume: Arc<AtomicU32>,
    muted: Arc<AtomicBool>,
}

#[cfg(feature = "audio")]
impl AudioOutput {
    /// Opens the default device; a missing one degrades to silent playback.
    async fn open(events: &UnboundedSender<AudioEvent>) -> Self {
        let device = cpal::default_host().default_output_device();
        if device.is_none() {
            let _ = events.send(AudioEvent::Log(
                "no audio output device; playing silently".to_owned(),
            ));
        }
        Self {
            device,
            stream: None,
            volume: Arc::new(AtomicU32::new(1_000)),
            muted: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Builds (once) and plays the output stream fed by `feed_rx`.
    fn stream(&mut self, feed_rx: &crossbeam_channel::Receiver<Vec<f32>>) {
        if self.stream.is_some() || self.device.is_none() {
            return;
        }
        let device = self.device.as_ref().expect("checked above");
        let Ok(supported) = device.default_output_config() else {
            return;
        };
        let config = supported.config();
        if config.sample_rate.0 == 0 || config.channels == 0 {
            return;
        }
        let rx = feed_rx.clone();
        let volume = Arc::clone(&self.volume);
        let muted = Arc::clone(&self.muted);
        match device.build_output_stream(
            &config,
            move |data: &mut [f32], _| {
                fill_output(data, &rx, &volume, &muted);
            },
            |error| {
                tracing::debug!(%error, "audio output stream error");
            },
        ) {
            Ok(stream) => {
                if let Err(error) = stream.play() {
                    tracing::warn!(%error, "audio stream refused to play");
                }
                self.stream = Some(stream);
            }
            Err(error) => {
                tracing::warn!(%error, "audio output unavailable");
            }
        }
    }

    fn set_volume(&self, percent: u8) {
        let scaled = u32::from(percent).saturating_mul(10).min(1_000);
        self.volume.store(scaled, Ordering::Relaxed);
    }

    fn set_mute(&self, muted: bool) {
        self.muted.store(muted, Ordering::Relaxed);
    }
}

/// Sample rate and channel layout of the current device.
#[cfg(feature = "audio")]
struct SpeakerConfig {
    sample_rate: u32,
    channels: u16,
}

#[cfg(feature = "audio")]
impl SpeakerConfig {
    fn read() -> Self {
        if let Some(device) = cpal::default_host().default_output_device() {
            if let Ok(config) = device.default_output_config() {
                return Self {
                    sample_rate: config.sample_rate().0,
                    channels: config.channels(),
                };
            }
        }
        Self {
            sample_rate: 48_000,
            channels: 2,
        }
    }
}

/// Device callback: drains the feed channel into `data`, applying volume and
/// mute; underruns produce silence.
#[cfg(feature = "audio")]
fn fill_output(
    data: &mut [f32],
    feed: &crossbeam_channel::Receiver<Vec<f32>>,
    volume: &AtomicU32,
    muted: &AtomicBool,
) {
    let gain = if muted.load(Ordering::Relaxed) {
        0.0
    } else {
        f32::from(volume.load(Ordering::Relaxed)) / 1_000.0
    };
    if gain == 0.0 {
        data.fill(0.0);
        return;
    }
    let mut spool: Vec<f32> = Vec::new();
    let mut spool_pos = 0_usize;
    for slot in data {
        if spool_pos >= spool.len() {
            match feed.try_recv() {
                Ok(chunk) => {
                    spool = chunk;
                    spool_pos = 0;
                }
                Err(_) => {
                    // Underrun: the rest of this callback is silence.
                    *slot = 0.0;
                    continue;
                }
            }
        }
        *slot = spool[spool_pos] * gain;
        spool_pos += 1;
    }
}
