//! Background orchestration for the player.
//!
//! One Tokio runtime hosts the audio driver and frame decoding; a dedicated
//! thread runs the whisper captioner (`LiveCaptioner`) and another the
//! thread-affine translation backend (`TranslateWorker`). Commands arrive
//! from the UI thread over an unbounded channel; events flow back to the UI
//! pump over a crossbeam channel.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crossbeam_channel::Sender;
use sublayer_ai::{LiveCaptioner, LiveEvent, LiveTranscriberConfig, ModelManager};
use sublayer_core::{SublayerPaths, VideoMetadata, WordToken};
use sublayer_media::{RgbaFrame, probe_video};
use sublayer_translate::{LocalPair, TranslationModelManager, find_local_pair};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::audio::{AudioCommand, AudioEvent, run_audio_driver};
use crate::transcript_cache;
use crate::translate_worker::{TranslateEvent, TranslateWorker};
use crate::video::{VideoCommand, VideoEvent, run_video_driver};

/// Caption window: longer lag, more reliable decodes.
const ASR_WINDOW_MS: u64 = 4_000;

/// Events the UI pump understands.
#[derive(Debug)]
pub enum PlayerEvent {
    /// Probe finished; the media can be played.
    Opened {
        title: String,
        duration_ms: u64,
        /// Full path of the media (transcript caching).
        path: PathBuf,
    },
    /// Status-line text.
    Status(String),
    /// A cached transcript was restored; captions are pre-built already.
    PrebuiltLoaded {
        words: Vec<WordToken>,
        cards: Vec<crate::session::PrebuiltCard>,
    },
    /// Recognized words (media-timeline timestamps).
    Words(Vec<WordToken>),
    /// Whisper settled on a language (auto mode).
    LanguageDetected(String),
    /// A finalized sentence was translated to English.
    Translated(Vec<String>),
    /// A decoded video frame for the current playhead.
    Frame {
        /// Media-timeline position of the frame in milliseconds.
        pts_ms: u64,
        /// The frame pixels.
        frame: RgbaFrame,
    },
    /// A full-file batch transcription finished; captions now come from the
    /// pre-built timeline (no whisper latency).
    PreTranscribed,
    /// Playback reached the end of the media.
    PlaybackEnded,
}

/// Open media metadata kept for frame decoding.
#[derive(Debug)]
pub struct VideoInfo {
    pub(crate) metadata: VideoMetadata,
}

/// UI-side handle over the orchestrator; cheap to clone for callback wiring.
#[derive(Clone)]
pub struct Bridge {
    inner: std::rc::Rc<BridgeInner>,
}

struct BridgeInner {
    commands: UnboundedSender<Command>,
    _runtime: tokio::runtime::Runtime,
}

impl Bridge {
    /// Spawns the orchestrator on a fresh Tokio runtime.
    pub fn new(events: Sender<PlayerEvent>) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("building the player runtime cannot fail");
        let (commands, receiver) = mpsc::unbounded_channel();
        runtime.spawn(orchestrator(receiver, events));
        Self {
            inner: std::rc::Rc::new(BridgeInner {
                commands,
                _runtime: runtime,
            }),
        }
    }

    /// Probes `path`; the result arrives as [`Opened`](PlayerEvent::Opened).
    pub fn open_video(&self, path: PathBuf) {
        self.send(Command::Open(path));
    }

    /// Starts (or resumes) playback at the current playhead.
    pub fn play_at(&self, offset_ms: u64) {
        self.send(Command::PlayAt(offset_ms));
    }

    /// Pauses playback (decoders stop; the playhead stays).
    pub fn pause(&self) {
        self.send(Command::Pause);
    }

    /// Stops playback and its decoders.
    pub fn stop(&self) {
        self.send(Command::Stop);
    }

    /// Seeks while playing; restarts the decoders at `offset_ms`.
    pub fn seek(&self, offset_ms: u64) {
        self.send(Command::Seek(offset_ms));
    }

    /// Restarts playback with a new model/language (decoders + captioner).
    pub fn restart(&self, offset_ms: u64) {
        self.send(Command::Restart(offset_ms));
    }

    /// Playback volume in percent (0..=100).
    pub fn set_volume(&self, percent: u8) {
        self.send(Command::Volume(percent));
    }

    /// Mute override.
    pub fn set_mute(&self, muted: bool) {
        self.send(Command::Mute(muted));
    }

    /// Switches the whisper model; applied on the next (re)start.
    pub fn set_model(&self, model: String) {
        self.send(Command::Model(model));
    }

    /// Pins the caption language (`None` = auto-detect); applied on restart.
    pub fn set_language(&self, language: Option<String>) {
        self.send(Command::Language(language));
    }

    /// Turns live translation to English on or off.
    pub fn set_translate(&self, enabled: bool) {
        self.send(Command::Translate(enabled));
    }

    /// Translates one finalized sentence (English output).
    pub fn translate_sentence(&self, sentence: String) {
        self.send(Command::TranslateSentence(sentence));
    }

    /// Starts a full-file batch transcription: the whole audio track is fed
    /// to whisper at full speed; words (and their translations, when
    /// enabled) arrive as `Words`/`Translated`, completed by
    /// `PreTranscribed`. Playback then shows captions from the pre-built
    /// timeline with no whisper latency.
    pub fn pre_transcribe(&self) {
        self.send(Command::PreTranscribe);
    }

    /// Advances the video decoder to `timestamp_ms`, emitting the frame that
    /// is due there; the result arrives as a `Frame`.
    pub fn request_frame(&self, timestamp_ms: u64) {
        self.send(Command::Frame(timestamp_ms));
    }

    fn send(&self, command: Command) {
        let _ = self.inner.commands.send(command);
    }
}

// NOTE: dropping the last `Bridge` clone drops `BridgeInner`, whose command
// sender close ends the orchestrator; the orchestrator then drops the
// workers (each joins its thread) and the runtime-bound tasks.

/// Commands from the UI thread (see the [`Bridge`] wrappers).
enum Command {
    Open(PathBuf),
    PlayAt(u64),
    Pause,
    Stop,
    Seek(u64),
    Restart(u64),
    Volume(u8),
    Mute(bool),
    Model(String),
    Language(Option<String>),
    Translate(bool),
    TranslateSentence(String),
    /// Full-file batch transcription (see [`Bridge::pre_transcribe`]).
    PreTranscribe,
    Frame(u64),
}

/// Player handles: the audio driver and the video driver.
#[derive(Default)]
struct Drivers {
    audio_tx: Option<UnboundedSender<AudioCommand>>,
    audio_events: Option<UnboundedReceiver<AudioEvent>>,
    video_tx: Option<UnboundedSender<VideoCommand>>,
    video_events: Option<UnboundedReceiver<VideoEvent>>,
}

impl Drivers {
    /// Stops the audio driver (its `Stop` handling exits the select loop).
    fn stop_audio(&mut self) {
        if let Some(tx) = self.audio_tx.take() {
            let _ = tx.send(AudioCommand::Stop);
        }
    }

    /// Stops the video driver (its `Stop` handling exits the select loop).
    fn stop_video(&mut self) {
        if let Some(tx) = self.video_tx.take() {
            let _ = tx.send(VideoCommand::Stop);
        }
    }
}

/// The whisper caption pipeline for the current model/language, plus the
/// values it was built with (a mismatch means a rebuild on next play).
struct CaptionPipeline {
    captioner: Option<Arc<LiveCaptioner>>,
    events: Option<UnboundedReceiver<LiveEvent>>,
    model: String,
    model_used: Option<String>,
    language: Option<String>,
    language_used: Option<Option<String>>,
}

impl CaptionPipeline {
    fn new() -> Self {
        Self {
            captioner: None,
            events: None,
            model: "tiny".to_owned(),
            model_used: None,
            language: None,
            language_used: None,
        }
    }

    fn stale(&self) -> bool {
        self.captioner.is_none()
            || self.model_used.as_deref() != Some(self.model.as_str())
            || self.language_used != Some(self.language.clone())
    }
}

struct TranslateState {
    enabled: bool,
    detected: Option<String>,
    pinned: Option<String>,
    worker: TranslateWorker,
    worker_events: UnboundedReceiver<TranslateEvent>,
    manager: TranslationModelManager,
    /// Pair prefixes whose bundles are verified on disk; re-ensuring them
    /// (and re-hashing dozens of MB) is avoided on every toggle.
    ensured: HashSet<String>,
}

impl TranslateState {
    fn new(models_root: PathBuf) -> Self {
        let mut worker = TranslateWorker::new(models_root.clone());
        let worker_events = worker.take_events();
        let manager = TranslationModelManager::new(models_root)
            .expect("constructing the translation model manager cannot fail");
        Self {
            enabled: false,
            detected: None,
            pinned: None,
            worker,
            worker_events,
            manager,
            ensured: HashSet::new(),
        }
    }

    /// The bundled pair for the current source language: pinned selection
    /// wins, otherwise whisper's detection; the pairs cover zh-CN/ja/ko/
    /// es/da/ru (whisper's `zh` maps to `zh-CN`).
    fn resolves_pair(&self) -> Option<LocalPair> {
        if !self.enabled {
            return None;
        }
        self.pinned
            .as_deref()
            .or(self.detected.as_deref())
            .map(|code| if code == "zh" { "zh-CN" } else { code })
            .and_then(find_local_pair)
            .copied()
    }

    /// Downloads the pair bundle when needed, then swaps the worker backend;
    /// failures surface in the status line and leave the previous backend
    /// unloaded.
    async fn apply_pair(&mut self, events: &Sender<PlayerEvent>) {
        let Some(pair) = self.resolves_pair() else {
            self.worker.set_pair(None);
            return;
        };
        if !self.ensured.contains(pair.model_prefix) {
            let _ = events.send(PlayerEvent::Status(format!(
                "downloading the {} → English model…",
                pair.display_name
            )));
            match self.manager.ensure_pair(&pair).await {
                Ok(_) => {
                    self.ensured.insert(pair.model_prefix.to_owned());
                }
                Err(error) => {
                    let _ = events.send(PlayerEvent::Status(format!(
                        "{} → English model unavailable: {error}",
                        pair.display_name
                    )));
                    self.worker.set_pair(None);
                    return;
                }
            }
        }
        self.worker.set_pair(Some(pair));
    }
}

async fn orchestrator(mut commands: UnboundedReceiver<Command>, events: Sender<PlayerEvent>) {
    let paths = SublayerPaths::resolve().expect("XDG base directories must be available");
    let model_manager =
        ModelManager::new(paths.clone()).expect("building the model manager failed");
    let translate_root = paths.data_dir().join("translate");

    let mut video: Option<(PathBuf, VideoInfo)> = None;
    let mut pipeline = CaptionPipeline::new();
    let mut translate = TranslateState::new(translate_root);
    let mut drivers = Drivers::default();
    // Batch pre-transcription state: the captioner is busy with the whole
    // file while `prebuilding`; once `prebuilt_ready`, playback skips the
    // live ASR entirely and shows the pre-built timeline. `prebuilt_model`
    // records the model the words came from (from cache or a fresh batch) so
    // a model switch invalidates them.
    let mut prebuilding = false;
    let mut prebuilt_ready = false;
    let mut prebuilt_model: Option<String> = None;
    let transcript_dir = paths.data_dir().join("transcripts");

    loop {
        tokio::select! {
            command = commands.recv() => {
                let Some(command) = command else { break };
                match command {
                    Command::Open(path) => {
                        // A new file replaces the current one: playback of
                        // the old video ends now, not when the probe lands.
                        drivers.stop_audio();
                        drivers.stop_video();
                        prebuilding = false;
                        prebuilt_ready = false;
                        prebuilt_model = None;
                        open(&mut video, &path, &events).await;
                        // Restore a cached transcript for this video if one
                        // exists for the current model/language.
                        if let Some((path, _)) = video.as_ref()
                            && let Some((words, cards)) = transcript_cache::load(
                                &transcript_dir,
                                path,
                                &pipeline.model,
                                pipeline.language.as_deref(),
                            )
                        {
                            prebuilt_ready = true;
                            prebuilt_model = Some(pipeline.model.clone());
                            let _ = events.send(PlayerEvent::PrebuiltLoaded { words, cards });
                        }
                    }
                    Command::PlayAt(offset_ms) => {
                        if video.is_none() {
                            let _ = events.send(PlayerEvent::Status("open a video first".to_owned()));
                            continue;
                        }
                        if pipeline.stale() {
                            let _ = events.send(PlayerEvent::Status(
                                "preparing the speech model…".to_owned(),
                            ));
                        }
                        let use_prebuilt = prebuilt_ready
                            && prebuilt_model.as_deref() == Some(pipeline.model.as_str());
                        let started = ensure_and_start(
                            &mut pipeline,
                            &model_manager,
                            &mut drivers,
                            video.as_ref().map(|(path, info)| (path.clone(), info.metadata.clone())),
                            offset_ms,
                            &events,
                            use_prebuilt,
                        )
                        .await;
                        if started {
                            let _ = events.send(PlayerEvent::Status("playing".to_owned()));
                        }
                    }
                    Command::PreTranscribe => {
                        if prebuilding {
                            let _ = events.send(PlayerEvent::Status(
                                "already pre-transcribing…".to_owned(),
                            ));
                            continue;
                        }
                        let Some((path, info)) = video.as_ref() else {
                            let _ = events.send(PlayerEvent::Status(
                                "open a video first".to_owned(),
                            ));
                            continue;
                        };
                        if !ensure_captioner(&mut pipeline, &model_manager, &events).await {
                            continue;
                        }
                        let Some(captioner) = pipeline.captioner.clone() else {
                            continue;
                        };
                        match captioner.reset(0) {
                            Ok(()) => {}
                            Err(error) => {
                                let _ = events.send(PlayerEvent::Status(format!(
                                    "whisper: {error}"
                                )));
                                continue;
                            }
                        }
                        let mut stream = match sublayer_media::stream_audio_16k(path).await {
                            Ok(stream) => stream,
                            Err(error) => {
                                let _ = events.send(PlayerEvent::Status(format!(
                                    "pre-transcribe failed: {error}"
                                )));
                                continue;
                            }
                        };
                        prebuilding = true;
                        prebuilt_ready = false;
                        // Feed the whole track at full speed, in window-sized
                        // pieces so the transcriber's buffer stays bounded;
                        // surface extraction progress in the status line.
                        let total_ms = (info.metadata.duration_seconds * 1_000.0) as u64;
                        let mut chunk = vec![0.0_f32; 64_000];
                        let mut pushed_samples = 0_u64;
                        let mut last_pct = 0_u8;
                        let mut stalled = false;
                        loop {
                            match stream.read_samples(&mut chunk).await {
                                Ok(0) => break,
                                Ok(n) => {
                                    pushed_samples += n as u64;
                                    if captioner
                                        .push(chunk[..n].to_vec())
                                        .is_err()
                                    {
                                        stalled = true;
                                        break;
                                    }
                                    if let Some(pct) = ((pushed_samples * 1_000 / 16_000) * 100)
                                        .checked_div(total_ms)
                                    {
                                        let pct = pct.min(99) as u8;
                                        if pct != last_pct {
                                            last_pct = pct;
                                            let _ = events.send(PlayerEvent::Status(format!(
                                                "pre-transcribing… {pct}%"
                                            )));
                                        }
                                    }
                                }
                                Err(error) => {
                                    let _ = events.send(PlayerEvent::Status(format!(
                                        "pre-transcribe failed: {error}"
                                    )));
                                    stalled = true;
                                    break;
                                }
                            }
                        }
                        if !stalled {
                            // Extraction is done; whisper still decodes the
                            // windows (its progress shows as word timestamps
                            // pass the playhead). The worker's `Flushed`
                            // marks the batch complete; playback then
                            // switches to the pre-built timeline.
                            let _ =
                                events.send(PlayerEvent::Status("pre-transcribing… decoding".to_owned()));
                            let _ = captioner.flush();
                        }
                    }
                    Command::Pause | Command::Stop => {
                        drivers.stop_audio();
                        drivers.stop_video();
                        let _ = events.send(PlayerEvent::Status("paused".to_owned()));
                    }
                    Command::Seek(offset_ms) => {
                        if let Some(tx) = &drivers.audio_tx {
                            // The driver restarts its decoders and resets the
                            // captioner timeline to the new offset.
                            let _ = tx.send(AudioCommand::Start { offset_ms });
                        }
                        if let Some(tx) = &drivers.video_tx {
                            let _ = tx.send(VideoCommand::Start { offset_ms });
                        }
                    }
                    Command::Restart(offset_ms) => {
                        drivers.stop_audio();
                        drivers.stop_video();
                        ensure_and_start(
                            &mut pipeline,
                            &model_manager,
                            &mut drivers,
                            video.as_ref().map(|(path, info)| (path.clone(), info.metadata.clone())),
                            offset_ms,
                            &events,
                            prebuilt_ready,
                        )
                        .await;
                    }
                    Command::Volume(percent) => {
                        if let Some(tx) = &drivers.audio_tx {
                            let _ = tx.send(AudioCommand::SetVolume(percent));
                        }
                    }
                    Command::Mute(muted) => {
                        if let Some(tx) = &drivers.audio_tx {
                            let _ = tx.send(AudioCommand::SetMute(muted));
                        }
                    }
                    Command::Model(model) => {
                        pipeline.model = model;
                        // Words from another model are stale.
                        prebuilt_ready = false;
                        prebuilt_model = None;
                    }
                    Command::Language(language) => {
                        pipeline.language = language.clone();
                        translate.pinned = language;
                        translate.apply_pair(&events).await;
                        // Words from another language pin are stale.
                        prebuilt_ready = false;
                        prebuilt_model = None;
                    }
                    Command::Translate(enabled) => {
                        translate.enabled = enabled;
                        translate.apply_pair(&events).await;
                    }
                    Command::TranslateSentence(sentence) => {
                        translate.worker.translate(vec![sentence]);
                    }
                    Command::Frame(timestamp_ms) => {
                        if let Some(tx) = &drivers.video_tx {
                            let _ = tx.send(VideoCommand::Advance {
                                playhead_ms: timestamp_ms,
                            });
                        }
                    }
                }
            }
            event = next_captioner_event(&mut pipeline.events) => {
                if let Some(event) = event {
                    match event {
                        LiveEvent::Words(words) => {
                            let _ = events.send(PlayerEvent::Words(words));
                        }
                        LiveEvent::LanguageDetected(language) => {
                            translate.detected = Some(language.clone());
                            translate.apply_pair(&events).await;
                            let _ = events.send(PlayerEvent::LanguageDetected(language));
                        }
                        LiveEvent::Error(error) => {
                            let _ = events.send(PlayerEvent::Status(format!("whisper: {error}")));
                        }
                        LiveEvent::Flushed => {
                            if prebuilding {
                                // The batch's last words were delivered before
                                // this marker; switch playback to the pre-built
                                // timeline and remember the model it came from.
                                prebuilding = false;
                                prebuilt_ready = true;
                                prebuilt_model = Some(pipeline.model.clone());
                                let _ = events.send(PlayerEvent::PreTranscribed);
                            }
                        }
                    }
                }
            }
            event = next_video_event(&mut drivers.video_events) => {
                if let Some(event) = event {
                    match event {
                        VideoEvent::Frame(decoded) => {
                            let _ = events.send(PlayerEvent::Frame {
                                pts_ms: decoded.timestamp_ms,
                                frame: decoded.frame,
                            });
                        }
                        VideoEvent::Log(message) => {
                            let _ = events.send(PlayerEvent::Status(message));
                        }
                    }
                }
            }
            event = next_audio_event(&mut drivers.audio_events) => {
                if let Some(event) = event {
                    match event {
                        AudioEvent::Ended => {
                            drivers.stop_audio();
                            let _ = events.send(PlayerEvent::PlaybackEnded);
                        }
                        AudioEvent::Log(message) => {
                            let _ = events.send(PlayerEvent::Status(message));
                        }
                    }
                }
            }
            event = translate.worker_events.recv() => {
                if let Some(event) = event {
                    match event {
                        TranslateEvent::Translated(out) => {
                            let _ = events.send(PlayerEvent::Translated(out));
                        }
                        TranslateEvent::Error(error) => {
                            let _ = events.send(PlayerEvent::Status(format!("translation: {error}")));
                        }
                    }
                }
            }
        }
    }
}

/// Select arm for the captioner event stream; parks when unset.
///
/// A closed channel (`None`) parks the arm for good — the sender was
/// dropped, so no event can ever arrive — instead of spinning the select.
async fn next_captioner_event(
    receiver: &mut Option<UnboundedReceiver<LiveEvent>>,
) -> Option<LiveEvent> {
    match receiver {
        Some(rx) => {
            let event = rx.recv().await;
            if event.is_none() {
                *receiver = None;
            }
            event
        }
        None => std::future::pending().await,
    }
}

/// Select arm for the video driver event stream; parks when unset or closed.
async fn next_video_event(
    receiver: &mut Option<UnboundedReceiver<VideoEvent>>,
) -> Option<VideoEvent> {
    match receiver {
        Some(rx) => {
            let event = rx.recv().await;
            if event.is_none() {
                *receiver = None;
            }
            event
        }
        None => std::future::pending().await,
    }
}

/// Select arm for the audio driver event stream; parks when unset or closed.
async fn next_audio_event(
    receiver: &mut Option<UnboundedReceiver<AudioEvent>>,
) -> Option<AudioEvent> {
    match receiver {
        Some(rx) => {
            let event = rx.recv().await;
            if event.is_none() {
                *receiver = None;
            }
            event
        }
        None => std::future::pending().await,
    }
}

/// Ensures the caption pipeline matches the pinned model/language, then
/// starts (or restarts) the audio and video drivers at `offset_ms`. `video`
/// is `None` after a failed open; with a pre-built transcript the audio
/// driver runs without the live ASR (captions come from the batch). Returns
/// whether playback started.
async fn ensure_and_start(
    pipeline: &mut CaptionPipeline,
    model_manager: &ModelManager,
    drivers: &mut Drivers,
    video: Option<(PathBuf, VideoMetadata)>,
    offset_ms: u64,
    events: &Sender<PlayerEvent>,
    prebuilt_ready: bool,
) -> bool {
    let Some((video, metadata)) = video else {
        let _ = events.send(PlayerEvent::Status("open a video first".to_owned()));
        return false;
    };
    if !ensure_captioner(pipeline, model_manager, events).await {
        return false;
    }
    let captioner = if prebuilt_ready {
        None
    } else {
        pipeline.captioner.clone()
    };
    let (events_tx, events_rx) = mpsc::unbounded_channel();
    drivers.audio_events = Some(events_rx);
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    tokio::spawn(run_audio_driver(
        video.clone(),
        captioner,
        command_rx,
        events_tx,
    ));
    let _ = command_tx.send(AudioCommand::Start { offset_ms });
    drivers.audio_tx = Some(command_tx);

    let (events_tx, events_rx) = mpsc::unbounded_channel();
    drivers.video_events = Some(events_rx);
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    tokio::spawn(run_video_driver(video, metadata, command_rx, events_tx));
    let _ = command_tx.send(VideoCommand::Start { offset_ms });
    drivers.video_tx = Some(command_tx);
    true
}

/// Rebuilds the captioner when the model/language changed; returns whether
/// the pipeline is ready to transcribe.
async fn ensure_captioner(
    pipeline: &mut CaptionPipeline,
    model_manager: &ModelManager,
    events: &Sender<PlayerEvent>,
) -> bool {
    if !pipeline.stale() {
        return true;
    }
    let _ = events.send(PlayerEvent::Status(
        "preparing the speech model…".to_owned(),
    ));
    match build_captioner(&pipeline.model, pipeline.language.as_deref(), model_manager).await {
        Ok(mut captioner) => {
            let events_rx = captioner.take_events();
            pipeline.captioner = Some(Arc::new(captioner));
            pipeline.events = Some(events_rx);
            pipeline.model_used = Some(pipeline.model.clone());
            pipeline.language_used = Some(pipeline.language.clone());
            true
        }
        Err(error) => {
            let _ = events.send(PlayerEvent::Status(error));
            false
        }
    }
}

async fn open(video: &mut Option<(PathBuf, VideoInfo)>, path: &Path, events: &Sender<PlayerEvent>) {
    match probe_video(path).await {
        Ok(metadata) => {
            let title = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string());
            let duration_ms = (metadata.duration_seconds * 1_000.0) as u64;
            *video = Some((path.to_path_buf(), VideoInfo { metadata }));
            let _ = events.send(PlayerEvent::Opened {
                title,
                duration_ms,
                path: path.to_path_buf(),
            });
        }
        Err(error) => {
            let _ = events.send(PlayerEvent::Status(format!("cannot open video: {error}")));
        }
    }
}

/// Ensures the whisper model is cached, then builds the caption pipeline.
async fn build_captioner(
    model: &str,
    language: Option<&str>,
    model_manager: &ModelManager,
) -> Result<LiveCaptioner, String> {
    let model_path = model_manager
        .ensure_cached(model)
        .await
        .map_err(|error| format!("model `{model}`: {error}"))?;
    let config = LiveTranscriberConfig {
        model_path,
        language: language.map(str::to_owned),
        enable_vad: true,
        use_gpu: false,
        threads: std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(4),
        window_ms: ASR_WINDOW_MS,
        min_flush_ms: 200,
    };
    LiveCaptioner::new(config).map_err(|error| format!("whisper: {error}"))
}
