//! Dedicated thread running the thread-affine local translation backend.
//!
//! [`sublayer_translate::LocalBackend`] owns a `!Send` engine, so it never
//! leaves its construction thread; this worker wraps that constraint in the
//! same command/event shape as the whisper `LiveCaptioner` (see
//! `sublayer-ai::live`).

use std::path::PathBuf;
use std::sync::mpsc;

use sublayer_translate::LocalPair;
use sublayer_translate::TranslationBackend;
use sublayer_translate::local::LocalBackend;
use tokio::sync::mpsc as tokio_mpsc;

/// Commands sent to the translation worker.
enum Command {
    /// Swaps the backend for a bundled pair (`None` unloads it).
    SetPair(Option<LocalPair>),
    /// Translates finalized sentences to English.
    Translate(Vec<String>),
    Shutdown,
}

/// Events emitted by the translation worker.
#[derive(Debug)]
pub enum TranslateEvent {
    /// One English string per input sentence, in order.
    Translated(Vec<String>),
    /// Backend load or translation failure; the pipeline keeps running.
    Error(String),
}

/// Async-friendly facade over a thread-affine [`LocalBackend`].
pub struct TranslateWorker {
    commands: mpsc::Sender<Command>,
    events: tokio_mpsc::UnboundedReceiver<TranslateEvent>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl TranslateWorker {
    /// Spawns the worker; `models_root` holds one bundle directory per
    /// bundled pair (`models_root/<prefix>/model.<prefix>.intgemm.alphas.bin`
    /// ...). No backend is loaded until [`set_pair`](Self::set_pair).
    pub fn new(models_root: PathBuf) -> Self {
        let (command_tx, command_rx) = mpsc::channel();
        let (event_tx, event_rx) = tokio_mpsc::unbounded_channel();
        let worker = std::thread::Builder::new()
            .name("sublayer-translate".to_owned())
            .spawn(move || run_worker(command_rx, event_tx, models_root))
            .expect("spawning the translate worker thread cannot fail");
        Self {
            commands: command_tx,
            events: event_rx,
            worker: Some(worker),
        }
    }

    /// Loads the backend for `pair` (or unloads it for `None`).
    pub fn set_pair(&self, pair: Option<LocalPair>) {
        let _ = self.commands.send(Command::SetPair(pair));
    }

    /// Translates sentences; the result arrives as a `Translated` event.
    pub fn translate(&self, sentences: Vec<String>) {
        let _ = self.commands.send(Command::Translate(sentences));
    }

    /// Hands out the event receiver; only one consumer may hold it.
    pub fn take_events(&mut self) -> tokio_mpsc::UnboundedReceiver<TranslateEvent> {
        std::mem::replace(&mut self.events, tokio_mpsc::unbounded_channel().1)
    }
}

impl Drop for TranslateWorker {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            let _ = self.commands.send(Command::Shutdown);
            let _ = worker.join();
        }
    }
}

fn run_worker(
    commands: mpsc::Receiver<Command>,
    events: tokio_mpsc::UnboundedSender<TranslateEvent>,
    models_root: PathBuf,
) {
    let mut backend: Option<LocalBackend> = None;
    while let Ok(command) = commands.recv() {
        match command {
            Command::SetPair(pair) => {
                backend = match pair {
                    None => None,
                    Some(pair) => {
                        let dir = models_root.join(pair.model_prefix);
                        match LocalBackend::new(&dir, pair.language) {
                            Ok(backend) => Some(backend),
                            Err(error) => {
                                tracing::warn!(%error, "translation model load failed");
                                let _ = events.send(TranslateEvent::Error(format!(
                                    "{}→en model: {error}",
                                    pair.display_name
                                )));
                                None
                            }
                        }
                    }
                };
            }
            Command::Translate(sentences) => match &backend {
                Some(backend) => match backend.translate(&sentences) {
                    Ok(out) => {
                        let _ = events.send(TranslateEvent::Translated(out));
                    }
                    Err(error) => {
                        let _ = events.send(TranslateEvent::Error(error.to_string()));
                    }
                },
                None => {
                    let _ = events.send(TranslateEvent::Error(
                        "no translation model loaded".to_owned(),
                    ));
                }
            },
            Command::Shutdown => break,
        }
    }
}
