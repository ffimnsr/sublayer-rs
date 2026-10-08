//! AI speech pipeline for Sublayer: Whisper GGML model management, an energy
//! threshold voice activity detector, and Whisper transcription with
//! word-level timestamps.
//!
//! The heavy inference step is executed on the blocking thread pool, so the
//! crate can be driven from Tokio workers without stalling them. GPU
//! acceleration is opt-in at build time through the `vulkan` feature; the
//! runtime falls back to CPU when the backend is not compiled in.

pub mod error;
pub mod models;
pub mod transcriber;
pub mod vad;

pub use error::AiError;
pub use models::{MODELS, ModelManager, ModelSpec};
pub use transcriber::{TranscriberConfig, WHISPER_SAMPLE_RATE, transcribe_audio};
pub use vad::{SpeechSegment, VadConfig, detect_speech};
