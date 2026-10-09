//! Errors raised by FFmpeg/FFprobe invocations, WAV decoding, and cache I/O.

use std::path::PathBuf;

/// Errors raised by [`sublayer-media`](crate) operations.
#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    /// A required external tool is not installed or not on `PATH`.
    #[error("required binary `{0}` was not found in PATH")]
    BinaryNotFound(&'static str),

    /// The external tool could not be started.
    #[error("failed to spawn `{binary}`: {source}")]
    Spawn {
        /// Binary that failed to start.
        binary: &'static str,
        /// Underlying spawn error.
        #[source]
        source: std::io::Error,
    },

    /// The external tool ran but exited with a non-zero status.
    #[error("`{binary}` failed ({status}): {stderr}")]
    CommandFailed {
        /// Binary that failed.
        binary: &'static str,
        /// Exit status reported by the process.
        status: std::process::ExitStatus,
        /// Bounded excerpt of the process' standard error output.
        stderr: String,
    },

    /// The input contains no video stream to caption.
    #[error("input `{0}` contains no video stream")]
    NoVideoStream(PathBuf),

    /// The input contains no audio stream to transcribe.
    #[error("input `{0}` contains no usable audio stream")]
    NoAudioStream(PathBuf),

    /// No frame could be decoded at the requested position (typically past the
    /// end of the stream).
    #[error("no video frame could be decoded at {timestamp_ms} ms")]
    NoVideoFrame {
        /// Requested position in milliseconds.
        timestamp_ms: u64,
    },

    /// The raw decoder produced an unexpected number of bytes.
    #[error("decoded frame data has {actual} bytes, expected {expected}")]
    InvalidFrameData {
        /// Byte count implied by the requested frame size.
        expected: usize,
        /// Byte count actually read from the decoder.
        actual: usize,
    },

    /// An operation exceeded its deadline; the child process was killed.
    #[error("{operation} timed out after {seconds}s")]
    Timeout {
        /// Human-readable operation name.
        operation: &'static str,
        /// Configured deadline in whole seconds.
        seconds: u64,
    },

    /// Extraction succeeded but produced a header-only WAV.
    #[error("audio extraction from `{input}` produced no samples")]
    EmptyAudioOutput {
        /// Input the (empty) extraction came from.
        input: PathBuf,
    },

    /// FFprobe printed output that does not match the expected schema.
    #[error("ffprobe returned malformed JSON: {0}")]
    InvalidProbeData(String),

    /// The WAV uses a bit depth or sample format this crate cannot decode.
    #[error(
        "unsupported WAV sample format {sample_format:?} with {bits_per_sample} bits per sample"
    )]
    UnsupportedSampleFormat {
        /// Sample format announced by the WAV header.
        sample_format: hound::SampleFormat,
        /// Bit depth announced by the WAV header.
        bits_per_sample: u16,
    },

    /// The WAV header is self-contradictory (for example zero channels).
    #[error("invalid WAV specification: {0}")]
    InvalidWavSpec(String),

    /// The WAV data chunk could not be decoded.
    #[error("WAV decoding failed: {0}")]
    Wav(#[from] hound::Error),

    /// A bucket rate of `0` cannot map samples to time.
    #[error("samples_per_sec must be greater than zero")]
    InvalidBucketRate,

    /// A `.waveform` cache file is malformed or truncated.
    #[error("invalid waveform cache: {0}")]
    InvalidCache(String),

    /// The cache file was written by a newer, unknown layout revision.
    #[error(
        "waveform cache version {found} is not supported (this build supports up to {supported})"
    )]
    UnsupportedCacheVersion {
        /// Layout version found in the file.
        found: u16,
        /// Highest layout version understood by this build.
        supported: u16,
    },

    /// The blocking waveform worker panicked.
    #[error("background waveform task failed: {0}")]
    TaskJoin(String),

    /// Filesystem interaction failed.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}
