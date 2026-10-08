//! Process helpers shared by the FFmpeg and FFprobe wrappers.
//!
//! Media tools are located on `PATH` unless the corresponding `SUBLAYER_*`
//! environment variable points at an explicit executable, which keeps the
//! crates usable inside Flatpak sandboxes and custom installations. Children
//! are started with `kill_on_drop`, so dropping a caller's future (or hitting a
//! `tokio::time::timeout`) terminates the encoder instead of leaking it.

use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Output, Stdio};

use tokio::process::Command;

use crate::MediaError;

/// Binary name of the FFmpeg CLI.
pub(crate) const FFMPEG: &str = "ffmpeg";
/// Binary name of the FFprobe CLI.
pub(crate) const FFPROBE: &str = "ffprobe";
/// Environment variable overriding the auto-detected `ffmpeg` executable.
pub(crate) const FFMPEG_ENV: &str = "SUBLAYER_FFMPEG";
/// Environment variable overriding the auto-detected `ffprobe` executable.
pub(crate) const FFPROBE_ENV: &str = "SUBLAYER_FFPROBE";

/// Longest stderr excerpt kept inside [`MediaError::CommandFailed`].
const STDERR_EXCERPT_CHARS: usize = 2_000;

/// Locates `binary` on `PATH`, preferring a non-empty `env_key` override.
pub(crate) fn resolve(binary: &'static str, env_key: &str) -> Result<PathBuf, MediaError> {
    if let Some(value) = std::env::var_os(env_key).filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(value));
    }
    which::which(binary).map_err(|_| MediaError::BinaryNotFound(binary))
}

/// Builds a command that never reads stdin and dies with its future.
pub(crate) fn command(program: &Path) -> Command {
    let mut command = Command::new(program);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    command
}

/// Runs `command` to completion, buffering stdout and stderr.
pub(crate) async fn run(binary: &'static str, command: &mut Command) -> Result<Output, MediaError> {
    tracing::debug!(command = %describe(command), "running media tool");
    command
        .output()
        .await
        .map_err(|source| MediaError::Spawn { binary, source })
}

/// Turns a non-zero exit status into a [`MediaError::CommandFailed`] carrying a
/// bounded stderr excerpt.
pub(crate) fn failure(binary: &'static str, status: ExitStatus, stderr: &[u8]) -> MediaError {
    MediaError::CommandFailed {
        binary,
        status,
        stderr: excerpt(stderr),
    }
}

/// Renders the command as a single line for debug logs.
fn describe(command: &Command) -> String {
    let std_command = command.as_std();
    let mut text = std_command.get_program().to_string_lossy().into_owned();
    for argument in std_command.get_args() {
        text.push(' ');
        text.push_str(&argument.to_string_lossy());
    }
    text
}

/// Keeps the last [`STDERR_EXCERPT_CHARS`] characters of `stderr`.
fn excerpt(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let trimmed = text.trim();
    let characters = trimmed.chars().count();
    if characters <= STDERR_EXCERPT_CHARS {
        return trimmed.to_owned();
    }
    trimmed
        .chars()
        .skip(characters - STDERR_EXCERPT_CHARS)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn excerpt_keeps_short_output_intact() {
        assert_eq!(excerpt(b"  boom  "), "boom");
        assert_eq!(excerpt(b""), "");
    }

    #[test]
    fn excerpt_bounds_long_output_to_tail() {
        let long = "x".repeat(STDERR_EXCERPT_CHARS + 500);
        let excerpted = excerpt(long.as_bytes());
        assert_eq!(excerpted.chars().count(), STDERR_EXCERPT_CHARS);
        assert_eq!(excerpted, long[500..]);
    }
}
