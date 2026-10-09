//! Routing of whisper.cpp and ggml diagnostics into `tracing`.
//!
//! whisper.cpp logs model loading, backend selection, and inference details
//! straight to stderr by default, which drowns CLI progress output and the
//! studio's status line. Installing the handler forwards every line into
//! `tracing` under the `whisper` target: warnings and errors stay visible at
//! default levels, while the verbose load/inference chatter moves to `debug`
//! and `trace`, where `RUST_LOG=whisper=debug` can bring it back.

use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::sync::Once;

use whisper_rs_sys as sys;

/// whisper.cpp installs one handler per process; so do we.
static INSTALL: Once = Once::new();

/// Installs the process-wide log handler; safe to call repeatedly.
///
/// Call this before creating a context if you want the first messages routed
/// too; otherwise the first `WhisperContext` may log a line to stderr.
pub fn install_log_handler() {
    INSTALL.call_once(|| {
        // SAFETY: `whisper_log_set` only stores the callback pointer. The
        // callback never unwinds (it only calls `tracing`, which does not
        // panic on its own) and ignores `user_data`.
        unsafe {
            sys::whisper_log_set(Some(forward), std::ptr::null_mut());
        }
    });
}

/// Forwards one whisper.cpp log line to [`tracing`].
///
/// # Safety
///
/// `text` must point at a NUL-terminated string for the duration of the call;
/// whisper.cpp guarantees that for every log callback invocation.
unsafe extern "C" fn forward(
    level: sys::ggml_log_level,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    if text.is_null() {
        return;
    }
    // SAFETY: guaranteed by the callback contract documented above.
    let message = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    let message = message.trim_end_matches(['\n', '\r']);
    if message.is_empty() {
        return;
    }

    match level {
        sys::ggml_log_level_GGML_LOG_LEVEL_ERROR => tracing::error!(target: "whisper", "{message}"),
        sys::ggml_log_level_GGML_LOG_LEVEL_WARN => tracing::warn!(target: "whisper", "{message}"),
        // Model loading and inference details are noise for normal runs.
        sys::ggml_log_level_GGML_LOG_LEVEL_INFO
        | sys::ggml_log_level_GGML_LOG_LEVEL_CONT
        | sys::ggml_log_level_GGML_LOG_LEVEL_DEBUG => {
            tracing::debug!(target: "whisper", "{message}");
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installing_twice_is_harmless() {
        install_log_handler();
        install_log_handler();
    }

    #[test]
    fn null_text_is_ignored() {
        // SAFETY: the callback explicitly tolerates a null pointer.
        unsafe {
            forward(
                sys::ggml_log_level_GGML_LOG_LEVEL_ERROR,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
    }
}
