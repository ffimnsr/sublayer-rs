//! Model loading: context parameters and the context handle.

use std::ffi::CString;
use std::path::Path;

use whisper_rs_sys as sys;

use crate::error::WhisperError;
use crate::state::WhisperState;

/// `whisper_context_default_params()` plus explicit builder overrides.
#[derive(Debug, Clone)]
pub struct ContextParams {
    fp: sys::whisper_context_params,
}

impl Default for ContextParams {
    fn default() -> Self {
        // Safety: whisper_context_default_params initializes the struct to
        // documented defaults; no arguments to validate.
        let mut fp = unsafe { sys::whisper_context_default_params() };
        // whisper.cpp itself defaults to requesting a GPU backend; normalize to
        // CPU so acceleration is always an explicit opt-in (a missing backend
        // otherwise emits confusing GPU errors at load time).
        fp.use_gpu = false;
        Self { fp }
    }
}

impl ContextParams {
    /// Request the GPU backend. Builds of whisper.cpp without a GPU backend
    /// (no `vulkan` feature) fall back to CPU.
    pub fn use_gpu(&mut self, enabled: bool) -> &mut Self {
        self.fp.use_gpu = enabled;
        self
    }

    /// Select the GPU device when multiple are present (default `0`).
    pub fn gpu_device(&mut self, device: i32) -> &mut Self {
        self.fp.gpu_device = device;
        self
    }

    /// Enable flash attention (default off; incompatible with DTW).
    pub fn flash_attn(&mut self, enabled: bool) -> &mut Self {
        self.fp.flash_attn = enabled;
        self
    }
}

/// A loaded whisper model. Freed in [`Drop`]; not `Send` or `Sync`, so create,
/// use, and drop it on one thread.
#[derive(Debug)]
pub struct WhisperContext {
    ctx: *mut sys::whisper_context,
}

impl WhisperContext {
    /// Loads a GGML whisper model file (as produced by a `sublayer-ai`
    /// `ModelManager`) with the given parameters.
    pub fn new_with_params(
        path: impl AsRef<Path>,
        params: ContextParams,
    ) -> Result<Self, WhisperError> {
        let path = c_string_path(path.as_ref())?;
        // Safety: `path` is a valid NUL-terminated C string and `params` is a
        // fully initialized copy of `whisper_context_default_params`; the
        // returned pointer is null-checked.
        let ctx = unsafe { sys::whisper_init_from_file_with_params(path.as_ptr(), params.fp) };
        if ctx.is_null() {
            return Err(WhisperError::ContextLoadFailed);
        }
        Ok(Self { ctx })
    }

    /// Creates an independent inference state bound to this context.
    pub fn create_state(&self) -> Result<WhisperState<'_>, WhisperError> {
        WhisperState::new(self)
    }

    pub(crate) fn raw(&self) -> *mut sys::whisper_context {
        self.ctx
    }
}

impl Drop for WhisperContext {
    fn drop(&mut self) {
        // Safety: `self.ctx` is either null or an owning pointer from
        // whisper_init_from_file_with_params; `whisper_free` releases it.
        // Drop runs exactly once per context.
        unsafe { sys::whisper_free(self.ctx) };
    }
}

fn c_string_path(path: &Path) -> Result<CString, WhisperError> {
    let value = path
        .as_os_str()
        .to_str()
        .ok_or_else(|| WhisperError::InvalidPath(path.to_path_buf()))?;
    CString::new(value).map_err(|_| WhisperError::InvalidPath(path.to_path_buf()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(params: &ContextParams) -> &sys::whisper_context_params {
        &params.fp
    }

    #[test]
    fn context_params_defaults_and_overrides() {
        let defaults = ContextParams::default();
        // use_gpu is normalized to CPU; flash_attn keeps whisper.cpp 1.8's
        // upstream default (enabled).
        assert!(!fp(&defaults).use_gpu);
        assert!(fp(&defaults).flash_attn);
        assert_eq!(fp(&defaults).gpu_device, 0);

        let mut params = ContextParams::default();
        params.use_gpu(true).flash_attn(false).gpu_device(2);
        assert!(fp(&params).use_gpu);
        assert!(!fp(&params).flash_attn);
        assert_eq!(fp(&params).gpu_device, 2);
    }

    #[test]
    fn garbage_model_files_fail_to_load() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("not-a-model.bin");
        std::fs::write(&path, b"this is not a ggml model").unwrap();

        let error = WhisperContext::new_with_params(&path, ContextParams::default()).unwrap_err();
        assert!(
            matches!(error, WhisperError::ContextLoadFailed),
            "got {error:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn paths_with_nul_bytes_are_rejected() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let path = Path::new(OsStr::from_bytes(b"model\0.bin"));
        let error = c_string_path(path).unwrap_err();
        assert!(
            matches!(error, WhisperError::InvalidPath(_)),
            "got {error:?}"
        );
    }
}
