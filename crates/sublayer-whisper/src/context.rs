//! Model loading: context parameters and the context handle.

use std::ffi::CString;
use std::path::Path;

use whisper_rs_sys as sys;

use crate::error::WhisperError;
use crate::state::WhisperState;

/// Alignment-head preset used for DTW token-level timestamps.
///
/// whisper.cpp 1.8 ships the OpenAI alignment heads for every model size as
/// compile-time presets, so DTW needs no extra weights: the preset selects
/// the cross-attention heads that map text tokens onto the audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentHeads {
    TinyEn,
    Tiny,
    BaseEn,
    Base,
    SmallEn,
    Small,
    MediumEn,
    Medium,
    LargeV1,
    LargeV2,
    LargeV3,
    LargeV3Turbo,
}

impl AlignmentHeads {
    /// Preset for a model file, inferred from its file name.
    ///
    /// Returns `None` for unrecognized names; DTW then stays off and the
    /// attention-based fallback (`t0`/`t1`) applies.
    pub fn for_model(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?.to_ascii_lowercase();
        if !name.contains("ggml") || !name.contains(".bin") {
            return None;
        }
        let english = name.contains(".en");
        let heads = if name.contains("turbo") {
            Self::LargeV3Turbo
        } else if name.contains("tiny") {
            if english { Self::TinyEn } else { Self::Tiny }
        } else if name.contains("small") {
            if english { Self::SmallEn } else { Self::Small }
        } else if name.contains("medium") {
            if english {
                Self::MediumEn
            } else {
                Self::Medium
            }
        } else if name.contains("base") {
            if english { Self::BaseEn } else { Self::Base }
        } else if name.contains("large") {
            if name.contains("v1") {
                Self::LargeV1
            } else if name.contains("v3") {
                Self::LargeV3
            } else {
                Self::LargeV2
            }
        } else {
            return None;
        };
        Some(heads)
    }

    fn as_sys(self) -> sys::whisper_alignment_heads_preset {
        match self {
            Self::TinyEn => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_TINY_EN,
            Self::Tiny => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_TINY,
            Self::BaseEn => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_BASE_EN,
            Self::Base => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_BASE,
            Self::SmallEn => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_SMALL_EN,
            Self::Small => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_SMALL,
            Self::MediumEn => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_MEDIUM_EN,
            Self::Medium => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_MEDIUM,
            Self::LargeV1 => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_LARGE_V1,
            Self::LargeV2 => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_LARGE_V2,
            Self::LargeV3 => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_LARGE_V3,
            Self::LargeV3Turbo => sys::whisper_alignment_heads_preset_WHISPER_AHEADS_LARGE_V3_TURBO,
        }
    }
}

/// `whisper_context_default_params()` plus explicit builder overrides.
#[derive(Debug, Clone)]
pub struct ContextParams {
    fp: sys::whisper_context_params,
    /// DTW alignment preset; `None` keeps the attention-based timestamps.
    dtw: Option<AlignmentHeads>,
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
        Self { fp, dtw: None }
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

    /// Enable flash attention (default off when DTW is in use; flash
    /// attention does not materialize the cross-attention scores DTW needs).
    pub fn flash_attn(&mut self, enabled: bool) -> &mut Self {
        self.fp.flash_attn = enabled;
        self
    }

    /// Align token timestamps with the DTW algorithm for `heads`.
    ///
    /// DTW maps the decoded text onto the audio with dynamic time warping,
    /// which stays accurate over long or music-heavy windows where the plain
    /// attention-based timestamps drift. Pass `None` to keep the whisper.cpp
    /// default alignment.
    pub fn dtw_timestamps(&mut self, heads: Option<AlignmentHeads>) -> &mut Self {
        self.dtw = heads;
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
        let fp = match params.dtw {
            Some(heads) => with_dtw(params.fp, heads),
            None => params.fp,
        };
        let ctx = unsafe { sys::whisper_init_from_file_with_params(path.as_ptr(), fp) };
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

/// Applies a DTW request to the raw parameters: enables DTW token
/// timestamps, selects the alignment heads, and switches flash attention off
/// (it does not materialize the cross-attention scores DTW reads).
fn with_dtw(
    mut fp: sys::whisper_context_params,
    heads: AlignmentHeads,
) -> sys::whisper_context_params {
    fp.dtw_token_timestamps = true;
    fp.dtw_aheads_preset = heads.as_sys();
    fp.flash_attn = false;
    fp
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
    fn dtw_requests_disable_flash_attention() {
        let params = ContextParams::default();
        let fp = with_dtw(params.fp, AlignmentHeads::BaseEn);

        assert!(fp.dtw_token_timestamps);
        assert_eq!(
            fp.dtw_aheads_preset,
            sys::whisper_alignment_heads_preset_WHISPER_AHEADS_BASE_EN
        );
        assert!(!fp.flash_attn, "DTW needs raw cross-attention scores");
    }

    #[test]
    fn alignment_presets_are_inferred_from_model_names() {
        for (name, expected) in [
            ("/models/ggml-tiny.en.bin", Some(AlignmentHeads::TinyEn)),
            ("/models/ggml-base.en.bin", Some(AlignmentHeads::BaseEn)),
            ("/models/ggml-small.en.bin", Some(AlignmentHeads::SmallEn)),
            ("/models/ggml-base.bin", Some(AlignmentHeads::Base)),
            (
                "/models/ggml-large-v3-turbo.bin",
                Some(AlignmentHeads::LargeV3Turbo),
            ),
            ("/models/ggml-large-v3.bin", Some(AlignmentHeads::LargeV3)),
            ("/models/ggml-large-v2.bin", Some(AlignmentHeads::LargeV2)),
            ("/models/ggml-medium.en.bin", Some(AlignmentHeads::MediumEn)),
            ("/models/my-custom.bin", None),
        ] {
            assert_eq!(
                AlignmentHeads::for_model(Path::new(name)),
                expected,
                "{name}"
            );
        }
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
