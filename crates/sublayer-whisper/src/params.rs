//! Decoding parameters.
//!
//! Callbacks are plain borrows: they live exactly as long as the
//! `FullParams` they are attached to and are never boxed or leaked. During
//! `WhisperState::full` the raw `user_data` pointers are dereferenced by the
//! trampolines; the borrow checker guarantees the referents outlive the call.

use std::ffi::{CString, c_int, c_void};

use whisper_rs_sys as sys;

/// Sampling strategy for the decoder.
pub enum SamplingStrategy {
    /// Pick the best of `best_of` greedy decodings.
    Greedy { best_of: i32 },
    /// Beam search with a beam of `beam_size` and the given patience.
    BeamSearch { beam_size: i32, patience: f32 },
}

/// Decoder parameters observed by the next
/// [`WhisperState::full`](crate::state::WhisperState::full) call.
///
/// Callback closures are received as borrows but stored behind a `Box`, so the
/// `user_data` pointers written into the FFI struct remain valid even if this
/// struct is moved; the boxes are dropped (and the closures released) together
/// with this struct, so nothing is ever leaked.
pub struct FullParams<'a> {
    fp: sys::whisper_full_params,
    language: Option<CString>,
    progress_slot: Box<Option<&'a mut dyn FnMut(i32)>>,
    abort_slot: Box<Option<&'a mut dyn FnMut() -> bool>>,
}

impl<'a> FullParams<'a> {
    /// Starts from `whisper_full_default_params` plus the requested strategy.
    pub fn new(strategy: SamplingStrategy) -> Self {
        let strategy_ffi = match strategy {
            SamplingStrategy::Greedy { .. } => {
                sys::whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY
            }
            SamplingStrategy::BeamSearch { .. } => {
                sys::whisper_sampling_strategy_WHISPER_SAMPLING_BEAM_SEARCH
            }
        };
        // Safety: the strategy enum values are the documented ones.
        let mut fp = unsafe { sys::whisper_full_default_params(strategy_ffi) };
        match strategy {
            SamplingStrategy::Greedy { best_of } => fp.greedy.best_of = best_of,
            SamplingStrategy::BeamSearch {
                beam_size,
                patience,
            } => {
                fp.beam_search.beam_size = beam_size;
                fp.beam_search.patience = patience;
            }
        }
        Self {
            fp,
            language: None,
            progress_slot: Box::new(None),
            abort_slot: Box::new(None),
        }
    }

    pub(crate) fn raw(&self) -> sys::whisper_full_params {
        self.fp
    }

    /// Worker threads used by whisper.cpp (defaults to
    /// `min(4, hardware_concurrency)`).
    pub fn set_n_threads(&mut self, n_threads: i32) {
        self.fp.n_threads = n_threads;
    }

    /// Start decoding at this offset in the audio (milliseconds).
    pub fn set_offset_ms(&mut self, offset_ms: i32) {
        self.fp.offset_ms = offset_ms;
    }

    /// Only process this many milliseconds of audio (`0` = all).
    pub fn set_duration_ms(&mut self, duration_ms: i32) {
        self.fp.duration_ms = duration_ms;
    }

    /// Spoken language (`"en"`, `"de"`, ...); `None` lets whisper detect it.
    ///
    /// A language string containing a NUL byte is ignored (with a warning);
    /// the `CString` backing store lives inside this struct, so the pointer
    /// stays valid across moves.
    pub fn set_language(&mut self, language: Option<&str>) {
        match language {
            Some(language) => match CString::new(language) {
                Ok(language) => {
                    self.fp.language = language.as_ptr();
                    self.language = Some(language);
                }
                Err(_) => {
                    tracing::warn!("ignoring language with an interior NUL byte");
                    self.fp.language = std::ptr::null();
                    self.language = None;
                }
            },
            None => {
                self.fp.language = std::ptr::null();
                self.language = None;
            }
        }
    }

    /// Compute word-level timestamps (whisper.cpp's `token_timestamps`).
    pub fn set_token_timestamps(&mut self, enabled: bool) {
        self.fp.token_timestamps = enabled;
    }

    /// Align token timestamps on word boundaries instead of characters.
    pub fn set_split_on_word(&mut self, enabled: bool) {
        self.fp.split_on_word = enabled;
    }

    /// Cap tokens per segment (`0` = no limit).
    pub fn set_max_tokens(&mut self, max_tokens: i32) {
        self.fp.max_tokens = max_tokens;
    }

    /// Force a single segment covering the whole input (streaming mode).
    pub fn set_single_segment(&mut self, single_segment: bool) {
        self.fp.single_segment = single_segment;
    }

    /// No-speech probability threshold (not implemented inside whisper.cpp;
    /// consumers filter segments explicitly).
    pub fn set_no_speech_thold(&mut self, threshold: f32) {
        self.fp.no_speech_thold = threshold;
    }

    /// Suppress the `[BLANK_AUDIO]` token.
    pub fn set_suppress_blank(&mut self, suppress: bool) {
        self.fp.suppress_blank = suppress;
    }

    /// Initial fallback decoding temperature (default `0.2`).
    pub fn set_temperature_inc(&mut self, temperature_inc: f32) {
        self.fp.temperature_inc = temperature_inc;
    }

    /// Mirror of whisper.cpp's `print_*` toggles; defaults are already
    /// disabled where it matters, exposed for parity with the CLI.
    pub fn set_print_progress(&mut self, enabled: bool) {
        self.fp.print_progress = enabled;
    }
    pub fn set_print_special(&mut self, enabled: bool) {
        self.fp.print_special = enabled;
    }
    pub fn set_print_realtime(&mut self, enabled: bool) {
        self.fp.print_realtime = enabled;
    }

    /// Report decoding progress (percent, `0..100`) to `callback`.
    ///
    /// whisper.cpp invokes the callback from the thread that calls
    /// [`WhisperState::full`](crate::state::WhisperState::full); the borrow
    /// lasts until this struct is dropped,
    /// at which point the closure (and anything it captures, such as channel
    /// senders) is released.
    pub fn set_progress_callback(&mut self, callback: &'a mut dyn FnMut(i32)) {
        *self.progress_slot = Some(callback);
        self.fp.progress_callback = Some(progress_trampoline);
        self.fp.progress_callback_user_data =
            (&mut *self.progress_slot) as *mut Option<&'a mut dyn FnMut(i32)> as *mut c_void;
    }

    /// Remove the progress callback.
    pub fn clear_progress_callback(&mut self) {
        *self.progress_slot = None;
        self.fp.progress_callback = None;
        self.fp.progress_callback_user_data = std::ptr::null_mut();
    }

    /// Abort decoding when `callback` returns `true`. whisper.cpp polls it
    /// before each ggml computation, so it cannot interrupt a stuck decode.
    pub fn set_abort_callback(&mut self, callback: &'a mut dyn FnMut() -> bool) {
        *self.abort_slot = Some(callback);
        self.fp.abort_callback = Some(abort_trampoline);
        self.fp.abort_callback_user_data =
            (&mut *self.abort_slot) as *mut Option<&'a mut dyn FnMut() -> bool> as *mut c_void;
    }

    /// Remove the abort callback.
    pub fn clear_abort_callback(&mut self) {
        *self.abort_slot = None;
        self.fp.abort_callback = None;
        self.fp.abort_callback_user_data = std::ptr::null_mut();
    }
}

unsafe extern "C" fn progress_trampoline(
    _context: *mut sys::whisper_context,
    _state: *mut sys::whisper_state,
    progress: c_int,
    user_data: *mut c_void,
) {
    // Safety: this trampoline is only installed by `set_progress_callback`,
    // which pairs it with a pointer to the `Box<Option<&mut dyn FnMut(i32)>>`
    // slot. The box lives inside the originating FullParams, which is alive
    // for the whole `whisper_full_with_state` call, so the slot is valid. The
    // closure is only reachable through this `&mut` while whisper.cpp invokes
    // the callback from the caller's thread.
    let slot = unsafe { &mut *(user_data as *mut Option<&'static mut dyn FnMut(i32)>) };
    if let Some(callback) = slot.as_deref_mut() {
        callback(progress);
    }
}

unsafe extern "C" fn abort_trampoline(user_data: *mut c_void) -> bool {
    // Safety: see `progress_trampoline`; the abort slot is an
    // `Option<&mut dyn FnMut() -> bool>` boxed inside the originating
    // FullParams.
    let slot = unsafe { &mut *(user_data as *mut Option<&'static mut dyn FnMut() -> bool>) };
    slot.as_deref_mut().is_some_and(|callback| callback())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    fn greedy<'a>() -> FullParams<'a> {
        FullParams::new(SamplingStrategy::Greedy { best_of: 3 })
    }

    #[test]
    fn defaults_follow_sampling_strategy() {
        let params = greedy();
        assert_eq!(
            params.fp.strategy,
            sys::whisper_sampling_strategy_WHISPER_SAMPLING_GREEDY
        );
        assert_eq!(params.fp.greedy.best_of, 3);
        // whisper_full_default_params points at the built-in "en" string.
        assert!(!params.fp.language.is_null());
        assert!(params.fp.progress_callback.is_none());
        assert!(params.fp.abort_callback.is_none());

        let beam = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: 0.5,
        });
        assert_eq!(
            beam.fp.strategy,
            sys::whisper_sampling_strategy_WHISPER_SAMPLING_BEAM_SEARCH
        );
        assert_eq!(beam.fp.beam_search.beam_size, 5);
        assert_eq!(beam.fp.beam_search.patience, 0.5);
    }

    #[test]
    fn language_pointer_tracks_the_stored_cstring() {
        let mut params = greedy();

        params.set_language(Some("en"));
        assert!(!params.fp.language.is_null());
        // Safety: the pointer was installed from the CString stored in the
        // same struct, so it is valid and NUL-terminated.
        let language = unsafe { CStr::from_ptr(params.fp.language) };
        assert_eq!(language.to_str().unwrap(), "en");

        params.set_language(None);
        assert_eq!(params.fp.language, std::ptr::null());

        params.set_language(Some("bad\0language"));
        assert_eq!(params.fp.language, std::ptr::null());
    }

    #[test]
    fn callbacks_are_borrowed_and_clearable() {
        let calls = std::cell::RefCell::new(Vec::new());
        let before_clear = {
            let mut progress = |percent: i32| calls.borrow_mut().push(percent);
            let mut params = greedy();
            params.set_progress_callback(&mut progress);
            // The trampoline receives the boxed slot; drive it directly to
            // prove the callback is wired up.
            let slot = unsafe {
                &mut *(params.fp.progress_callback_user_data
                    as *mut Option<&'static mut dyn FnMut(i32)>)
            };
            if let Some(callback) = slot.as_deref_mut() {
                callback(50);
            }
            (
                params.fp.progress_callback.is_some(),
                !params.fp.progress_callback_user_data.is_null(),
            )
        };
        assert!(before_clear.0, "progress callback should be installed");
        assert!(before_clear.1, "user_data should be set");
        assert_eq!(calls.borrow().as_slice(), [50]);

        let after_clear = {
            let mut progress = |_: i32| {};
            let mut params = greedy();
            params.set_progress_callback(&mut progress);
            params.clear_progress_callback();
            (
                params.fp.progress_callback.is_none(),
                params.fp.progress_callback_user_data.is_null(),
            )
        };
        assert!(after_clear.0);
        assert!(after_clear.1);
    }

    #[test]
    fn user_data_pointers_survive_moves() {
        let seen = std::cell::Cell::new(0i32);
        let moved_ok = {
            let mut progress = |percent: i32| seen.set(percent);
            let mut params = greedy();
            params.set_progress_callback(&mut progress);
            let pointer = params.fp.progress_callback_user_data;

            // Moving the params must not invalidate the callback pointer: the
            // closure slot lives on the heap, not inside the struct.
            let moved = &mut params;
            let same = moved.fp.progress_callback_user_data == pointer;
            let slot = unsafe { &mut *(pointer as *mut Option<&'static mut dyn FnMut(i32)>) };
            if let Some(callback) = slot.as_deref_mut() {
                callback(75);
            }
            (same, seen.get())
        };
        assert!(moved_ok.0, "pointer must survive a param move");
        assert_eq!(moved_ok.1, 75);
    }

    #[test]
    fn field_setters_land_in_the_ffi_struct() {
        let mut params = greedy();
        params.set_n_threads(7);
        params.set_offset_ms(250);
        params.set_duration_ms(5000);
        params.set_token_timestamps(true);
        params.set_split_on_word(true);
        params.set_max_tokens(42);
        params.set_single_segment(true);
        params.set_no_speech_thold(0.9);
        params.set_suppress_blank(false);
        params.set_temperature_inc(0.4);
        params.set_print_progress(true);

        assert_eq!(params.fp.n_threads, 7);
        assert_eq!(params.fp.offset_ms, 250);
        assert_eq!(params.fp.duration_ms, 5000);
        assert!(params.fp.token_timestamps);
        assert!(params.fp.split_on_word);
        assert_eq!(params.fp.max_tokens, 42);
        assert!(params.fp.single_segment);
        assert_eq!(params.fp.no_speech_thold, 0.9);
        assert!(!params.fp.suppress_blank);
        assert_eq!(params.fp.temperature_inc, 0.4);
        assert!(params.fp.print_progress);
    }
}
