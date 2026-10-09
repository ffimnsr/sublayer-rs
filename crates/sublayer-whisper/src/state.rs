//! Transcription state and result iteration.

use std::ffi::{CStr, c_char, c_int};

use whisper_rs_sys as sys;

use crate::context::WhisperContext;
use crate::error::WhisperError;
use crate::params::FullParams;

/// Mutable inference state bound to a [`WhisperContext`]; freed in [`Drop`].
///
/// Not `Send` or `Sync`: create, use, and drop it on one thread. Its lifetime
/// is tied to the context it was created from.
pub struct WhisperState<'ctx> {
    context: &'ctx WhisperContext,
    state: *mut sys::whisper_state,
}

impl<'ctx> WhisperState<'ctx> {
    pub(crate) fn new(context: &'ctx WhisperContext) -> Result<Self, WhisperError> {
        // Safety: `context.raw()` is a live context pointer owned by
        // `context`; the result is null-checked and freed in Drop.
        let state = unsafe { sys::whisper_init_state(context.raw()) };
        if state.is_null() {
            return Err(WhisperError::StateInitFailed);
        }
        Ok(Self { context, state })
    }

    /// Runs the full transcription pipeline over `pcm` (16 kHz mono `f32`),
    /// invoking any callbacks installed on `params` from the calling thread.
    pub fn full(&mut self, params: &mut FullParams<'_>, pcm: &[f32]) -> Result<(), WhisperError> {
        if pcm.is_empty() {
            // whisper.cpp can segfault on empty buffers; refuse up front.
            return Err(WhisperError::NoSamples);
        }
        if pcm.len() > c_int::MAX as usize {
            // The length crosses into C as an `int`; a wrap would make
            // whisper.cpp read out of bounds later.
            return Err(WhisperError::SamplesTooLarge);
        }
        // Safety: `context`/`state` are live owning pointers, `params.raw()`
        // is a fully initialized parameter struct whose callback pointers are
        // backed by the borrow checker, and `pcm` is a valid slice whose
        // length fits in a c_int for realistic inputs.
        let ret = unsafe {
            sys::whisper_full_with_state(
                self.context.raw(),
                self.state,
                params.raw(),
                pcm.as_ptr(),
                pcm.len() as c_int,
            )
        };
        match ret {
            0 => Ok(()),
            -1 => Err(WhisperError::UnableToCalculateSpectrogram),
            7 => Err(WhisperError::FailedToEncode),
            8 => Err(WhisperError::FailedToDecode),
            other => Err(WhisperError::Generic(other)),
        }
    }

    /// Number of decoded text segments.
    pub fn n_segments(&self) -> i32 {
        // Safety: `state` is a live owning pointer.
        unsafe { sys::whisper_full_n_segments_from_state(self.state) }
    }

    /// Iterates every decoded segment.
    pub fn segments(&self) -> Segments<'_, 'ctx> {
        Segments {
            state: self,
            next: 0,
            count: self.n_segments(),
        }
    }

    pub(crate) fn raw(&self) -> *mut sys::whisper_state {
        self.state
    }

    pub(crate) fn context_raw(&self) -> *mut sys::whisper_context {
        self.context.raw()
    }
}

impl Drop for WhisperState<'_> {
    fn drop(&mut self) {
        // Safety: `self.state` is either null or an owning pointer from
        // whisper_init_state; Drop runs exactly once per state.
        unsafe { sys::whisper_free_state(self.state) };
    }
}

/// One decoded segment.
#[derive(Clone, Copy)]
pub struct Segment<'a, 'ctx> {
    state: &'a WhisperState<'ctx>,
    index: i32,
}

impl<'a, 'ctx> Segment<'a, 'ctx> {
    /// Segment index in the state.
    pub fn index(&self) -> i32 {
        self.index
    }

    /// Segment text; invalid UTF-8 is replaced with the replacement character.
    pub fn text_lossy(&self) -> String {
        self.c_string(sys::whisper_full_get_segment_text_from_state)
            .map(|text| text.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// Start time in milliseconds; whisper.cpp reports 10 ms units.
    pub fn start_ms(&self) -> i64 {
        // Safety: `state` is live and `index` is in bounds (iteration is
        // bounds-checked or derived from `n_segments`).
        let centiseconds =
            unsafe { sys::whisper_full_get_segment_t0_from_state(self.state.raw(), self.index) };
        centiseconds * 10
    }

    /// End time in milliseconds; whisper.cpp reports 10 ms units.
    pub fn end_ms(&self) -> i64 {
        // Safety: see `start_ms`.
        let centiseconds =
            unsafe { sys::whisper_full_get_segment_t1_from_state(self.state.raw(), self.index) };
        centiseconds * 10
    }

    /// Number of tokens in this segment.
    pub fn n_tokens(&self) -> i32 {
        // Safety: `state` is live and `index` is in bounds.
        unsafe { sys::whisper_full_n_tokens_from_state(self.state.raw(), self.index) }
    }

    /// Probability that this segment is not speech; silence and music windows
    /// score high and their text is usually hallucinated.
    pub fn no_speech_probability(&self) -> f32 {
        // Safety: `state` is live and `index` is in bounds.
        unsafe {
            sys::whisper_full_get_segment_no_speech_prob_from_state(self.state.raw(), self.index)
        }
    }

    /// Iterates every token of this segment.
    pub fn tokens(&self) -> Tokens<'a, 'ctx> {
        Tokens {
            segment: *self,
            next: 0,
            count: self.n_tokens(),
        }
    }

    /// Token at `index`, bounds-checked.
    pub fn token(&self, index: i32) -> Option<Token<'a, 'ctx>> {
        (index >= 0 && index < self.n_tokens()).then_some(Token {
            segment: *self,
            index,
        })
    }

    fn c_string(
        &self,
        function: unsafe extern "C" fn(*mut sys::whisper_state, c_int) -> *const c_char,
    ) -> Option<&CStr> {
        // Safety: `function` is a whisper.cpp accessor returning a
        // NUL-terminated string for in-bounds segments, which is guaranteed by
        // the bounds-checked iterator API.
        let ptr = unsafe { function(self.state.raw(), self.index) };
        if ptr.is_null() {
            None
        } else {
            // Safety: the returned pointer is NUL-terminated (see above).
            Some(unsafe { CStr::from_ptr(ptr) })
        }
    }
}

/// Iterator over a state's segments.
pub struct Segments<'a, 'ctx> {
    state: &'a WhisperState<'ctx>,
    next: i32,
    count: i32,
}

impl<'a, 'ctx> Iterator for Segments<'a, 'ctx> {
    type Item = Segment<'a, 'ctx>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.count {
            return None;
        }
        let segment = Segment {
            state: self.state,
            index: self.next,
        };
        self.next += 1;
        Some(segment)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.count - self.next).max(0) as usize;
        (remaining, Some(remaining))
    }
}

/// One token inside a segment.
#[derive(Clone, Copy)]
pub struct Token<'a, 'ctx> {
    segment: Segment<'a, 'ctx>,
    index: i32,
}

impl<'a, 'ctx> Token<'a, 'ctx> {
    /// Token index within its segment.
    pub fn index(&self) -> i32 {
        self.index
    }

    /// Token text; invalid UTF-8 is replaced with the replacement character.
    pub fn text_lossy(&self) -> String {
        // Safety: `state` is live, segment/token indices are in bounds, and
        // whisper.cpp returns a NUL-terminated string.
        let ptr = unsafe {
            sys::whisper_full_get_token_text_from_state(
                self.segment.state.context_raw(),
                self.segment.state.raw(),
                self.segment.index,
                self.index,
            )
        };
        if ptr.is_null() {
            return String::new();
        }
        // Safety: NUL-terminated (see above).
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    /// Start time in milliseconds.
    ///
    /// When the context was created with DTW alignment enabled, this prefers
    /// the DTW time (`t_dtw`) and falls back to the attention-based `t0`;
    /// both are centiseconds in whisper.cpp, so either way the value is
    /// scaled to milliseconds here.
    pub fn start_ms(&self) -> i64 {
        let data = self.data();
        let centiseconds = if data.t_dtw > 0 { data.t_dtw } else { data.t0 };
        centiseconds * 10
    }

    /// End time in milliseconds; token data uses the same 10 ms units as
    /// segment timestamps.
    pub fn end_ms(&self) -> i64 {
        self.data().t1 * 10
    }

    /// Probability of this token.
    pub fn probability(&self) -> f32 {
        self.data().p
    }

    fn data(&self) -> sys::whisper_token_data {
        // Safety: `state` is live and both indices are in bounds.
        unsafe {
            sys::whisper_full_get_token_data_from_state(
                self.segment.state.raw(),
                self.segment.index,
                self.index,
            )
        }
    }
}

/// Iterator over a segment's tokens.
pub struct Tokens<'a, 'ctx> {
    segment: Segment<'a, 'ctx>,
    next: i32,
    count: i32,
}

impl<'a, 'ctx> Iterator for Tokens<'a, 'ctx> {
    type Item = Token<'a, 'ctx>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.count {
            return None;
        }
        let token = Token {
            segment: self.segment,
            index: self.next,
        };
        self.next += 1;
        Some(token)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.count - self.next).max(0) as usize;
        (remaining, Some(remaining))
    }
}
