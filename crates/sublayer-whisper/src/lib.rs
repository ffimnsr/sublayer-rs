//! Leak-free safe bindings to whisper.cpp.
//!
//! This crate wraps the raw [`whisper-rs-sys`](https://crates.io/crates/whisper-rs-sys)
//! FFI with an ownership model that whisper-rs's safe layer lacks:
//!
//! - **Callbacks are borrowed, not boxed.** Progress and abort callbacks are
//!   `&mut dyn FnMut` borrows that live exactly as long as the [`FullParams`]
//!   they are attached to. Nothing is leaked, and a caller's `mpsc` senders
//!   held by a callback are dropped as soon as transcription returns, so
//!   channel-close semantics behave normally.
//! - **Pointers never escape.** [`WhisperContext`] and [`WhisperState`] are the
//!   only owners of raw pointers; they are freed in `Drop` and never exposed
//!   through the public API.
//! - **Bounds-checked iteration.** Segments and tokens are validated before
//!   crossing into C; out-of-range access returns `None` instead of relying on
//!   C-side behavior.
//!
//! # Safety audit
//!
//! All `unsafe` blocks in this crate are one of:
//!
//! 1. FFI calls whose arguments are validated non-null / non-empty first;
//! 2. `CStr` reads of pointers whisper.cpp guarantees to be NUL-terminated for
//!    live segments and tokens;
//! 3. the two callback trampolines, which reborrow `user_data` pointers
//!    installed by the `FullParams` builder. whisper.cpp only invokes them
//!    from the thread calling [`WhisperState::full`], while the originating
//!    `FullParams` is still alive, so the raw pointers are backed by the
//!    borrow checker for their entire reachable lifetime.

pub mod context;
pub mod error;
pub mod params;
pub mod state;

pub use context::{ContextParams, WhisperContext};
pub use error::WhisperError;
pub use params::{FullParams, SamplingStrategy};
pub use state::{Segment, Segments, Token, Tokens, WhisperState};
