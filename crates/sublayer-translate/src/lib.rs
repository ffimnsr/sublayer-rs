//! Sentence-level machine translation for live captions.
//!
//! Backends translate *finalized sentences* (caption cards), never partial
//! words, so latency and quality stay predictable. The [`TranslationBackend`]
//! trait is synchronous because both backends are blocking by nature:
//!
//! - [`local::LocalBackend`] (feature `local-translate`) runs the pure-Rust
//!   Firefox Translations / Bergamot inference engine (`fxtranslate`) offline
//!   over the bundled source → English pairs in [`LOCAL_PAIRS`] (see
//!   [`pairs`]); it is thread-affine (its engine is `!Send`), so construct
//!   and use it on one worker thread.
//! - [`remote::RemoteBackend`] (feature `remote-translate`, default) talks to
//!   any OpenAI-compatible chat-completions endpoint over blocking HTTP.
//!
//! Callers run backends on worker threads; [`translate_async`] wraps that for
//! Tokio contexts when the backend is `Send + Sync`.

#[cfg(feature = "local-translate")]
pub mod local;
#[cfg(feature = "local-translate")]
pub mod pairs;
#[cfg(feature = "remote-translate")]
pub mod remote;
#[cfg(all(feature = "download", feature = "local-translate"))]
pub mod translation_models;

#[cfg(feature = "local-translate")]
pub use pairs::{LOCAL_PAIRS, LocalPair, find_local_pair};
#[cfg(all(feature = "download", feature = "local-translate"))]
pub use translation_models::{PinnedBundle, PinnedFile, TranslationModelManager, pinned_bundle};

/// A sentence-level translation backend.
pub trait TranslationBackend {
    /// Stable identifier for UI labels and logs (e.g. `"bergamot-fxtranslate"`).
    fn name(&self) -> &str;

    /// Language of the input sentences, or `None` when the backend accepts
    /// any language (remote LLM backends, the identity pass-through).
    fn source_language(&self) -> Option<&str>;

    /// Language translations are produced in, or `None` when the backend
    /// leaves the input unchanged (see [`IdentityBackend`]).
    fn target_language(&self) -> Option<&str>;

    /// Translates each sentence, returning one target-language string per
    /// input, in order. Blocking I/O is expected; run backends on worker
    /// threads in interactive contexts.
    fn translate(&self, sentences: &[String]) -> Result<Vec<String>, TranslateError>;
}

/// Pass-through backend for when translation is disabled: captions are shown
/// in the source language unchanged.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityBackend;

impl TranslationBackend for IdentityBackend {
    fn name(&self) -> &str {
        "identity"
    }

    fn source_language(&self) -> Option<&str> {
        None
    }

    fn target_language(&self) -> Option<&str> {
        None
    }

    fn translate(&self, sentences: &[String]) -> Result<Vec<String>, TranslateError> {
        Ok(sentences.to_vec())
    }
}

/// Runs a `Send + Sync` backend on Tokio's blocking pool.
///
/// `LocalBackend` does not qualify (its engine is thread-affine); keep it on
/// a dedicated worker thread instead.
pub async fn translate_async<B>(
    backend: B,
    sentences: Vec<String>,
) -> Result<Vec<String>, TranslateError>
where
    B: TranslationBackend + Send + Sync + 'static,
{
    tokio::task::spawn_blocking(move || backend.translate(&sentences))
        .await
        .map_err(|error| TranslateError::TaskJoin(error.to_string()))?
}

/// Errors raised by translation backends.
#[derive(Debug, thiserror::Error)]
pub enum TranslateError {
    /// The backend could not be constructed (missing model files, engine
    /// load failure, invalid configuration).
    #[error("failed to load translation backend: {0}")]
    Load(String),

    /// The remote endpoint could not be reached or the request failed at the
    /// transport level.
    #[cfg(any(feature = "remote-translate", feature = "download"))]
    #[error("translation request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// The remote endpoint answered with a non-success status.
    #[cfg(any(feature = "remote-translate", feature = "download"))]
    #[error("translation endpoint returned HTTP {status}: {detail}")]
    HttpStatus {
        /// HTTP status code received.
        status: u16,
        /// Bounded excerpt of the response body.
        detail: String,
    },

    /// Filesystem interaction failed.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// The backend answered with something that is not a usable translation.
    #[error("translation backend returned an unexpected response: {0}")]
    InvalidResponse(String),

    /// The blocking translation worker panicked.
    #[error("background translation task failed: {0}")]
    TaskJoin(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_backend_passes_sentences_through() {
        let sentences = vec!["hello".to_owned(), "world".to_owned()];
        assert_eq!(IdentityBackend.translate(&sentences).unwrap(), sentences);
        assert_eq!(IdentityBackend.name(), "identity");
        assert_eq!(IdentityBackend.source_language(), None);
        assert_eq!(IdentityBackend.target_language(), None);
    }

    #[tokio::test]
    async fn translate_async_runs_send_backends_on_the_blocking_pool() {
        let sentences = vec!["a".to_owned(), "b".to_owned()];
        let out = translate_async(IdentityBackend, sentences.clone())
            .await
            .unwrap();
        assert_eq!(out, sentences);
    }
}
