//! Offline NMT backend built on `fxtranslate`, the pure-Rust port of the
//! Firefox Translations (Bergamot / opus-mt) inference engine.
//!
//! The bundled direction is **source language → English** (see
//! [`crate::LOCAL_PAIRS`]).
//!
//! # Model bundle
//!
//! A language pair is a directory holding one model, vocabulary file(s), and
//! one lexical shortlist. Firefox ships two layouts, both detected by this
//! backend:
//!
//! ```text
//! shared vocabulary (most pairs):
//!   model.koen.intgemm.alphas.bin   marian int8 weights (opus-mt)
//!   lex.50.50.koen.s2t.bin          lexical shortlist for decoding
//!   vocab.koen.spm                  SentencePiece vocabulary (src = trg)
//!
//! split vocabularies (zh-CN → en):
//!   model.zhen.intgemm.alphas.bin
//!   lex.50.50.zhen.s2t.bin
//!   srcvocab.zhen.spm               source (Mandarin) vocabulary
//!   trgvocab.zhen.spm               target (English) vocabulary
//! ```
//!
//! # Thread affinity
//!
//! The engine memory-maps its weights (SIMD fast path), which makes it
//! `!Send`: construct the backend and call [`translate`] on one thread only.
//! A live-caption player keeps it on a dedicated translation worker instead
//! of moving it across threads.

use std::path::{Path, PathBuf};

use fxtranslate::{engine::Engine, shortlist::Shortlist};

use crate::{TranslateError, TranslationBackend};

/// Offline source → English translator over one local Firefox Translations
/// bundle.
///
/// Model loading happens in [`new`](Self::new); translation batches whole
/// sentences through the engine (`translate_batch`), which encodes and
/// decodes them together.
pub struct LocalBackend {
    engine: Engine,
    source_language: String,
}

impl LocalBackend {
    /// Loads the engine from `model_dir` (one file per pattern, see the
    /// module docs). `source_language` labels the input (`"ko"`, `"zh-CN"`);
    /// the output is always English. Fails with the name of the first missing
    /// or ambiguous file.
    pub fn new(
        model_dir: &Path,
        source_language: impl Into<String>,
    ) -> Result<Self, TranslateError> {
        let source_language = source_language.into();
        let model_path = find_unique(model_dir, ".intgemm.alphas.bin", "model")?;
        let shortlist_path = find_unique(model_dir, ".s2t.bin", "lexical shortlist")?;
        let (src_vocab, trg_vocab) = find_vocabs(model_dir)?;
        let engine = Engine::load_mmapped(&model_path, &src_vocab, &trg_vocab)
            .map_err(TranslateError::Load)?
            .with_shortlist(
                Shortlist::load(&shortlist_path)
                    .map_err(|error| TranslateError::Load(format!("{error:?}")))?,
            );
        Ok(Self {
            engine,
            source_language,
        })
    }
}

impl TranslationBackend for LocalBackend {
    fn name(&self) -> &str {
        "bergamot-fxtranslate"
    }

    fn source_language(&self) -> Option<&str> {
        Some(&self.source_language)
    }

    fn target_language(&self) -> Option<&str> {
        Some("en")
    }

    fn translate(&self, sentences: &[String]) -> Result<Vec<String>, TranslateError> {
        if sentences.is_empty() {
            return Ok(Vec::new());
        }
        let refs: Vec<&str> = sentences.iter().map(String::as_str).collect();
        Ok(self.engine.translate_batch(&refs))
    }
}

// The engine exposes no `Debug`; keep it out of the derived output.
impl std::fmt::Debug for LocalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalBackend")
            .field("name", &self.name())
            .field("source_language", &self.source_language)
            .field("target_language", &self.target_language())
            .finish_non_exhaustive()
    }
}

/// Finds the unique file in `dir` whose name ends with `pattern`.
fn find_unique(dir: &Path, pattern: &str, what: &str) -> Result<PathBuf, TranslateError> {
    let mut matches: Vec<PathBuf> = read_dir_files(dir, |name| name.ends_with(pattern));
    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => Err(TranslateError::Load(format!(
            "{what} file (*{pattern}) missing in {}",
            dir.display()
        ))),
        _ => Err(TranslateError::Load(format!(
            "{what} is ambiguous in {} ({} files match *{pattern})",
            dir.display(),
            matches.len()
        ))),
    }
}

/// Resolves the vocabulary files: one shared `.spm` or a split
/// `srcvocab.*.spm` + `trgvocab.*.spm` pair (the layout Firefox uses for
/// zh-CN → en).
fn find_vocabs(dir: &Path) -> Result<(PathBuf, PathBuf), TranslateError> {
    let mut spm = read_dir_files(dir, |name| name.ends_with(".spm"));
    spm.sort();
    match spm.len() {
        1 => {
            let shared = spm.remove(0);
            Ok((shared.clone(), shared))
        }
        2 => {
            let named = |needle: &str| {
                spm.iter().find(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.contains(needle))
                })
            };
            match (named("srcvocab"), named("trgvocab")) {
                (Some(src), Some(trg)) => Ok((src.clone(), trg.clone())),
                _ => Err(TranslateError::Load(format!(
                    "vocabulary files are ambiguous in {} ({}); expected one \
                     shared vocab or a srcvocab.*.spm + trgvocab.*.spm pair",
                    dir.display(),
                    spm.iter()
                        .filter_map(|path| path.file_name())
                        .filter_map(|name| name.to_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))),
            }
        }
        0 => Err(TranslateError::Load(format!(
            "vocabulary file (*.spm) missing in {}",
            dir.display()
        ))),
        count => Err(TranslateError::Load(format!(
            "vocabulary files are ambiguous in {} ({count} files match *.spm)",
            dir.display()
        ))),
    }
}

/// Names of the regular files in `dir` ending with `pattern`, in arbitrary
/// order; unreadable entries are skipped.
fn read_dir_files(dir: &Path, matches: impl Fn(&str) -> bool) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    path.is_file()
                        && path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .is_some_and(&matches)
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_files_are_reported_by_name() {
        let directory = tempfile::tempdir().unwrap();
        let error = LocalBackend::new(directory.path(), "ko")
            .expect_err("an empty directory must not load");

        let message = error.to_string();
        assert!(message.contains(".intgemm.alphas.bin"), "got: {message}");
    }

    #[test]
    fn missing_shortlist_is_reported_after_a_valid_model() {
        let directory = tempfile::tempdir().unwrap();
        // A zero-byte model parses as garbage, but the shortlist check runs
        // first; the error must name the shortlist, not the model content.
        std::fs::write(directory.path().join("model.koen.intgemm.alphas.bin"), []).unwrap();
        std::fs::write(directory.path().join("vocab.koen.spm"), []).unwrap();

        let error =
            LocalBackend::new(directory.path(), "ko").expect_err("the shortlist must be demanded");
        let message = error.to_string();
        assert!(message.contains(".s2t.bin"), "got: {message}");
    }

    #[test]
    fn a_shared_vocabulary_is_used_for_both_sides() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("vocab.koen.spm"), []).unwrap();
        let (src, trg) = find_vocabs(directory.path()).unwrap();
        assert_eq!(src, trg);
    }

    #[test]
    fn split_vocabularies_are_ordered_source_then_target() {
        let directory = tempfile::tempdir().unwrap();
        let src = directory.path().join("srcvocab.zhen.spm");
        let trg = directory.path().join("trgvocab.zhen.spm");
        std::fs::write(&src, []).unwrap();
        std::fs::write(&trg, []).unwrap();

        let (found_src, found_trg) = find_vocabs(directory.path()).unwrap();
        assert_eq!(found_src, src);
        assert_eq!(found_trg, trg);
    }

    #[test]
    fn unsupported_vocab_layouts_are_reported_ambiguously() {
        // Two generically named vocab files: not a recognised split layout.
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("vocab.a.spm"), []).unwrap();
        std::fs::write(directory.path().join("vocab.b.spm"), []).unwrap();
        let error = find_vocabs(directory.path()).expect_err("must be ambiguous");
        assert!(error.to_string().contains("ambiguous"), "got: {error}");

        // Two split-looking files that do not form a src/trg pair.
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("srcvocab.a.spm"), []).unwrap();
        std::fs::write(directory.path().join("srcvocab.b.spm"), []).unwrap();
        let error = find_vocabs(directory.path()).expect_err("must be ambiguous");
        assert!(error.to_string().contains("ambiguous"), "got: {error}");

        // Three vocab files are ambiguous even with a src/trg pair inside.
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("srcvocab.zhen.spm"), []).unwrap();
        std::fs::write(directory.path().join("trgvocab.zhen.spm"), []).unwrap();
        std::fs::write(directory.path().join("vocab.koen.spm"), []).unwrap();
        let error = find_vocabs(directory.path()).expect_err("must be ambiguous");
        assert!(error.to_string().contains("ambiguous"), "got: {error}");
    }

    #[test]
    fn missing_vocabularies_are_reported() {
        let directory = tempfile::tempdir().unwrap();
        let error = find_vocabs(directory.path()).expect_err("must fail");
        assert!(error.to_string().contains(".spm"), "got: {error}");
    }

    #[test]
    fn unreadable_directories_surface_as_missing_files() {
        let error = find_vocabs(Path::new("/nonexistent/sublayer-test")).expect_err("must fail");
        assert!(error.to_string().contains("missing"), "got: {error}");
    }

    /// End-to-end through the engine with a real Firefox Translations bundle
    /// (any directory with one `*.intgemm.alphas.bin`, one `*.s2t.bin`, and
    /// one shared `vocab.*.spm` or a `srcvocab.*.spm`/`trgvocab.*.spm` pair).
    ///
    /// ```text
    /// SUBLAYER_TEST_TRANSLATE_DIR=/path/to/koen-bundle \
    /// SUBLAYER_TEST_TRANSLATE_LANG=ko \
    /// SUBLAYER_TEST_TRANSLATE_TEXT='오늘 날씨가 아주 좋아요.' \
    ///   cargo test -p sublayer-translate --features local-translate -- \
    ///   --ignored translates_x_to_english_end_to_end
    /// ```
    #[ignore]
    #[test]
    fn translates_x_to_english_end_to_end() {
        let dir = std::env::var("SUBLAYER_TEST_TRANSLATE_DIR")
            .unwrap_or_else(|_| "/tmp/fx-koen".to_owned());
        let language =
            std::env::var("SUBLAYER_TEST_TRANSLATE_LANG").unwrap_or_else(|_| "ko".to_owned());
        let text = std::env::var("SUBLAYER_TEST_TRANSLATE_TEXT")
            .unwrap_or_else(|_| "오늘 날씨가 아주 좋아요.".to_owned());
        let has_model = std::fs::read_dir(&dir).is_ok_and(|entries| {
            entries.filter_map(|entry| entry.ok()).any(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.ends_with(".intgemm.alphas.bin"))
            })
        });
        if !has_model {
            eprintln!("skipping: set SUBLAYER_TEST_TRANSLATE_DIR to a model bundle");
            return;
        }

        let backend = LocalBackend::new(std::path::Path::new(&dir), language.clone()).unwrap();
        assert_eq!(backend.source_language().unwrap(), language);
        assert_eq!(backend.target_language().unwrap(), "en");

        let out = backend.translate(std::slice::from_ref(&text)).unwrap();
        eprintln!("translated ({language} → en): {:?}", out);
        assert_eq!(out.len(), 1);
        let translation = &out[0];
        assert!(!translation.trim().is_empty(), "must produce a translation");
        // English output must not leak the source script.
        assert!(
            !translation.chars().any(is_cjk),
            "translation contains source-script characters: {translation}"
        );
        assert!(translation.chars().any(|c| c.is_ascii_alphabetic()));
    }

    /// Whether `c` falls in the CJK Unified Ideographs / Hiragana / Katakana /
    /// Hangul Syllables blocks.
    fn is_cjk(c: char) -> bool {
        matches!(c as u32,
            0x4E00..=0x9FFF      // CJK Unified Ideographs
            | 0x3040..=0x30FF    // Hiragana + Katakana
            | 0xAC00..=0xD7AF    // Hangul Syllables
        )
    }
}
