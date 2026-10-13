//! On-disk cache for pre-built transcripts.
//!
//! A batch pre-transcription (words + timed cards + translations) is stored
//! under the XDG data directory, keyed by the whisper model, the pinned
//! language, and the video's path. The video's size and modification time are
//! recorded inside the file, so a replaced or re-encoded source invalidates
//! the entry automatically; opening the same video with the same model shows
//! captions instantly without re-running whisper.
//!
//! The format is versioned JSON; unknown fields are ignored on read, so
//! newer files degrade safely on older builds.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sublayer_core::WordToken;

use crate::session::PrebuiltCard;

/// Bumped whenever the stored layout changes incompatibly.
const CACHE_VERSION: u32 = 1;

/// One cached transcript file.
#[derive(Debug, Serialize, Deserialize)]
struct TranscriptFile {
    version: u32,
    /// Whisper model the words came from.
    model: String,
    /// Pinned language used for the run.
    language: Option<String>,
    /// Video the transcript belongs to.
    source: PathBuf,
    /// Byte size of the video when cached.
    size: u64,
    /// Modification time of the video when cached, in ms since the epoch.
    modified_ms: u64,
    /// Word timeline, in decode order.
    words: Vec<WordToken>,
    /// Finalized sentence cards with their translation slots.
    cards: Vec<PrebuiltCard>,
}

/// Looks up the cached transcript for `video` under `model`/`language`.
///
/// Returns `None` when there is no entry, the file moved on (size or
/// modification time changed), or the entry is unreadable or from another
/// model/language. The caller then runs a fresh batch instead.
pub(crate) fn load(
    dir: &Path,
    video: &Path,
    model: &str,
    language: Option<&str>,
) -> Option<(Vec<WordToken>, Vec<PrebuiltCard>)> {
    let current = metadata_fingerprint(video)?;
    let bytes = fs::read(cache_path(dir, video, model, language)).ok()?;
    let file: TranscriptFile = serde_json::from_slice(&bytes).ok()?;
    if file.version != CACHE_VERSION
        || file.model != model
        || file.language != language.map(str::to_owned)
        || file.source != video
        || file.size != current.0
        || file.modified_ms != current.1
    {
        return None;
    }
    Some((file.words, file.cards))
}

/// Stores (or replaces) the transcript for `video` under `model`/`language`.
pub(crate) fn save(
    dir: &Path,
    video: &Path,
    model: &str,
    language: Option<&str>,
    words: &[WordToken],
    cards: &[PrebuiltCard],
) -> Result<(), String> {
    let (size, modified_ms) = metadata_fingerprint(video).ok_or("no metadata for the video")?;
    let file = TranscriptFile {
        version: CACHE_VERSION,
        model: model.to_owned(),
        language: language.map(str::to_owned),
        source: video.to_path_buf(),
        size,
        modified_ms,
        words: words.to_vec(),
        cards: cards.to_vec(),
    };
    let bytes = serde_json::to_vec(&file).map_err(|error| error.to_string())?;
    fs::create_dir_all(dir).map_err(|error| error.to_string())?;
    fs::write(cache_path(dir, video, model, language), bytes).map_err(|error| error.to_string())
}

/// `<dir>/<sha256(model|language|path)>.json`.
fn cache_path(dir: &Path, video: &Path, model: &str, language: Option<&str>) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(model.as_bytes());
    hasher.update([0]);
    hasher.update(language.unwrap_or("").as_bytes());
    hasher.update([0]);
    hasher.update(video.as_os_str().as_encoded_bytes());
    let key = hasher.finalize();
    dir.join(format!("{}.json", hex(&key[..16])))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// `(size, modified_ms)` of `path`, or `None` when the file cannot be stat'ed.
fn metadata_fingerprint(path: &Path) -> Option<(u64, u64)> {
    let metadata = fs::metadata(path).ok()?;
    let modified_ms = metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_millis() as u64;
    Some((metadata.len(), modified_ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::PrebuiltCard;

    fn card(start_ms: u64, source: &str) -> PrebuiltCard {
        PrebuiltCard {
            start_ms,
            end_ms: start_ms + 500,
            source: source.to_owned(),
            translation: None,
        }
    }

    fn word(text: &str, start_ms: u64) -> WordToken {
        WordToken::new(text.to_owned(), start_ms, start_ms + 300)
    }

    fn video_file(dir: &Path) -> PathBuf {
        let path = dir.join("clip.mp4");
        fs::write(&path, b"fake video bytes").expect("write fixture");
        path
    }

    #[test]
    fn round_trips_the_transcript_payload() {
        let directory = tempfile::tempdir().unwrap();
        let video = video_file(directory.path());
        let words = vec![word("Hallo", 0), word("Welt.", 400)];
        let cards = vec![card(0, "Hallo Welt.")];

        save(
            &directory.path().join("cache"),
            &video,
            "tiny",
            Some("en"),
            &words,
            &cards,
        )
        .expect("save must succeed");

        let (loaded_words, loaded_cards) =
            load(&directory.path().join("cache"), &video, "tiny", Some("en"))
                .expect("load must hit the cache");
        assert_eq!(loaded_words, words);
        assert_eq!(loaded_cards, cards);
    }

    #[test]
    fn a_different_model_or_language_is_a_miss() {
        let directory = tempfile::tempdir().unwrap();
        let video = video_file(directory.path());
        save(
            &directory.path().join("cache"),
            &video,
            "tiny",
            Some("en"),
            &[word("Hallo", 0)],
            &[card(0, "Hallo")],
        )
        .expect("save must succeed");

        assert!(
            load(
                &directory.path().join("cache"),
                &video,
                "base.en",
                Some("en")
            )
            .is_none()
        );
        assert!(load(&directory.path().join("cache"), &video, "tiny", None).is_none());
    }

    #[test]
    fn a_replaced_video_invalidates_the_entry() {
        let directory = tempfile::tempdir().unwrap();
        let video = video_file(directory.path());
        save(
            &directory.path().join("cache"),
            &video,
            "tiny",
            None,
            &[word("Hallo", 0)],
            &[card(0, "Hallo")],
        )
        .expect("save must succeed");
        assert!(load(&directory.path().join("cache"), &video, "tiny", None).is_some());

        // A re-encoded file (different size) must not reuse the transcript.
        fs::write(&video, b"re-encoded video bytes, longer than before").unwrap();
        assert!(load(&directory.path().join("cache"), &video, "tiny", None).is_none());
    }
}
