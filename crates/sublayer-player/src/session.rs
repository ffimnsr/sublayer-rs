//! Playback and caption state for the player window.
//!
//! Everything here is plain data mutated by the UI pump; heavy work (audio,
//! translation, decoding) lives in `bridge`. Captions always come from a
//! pre-transcribed batch: recognized words accumulate on the media timeline
//! and are segmented into sentence cards (on a long pause or sentence-ending
//! punctuation); the card covering the playhead is what gets shown and
//! translated. There is no live ASR — the user clicks Pre-transcribe.

use std::collections::VecDeque;

use sublayer_core::WordToken;
use sublayer_translate::{LocalPair, find_local_pair};

/// Language choices shown in the UI; `None` = let whisper detect.
/// Whisper reports Mandarin as `"zh"`, which maps to the bundled `zh-CN` pair.
pub const PLAYER_LANGUAGES: &[(&str, Option<&str>)] = &[
    ("Auto-detect", None),
    ("Korean", Some("ko")),
    ("Japanese", Some("ja")),
    ("Chinese (Simplified)", Some("zh-CN")),
    ("Spanish", Some("es")),
    ("Danish", Some("da")),
    ("Russian", Some("ru")),
    ("English", Some("en")),
];

/// Gap between words that ends the current live caption line.
const FINALIZE_GAP_MS: u64 = 900;

/// Rolling window for pre-built (batch-transcribed) captions: they have no
/// whisper latency, so a short linger keeps the line current.
const PREBUILT_LINGER_MS: u64 = 3_000;

/// How far past the playhead a pre-built word may start and still be shown.
const PREBUILT_LOOKAHEAD_MS: u64 = 300;

/// Sentence-ending punctuation; a word ending in one of these finalizes.
const SENTENCE_END: &[char] = &['.', '?', '!', '。', '！', '？', '…'];

/// Playback duration of the opened media.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MediaInfo {
    pub duration_ms: u64,
}

/// One finalized pre-built caption card, with the translation slot filled
/// in by the translation worker during the batch. Serde keeps the transcript
/// cache format (see `transcript_cache`).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PrebuiltCard {
    /// Media-timeline start of the first word of the sentence.
    pub(crate) start_ms: u64,
    /// Media-timeline end of the last word of the sentence.
    pub(crate) end_ms: u64,
    /// Source-language sentence.
    pub(crate) source: String,
    /// English translation once the worker delivered it.
    pub(crate) translation: Option<String>,
}

/// Live-caption player state.
#[derive(Debug, Default)]
pub struct Session {
    /// Header text (file name).
    pub file_title: String,
    /// Status line under the video.
    pub status: String,
    /// Opened media; `None` until a file is picked.
    pub video: Option<MediaInfo>,
    /// Full path of the opened media (transcript caching).
    pub video_path: Option<std::path::PathBuf>,
    /// Directory for on-disk transcript caches; `None` disables caching.
    pub transcript_dir: Option<std::path::PathBuf>,
    /// Transport state.
    pub playing: bool,
    pub playhead_ms: u64,
    /// Playback volume in percent `0..=150` (100 = unity, above = boost).
    pub volume_percent: u8,
    pub muted: bool,
    /// Whisper model name (see `sublayer_ai::MODELS`).
    pub model: String,
    /// Pinned caption language; `None` = auto-detect.
    pub language: Option<String>,
    /// Whether live captions are translated to English.
    pub translate_enabled: bool,
    /// Whisper's auto-detected language (auto mode).
    pub detected_language: Option<String>,
    /// The user is dragging the seek slider; the pump holds the playhead
    /// still so the scrub position does not fight the wall clock.
    pub scrubbing: bool,
    /// Text shown in the overlay.
    pub caption_text: String,
    /// A full-file batch transcription is being built (see the player's
    /// pre-transcribe button); words are routed here instead of the live line.
    pub prebuilding: bool,
    /// The batch finished; captions come from `prebuilt` (and `prebuilt_cards`
    /// when translating), with no whisper latency.
    pub prebuilt_ready: bool,
    /// Batch words on the media timeline, in decode (≈ timeline) order.
    prebuilt: Vec<WordToken>,
    /// Batch sentences, in finalization order, with translation slots.
    prebuilt_cards: Vec<PrebuiltCard>,
    /// Words still accumulating into the card being segmented.
    pending: VecDeque<WordToken>,
    last_word_end_ms: u64,
}

impl Session {
    /// A video was opened; resets playback, captions, and the pre-built
    /// transcript (it belongs to the previous media). A cached transcript
    /// for `path` is restored by the bridge right after this event.
    pub fn open_video(
        &mut self,
        title: impl Into<String>,
        duration_ms: u64,
        path: std::path::PathBuf,
    ) {
        self.file_title = title.into();
        self.video = Some(MediaInfo { duration_ms });
        self.video_path = Some(path);
        self.playing = false;
        self.playhead_ms = 0;
        self.scrubbing = false;
        self.clear_captions();
        self.prebuilding = false;
        self.prebuilt_ready = false;
        self.prebuilt.clear();
        self.prebuilt_cards.clear();
    }

    /// Duration of the opened media, or zero.
    pub fn duration_ms(&self) -> u64 {
        self.video.as_ref().map_or(0, |video| video.duration_ms)
    }

    /// Stop button: halts transport, rewinds, and clears captions.
    pub fn stop_transport(&mut self) {
        self.playing = false;
        self.playhead_ms = 0;
        self.scrubbing = false;
        self.clear_captions();
    }

    /// Drops all caption state (stop / restart / new video).
    pub fn clear_captions(&mut self) {
        self.pending.clear();
        self.caption_text.clear();
        self.last_word_end_ms = 0;
    }

    /// Feeds recognized words onto the pre-built timeline, returning the
    /// finalized sentences (usually zero or one) that should be translated.
    ///
    /// Words always land on the timeline; while a batch is running they are
    /// also segmented into timed cards so playback can show them (and their
    /// translations) with no whisper latency.
    pub fn feed_words(&mut self, words: Vec<WordToken>) -> Vec<String> {
        let mut finalized = Vec::new();
        for word in words {
            self.prebuilt.push(word.clone());
            if !self.pending.is_empty()
                && word.start_ms.saturating_sub(self.last_word_end_ms) >= FINALIZE_GAP_MS
            {
                self.finalize_into(&mut finalized);
            }
            let previous = self.pending.back();
            if previous.is_some_and(|word| word.text.ends_with(SENTENCE_END)) {
                self.finalize_into(&mut finalized);
            }
            let end_ms = word.end_ms;
            self.pending.push_back(word);
            self.last_word_end_ms = end_ms;
            if self
                .pending
                .back()
                .is_some_and(|word| word.text.ends_with(SENTENCE_END))
            {
                self.finalize_into(&mut finalized);
            }
        }
        finalized
    }

    /// Finalizes the pending words into a timed card, collecting non-empty
    /// sentences into `finalized` (for translation).
    fn finalize_into(&mut self, finalized: &mut Vec<String>) {
        let start_ms = self.pending.front().map(|word| word.start_ms).unwrap_or(0);
        let end_ms = self
            .pending
            .back()
            .map(|word| word.end_ms)
            .unwrap_or(start_ms);
        let text = self.pending_text();
        self.pending.clear();
        if text.is_empty() {
            return;
        }
        if self.prebuilding {
            self.prebuilt_cards.push(PrebuiltCard {
                start_ms,
                end_ms,
                source: text.clone(),
                translation: None,
            });
        }
        finalized.push(text);
    }

    /// The pending card's words, joined with spaces except between CJK
    /// characters (Chinese/Japanese/Korean are written unspaced).
    fn pending_text(&self) -> String {
        let mut text = String::new();
        for word in &self.pending {
            join_word(&mut text, &word.text);
        }
        text
    }

    /// Recomputes the overlay text; call every pump tick (playhead moves).
    ///
    /// Captions come exclusively from the pre-built timeline (see
    /// [`Self::prebuilt_caption`]); before a batch finishes the overlay is
    /// empty — there is no live ASR.
    pub fn refresh_caption(&mut self, playhead_ms: u64) {
        if self.prebuilt_ready {
            self.caption_text = self.prebuilt_caption(playhead_ms);
        } else {
            self.caption_text.clear();
        }
    }

    /// Overlay text from the batch transcript at `playhead_ms`.
    ///
    /// Subtitle-style: the sentence covering the playhead, translated when
    /// enabled (source until the translation arrives). Before the first
    /// card, in inter-card gaps, and for a while after a card ends it falls
    /// back to the rolling word line.
    fn prebuilt_caption(&self, playhead_ms: u64) -> String {
        let index = self
            .prebuilt_cards
            .partition_point(|card| card.start_ms <= playhead_ms)
            .saturating_sub(1);
        if let Some(card) = self.prebuilt_cards.get(index)
            && playhead_ms <= card.end_ms.saturating_add(PREBUILT_LINGER_MS)
        {
            if self.translate_enabled
                && let Some(translated) = &card.translation
            {
                return translated.clone();
            }
            return card.source.clone();
        }
        self.prebuilt_word_line(playhead_ms)
    }

    /// The rolling word line: words finished within the linger window and
    /// not starting more than a lookahead ahead of the playhead.
    fn prebuilt_word_line(&self, playhead_ms: u64) -> String {
        let keep_from = playhead_ms.saturating_sub(PREBUILT_LINGER_MS);
        let lookahead = playhead_ms.saturating_add(PREBUILT_LOOKAHEAD_MS);
        let mut text = String::new();
        for word in &self.prebuilt {
            if word.end_ms < keep_from {
                continue;
            }
            if word.start_ms > lookahead {
                break; // sorted by timeline; nothing later qualifies
            }
            join_word(&mut text, &word.text);
        }
        text
    }

    /// The batch transcript is cleared for a fresh run (cards included).
    pub fn prebuilt_clear_for_batch(&mut self) {
        self.prebuilt.clear();
        self.prebuilt_cards.clear();
        self.pending.clear();
        self.last_word_end_ms = 0;
    }

    /// Drops the pre-built transcript because the model or pinned language
    /// changed: the words no longer match the selection, so captions stay
    /// off until Pre-transcribe runs again.
    pub fn invalidate_prebuilt(&mut self) {
        self.prebuilt_ready = false;
        self.prebuilt_clear_for_batch();
        self.clear_captions();
    }

    /// The batch finished: sort and dedupe the words (live and batch words
    /// can interleave when play started mid-build), switch the display to
    /// the pre-built timeline, and cache the transcript on disk so the next
    /// open of the same video is instant.
    pub fn mark_prebuilt_ready(&mut self) {
        self.prebuilding = false;
        self.prebuilt_ready = true;
        self.prebuilt.sort_by_key(|word| word.start_ms);
        self.prebuilt
            .dedup_by(|a, b| a.start_ms == b.start_ms && a.end_ms == b.end_ms && a.text == b.text);
        self.save_transcript_cache();
    }

    /// Installs a transcript restored from the on-disk cache (see
    /// `transcript_cache`); the batch itself is not re-run.
    pub fn install_prebuilt(&mut self, words: Vec<WordToken>, cards: Vec<PrebuiltCard>) {
        self.prebuilt = words;
        self.prebuilt_cards = cards;
    }

    /// Stores the current transcript under the XDG cache directory when one
    /// is configured; failures are silent (the batch simply re-runs next
    /// time).
    fn save_transcript_cache(&self) {
        let (Some(dir), Some(path)) = (&self.transcript_dir, &self.video_path) else {
            return;
        };
        let _ = crate::transcript_cache::save(
            dir,
            path,
            &self.model,
            self.language.as_deref(),
            &self.prebuilt,
            &self.prebuilt_cards,
        );
    }

    /// Attaches an English translation to the oldest pre-built card that
    /// still lacks one (worker results arrive in submission order).
    pub fn attach_prebuilt_translation(&mut self, text: String) {
        if let Some(card) = self
            .prebuilt_cards
            .iter_mut()
            .find(|card| card.translation.is_none())
        {
            card.translation = Some(text);
        }
    }

    /// Whether every pre-built card has its translation.
    pub fn translations_done(&self) -> bool {
        self.prebuilt_cards
            .iter()
            .all(|card| card.translation.is_some())
    }

    /// Word count on the pre-built timeline (for the status line).
    pub fn prebuilt_word_count(&self) -> usize {
        self.prebuilt.len()
    }

    /// Source language used to pick a bundled pair: the pinned selection
    /// wins; otherwise whisper's detection. Whisper's `"zh"` becomes the
    /// bundled `"zh-CN"` pair.
    pub fn translation_code(&self) -> Option<&str> {
        let code = self
            .language
            .as_deref()
            .or(self.detected_language.as_deref())?;
        Some(if code == "zh" { "zh-CN" } else { code })
    }

    /// Bundled translation pair for the current source language.
    pub fn translation_pair(&self) -> Option<LocalPair> {
        self.translation_code().and_then(find_local_pair).copied()
    }

    /// Whether live translation can run for the current source language.
    pub fn translation_available(&self) -> bool {
        self.translation_pair().is_some()
    }
}

/// Appends `word` to `text` with a space unless both sides are CJK.
fn join_word(text: &mut String, word: &str) {
    if text.is_empty() {
        text.push_str(word);
        return;
    }
    let previous = text.chars().last().expect("text is not empty");
    let next = word.chars().next().expect("word is not empty");
    if is_cjk(previous) && is_cjk(next) {
        text.push_str(word);
    } else {
        text.push(' ');
        text.push_str(word);
    }
}

/// CJK Unified Ideographs, Extension A, Hiragana/Katakana, Hangul Syllables.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0x3040..=0x30FF
        | 0xAC00..=0xD7AF
    )
}

/// Formats milliseconds as `m:ss`.
pub fn format_time(ms: u64) -> String {
    let total_seconds = ms / 1_000;
    format!("{}:{:02}", total_seconds / 60, total_seconds % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start_ms: u64) -> WordToken {
        WordToken::new(text, start_ms, start_ms + 300)
    }

    fn install_batch(session: &mut Session, words: &[(&str, u64)]) {
        session.prebuilding = true;
        session.feed_words(
            words
                .iter()
                .map(|(text, start)| word(text, *start))
                .collect(),
        );
        session.mark_prebuilt_ready();
    }

    #[test]
    fn prebuilt_word_line_joins_latin_with_spaces() {
        let mut session = Session::default();
        install_batch(&mut session, &[("Hello", 0), ("world", 400)]);
        session.refresh_caption(1_000);
        assert_eq!(session.caption_text, "Hello world");
    }

    #[test]
    fn prebuilt_word_line_joins_cjk_without_spaces() {
        let mut session = Session::default();
        install_batch(&mut session, &[("今天", 0), ("天气", 400), ("很好", 800)]);
        session.refresh_caption(2_000);
        assert_eq!(session.caption_text, "今天天气很好");
    }

    #[test]
    fn long_pauses_finalize_the_card() {
        let mut session = Session {
            prebuilding: true,
            ..Session::default()
        };
        let finalized = session.feed_words(vec![
            word("Hello", 0),
            word("world", 300),
            word("Next", 2_000), // gap >= 900 ms
        ]);
        assert_eq!(finalized, ["Hello world"]);
        assert_eq!(session.prebuilt_cards.len(), 1);
        assert_eq!(session.prebuilt_cards[0].source, "Hello world");
    }

    #[test]
    fn sentence_end_punctuation_finalizes() {
        let mut session = Session {
            prebuilding: true,
            ..Session::default()
        };
        let finalized = session.feed_words(vec![
            word("Hello", 0),
            word("world.", 300),
            word("How", 700),
        ]);
        assert_eq!(finalized, ["Hello world."]);
        assert_eq!(session.prebuilt_cards.len(), 1);
    }

    #[test]
    fn no_batch_yet_means_no_captions() {
        // There is no live ASR: before a pre-transcription finishes the
        // overlay stays empty even when words have been fed.
        let mut session = Session::default();
        session.feed_words(vec![word("Hello", 0), word("world.", 300)]);
        session.refresh_caption(2_000);
        assert!(session.caption_text.is_empty());

        session.mark_prebuilt_ready();
        session.refresh_caption(400);
        assert_eq!(session.caption_text, "Hello world.");
    }

    #[test]
    fn mandarin_whisper_code_maps_to_the_bundled_pair() {
        let session = Session {
            detected_language: Some("zh".to_owned()),
            ..Session::default()
        };
        assert_eq!(session.translation_code(), Some("zh-CN"));
        let pair = session.translation_pair().expect("zh-CN is bundled");
        assert_eq!(pair.model_prefix, "zhen");
    }

    #[test]
    fn pinned_language_beats_detection() {
        let session = Session {
            language: Some("ko".to_owned()),
            detected_language: Some("zh".to_owned()),
            ..Session::default()
        };
        assert_eq!(session.translation_code(), Some("ko"));
    }

    #[test]
    fn bundled_pairs_cover_the_six_languages_only() {
        for code in ["ko", "zh-CN", "ja", "es", "da", "ru"] {
            let session = Session {
                detected_language: Some(code.to_owned()),
                ..Session::default()
            };
            assert!(session.translation_available(), "{code} must be bundled");
        }
        // No Traditional-Chinese model exists in Mozilla's collection yet,
        // so zh-TW must report unavailable until one is bundled.
        for code in ["en", "de", "fr", "sw", "zh-TW"] {
            let session = Session {
                detected_language: Some(code.to_owned()),
                ..Session::default()
            };
            assert!(
                !session.translation_available(),
                "{code} must not be bundled"
            );
        }
    }

    #[test]
    fn open_video_resets_transport_and_captions() {
        let mut session = Session {
            playing: true,
            scrubbing: true,
            playhead_ms: 45_000,
            ..Session::default()
        };
        install_batch(&mut session, &[("leftover", 0)]);
        session.open_video("clip.mp4", 60_000, std::path::PathBuf::from("clip.mp4"));
        assert_eq!(session.playhead_ms, 0);
        assert!(!session.playing);
        assert!(!session.scrubbing);
        assert_eq!(session.duration_ms(), 60_000);
        assert_eq!(session.caption_text, "");
        assert!(!session.prebuilt_ready, "captions belong to the old media");
    }

    #[test]
    fn batch_words_feed_the_prebuilt_timeline_and_cards() {
        let mut session = Session {
            prebuilding: true,
            ..Session::default()
        };
        // Two sentences plus a trailing partial line.
        let sentences = session.feed_words(vec![
            word("Hallo", 0),
            word("Welt.", 300),
            word("Wie", 1_500),
            word("geht", 1_800),
            word("es?", 2_100),
            word("Mir", 3_500),
        ]);
        assert_eq!(sentences, ["Hallo Welt.", "Wie geht es?"]);
        assert_eq!(
            session.prebuilt.len(),
            6,
            "every word lands on the timeline"
        );
        assert_eq!(session.prebuilt_cards.len(), 2);
        assert_eq!(session.prebuilt_cards[0].start_ms, 0);
        assert_eq!(session.prebuilt_cards[1].start_ms, 1_500);
    }

    #[test]
    fn prebuilt_captions_are_instant_at_the_playhead() {
        let mut session = Session {
            prebuilding: true,
            ..Session::default()
        };
        install_batch(
            &mut session,
            &[
                ("Hallo", 0),
                ("Welt.", 300),
                ("Wie", 1_500),
                ("geht", 1_800),
                ("es?", 2_100),
            ],
        );

        session.refresh_caption(400);
        assert_eq!(session.caption_text, "Hallo Welt.");
        // Straight to the next sentence the moment it starts; no whisper lag.
        session.refresh_caption(1_600);
        assert_eq!(session.caption_text, "Wie geht es?");
        // Words finish their 3 s linger, then the overlay empties.
        session.refresh_caption(6_000);
        assert!(session.caption_text.is_empty());
    }

    #[test]
    fn translating_during_the_batch_replaces_cards_at_their_times() {
        let mut session = Session {
            prebuilding: true,
            translate_enabled: true,
            ..Session::default()
        };
        install_batch(
            &mut session,
            &[
                ("Hallo", 0),
                ("Welt.", 300),
                ("Wie", 1_500),
                ("geht", 1_800),
                ("es?", 2_100),
            ],
        );
        session.attach_prebuilt_translation("Hello world.".to_owned());
        session.attach_prebuilt_translation("How are you?".to_owned());

        session.refresh_caption(400);
        assert_eq!(session.caption_text, "Hello world.");
        session.refresh_caption(1_600);
        assert_eq!(session.caption_text, "How are you?");

        // Untranslated cards fall back to the source sentence.
        let mut session = Session {
            prebuilding: true,
            translate_enabled: true,
            ..Session::default()
        };
        install_batch(&mut session, &[("Hallo", 0), ("Welt.", 300)]);
        session.refresh_caption(400);
        assert_eq!(session.caption_text, "Hallo Welt.");
    }

    #[test]
    fn time_labels_format_minutes_and_seconds() {
        assert_eq!(format_time(0), "0:00");
        assert_eq!(format_time(59_999), "0:59");
        assert_eq!(format_time(60_000), "1:00");
        assert_eq!(format_time(3_725_000), "62:05");
    }
}
