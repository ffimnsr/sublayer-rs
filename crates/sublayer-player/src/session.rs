//! Playback and caption state for the player window.
//!
//! Everything here is plain data mutated by the UI pump; heavy work (audio,
//! ASR, translation, decoding) lives in `bridge`. Caption assembly is the
//! live part: recognized words accumulate into a rolling line that is
//! finalized (a "card") on a long pause or sentence-ending punctuation, and
//! finalized cards are what get translated.

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

/// Words older than this (relative to the playhead) leave the rolling line.
///
/// Sized for whisper latency: words land seconds after they were spoken
/// (a 4 s ASR window plus decode time), so the window must comfortably
/// cover capture + decode + a readable linger; 12 s keeps a slow model's
/// captions visible instead of pruning them before display.
const LIVE_WINDOW_MS: u64 = 12_000;

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
/// in by the translation worker during the batch.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PrebuiltCard {
    /// Media-timeline start of the first word of the sentence.
    start_ms: u64,
    /// Media-timeline end of the last word of the sentence.
    end_ms: u64,
    /// Source-language sentence.
    source: String,
    /// English translation once the worker delivered it.
    translation: Option<String>,
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
    /// Transport state.
    pub playing: bool,
    pub playhead_ms: u64,
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
    /// Playhead of the last seek actually sent to the audio driver; closer
    /// slider movements update the display only.
    pub last_bridge_seek_ms: u64,
    /// Text shown in the overlay.
    pub caption_text: String,
    /// Last finalized source sentence.
    pub current_card: String,
    /// Latest translated sentence (English).
    pub translated_text: String,
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
    /// Words still accumulating into the live line.
    live_words: VecDeque<WordToken>,
    last_word_end_ms: u64,
}

impl Session {
    /// A video was opened; resets playback, captions, and the pre-built
    /// transcript (it belongs to the previous media).
    pub fn open_video(&mut self, title: impl Into<String>, duration_ms: u64) {
        self.file_title = title.into();
        self.video = Some(MediaInfo { duration_ms });
        self.playing = false;
        self.playhead_ms = 0;
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
        self.clear_captions();
    }

    /// Drops all caption state (stop / restart / new video).
    pub fn clear_captions(&mut self) {
        self.live_words.clear();
        self.current_card.clear();
        self.translated_text.clear();
        self.caption_text.clear();
        self.last_word_end_ms = 0;
    }

    /// Feeds recognized words into the live line, returning the finalized
    /// sentences (usually zero or one) that should be translated.
    ///
    /// During a batch pre-transcription the words feed the pre-built
    /// transcript instead, and the finalized sentences are recorded as timed
    /// cards so their translations can be shown during playback.
    pub fn feed_words(&mut self, words: Vec<WordToken>) -> Vec<String> {
        let mut finalized = Vec::new();
        for word in words {
            if self.prebuilding {
                // Keep a full-timeline copy for the pre-built display; the
                // live line below does the sentence segmentation.
                self.prebuilt.push(word.clone());
            }
            if !self.live_words.is_empty()
                && word.start_ms.saturating_sub(self.last_word_end_ms) >= FINALIZE_GAP_MS
            {
                self.finalize_into(&mut finalized);
            }
            let previous = self.live_words.back();
            if previous.is_some_and(|word| word.text.ends_with(SENTENCE_END)) {
                self.finalize_into(&mut finalized);
            }
            let end_ms = word.end_ms;
            self.live_words.push_back(word);
            self.last_word_end_ms = end_ms;
            if self
                .live_words
                .back()
                .is_some_and(|word| word.text.ends_with(SENTENCE_END))
            {
                self.finalize_into(&mut finalized);
            }
        }
        finalized
    }

    /// Finalizes the live line into `current_card`, collecting non-empty
    /// sentences into `finalized`; during a batch the sentence is also
    /// recorded as a timed card for translation.
    fn finalize_into(&mut self, finalized: &mut Vec<String>) {
        let start_ms = self
            .live_words
            .front()
            .map(|word| word.start_ms)
            .unwrap_or(0);
        let end_ms = self
            .live_words
            .back()
            .map(|word| word.end_ms)
            .unwrap_or(start_ms);
        let text = self.live_text();
        self.live_words.clear();
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
        self.current_card = text.clone();
        finalized.push(text);
    }

    /// The still-accumulating line, words joined with spaces except between
    /// CJK characters (Chinese/Japanese/Korean are written unspaced).
    fn live_text(&self) -> String {
        let mut text = String::new();
        for word in &self.live_words {
            join_word(&mut text, &word.text);
        }
        text
    }

    /// Recomputes the overlay text; call every pump tick (playhead moves).
    ///
    /// Once a batch transcript is ready it wins over the live line: words
    /// are looked up on the timeline directly, so captions (and their
    /// translations) appear instantly with no whisper latency.
    pub fn refresh_caption(&mut self, playhead_ms: u64) {
        if self.prebuilt_ready {
            self.caption_text = self.prebuilt_caption(playhead_ms);
            return;
        }
        let keep_from = playhead_ms.saturating_sub(LIVE_WINDOW_MS);
        while self
            .live_words
            .front()
            .is_some_and(|word| word.end_ms < keep_from)
        {
            self.live_words.pop_front();
        }
        let live = self.live_text();
        self.caption_text = if self.translate_enabled {
            if !self.translated_text.is_empty() {
                self.translated_text.clone()
            } else if !self.current_card.is_empty() {
                self.current_card.clone()
            } else {
                live
            }
        } else if !live.is_empty() {
            live
        } else {
            self.current_card.clone()
        };
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
    }

    /// The batch finished: sort and dedupe the words (live and batch words
    /// can interleave when play started mid-build) and switch the display to
    /// the pre-built timeline.
    pub fn mark_prebuilt_ready(&mut self) {
        self.prebuilding = false;
        self.prebuilt_ready = true;
        self.prebuilt.sort_by_key(|word| word.start_ms);
        self.prebuilt
            .dedup_by(|a, b| a.start_ms == b.start_ms && a.end_ms == b.end_ms && a.text == b.text);
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

    /// Whether any pre-built card still waits for its translation.
    pub fn has_untranslated_cards(&self) -> bool {
        self.prebuilt_cards
            .iter()
            .any(|card| card.translation.is_none())
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

    /// Stores a translation of the latest finalized card.
    pub fn set_translation(&mut self, text: impl Into<String>) {
        self.translated_text = text.into();
        if self.translate_enabled {
            self.caption_text = self.translated_text.clone();
        }
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

    #[test]
    fn words_join_with_spaces_for_latin() {
        let mut session = Session::default();
        session.feed_words(vec![word("Hello", 0), word("world", 400)]);
        session.refresh_caption(1_000);
        assert_eq!(session.caption_text, "Hello world");
    }

    #[test]
    fn cjk_words_join_without_spaces() {
        let mut session = Session::default();
        session.feed_words(vec![word("今天", 0), word("天气", 400), word("很好", 800)]);
        session.refresh_caption(2_000);
        assert_eq!(session.caption_text, "今天天气很好");
    }

    #[test]
    fn long_pauses_finalize_the_line() {
        let mut session = Session::default();
        let finalized = session.feed_words(vec![
            word("Hello", 0),
            word("world", 300),
            word("Next", 2_000), // gap >= 900 ms
        ]);
        assert_eq!(finalized, ["Hello world"]);
        session.refresh_caption(3_000);
        assert_eq!(session.caption_text, "Next");
        assert_eq!(session.current_card, "Hello world");
    }

    #[test]
    fn sentence_end_punctuation_finalizes() {
        let mut session = Session::default();
        let finalized = session.feed_words(vec![
            word("Hello", 0),
            word("world.", 300),
            word("How", 700),
        ]);
        assert_eq!(finalized, ["Hello world."]);
        session.refresh_caption(2_000);
        assert_eq!(session.caption_text, "How");
        assert_eq!(session.current_card, "Hello world.");
    }

    #[test]
    fn stale_live_words_drop_out_of_the_window() {
        let mut session = Session::default();
        session.feed_words(vec![word("old", 0), word("new", 8_000)]);
        session.refresh_caption(15_000);
        assert!(!session.caption_text.contains("old"));
        assert!(session.caption_text.contains("new"));
    }

    #[test]
    fn words_survive_whisper_latency_while_playing() {
        // Whisper emits words seconds after capture: a word spoken at 0 ms
        // lands at the playhead around 5-8 s on a slow model and must still
        // be on screen, not pruned by the live window.
        let mut session = Session::default();
        session.feed_words(vec![word("spoken", 0), word("early", 300)]);
        session.refresh_caption(6_000);
        assert_eq!(session.caption_text, "spoken early");
        session.refresh_caption(11_000);
        assert_eq!(session.caption_text, "spoken early");
        session.refresh_caption(13_000);
        assert!(session.caption_text.is_empty());
    }

    #[test]
    fn translation_overlay_has_priority_when_enabled() {
        let mut session = Session {
            translate_enabled: true,
            ..Session::default()
        };
        session.feed_words(vec![word("Hallo", 0), word("Welt.", 300)]);
        session.set_translation("Hello world.");
        session.refresh_caption(2_000);
        assert_eq!(session.caption_text, "Hello world.");

        session.translate_enabled = false;
        session.refresh_caption(2_000);
        assert_eq!(session.caption_text, "Hallo Welt.");
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
        for code in ["en", "de", "fr", "sw"] {
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
            playhead_ms: 45_000,
            ..Session::default()
        };
        session.feed_words(vec![word("leftover", 0)]);
        session.open_video("clip.mp4", 60_000);
        assert_eq!(session.playhead_ms, 0);
        assert!(!session.playing);
        assert_eq!(session.duration_ms(), 60_000);
        assert_eq!(session.caption_text, "");
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
        session.feed_words(vec![
            word("Hallo", 0),
            word("Welt.", 300),
            word("Wie", 1_500),
            word("geht", 1_800),
            word("es?", 2_100),
        ]);
        session.mark_prebuilt_ready();

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
        session.feed_words(vec![
            word("Hallo", 0),
            word("Welt.", 300),
            word("Wie", 1_500),
            word("geht", 1_800),
            word("es?", 2_100),
        ]);
        session.attach_prebuilt_translation("Hello world.".to_owned());
        session.attach_prebuilt_translation("How are you?".to_owned());
        session.mark_prebuilt_ready();

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
        session.feed_words(vec![word("Hallo", 0), word("Welt.", 300)]);
        session.mark_prebuilt_ready();
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
