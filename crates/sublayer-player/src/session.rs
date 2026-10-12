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
const LIVE_WINDOW_MS: u64 = 5_000;

/// Sentence-ending punctuation; a word ending in one of these finalizes.
const SENTENCE_END: &[char] = &['.', '?', '!', '。', '！', '？', '…'];

/// Playback duration of the opened media.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MediaInfo {
    pub duration_ms: u64,
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
    /// Words still accumulating into the live line.
    live_words: VecDeque<WordToken>,
    last_word_end_ms: u64,
}

impl Session {
    /// A video was opened; resets playback and captions.
    pub fn open_video(&mut self, title: impl Into<String>, duration_ms: u64) {
        self.file_title = title.into();
        self.video = Some(MediaInfo { duration_ms });
        self.playing = false;
        self.playhead_ms = 0;
        self.clear_captions();
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
    pub fn feed_words(&mut self, words: Vec<WordToken>) -> Vec<String> {
        let mut finalized = Vec::new();
        for word in words {
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
    /// sentences into `finalized`.
    fn finalize_into(&mut self, finalized: &mut Vec<String>) {
        let text = self.live_text();
        self.live_words.clear();
        if text.is_empty() {
            return;
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
    pub fn refresh_caption(&mut self, playhead_ms: u64) {
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
        session.refresh_caption(10_000);
        assert!(!session.caption_text.contains("old"));
        assert!(session.caption_text.contains("new"));
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
    fn time_labels_format_minutes_and_seconds() {
        assert_eq!(format_time(0), "0:00");
        assert_eq!(format_time(59_999), "0:59");
        assert_eq!(format_time(60_000), "1:00");
        assert_eq!(format_time(3_725_000), "62:05");
    }
}
