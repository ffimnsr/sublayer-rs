//! Word clustering: turns a continuous word stream into caption cards.
//!
//! A new card starts when any of these conditions holds:
//!
//! * the silence between two words reaches [`SegmenterConfig::min_pause_ms`]
//!   (the classic short-form `>= 350 ms` rule),
//! * a sentence-ending word (`.`, `!`, `?`, `…`) is followed by at least
//!   [`SegmenterConfig::min_punctuation_pause_ms`] of silence,
//! * the card would exceed [`SegmenterConfig::max_chars`] characters — the
//!   split then prefers the last clause boundary (`,`, `;`, `:`) inside the
//!   card so the break lands on a natural pause,
//! * the card would exceed [`SegmenterConfig::max_duration_ms`].
//!
//! Pauses are always measured between the previous word's end and the next
//! word's start, regardless of card boundaries.
//!
//! A trailing card with fewer than [`SegmenterConfig::min_words`] words is
//! folded back into its predecessor when it still fits. The default of `1`
//! disables folding, keeping the pause heuristics fully visible; raise it to
//! avoid lone one-word cards at the end of a caption run.

use sublayer_core::{CaptionSegment, WordToken};

use serde::{Deserialize, Serialize};

/// Tuning knobs for [`segment_words`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmenterConfig {
    /// Hard character ceiling per caption card (whitespace included).
    pub max_chars: usize,
    /// Silence that always starts a new card, in milliseconds.
    pub min_pause_ms: u64,
    /// Silence after sentence punctuation that starts a new card, in
    /// milliseconds. Must be below `min_pause_ms` to be reachable.
    pub min_punctuation_pause_ms: u64,
    /// Hard duration ceiling per card, in milliseconds.
    pub max_duration_ms: u64,
    /// Fewest words a card may hold; undersized trailing cards are merged into
    /// the previous card when they fit.
    pub min_words: usize,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            max_chars: 28,
            min_pause_ms: 350,
            min_punctuation_pause_ms: 120,
            max_duration_ms: 3_000,
            min_words: 1,
        }
    }
}

/// Clusters `words` (already in speech order) into caption cards.
pub fn segment_words(words: &[WordToken], config: &SegmenterConfig) -> Vec<CaptionSegment> {
    let mut segments: Vec<CaptionSegment> = Vec::new();
    let mut current: Vec<WordToken> = Vec::new();
    let mut prev: Option<&WordToken> = None;

    for word in words {
        if word.text.trim().is_empty() {
            continue;
        }
        if let Some(prev_word) = prev {
            if !current.is_empty() && should_split(&current, word, prev_word, config) {
                split_current(&mut segments, &mut current);
            }
        }
        current.push(word.clone());
        prev = Some(word);
    }
    flush(&mut segments, &mut current);
    merge_orphans(&mut segments, config);
    segments
}

/// Whether `word` must open a new card instead of joining `current`.
fn should_split(
    current: &[WordToken],
    word: &WordToken,
    prev: &WordToken,
    config: &SegmenterConfig,
) -> bool {
    let gap = word.start_ms.saturating_sub(prev.end_ms);
    if gap >= config.min_pause_ms {
        return true;
    }
    if gap >= config.min_punctuation_pause_ms && ends_sentence(&prev.text) {
        return true;
    }
    if card_duration_ms(current, prev) >= config.max_duration_ms {
        return true;
    }
    card_chars(current) + 1 + word.text.chars().count() > config.max_chars
}

/// Breaks the working card apart. When the card holds a clause boundary with
/// words on both sides, the split lands after the last such boundary so the
/// break feels natural; otherwise the split is a hard word boundary.
fn split_current(segments: &mut Vec<CaptionSegment>, current: &mut Vec<WordToken>) {
    let len = current.len();
    for index in (1..len.saturating_sub(1)).rev() {
        if ends_clause(&current[index].text) {
            let tail = current.split_off(index + 1);
            flush(segments, current);
            *current = tail;
            return;
        }
    }
    flush(segments, current);
}

/// Accumulated character count of a card, spaces between words included.
fn card_chars(words: &[WordToken]) -> usize {
    words
        .iter()
        .map(|word| word.text.chars().count())
        .sum::<usize>()
        + words.len().saturating_sub(1)
}

/// Span from the first word's start to the last word's end.
fn card_duration_ms(words: &[WordToken], last: &WordToken) -> u64 {
    let first = words
        .first()
        .map(|word| word.start_ms)
        .unwrap_or(last.start_ms);
    last.end_ms.saturating_sub(first)
}

/// Empties `current` into `segments` as a finished card.
fn flush(segments: &mut Vec<CaptionSegment>, current: &mut Vec<WordToken>) {
    if !current.is_empty() {
        segments.push(CaptionSegment::new(std::mem::take(current)));
    }
}

/// Folds undersized trailing cards back into their predecessor.
///
/// Only trailing cards matter: a one-word card in the middle of the run is a
/// deliberate pause split, while a lone word at the very end usually reads as
/// a stray interjection.
fn merge_orphans(segments: &mut Vec<CaptionSegment>, config: &SegmenterConfig) {
    // `min_words < 2` keeps every pause split visible; only an explicit
    // minimum turns folding on.
    while segments.len() >= 2 && config.min_words >= 2 {
        let tail_words = segments
            .last()
            .map(|segment| segment.words.len())
            .unwrap_or(0);
        if tail_words >= config.min_words {
            break;
        }
        let Some(tail) = segments.pop() else { break };
        let Some(prev) = segments.last_mut() else {
            segments.push(tail);
            break;
        };
        let fits = card_chars(&prev.words) + 1 + card_chars(&tail.words) <= config.max_chars;
        let within_duration = tail.words.last().is_some_and(|word| {
            word.end_ms.saturating_sub(prev.start_ms().unwrap_or(0)) <= config.max_duration_ms
        });
        if fits && within_duration {
            prev.words.extend(tail.words);
        } else {
            segments.push(tail);
            break;
        }
    }
}

/// Whether the word closes a sentence.
fn ends_sentence(text: &str) -> bool {
    text.ends_with(['.', '!', '?', '…'])
}

/// Whether the word ends a clause that makes a natural (soft) break point.
fn ends_clause(text: &str) -> bool {
    text.ends_with([',', ';', ':'])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start_ms: u64, end_ms: u64) -> WordToken {
        WordToken::new(text, start_ms, end_ms)
    }

    fn text_of(segments: &[CaptionSegment]) -> Vec<String> {
        segments.iter().map(CaptionSegment::text).collect()
    }

    #[test]
    fn long_pause_splits_cards() {
        let words = vec![
            word("Hello", 0, 300),
            word("world", 700, 1000),
            word("again", 1400, 1700),
        ];
        let segments = segment_words(&words, &SegmenterConfig::default());
        assert_eq!(text_of(&segments), vec!["Hello", "world", "again"]);
    }

    #[test]
    fn short_pause_keeps_one_card() {
        let words = vec![word("Hello", 0, 300), word("world", 400, 700)];
        let segments = segment_words(&words, &SegmenterConfig::default());
        assert_eq!(text_of(&segments), vec!["Hello world"]);
    }

    #[test]
    fn sentence_punctuation_splits_with_enough_pause() {
        let words = vec![
            word("Hello", 0, 300),
            word("world.", 300, 600),
            word("Next", 800, 1000),
        ];
        let segments = segment_words(&words, &SegmenterConfig::default());
        assert_eq!(text_of(&segments), vec!["Hello world.", "Next"]);
    }

    #[test]
    fn punctuation_without_pause_stays_in_card() {
        let words = vec![
            word("Hello", 0, 300),
            word("world.", 320, 600),
            word("Next", 620, 800),
        ];
        let segments = segment_words(&words, &SegmenterConfig::default());
        assert_eq!(text_of(&segments), vec!["Hello world. Next"]);
    }

    #[test]
    fn char_limit_splits_even_without_pause() {
        let config = SegmenterConfig {
            max_chars: 15,
            ..SegmenterConfig::default()
        };
        let words = vec![
            word("The", 0, 100),
            word("quick", 100, 200),
            word("brown", 200, 300),
            word("fox", 300, 400),
        ];
        let segments = segment_words(&words, &config);
        assert_eq!(text_of(&segments), vec!["The quick brown", "fox"]);
    }

    #[test]
    fn char_limit_prefers_clause_boundary() {
        let config = SegmenterConfig {
            max_chars: 20,
            ..SegmenterConfig::default()
        };
        let words = vec![
            word("One,", 0, 100),
            word("two,", 100, 200),
            word("three,", 200, 300),
            word("four", 300, 400),
        ];
        let segments = segment_words(&words, &config);
        assert_eq!(text_of(&segments), vec!["One, two,", "three, four"]);
    }

    #[test]
    fn duration_cap_splits_cards() {
        let config = SegmenterConfig {
            max_duration_ms: 900,
            ..SegmenterConfig::default()
        };
        let words = vec![
            word("One", 0, 100),
            word("two", 200, 300),
            word("three", 400, 500),
            word("four", 1000, 1100),
        ];
        let segments = segment_words(&words, &config);
        assert_eq!(text_of(&segments), vec!["One two three", "four"]);
    }

    #[test]
    fn single_word_orphan_merges_back_when_configured() {
        let config = SegmenterConfig {
            min_words: 2,
            ..SegmenterConfig::default()
        };
        let words = vec![
            word("Hello", 0, 300),
            word("there", 400, 700),
            word("friend", 1800, 2000),
        ];
        let segments = segment_words(&words, &config);
        // "friend" follows a 1100 ms pause, but the min_words pass folds the
        // lone trailing word back into the previous card.
        assert_eq!(text_of(&segments), vec!["Hello there friend"]);
    }

    #[test]
    fn default_config_keeps_pause_split_orphans() {
        let words = vec![
            word("Hello", 0, 300),
            word("there", 400, 700),
            word("friend", 1800, 2000),
        ];
        let segments = segment_words(&words, &SegmenterConfig::default());
        assert_eq!(text_of(&segments), vec!["Hello there", "friend"]);
    }

    #[test]
    fn orphan_merge_respects_char_limit() {
        let config = SegmenterConfig {
            max_chars: 12,
            min_words: 2,
            ..SegmenterConfig::default()
        };
        let words = vec![
            word("Hello", 0, 300),
            word("there", 400, 700),
            word("friend", 1800, 2000),
        ];
        let segments = segment_words(&words, &config);
        assert_eq!(text_of(&segments), vec!["Hello there", "friend"]);
    }

    #[test]
    fn words_longer_than_limit_survive_alone() {
        let config = SegmenterConfig {
            max_chars: 5,
            ..SegmenterConfig::default()
        };
        let words = vec![word("antidisestablishmentarianism", 0, 800)];
        let segments = segment_words(&words, &config);
        assert_eq!(text_of(&segments), vec!["antidisestablishmentarianism"]);
    }

    #[test]
    fn empty_and_blank_input_yields_no_cards() {
        assert!(segment_words(&[], &SegmenterConfig::default()).is_empty());
        let words = vec![word("", 0, 100), word("   ", 100, 200)];
        assert!(segment_words(&words, &SegmenterConfig::default()).is_empty());
    }

    #[test]
    fn gap_is_measured_across_card_boundaries() {
        // A pause split must leave the next gap measured from the last word
        // of the previous card, not from its first word: "cccc" at 320 ms is
        // only 120 ms after "bbbb" (no split), but 320 ms after the card's
        // first word (which would be a split if measured that way).
        let words = vec![
            word("aaaa", 0, 100),
            word("bbbb", 100, 200),
            word("cccc", 320, 420),
            word("dddd", 770, 870),
        ];
        let segments = segment_words(&words, &SegmenterConfig::default());
        assert_eq!(text_of(&segments), vec!["aaaa bbbb cccc", "dddd"]);
    }

    #[test]
    fn custom_min_words_merges_longer_orphans() {
        let config = SegmenterConfig {
            min_words: 3,
            ..SegmenterConfig::default()
        };
        let words = vec![
            word("One", 0, 100),
            word("two", 200, 300),
            word("three", 500, 600),
            word("four", 2000, 2100),
            word("five", 2400, 2500),
        ];
        let segments = segment_words(&words, &config);
        assert_eq!(text_of(&segments), vec!["One two three four five"]);
    }
}
