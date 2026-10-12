//! Bundled local translation pairs: source language → English.
//!
//! `sublayer-translate` ships a curated set of source languages; each entry
//! in [`LOCAL_PAIRS`] maps to a real Firefox Translations bundle published
//! through Mozilla's Remote Settings (files named
//! `model.<src><trg>.intgemm.alphas.bin`), which the
//! [`LocalBackend`](crate::local::LocalBackend) loads from a downloaded
//! directory. The destination language is **always English**; `language` is
//! the source audio/live-caption language.
//!
//! Bundled languages: Chinese (Simplified, Mandarin), Japanese, Korean,
//! Spanish, Danish, Russian. The Chinese pair ships with split source/target
//! vocabularies (`srcvocab.zhen.spm` + `trgvocab.zhen.spm`), which the
//! backend loads explicitly.

/// One source language the local backend can translate from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalPair {
    /// Firefox model prefix: the bundle files are named
    /// `model.{prefix}.intgemm.alphas.bin`,
    /// `lex.50.{50,12k}.{prefix}.s2t.bin` and `vocab.{prefix}.spm` (or
    /// `srcvocab.{prefix}.spm` / `trgvocab.{prefix}.spm`).
    pub model_prefix: &'static str,
    /// BCP-47 language code of the **source** language (`"zh-CN"`, `"ja"`,
    /// `"ko"`, ...). Whisper uses `"zh"` for Mandarin; map it to `"zh-CN"`
    /// when picking a pair.
    pub language: &'static str,
    /// Human-readable name for UI lists.
    pub display_name: &'static str,
}

/// Bundled source → English pairs (destination is always English).
pub const LOCAL_PAIRS: &[LocalPair] = &[
    LocalPair {
        model_prefix: "zhen",
        language: "zh-CN",
        display_name: "Chinese (Simplified, Mandarin)",
    },
    LocalPair {
        model_prefix: "jaen",
        language: "ja",
        display_name: "Japanese",
    },
    LocalPair {
        model_prefix: "koen",
        language: "ko",
        display_name: "Korean",
    },
    LocalPair {
        model_prefix: "esen",
        language: "es",
        display_name: "Spanish",
    },
    LocalPair {
        model_prefix: "daen",
        language: "da",
        display_name: "Danish",
    },
    LocalPair {
        model_prefix: "ruen",
        language: "ru",
        display_name: "Russian",
    },
];

/// Looks up a pair by its source language code, case-insensitively
/// (`"zh-cn"` and `"ZH-CN"` both match the `zh-CN` entry).
pub fn find_local_pair(language: &str) -> Option<&'static LocalPair> {
    LOCAL_PAIRS
        .iter()
        .find(|pair| pair.language.eq_ignore_ascii_case(language))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_languages_are_exactly_the_curated_six() {
        let mut bundled: Vec<&str> = LOCAL_PAIRS.iter().map(|pair| pair.language).collect();
        bundled.sort_unstable();
        assert_eq!(bundled, ["da", "es", "ja", "ko", "ru", "zh-CN"]);
    }

    #[test]
    fn bundled_languages_include_korean_chinese_and_japanese() {
        let korean = find_local_pair("ko").expect("Korean must be bundled");
        assert_eq!(korean.model_prefix, "koen");
        assert_eq!(korean.display_name, "Korean");

        let chinese = find_local_pair("zh-CN").expect("Simplified Chinese must be bundled");
        assert_eq!(chinese.model_prefix, "zhen");
        assert_eq!(chinese.display_name, "Chinese (Simplified, Mandarin)");

        let japanese = find_local_pair("ja").expect("Japanese must be bundled");
        assert_eq!(japanese.model_prefix, "jaen");
        assert_eq!(japanese.display_name, "Japanese");

        assert_eq!(find_local_pair("es").unwrap().model_prefix, "esen");
        assert_eq!(find_local_pair("da").unwrap().model_prefix, "daen");
        assert_eq!(find_local_pair("ru").unwrap().model_prefix, "ruen");
    }

    #[test]
    fn lookups_are_case_insensitive() {
        assert_eq!(find_local_pair("KO").map(|pair| pair.language), Some("ko"));
        assert_eq!(
            find_local_pair("Zh-cn").map(|pair| pair.language),
            Some("zh-CN")
        );
        assert_eq!(find_local_pair("JA").map(|pair| pair.language), Some("ja"));
        assert_eq!(find_local_pair("swahili"), None);
    }

    #[test]
    fn every_entry_targets_english_and_is_unique() {
        let mut languages: Vec<&str> = LOCAL_PAIRS.iter().map(|pair| pair.language).collect();
        languages.sort_unstable();
        assert!(
            languages.windows(2).all(|w| w[0] < w[1]),
            "languages must be unique"
        );

        let mut prefixes: Vec<&str> = LOCAL_PAIRS.iter().map(|pair| pair.model_prefix).collect();
        prefixes.sort_unstable();
        assert!(
            prefixes.windows(2).all(|w| w[0] < w[1]),
            "prefixes must be unique"
        );

        for pair in LOCAL_PAIRS {
            assert!(
                pair.model_prefix.ends_with("en"),
                "pair must target English"
            );
            assert!(pair.model_prefix.len() >= 4);
            assert!(!pair.display_name.is_empty());
        }
    }
}
