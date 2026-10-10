//! Built-in theme presets and custom-theme (de)serialization.
//!
//! Presets cover the mainstream short-form caption aesthetics. All fonts
//! referenced here ship in `assets/fonts/` except when a system font of the
//! same family exists; libass falls back to a default font when neither is
//! available, so a missing font degrades gracefully.

use std::path::Path;

use sublayer_core::{AnimationType, Rgba, ThemeStyle};

use crate::SubtitleError;

/// TikTok Classic — the archetypal short-form look: white extra-bold caps
/// with a heavy black outline, a brand-pink highlight, and a springy bounce.
pub fn tiktok_classic() -> ThemeStyle {
    ThemeStyle {
        name: "TikTok Classic".to_owned(),
        font_name: "Montserrat".to_owned(),
        font_size: 72,
        primary_color: Rgba::WHITE,
        highlight_color: Rgba::opaque(0xFE, 0x2C, 0x55),
        outline_color: Rgba::BLACK,
        outline_width: 6.0,
        shadow: 0.0,
        bold: true,
        uppercase: true,
        alignment: 2,
        margin_v: 320,
        animation: AnimationType::Bounce,
    }
}

/// Hormozi Bold — high-contrast white-on-black with the signature yellow
/// accent on the spoken word.
pub fn hormozi_bold() -> ThemeStyle {
    ThemeStyle {
        name: "Hormozi Bold".to_owned(),
        font_name: "Montserrat".to_owned(),
        font_size: 76,
        primary_color: Rgba::WHITE,
        highlight_color: Rgba::opaque(0xFF, 0xD4, 0x00),
        outline_color: Rgba::BLACK,
        outline_width: 5.0,
        shadow: 0.0,
        bold: true,
        uppercase: true,
        alignment: 2,
        margin_v: 300,
        animation: AnimationType::WordPop,
    }
}

/// Clean Podcast — understated top-aligned cards with a soft karaoke sweep,
/// aimed at talking-head and interview footage.
pub fn clean_podcast() -> ThemeStyle {
    ThemeStyle {
        name: "Clean Podcast".to_owned(),
        font_name: "Inter".to_owned(),
        font_size: 52,
        primary_color: Rgba::WHITE,
        highlight_color: Rgba::opaque(0x8A, 0xFF, 0xC1),
        outline_color: Rgba::opaque(0x00, 0x00, 0x00),
        outline_width: 2.0,
        shadow: 1.0,
        bold: false,
        uppercase: false,
        alignment: 8,
        margin_v: 220,
        animation: AnimationType::Karaoke,
    }
}

/// Cyber Gaming — neon teal on black with a magenta active word and a soft
/// glow shadow, for gaming and tech content.
pub fn cyber_gaming() -> ThemeStyle {
    ThemeStyle {
        name: "Cyber Gaming".to_owned(),
        font_name: "Space Grotesk".to_owned(),
        font_size: 64,
        primary_color: Rgba::opaque(0x00, 0xFF, 0xF0),
        highlight_color: Rgba::opaque(0xFF, 0x00, 0xE5),
        outline_color: Rgba::BLACK,
        outline_width: 3.0,
        shadow: 2.5,
        bold: true,
        uppercase: true,
        alignment: 2,
        margin_v: 320,
        animation: AnimationType::Karaoke,
    }
}

/// Minimal Cinematic — quiet sentence case with no outline and a whisper-soft
/// shadow, letting the footage carry the look.
pub fn minimal_cinematic() -> ThemeStyle {
    ThemeStyle {
        name: "Minimal Cinematic".to_owned(),
        font_name: "Inter".to_owned(),
        font_size: 56,
        primary_color: Rgba::WHITE,
        highlight_color: Rgba::opaque(0xFF, 0xC8, 0x5C),
        outline_color: Rgba::BLACK,
        outline_width: 0.0,
        shadow: 1.5,
        bold: false,
        uppercase: false,
        alignment: 2,
        margin_v: 260,
        animation: AnimationType::None,
    }
}

/// Printable names of every built-in preset, in display order.
pub const PRESET_NAMES: &[&str] = &[
    "TikTok Classic",
    "Hormozi Bold",
    "Clean Podcast",
    "Cyber Gaming",
    "Minimal Cinematic",
];

/// Iterator over the printable preset names.
pub fn preset_names() -> impl Iterator<Item = &'static str> {
    PRESET_NAMES.iter().copied()
}

/// Looks up a preset by its full name or a case-insensitive alias
/// (`"tiktok"`, `"hormozi"`, `"podcast"`, `"cyber"`, `"cinematic"`, ...).
pub fn preset(name: &str) -> Option<ThemeStyle> {
    Some(match name.to_ascii_lowercase().as_str() {
        "tiktok" | "tiktok-classic" | "tiktok classic" => tiktok_classic(),
        "hormozi" | "hormozi-bold" | "hormozi bold" => hormozi_bold(),
        "podcast" | "clean-podcast" | "clean podcast" => clean_podcast(),
        "cyber" | "gaming" | "cyber-gaming" | "cyber gaming" => cyber_gaming(),
        "minimal" | "cinematic" | "minimal-cinematic" | "minimal cinematic" => minimal_cinematic(),
        _ => return None,
    })
}

/// Serializes a theme as JSON; round-trips through [`from_json`].
pub fn to_json(theme: &ThemeStyle) -> Result<String, SubtitleError> {
    Ok(serde_json::to_string_pretty(theme)?)
}

/// Parses a custom theme from JSON. Colors must be `#RRGGBB` or `#RRGGBBAA`.
pub fn from_json(json: &str) -> Result<ThemeStyle, SubtitleError> {
    Ok(serde_json::from_str(json)?)
}

/// Loads a custom theme from a JSON file.
pub fn load_json_file(path: &Path) -> Result<ThemeStyle, SubtitleError> {
    let json = std::fs::read_to_string(path)?;
    from_json(&json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_distinct_and_well_formed() {
        let names: Vec<&str> = preset_names().collect();
        assert_eq!(names.len(), 5);
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), names.len(), "preset names must be unique");

        for name in names {
            let theme = preset(name).expect("name must resolve");
            assert!(!theme.font_name.is_empty());
            assert!(theme.font_size > 0);
            assert!((1..=9).contains(&theme.alignment));
        }
    }

    #[test]
    fn preset_lookup_is_case_insensitive_and_aliased() {
        assert_eq!(preset("tiktok"), Some(tiktok_classic()));
        assert_eq!(preset("TIKTOK CLASSIC"), Some(tiktok_classic()));
        assert_eq!(preset("hormozi"), Some(hormozi_bold()));
        assert_eq!(preset("podcast"), Some(clean_podcast()));
        assert_eq!(preset("cyber"), Some(cyber_gaming()));
        assert_eq!(preset("cinematic"), Some(minimal_cinematic()));
        assert_eq!(preset("nope"), None);
    }

    #[test]
    fn theme_json_roundtrips() {
        let json = to_json(&tiktok_classic()).unwrap();
        let theme = from_json(&json).unwrap();
        assert_eq!(theme, tiktok_classic());
    }

    #[test]
    fn theme_json_rejects_bad_colors() {
        let json = r##"{"name":"X","font_name":"M","font_size":10,
            "primary_color":"#12345","highlight_color":"#FF0000",
            "outline_color":"#000000","outline_width":1.0,"shadow":0.0,
            "bold":true,"uppercase":true,"alignment":2,"margin_v":10,
            "animation":"none"}"##;
        assert!(from_json(json).is_err());
    }

    #[test]
    fn theme_json_accepts_alpha_colors() {
        let json = r##"{"name":"X","font_name":"M","font_size":10,
            "primary_color":"#FFFFFFFF","highlight_color":"#FF0000",
            "outline_color":"#000000","outline_width":1.0,"shadow":0.0,
            "bold":true,"uppercase":true,"alignment":2,"margin_v":10,
            "animation":"word_pop"}"##;
        let theme = from_json(json).unwrap();
        assert_eq!(theme.primary_color.a, 0xFF);
    }

    #[test]
    fn theme_json_accepts_highlight_box_and_pill() {
        let json = r##"{"name":"Custom","font_name":"Montserrat","font_size":60,
            "primary_color":"#FFFFFF","highlight_color":"#FFD400",
            "outline_color":"#000000","outline_width":4.0,"shadow":0.0,
            "bold":true,"uppercase":true,"alignment":2,"margin_v":200,
            "animation":"highlight_box"}"##;
        let theme = from_json(json).unwrap();
        assert_eq!(theme.animation, AnimationType::HighlightBox);

        let json_pill = r##"{"name":"Custom","font_name":"Montserrat","font_size":60,
            "primary_color":"#FFFFFF","highlight_color":"#FFD400",
            "outline_color":"#000000","outline_width":4.0,"shadow":0.0,
            "bold":true,"uppercase":true,"alignment":2,"margin_v":200,
            "animation":"highlight_pill"}"##;
        let theme_pill = from_json(json_pill).unwrap();
        assert_eq!(theme_pill.animation, AnimationType::HighlightBox);
    }

    #[test]
    fn unknown_animation_is_rejected() {
        let json = r##"{"name":"X","font_name":"M","font_size":10,
            "primary_color":"#FFFFFF","highlight_color":"#FF0000",
            "outline_color":"#000000","outline_width":1.0,"shadow":0.0,
            "bold":true,"uppercase":true,"alignment":2,"margin_v":10,
            "animation":"flip"}"##;
        assert!(from_json(json).is_err());
    }
}
