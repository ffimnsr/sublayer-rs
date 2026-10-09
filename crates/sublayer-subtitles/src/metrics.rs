//! Text measurement for auto-fitting caption cards to the frame.
//!
//! The theme presets are authored for a 1080p reference and scale with the
//! video's height, so a vertical 1080x1920 frame renders them 1.78× larger. A
//! card with several words then grows wider than the frame, and the compiler
//! never wraps cards (`WrapStyle: 2` keeps each animation on one line), so
//! oversized cards would run off both edges. Instead the compiler measures the
//! card against the theme font and shrinks only the cards that would overflow.
//!
//! Measurement uses the actual font file through `ttf-parser`, so wide and
//! narrow glyphs are accounted for. libass sizes glyphs with FreeType's
//! `FT_SIZE_REQUEST_TYPE_REAL_DIM` over the OS/2 Windows metrics, so a style
//! font size of N produces an em of `N * unitsPerEm / (winAscent + winDescent)`
//! — noticeably smaller than N for fonts with large Windows metrics (1.56× for
//! Montserrat). The measurement mirrors that conversion, otherwise "fitted"
//! cards would still render far wider or narrower than measured. When the
//! theme's font file cannot be found (a custom theme, or a font installed
//! system-wide), a conservative per-character estimate keeps the compiler
//! working without libass.

use std::path::{Path, PathBuf};

/// Fraction of the frame width a card may occupy. The style's fixed horizontal
/// margins (2×40 px) plus a safety lead keep the text clear of the edges.
const MAX_WIDTH_FRACTION: f32 = 0.9;

/// Advance width per non-space character (in em) used by the fallback
/// estimator. Montserrat ExtraBold uppercase averages ≈ 0.65 em; 0.72 errs
/// toward shrinking slightly more than needed.
const FALLBACK_GLYPH_EM: f32 = 0.72;

/// Advance width of a space (in em) used by the fallback estimator.
const FALLBACK_SPACE_EM: f32 = 0.28;

/// Floor for the fitted font size, in script pixels; only pathological cards
/// reach it.
const MIN_FONT_SIZE: u32 = 16;

/// Largest text width a card may use at this frame width, in script pixels.
pub(crate) fn max_card_width(frame_width: u32) -> f32 {
    frame_width as f32 * MAX_WIDTH_FRACTION
}

/// Measures card text in the theme font.
pub(crate) struct Measurer<'a> {
    face: Option<ttf_parser::Face<'a>>,
    /// Converts a style font size into the em size libass renders
    /// (`unitsPerEm / (winAscent + winDescent)`); `1.0` without a face.
    size_scale: f32,
}

impl<'a> Measurer<'a> {
    /// Parses `data` as a font; a missing or unparsable font selects the
    /// fallback estimator.
    pub(crate) fn new(data: Option<&'a [u8]>) -> Self {
        let face = data.and_then(|data| ttf_parser::Face::parse(data, 0).ok());
        let size_scale = match (face.as_ref(), data) {
            (Some(face), Some(data)) => real_dim_scale(face, data),
            _ => 1.0,
        };
        Self { face, size_scale }
    }

    /// Largest size ≤ `font_size` at which `text` fits `max_width_px`.
    pub(crate) fn fitted_font_size(&self, text: &str, font_size: u32, max_width_px: f32) -> u32 {
        let width = self.text_width_px(text, font_size as f32);
        if width <= max_width_px || width <= 0.0 {
            return font_size;
        }
        // Floor instead of round: rounding up can leave the card a pixel wider
        // than its budget.
        let fitted = (font_size as f32 * max_width_px / width).floor() as u32;
        fitted.clamp(MIN_FONT_SIZE, font_size)
    }

    /// Width of `text` at `font_size`, by glyph advances when the font is
    /// available and by the fallback estimate otherwise.
    fn text_width_px(&self, text: &str, font_size: f32) -> f32 {
        let Some(face) = self.face.as_ref() else {
            return fallback_width_px(text, font_size);
        };
        let units = f32::from(face.units_per_em());
        if units <= 0.0 {
            return fallback_width_px(text, font_size);
        }
        let em = font_size * self.size_scale;
        let advances: f32 = text
            .chars()
            .map(|c| match face.glyph_index(c) {
                Some(glyph) => f32::from(face.glyph_hor_advance(glyph).unwrap_or(0)),
                // A missing glyph renders as a wide `.notdef` box.
                None => units * FALLBACK_GLYPH_EM,
            })
            .sum();
        advances / units * em
    }
}

/// The em size libass renders for a style font size: FreeType's REAL_DIM sizing
/// targets `winAscent + winDescent` pixels, so the em is
/// `unitsPerEm / (winAscent + winDescent)` per script pixel.
fn real_dim_scale(face: &ttf_parser::Face<'_>, data: &[u8]) -> f32 {
    let units = f32::from(face.units_per_em());
    if units <= 0.0 {
        return 1.0;
    }
    let vertical = win_vertical_metrics(data).unwrap_or_else(|| {
        // Fonts without OS/2 fall back to the face's own ascender/descender,
        // which is what FreeType and therefore libass would use as well.
        (
            face.ascender().max(0) as u16,
            face.descender().unsigned_abs(),
        )
    });
    let sum = u32::from(vertical.0) + u32::from(vertical.1);
    if sum == 0 { 1.0 } else { units / sum as f32 }
}

/// `(usWinAscent, usWinDescent)` from the OS/2 table, when present.
fn win_vertical_metrics(data: &[u8]) -> Option<(u16, u16)> {
    let num_tables = u16::from_be_bytes(data.get(4..6)?.try_into().ok()?) as usize;
    for index in 0..num_tables {
        let entry = data.get(12 + 16 * index..12 + 16 * index + 16)?;
        if &entry[0..4] != b"OS/2" {
            continue;
        }
        let offset = u32::from_be_bytes(entry[8..12].try_into().ok()?) as usize;
        let ascent = u16::from_be_bytes(data.get(offset + 74..offset + 76)?.try_into().ok()?);
        let descent = u16::from_be_bytes(data.get(offset + 76..offset + 78)?.try_into().ok()?);
        return Some((ascent, descent));
    }
    None
}

/// Estimated width of `text` at `font_size` without font metrics.
fn fallback_width_px(text: &str, font_size: f32) -> f32 {
    text.chars()
        .map(|c| {
            if c == ' ' {
                FALLBACK_SPACE_EM
            } else {
                FALLBACK_GLYPH_EM
            }
        })
        .sum::<f32>()
        * font_size
}

/// Reads the data of the font file backing `font_name` from `fonts_dir`.
pub(crate) fn theme_font_data(fonts_dir: &Path, font_name: &str) -> Option<Vec<u8>> {
    std::fs::read(font_file(fonts_dir, font_name)?).ok()
}

/// Picks the font file for `font_name`.
///
/// Names compare with case and separators removed, so `Space Grotesk` matches
/// `SpaceGrotesk.ttf` and `Montserrat` matches `Montserrat-Bold.ttf`. Among
/// matches, the weight nearest the caption styles' bold request wins (`Bold`
/// over `ExtraBold`), which is also what libass selects for `Bold: -1`.
fn font_file(fonts_dir: &Path, font_name: &str) -> Option<PathBuf> {
    let needle = normalize(font_name);
    if needle.is_empty() {
        return None;
    }
    let mut best: Option<(u8, PathBuf)> = None;
    for entry in std::fs::read_dir(fonts_dir).ok()?.flatten() {
        let path = entry.path();
        if !is_font_file(&path) {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let stem = normalize(stem);
        if !stem.starts_with(&needle) {
            continue;
        }
        let rank = weight_rank(&stem);
        if best.as_ref().is_none_or(|(best_rank, _)| rank > *best_rank) {
            best = Some((rank, path));
        }
    }
    best.map(|(_, path)| path)
}

fn is_font_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "ttf" | "otf" | "ttc"
            )
        })
}

/// Weight ranking: plain `Bold` (rank 2) matches the style's `Bold: -1` best,
/// heavier faces (rank 1) second, regular faces last.
fn weight_rank(normalized_stem: &str) -> u8 {
    if normalized_stem.contains("extrabold")
        || normalized_stem.contains("black")
        || normalized_stem.contains("heavy")
    {
        1
    } else if normalized_stem.contains("bold") {
        2
    } else {
        0
    }
}

/// Lowercases and strips separators so font names match file stems.
fn normalize(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_estimate_shrinks_long_cards_and_keeps_short_ones() {
        let measurer = Measurer::new(None);
        let max = max_card_width(1080);
        let long = "WHOO HOO! GOLD NUMBER ONE!";
        let fitted = measurer.fitted_font_size(long, 128, max);
        assert!(fitted < 128, "expected a shrink, got {fitted}");
        assert!(measurer.text_width_px(long, fitted as f32) <= max);
        assert_eq!(measurer.fitted_font_size("YEAH", 128, max), 128);
    }

    #[test]
    fn bundled_fonts_resolve_and_unknown_fonts_do_not() {
        let dir = crate::fonts::resolve_fonts_dir();
        assert!(theme_font_data(&dir, "Montserrat").is_some());
        assert!(theme_font_data(&dir, "Space Grotesk").is_some());
        assert!(theme_font_data(&dir, "Komika Axis").is_some());
        assert!(theme_font_data(&dir, "No Such Font").is_none());
    }

    #[test]
    fn real_font_metrics_shrink_a_wide_card_inside_the_frame() {
        let dir = crate::fonts::resolve_fonts_dir();
        let data = theme_font_data(&dir, "Montserrat").expect("bundled Montserrat");
        let measurer = Measurer::new(Some(&data));
        let max = max_card_width(1080);
        let card = "WHOO HOO! GOLD NUMBER ONE!";
        let fitted = measurer.fitted_font_size(card, 128, max);
        assert!(fitted < 128, "expected a shrink, got {fitted}");
        let width = measurer.text_width_px(card, fitted as f32);
        assert!(width <= max, "{width} exceeds {max}");
        // The floor leaves the card within a percent of the budget.
        assert!(width > max * 0.9, "{width} wastes frame width");
        assert_eq!(measurer.fitted_font_size("GO", 128, max), 128);
    }

    #[test]
    fn libass_real_dim_sizing_is_mirrored() {
        // Montserrat's OS/2 metrics are 1109 + 453 units of a 1000-unit em, so
        // libass renders a style size of 128 at an 81.9 px em, not 128 px.
        let dir = crate::fonts::resolve_fonts_dir();
        let data = theme_font_data(&dir, "Montserrat").expect("bundled Montserrat");
        let measurer = Measurer::new(Some(&data));
        let expected = 128.0 * 1000.0 / (1109.0 + 453.0);
        assert!(
            (measurer.size_scale * 128.0 - expected).abs() < 0.5,
            "{} != {expected}",
            measurer.size_scale * 128.0
        );
    }

    #[test]
    fn empty_text_keeps_the_style_size() {
        let measurer = Measurer::new(None);
        assert_eq!(
            measurer.fitted_font_size("", 128, max_card_width(1080)),
            128
        );
    }
}
