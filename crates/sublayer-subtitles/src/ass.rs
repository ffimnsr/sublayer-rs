//! Advanced SubStation Alpha (`.ass`) compiler with per-word animations.
//!
//! The script is authored at the source video's native resolution; the theme's
//! font size, outline, shadow, and margins are scaled from the 1080p reference
//! the theme presets are tuned for.
//!
//! Per-word animation rides one of two layouts, chosen per theme:
//!
//! * **Karaoke** keeps one `Dialogue` line per card and animates only with
//!   `\k` fills, which libass keys per word (sweep keyed to the word's own
//!   duration; the Secondary Colour carries the highlight).
//! * **WordPop** and **Bounce** need transforms (`\fscx`, `\1c`, …). A
//!   transform override on a shared line changes the fill state for **every
//!   later word**, so the whole caption would scale along with the spoken
//!   word. The compiler instead emits one `Dialogue` event per word, each
//!   anchored with `\an5\pos(x, y)` at the position the word would occupy in
//!   the composed line (computed from the same font metrics the auto-fit
//!   uses). All events span the whole card, so every word of the sentence
//!   stays on screen while the transforms are keyed at the spoken word's
//!   offset — at any moment only that word's event animates: WordPop springs
//!   the word to 115 %, Bounce overshoots through 125 %.
//!
//! Layout: cards are auto-fitted (never wrapped, `WrapStyle: 2`), and pop
//! events reuse the fitted size so both layouts render the same geometry.
//!
//! When `fonts_dir` is non-empty, a `[Fonts]` section pins libass to the
//! bundled font directory so rendering is deterministic across systems.
//!
//! Cards stay on one line (`WrapStyle: 2`), so the compiler measures each card
//! against the theme font and shrinks the font of cards that would overflow the
//! frame; see `crate::metrics`.

use std::path::Path;

use sublayer_core::{AnimationType, CaptionSegment, Rgba, ThemeStyle, VideoMetadata, WordToken};

use crate::SubtitleError;
use crate::metrics::{self, Measurer};

/// Playback resolution used when the video metadata carries no dimensions.
const FALLBACK_RESOLUTION: (u32, u32) = (1920, 1080);

/// The reference height the theme presets are tuned against.
const REFERENCE_HEIGHT: f32 = 1080.0;

/// Horizontal style margin from the `[V4+ Styles]` block (`MarginL/MarginR`),
/// in script pixels; pop-line positioning mirrors it.
const STYLE_MARGIN_H: f32 = 40.0;

/// Style name used by every dialogue line.
const STYLE_NAME: &str = "Caption";

/// Compiles `segments` into a complete ASS script.
///
/// `theme` controls the style block and the per-word animation; `video_meta`
/// provides the playback resolution; `fonts_dir` scopes font lookup (pass an
/// empty path to omit the section and let libass find fonts on the system).
pub fn build_ass_script(
    segments: &[CaptionSegment],
    theme: &ThemeStyle,
    video_meta: &VideoMetadata,
    fonts_dir: &Path,
) -> Result<String, SubtitleError> {
    let (width, height) = if video_meta.width > 0 && video_meta.height > 0 {
        (video_meta.width, video_meta.height)
    } else {
        FALLBACK_RESOLUTION
    };
    let scale = height as f32 / REFERENCE_HEIGHT;
    let font_size = scaled_font_size(theme, scale);
    let font_data = metrics::theme_font_data(fonts_dir, &theme.font_name);
    let measurer = Measurer::new(font_data.as_deref());

    let mut script = String::new();
    script_info(&mut script, width, height);
    style_block(&mut script, theme, scale, font_size);
    if !fonts_dir.as_os_str().is_empty() {
        fonts_section(&mut script, fonts_dir);
    }
    events_header(&mut script);
    for segment in segments {
        dialogue(
            &mut script,
            segment,
            theme,
            &measurer,
            font_size,
            width,
            height,
        );
    }
    Ok(script)
}

/// Theme font size at the script's resolution.
fn scaled_font_size(theme: &ThemeStyle, scale: f32) -> u32 {
    (theme.font_size as f32 * scale).round().max(1.0) as u32
}

/// Calculates the fitted font size for `segment` at the video's resolution,
/// matching the auto-fitting logic applied when compiling the ASS script.
pub fn fitted_font_size(
    segment: &CaptionSegment,
    theme: &ThemeStyle,
    video_meta: &VideoMetadata,
    fonts_dir: &Path,
) -> u32 {
    let (width, height) = if video_meta.width > 0 && video_meta.height > 0 {
        (video_meta.width, video_meta.height)
    } else {
        FALLBACK_RESOLUTION
    };
    let scale = height as f32 / REFERENCE_HEIGHT;
    let font_size = scaled_font_size(theme, scale);
    let font_data = metrics::theme_font_data(fonts_dir, &theme.font_name);
    let measurer = Measurer::new(font_data.as_deref());
    measurer.fitted_font_size(
        &card_text(segment, theme),
        font_size,
        metrics::max_card_width(width),
    )
}

/// Base font size scaled to the video resolution, before auto-fitting.
pub fn scaled_font_size_for(theme: &ThemeStyle, video_meta: &VideoMetadata) -> u32 {
    let height = if video_meta.height > 0 {
        video_meta.height
    } else {
        FALLBACK_RESOLUTION.1
    };
    let scale = height as f32 / REFERENCE_HEIGHT;
    scaled_font_size(theme, scale)
}

/// One measured word of a caption card, in script (video) pixels.
///
/// `x`/`y` are the `\an5` anchors the per-word dialogue events use: `x` is
/// the word's horizontal center on the composed line, `y` the line's vertical
/// center (margin-aware). `width` is the word's advance width and
/// `pill_pad_x` the HighlightBox pill padding (`0.0` for other animations).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WordPlacement {
    /// Word center x, in script pixels.
    pub x: f32,
    /// Line center y, in script pixels.
    pub y: f32,
    /// Word advance width, in script pixels.
    pub width: f32,
    /// HighlightBox pill horizontal padding, in script pixels.
    pub pill_pad_x: f32,
}

/// Measures every word of `segment` with the metrics the ASS compiler uses
/// for its per-word events (`\an5\pos`), honoring `theme`'s alignment,
/// margin, and animation. `None` when the segment has no measurable words.
///
/// The studio preview feeds these values into its overlay so the caption on
/// screen sits exactly where the burned-in render places it.
pub fn place_words(
    segment: &CaptionSegment,
    theme: &ThemeStyle,
    video_meta: &VideoMetadata,
    fonts_dir: &Path,
) -> Option<Vec<WordPlacement>> {
    let (width, height) = if video_meta.width > 0 && video_meta.height > 0 {
        (video_meta.width, video_meta.height)
    } else {
        FALLBACK_RESOLUTION
    };
    let scale = height as f32 / REFERENCE_HEIGHT;
    let font_size = scaled_font_size(theme, scale);
    let font_data = metrics::theme_font_data(fonts_dir, &theme.font_name);
    let measurer = Measurer::new(font_data.as_deref());
    place_words_with(segment, theme, &measurer, font_size, width, height)
        .map(|(_, placements)| placements)
}

/// Shared measurement core of [`place_words`] and the pop event compiler.
///
/// Returns the fitted font size alongside the placements; the compiler needs
/// it for its `\fs` override and pill sizing.
fn place_words_with(
    segment: &CaptionSegment,
    theme: &ThemeStyle,
    measurer: &Measurer<'_>,
    font_size: u32,
    frame_width: u32,
    frame_height: u32,
) -> Option<(u32, Vec<WordPlacement>)> {
    let words: Vec<&WordToken> = segment
        .words
        .iter()
        .filter(|word| !word.text.trim().is_empty())
        .collect();
    if words.is_empty() {
        return None;
    }
    let fitted_size = measurer.fitted_font_size(
        &card_text(segment, theme),
        font_size,
        metrics::max_card_width(frame_width),
    );
    let size = fitted_size as f32;
    let display: Vec<String> = words
        .iter()
        .map(|word| {
            if theme.uppercase {
                word.text.to_uppercase()
            } else {
                word.text.clone()
            }
        })
        .collect();
    let widths: Vec<f32> = display
        .iter()
        .map(|text| measurer.text_width_px(text, size))
        .collect();
    let base_space = measurer.text_width_px(" ", size);
    let space = if theme.animation == AnimationType::HighlightBox {
        (base_space * 1.30).round()
    } else {
        base_space
    };
    let total: f32 = widths.iter().sum::<f32>() + space * (words.len() - 1) as f32;
    let frame_width = frame_width as f32;
    let frame_height = frame_height as f32;

    let line_left = match theme.alignment.clamp(1, 9) {
        1 | 4 | 7 => STYLE_MARGIN_H,
        3 | 6 | 9 => frame_width - STYLE_MARGIN_H - total,
        _ => (frame_width - total) / 2.0,
    };
    let (ascent, descent) = measurer.line_vertical_extent(size);
    let line_height = ascent + descent;
    let margin_v = (theme.margin_v as f32 * (frame_height / REFERENCE_HEIGHT)).round();
    let center_y = match theme.alignment.clamp(1, 9) {
        7..=9 => margin_v + line_height / 2.0,
        4..=6 => frame_height / 2.0,
        _ => frame_height - margin_v - line_height / 2.0,
    };
    let pad_x = if theme.animation == AnimationType::HighlightBox {
        (size * 0.05).clamp(3.0, 7.0)
    } else {
        0.0
    };

    let mut cursor = line_left;
    let placements: Vec<WordPlacement> = widths
        .into_iter()
        .map(|width| {
            let placement = WordPlacement {
                x: cursor + width / 2.0,
                y: center_y,
                width,
                pill_pad_x: pad_x,
            };
            cursor += width + space;
            placement
        })
        .collect();
    Some((fitted_size, placements))
}

/// `[Script Info]` block.
fn script_info(out: &mut String, width: u32, height: u32) {
    out.push_str(
        "[Script Info]\n\
         ScriptType: v4.00+\n\
         WrapStyle: 2\n\
         ScaledBorderAndShadow: yes\n",
    );
    out.push_str(&format!("PlayResX: {width}\nPlayResY: {height}\n\n"));
}

/// `[V4+ Styles]` block with a single `Caption` style.
///
/// The secondary colour carries the karaoke fill; the back colour carries the
/// shadow (and the outline colour when no shadow is used, which keeps the
/// border solid for the pop animations).
fn style_block(out: &mut String, theme: &ThemeStyle, scale: f32, font_size: u32) {
    let outline = theme.outline_width * scale;
    let shadow = theme.shadow * scale;
    let margin_v = (theme.margin_v as f32 * scale).round() as u32;
    let bold = if theme.bold { "-1" } else { "0" };
    let alignment = theme.alignment.clamp(1, 9);
    let outline_color = ass_color(theme.outline_color);

    out.push_str(
        "[V4+ Styles]\n\
         Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n",
    );
    out.push_str(&format!(
        "Style: {STYLE_NAME},{font},{font_size},{primary},{secondary},{outline_color},{outline_color},{bold},0,0,0,100,100,0,0,1,{outline:.1},{shadow:.1},{alignment},40,40,{margin_v},1\n\n",
        font = theme.font_name,
        primary = ass_color(theme.primary_color),
        secondary = ass_color(theme.highlight_color),
    ));
}

/// `[Fonts]` section pinning libass to `fonts_dir`.
fn fonts_section(out: &mut String, fonts_dir: &Path) {
    // Backslashes are rare on Linux but escape them anyway; libass parses the
    // value literally otherwise.
    let dir = fonts_dir.to_string_lossy().replace('\\', "\\\\");
    out.push_str(&format!("[Fonts]\nfontsdir: {dir}\n\n"));
}

/// `[Events]` header.
fn events_header(out: &mut String) {
    out.push_str(
        "[Events]\n\
         Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
    );
}

/// Appends the `Dialogue` line(s) for `segment`, shrinking the font when the
/// card would overflow the frame.
///
/// Karaoke and static cards produce one line; WordPop and Bounce produce one
/// positioned event per word (see the module docs), so their transforms only
/// ever cover the spoken word.
fn dialogue(
    out: &mut String,
    segment: &CaptionSegment,
    theme: &ThemeStyle,
    measurer: &Measurer<'_>,
    font_size: u32,
    frame_width: u32,
    frame_height: u32,
) {
    let Some(first) = segment.start_ms() else {
        return;
    };
    let end = segment.end_ms().unwrap_or(first).max(first);
    let fitted = measurer.fitted_font_size(
        &card_text(segment, theme),
        font_size,
        metrics::max_card_width(frame_width),
    );
    if matches!(
        theme.animation,
        AnimationType::WordPop | AnimationType::Bounce | AnimationType::HighlightBox
    ) {
        pop_dialogues(
            out,
            segment,
            theme,
            measurer,
            font_size,
            frame_width,
            frame_height,
        );
        return;
    }
    out.push_str(&format!(
        "Dialogue: 0,{start},{end},{STYLE_NAME},,0,0,0,,{text}\n",
        start = ass_time(first),
        end = ass_time(end),
        text = dialogue_text(segment, theme, fitted, font_size),
    ));
}

/// Emits one `\an5\pos`-anchored `Dialogue` event per word for the transform-
/// based animations.
///
/// Every event spans the **whole card**, so all words of the sentence stay on
/// screen; word positions come from the same font metrics as the auto-fit,
/// each word sitting where the composed line would place it. The pop
/// transform windows are keyed at the word's offset into the card, so at any
/// moment only the spoken word's event animates — the other events render
/// their words at rest.
#[allow(clippy::too_many_arguments)]
fn pop_dialogues(
    out: &mut String,
    segment: &CaptionSegment,
    theme: &ThemeStyle,
    measurer: &Measurer<'_>,
    font_size: u32,
    frame_width: u32,
    frame_height: u32,
) {
    let Some(card_start) = segment.start_ms() else {
        return;
    };
    let card_end = segment.end_ms().unwrap_or(card_start).max(card_start + 1);
    let words: Vec<&WordToken> = segment
        .words
        .iter()
        .filter(|word| !word.text.trim().is_empty())
        .collect();
    let Some((fitted_size, placements)) = place_words_with(
        segment,
        theme,
        measurer,
        font_size,
        frame_width,
        frame_height,
    ) else {
        return;
    };
    let scale = frame_height as f32 / REFERENCE_HEIGHT;

    // HighlightBox draws an underlying rounded pill behind the active word
    // on Layer 0 during that word's speech window.
    if theme.animation == AnimationType::HighlightBox {
        let pill_h = (fitted_size as f32 * 0.88).round().max(16.0);
        let radius = (pill_h * 0.22).clamp(3.0, 10.0);
        for (word, placement) in words.iter().zip(&placements) {
            let pill_w = (placement.width + placement.pill_pad_x * 2.0).round();
            let draw = pill_drawing(pill_w, pill_h, radius);
            let pill_start = word.start_ms.max(card_start);
            let pill_end = word.end_ms.min(card_end).max(pill_start + 1);

            out.push_str(&format!(
                "Dialogue: 0,{start_time},{end_time},{STYLE_NAME},,0,0,0,,{{\\an5\\pos({x:.1},{y:.1})\\p1\\1c{highlight}&\\3c{highlight}&\\bord0\\shad0\\t(0,50,\\fscx102\\fscy104)\\t(50,140,\\fscx100\\fscy100)}}{draw}{{\\p0}}\n",
                start_time = ass_time(pill_start),
                end_time = ass_time(pill_end),
                x = placement.x,
                y = placement.y,
                highlight = ass_color(theme.highlight_color),
                draw = draw,
            ));
        }
    }

    let layer = if theme.animation == AnimationType::HighlightBox {
        1
    } else {
        0
    };
    for (word, placement) in words.into_iter().zip(&placements) {
        let tags = match theme.animation {
            // Transform windows are keyed at the word's offset into the card
            // (the event clock starts at the card, covering all its words).
            AnimationType::Bounce => bounce_tags(word, card_start, theme),
            AnimationType::HighlightBox => highlight_box_tags(word, card_start, theme, scale),
            _ => pop_tags(word, card_start, theme),
        };
        out.push_str(&format!(
            "Dialogue: {layer},{start_time},{end_time},{STYLE_NAME},,0,0,0,,{{\\an5\\pos({x:.1},{y:.1})}}",
            start_time = ass_time(card_start),
            end_time = ass_time(card_end),
            x = placement.x,
            y = placement.y,
        ));
        if fitted_size < font_size {
            out.push_str(&format!("{{\\fs{fitted_size}}}"));
        }
        out.push_str(&tags);
        out.push_str(&word_text(word, theme));
        out.push('\n');
    }
}

/// The card's visible text, used for width measurement.
fn card_text(segment: &CaptionSegment, theme: &ThemeStyle) -> String {
    let mut text = String::new();
    for word in &segment.words {
        if !text.is_empty() {
            text.push(' ');
        }
        if theme.uppercase {
            text.push_str(&word.text.to_uppercase());
        } else {
            text.push_str(&word.text);
        }
    }
    text
}

/// Builds the animated text of one card.
fn dialogue_text(
    segment: &CaptionSegment,
    theme: &ThemeStyle,
    fitted_size: u32,
    font_size: u32,
) -> String {
    let mut out = String::new();
    // One override before the first word scales the whole card; the per-word
    // animation tags stay relative to whatever size is current.
    if fitted_size < font_size {
        out.push_str(&format!("{{\\fs{fitted_size}}}"));
    }
    for word in &segment.words {
        match theme.animation {
            AnimationType::None => out.push_str(&word_text(word, theme)),
            AnimationType::Karaoke => {
                out.push_str(&format!("{{\\k{}}}", karaoke_cs(word)));
                out.push_str(&word_text(word, theme));
            }
            // Pop and pill families are emitted as per-word events in `pop_dialogues`.
            AnimationType::WordPop | AnimationType::Bounce | AnimationType::HighlightBox => {
                unreachable!("pop and pill cards render as per-word dialogue events")
            }
        }
        out.push(' ');
    }
    out.trim_end().to_owned()
}

/// Escaped word text, uppercased when the theme asks for it.
fn word_text(word: &WordToken, theme: &ThemeStyle) -> String {
    let text = if theme.uppercase {
        word.text.to_uppercase()
    } else {
        word.text.clone()
    };
    escape_ass(&text)
}

/// Karaoke duration in centiseconds, clamped so `{\k0}` never swallows a
/// visible word.
fn karaoke_cs(word: &WordToken) -> u64 {
    (word.duration_ms() / 10).clamp(1, 6000)
}

/// Word-pop override for a per-word event (whose clock starts at the word):
/// the word springs out to 115 % and shifts to the highlight over 80 ms, then
/// settles back to the base scale and colour by +200 ms.
///
/// The event covers exactly one word, so libass's per-line fill state cannot
/// leak the transform into neighbouring words.
fn pop_tags(word: &WordToken, line_start_ms: u64, theme: &ThemeStyle) -> String {
    let start = word.start_ms.saturating_sub(line_start_ms);
    let grow = start + 80;
    let settle = start + 200;
    format!(
        "{{\\1c{base}\\t({start},{grow},\\fscx115\\fscy115\\1c{highlight}&)\\t({grow},{settle},\\fscx100\\fscy100\\1c{base}&)}}",
        highlight = ass_color(theme.highlight_color),
        base = ass_color(theme.primary_color),
    )
}

/// Bounce override for a per-word event: a deep spring — the word jumps to
/// 145 % over 60 ms, dips through 95 %, overshoots to 112 % and settles by
/// +340 ms; the colour rides the first phase up to the highlight and fades
/// back to base during the settle.
///
/// The phases are deliberately longer and deeper than [`pop_tags`]: at video
/// frame rates the pop's 115 % puff reads as a blip, and a 125 % spring with
/// 50 ms phases landed between frames — indistinguishable from the pop.
fn bounce_tags(word: &WordToken, line_start_ms: u64, theme: &ThemeStyle) -> String {
    let start = word.start_ms.saturating_sub(line_start_ms);
    let peak = start + 60;
    let trough = start + 150;
    let overshoot = start + 230;
    let settle = start + 340;
    format!(
        "{{\\1c{base}\\t({start},{peak},\\fscx145\\fscy145\\1c{highlight}&)\\t({peak},{trough},\\fscx95\\fscy95)\\t({trough},{overshoot},\\fscx112\\fscy112)\\t({overshoot},{settle},\\fscx100\\fscy100\\1c{base}&)}}",
        highlight = ass_color(theme.highlight_color),
        base = ass_color(theme.primary_color),
    )
}

/// Highlight box override for a per-word event:
/// When the word is spoken, its text contrasts with the highlight pill (e.g.
/// dark text over bright yellow/green/cyan, or light text over dark colors),
/// hides the stroke outline so text is crisp on the pill, pops with the pill
/// to 108 %, settles back to 100 %, and returns to base primary styling when the
/// word finishes.
fn highlight_box_tags(
    word: &WordToken,
    line_start_ms: u64,
    theme: &ThemeStyle,
    scale: f32,
) -> String {
    let start = word.start_ms.saturating_sub(line_start_ms);
    let grow = start + 50;
    let settle = start + 140;
    let end = word.end_ms.saturating_sub(line_start_ms).max(settle);
    let revert = end + 20;

    let base = ass_color(theme.primary_color);
    let outline = ass_color(theme.outline_color);
    let pill_text = ass_color(pill_text_color(theme));
    let scaled_outline = theme.outline_width * scale;
    let scaled_shadow = theme.shadow * scale;

    format!(
        "{{\\1c{base}&\\3c{outline}&\\bord{scaled_outline:.1}\\shad{scaled_shadow:.1}\\\t({start},{grow},\\1c{pill_text}&\\3c{pill_text}&\\bord0\\shad0\\fscx102\\fscy104)\\\t({grow},{settle},\\fscx100\\fscy100)\\\t({end},{revert},\\1c{base}&\\3c{outline}&\\bord{scaled_outline:.1}\\shad{scaled_shadow:.1})}}"
    )
}

/// Picks a contrasting text color for text drawn over the highlight pill.
fn pill_text_color(theme: &ThemeStyle) -> Rgba {
    let lum = (u32::from(theme.highlight_color.r) * 299
        + u32::from(theme.highlight_color.g) * 587
        + u32::from(theme.highlight_color.b) * 114)
        / 1000;
    if lum >= 128 {
        let outline_lum = (u32::from(theme.outline_color.r) * 299
            + u32::from(theme.outline_color.g) * 587
            + u32::from(theme.outline_color.b) * 114)
            / 1000;
        if outline_lum < 100 {
            theme.outline_color
        } else {
            Rgba::BLACK
        }
    } else {
        Rgba::WHITE
    }
}

/// Generates an ASS vector drawing (`\p1`) path for a rounded rectangle pill.
fn pill_drawing(width: f32, height: f32, radius: f32) -> String {
    let c = 0.552_284_8 * radius;
    format!(
        "m {r:.1} 0 l {w_r:.1} 0 b {cp1_x:.1} 0 {w:.1} {cp1_y:.1} {w:.1} {r:.1}         l {w:.1} {h_r:.1} b {w:.1} {cp2_y:.1} {cp1_x:.1} {h:.1} {w_r:.1} {h:.1}         l {r:.1} {h:.1} b {cp3_x:.1} {h:.1} 0 {cp2_y:.1} 0 {h_r:.1}         l 0 {r:.1} b 0 {cp1_y:.1} {cp3_x:.1} 0 {r:.1} 0",
        r = radius,
        w = width,
        h = height,
        w_r = width - radius,
        h_r = height - radius,
        cp1_x = width - radius + c,
        cp1_y = radius - c,
        cp2_y = height - radius + c,
        cp3_x = radius - c,
    )
}

/// Formats milliseconds as `H:MM:SS.cc` (ASS centisecond timestamps).
pub fn ass_time(ms: u64) -> String {
    let centiseconds = ms % 1000 / 10;
    let total_seconds = ms / 1000;
    let seconds = total_seconds % 60;
    let minutes = total_seconds / 60 % 60;
    let hours = total_seconds / 3600;
    format!("{hours}:{minutes:02}:{seconds:02}.{centiseconds:02}")
}

/// Formats an RGBA color as ASS's `&HAABBGGRR` (alpha 00 = opaque).
pub fn ass_color(color: Rgba) -> String {
    format!(
        "&H{:02X}{:02X}{:02X}{:02X}",
        255 - color.a,
        color.b,
        color.g,
        color.r
    )
}

/// Escapes text for the ASS `Text` field: braces open override blocks and
/// backslashes introduce commands, so all three are backslash-escaped and
/// newlines collapse to spaces (cards render on a single line).
fn escape_ass(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('{', "\\{")
        .replace('}', "\\}")
        .replace(['\n', '\r'], " ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::themes;
    use sublayer_core::{AnimationType, WordToken};

    fn metadata(width: u32, height: u32) -> VideoMetadata {
        VideoMetadata {
            duration_seconds: 10.0,
            width,
            height,
            fps: 30.0,
            video_codec: Some("h264".to_owned()),
            audio_codec: Some("aac".to_owned()),
            audio_channels: Some(2),
            audio_sample_rate: Some(48_000),
        }
    }

    fn segment(words: &[&str]) -> CaptionSegment {
        CaptionSegment::new(
            words
                .iter()
                .enumerate()
                .map(|(index, text)| {
                    WordToken::new(*text, index as u64 * 500, index as u64 * 500 + 400)
                })
                .collect(),
        )
    }

    #[test]
    fn empty_input_still_yields_a_valid_script() {
        let script = build_ass_script(
            &[],
            &themes::tiktok_classic(),
            &metadata(1080, 1920),
            Path::new(""),
        )
        .unwrap();
        assert!(script.contains("[Script Info]"));
        assert!(script.contains("[V4+ Styles]"));
        assert!(script.contains("[Events]"));
        assert!(!script.contains("[Fonts]"));
        assert!(!script.contains("Dialogue:"));
    }

    #[test]
    fn script_info_uses_video_resolution() {
        let script = build_ass_script(
            &[],
            &themes::tiktok_classic(),
            &metadata(1080, 1920),
            Path::new(""),
        )
        .unwrap();
        assert!(script.contains("PlayResX: 1080\nPlayResY: 1920"));
    }

    #[test]
    fn style_block_scales_from_1080p_reference() {
        let script = build_ass_script(
            &[],
            &themes::tiktok_classic(),
            &metadata(540, 960),
            Path::new(""),
        )
        .unwrap();
        // 72 * 960/1080 = 64, outline 6 * 0.888... = 5.3, margin 320 * 0.888 = 284.
        assert!(script.contains("Montserrat,64,&H00FFFFFF"));
        assert!(script.contains(",1,5.3,0.0,2,40,40,284,1"));
    }

    #[test]
    fn fonts_section_is_written_when_dir_given() {
        let script = build_ass_script(
            &[],
            &themes::tiktok_classic(),
            &metadata(1080, 1920),
            Path::new("/opt/sublayer/fonts"),
        )
        .unwrap();
        assert!(script.contains("[Fonts]\nfontsdir: /opt/sublayer/fonts"));
    }

    #[test]
    fn ass_timestamps_are_hms_centiseconds() {
        assert_eq!(ass_time(0), "0:00:00.00");
        assert_eq!(ass_time(61_230), "0:01:01.23");
        assert_eq!(ass_time(3_600_000), "1:00:00.00");
    }

    #[test]
    fn ass_colors_are_alpha_bgr() {
        assert_eq!(ass_color(Rgba::WHITE), "&H00FFFFFF");
        assert_eq!(ass_color(Rgba::BLACK), "&H00000000");
        assert_eq!(ass_color(Rgba::opaque(0xFE, 0x2C, 0x55)), "&H00552CFE");
        assert_eq!(ass_color(Rgba::new(0x12, 0x34, 0x56, 0x78)), "&H87563412");
    }

    #[test]
    fn karaoke_lines_carry_k_fills_and_highlight_secondary() {
        let theme = themes::clean_podcast();
        let script = build_ass_script(
            &[segment(&["Hello", "world!"])],
            &theme,
            &metadata(1080, 1920),
            Path::new(""),
        )
        .unwrap();
        // Words run 0-400 and 500-900 ms; k durations 40 and 40 cs.
        assert!(script.contains(
            "Dialogue: 0,0:00:00.00,0:00:00.90,Caption,,0,0,0,,{\\k40}Hello {\\k40}world!"
        ));
        assert!(script.contains("SecondaryColour, OutlineColour"));
        assert!(script.contains("&H00C1FF8A")); // highlight 0x8AFFC1 -> &HAABBGGRR
    }

    #[test]
    fn word_pop_lines_are_one_event_per_word() {
        let theme = themes::hormozi_bold();
        let script = build_ass_script(
            &[segment(&["Go"])],
            &theme,
            &metadata(1080, 1920),
            Path::new(""),
        )
        .unwrap();
        // Font 76 -> 135 script px; "GO" measures 194.4 px wide (fallback
        // estimator), centered at x=540; the line box centers at
        // 1920 - 533 (scaled margin) - 135/2 = 1319.5. The transforms live on
        // the word's own event, so nothing else can animate along.
        assert!(script.contains(
            "Dialogue: 0,0:00:00.00,0:00:00.40,Caption,,0,0,0,,{\\an5\\pos(540.0,1319.5)}{\\1c&H00FFFFFF\\t(0,80,\\fscx115\\fscy115\\1c&H0000D4FF&)\\t(80,200,\\fscx100\\fscy100\\1c&H00FFFFFF&)}GO"
        ));
    }

    #[test]
    fn pop_words_own_their_dialogue_event() {
        // Regression: a `\t`/`\fscx` override sharing a line with other words
        // animates all of them. Every transformed event must hold exactly one
        // word, anchored by its own `\pos`.
        for theme in [themes::hormozi_bold(), themes::tiktok_classic()] {
            let script = build_ass_script(
                &[segment(&["Hello", "world!"])],
                &theme,
                &metadata(1080, 1920),
                Path::new(""),
            )
            .unwrap();
            let animated: Vec<&str> = script
                .lines()
                .filter(|line| line.starts_with("Dialogue:") && line.contains("\\fscx"))
                .collect();
            assert_eq!(animated.len(), 2, "{theme:?}: one event per word");
            for line in animated {
                assert!(line.contains("\\an5\\pos("), "{theme:?}: positioned");
                let words = ["HELLO", "WORLD!"]
                    .iter()
                    .filter(|word| line.contains(**word))
                    .count();
                assert_eq!(words, 1, "{theme:?}: exactly one word per event: {line}");
            }
            // The card's timed events span the whole card: the sentence
            // stays visible and each word pops inside its own span.
            assert!(
                script.contains("Dialogue: 0,0:00:00.00,0:00:00.90,"),
                "{script}"
            );
        }
    }

    #[test]
    fn bounce_lines_are_one_event_per_word() {
        // Font 72 -> 128 script px; "YES" measures 276.48 px, centered at
        // x=540; the line box centers at 1920 - 569 (scaled margin) - 64.
        let theme = themes::tiktok_classic();
        let script = build_ass_script(
            &[segment(&["Yes"])],
            &theme,
            &metadata(1080, 1920),
            Path::new(""),
        )
        .unwrap();
        assert!(script.contains(
            "{\\an5\\pos(540.0,1287.0)}{\\1c&H00FFFFFF\\t(0,60,\\fscx145\\fscy145\\1c&H00552CFE&)\\t(60,150,\\fscx95\\fscy95)\\t(150,230,\\fscx112\\fscy112)\\t(230,340,\\fscx100\\fscy100\\1c&H00FFFFFF&)}YES"
        ));
    }

    #[test]
    fn static_themes_render_plain_uppercased_text() {
        let theme = themes::minimal_cinematic();
        let script = build_ass_script(
            &[segment(&["soft", "light"])],
            &theme,
            &metadata(1080, 1920),
            Path::new(""),
        )
        .unwrap();
        // Sentence case: no uppercase, no override tags.
        assert!(script.contains("Dialogue: 0,0:00:00.00,0:00:00.90,Caption,,0,0,0,,soft light"));

        let ghost = ThemeStyle {
            animation: AnimationType::None,
            uppercase: true,
            ..themes::clean_podcast()
        };
        let script = build_ass_script(
            &[segment(&["soft", "light"])],
            &ghost,
            &metadata(1080, 1920),
            Path::new(""),
        )
        .unwrap();
        assert!(script.contains(",,SOFT LIGHT"));
    }

    #[test]
    fn ass_text_escapes_braces_and_backslashes() {
        let words = vec![WordToken::new("a{b} \\c", 0, 100)];
        let segments = vec![CaptionSegment::new(words)];
        let script = build_ass_script(
            &segments,
            &themes::clean_podcast(),
            &metadata(1080, 1920),
            Path::new(""),
        )
        .unwrap();
        assert!(script.contains("a\\{b\\} \\\\c"));
    }

    #[test]
    fn long_cards_shrink_to_fit_the_frame() {
        let theme = themes::tiktok_classic();
        let script = build_ass_script(
            &[segment(&["Whos", "hoo", "gold", "number", "one"])],
            &theme,
            &metadata(1080, 1920),
            Path::new(""),
        )
        .unwrap();
        // Bounce renders per-word events; the first word's event carries the
        // fitted `\fs` (the 128 px style font would overflow the frame).
        let line = script
            .lines()
            .find(|line| line.starts_with("Dialogue:") && line.contains("WHOS"))
            .expect("dialogue line");
        let (_, text) = line.rsplit_once(",,").unwrap();
        assert!(text.contains("{\\fs"), "{text}");
        let size: u32 = text
            .split("{\\fs")
            .nth(1)
            .and_then(|rest| rest.split('}').next())
            .and_then(|value| value.parse().ok())
            .unwrap();
        assert!((16..128).contains(&size), "fitted to {size}");
    }

    #[test]
    fn short_cards_keep_the_style_font_size() {
        let theme = themes::tiktok_classic();
        let script = build_ass_script(
            &[segment(&["Yes"])],
            &theme,
            &metadata(1080, 1920),
            Path::new(""),
        )
        .unwrap();
        let line = script
            .lines()
            .find(|line| line.starts_with("Dialogue:") && line.contains("YES"))
            .unwrap();
        let (_, text) = line.rsplit_once(",,").unwrap();
        assert!(!text.contains("{\\fs"), "{text}");
    }

    #[test]
    fn highlight_box_emits_layer_0_pill_and_layer_1_text() {
        let theme = ThemeStyle {
            animation: AnimationType::HighlightBox,
            ..themes::hormozi_bold()
        };
        let script = build_ass_script(
            &[segment(&["Hormozi", "style"])],
            &theme,
            &metadata(1080, 1920),
            Path::new(""),
        )
        .unwrap();

        // Layer 0: pill vector drawing for each word
        let pills: Vec<&str> = script
            .lines()
            .filter(|line| line.starts_with("Dialogue: 0,") && line.contains(r"\p1"))
            .collect();
        assert_eq!(pills.len(), 2, "one pill drawing per word on Layer 0");
        for pill in pills {
            assert!(pill.contains(r"\bord0\shad0"));
            assert!(pill.contains(r"\fscx102\fscy104"));
            assert!(pill.contains("m "));
            assert!(pill.contains("b "));
            assert!(pill.contains(r"{\p0}"));
        }

        // Layer 1: positioned text for each word
        let text_events: Vec<&str> = script
            .lines()
            .filter(|line| line.starts_with("Dialogue: 1,"))
            .collect();
        assert_eq!(text_events.len(), 2, "one text event per word on Layer 1");
        for line in text_events {
            assert!(line.contains(r"\an5\pos("));
            assert!(line.contains(r"\fscx102\fscy104"));
            // Hormozi yellow is bright, so text inside the pill contrasts with dark/black
            assert!(line.contains(&ass_color(Rgba::BLACK)));
        }
    }

    #[test]
    fn pill_text_color_adapts_to_luminance() {
        // Bright highlight: yellow -> black text
        let bright = themes::hormozi_bold();
        assert_eq!(pill_text_color(&bright), Rgba::BLACK);

        // Dark highlight: dark navy (#001133) -> white text
        let mut dark = themes::hormozi_bold();
        dark.highlight_color = Rgba::opaque(0x00, 0x11, 0x33);
        assert_eq!(pill_text_color(&dark), Rgba::WHITE);
    }

    #[test]
    fn zero_duration_words_still_get_visible_karaoke() {
        let words = vec![WordToken::new("hi", 500, 500)];
        let segments = vec![CaptionSegment::new(words)];
        let script = build_ass_script(
            &segments,
            &themes::clean_podcast(),
            &metadata(1080, 1920),
            Path::new(""),
        )
        .unwrap();
        assert!(script.contains("{\\k1}hi"));
    }

    #[test]
    fn alignment_affects_dialogue_positions_in_rendered_ass() {
        for (alignment, expected_y_cmp, expected_x_cmp) in [
            (7, "top", "left"),
            (8, "top", "center"),
            (9, "top", "right"),
            (4, "middle", "left"),
            (5, "middle", "center"),
            (6, "middle", "right"),
            (1, "bottom", "left"),
            (2, "bottom", "center"),
            (3, "bottom", "right"),
        ] {
            let mut theme = themes::hormozi_bold();
            theme.alignment = alignment;
            let script = build_ass_script(
                &[segment(&["Hello"])],
                &theme,
                &metadata(1080, 1920),
                Path::new(""),
            )
            .unwrap();
            let pos_line = script.lines().find(|l| l.contains(r"\pos(")).unwrap();
            let start = pos_line.find(r"\pos(").unwrap() + 5;
            let end = pos_line[start..].find(")").unwrap() + start;
            let parts: Vec<f32> = pos_line[start..end]
                .split(",")
                .map(|s| s.parse().unwrap())
                .collect();
            let (x, y) = (parts[0], parts[1]);

            match expected_y_cmp {
                "top" => assert!(
                    y < 700.0,
                    "top alignment {alignment} must place y near top: {y}"
                ),
                "middle" => assert!(
                    (y - 960.0).abs() < 50.0,
                    "middle alignment {alignment} must place y near 960: {y}"
                ),
                "bottom" => assert!(
                    y > 1200.0,
                    "bottom alignment {alignment} must place y near bottom: {y}"
                ),
                _ => unreachable!(),
            }
            match expected_x_cmp {
                "left" => assert!(
                    x < 400.0,
                    "left alignment {alignment} must place first word near left: {x}"
                ),
                "center" => assert!(
                    (x - 450.0).abs() < 100.0,
                    "center alignment {alignment} must place first word near center: {x}"
                ),
                "right" => assert!(
                    x > 500.0,
                    "right alignment {alignment} must place first word toward right: {x}"
                ),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn alignment_affects_style_block_for_karaoke_and_none() {
        let mut theme = themes::clean_podcast();
        theme.alignment = 7;
        let script = build_ass_script(&[], &theme, &metadata(1080, 1920), Path::new("")).unwrap();
        assert!(script.contains(",7,40,40,"));
    }

    /// `place_words` must hand the studio preview exactly the `\an5\pos`
    /// anchors the compiler emits for the same card, so the on-screen caption
    /// and the burned-in render share one geometry.
    #[test]
    fn place_words_match_emitted_dialogue_positions() {
        let fonts_dir = crate::fonts::resolve_fonts_dir();
        let segment = segment(&["Hello", "World"]);
        for alignment in 1..=9u8 {
            for animation in [
                AnimationType::WordPop,
                AnimationType::Bounce,
                AnimationType::HighlightBox,
            ] {
                let mut theme = themes::hormozi_bold();
                theme.alignment = alignment;
                theme.animation = animation;
                let meta = metadata(1080, 1920);
                let placements = place_words(&segment, &theme, &meta, &fonts_dir).expect("words");
                let script =
                    build_ass_script(std::slice::from_ref(&segment), &theme, &meta, &fonts_dir)
                        .unwrap();

                let positions: Vec<(f32, f32)> = script
                    .lines()
                    .filter(|line| line.contains(r"\pos("))
                    .map(|line| {
                        let start = line.find(r"\pos(").unwrap() + 5;
                        let end = line[start..].find(")").unwrap() + start;
                        let parts: Vec<f32> = line[start..end]
                            .split(',')
                            .map(|value| value.parse().unwrap())
                            .collect();
                        (parts[0], parts[1])
                    })
                    .collect();

                // HighlightBox also emits one pill event per word with the
                // same anchors, ahead of the text events.
                let expected = match animation {
                    AnimationType::HighlightBox => placements
                        .iter()
                        .chain(placements.iter())
                        .map(|p| (p.x, p.y))
                        .collect::<Vec<_>>(),
                    _ => placements.iter().map(|p| (p.x, p.y)).collect(),
                };
                assert_eq!(positions.len(), expected.len(), "alignment {alignment}");
                for (index, ((x, y), (ex, ey))) in positions.into_iter().zip(expected).enumerate() {
                    assert!(
                        (x - ex).abs() < 0.1 && (y - ey).abs() < 0.1,
                        "alignment {alignment} {animation:?} word {index}: \
                         \\pos({x:.1},{y:.1}) vs placement ({ex:.1},{ey:.1})"
                    );
                }
            }
        }
    }
}
