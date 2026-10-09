//! Advanced SubStation Alpha (`.ass`) compiler with per-word animations.
//!
//! The script is authored at the source video's native resolution; the theme's
//! font size, outline, shadow, and margins are scaled from the 1080p reference
//! the theme presets are tuned for.
//!
//! Animations reuse whisper's own word timings, shifted to the card's local
//! timeline — libass applies each override block the moment its clock reaches
//! the word's start, so the pop/bounce lands on the spoken word even though
//! the whole card is one `Dialogue` line.
//!
//! Per-word animation is safe only because that line uses `\k`-style karaoke
//! tags, which are keyed per word. Transform overrides (`\fscx`, `\1c`, …)
//! ride the line's fill state, so during any word's pop window **every
//! later word would animate along** — the whole caption visibly scaling
//! together. The pop families therefore highlight the spoken word instead
//! of scaling it:
//!
//! * **Karaoke** — `{\k}` sweep keyed to the word duration; the Secondary
//!   Colour (set to the theme's highlight) fills over the base colour.
//! * **WordPop** — `{\K}` instant fill: the spoken word snaps into the
//!   highlight for exactly its duration and snaps back.
//! * **Bounce** — `{\k}` sweep like karaoke; kept as a preset-pickable
//!   variant that does not scale the line.
//!
//! Layout: the card is auto-fitted (never wrapped, `WrapStyle: 2`) so the
//! whole line lands on one baseline; karaoke fills change no geometry.
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
    let card_width = metrics::max_card_width(width);

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
            card_width,
        );
    }
    Ok(script)
}

/// Theme font size at the script's resolution.
fn scaled_font_size(theme: &ThemeStyle, scale: f32) -> u32 {
    (theme.font_size as f32 * scale).round().max(1.0) as u32
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

/// Appends one `Dialogue` line for `segment`, shrinking the font when the card
/// would overflow the frame.
fn dialogue(
    out: &mut String,
    segment: &CaptionSegment,
    theme: &ThemeStyle,
    measurer: &Measurer<'_>,
    font_size: u32,
    card_width: f32,
) {
    let Some(first) = segment.start_ms() else {
        return;
    };
    let end = segment.end_ms().unwrap_or(first).max(first);
    let fitted = measurer.fitted_font_size(&card_text(segment, theme), font_size, card_width);
    out.push_str(&format!(
        "Dialogue: 0,{start},{end},{STYLE_NAME},,0,0,0,,{text}\n",
        start = ass_time(first),
        end = ass_time(end),
        text = dialogue_text(segment, theme, fitted, font_size),
    ));
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
            AnimationType::WordPop => {
                out.push_str(&format!("{{\\K{}}}", karaoke_cs(word)));
                out.push_str(&word_text(word, theme));
            }
            AnimationType::Bounce => {
                out.push_str(&format!("{{\\k{}}}", karaoke_cs(word)));
                out.push_str(&word_text(word, theme));
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
    fn word_pop_lines_highlight_the_spoken_word() {
        let theme = themes::hormozi_bold();
        let script = build_ass_script(
            &[segment(&["Go"])],
            &theme,
            &metadata(1080, 1920),
            Path::new(""),
        )
        .unwrap();
        // The 400 ms word gets an instant `\K` fill of 40 cs: the word snaps
        // into the highlight (the style's SecondaryColour) for its duration.
        assert!(script.contains("Dialogue: 0,0:00:00.00,0:00:00.40,Caption,,0,0,0,,{\\K40}GO"));
    }

    #[test]
    fn pop_lines_never_scale_the_whole_caption() {
        // Regression: a `\t`/`\fscx` override on one word of a single
        // Dialogue line animates every later word along with it, so the whole
        // caption grew in unison. Pop families must stay fill-only.
        for theme in [themes::hormozi_bold(), themes::tiktok_classic()] {
            let script = build_ass_script(
                &[segment(&["Hello", "world!"])],
                &theme,
                &metadata(1080, 1920),
                Path::new(""),
            )
            .unwrap();
            assert!(!script.contains(r"\fscx"), "{theme:?}: no scale overrides");
            assert!(!script.contains(r"\t("), "{theme:?}: no animated overrides");
            let fill = if theme.animation == AnimationType::WordPop {
                "{\\K40}HELLO {\\K40}WORLD!"
            } else {
                "{\\k40}HELLO {\\k40}WORLD!"
            };
            assert!(script.contains(fill), "{theme:?}: per-word fill");
        }
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
        let line = script
            .lines()
            .find(|line| line.starts_with("Dialogue:"))
            .expect("dialogue line");
        let (_, text) = line.rsplit_once(",,").unwrap();
        // The style font is 72 * 1920/1080 = 128; the card is too wide for the
        // 1080 px frame and must carry a smaller `\fs` override.
        assert!(text.starts_with("{\\fs"), "{text}");
        let size: u32 = text[4..].split('}').next().unwrap().parse().unwrap();
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
            .find(|line| line.starts_with("Dialogue:"))
            .unwrap();
        let (_, text) = line.rsplit_once(",,").unwrap();
        assert!(!text.starts_with("{\\fs"), "{text}");
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
}
