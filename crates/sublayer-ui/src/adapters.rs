//! Conversions between the domain models and the Slint models/properties.
//!
//! All formatting and clamping rules live here so the Slint files stay
//! declarative and the values pushed into them are always in range.

use slint::{Image, SharedPixelBuffer, SharedString};
use sublayer_core::{AnimationType, CaptionSegment, Project, Rgba, ThemeStyle};
use sublayer_media::{RgbaFrame, WaveformCache};
use sublayer_subtitles::PRESET_NAMES;

use crate::{BucketItem, CaptionRow, CaptionWord, SegmentItem, ThemeData};

/// Caption animations in `AnimationType` declaration order, matching the
/// inspector dropdown.
const ANIMATIONS: [AnimationType; 4] = [
    AnimationType::None,
    AnimationType::WordPop,
    AnimationType::Karaoke,
    AnimationType::Bounce,
];

/// Builds the inspector snapshot for `theme`.
pub fn theme_data_from(theme: &ThemeStyle) -> ThemeData {
    ThemeData {
        preset_index: preset_index_of(&theme.name),
        font_size: theme.font_size.min(i32::MAX as u32) as i32,
        primary_hex: color_hex(theme.primary_color),
        highlight_hex: color_hex(theme.highlight_color),
        outline_hex: color_hex(theme.outline_color),
        outline_width: theme.outline_width,
        shadow: theme.shadow,
        margin_v: theme.margin_v.min(i32::MAX as u32) as i32,
        bold: theme.bold,
        uppercase: theme.uppercase,
        alignment_index: i32::from(theme.alignment.clamp(1, 9)) - 1,
        animation_index: ANIMATIONS
            .iter()
            .position(|candidate| *candidate == theme.animation)
            .unwrap_or(0) as i32,
    }
}

/// Merges an inspector snapshot into `theme`.
///
/// Unparseable colors keep their previous value, numeric fields are clamped,
/// and the preset selection is *not* applied here: switching presets replaces
/// the whole style, which the app handles before merging.
pub fn apply_theme_data(data: &ThemeData, theme: &mut ThemeStyle) {
    if let Some(color) = parse_color(&data.primary_hex) {
        theme.primary_color = color;
    }
    if let Some(color) = parse_color(&data.highlight_hex) {
        theme.highlight_color = color;
    }
    if let Some(color) = parse_color(&data.outline_hex) {
        theme.outline_color = color;
    }
    theme.font_size = u32::try_from(data.font_size)
        .unwrap_or(theme.font_size)
        .clamp(16, 200);
    theme.font_size = data.font_size.clamp(16, 200) as u32;
    theme.outline_width = data.outline_width.clamp(0.0, 16.0);
    theme.shadow = data.shadow.clamp(0.0, 10.0);
    theme.margin_v = data.margin_v.clamp(0, 1_200) as u32;
    theme.bold = data.bold;
    theme.uppercase = data.uppercase;
    theme.alignment = u8::try_from(data.alignment_index.clamp(0, 8)).unwrap_or(0) + 1;
    theme.animation = ANIMATIONS
        .get(usize::try_from(data.animation_index).unwrap_or(0))
        .copied()
        .unwrap_or(AnimationType::None);
}

/// Index of the preset a theme name refers to; unknown names map to the first
/// preset.
pub fn preset_index_of(name: &str) -> i32 {
    let lowered = name.to_ascii_lowercase();
    PRESET_NAMES
        .iter()
        .position(|preset| {
            let preset = preset.to_ascii_lowercase();
            preset == lowered || lowered.contains(&preset)
        })
        .unwrap_or(0) as i32
}

/// Resolves the preset at `index`, including the alias spelling used by
/// [`preset_index_of`].
pub fn preset_by_index(index: i32) -> Option<ThemeStyle> {
    let name = PRESET_NAMES.get(usize::try_from(index).ok()?)?;
    sublayer_subtitles::preset(name)
}

/// Formats a color as uppercase `#RRGGBBAA`, the format the hex fields accept.
pub fn color_hex(color: Rgba) -> SharedString {
    color.to_hex().to_uppercase().into()
}

/// Parses a hex color field, tolerating surrounding whitespace.
pub fn parse_color(text: &str) -> Option<Rgba> {
    Rgba::from_hex(text.trim()).ok()
}

/// Builds the timeline rows of caption cards.
pub fn segment_items(segments: &[CaptionSegment], selected: Option<usize>) -> Vec<SegmentItem> {
    segments
        .iter()
        .enumerate()
        .map(|(index, segment)| SegmentItem {
            index: index as i32,
            start_ms: clamp_ms(segment.start_ms().unwrap_or(0)),
            end_ms: clamp_ms(segment.end_ms().unwrap_or(0)),
            text: segment.text().into(),
            selected: selected == Some(index),
        })
        .collect()
}

/// Builds the rows of the captions drawer, in timeline order.
pub fn caption_row_items(segments: &[CaptionSegment], selected: Option<usize>) -> Vec<CaptionRow> {
    segments
        .iter()
        .enumerate()
        .map(|(index, segment)| CaptionRow {
            index: index as i32,
            label: format!(
                "{} → {}",
                format_timecode(segment.start_ms().unwrap_or(0)),
                format_timecode(segment.end_ms().unwrap_or(0))
            )
            .into(),
            text: segment.text().into(),
            selected: selected == Some(index),
        })
        .collect()
}

/// The word under `playhead_ms` of the caption at that time, with every word
/// of the card; `active` marks the spoken one.
pub fn caption_words_at(project: &Project, playhead_ms: u64) -> Option<Vec<CaptionWord>> {
    let segment = project.segments.iter().find(|segment| {
        let start = segment.start_ms().unwrap_or(0);
        let end = segment.end_ms().unwrap_or(start).max(start + 1);
        start <= playhead_ms && playhead_ms < end
    })?;
    Some(
        segment
            .words
            .iter()
            .enumerate()
            .map(|(index, word)| CaptionWord {
                index: index as i32,
                text: word.text.clone().into(),
                active: word.start_ms <= playhead_ms
                    && playhead_ms < word.end_ms.max(word.start_ms + 1),
            })
            .collect(),
    )
}

/// Aggregates `cache` into at most `columns` render columns starting at
/// `base_ms` (which must be aligned to the column grid).
pub fn waveform_window(
    cache: &WaveformCache,
    base_ms: u64,
    column_ms: u64,
    columns: usize,
) -> Vec<BucketItem> {
    if column_ms == 0 {
        return Vec::new();
    }
    let buckets_per_sec = u64::from(cache.samples_per_sec());
    let all = cache.buckets();
    let total = all.len() as u64;

    (0..columns as u64)
        .map(|column| {
            let start_ms = base_ms + column * column_ms;
            let end_ms = start_ms + column_ms;
            let first = (start_ms.saturating_mul(buckets_per_sec) / 1_000).min(total) as usize;
            let last = (end_ms.saturating_mul(buckets_per_sec) / 1_000).min(total) as usize;
            if last <= first {
                return BucketItem {
                    low: 0.0,
                    high: 0.0,
                    rms: 0.0,
                };
            }
            let slice = &all[first..last];
            let mut low = 0.0_f32;
            let mut high = 0.0_f32;
            let mut sum_of_squares = 0.0_f64;
            for bucket in slice {
                low = low.max(-bucket.min_amplitude);
                high = high.max(bucket.max_amplitude);
                sum_of_squares += f64::from(bucket.rms) * f64::from(bucket.rms);
            }
            let rms = (sum_of_squares / slice.len() as f64).sqrt() as f32;
            BucketItem {
                low: low.clamp(0.0, 1.0),
                high: high.clamp(0.0, 1.0),
                rms: rms.clamp(0.0, 1.0),
            }
        })
        .collect()
}

/// Converts a decoded RGBA frame into a Slint image.
pub fn frame_to_image(frame: &RgbaFrame) -> Image {
    let buffer = SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(
        &frame.pixels,
        frame.width,
        frame.height,
    );
    Image::from_rgba8(buffer)
}

/// Caption card displayed by the viewport at `playhead_ms`.
pub fn caption_at(project: &Project, playhead_ms: u64) -> Option<SharedString> {
    project
        .segments
        .iter()
        .find(|segment| {
            let start = segment.start_ms().unwrap_or(0);
            let end = segment.end_ms().unwrap_or(0);
            start <= playhead_ms && playhead_ms < end
        })
        .map(|segment| segment.text().into())
}

/// One-line description of a card for the inspector header.
pub fn segment_meta(segments: &[CaptionSegment], index: usize) -> SharedString {
    let Some(segment) = segments.get(index) else {
        return SharedString::default();
    };
    let start = segment.start_ms().unwrap_or(0);
    let end = segment.end_ms().unwrap_or(0);
    format!(
        "Card {} · {} → {} · {} {}",
        index + 1,
        format_timecode(start),
        format_timecode(end),
        segment.words.len(),
        if segment.words.len() == 1 {
            "word"
        } else {
            "words"
        }
    )
    .into()
}

/// `M:SS`, compact duration used for progress ETAs.
pub fn format_eta(seconds: f64) -> String {
    let seconds = seconds.max(0.0).round() as u64;
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

/// `MM:SS.mmm` timecode, mirroring the Slint `Format.timecode` helper.
pub fn format_timecode(ms: u64) -> String {
    let minutes = ms / 60_000;
    let seconds = (ms / 1_000) % 60;
    format!("{minutes:02}:{seconds:02}.{:03}", ms % 1_000)
}

/// Clamps a millisecond timestamp into the `int` range the Slint properties
/// use (about 24 days, far beyond any video).
fn clamp_ms(ms: u64) -> i32 {
    ms.min(i32::MAX as u64) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use sublayer_core::{VideoMetadata, WordToken};
    use sublayer_media::WaveformBucket;

    fn metadata() -> VideoMetadata {
        VideoMetadata {
            duration_seconds: 12.0,
            width: 1_920,
            height: 1_080,
            fps: 30.0,
            video_codec: Some("h264".to_owned()),
            audio_codec: Some("aac".to_owned()),
            audio_channels: Some(2),
            audio_sample_rate: Some(48_000),
        }
    }

    fn segment(start_ms: u64, end_ms: u64, text: &str) -> CaptionSegment {
        CaptionSegment::new(vec![WordToken::new(text, start_ms, end_ms)])
    }

    #[test]
    fn theme_data_roundtrips_through_the_inspector_format() {
        for preset in sublayer_subtitles::preset_names() {
            let theme = sublayer_subtitles::preset(preset).unwrap();
            let data = theme_data_from(&theme);
            let mut merged = theme.clone();
            apply_theme_data(&data, &mut merged);
            assert_eq!(merged, theme, "preset `{preset}` must survive a merge");
        }

        let mut theme = ThemeStyle::default();
        let mut data = theme_data_from(&theme);
        data.primary_hex = "#112233".into();
        data.font_size = 88;
        data.alignment_index = 8;
        data.animation_index = 0;
        apply_theme_data(&data, &mut theme);
        assert_eq!(theme.primary_color, Rgba::opaque(0x11, 0x22, 0x33));
        assert_eq!(theme.font_size, 88);
        assert_eq!(theme.alignment, 9);
        assert_eq!(theme.animation, AnimationType::None);

        // Garbage hex and out-of-range numbers must not corrupt the theme.
        let mut data = theme_data_from(&theme);
        data.primary_hex = "#zzz".into();
        data.font_size = 10_000;
        data.margin_v = -50;
        apply_theme_data(&data, &mut theme);
        assert_eq!(theme.primary_color, Rgba::opaque(0x11, 0x22, 0x33));
        assert_eq!(theme.font_size, 200);
        assert_eq!(theme.margin_v, 0);
    }

    #[test]
    fn preset_lookup_matches_names_and_aliases() {
        assert_eq!(preset_index_of("TikTok Classic"), 0);
        assert_eq!(preset_index_of("Hormozi Bold"), 1);
        assert_eq!(preset_index_of("custom"), 0);
        assert!(preset_by_index(1).is_some());
        assert!(preset_by_index(99).is_none());
    }

    #[test]
    fn segment_items_mark_the_selection() {
        let segments = vec![segment(0, 500, "hello"), segment(600, 900, "world")];
        let items = segment_items(&segments, Some(1));
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].text, "hello");
        assert!(!items[0].selected);
        assert_eq!(items[1].text, "world");
        assert!(items[1].selected);
        assert_eq!((items[1].start_ms, items[1].end_ms), (600, 900));
    }

    #[test]
    fn waveform_window_aggregates_peaks_and_rms() {
        let cache = WaveformCache::new(
            16_000,
            10,
            160_000,
            vec![
                WaveformBucket {
                    min_amplitude: -0.5,
                    max_amplitude: 0.25,
                    rms: 0.5,
                },
                WaveformBucket {
                    min_amplitude: -0.1,
                    max_amplitude: 0.9,
                    rms: 0.5,
                },
                WaveformBucket {
                    min_amplitude: -0.8,
                    max_amplitude: 0.2,
                    rms: 0.5,
                },
                WaveformBucket {
                    min_amplitude: -0.4,
                    max_amplitude: 0.4,
                    rms: 0.5,
                },
            ],
        );
        // Ten buckets per second: two columns of 200 ms each.
        let columns = waveform_window(&cache, 0, 200, 2);
        assert_eq!(columns.len(), 2);
        assert!((columns[0].low - 0.5).abs() < 1e-6);
        assert!((columns[0].high - 0.9).abs() < 1e-6);
        assert!((columns[0].rms - 0.5).abs() < 1e-6);
        assert!((columns[1].high - 0.4).abs() < 1e-6);
        assert!((columns[1].low - 0.8).abs() < 1e-6);

        // Beyond the cached range the columns are silent, not missing.
        let tail = waveform_window(&cache, 10_000, 200, 2);
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].rms, 0.0);
    }

    #[test]
    fn caption_lookup_respects_half_open_bounds() {
        let mut project = Project::new("test", "/tmp/in.mp4", metadata());
        project.segments = vec![segment(1_000, 2_000, "one"), segment(2_500, 3_000, "two")];
        assert_eq!(caption_at(&project, 1_000).as_deref(), Some("one"));
        assert_eq!(caption_at(&project, 1_999).as_deref(), Some("one"));
        assert!(caption_at(&project, 2_000).is_none());
        assert_eq!(caption_at(&project, 2_500).as_deref(), Some("two"));
        assert!(caption_at(&project, 9_999).is_none());
    }

    #[test]
    fn segment_meta_reports_bounds_and_words() {
        let segments = vec![segment(1_000, 2_500, "hello")];
        assert_eq!(
            segment_meta(&segments, 0).as_str(),
            "Card 1 · 00:01.000 → 00:02.500 · 1 word"
        );
        assert_eq!(segment_meta(&segments, 5).as_str(), "");
        assert_eq!(format_timecode(3_723_004), "62:03.004");
        assert_eq!(format_eta(0.0), "00:00");
        assert_eq!(format_eta(95.4), "01:35");
        assert_eq!(format_eta(-3.0), "00:00");
    }

    #[test]
    fn frame_conversion_keeps_pixel_data() {
        let frame = RgbaFrame {
            width: 2,
            height: 1,
            pixels: vec![255, 0, 0, 255, 0, 255, 0, 128],
        };
        let image = frame_to_image(&frame);
        assert_eq!(image.size().width, 2);
        assert_eq!(image.size().height, 1);
    }
}
