//! Plain-text caption exporters: SubRip (`.srt`) and WebVTT (`.vtt`).
//!
//! Both formats carry only the compiled card text — word-level karaoke and
//! pop animations are ASS-only features, so SRT/VTT are meant for
//! accessibility and interchange rather than stylized burn-in.

use sublayer_core::CaptionSegment;

/// Compiles `segments` into a SubRip document.
pub fn build_srt(segments: &[CaptionSegment]) -> String {
    let mut out = String::new();
    let mut index = 0;
    for segment in segments {
        let (Some(start), Some(end)) = (segment.start_ms(), segment.end_ms()) else {
            continue;
        };
        index += 1;
        out.push_str(&format!(
            "{index}\n{} --> {}\n{}\n\n",
            format_srt_time(start),
            format_srt_time(end.max(start)),
            sanitize(&segment.text()),
        ));
    }
    out
}

/// Compiles `segments` into a WebVTT document.
pub fn build_vtt(segments: &[CaptionSegment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for segment in segments {
        let (Some(start), Some(end)) = (segment.start_ms(), segment.end_ms()) else {
            continue;
        };
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            format_vtt_time(start),
            format_vtt_time(end.max(start)),
            sanitize(&segment.text()),
        ));
    }
    out
}

/// Formats milliseconds as `HH:MM:SS,mmm` (SRT timestamp).
pub fn format_srt_time(ms: u64) -> String {
    let millis = ms % 1000;
    let total_seconds = ms / 1000;
    let seconds = total_seconds % 60;
    let minutes = total_seconds / 60 % 60;
    let hours = total_seconds / 3600;
    format!("{hours:02}:{minutes:02}:{seconds:02},{millis:03}")
}

/// Formats milliseconds as `HH:MM:SS.mmm` (WebVTT timestamp).
pub fn format_vtt_time(ms: u64) -> String {
    let millis = ms % 1000;
    let total_seconds = ms / 1000;
    let seconds = total_seconds % 60;
    let minutes = total_seconds / 60 % 60;
    let hours = total_seconds / 3600;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

/// Strips line breaks from card text (captions render as a single line).
fn sanitize(text: &str) -> String {
    text.replace(['\r', '\n'], " ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use sublayer_core::WordToken;

    fn segment(words: &[(&str, u64, u64)]) -> CaptionSegment {
        CaptionSegment::new(
            words
                .iter()
                .map(|(text, start, end)| WordToken::new(*text, *start, *end))
                .collect(),
        )
    }

    #[test]
    fn srt_numbering_and_timestamps() {
        let segments = vec![segment(&[("Hello", 1_000, 2_500)])];
        assert_eq!(
            build_srt(&segments),
            "1\n00:00:01,000 --> 00:00:02,500\nHello\n\n"
        );
    }

    #[test]
    fn srt_skips_empty_segments_but_keeps_numbering() {
        let segments = vec![
            CaptionSegment::default(),
            segment(&[("Hi", 0, 100)]),
            CaptionSegment::default(),
        ];
        assert_eq!(
            build_srt(&segments),
            "1\n00:00:00,000 --> 00:00:00,100\nHi\n\n"
        );
    }

    #[test]
    fn vtt_header_and_dot_timestamps() {
        let segments = vec![segment(&[("Hello", 1_000, 2_500)])];
        assert_eq!(
            build_vtt(&segments),
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.500\nHello\n\n"
        );
    }

    #[test]
    fn vtt_is_empty_for_no_segments() {
        assert_eq!(build_vtt(&[]), "WEBVTT\n\n");
    }

    #[test]
    fn timestamps_roll_over_at_hour_boundaries() {
        assert_eq!(format_srt_time(3_661_234), "01:01:01,234");
        assert_eq!(format_vtt_time(3_661_234), "01:01:01.234");
    }

    #[test]
    fn newlines_collapse_to_spaces() {
        let dirty = CaptionSegment::new(vec![WordToken::new("a\nb", 0, 100)]);
        assert!(build_srt(&[dirty]).contains("a b"));
    }
}
