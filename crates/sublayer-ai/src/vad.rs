//! Energy-threshold voice activity detection.
//!
//! Splits mono PCM audio into speech spans so the transcriber can skip silence
//! and music, which are where Whisper tends to hallucinate. Audio is framed at
//! a fixed window with a hop; frames above the RMS threshold are speech, then
//! short gaps are bridged and short blips dropped.

/// Parameters of the energy-based voice activity detector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadConfig {
    /// Analysis window length in milliseconds (default 30).
    pub frame_ms: u32,
    /// Hop between windows in milliseconds (default 10).
    pub hop_ms: u32,
    /// RMS threshold in dBFS; frames at or above it count as speech
    /// (default −40 dBFS).
    pub threshold_db: f32,
    /// Spans shorter than this are discarded as noise (default 200 ms).
    pub min_speech_ms: u32,
    /// Gaps shorter than this are bridged into one span (default 350 ms).
    pub min_silence_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            frame_ms: 30,
            hop_ms: 10,
            threshold_db: -40.0,
            min_speech_ms: 200,
            min_silence_ms: 350,
        }
    }
}

/// A detected speech span in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechSegment {
    /// Inclusive start on the timeline.
    pub start_ms: u64,
    /// Exclusive end on the timeline.
    pub end_ms: u64,
}

impl SpeechSegment {
    /// Length of the span.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// First sample index at `sample_rate`.
    pub fn start_sample(&self, sample_rate: u32) -> usize {
        (self.start_ms * u64::from(sample_rate) / 1_000) as usize
    }

    /// One-past-last sample index at `sample_rate`.
    pub fn end_sample(&self, sample_rate: u32) -> usize {
        (self.end_ms * u64::from(sample_rate) / 1_000) as usize
    }
}

/// Splits mono PCM samples (normalized to `-1.0..=1.0`) into speech spans.
///
/// Returns an empty list for silent input or degenerate configurations
/// (`sample_rate == 0`, `frame_ms == 0`, `hop_ms > frame_ms`).
pub fn detect_speech(samples: &[f32], sample_rate: u32, config: &VadConfig) -> Vec<SpeechSegment> {
    if samples.is_empty()
        || sample_rate == 0
        || config.frame_ms == 0
        || config.hop_ms == 0
        || config.hop_ms > config.frame_ms
    {
        return Vec::new();
    }

    let rate = u64::from(sample_rate);
    let frame_size = (sample_rate * config.frame_ms / 1_000) as usize;
    let hop_size = (sample_rate * config.hop_ms / 1_000) as usize;
    // Sub-frame rates (e.g. 30 ms frames at sample_rate < 1000) would yield
    // zero-length windows, an empty slice, and an infinite hop loop.
    if frame_size == 0 || hop_size == 0 {
        return Vec::new();
    }
    let threshold = config.threshold_db;

    // Raw speech runs from the frame classifier.
    let mut runs: Vec<SpeechSegment> = Vec::new();
    let mut index = 0_usize;
    while index < samples.len() {
        let end = (index + frame_size).min(samples.len());
        let db = frame_db(&samples[index..end]);
        if db >= threshold {
            let span = SpeechSegment {
                start_ms: index as u64 * 1_000 / rate,
                end_ms: end as u64 * 1_000 / rate,
            };
            match runs.last_mut() {
                Some(last) if span.start_ms <= last.end_ms => {
                    last.end_ms = last.end_ms.max(span.end_ms)
                }
                _ => runs.push(span),
            }
        }
        index += hop_size;
    }

    // Bridge gaps shorter than `min_silence_ms`, then drop blips shorter than
    // `min_speech_ms`.
    let min_silence_ms = u64::from(config.min_silence_ms);
    let min_speech_ms = u64::from(config.min_speech_ms);
    let mut merged: Vec<SpeechSegment> = Vec::new();
    for run in runs {
        match merged.last_mut() {
            Some(last) if run.start_ms < last.end_ms.saturating_add(min_silence_ms) => {
                last.end_ms = last.end_ms.max(run.end_ms);
            }
            _ => merged.push(run),
        }
    }
    merged
        .into_iter()
        .filter(|span| span.duration_ms() >= min_speech_ms)
        .collect()
}

/// RMS level of `samples` in dBFS, floored at `-180.0` for digital silence.
fn frame_db(samples: &[f32]) -> f32 {
    let mut sum_of_squares = 0.0_f64;
    for sample in samples {
        let value = f64::from(*sample);
        sum_of_squares += value * value;
    }
    let rms = (sum_of_squares / samples.len() as f64).sqrt().max(1e-9);
    (20.0 * rms.log10()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    fn tone(amplitude: f32, ms: u64) -> Vec<f32> {
        vec![amplitude; (ms * u64::from(RATE) / 1_000) as usize]
    }

    #[test]
    fn silence_produces_no_segments() {
        let samples = vec![0.0_f32; RATE as usize];
        assert!(detect_speech(&samples, RATE, &VadConfig::default()).is_empty());
    }

    #[test]
    fn continuous_tone_is_one_span() {
        let samples = tone(0.1, 1_000); // −20 dBFS, above the −40 dBFS threshold
        let segments = detect_speech(&samples, RATE, &VadConfig::default());

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_ms, 0);
        assert_eq!(segments[0].end_ms, 1_000);
    }

    #[test]
    fn quiet_audio_is_treated_as_silence() {
        let samples = tone(0.001, 1_000); // −60 dBFS, below the threshold
        assert!(detect_speech(&samples, RATE, &VadConfig::default()).is_empty());
    }

    #[test]
    fn long_gaps_split_spans() {
        // Keep short spans by relaxing the default min_speech_ms.
        let config = VadConfig {
            min_speech_ms: 50,
            ..VadConfig::default()
        };
        let mut samples = tone(0.1, 100);
        samples.extend(tone(0.0, 500)); // gap ≥ min_silence_ms (350)
        samples.extend(tone(0.1, 100));
        samples.extend(tone(0.0, 100));

        let segments = detect_speech(&samples, RATE, &config);

        assert_eq!(segments.len(), 2);
        // Frame windows bleed up to ~30 ms past the tone boundaries.
        assert_eq!(segments[0].start_ms, 0);
        assert_eq!(segments[0].end_ms, 120);
        assert_eq!(segments[1].start_ms, 580);
        assert_eq!(segments[1].end_ms, 720);
    }

    #[test]
    fn short_gaps_are_bridged() {
        let config = VadConfig::default();
        let mut samples = tone(0.1, 100);
        samples.extend(tone(0.0, 200)); // gap < min_silence_ms (350)
        samples.extend(tone(0.1, 100));

        let segments = detect_speech(&samples, RATE, &config);

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_ms, 0);
        assert_eq!(segments[0].end_ms, 400);
    }

    #[test]
    fn brief_blips_are_dropped() {
        let config = VadConfig::default();
        let mut samples = tone(0.0, 500);
        samples.extend(tone(0.1, 50)); // blip < min_speech_ms (200)
        samples.extend(tone(0.0, 500));

        let segments = detect_speech(&samples, RATE, &config);

        assert!(segments.is_empty());
    }

    #[test]
    fn span_exactly_at_min_speech_length_is_kept() {
        let config = VadConfig::default();
        let samples = tone(0.1, 200);
        let segments = detect_speech(&samples, RATE, &config);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].duration_ms(), 200);
    }

    #[test]
    fn degenerate_configs_yield_no_segments() {
        let samples = tone(0.1, 100);
        assert!(detect_speech(&samples, 0, &VadConfig::default()).is_empty());
        assert!(
            detect_speech(
                &samples,
                RATE,
                &VadConfig {
                    hop_ms: 40,
                    frame_ms: 30,
                    ..VadConfig::default()
                }
            )
            .is_empty()
        );
        assert!(detect_speech(&[], RATE, &VadConfig::default()).is_empty());
    }

    #[test]
    fn sub_frame_sample_rates_are_rejected_without_looping() {
        // sample_rate below ~33 Hz makes the 30 ms frame window zero samples
        // long; the detector must bail out instead of spinning forever.
        let samples = vec![0.1_f32; 32];
        assert!(detect_speech(&samples, 1, &VadConfig::default()).is_empty(),);
        assert!(detect_speech(&samples, 20, &VadConfig::default()).is_empty());
    }

    #[test]
    fn sample_mapping_roundtrips() {
        let segment = SpeechSegment {
            start_ms: 500,
            end_ms: 1_000,
        };
        assert_eq!(segment.start_sample(RATE), 8_000);
        assert_eq!(segment.end_sample(RATE), 16_000);
        assert_eq!(segment.duration_ms(), 500);
    }
}
