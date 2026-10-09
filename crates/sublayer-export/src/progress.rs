//! Parser for FFmpeg's `-progress pipe:1` stream and ETA estimation.
//!
//! FFmpeg writes one `key=value` block per stats period and terminates it with
//! `progress=continue` (or `progress=end` for the final block), so the parser
//! accumulates a block and emits a [`ProgressReport`] on each terminator.
//!
//! Note: despite its name, FFmpeg reports `out_time_ms` in microseconds, the
//! same unit as `out_time_us`; both keys are accepted.

use std::time::Duration;

use crate::runner::ExportProgress;

/// One complete progress block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProgressReport {
    /// Encoded output time, in milliseconds.
    pub out_time_ms: u64,
    /// Frames written so far.
    pub frame: u64,
    /// Instantaneous encoding speed in frames per second.
    pub fps: f32,
    /// Encoding speed relative to realtime (`1.0` = realtime).
    pub speed: f32,
    /// Whether this is the final report (`progress=end`).
    pub done: bool,
}

/// Block currently being accumulated.
#[derive(Debug, Default, Clone, Copy)]
struct PendingReport {
    out_time_ms: u64,
    frame: u64,
    fps: f32,
    speed: f32,
}

impl PendingReport {
    fn finish(&self, done: bool) -> ProgressReport {
        ProgressReport {
            out_time_ms: self.out_time_ms,
            frame: self.frame,
            fps: self.fps,
            speed: self.speed,
            done,
        }
    }
}

/// Incremental parser fed with the lines of FFmpeg's progress stream.
#[derive(Debug, Default)]
pub struct ProgressParser {
    pending: PendingReport,
    seen_key: bool,
}

impl ProgressParser {
    /// Feeds one line; returns the report when a block terminates.
    ///
    /// Unknown keys and unparseable values are ignored, which keeps the
    /// parser working across FFmpeg versions.
    pub fn push_line(&mut self, line: &str) -> Option<ProgressReport> {
        let line = line.trim();
        let (key, value) = line.split_once('=')?;
        let value = value.trim();
        match key {
            "out_time_us" | "out_time_ms" => {
                if let Ok(micros) = value.parse::<i64>() {
                    self.pending.out_time_ms = (micros.max(0) / 1_000) as u64;
                    self.seen_key = true;
                }
            }
            "frame" => {
                if let Ok(frame) = value.parse::<u64>() {
                    self.pending.frame = frame;
                    self.seen_key = true;
                }
            }
            "fps" => {
                if let Ok(fps) = value.parse::<f32>() {
                    self.pending.fps = fps;
                    self.seen_key = true;
                }
            }
            "speed" => {
                if let Some(speed) = value.strip_suffix('x').and_then(|v| v.parse::<f32>().ok()) {
                    self.pending.speed = speed;
                    self.seen_key = true;
                }
            }
            "progress" => {
                // A lone `progress=` without any prior key is not a block.
                if !self.seen_key {
                    return None;
                }
                let done = value == "end";
                let report = self.pending.finish(done);
                self.pending = PendingReport::default();
                self.seen_key = false;
                return Some(report);
            }
            _ => {}
        }
        None
    }
}

/// Folds reports into [`ExportProgress`] values.
///
/// Elapsed time is injected on every update so the arithmetic is testable
/// without wall-clock assertions.
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    duration_ms: u64,
}

impl ProgressTracker {
    /// Builds a tracker for a video of `duration_ms`; `0` means unknown, and
    /// the ETA then comes from FFmpeg's realtime speed factor.
    pub fn new(duration_ms: u64) -> Self {
        Self { duration_ms }
    }

    /// Combines a report with the measured elapsed time.
    pub fn update(&self, report: &ProgressReport, elapsed: Duration) -> ExportProgress {
        let percentage = if self.duration_ms > 0 {
            (report.out_time_ms as f64 / self.duration_ms as f64).clamp(0.0, 1.0) as f32
        } else {
            0.0
        };
        let elapsed_seconds = elapsed.as_secs_f64();

        let eta_seconds = if report.done {
            0.0
        } else if percentage > 0.001 {
            elapsed_seconds * (1.0 - f64::from(percentage)) / f64::from(percentage)
        } else if report.speed > 0.0 {
            // Encoding slower than realtime (`speed < 1`) still yields an ETA.
            elapsed_seconds * (1.0 / f64::from(report.speed) - 1.0).max(0.0)
        } else {
            0.0
        };

        let current_fps = if report.fps > 0.0 {
            report.fps
        } else if elapsed_seconds > 0.0 {
            report.frame as f32 / elapsed_seconds as f32
        } else {
            0.0
        };

        ExportProgress {
            percentage,
            current_fps,
            elapsed_seconds,
            eta_seconds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A realistic `-progress` block, split into the lines FFmpeg writes.
    const BLOCK: &[&str] = &[
        "frame=120",
        "fps=30.02",
        "stream_0_0_q=24.0",
        "bitrate=1234.5kbits/s",
        "total_size=524288",
        "out_time_us=4000000",
        "out_time_ms=4000000",
        "out_time=00:00:04.000000",
        "dup_frames=0",
        "drop_frames=0",
        "speed=1.48x",
        "progress=continue",
    ];

    #[test]
    fn parses_a_full_block() {
        let mut parser = ProgressParser::default();
        let mut report = None;
        for line in BLOCK {
            report = parser.push_line(line).or(report);
        }
        let report = report.expect("block must terminate with a report");
        assert_eq!(report.out_time_ms, 4_000);
        assert_eq!(report.frame, 120);
        assert!((report.fps - 30.02).abs() < 0.01);
        assert!((report.speed - 1.48).abs() < 0.01);
        assert!(!report.done);
    }

    #[test]
    fn parses_consecutive_blocks_independently() {
        let mut parser = ProgressParser::default();
        for line in BLOCK {
            parser.push_line(line);
        }
        let mut final_report = None;
        for line in [
            "frame=300",
            "out_time_us=10000000",
            "speed=N/A",
            "progress=end",
        ] {
            final_report = parser.push_line(line).or(final_report);
        }
        let report = final_report.unwrap();
        assert_eq!(report.out_time_ms, 10_000);
        assert_eq!(report.frame, 300);
        assert_eq!(
            report.fps, 0.0,
            "stale fps must not leak into the new block"
        );
        assert_eq!(report.speed, 0.0, "`N/A` speed is ignored");
        assert!(report.done);
    }

    #[test]
    fn ignores_noise_and_blocks_without_keys() {
        let mut parser = ProgressParser::default();
        assert!(parser.push_line("progress=end").is_none());
        assert!(parser.push_line("").is_none());
        assert!(parser.push_line("no equals sign").is_none());
        assert!(parser.push_line("out_time_us=not-a-number").is_none());
        assert!(parser.push_line("progress=continue").is_none());
    }

    #[test]
    fn tracker_derives_percentage_and_eta_from_duration() {
        let tracker = ProgressTracker::new(10_000);
        let report = ProgressReport {
            out_time_ms: 2_500,
            frame: 75,
            fps: 30.0,
            speed: 1.0,
            done: false,
        };
        let progress = tracker.update(&report, Duration::from_secs(5));
        assert!((progress.percentage - 0.25).abs() < 1e-6);
        assert!((progress.current_fps - 30.0).abs() < 1e-6);
        assert!((progress.elapsed_seconds - 5.0).abs() < 1e-6);
        // 25% done after 5 s → 15 s remaining.
        assert!((progress.eta_seconds - 15.0).abs() < 1e-6);
    }

    #[test]
    fn tracker_falls_back_to_speed_when_duration_is_unknown() {
        let tracker = ProgressTracker::new(0);
        let report = ProgressReport {
            out_time_ms: 4_000,
            frame: 0,
            fps: 0.0,
            speed: 0.5,
            done: false,
        };
        let progress = tracker.update(&report, Duration::from_secs(8));
        assert_eq!(progress.percentage, 0.0);
        // Half realtime: another 8 s of encoding for the same span.
        assert!((progress.eta_seconds - 8.0).abs() < 1e-6);
        // Without an fps line the frame counter and elapsed time are used.
        assert!((progress.current_fps - 0.0).abs() < 1e-6);

        let report = ProgressReport {
            frame: 40,
            ..report
        };
        let progress = tracker.update(&report, Duration::from_secs(8));
        assert!((progress.current_fps - 5.0).abs() < 1e-6);
    }

    #[test]
    fn final_report_pins_the_eta_to_zero() {
        let tracker = ProgressTracker::new(10_000);
        let report = ProgressReport {
            out_time_ms: 10_000,
            frame: 300,
            fps: 30.0,
            speed: 1.0,
            done: true,
        };
        let progress = tracker.update(&report, Duration::from_secs(10));
        assert!((progress.percentage - 1.0).abs() < 1e-6);
        assert_eq!(progress.eta_seconds, 0.0);
    }
}
