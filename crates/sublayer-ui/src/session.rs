//! Editor session: the loaded project plus every piece of UI state the
//! renderer reads back.
//!
//! The session is deliberately free of Slint types so its editing rules
//! (segment retiming, zoom clamping, preview scheduling) are unit-testable
//! without a display server.

use std::path::PathBuf;
use std::sync::Arc;

use sublayer_core::{CaptionSegment, Project, ThemeStyle, WordToken};
use sublayer_export::{HardwareEncoder, HardwareProbe};
use sublayer_media::WaveformCache;

/// Zoom bounds of the timeline, in pixels per second. The upper bound keeps a
/// waveform column at or below ~8 px so the bars never stretch wider than the
/// source bucket resolution.
pub const MIN_PIXELS_PER_SECOND: f32 = 4.0;
/// Upper zoom bound; one millisecond already covers multiple pixels.
pub const MAX_PIXELS_PER_SECOND: f32 = 400.0;
/// Zoom used before the first fit.
pub const DEFAULT_PIXELS_PER_SECOND: f32 = 60.0;
/// Shortest caption card a trim gesture may produce.
pub const MIN_SEGMENT_MS: u64 = 100;
/// Rendered width of one waveform column, in pixels.
const COLUMN_PIXEL_WIDTH: f64 = 2.0;
/// Ceiling on the columns in one waveform window, bounding render cost.
const MAX_WAVEFORM_COLUMNS: usize = 4_096;
/// Shortest word the boundary pinning may produce.
const MIN_WORD_MS: u64 = 20;

/// What a timeline drag gesture manipulates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DragMode {
    /// Move the whole card.
    Move,
    /// Trim the start of the card.
    TrimStart,
    /// Trim the end of the card.
    TrimEnd,
}

impl DragMode {
    /// Decodes the mode code used by the Slint track component.
    pub fn from_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::Move),
            1 => Some(Self::TrimStart),
            2 => Some(Self::TrimEnd),
            _ => None,
        }
    }
}

/// Bounds captured when a drag gesture starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DragState {
    index: usize,
    mode: DragMode,
    origin_start_ms: u64,
    origin_end_ms: u64,
}

/// Outcome of a finished preview decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewOutcome {
    /// Nothing queued; the viewport is up to date.
    Idle,
    /// The playhead moved while decoding; decode `ms` next.
    Retry(u64),
}

/// Scheduling state of the viewport frame decoder.
#[derive(Debug, Default, Clone)]
pub struct PreviewState {
    /// Whether a decode task is currently running.
    pub in_flight: bool,
    /// Most recent request while a decode was running.
    pending_ms: Option<u64>,
    /// Identifier of the newest request; stale results are dropped.
    pub token: u64,
    /// Timestamp of the frame currently displayed.
    pub shown_ms: Option<u64>,
}

/// Busy indicator shown in the header and status bar.
#[derive(Debug, Clone)]
pub struct TaskState {
    /// Whether a background task is running.
    pub busy: bool,
    /// Fractional progress in `0.0..=1.0`; `0.0` means indeterminate.
    pub progress: f32,
    /// Human-readable status line.
    pub status: String,
}

impl Default for TaskState {
    fn default() -> Self {
        Self {
            busy: false,
            progress: 0.0,
            status: "Ready".to_owned(),
        }
    }
}

/// Everything the editor knows about the open document and the view.
#[derive(Debug)]
pub struct Session {
    /// Loaded project, if any.
    pub project: Option<Project>,
    /// Location the project was loaded from or last saved to.
    pub project_path: Option<PathBuf>,
    /// Waveform of the source audio, once generated.
    pub waveform: Option<Arc<WaveformCache>>,
    /// Index of the selected caption card.
    pub selected: Option<usize>,
    /// Playhead position in milliseconds.
    pub playhead_ms: u64,
    /// Time at the left edge of the visible timeline window.
    pub scroll_ms: u64,
    /// Timeline zoom.
    pub pixels_per_second: f32,
    /// Whisper model selected for the next transcription run.
    pub model_name: String,
    /// Request the GPU backend for transcription.
    pub use_gpu: bool,
    /// Run the energy VAD before transcription.
    pub enable_vad: bool,
    /// Index into the preset list currently applied to the theme.
    pub preset_index: i32,
    /// Encoders detected on this machine.
    pub render_probe: HardwareProbe,
    /// Encoder the next render will use.
    pub render_encoder: HardwareEncoder,
    /// x264 CRF / NVENC & VA-API quality for renders.
    pub render_quality: u8,
    /// Running task indicator.
    pub task: TaskState,
    /// Preview decoder state.
    pub preview: PreviewState,
    drag: Option<DragState>,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            project: None,
            project_path: None,
            waveform: None,
            selected: None,
            playhead_ms: 0,
            scroll_ms: 0,
            pixels_per_second: DEFAULT_PIXELS_PER_SECOND,
            model_name: "base.en".to_owned(),
            use_gpu: false,
            enable_vad: true,
            preset_index: 0,
            render_probe: HardwareProbe::default(),
            render_encoder: HardwareEncoder::Cpu,
            render_quality: 20,
            task: TaskState::default(),
            preview: PreviewState::default(),
            drag: None,
        }
    }
}

impl Session {
    /// Video duration in milliseconds; `0` without a project.
    pub fn duration_ms(&self) -> u64 {
        self.project
            .as_ref()
            .map(|project| project.video_metadata.duration_ms())
            .unwrap_or(0)
    }

    /// Adopts a freshly probed video as an empty project.
    pub fn install_video(&mut self, project: Project, preset_index: i32) {
        self.project_path = None;
        self.install(project, preset_index);
    }

    /// Adopts a project loaded from disk.
    pub fn install_loaded(&mut self, project: Project, path: PathBuf, preset_index: i32) {
        self.install(project, preset_index);
        self.project_path = Some(path);
    }

    /// Resets the view around `project`; the waveform arrives separately.
    fn install(&mut self, project: Project, preset_index: i32) {
        self.project = Some(project);
        self.waveform = None;
        self.selected = None;
        self.playhead_ms = 0;
        self.scroll_ms = 0;
        self.preset_index = preset_index;
        self.drag = None;
        // Bumping the token invalidates any decode still running for the
        // previous document.
        self.preview = PreviewState {
            token: self.preview.token.wrapping_add(1),
            ..PreviewState::default()
        };
    }

    /// Selects a caption card; returns `true` when the index exists.
    pub fn select(&mut self, index: usize) -> bool {
        let exists = self
            .project
            .as_ref()
            .is_some_and(|project| index < project.segments.len());
        if exists {
            self.selected = Some(index);
        }
        exists
    }

    /// Card selected in the timeline.
    pub fn selected_segment(&self) -> Option<(usize, &CaptionSegment)> {
        let index = self.selected?;
        let segment = self.project.as_ref()?.segments.get(index)?;
        Some((index, segment))
    }

    /// Replaces the text of a card with a single token spanning its bounds.
    ///
    /// Caption text is re-typed as a whole, so word-level timings inside the
    /// card are replaced by one span; whitespace-only input is rejected.
    pub fn set_segment_text(&mut self, index: usize, text: &str) -> bool {
        let text = text.trim();
        if text.is_empty() {
            return false;
        }
        let Some(project) = self.project.as_mut() else {
            return false;
        };
        let Some(segment) = project.segments.get_mut(index) else {
            return false;
        };
        let (start_ms, end_ms) = (segment_start(segment), segment_end(segment));
        segment.words = vec![WordToken::new(text, start_ms, end_ms)];
        project.touch();
        true
    }

    /// Removes a caption card; returns `true` when the index existed.
    pub fn delete_segment(&mut self, index: usize) -> bool {
        let Some(project) = self.project.as_mut() else {
            return false;
        };
        if index >= project.segments.len() {
            return false;
        }
        project.segments.remove(index);
        project.touch();
        if self
            .selected
            .is_some_and(|selected| selected >= project.segments.len())
        {
            self.selected = None;
        }
        true
    }

    /// Captures the bounds of the card a drag gesture will retime.
    pub fn begin_drag(&mut self, index: usize, mode: DragMode) -> bool {
        let Some(project) = self.project.as_ref() else {
            return false;
        };
        let Some(segment) = project.segments.get(index) else {
            return false;
        };
        self.drag = Some(DragState {
            index,
            mode,
            origin_start_ms: segment_start(segment),
            origin_end_ms: segment_end(segment),
        });
        self.selected = Some(index);
        true
    }

    /// Applies a drag delta relative to the gesture origin.
    ///
    /// Cards may not overlap their neighbours and keep at least
    /// [`MIN_SEGMENT_MS`] of duration. Returns the new bounds when the model
    /// changed.
    pub fn apply_drag(&mut self, delta_ms: i64) -> Option<(u64, u64)> {
        let drag = self.drag?;
        let project = self.project.as_mut()?;
        if drag.index >= project.segments.len() {
            return None;
        }
        let duration_ms = project.video_metadata.duration_ms();
        let previous_end = if drag.index > 0 {
            segment_end(&project.segments[drag.index - 1])
        } else {
            0
        };
        let next_start = project
            .segments
            .get(drag.index + 1)
            .map_or(duration_ms, segment_start);

        let (start, end) = (drag.origin_start_ms, drag.origin_end_ms);
        let (new_start, new_end) = match drag.mode {
            DragMode::Move => {
                let length = end.saturating_sub(start);
                let upper = next_start.saturating_sub(length);
                if previous_end > upper {
                    return None;
                }
                let shifted = shift(start, delta_ms).clamp(previous_end, upper);
                (shifted, shifted + length)
            }
            DragMode::TrimStart => {
                let upper = end.saturating_sub(MIN_SEGMENT_MS);
                if previous_end > upper {
                    return None;
                }
                let trimmed = shift(start, delta_ms).clamp(previous_end, upper);
                (trimmed, end)
            }
            DragMode::TrimEnd => {
                let lower = start.saturating_add(MIN_SEGMENT_MS);
                if lower > next_start {
                    return None;
                }
                let trimmed = shift(end, delta_ms).clamp(lower, next_start);
                (start, trimmed)
            }
        };
        if (new_start, new_end) == (start, end) {
            return None;
        }

        retime(&mut project.segments[drag.index], start, new_start, new_end);
        project.touch();
        Some((new_start, new_end))
    }

    /// Ends a drag gesture.
    pub fn end_drag(&mut self) -> bool {
        self.drag.take().is_some()
    }

    /// Milliseconds visible across `viewport_px` pixels.
    pub fn viewport_ms(&self, viewport_px: f32) -> u64 {
        if viewport_px <= 0.0 || self.pixels_per_second <= 0.0 {
            return 0;
        }
        (f64::from(viewport_px) / f64::from(self.pixels_per_second) * 1_000.0).round() as u64
    }

    /// Zooms so the whole video fits the visible width.
    pub fn zoom_fit(&mut self, viewport_px: f32) {
        let duration_ms = self.duration_ms();
        if duration_ms == 0 || viewport_px <= 0.0 {
            return;
        }
        let pixels_per_second = f64::from(viewport_px) * 1_000.0 / duration_ms as f64;
        self.pixels_per_second =
            (pixels_per_second as f32).clamp(MIN_PIXELS_PER_SECOND, MAX_PIXELS_PER_SECOND);
        self.scroll_ms = 0;
    }

    /// Multiplies the zoom, keeping the playhead pinned in the viewport.
    pub fn zoom_by(&mut self, factor: f32, viewport_px: f32) {
        let duration_ms = self.duration_ms();
        if duration_ms == 0 || viewport_px <= 0.0 || !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let anchor_ms = self.playhead_ms.min(duration_ms);
        let anchor_px = self.time_to_px(anchor_ms);

        self.pixels_per_second =
            (self.pixels_per_second * factor).clamp(MIN_PIXELS_PER_SECOND, MAX_PIXELS_PER_SECOND);
        let scroll_ms =
            anchor_ms as f64 - f64::from(anchor_px) / f64::from(self.pixels_per_second) * 1_000.0;
        self.set_scroll_ms(scroll_ms, viewport_px);
    }

    /// Horizontal position of `ms` inside the viewport, in pixels.
    pub fn time_to_px(&self, ms: u64) -> f32 {
        let offset_ms = ms as f64 - self.scroll_ms as f64;
        (offset_ms * f64::from(self.pixels_per_second) / 1_000.0) as f32
    }

    /// Scrolls by `delta_ms`, clamped to the document.
    pub fn scroll_by(&mut self, delta_ms: i64, viewport_px: f32) {
        let target = self.scroll_ms as f64 + delta_ms as f64;
        self.set_scroll_ms(target, viewport_px);
    }

    /// Sets the scroll position, clamped to `0..=duration - viewport`.
    fn set_scroll_ms(&mut self, target_ms: f64, viewport_px: f32) {
        let duration_ms = self.duration_ms();
        let viewport_ms = self.viewport_ms(viewport_px) as f64;
        let max_scroll = (duration_ms as f64 - viewport_ms).max(0.0);
        self.scroll_ms = target_ms.clamp(0.0, max_scroll).round() as u64;
    }

    /// Moves the playhead, clamped to the document.
    pub fn set_playhead(&mut self, ms: u64) {
        self.playhead_ms = ms.min(self.duration_ms());
    }

    /// Grid of waveform columns covering the visible window.
    ///
    /// Returns `(base_ms, column_ms, count)`, where `base_ms` is snapped to the
    /// column grid so columns do not jitter while scrolling. Columns are never
    /// narrower than one source bucket, so zooming past the cache resolution
    /// cannot produce empty columns.
    pub fn column_plan(&self, viewport_px: f32) -> (u64, u64, usize) {
        let duration_ms = self.duration_ms();
        if duration_ms == 0 || viewport_px <= 0.0 {
            return (0, 100, 0);
        }
        let viewport_ms = self.viewport_ms(viewport_px).max(1);
        let ideal = 1_000.0 / f64::from(self.pixels_per_second) * COLUMN_PIXEL_WIDTH;
        let bucket_ms = self
            .waveform
            .as_ref()
            .map_or(1.0, |cache| 1_000.0 / f64::from(cache.samples_per_sec()));
        let mut column_ms = ideal.max(bucket_ms).max(1.0).round() as u64;
        let mut count = viewport_ms / column_ms + 3;
        if count as usize > MAX_WAVEFORM_COLUMNS {
            column_ms = viewport_ms.div_ceil(MAX_WAVEFORM_COLUMNS as u64).max(1);
            count = viewport_ms / column_ms + 3;
        }
        let base_ms = self.scroll_ms / column_ms * column_ms;
        (base_ms, column_ms, count as usize)
    }

    /// Registers a preview request for `timestamp_ms`.
    ///
    /// Returns the token and timestamp the caller must decode, or `None` when
    /// a decode is already running (the request is queued) or when the frame is
    /// already on screen.
    pub fn request_preview(&mut self, timestamp_ms: u64) -> Option<(u64, u64)> {
        self.project.as_ref()?;
        if self.preview.in_flight {
            self.preview.pending_ms = Some(timestamp_ms);
            return None;
        }
        if self.preview.shown_ms == Some(timestamp_ms) {
            return None;
        }
        self.preview.in_flight = true;
        self.preview.token = self.preview.token.wrapping_add(1);
        Some((self.preview.token, timestamp_ms))
    }

    /// Records a decoded frame and reports whether another decode is queued.
    pub fn preview_ready(&mut self, token: u64, timestamp_ms: u64) -> PreviewOutcome {
        if token != self.preview.token {
            return PreviewOutcome::Idle;
        }
        self.preview.in_flight = false;
        self.preview.shown_ms = Some(timestamp_ms);
        match self.preview.pending_ms.take() {
            Some(next) if next != timestamp_ms => PreviewOutcome::Retry(next),
            _ => PreviewOutcome::Idle,
        }
    }

    /// Records a failed decode; queued follow-ups are dropped to avoid loops.
    pub fn preview_failed(&mut self, token: u64) {
        if token != self.preview.token {
            return;
        }
        self.preview.in_flight = false;
        self.preview.pending_ms = None;
    }

    /// Switches to a new preset, replacing the whole style.
    pub fn apply_preset(&mut self, index: i32, theme: ThemeStyle) {
        self.preset_index = index;
        if let Some(project) = self.project.as_mut() {
            project.theme = theme;
            project.touch();
        }
    }

    /// Marks the start of a background task.
    pub fn begin_task(&mut self, label: impl Into<String>) {
        self.task = TaskState {
            busy: true,
            progress: 0.0,
            status: label.into(),
        };
    }

    /// Updates the running task's progress (`0.0` keeps it indeterminate).
    pub fn set_task_progress(&mut self, fraction: f32) {
        self.task.progress = fraction.clamp(0.0, 1.0);
    }

    /// Updates the render progress, including the status line summary.
    pub fn set_render_progress(&mut self, percentage: f32, fps: f32, eta_seconds: f64) {
        let percentage = percentage.clamp(0.0, 1.0);
        self.task.progress = percentage;
        self.task.status = format!(
            "Rendering {}% · {:.1} fps · ETA {}",
            (percentage * 100.0).round(),
            fps,
            crate::adapters::format_eta(eta_seconds)
        );
    }

    /// Adopts the encoder resolved by the hardware probe.
    pub fn set_render_encoder(&mut self, probe: HardwareProbe, encoder: HardwareEncoder) {
        self.render_probe = probe;
        self.render_encoder = encoder;
    }

    /// Marks the running task as finished.
    pub fn finish_task(&mut self, status: impl Into<String>) {
        self.task = TaskState {
            busy: false,
            progress: 0.0,
            status: status.into(),
        };
    }
}

/// Card start, or `0` when the card holds no words.
fn segment_start(segment: &CaptionSegment) -> u64 {
    segment.start_ms().unwrap_or(0)
}

/// Card end, or `0` when the card holds no words.
fn segment_end(segment: &CaptionSegment) -> u64 {
    segment.end_ms().unwrap_or(0)
}

/// Applies a signed millisecond delta to a timestamp.
fn shift(value: u64, delta_ms: i64) -> u64 {
    if delta_ms >= 0 {
        value.saturating_add(delta_ms as u64)
    } else {
        value.saturating_sub(delta_ms.unsigned_abs())
    }
}

/// Rewrites a card's word timings to the new bounds.
///
/// Every word shifts by the start delta, then the boundary words are pinned to
/// the requested bounds and all words are clamped into `[start, end]`, so a
/// word stream that loses its timing granularity still stays monotonic.
fn retime(segment: &mut CaptionSegment, old_start: u64, start: u64, end: u64) {
    if segment.words.is_empty() {
        segment
            .words
            .push(WordToken::new(String::new(), start, end));
        return;
    }
    let start_shift = start as i64 - old_start as i64;
    for word in &mut segment.words {
        word.start_ms = shift(word.start_ms, start_shift);
        word.end_ms = shift(word.end_ms, start_shift);
    }
    let last = segment.words.len() - 1;
    segment.words[0].start_ms = start;
    segment.words[last].end_ms = end;
    for word in &mut segment.words {
        word.start_ms = word.start_ms.clamp(start, end.saturating_sub(MIN_WORD_MS));
        word.end_ms = word.end_ms.clamp(
            word.start_ms.saturating_add(MIN_WORD_MS),
            end.max(start + MIN_WORD_MS),
        );
    }
    // The pins survive clamping because `end >= start + MIN_SEGMENT_MS`.
    debug_assert_eq!(segment_start(segment), start);
    debug_assert_eq!(segment_end(segment), end);
}

#[cfg(test)]
mod tests {
    use super::*;
    use sublayer_core::VideoMetadata;

    fn metadata(duration_ms: u64) -> VideoMetadata {
        VideoMetadata {
            duration_seconds: duration_ms as f64 / 1_000.0,
            width: 1_920,
            height: 1_080,
            fps: 30.0,
            video_codec: Some("h264".to_owned()),
            audio_codec: Some("aac".to_owned()),
            audio_channels: Some(2),
            audio_sample_rate: Some(48_000),
        }
    }

    fn project_with_segments(bounds: &[(u64, u64)]) -> Project {
        let mut project = Project::new("test", "/tmp/in.mp4", metadata(30_000));
        project.segments = bounds
            .iter()
            .map(|&(start_ms, end_ms)| {
                let middle = start_ms + (end_ms - start_ms) / 2;
                CaptionSegment::new(vec![
                    WordToken::new("one", start_ms, middle),
                    WordToken::new("two", middle, end_ms),
                ])
            })
            .collect();
        project
    }

    fn session_with(bounds: &[(u64, u64)]) -> Session {
        let mut session = Session::default();
        session.install_video(project_with_segments(bounds), 0);
        session
    }

    #[test]
    fn move_is_clamped_by_neighbours() {
        let mut session = session_with(&[(1_000, 2_000), (3_000, 4_000)]);
        session.begin_drag(0, DragMode::Move);
        // Try to push the first card past the second one.
        assert_eq!(session.apply_drag(5_000).unwrap(), (2_000, 3_000));
    }

    #[test]
    fn move_stays_inside_the_document() {
        let mut session = session_with(&[(1_000, 2_000)]);
        session.begin_drag(0, DragMode::Move);
        assert_eq!(session.apply_drag(-10_000).unwrap(), (0, 1_000));

        session.begin_drag(0, DragMode::Move);
        assert_eq!(session.apply_drag(500_000).unwrap(), (29_000, 30_000));
    }

    #[test]
    fn trim_start_respects_minimum_duration() {
        let mut session = session_with(&[(1_000, 2_000)]);
        session.begin_drag(0, DragMode::TrimStart);
        let (start, end) = session.apply_drag(10_000).unwrap();
        assert_eq!(end, 2_000);
        assert_eq!(start, 2_000 - MIN_SEGMENT_MS);
    }

    #[test]
    fn trim_end_respects_minimum_duration_and_neighbours() {
        let mut session = session_with(&[(1_000, 2_000), (3_000, 4_000)]);
        session.begin_drag(0, DragMode::TrimEnd);
        let (start, end) = session.apply_drag(10_000).unwrap();
        assert_eq!(start, 1_000);
        assert_eq!(end, 3_000);
    }

    #[test]
    fn drag_without_movement_reports_no_change() {
        let mut session = session_with(&[(1_000, 2_000)]);
        session.begin_drag(0, DragMode::Move);
        assert!(session.apply_drag(0).is_none());
    }

    #[test]
    fn retimed_card_keeps_word_time_bounds() {
        let mut session = session_with(&[(1_000, 2_000)]);
        session.begin_drag(0, DragMode::Move);
        session.apply_drag(500).unwrap();
        let (index, segment) = session.selected_segment().unwrap();
        assert_eq!(index, 0);
        assert_eq!(segment.start_ms(), Some(1_500));
        assert_eq!(segment.end_ms(), Some(2_500));
        // Interior words follow the shift.
        assert_eq!(segment.words[1].start_ms, 2_000);
    }

    #[test]
    fn trimming_an_end_keeps_words_ordered() {
        let mut session = session_with(&[(1_000, 3_000)]);
        session.begin_drag(0, DragMode::TrimEnd);
        let (_, end) = session.apply_drag(-1_500).unwrap();
        assert_eq!(end, 1_500);
        let segment = &session.project.as_ref().unwrap().segments[0];
        assert_eq!(segment.end_ms(), Some(1_500));
        for pair in segment.words.windows(2) {
            assert!(pair[0].end_ms <= pair[1].end_ms);
            assert!(pair[0].start_ms <= pair[1].start_ms);
        }
        assert!(segment.words.iter().all(|word| word.end_ms > word.start_ms));
    }

    #[test]
    fn text_edits_collapse_the_card_to_one_token() {
        let mut session = session_with(&[(1_000, 2_000)]);
        assert!(session.set_segment_text(0, "  rewritten  "));
        let segment = &session.project.as_ref().unwrap().segments[0];
        assert_eq!(segment.words.len(), 1);
        assert_eq!(segment.text(), "rewritten");
        assert_eq!(
            (segment.start_ms(), segment.end_ms()),
            (Some(1_000), Some(2_000))
        );
        assert!(!session.set_segment_text(0, "   "));
    }

    #[test]
    fn deleting_a_card_drops_a_stale_selection() {
        let mut session = session_with(&[(1_000, 2_000), (3_000, 4_000)]);
        session.select(1);
        assert!(session.delete_segment(1));
        assert!(session.selected.is_none());
        assert!(!session.delete_segment(9));
    }

    #[test]
    fn zoom_fit_shows_the_whole_video() {
        let mut session = session_with(&[(0, 1_000)]);
        session.zoom_fit(900.0);
        // 30 s of video across 900 px.
        assert!((session.pixels_per_second - 30.0).abs() < 0.001);
        assert_eq!(session.viewport_ms(900.0), 30_000);
    }

    #[test]
    fn zoom_keeps_the_playhead_in_place() {
        let mut session = session_with(&[(0, 1_000)]);
        session.zoom_fit(900.0);
        session.set_playhead(15_000);
        let before = session.time_to_px(15_000);
        session.zoom_by(2.0, 900.0);
        let after = session.time_to_px(15_000);
        assert!((before - after).abs() <= 1.0, "{before} vs {after}");
    }

    #[test]
    fn scroll_is_clamped_to_the_document() {
        let mut session = session_with(&[(0, 1_000)]);
        session.zoom_fit(900.0);
        session.scroll_by(-5_000, 900.0);
        assert_eq!(session.scroll_ms, 0);
        session.scroll_by(500_000, 900.0);
        assert_eq!(
            session.scroll_ms, 0,
            "a fully fitted timeline cannot scroll"
        );

        session.zoom_by(4.0, 900.0);
        session.scroll_by(500_000, 900.0);
        assert_eq!(session.scroll_ms + session.viewport_ms(900.0), 30_000);
    }

    #[test]
    fn column_plan_snaps_to_the_grid_and_covers_the_window() {
        let mut session = session_with(&[(0, 1_000)]);
        session.zoom_fit(900.0);
        session.zoom_by(4.0, 900.0);
        session.scroll_by(3_700, 900.0);

        let (base_ms, column_ms, count) = session.column_plan(900.0);
        assert_eq!(base_ms % column_ms, 0);
        assert!(base_ms <= session.scroll_ms);
        assert!(
            base_ms + column_ms * count as u64 >= session.scroll_ms + session.viewport_ms(900.0),
            "columns must cover the visible window"
        );
        assert!(count <= MAX_WAVEFORM_COLUMNS + 3);
    }

    #[test]
    fn preview_requests_coalesce_while_decoding() {
        let mut session = session_with(&[(0, 1_000)]);
        let (token, ms) = session.request_preview(500).unwrap();
        assert_eq!(ms, 500);
        assert!(session.request_preview(700).is_none());
        assert_eq!(
            session.preview_ready(token, 500),
            PreviewOutcome::Retry(700)
        );
        assert_eq!(session.preview.shown_ms, Some(500));
        // The queued request is picked up next.
        let (next_token, next_ms) = session.request_preview(700).unwrap();
        assert_eq!(next_ms, 700);
        assert_eq!(session.preview_ready(next_token, 700), PreviewOutcome::Idle);
        // Decoding the same frame twice is a no-op.
        assert!(session.request_preview(700).is_none());
    }

    #[test]
    fn installing_a_video_invalidates_running_decodes() {
        let mut session = session_with(&[(0, 1_000)]);
        let (token, _) = session.request_preview(0).unwrap();
        session.install_video(project_with_segments(&[(0, 500)]), 0);
        assert_eq!(session.preview_ready(token, 0), PreviewOutcome::Idle);
        assert_eq!(session.preview.shown_ms, None);
    }

    #[test]
    fn render_progress_updates_the_status_line() {
        let mut session = session_with(&[(0, 1_000)]);
        session.begin_task("Rendering with CPU (libx264)");
        session.set_render_progress(0.42, 25.4, 12.6);
        assert!(session.task.busy);
        assert!((session.task.progress - 0.42).abs() < 1e-6);
        assert_eq!(session.task.status, "Rendering 42% · 25.4 fps · ETA 00:13");
    }

    #[test]
    fn probed_encoder_is_adopted_for_renders() {
        use sublayer_export::{HardwareEncoder, HardwareProbe};

        let mut session = Session::default();
        session.set_render_encoder(
            HardwareProbe {
                vaapi: false,
                nvenc: true,
                vaapi_device: None,
            },
            HardwareEncoder::Nvenc,
        );
        assert_eq!(session.render_encoder, HardwareEncoder::Nvenc);
        assert!(session.render_probe.nvenc);
    }
}
