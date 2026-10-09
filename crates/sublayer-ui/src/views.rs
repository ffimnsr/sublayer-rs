//! Property projection: pushes [`Session`] state into the Slint window.
//!
//! Every function here is a one-way render of the session into one region of
//! the window, so an event only refreshes the panels it actually changed.

use std::rc::Rc;

use slint::{Color, Model, ModelRc, SharedString, VecModel};
use sublayer_core::{CaptionSegment, Rgba};
use sublayer_subtitles::PRESET_NAMES;

use crate::adapters;
use crate::session::Session;
use crate::{BucketItem, MainWindow, SegmentItem};

/// Long-lived Slint models of the studio window.
///
/// Assigning a fresh model to a property rebuilds every repeated item. Keeping
/// the instances stable and updating rows in place preserves the item the user
/// is holding, which is what makes dragging a caption card work at all.
pub(crate) struct UiModels {
    /// Caption cards of the timeline track.
    pub(crate) segments: Rc<VecModel<SegmentItem>>,
    /// Aggregated waveform columns.
    pub(crate) buckets: Rc<VecModel<BucketItem>>,
    /// Rows of the captions drawer.
    pub(crate) caption_rows: Rc<VecModel<crate::CaptionRow>>,
    /// Words of the caption shown over the preview frame.
    pub(crate) caption_words: Rc<VecModel<crate::CaptionWord>>,
}

impl UiModels {
    /// Creates the (empty) model instances.
    pub(crate) fn new() -> Self {
        Self {
            segments: Rc::new(VecModel::default()),
            buckets: Rc::new(VecModel::default()),
            caption_rows: Rc::new(VecModel::default()),
            caption_words: Rc::new(VecModel::default()),
        }
    }

    /// Installs the instances on the window; call once at startup.
    pub(crate) fn attach(&self, ui: &MainWindow) {
        ui.set_segments(ModelRc::new(self.segments.clone()));
        ui.set_buckets(ModelRc::new(self.buckets.clone()));
        ui.set_caption_rows(ModelRc::new(self.caption_rows.clone()));
        ui.set_caption_words(ModelRc::new(self.caption_words.clone()));
    }
}

/// Copies `items` into `model`, leaving unchanged rows (and their items) be.
fn sync_rows<T: Clone + PartialEq + 'static>(model: &VecModel<T>, items: Vec<T>) {
    if model.row_count() == items.len() {
        for (index, item) in items.into_iter().enumerate() {
            if model.row_data(index).as_ref() != Some(&item) {
                model.set_row_data(index, item);
            }
        }
    } else {
        model.set_vec(items);
    }
}

/// Fills the static option lists (models, presets, alignments, animations).
pub(crate) fn install_static_options(ui: &MainWindow) {
    let names: Vec<SharedString> = crate::bridge::model_names()
        .into_iter()
        .map(SharedString::from)
        .collect();
    let default_index = names
        .iter()
        .position(|name| name.as_str() == "base.en")
        .unwrap_or(0) as i32;
    ui.set_model_names(ModelRc::new(VecModel::from(names)));
    ui.set_model_index(default_index);

    ui.set_preset_names(strings(PRESET_NAMES));
    ui.set_alignment_names(strings(&[
        "Bottom left",
        "Bottom center",
        "Bottom right",
        "Middle left",
        "Middle center",
        "Middle right",
        "Top left",
        "Top center",
        "Top right",
    ]));
    ui.set_animation_names(strings(&["None", "Word pop", "Karaoke", "Bounce"]));
}

/// Builds a Slint string model from a slice of string literals.
fn strings(values: &[&str]) -> ModelRc<SharedString> {
    let values: Vec<SharedString> = values
        .iter()
        .map(|value| SharedString::from(*value))
        .collect();
    ModelRc::new(VecModel::from(values))
}

/// Pushes the project identity and duration.
pub(crate) fn refresh_document(ui: &MainWindow, session: &Session) {
    match session.project.as_ref() {
        Some(project) => {
            ui.set_has_project(true);
            ui.set_project_name(project.name.as_str().into());
            ui.set_video_path(project.video_path.to_string_lossy().into_owned().into());
            ui.set_video_width(project.video_metadata.width.min(i32::MAX as u32) as i32);
            ui.set_video_height(project.video_metadata.height.min(i32::MAX as u32) as i32);
            ui.set_duration_ms(ui_ms(project.video_metadata.duration_ms()));
        }
        None => {
            ui.set_has_project(false);
            ui.set_project_name("Untitled project".into());
            ui.set_duration_ms(0);
        }
    }
}

/// Pushes the theme into the inspector and the preview overlay.
pub(crate) fn refresh_theme(ui: &MainWindow, session: &Session) {
    let Some(project) = session.project.as_ref() else {
        return;
    };
    let theme = &project.theme;
    ui.set_theme(adapters::theme_data_from(theme));
    ui.set_primary_color(slint_color(theme.primary_color));
    ui.set_highlight_color(slint_color(theme.highlight_color));
    ui.set_outline_color(slint_color(theme.outline_color));
}

/// Pushes the caption cards and the inspector's text editor.
pub(crate) fn refresh_segments(ui: &MainWindow, session: &Session, models: &UiModels) {
    let (segments, selected) = match session.project.as_ref() {
        Some(project) => (project.segments.as_slice(), session.selected),
        None => (&[][..] as &[CaptionSegment], None),
    };
    sync_rows(
        &models.segments,
        adapters::segment_items(segments, selected),
    );
    ui.set_has_selection(session.selected.is_some());
    match session.selected_segment() {
        Some((_, segment)) => {
            ui.set_caption_text(segment.text().as_str().into());
            ui.set_caption_meta(
                session
                    .selected
                    .map(|index| adapters::segment_meta(segments, index))
                    .unwrap_or_default(),
            );
        }
        None => {
            ui.set_caption_text(SharedString::default());
            ui.set_caption_meta(
                if segments.is_empty() {
                    "Transcribe to create caption cards"
                } else {
                    "Select a card on the timeline"
                }
                .into(),
            );
        }
    }
    refresh_caption_drawer(ui, session, models);
}

/// Pushes the captions drawer: its visibility and its rows.
pub(crate) fn refresh_caption_drawer(ui: &MainWindow, session: &Session, models: &UiModels) {
    ui.set_caption_drawer_open(session.caption_drawer_open);
    let (segments, selected) = match session.project.as_ref() {
        Some(project) => (project.segments.as_slice(), session.selected),
        None => (&[][..] as &[CaptionSegment], None),
    };
    sync_rows(
        &models.caption_rows,
        adapters::caption_row_items(segments, selected),
    );
}

/// Pushes the playhead, timecode, and the word-level caption shown over the
/// frame (the spoken word pops into the highlight color).
pub(crate) fn refresh_playhead(ui: &MainWindow, session: &Session, models: &UiModels) {
    ui.set_playhead_ms(ui_ms(session.playhead_ms));
    ui.set_preview_timecode(adapters::format_timecode(session.playhead_ms).into());
    let caption = session
        .project
        .as_ref()
        .and_then(|project| adapters::caption_at(project, session.playhead_ms));
    ui.set_show_caption(caption.is_some());
    ui.set_active_caption(caption.unwrap_or_default());
    let words = session
        .project
        .as_ref()
        .and_then(|project| adapters::caption_words_at(project, session.playhead_ms))
        .unwrap_or_default();
    sync_rows(&models.caption_words, words);
}

/// Pushes zoom, scroll, and the waveform window.
pub(crate) fn refresh_timeline(ui: &MainWindow, session: &Session, models: &UiModels) {
    ui.set_scroll_ms(ui_ms(session.scroll_ms));
    ui.set_pixels_per_second(session.pixels_per_second);

    let viewport_px = ui.get_timeline_pixels();
    let (base_ms, column_ms, columns) = session.column_plan(viewport_px);
    let buckets = session
        .waveform
        .as_ref()
        .map(|cache| adapters::waveform_window(cache, base_ms, column_ms, columns))
        .unwrap_or_default();
    ui.set_column_base_ms(ui_ms(base_ms));
    ui.set_column_ms(ui_ms(column_ms));
    sync_rows(&models.buckets, buckets);
}

/// Pushes the busy indicator and the status line.
pub(crate) fn refresh_task(ui: &MainWindow, session: &Session) {
    ui.set_busy(session.task.busy);
    ui.set_task_progress(session.task.progress);
    ui.set_status(session.task.status.as_str().into());
}

/// Pushes the encoder chosen for the next render.
///
/// A non-empty fallback list means `auto` probing picked the encoder, so the
/// label says so: the render may still finish on another backend.
pub(crate) fn refresh_encoder(ui: &MainWindow, session: &Session) {
    let suffix = if session.render_fallbacks.is_empty() {
        ""
    } else {
        " (auto)"
    };
    ui.set_encoder_label(format!("Encoder: {}{suffix}", session.render_encoder.label()).into());
}

/// Converts a domain color into a Slint color.
fn slint_color(color: Rgba) -> Color {
    Color::from_argb_u8(color.a, color.r, color.g, color.b)
}

/// Converts a millisecond count into the `int` range used by the UI.
fn ui_ms(ms: u64) -> i32 {
    ms.min(i32::MAX as u64) as i32
}
