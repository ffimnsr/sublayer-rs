//! Serializable domain models describing a captioning project.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::CoreError;

/// On-disk schema version written into every `.sublayer` project file.
pub const PROJECT_SCHEMA_VERSION: u32 = 1;

/// Metadata describing the source video, produced by `sublayer-media` probing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoMetadata {
    /// Total duration in seconds.
    pub duration_seconds: f64,
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Frames per second; may be fractional (for example `29.97`).
    pub fps: f64,
    /// Video codec reported by FFprobe (for example `h264`).
    pub video_codec: Option<String>,
    /// Audio codec reported by FFprobe (for example `aac`).
    pub audio_codec: Option<String>,
    /// Number of audio channels, when an audio stream exists.
    pub audio_channels: Option<u32>,
    /// Audio sample rate in Hz, when an audio stream exists.
    pub audio_sample_rate: Option<u32>,
}

impl VideoMetadata {
    /// Whether the source has an audio stream that can be transcribed.
    pub fn has_audio(&self) -> bool {
        self.audio_codec.is_some() || self.audio_channels.is_some()
    }

    /// Total duration in whole milliseconds.
    pub fn duration_ms(&self) -> u64 {
        (self.duration_seconds.max(0.0) * 1000.0).round() as u64
    }
}

/// A single transcribed word with word-level timing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordToken {
    /// Recognized text without surrounding whitespace.
    pub text: String,
    /// Start of the word on the timeline, in milliseconds.
    pub start_ms: u64,
    /// End of the word on the timeline, in milliseconds.
    pub end_ms: u64,
}

impl WordToken {
    /// Creates a token from its text and millisecond bounds.
    pub fn new(text: impl Into<String>, start_ms: u64, end_ms: u64) -> Self {
        Self {
            text: text.into(),
            start_ms,
            end_ms,
        }
    }

    /// Duration of the word in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// One caption card: an ordered list of words displayed together.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptionSegment {
    /// Words belonging to this caption, in speech order.
    pub words: Vec<WordToken>,
}

impl CaptionSegment {
    /// Creates a segment from its words.
    pub fn new(words: Vec<WordToken>) -> Self {
        Self { words }
    }

    /// Whether the segment holds no words.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Earliest word start, if any words exist.
    pub fn start_ms(&self) -> Option<u64> {
        self.words.iter().map(|word| word.start_ms).min()
    }

    /// Latest word end, if any words exist.
    pub fn end_ms(&self) -> Option<u64> {
        self.words.iter().map(|word| word.end_ms).max()
    }

    /// Segment text with single spaces between words.
    pub fn text(&self) -> String {
        let mut text = String::new();
        for word in &self.words {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&word.text);
        }
        text
    }
}

/// Built-in caption animation styles.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnimationType {
    /// Static captions; no movement or highlight.
    #[default]
    None,
    /// Scale-up "pop" applied to each word as it is spoken.
    WordPop,
    /// Karaoke fill that sweeps across the active word.
    Karaoke,
    /// Overshooting bounce used by short-form caption styles.
    Bounce,
}

/// Straight (non-premultiplied) RGBA color with 8 bits per channel.
///
/// Serializes as `#RRGGBB` when opaque and `#RRGGBBAA` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Rgba {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Alpha channel; `255` is fully opaque.
    pub a: u8,
}

impl Rgba {
    /// Opaque white.
    pub const WHITE: Self = Self::opaque(0xFF, 0xFF, 0xFF);
    /// Opaque black.
    pub const BLACK: Self = Self::opaque(0x00, 0x00, 0x00);

    /// Builds a color from all four channels.
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Builds a fully opaque color.
    pub const fn opaque(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 0xFF)
    }

    /// Parses `#RRGGBB` or `#RRGGBBAA` (the leading `#` is optional).
    pub fn from_hex(value: &str) -> Result<Self, CoreError> {
        let digits = value.strip_prefix('#').unwrap_or(value);
        if !digits.is_ascii() {
            return Err(CoreError::InvalidColor(value.to_owned()));
        }
        let pair = |range: std::ops::Range<usize>| -> Result<u8, CoreError> {
            u8::from_str_radix(&digits[range], 16)
                .map_err(|_| CoreError::InvalidColor(value.to_owned()))
        };
        match digits.len() {
            6 => Ok(Self::opaque(pair(0..2)?, pair(2..4)?, pair(4..6)?)),
            8 => Ok(Self::new(
                pair(0..2)?,
                pair(2..4)?,
                pair(4..6)?,
                pair(6..8)?,
            )),
            _ => Err(CoreError::InvalidColor(value.to_owned())),
        }
    }

    /// Formats the color as `#RRGGBB` (opaque) or `#RRGGBBAA`.
    pub fn to_hex(self) -> String {
        if self.a == 0xFF {
            format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
        } else {
            format!("#{:02X}{:02X}{:02X}{:02X}", self.r, self.g, self.b, self.a)
        }
    }
}

impl From<Rgba> for String {
    fn from(color: Rgba) -> Self {
        color.to_hex()
    }
}

impl TryFrom<String> for Rgba {
    type Error = CoreError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::from_hex(&value)
    }
}

/// Visual styling applied to rendered captions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThemeStyle {
    /// Human-readable name of the style or preset.
    pub name: String,
    /// Font family requested from libass.
    pub font_name: String,
    /// Font size in ASS script units (roughly pixels at 1080p).
    pub font_size: u32,
    /// Base text color.
    pub primary_color: Rgba,
    /// Color of the active word in karaoke/pop animations.
    pub highlight_color: Rgba,
    /// Text outline color.
    pub outline_color: Rgba,
    /// Outline thickness in pixels; `0.0` disables the outline.
    pub outline_width: f32,
    /// Drop-shadow distance in pixels; `0.0` disables the shadow.
    pub shadow: f32,
    /// Whether text is rendered bold.
    pub bold: bool,
    /// Whether text is uppercased before rendering.
    pub uppercase: bool,
    /// ASS numpad alignment (`1`-`9`); `2` is bottom-center.
    pub alignment: u8,
    /// Vertical margin from the aligned edge, in pixels.
    pub margin_v: u32,
    /// Animation applied to the captions.
    pub animation: AnimationType,
}

impl Default for ThemeStyle {
    /// Neutral short-form starting point; built-in presets live in
    /// `sublayer-subtitles`.
    fn default() -> Self {
        Self {
            name: "Default".to_owned(),
            font_name: "Montserrat".to_owned(),
            font_size: 64,
            primary_color: Rgba::WHITE,
            highlight_color: Rgba::opaque(0xFE, 0x2C, 0x55),
            outline_color: Rgba::BLACK,
            outline_width: 4.0,
            shadow: 0.0,
            bold: true,
            uppercase: true,
            alignment: 2,
            margin_v: 320,
            animation: AnimationType::WordPop,
        }
    }
}

/// Root document of a `.sublayer` project file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    /// Stable identifier of this project.
    pub id: Uuid,
    /// Schema version this document was written with.
    pub version: u32,
    /// User-facing project name.
    pub name: String,
    /// Path to the source video.
    pub video_path: PathBuf,
    /// Probed metadata of the source video.
    pub video_metadata: VideoMetadata,
    /// Caption segments, ordered by start time.
    pub segments: Vec<CaptionSegment>,
    /// Theme applied when rendering captions.
    pub theme: ThemeStyle,
    /// Creation timestamp (UTC).
    pub created_at: DateTime<Utc>,
    /// Last modification timestamp (UTC).
    pub updated_at: DateTime<Utc>,
}

impl Project {
    /// Creates an empty project for `video_path` at the current schema version.
    pub fn new(
        name: impl Into<String>,
        video_path: impl Into<PathBuf>,
        video_metadata: VideoMetadata,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            version: PROJECT_SCHEMA_VERSION,
            name: name.into(),
            video_path: video_path.into(),
            video_metadata,
            segments: Vec::new(),
            theme: ThemeStyle::default(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Marks the project as modified now.
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }

    /// Atomically writes the project to `path` as pretty-printed JSON.
    ///
    /// The document is written to a temporary file in the destination directory
    /// and renamed into place, so a crash can never leave a half-written
    /// project behind. Missing parent directories are created.
    pub fn save(&self, path: &Path) -> Result<(), CoreError> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .ok_or_else(|| CoreError::MissingParent {
                path: path.to_path_buf(),
            })?;
        std::fs::create_dir_all(parent)?;

        let json = serde_json::to_vec_pretty(self)?;
        let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
        std::io::Write::write_all(&mut temporary, &json)?;
        // Flush to disk before the rename so the visible file is never partial.
        temporary.as_file_mut().sync_all()?;
        temporary
            .persist(path)
            .map_err(|error| CoreError::Io(error.error))?;
        Ok(())
    }

    /// Loads a project from `path`, rejecting unsupported schema versions.
    pub fn load(path: &Path) -> Result<Self, CoreError> {
        let bytes = std::fs::read(path)?;
        let project: Self = serde_json::from_slice(&bytes)?;
        if project.version == 0 || project.version > PROJECT_SCHEMA_VERSION {
            return Err(CoreError::UnsupportedSchemaVersion {
                found: project.version,
                supported: PROJECT_SCHEMA_VERSION,
            });
        }
        Ok(project)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> VideoMetadata {
        VideoMetadata {
            duration_seconds: 12.5,
            width: 1080,
            height: 1920,
            fps: 30.0,
            video_codec: Some("h264".to_owned()),
            audio_codec: Some("aac".to_owned()),
            audio_channels: Some(2),
            audio_sample_rate: Some(48_000),
        }
    }

    fn sample_project() -> Project {
        let mut project = Project::new("Demo", "/videos/clip.mp4", sample_metadata());
        project.theme.animation = AnimationType::Karaoke;
        project.segments.push(CaptionSegment::new(vec![
            WordToken::new("hello", 0, 320),
            WordToken::new("world", 320, 700),
        ]));
        project
    }

    #[test]
    fn metadata_derives_audio_flags_and_duration() {
        let metadata = sample_metadata();
        assert!(metadata.has_audio());
        assert_eq!(metadata.duration_ms(), 12_500);

        let silent = VideoMetadata {
            audio_codec: None,
            audio_channels: None,
            duration_seconds: 0.0,
            ..sample_metadata()
        };
        assert!(!silent.has_audio());
        assert_eq!(silent.duration_ms(), 0);
    }

    #[test]
    fn rgba_parses_six_and_eight_digit_hex() {
        assert_eq!(
            Rgba::from_hex("#FE2C55").unwrap(),
            Rgba::opaque(0xFE, 0x2C, 0x55)
        );
        assert_eq!(
            Rgba::from_hex("fe2c5580").unwrap(),
            Rgba::new(0xFE, 0x2C, 0x55, 0x80)
        );
        assert_eq!(Rgba::from_hex("#00000000").unwrap().to_hex(), "#00000000");
        assert_eq!(Rgba::WHITE.to_hex(), "#FFFFFF");
    }

    #[test]
    fn rgba_rejects_malformed_hex() {
        for value in [
            "",
            "#",
            "#12345",
            "#1234567",
            "#GGGGGG",
            "#12345G",
            "éééééé",
        ] {
            assert!(
                matches!(Rgba::from_hex(value), Err(CoreError::InvalidColor(_))),
                "{value:?} should be rejected"
            );
        }
    }

    #[test]
    fn rgba_roundtrips_through_json() {
        let color = Rgba::opaque(0xFE, 0x2C, 0x55);
        let json = serde_json::to_string(&color).unwrap();
        assert_eq!(json, "\"#FE2C55\"");
        assert_eq!(serde_json::from_str::<Rgba>(&json).unwrap(), color);

        let translucent = Rgba::new(0x10, 0x20, 0x30, 0x40);
        let json = serde_json::to_string(&translucent).unwrap();
        assert_eq!(json, "\"#10203040\"");
        assert_eq!(serde_json::from_str::<Rgba>(&json).unwrap(), translucent);
    }

    #[test]
    fn word_token_reports_duration() {
        let word = WordToken::new("captions", 1_000, 1_450);
        assert_eq!(word.duration_ms(), 450);
        assert_eq!(WordToken::new("odd", 500, 400).duration_ms(), 0);
    }

    #[test]
    fn caption_segment_derives_bounds_and_text() {
        let segment = CaptionSegment::new(vec![
            WordToken::new("world", 400, 700),
            WordToken::new("hello", 100, 380),
        ]);
        assert_eq!(segment.start_ms(), Some(100));
        assert_eq!(segment.end_ms(), Some(700));
        assert_eq!(segment.text(), "world hello");
        assert!(CaptionSegment::default().is_empty());
        assert_eq!(CaptionSegment::default().start_ms(), None);
    }

    #[test]
    fn project_new_initializes_defaults() {
        let project = Project::new("Demo", "/videos/clip.mp4", sample_metadata());
        assert_eq!(project.version, PROJECT_SCHEMA_VERSION);
        assert!(project.segments.is_empty());
        assert_eq!(project.created_at, project.updated_at);
    }

    #[test]
    fn project_roundtrips_through_disk() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("nested").join("demo.sublayer");
        let project = sample_project();

        project.save(&path).unwrap();
        let loaded = Project::load(&path).unwrap();

        assert_eq!(loaded, project);
    }

    #[test]
    fn project_save_overwrites_existing_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("demo.sublayer");
        let mut project = sample_project();

        project.save(&path).unwrap();
        project.name = "Renamed".to_owned();
        project.save(&path).unwrap();

        assert_eq!(Project::load(&path).unwrap().name, "Renamed");
    }

    #[test]
    fn project_save_leaves_no_temporary_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("demo.sublayer");
        sample_project().save(&path).unwrap();

        let entries: Vec<_> = std::fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(entries, vec![std::ffi::OsString::from("demo.sublayer")]);
    }

    #[test]
    fn project_save_requires_a_parent_directory() {
        let error = sample_project()
            .save(Path::new("demo.sublayer"))
            .unwrap_err();
        assert!(matches!(error, CoreError::MissingParent { .. }));
    }

    #[test]
    fn project_load_rejects_unknown_schema_versions() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("future.sublayer");

        let mut project = sample_project();
        project.version = PROJECT_SCHEMA_VERSION + 1;
        std::fs::write(&path, serde_json::to_vec(&project).unwrap()).unwrap();
        assert!(matches!(
            Project::load(&path),
            Err(CoreError::UnsupportedSchemaVersion { found, supported })
                if found == PROJECT_SCHEMA_VERSION + 1 && supported == PROJECT_SCHEMA_VERSION
        ));

        project.version = 0;
        std::fs::write(&path, serde_json::to_vec(&project).unwrap()).unwrap();
        assert!(matches!(
            Project::load(&path),
            Err(CoreError::UnsupportedSchemaVersion { found: 0, .. })
        ));
    }
}
