//! XDG base directory resolution for models, configuration, and caches.

use std::fs;
use std::path::{Path, PathBuf};

use directories::ProjectDirs;

use crate::CoreError;

/// Reverse-DNS-style identity handed to [`ProjectDirs`].
const QUALIFIER: &str = "rs";
const ORGANIZATION: &str = "sublayer";
const APPLICATION: &str = "sublayer";

/// Resolved application directories:
///
/// | Purpose | Location |
/// | --- | --- |
/// | Whisper models | `$XDG_DATA_HOME/sublayer/models/` |
/// | Config & themes | `$XDG_CONFIG_HOME/sublayer/` |
/// | Waveforms & scratch data | `$XDG_CACHE_HOME/sublayer/` |
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SublayerPaths {
    data_dir: PathBuf,
    config_dir: PathBuf,
    cache_dir: PathBuf,
}

impl SublayerPaths {
    /// Resolves the XDG directories of the current user.
    pub fn resolve() -> Result<Self, CoreError> {
        let dirs = ProjectDirs::from(QUALIFIER, ORGANIZATION, APPLICATION)
            .ok_or(CoreError::BaseDirectoriesUnavailable)?;
        Ok(Self::from_project_dirs(&dirs))
    }

    /// Adopts directories already resolved by the caller.
    pub fn from_project_dirs(dirs: &ProjectDirs) -> Self {
        Self {
            data_dir: dirs.data_dir().to_path_buf(),
            config_dir: dirs.config_dir().to_path_buf(),
            cache_dir: dirs.cache_dir().to_path_buf(),
        }
    }

    /// Builds an explicit layout; used by tests and portable installations.
    pub const fn new(data_dir: PathBuf, config_dir: PathBuf, cache_dir: PathBuf) -> Self {
        Self {
            data_dir,
            config_dir,
            cache_dir,
        }
    }

    /// Root of persistent application data (`$XDG_DATA_HOME/sublayer`).
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Root of user configuration (`$XDG_CONFIG_HOME/sublayer`).
    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    /// Root of disposable caches (`$XDG_CACHE_HOME/sublayer`).
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Directory holding downloaded Whisper GGML models.
    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    /// Path of a specific Whisper model file.
    pub fn model_path(&self, file_name: &str) -> PathBuf {
        self.models_dir().join(file_name)
    }

    /// Path of a configuration file (preferences, custom themes, ...).
    pub fn config_file(&self, file_name: &str) -> PathBuf {
        self.config_dir.join(file_name)
    }

    /// Path of a cache file (waveforms, thumbnails, ...).
    pub fn cache_file(&self, file_name: &str) -> PathBuf {
        self.cache_dir.join(file_name)
    }

    /// Creates every application directory, ignoring pre-existing ones.
    pub fn ensure_dirs(&self) -> Result<(), CoreError> {
        let models = self.models_dir();
        for dir in [
            self.data_dir.as_path(),
            self.config_dir.as_path(),
            self.cache_dir.as_path(),
            models.as_path(),
        ] {
            fs::create_dir_all(dir)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_paths(root: &Path) -> SublayerPaths {
        SublayerPaths::new(root.join("data"), root.join("config"), root.join("cache"))
    }

    #[test]
    fn layouts_join_expected_subpaths() {
        let directory = tempfile::tempdir().unwrap();
        let paths = temp_paths(directory.path());

        assert_eq!(paths.models_dir(), paths.data_dir().join("models"));
        assert_eq!(
            paths.model_path("base.en.bin"),
            paths.models_dir().join("base.en.bin")
        );
        assert_eq!(
            paths.config_file("themes.json"),
            paths.config_dir().join("themes.json")
        );
        assert_eq!(
            paths.cache_file("clip.waveform"),
            paths.cache_dir().join("clip.waveform")
        );
    }

    #[test]
    fn ensure_dirs_creates_every_directory_idempotently() {
        let directory = tempfile::tempdir().unwrap();
        let paths = temp_paths(directory.path());
        let models = paths.models_dir();

        paths.ensure_dirs().unwrap();
        paths.ensure_dirs().unwrap();

        for path in [
            paths.data_dir(),
            paths.config_dir(),
            paths.cache_dir(),
            models.as_path(),
        ] {
            assert!(path.is_dir(), "{path:?} should exist");
        }
    }
}
