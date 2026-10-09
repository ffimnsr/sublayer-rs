//! Resolution of the font directory handed to libass.
//!
//! Bundled theme fonts live in the workspace `assets/fonts` directory; release
//! packages install them elsewhere, so deployments set `SUBLAYER_FONTS_DIR`.
//! The resolved directory is passed to both the ASS script header
//! (`[Fonts]`-adjacent metadata) and the FFmpeg `ass=...:fontsdir=...` filter.

use std::path::PathBuf;

/// Environment variable overriding the auto-detected font directory.
pub const FONTS_DIR_ENV: &str = "SUBLAYER_FONTS_DIR";

/// Font directory used for rendering, resolved in order:
///
/// 1. `SUBLAYER_FONTS_DIR`,
/// 2. a bundled `assets/fonts` next to the working directory,
/// 3. (for dev builds) the workspace `assets/fonts` relative to this crate's
///    manifest.
///
/// Falls back to an empty path when nothing exists; libass then uses its
/// default font matching.
pub fn resolve_fonts_dir() -> PathBuf {
    resolve_fonts_dir_from(std::env::var(FONTS_DIR_ENV).ok())
}

/// [`resolve_fonts_dir`] with the environment override injected, so tests can
/// exercise the override without mutating process state.
pub fn resolve_fonts_dir_from(env_override: Option<String>) -> PathBuf {
    if let Some(dir) = env_override {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    let candidates = [
        PathBuf::from("assets/fonts"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/fonts"),
    ];
    candidates
        .into_iter()
        .find(|dir| dir.is_dir())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fonts_dir_prefers_environment() {
        assert_eq!(
            resolve_fonts_dir_from(Some("/tmp/sublayer-fonts".to_owned())),
            PathBuf::from("/tmp/sublayer-fonts")
        );
    }

    #[test]
    fn fonts_dir_falls_back_to_workspace_bundle() {
        // No env override: the dev fallback resolves to the workspace
        // `assets/fonts` directory, which exists in this tree.
        let dir = resolve_fonts_dir_from(Some(String::new()));
        assert!(dir.is_dir(), "{dir:?} should exist");
        assert!(dir.ends_with("assets/fonts"));
    }
}
