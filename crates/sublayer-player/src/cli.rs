//! `sublayer-player download` subcommand: pre-fetch the bundled translation
//! pairs without launching the window.
//!
//! Usage: `sublayer-player download <language…|all>`
//!
//! Languages are the bundled source codes (`ko`, `ja`, `zh-cn`, `es`, `da`,
//! `ru`; `zh` and `zh-cn` both mean Simplified Mandarin). Bundles land in
//! `$XDG_DATA_HOME/sublayer/translate/<prefix>/`, verified against the code
//! pinned digests before they are renamed into place.

use sublayer_core::SublayerPaths;
use sublayer_translate::{LOCAL_PAIRS, LocalPair, TranslationModelManager, find_local_pair};

/// Runs the subcommand; `args` are everything after `download`.
pub fn run(args: &[String]) -> Result<(), String> {
    let pairs: Vec<LocalPair> = if args.iter().any(|arg| arg == "all") {
        LOCAL_PAIRS.to_vec()
    } else {
        args.iter()
            .map(|arg| {
                pair_for(arg).ok_or_else(|| format!("unknown language `{arg}`; bundled: {BUNDLED}"))
            })
            .collect::<Result<_, String>>()?
    };

    let paths = SublayerPaths::resolve().map_err(|error| error.to_string())?;
    let root = paths.data_dir().join("translate");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    let manager = TranslationModelManager::new(&root).map_err(|error| error.to_string())?;

    for pair in pairs {
        println!("downloading {} → English…", pair.display_name);
        runtime
            .block_on(manager.ensure_pair(&pair))
            .map_err(|error| format!("{}: {error}", pair.display_name))?;
        println!("ok: {}", root.join(pair.model_prefix).display());
    }
    Ok(())
}

/// Human-readable usage line for `--help` and error messages.
pub fn usage() -> String {
    format!(
        "usage: sublayer-player [download <language…|all>]\n\
         (no arguments starts the player window; a video path opens it)\n\
         bundled languages: {BUNDLED}\n\
         downloads are verified against pinned SHA-256 digests",
    )
}

/// Bundled source codes, for error messages and the help text.
const BUNDLED: &str = "ko, ja, zh (zh-cn), es, da, ru";

/// Maps a CLI argument to a bundled pair; whisper's `zh` maps to `zh-CN`.
fn pair_for(arg: &str) -> Option<LocalPair> {
    let code = if arg.eq_ignore_ascii_case("zh") || arg.eq_ignore_ascii_case("zh-cn") {
        "zh-CN"
    } else {
        arg
    };
    find_local_pair(code).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_arguments_resolve_to_bundled_pairs() {
        assert_eq!(pair_for("ko").map(|pair| pair.model_prefix), Some("koen"));
        assert_eq!(pair_for("KO").map(|pair| pair.model_prefix), Some("koen"));
        assert_eq!(pair_for("ja").map(|pair| pair.model_prefix), Some("jaen"));
        assert_eq!(pair_for("zh").map(|pair| pair.model_prefix), Some("zhen"));
        assert_eq!(
            pair_for("zh-CN").map(|pair| pair.model_prefix),
            Some("zhen")
        );
        assert_eq!(pair_for("es").map(|pair| pair.model_prefix), Some("esen"));
        assert_eq!(pair_for("da").map(|pair| pair.model_prefix), Some("daen"));
        assert_eq!(pair_for("ru").map(|pair| pair.model_prefix), Some("ruen"));
        assert_eq!(pair_for("fr"), None, "French is not bundled");
    }

    #[test]
    fn unknown_arguments_are_rejected() {
        let error = run(&["swahili".to_owned()]).expect_err("unknown language must fail");
        assert!(error.contains("swahili"));
        assert!(error.contains("bundled"));
    }
}
