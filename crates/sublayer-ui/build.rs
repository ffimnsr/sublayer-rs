//! Compiles the Slint templates of the studio UI into Rust code.
//!
//! `app.slint` pulls in the header, viewport, inspector, and timeline
//! components, so a single `include_modules!()` in `main.rs` exposes the whole
//! window. Any change below `ui/` re-runs this script.
//!
//! Debug builds embed element names, which the headless tests use to find and
//! drive individual elements; release builds skip that metadata.

use std::path::Path;

fn main() {
    let ui_entry = Path::new("ui/app.slint");
    println!("cargo:rerun-if-changed={}", ui_entry.display());
    println!("cargo:rerun-if-changed=ui");

    let debug_build = std::env::var_os("PROFILE").is_none_or(|profile| profile == "debug");
    let config = slint_build::CompilerConfiguration::new().with_debug_info(debug_build);
    slint_build::compile_with_config(ui_entry, config)
        .expect("failed to compile the Slint templates");
}
