//! Compiles the Slint templates of the studio UI into Rust code.
//!
//! `app.slint` pulls in the header, viewport, inspector, and timeline
//! components, so a single `include_modules!()` in `main.rs` exposes the whole
//! window. Any change below `ui/` re-runs this script.

use std::path::Path;

fn main() {
    let ui_entry = Path::new("ui/app.slint");
    println!("cargo:rerun-if-changed={}", ui_entry.display());
    println!("cargo:rerun-if-changed=ui");

    slint_build::compile(ui_entry).expect("failed to compile the Slint templates");
}
