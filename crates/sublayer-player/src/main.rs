//! Sublayer player: a Slint window that plays media with live captions,
//! automatic language detection, and optional live translation to English.
//!
//! The window is a thin shell over [`app::App`]: all playback and caption
//! state lives in `session`, all background work (audio decoders, whisper,
//! translation, frame decoding) runs in `bridge`, and `app` wires the two.

// The Slint-generated bootstrap code contains `todo!()` stubs for embedded
// Rust components; see the studio's note in sublayer-ui/src/main.rs.
#![allow(clippy::todo)]

mod app;
mod audio;
mod bridge;
mod error;
mod session;
mod translate_worker;
#[cfg(test)]
mod window_tests;

use tracing_subscriber::EnvFilter;

slint::include_modules!();

fn main() {
    init_tracing();
    match app::App::new().and_then(|app| app.run()) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}

/// Initializes diagnostics logging; `RUST_LOG` overrides the `info` default.
fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}
