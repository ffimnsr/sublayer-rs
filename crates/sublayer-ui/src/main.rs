//! Sublayer desktop studio: a Slint front end for the caption pipeline.
//!
//! The window is a thin shell over [`app::App`]: all document state lives in
//! `session`, all background work runs in `bridge`, and `adapters` translates
//! between the domain models and the Slint models.

// The Slint-generated bootstrap code contains `todo!()` stubs for embedded
// Rust components, which the workspace's `clippy::todo` lint would flag; the
// crate itself keeps the lint meaningful by having no `todo!()` at all.
#![allow(clippy::todo)]

mod adapters;
mod app;
mod bridge;
mod error;
mod session;
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
