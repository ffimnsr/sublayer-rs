//! Sublayer player: a Slint window that plays media with pre-transcribed
//! captions, automatic language detection, optional translation to English,
//! and a scrubbing seek bar that keeps picture, sound and captions in sync.
//!
//! The window is a thin shell over [`app::App`]: all playback and caption
//! state lives in `session`, all background work (audio decoders, whisper
//! batches, translation, frame decoding) runs in `bridge`, and `app` wires
//! the two. There is no live ASR — captions appear only after the user
//! clicks Pre-transcribe.

// The Slint-generated bootstrap code contains `todo!()` stubs for embedded
// Rust components; see the studio's note in sublayer-ui/src/main.rs.
#![allow(clippy::todo)]

mod app;
mod audio;
mod bridge;
mod cli;
mod error;
mod session;
mod transcript_cache;
mod translate_worker;
mod video;
#[cfg(test)]
mod window_tests;

use tracing_subscriber::EnvFilter;

slint::include_modules!();

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let initial_video = if args.is_empty() {
        None
    } else {
        match args[0].as_str() {
            "download" => {
                std::process::exit(match cli::run(&args[1..]) {
                    Ok(()) => 0,
                    Err(error) => {
                        eprintln!("error: {error}\n{}", cli::usage());
                        1
                    }
                });
            }
            "--help" | "-h" | "help" => {
                println!("{}", cli::usage());
                std::process::exit(0);
            }
            // A path that exists opens as the initial video: drag a file
            // onto the player (or an "Open with" launcher) and it plays
            // right away. Slint 1.18's winit backend has no external
            // drag-and-drop API, so the file manager drop lands here.
            other => {
                let path = std::path::Path::new(other);
                if path.is_file() {
                    Some(path.to_path_buf())
                } else {
                    eprintln!("unknown subcommand `{other}`\n{}", cli::usage());
                    std::process::exit(2);
                }
            }
        }
    };
    init_tracing();
    match app::App::new(initial_video).and_then(|app| app.run()) {
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
