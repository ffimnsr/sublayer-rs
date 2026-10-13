fn main() {
    // Debug info lets the headless tests address widgets by id
    // (`ElementHandle` in `window_tests`); release builds skip it.
    let config = slint_build::CompilerConfiguration::new().with_debug_info(cfg!(debug_assertions));
    slint_build::compile_with_config("ui/player.slint", config)
        .expect("failed to compile player.slint");
}
