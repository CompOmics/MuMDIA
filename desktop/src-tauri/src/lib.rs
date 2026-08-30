//! The console's logic, as a library so integration tests can drive a real search
//! without a window.
//!
//! `main.rs` is the Tauri shell over this: it owns the window, the command surface
//! and the run registry, and nothing else.

pub mod components;
pub mod engine;
pub mod preflight;
pub mod run;
pub mod settings;
