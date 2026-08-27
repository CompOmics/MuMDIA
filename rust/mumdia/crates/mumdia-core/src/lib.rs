//! mumdia-core: the shared vocabulary of every stage (docs/02_config_and_data_model.md).
//!
//! Types, config, the run manifest, a ProForma/UniMod mass model, physical
//! constants, and error types. No stage-specific logic lives here.

pub mod config;
pub mod constants;
pub mod error;
pub mod manifest;
pub mod mass;
pub mod rejection;
pub mod schema;
pub mod types;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
