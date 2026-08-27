//! Typed errors for the mass model and core config: misconfiguration fails
//! loudly (docs/02_config_and_data_model.md).

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MassError {
    #[error("peptidoform parse error: {0}")]
    Parse(String),
    #[error("ambiguous or non-standard residue '{0}'")]
    AmbiguousResidue(char),
    #[error("unknown modification '{0}' (not in the UniMod subset)")]
    UnknownModification(String),
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config parse error: {0}")]
    Parse(String),
    #[error("invalid config: {0}")]
    Invalid(String),
}
