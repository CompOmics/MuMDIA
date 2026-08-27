//! MuMDIA pipeline library: stage implementations and shared machinery.
//! The `mumdia` binary is a thin CLI over these modules; exposing them as a lib
//! lets integration tests drive stages directly (docs/01_overview_and_dataflow.md).

pub mod calibrate;
pub mod fdr;
pub mod index;
pub mod matchers;
pub mod peaks;
pub mod predict;
pub mod quant_lfq;
pub mod rescoring;
pub mod sidecar;
pub mod solve;
pub mod spectra;
pub mod stages;
pub mod stats;
