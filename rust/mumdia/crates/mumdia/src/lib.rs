//! MuMDIA pipeline library: stage implementations and shared machinery.
//! The `mumdia` binary is a thin CLI over these modules; exposing them as a lib
//! lets integration tests drive stages directly (PLAN.md Section 3.5).

pub mod calibrate;
pub mod fdr;
pub mod index;
pub mod matchers;
pub mod peaks;
pub mod predict;
pub mod quant_lfq;
pub mod rescoring;
pub mod sidecar;
pub mod spectra;
pub mod stages;
pub mod stats;
