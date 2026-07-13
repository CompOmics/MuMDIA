//! Core data-model types (PLAN.md Section 2). Ion mobility is `Option` /
//! nullable throughout so the same model serves 3D and 4D runs; MVP is 3D so
//! IM is always `None`.

use serde::{Deserialize, Serialize};

/// One observed peak. `ion_mobility` is `None` for Orbitrap DIA (MVP).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    pub mz: f64,
    pub intensity: f32,
    pub ion_mobility: Option<f32>,
}

/// A DIA isolation window in (m/z, 1/K0). IM bounds are `None` for plain DIA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsolationWindow {
    pub target_mz: f64,
    pub lower_mz: f64,
    pub upper_mz: f64,
    pub im_lower: Option<f32>,
    pub im_upper: Option<f32>,
}

impl IsolationWindow {
    #[inline]
    pub fn covers(&self, mz: f64) -> bool {
        mz >= self.lower_mz && mz <= self.upper_mz
    }
}

/// Whether a record is a target or a decoy, and which strategy made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Label {
    Target,
    Decoy,
}

impl Label {
    /// Percolator label: +1 target, -1 decoy.
    pub fn pin(&self) -> i32 {
        match self {
            Label::Target => 1,
            Label::Decoy => -1,
        }
    }
    pub fn is_decoy(&self) -> bool {
        matches!(self, Label::Decoy)
    }
}

/// Minimal in-memory MS2 scan handed to the seed search and extractor.
#[derive(Clone, Debug)]
pub struct Ms2Scan {
    pub scan_index: u32,
    pub id: String,
    pub rt_seconds: f64,
    pub window: IsolationWindow,
    /// m/z sorted peaks.
    pub peaks: Vec<Peak>,
}
