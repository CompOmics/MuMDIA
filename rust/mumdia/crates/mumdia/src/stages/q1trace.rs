//! Cycle-level traces of scanning-quadrupole spectra (`extract.q1_traces`), SCIEX ZT Scan DIA.
//!
//! **Cycle traces.** On a ZT Scan run a candidate's chromatogram grid holds every scan of the
//! isolation windows that cover its precursor: about five Q1 steps a cycle, 1.3 ms apart, then
//! a gap of about a second to the next cycle. Read as one series, the trace is a sawtooth.
//! Under `extract.q1_traces` extract writes every accepted candidate's traces once more with one
//! point per cycle (`chromatograms_q1.parquet`, beside the chromatograms, which stay as they
//! are): each fragment's samples of one cycle collapsed to the least-squares amplitude
//! `sum(w x) / sum(w^2)` under `w`, the fragment's expected Q1 profile at the step's target and
//! the precursor's calibrated expected centre (`CandQ1::expected`), the estimator of
//! `quant.q1_fragments` ([`super::q1quant::collapse_trace`]), at the retention time of the
//! cycle's representative step, the covering step whose target is nearest the expected centre
//! (a scan of the run). [`CycleSpans`] groups a grid into cycles and picks those steps;
//! [`ls_amplitude`] is the estimator. Read in place of the interleaved traces (the features, the
//! apex, the gate), the cycle traces measured fewer identifications (docs/37_ztscan.md, section
//! 15), so no stage reads them.
//!
//! Deterministic: every sum runs in grid order; ties keep the first.

/// The cycles of a candidate's per-step grid (`extract.q1_traces`): consecutive grid samples
/// of one Q1 cycle form one span, and each span has a representative sample.
#[derive(Clone, Debug, PartialEq)]
pub struct CycleSpans {
    /// Per cycle, in grid order: the grid samples `start..end` it holds.
    pub spans: Vec<(usize, usize)>,
    /// Per cycle: the grid index of its representative sample, the one whose step target is
    /// nearest the expected Q1 centre (the first on a tie).
    pub rep: Vec<usize>,
}

impl CycleSpans {
    /// The cycles of a grid from each sample's `(cycle, step target)`, with the representative
    /// samples nearest `centre`. `None` for an empty grid, a sample outside the sweep
    /// (`u32::MAX`), or cycles that do not ascend along the grid (a cycle that comes back
    /// after another).
    pub fn of(place: &[(u32, f64)], centre: f64) -> Option<CycleSpans> {
        if place.is_empty() {
            return None;
        }
        let (mut spans, mut rep) = (Vec::new(), Vec::new());
        let mut last: Option<u32> = None;
        let mut i = 0usize;
        while i < place.len() {
            let c = place[i].0;
            if c == u32::MAX || last.is_some_and(|l| c <= l) {
                return None;
            }
            let a = i;
            let mut best = i;
            while i < place.len() && place[i].0 == c {
                if (place[i].1 - centre).abs() < (place[best].1 - centre).abs() {
                    best = i;
                }
                i += 1;
            }
            spans.push((a, i));
            rep.push(best);
            last = Some(c);
        }
        Some(CycleSpans { spans, rep })
    }

    /// Cycles.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

/// The least-squares amplitude of samples `x` under weights `w`, `sum(w x) / sum(w^2)`, summed
/// in order; 0 when every weight is 0.
pub fn ls_amplitude(xw: impl Iterator<Item = (f64, f64)>) -> f64 {
    let (mut swx, mut sww) = (0.0f64, 0.0f64);
    for (x, w) in xw {
        swx += w * x;
        sww += w * w;
    }
    if sww > 0.0 {
        swx / sww
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycles_are_spans_of_one_cycle_with_the_step_nearest_the_centre() {
        let place = [
            (5, 599.0),
            (5, 599.8),
            (5, 600.5),
            (6, 599.0),
            (6, 599.8),
            (6, 600.5),
            (7, 600.5),
        ];
        let c = CycleSpans::of(&place, 600.3).expect("cycles");
        assert_eq!(c.spans, vec![(0, 3), (3, 6), (6, 7)]);
        assert_eq!(c.rep, vec![2, 5, 6]);
        assert_eq!(c.len(), 3);
        // Outside the sweep, or a cycle that comes back: no cycles.
        assert!(CycleSpans::of(&[(5, 1.0), (u32::MAX, 1.0)], 1.0).is_none());
        assert!(CycleSpans::of(&[(5, 1.0), (6, 1.0), (5, 1.0)], 1.0).is_none());
        assert!(CycleSpans::of(&[], 1.0).is_none());
    }

    #[test]
    fn the_amplitude_is_the_least_squares_one() {
        // x = 3 w exactly.
        let x = [1.5, 3.0, 2.4];
        let w = [0.5, 1.0, 0.8];
        let a = ls_amplitude(x.iter().copied().zip(w.iter().copied()));
        assert!((a - 3.0).abs() < 1e-12);
        assert_eq!(ls_amplitude([(1.0, 0.0)].into_iter()), 0.0);
    }
}
