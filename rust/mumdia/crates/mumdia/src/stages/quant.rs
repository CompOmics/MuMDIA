//! Stage G `mumdia quant` (docs/12_quant_lfq_align_mbr_report_audit.md): quantify
//! identified peptidoforms and roll up to protein groups. Integrate each fragment
//! chromatogram over the apex region by the trapezoidal rule, sum the top-N
//! fragments into a per-run peptidoform quantity, then roll up to protein groups.
//! MVP is single-run, so cross-run normalization and MaxLFQ/directLFQ (which need
//! multiple runs) reduce to a top-N sum; the method is a config strategy for
//! later.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use anyhow::Result;
use mumdia_core::config::{
    FragmentSelection, NormalizeMethod, PeakWindowMode, QuantConfig, QuantQColumn, RollupMethod,
};
use mumdia_core::schema::artifact;
use mumdia_io::report::ArtifactReport;
use mumdia_io::table::{column_names, write_table, Col, Table};
use rayon::prelude::*;
use serde_json::json;
use tracing::{info, warn};

pub struct QuantParams<'a> {
    pub psms_scored: &'a str,
    pub chromatograms: &'a str,
    pub out_peptide: &'a str,
    pub out_protein: &'a str,
    /// Optional per-fragment area export (for ion-level directLFQ across runs).
    pub out_fragment: Option<&'a str>,
    /// Optional per-candidate peak-window diagnostic (candidate_id, lo_rt, hi_rt,
    /// width_s). Emitted only when `bound_peak` is on; a diagnostic of the
    /// integration windows, not part of the quant contract.
    pub out_peak_bounds: Option<&'a str>,
    pub cfg: &'a QuantConfig,
    pub config_hash: &'a str,
}

/// Trapezoidal integral of an intensity trace over RT (seconds). A single point
/// yields its raw intensity.
fn trapezoid(rt: &[f32], inten: &[f32]) -> f64 {
    if rt.len() < 2 {
        return inten.first().copied().unwrap_or(0.0) as f64;
    }
    let mut area = 0.0f64;
    for i in 0..rt.len() - 1 {
        let dt = (rt[i + 1] - rt[i]) as f64;
        area += dt * (inten[i] + inten[i + 1]) as f64 * 0.5;
    }
    area
}

/// Apex-outward interference-correction envelope: walking outward from the apex
/// (the max sample) in each direction, cap every sample at the running minimum so
/// far. A co-eluting interferent that lifts the peak wings back up is clipped to
/// the trough between it and the true peak, so it no longer inflates the
/// integrated area. Returns the corrected intensities aligned to the input.
/// Behavior-preserving when there is no wing interference (a clean monotone peak
/// is unchanged). Deterministic.
fn center_envelope_1d(inten: &[f32]) -> Vec<f32> {
    let n = inten.len();
    if n < 3 {
        return inten.to_vec();
    }
    let apex = (0..n).fold(0usize, |b, i| if inten[i] > inten[b] { i } else { b });
    let mut out = inten.to_vec();
    let mut m = inten[apex];
    for i in (0..apex).rev() {
        m = m.min(inten[i]);
        out[i] = m;
    }
    let mut m = inten[apex];
    for i in (apex + 1)..n {
        m = m.min(inten[i]);
        out[i] = m;
    }
    out
}

/// Trapezoidal integral of one fragment trace restricted to RT in `[lo, hi]`.
/// Reuses [`trapezoid`] on the in-window samples so the single-sample rule is
/// identical; an empty window integrates to 0. When `envelope` is set, the
/// in-window intensities are passed through [`center_envelope_1d`] first to strip
/// co-eluting interference in the peak wings before integration.
fn trapezoid_window(rt: &[f32], inten: &[f32], lo: f64, hi: f64, envelope: bool) -> f64 {
    let mut wr: Vec<f32> = Vec::new();
    let mut wi: Vec<f32> = Vec::new();
    for k in 0..rt.len() {
        let r = rt[k] as f64;
        if r >= lo && r <= hi {
            wr.push(rt[k]);
            wi.push(inten[k]);
        }
    }
    if wr.is_empty() {
        return 0.0;
    }
    if envelope {
        let ev = center_envelope_1d(&wi);
        trapezoid(&wr, &ev)
    } else {
        trapezoid(&wr, &wi)
    }
}

/// Background level for a fixed-scan window `[lo, hi)`: the `quantile` quantile of
/// the intensities in the flanks (`flank` samples on each side, clipped to the
/// trace). Returns 0 when no flank sample exists.
fn flank_baseline(inten: &[f32], lo: usize, hi: usize, flank: usize, quantile: f64) -> f32 {
    let mut v: Vec<f32> = Vec::with_capacity(2 * flank);
    let fl = lo.saturating_sub(flank);
    v.extend_from_slice(&inten[fl..lo]);
    let fh = (hi + flank).min(inten.len());
    if hi < fh {
        v.extend_from_slice(&inten[hi..fh]);
    }
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let pos = ((v.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
    v[pos.min(v.len() - 1)]
}

/// Sample index range `[lo, hi)` covered by a fixed window centred on the sample
/// nearest to `apex`. `half_s > 0` selects the samples within `half_s` seconds of
/// the apex (always at least that nearest sample) and overrides `half`, which
/// otherwise takes `half` scans on each side. `None` for an empty trace.
///
/// Shared by the integration below and by the applied-window contract in [`run`],
/// so the bounds reported for a quantity are the bounds it was integrated over.
fn fixed_window_indices(rt: &[f32], apex: f64, half: usize, half_s: f64) -> Option<(usize, usize)> {
    if rt.is_empty() {
        return None;
    }
    let mut k = 0usize;
    let mut best = f64::INFINITY;
    for (i, &r) in rt.iter().enumerate() {
        let d = (r as f64 - apex).abs();
        if d < best {
            best = d;
            k = i;
        }
    }
    if half_s > 0.0 {
        let mut lo = k;
        while lo > 0 && (apex - rt[lo - 1] as f64) <= half_s {
            lo -= 1;
        }
        let mut hi = k + 1;
        while hi < rt.len() && (rt[hi] as f64 - apex) <= half_s {
            hi += 1;
        }
        Some((lo, hi))
    } else {
        Some((k.saturating_sub(half), (k + half + 1).min(rt.len())))
    }
}

/// Fixed-window integration over the samples chosen by [`fixed_window_indices`],
/// with optional apex-outward envelope and optional flank-baseline subtraction
/// (`baseline = Some((flank, quantile))`). Empty trace integrates to 0.
fn trapezoid_fixed_opts(
    rt: &[f32],
    inten: &[f32],
    apex: f64,
    half: usize,
    half_s: f64,
    envelope: bool,
    baseline: Option<(usize, f64)>,
) -> f64 {
    let Some((lo, hi)) = fixed_window_indices(rt, apex, half, half_s) else {
        return 0.0;
    };
    let mut w: Vec<f32> = inten[lo..hi].to_vec();
    if let Some((flank, quantile)) = baseline {
        let b = flank_baseline(inten, lo, hi, flank, quantile);
        for x in w.iter_mut() {
            *x = (*x - b).max(0.0);
        }
    }
    if envelope {
        w = center_envelope_1d(&w);
    }
    trapezoid(&rt[lo..hi], &w)
}

/// Top-N sum with the fragment ranking chosen by `selection`. `observed_area`
/// delegates to [`summarize_fragment_areas`] (legacy, byte-identical); `predicted`
/// ranks the positive finite areas by library intensity and sums the top N.
fn select_fragment_areas(
    areas: Option<&[(f64, f32)]>,
    top_n: usize,
    selection: FragmentSelection,
) -> (Option<f64>, usize, &'static str) {
    match selection {
        FragmentSelection::ObservedArea => {
            let plain: Option<Vec<f64>> = areas.map(|a| a.iter().map(|x| x.0).collect());
            summarize_fragment_areas(plain.as_deref(), top_n)
        }
        FragmentSelection::Predicted => {
            let Some(areas) = areas else {
                return (None, 0, "no_fragment_traces");
            };
            let mut positive: Vec<(f64, f32)> = areas
                .iter()
                .copied()
                .filter(|(area, _)| area.is_finite() && *area > 0.0)
                .collect();
            if positive.is_empty() {
                return (None, 0, "no_positive_fragment_area");
            }
            if top_n == 0 {
                return (None, 0, "no_fragments_selected");
            }
            positive.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.0.total_cmp(&a.0)));
            let used = positive.len().min(top_n);
            let quantity: f64 = positive.iter().take(used).map(|x| x.0).sum();
            if quantity.is_finite() && quantity > 0.0 {
                (Some(quantity), used, "quantified")
            } else {
                (None, used, "nonfinite_quantity")
            }
        }
    }
}

/// Elution-peak RT window `[lo, hi]` for one candidate, from the summed XIC across
/// all its fragment chromatograms. Fragments are aligned on the union of their RT
/// samples via a BTreeMap keyed by the f32 RT bit pattern: for the non-negative
/// RTs here the bit order matches the value order, so both the union axis and the
/// f64 summation order are fixed (determinism,
/// docs/14_build_test_deploy_gotchas.md). When a finite identification apex is
/// available, the nearest sampled RT anchors the outward
/// [`super::features::peak_bounds`] walk. This prevents a brighter off-apex
/// interferent from moving quantification to a different peak. Older scored
/// artifacts and missing/non-finite hints retain the legacy co-elution apex
/// detector. Returns an unbounded window when there are fewer than two distinct
/// RT samples (nothing to bound).
fn peak_window(
    rows: &[usize],
    ch_rt: &[Vec<f32>],
    ch_int: &[Vec<f32>],
    frac: f64,
    grace: usize,
    apex_hint: Option<f64>,
) -> (f64, f64, f64) {
    let mut prof_map: BTreeMap<u32, f64> = BTreeMap::new();
    // Per-scan count of co-eluting (nonzero) fragments, aligned to prof_map keys.
    let mut cnt_map: BTreeMap<u32, u32> = BTreeMap::new();
    for &i in rows {
        let rts = &ch_rt[i];
        let ins = &ch_int[i];
        for k in 0..rts.len() {
            *prof_map.entry(rts[k].to_bits()).or_insert(0.0) += ins[k] as f64;
            if ins[k] > 0.0 {
                *cnt_map.entry(rts[k].to_bits()).or_insert(0) += 1;
            }
        }
    }
    if prof_map.len() < 2 {
        // Nothing to bound; apex is the lone RT if present, else NaN.
        let apex = prof_map
            .keys()
            .next()
            .map_or(f64::NAN, |b| f32::from_bits(*b) as f64);
        return (f64::NEG_INFINITY, f64::INFINITY, apex);
    }
    let axis: Vec<f64> = prof_map.keys().map(|b| f32::from_bits(*b) as f64).collect();
    let prof: Vec<f64> = prof_map.values().cloned().collect();
    let cnt: Vec<u32> = prof_map
        .keys()
        .map(|b| *cnt_map.get(b).unwrap_or(&0))
        .collect();
    let ai = if let Some(hint) = apex_hint.filter(|v| v.is_finite()) {
        // The identified apex need not exactly equal a chromatogram sample (for
        // example after serialization/calibration), so anchor to the nearest RT.
        let mut nearest = 0usize;
        let mut distance = f64::INFINITY;
        for (i, &rt) in axis.iter().enumerate() {
            let d = (rt - hint).abs();
            if d < distance {
                nearest = i;
                distance = d;
            }
        }
        nearest
    } else {
        // Legacy robust apex: among scans whose co-eluting-fragment count is
        // within 1 of the maximum ("-1 for robustness"), take the one with the
        // highest summed intensity. This rejects a lone tall interferent fragment
        // in favor of a region where many fragments co-elute. Falls back to the
        // summed argmax only if no scan has a fragment.
        let max_cnt = cnt.iter().copied().max().unwrap_or(0);
        let thresh = max_cnt.saturating_sub(1).max(1);
        let mut ai = 0usize;
        let mut best = f64::NEG_INFINITY;
        let mut found = false;
        for (i, &v) in prof.iter().enumerate() {
            if cnt[i] >= thresh && v > best {
                best = v;
                ai = i;
                found = true;
            }
        }
        if !found {
            best = f64::NEG_INFINITY;
            for (i, &v) in prof.iter().enumerate() {
                if v > best {
                    best = v;
                    ai = i;
                }
            }
        }
        ai
    };
    let (mut lo, mut hi) = super::features::peak_bounds(&prof, ai, frac, grace);
    // Guarantee a nonzero-width window. A near-1-scan summed XIC (both apex
    // shoulders below the threshold) collapses to lo==hi; trapezoid_window would
    // then hit the single-sample rule and return the apex HEIGHT, not an area
    // (intensity x seconds), mixing units against broad-peak peptides in the same
    // run and corrupting relative/LFQ quantities. Widen to the adjacent grid scans
    // so at least two samples are always integrated (prof.len() >= 2 here).
    if lo == hi {
        if hi + 1 < prof.len() {
            hi += 1;
        }
        lo = lo.saturating_sub(1);
    }
    (axis[lo], axis[hi], axis[ai])
}

fn finite_option(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn passes_quant_filter(label: &str, q_value: f64, threshold: f64) -> bool {
    label != "decoy" && q_value.is_finite() && q_value <= threshold
}

/// Select positive, finite fragment areas and sum the top N. Missing traces and
/// all-zero/non-finite traces are deliberately nullable, not quantitative zero:
/// zero would be indistinguishable from a measured biological absence and would
/// bias protein rollups and downstream ratios.
fn summarize_fragment_areas(
    areas: Option<&[f64]>,
    top_n: usize,
) -> (Option<f64>, usize, &'static str) {
    let Some(areas) = areas else {
        return (None, 0, "no_fragment_traces");
    };
    let mut positive: Vec<f64> = areas
        .iter()
        .copied()
        .filter(|area| area.is_finite() && *area > 0.0)
        .collect();
    if positive.is_empty() {
        return (None, 0, "no_positive_fragment_area");
    }
    if top_n == 0 {
        return (None, 0, "no_fragments_selected");
    }
    positive.sort_by(|a, b| b.total_cmp(a));
    let used = positive.len().min(top_n);
    let quantity: f64 = positive.iter().take(used).sum();
    if quantity.is_finite() && quantity > 0.0 {
        (Some(quantity), used, "quantified")
    } else {
        (None, used, "nonfinite_quantity")
    }
}

type ProteinBaseQuant = BTreeMap<String, BTreeMap<u32, f64>>;

/// Record one identified row for protein rollup. Multiple charge/mod precursor
/// rows belonging to the same base peptide contribute only their maximum
/// quantity, preventing repeated identifications from inflating Top-N. This max
/// is a single-run representative only; proper cross-run abundance estimation
/// must combine per-run quant tables rather than roll pooled scored rows here.
fn add_protein_base_quantity(
    groups: &mut ProteinBaseQuant,
    protein_group: &str,
    base_peptide_id: u32,
    quantity: Option<f64>,
) {
    let bases = groups.entry(protein_group.to_string()).or_default();
    if let Some(quantity) = quantity.filter(|v| v.is_finite() && *v > 0.0) {
        bases
            .entry(base_peptide_id)
            .and_modify(|current| *current = current.max(quantity))
            .or_insert(quantity);
    }
}

/// Roll up unique quantifiable base peptides. `n_peptides` is the number of
/// unique positive bases before Top-N truncation, not the number of precursor
/// rows and not the number selected into the sum.
fn rollup_protein_bases(
    bases: &BTreeMap<u32, f64>,
    rollup: RollupMethod,
    top_n: usize,
) -> (Option<f64>, usize, &'static str) {
    if bases.is_empty() {
        return (None, 0, "no_quantifiable_peptide");
    }
    let mut values: Vec<f64> = bases.values().copied().collect();
    values.sort_by(|a, b| b.total_cmp(a));
    let quantity: f64 = match rollup {
        RollupMethod::TopNSum => values.iter().take(top_n).sum(),
        RollupMethod::Sum => values.iter().sum(),
    };
    if quantity.is_finite() && quantity > 0.0 {
        (Some(quantity), bases.len(), "quantified")
    } else {
        (None, bases.len(), "no_quantifiable_peptide")
    }
}

pub fn run(p: QuantParams) -> Result<(u64, u64)> {
    let t0 = Instant::now();

    // Identified target PSMs below the peptide q threshold.
    let ps = Table::read(p.psms_scored)?;
    let cid = ps.u32("candidate_id")?;
    let pform = ps.str("peptidoform")?;
    let charge = ps.i32("charge")?;
    let label = ps.str("label")?;
    let pg = ps.str("protein_group")?;
    let base = ps.u32("base_peptide_id")?;
    // Q-value column to filter on. Peptide/precursor q is per-run only when the
    // rescore itself is single-run; grouped q-values are experiment-wide otherwise.
    // For per-run slices of a pooled rescore, run_psm_q is the run-local FDR gate.
    let pep_q = match p.cfg.q_filter {
        QuantQColumn::PeptideQ => ps.f64("peptide_q_value")?,
        QuantQColumn::PrecursorQ => ps.f64("precursor_q")?,
        QuantQColumn::PsmQ => ps.f64("q_value")?,
        QuantQColumn::RunPsmQ => ps.f64("run_psm_q")?,
    };
    // `psms_scored` carries the exact identification apex (schema psms_scored v4;
    // the column has been present since v3). Older artifacts, or rows whose apex is
    // null (read as NaN) or non-finite, carry no hint, and quant then re-detects the
    // apex from the chromatogram itself.
    //
    // That fallback is not equivalent: re-detection reproduces the identification's
    // apex only about half the time (CLAUDE.md, "the selected apex was historically
    // correct/strongest only about 48-52% of the time"), so a quantity integrated
    // around a re-detected apex can belong to a different peak than the one that was
    // identified. It must therefore be visible rather than silent: warn, and record
    // the coverage in the artifact report so a downstream reader can tell which apex
    // source a quantity actually used.
    let mut apex_by_cid: HashMap<u32, f64> = HashMap::new();
    let apex_column_present = ps.f64("apex_rt").is_ok();
    if let Ok(apex_rt) = ps.f64("apex_rt") {
        for i in 0..ps.nrows {
            if apex_rt[i].is_finite() {
                apex_by_cid.entry(cid[i]).or_insert(apex_rt[i]);
            }
        }
    }
    if !apex_column_present {
        warn!(
            psms_scored = p.psms_scored,
            "quant: scored table has no apex_rt column (pre-v3 artifact); every              quantity will be integrated around a RE-DETECTED apex, which reproduces              the identification apex only about half the time"
        );
    }

    // Chromatograms grouped by candidate.
    // Project: quant reads at most five of the chromatogram table's seven columns, and the
    // table is the largest artifact in the run (tens of millions of rows with two big list
    // columns). Unprojected, frag_mz / frag_obs_mz were decoded and held for nothing.
    //
    // `predicted_intensity` is OPTIONAL. Chromatogram artifacts written before that column
    // existed do not carry it, and the default `observed_area` ranking never reads it, so
    // probe the footer (which decodes no data) and project only what is present. Demanding
    // the column unconditionally made every older artifact unquantifiable.
    let has_pred = column_names(p.chromatograms)?
        .iter()
        .any(|c| c == "predicted_intensity");
    if !has_pred && p.cfg.fragment_selection == FragmentSelection::Predicted {
        anyhow::bail!(
            "quant.fragment_selection = predicted ranks fragments by the \
             `predicted_intensity` column, which {} does not carry. Re-run `extract` to \
             write a current chromatogram artifact, or set \
             quant.fragment_selection = observed_area.",
            p.chromatograms
        );
    }
    let ch = if has_pred {
        Table::read_cols(
            p.chromatograms,
            &[
                "candidate_id",
                "frag_name",
                "predicted_intensity",
                "rt",
                "intensity",
            ],
        )?
    } else {
        Table::read_cols(
            p.chromatograms,
            &["candidate_id", "frag_name", "rt", "intensity"],
        )?
    };
    let ch_cid = ch.u32("candidate_id")?;
    let ch_name = ch.str("frag_name")?;
    // Zero, not NaN, for the absent column: the value is only a ranking key, and
    // `total_cmp` orders NaN above every real intensity, which would silently invert the
    // `predicted` ranking rather than fail.
    let ch_pred: Vec<f32> = if has_pred {
        ch.f32("predicted_intensity")?
    } else {
        vec![0.0; ch_cid.len()]
    };
    let ch_rt = ch.list_f32("rt")?;
    let ch_int = ch.list_f32("intensity")?;
    // Group the b/y fragment chromatogram rows by candidate. The MS1 isotope XIC
    // pseudo-traces (frag_name "ms1_*") are precursor channels, not fragment ions,
    // and are excluded from both the peak-window detection and the top-N sum. The
    // BTreeMap keeps candidate iteration order deterministic.
    let mut cand_rows: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for i in 0..ch.nrows {
        if ch_name[i].starts_with("ms1_") {
            continue;
        }
        cand_rows.entry(ch_cid[i]).or_default().push(i);
    }
    let mut areas: HashMap<u32, Vec<(f64, f32)>> = HashMap::new();
    // Store the fragment name by reference (borrowed from `ch_name`, which outlives
    // this map) to avoid a per-row String clone; it is materialized once at export.
    let mut frag_areas: HashMap<u32, Vec<(&str, f64)>> = HashMap::new();
    // Optional peak-window diagnostic: (candidate_id, lo_rt, hi_rt) for finite windows.
    let emit_bounds = p.out_peak_bounds.is_some() && p.cfg.bound_peak;
    let (mut pb_cid, mut pb_lo, mut pb_hi) = (Vec::new(), Vec::new(), Vec::new());

    // Phase 1: per-candidate summed-XIC window (lo_rt, hi_rt, apex_rt), anchored at
    // the identification apex when available and otherwise using the legacy robust
    // co-elution detector. Kept keyed by candidate for the consensus estimate.
    let mut win: BTreeMap<u32, (f64, f64, f64)> = BTreeMap::new();
    if p.cfg.bound_peak {
        // Each candidate's window depends only on its own chromatogram rows, so this is
        // embarrassingly parallel. Results are collected into a Vec and only then folded
        // into the BTreeMap, so the map is built from a deterministically ordered sequence
        // and every float inside `peak_window` is still reduced per candidate in the same
        // order as before -- bit-identical, not merely equivalent.
        let computed: Vec<(u32, (f64, f64, f64))> = cand_rows
            .par_iter()
            .map(|(&c, rows)| {
                (
                    c,
                    peak_window(
                        rows,
                        &ch_rt,
                        &ch_int,
                        p.cfg.peak_fraction,
                        p.cfg.peak_grace,
                        apex_by_cid.get(&c).copied(),
                    ),
                )
            })
            .collect();
        win.extend(computed);
    }

    // Consensus mode: peak width is a near-constant instrument/gradient property, so
    // take the median left/right half-width over CONFIDENT peptides (q <= reliable_q)
    // and apply it around each candidate's apex. The median ignores the interference-
    // stretched and collapsed per-candidate windows. It is estimated independently
    // for each quant invocation/run; cross-run-identical widths require an external
    // shared policy. Falls back to per-candidate if too few confident anchors.
    let consensus: Option<(f64, f64)> =
        if p.cfg.bound_peak && p.cfg.peak_window_mode == PeakWindowMode::Consensus {
            let mut q_by_cid: HashMap<u32, f64> = HashMap::new();
            for i in 0..ps.nrows {
                if label[i] == "target" {
                    let e = q_by_cid.entry(cid[i]).or_insert(f64::INFINITY);
                    if pep_q[i] < *e {
                        *e = pep_q[i];
                    }
                }
            }
            let (mut left, mut right) = (Vec::new(), Vec::new());
            for (c, &(lo, hi, apex)) in &win {
                if lo.is_finite()
                    && hi.is_finite()
                    && apex.is_finite()
                    && q_by_cid.get(c).is_some_and(|&q| q <= p.cfg.reliable_q)
                {
                    left.push(apex - lo);
                    right.push(hi - apex);
                }
            }
            if left.len() >= 20 {
                let ml = median_sorted(&mut left);
                let mr = median_sorted(&mut right);
                info!(
                    anchors = left.len(),
                    med_left_s = ml,
                    med_right_s = mr,
                    "quant: consensus peak window"
                );
                Some((ml, mr))
            } else {
                info!(
                    anchors = left.len(),
                    "quant: too few confident anchors, using per-candidate windows"
                );
                None
            }
        } else {
            None
        };

    // Phase 2: integrate each fragment over the chosen window and retain the
    // actually applied apex/bounds for the peptide-quant contract.
    let mut applied_win: BTreeMap<u32, (f64, f64, f64)> = BTreeMap::new();
    // `(candidate_id, (lo_rt, hi_rt, integration_apex), one area per chromatogram row)`.
    type Integrated = (u32, (f64, f64, f64), Vec<f64>);
    // Integrate each candidate's fragment traces. Parallel across candidates for the same
    // reason the peak-window phase above is: a candidate reads only its own chromatogram
    // rows and every float reduction happens inside one candidate's `trapezoid*` call.
    // Rayon's indexed `collect` keeps candidate order, and `cand_rows` is a `BTreeMap`, so
    // the maps below are filled in exactly the order the serial loop filled them.
    let integrated: Vec<Integrated> = cand_rows
        .par_iter()
        .map(|(&c, rows)| {
            let (lo_rt, hi_rt, integration_apex) = if !p.cfg.bound_peak {
                (f64::NEG_INFINITY, f64::INFINITY, f64::NAN)
            } else {
                let (lo, hi, apex) = win[&c];
                match consensus {
                    Some((ml, mr)) if apex.is_finite() => (apex - ml, apex + mr, apex),
                    _ => (lo, hi, apex),
                }
            };
            // A fixed window replaces the walked bounds entirely; it needs a finite apex
            // to centre on, so an unknown apex falls back to the configured window.
            let fixed = (p.cfg.fixed_scan_halfwidth > 0 || p.cfg.fixed_window_s > 0.0)
                && integration_apex.is_finite();
            let a: Vec<f64> = rows
                .iter()
                .map(|&i| {
                    if fixed {
                        trapezoid_fixed_opts(
                            &ch_rt[i],
                            &ch_int[i],
                            integration_apex,
                            p.cfg.fixed_scan_halfwidth,
                            p.cfg.fixed_window_s,
                            p.cfg.interference_envelope,
                            if p.cfg.baseline_subtract {
                                Some((p.cfg.baseline_flank_scans, p.cfg.baseline_quantile))
                            } else {
                                None
                            },
                        )
                    } else if p.cfg.bound_peak {
                        trapezoid_window(
                            &ch_rt[i],
                            &ch_int[i],
                            lo_rt,
                            hi_rt,
                            p.cfg.interference_envelope,
                        )
                    } else {
                        trapezoid(&ch_rt[i], &ch_int[i])
                    }
                })
                .collect();
            // Applied-window contract: under a fixed window the walked bounds are NOT the
            // integration range, so report the RT extent actually covered (union over this
            // candidate's traces, whose sample grids may differ). Otherwise
            // `integration_lo_rt`/`integration_hi_rt` and the peak-bounds diagnostic would
            // describe a window that produced no part of `quantity`.
            let (lo_rt, hi_rt) = if fixed {
                let mut flo = f64::INFINITY;
                let mut fhi = f64::NEG_INFINITY;
                for &i in rows.iter() {
                    if let Some((lo, hi)) = fixed_window_indices(
                        &ch_rt[i],
                        integration_apex,
                        p.cfg.fixed_scan_halfwidth,
                        p.cfg.fixed_window_s,
                    ) {
                        flo = flo.min(ch_rt[i][lo] as f64);
                        fhi = fhi.max(ch_rt[i][hi - 1] as f64);
                    }
                }
                if flo.is_finite() && fhi.is_finite() {
                    (flo, fhi)
                } else {
                    (lo_rt, hi_rt)
                }
            } else {
                (lo_rt, hi_rt)
            };
            (c, (lo_rt, hi_rt, integration_apex), a)
        })
        .collect();
    for ((c, w, computed), rows) in integrated.into_iter().zip(cand_rows.values()) {
        let (lo_rt, hi_rt, _) = w;
        applied_win.insert(c, w);
        if emit_bounds && lo_rt.is_finite() && hi_rt.is_finite() {
            pb_cid.push(c);
            pb_lo.push(lo_rt);
            pb_hi.push(hi_rt);
        }
        for (&i, a) in rows.iter().zip(computed) {
            areas.entry(c).or_default().push((a, ch_pred[i]));
            frag_areas
                .entry(c)
                .or_default()
                .push((ch_name[i].as_str(), a));
        }
    }

    // Optional peak-window diagnostic export.
    if let Some(pbpath) = p.out_peak_bounds {
        let width: Vec<f64> = pb_lo.iter().zip(&pb_hi).map(|(l, h)| h - l).collect();
        write_table(
            pbpath,
            vec![
                Col::U32("candidate_id".into(), pb_cid),
                Col::F64("lo_rt".into(), pb_lo),
                Col::F64("hi_rt".into(), pb_hi),
                Col::F64("width_s".into(), width),
            ],
        )?;
    }

    // Per-peptidoform quantity = sum of the top-N positive fragment areas.
    let (
        mut q_cid,
        mut q_base,
        mut q_pform,
        mut q_z,
        mut q_pg,
        mut q_val,
        mut q_status,
        mut q_nfrag,
        mut q_apex,
        mut q_lo,
        mut q_hi,
    ) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let mut per_group: ProteinBaseQuant = BTreeMap::new();
    let mut n_quantified_peptides = 0u64;
    for i in 0..ps.nrows {
        if !passes_quant_filter(&label[i], pep_q[i], p.cfg.q_threshold) {
            continue;
        }
        let (quantity, used, status) = select_fragment_areas(
            areas.get(&cid[i]).map(Vec::as_slice),
            p.cfg.top_n_fragments,
            p.cfg.fragment_selection,
        );
        if quantity.is_some() {
            n_quantified_peptides += 1;
        }
        let (integration_lo, integration_hi, integration_apex) = match applied_win.get(&cid[i]) {
            Some(&(lo, hi, apex)) => (finite_option(lo), finite_option(hi), finite_option(apex)),
            None => (None, None, None),
        };
        q_cid.push(cid[i]);
        q_base.push(base[i]);
        q_pform.push(pform[i].clone());
        q_z.push(charge[i]);
        q_pg.push(pg[i].clone());
        q_val.push(quantity);
        q_status.push(status.to_string());
        q_nfrag.push(used as i32);
        q_apex.push(integration_apex);
        q_lo.push(integration_lo);
        q_hi.push(integration_hi);
        add_protein_base_quantity(&mut per_group, &pg[i], base[i], quantity);
    }

    let n_pep = write_table(
        p.out_peptide,
        vec![
            Col::U32("candidate_id".into(), q_cid),
            Col::U32("base_peptide_id".into(), q_base),
            Col::Str("peptidoform".into(), q_pform),
            Col::I32("charge".into(), q_z),
            Col::Str("protein_group".into(), q_pg),
            Col::OptF64("quantity".into(), q_val),
            Col::Str("quant_status".into(), q_status),
            Col::I32("n_fragments_used".into(), q_nfrag),
            Col::OptF64("integration_apex_rt".into(), q_apex),
            Col::OptF64("integration_lo_rt".into(), q_lo),
            Col::OptF64("integration_hi_rt".into(), q_hi),
        ],
    )?;

    // Protein-group rollup over unique, quantifiable base peptides only.
    let (mut g_name, mut g_val, mut g_status, mut g_npep) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut n_quantified_protein_groups = 0u64;
    for (group, bases) in &per_group {
        let (quantity, n_bases, status) =
            rollup_protein_bases(bases, p.cfg.rollup, p.cfg.top_n_peptides);
        if quantity.is_some() {
            n_quantified_protein_groups += 1;
        }
        g_name.push(group.clone());
        g_val.push(quantity);
        g_status.push(status.to_string());
        g_npep.push(n_bases as i32);
    }
    let n_pg = write_table(
        p.out_protein,
        vec![
            Col::Str("protein_group".into(), g_name),
            Col::OptF64("quantity".into(), g_val),
            Col::Str("quant_status".into(), g_status),
            Col::I32("n_peptides".into(), g_npep),
        ],
    )?;

    // Optional per-fragment area export for ion-level directLFQ across runs.
    let mut fragment_output: Option<(&str, u64)> = None;
    if let Some(fpath) = p.out_fragment {
        let (mut f_cid, mut f_pf, mut f_z, mut f_pg, mut f_name, mut f_area) = (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        for i in 0..ps.nrows {
            if !passes_quant_filter(&label[i], pep_q[i], p.cfg.q_threshold) {
                continue;
            }
            if let Some(fa) = frag_areas.get(&cid[i]) {
                for (nm, a) in fa {
                    if !a.is_finite() || *a <= 0.0 {
                        continue;
                    }
                    f_cid.push(cid[i]);
                    f_pf.push(pform[i].clone());
                    f_z.push(charge[i]);
                    f_pg.push(pg[i].clone());
                    f_name.push(nm.to_string());
                    f_area.push(*a);
                }
            }
        }
        let fragment_rows = write_table(
            fpath,
            vec![
                Col::U32("candidate_id".into(), f_cid),
                Col::Str("peptidoform".into(), f_pf),
                Col::I32("charge".into(), f_z),
                Col::Str("protein_group".into(), f_pg),
                Col::Str("fragment_name".into(), f_name),
                Col::F64("quantity".into(), f_area),
            ],
        )?;
        fragment_output = Some((fpath, fragment_rows));
    }

    let elapsed = t0.elapsed().as_millis();
    let mut stats = std::collections::BTreeMap::new();
    stats.insert("peptide_rows".to_string(), json!(n_pep));
    stats.insert(
        "quantified_peptides".to_string(),
        json!(n_quantified_peptides),
    );
    stats.insert(
        "nonquantifiable_peptides".to_string(),
        json!(n_pep.saturating_sub(n_quantified_peptides)),
    );
    stats.insert("protein_group_rows".to_string(), json!(n_pg));
    stats.insert(
        "quantified_protein_groups".to_string(),
        json!(n_quantified_protein_groups),
    );
    let report_params = json!({
        "q_threshold": p.cfg.q_threshold,
        "top_n_fragments": p.cfg.top_n_fragments,
        "fragment_selection": format!("{:?}", p.cfg.fragment_selection),
        "fixed_scan_halfwidth": p.cfg.fixed_scan_halfwidth,
        "fixed_window_s": p.cfg.fixed_window_s,
        "baseline_subtract": p.cfg.baseline_subtract,
        "baseline_flank_scans": p.cfg.baseline_flank_scans,
        "baseline_quantile": p.cfg.baseline_quantile,
        "top_n_peptides": p.cfg.top_n_peptides,
        "rollup": format!("{:?}", p.cfg.rollup),
        "bound_peak": p.cfg.bound_peak,
        "peak_fraction": p.cfg.peak_fraction,
        "peak_grace": p.cfg.peak_grace,
        "peak_window_mode": format!("{:?}", p.cfg.peak_window_mode),
        "reliable_q": p.cfg.reliable_q,
        "q_filter": format!("{:?}", p.cfg.q_filter),
        "config_hash": p.config_hash,
        "psms_scored": p.psms_scored,
        "chromatograms": p.chromatograms,
        // Which apex each quantity was integrated around: `scored_apex` rows reuse the
        // identification apex, `redetected` rows fell back to quant's own peak pick.
        "apex_rt_column_present": apex_column_present,
        "candidates_with_scored_apex": apex_by_cid.len(),
    });
    for (path, schema, rows) in [
        (p.out_peptide, artifact::PEPTIDE_QUANT, n_pep),
        (p.out_protein, artifact::PROTEIN_GROUP_QUANT, n_pg),
    ] {
        ArtifactReport {
            logical_name: schema.0.to_string(),
            schema_name: schema.0.to_string(),
            schema_version: schema.1,
            stage: "quant".to_string(),
            rows,
            content_hash: mumdia_io::hash::blake3_file(path)?,
            params: report_params.clone(),
            stats: stats.clone(),
            model_identity: None,
            elapsed_ms: elapsed,
        }
        .write_for(path)?;
    }
    if let Some((path, rows)) = fragment_output {
        ArtifactReport {
            logical_name: artifact::FRAGMENT_QUANT.0.to_string(),
            schema_name: artifact::FRAGMENT_QUANT.0.to_string(),
            schema_version: artifact::FRAGMENT_QUANT.1,
            stage: "quant".to_string(),
            rows,
            content_hash: mumdia_io::hash::blake3_file(path)?,
            params: report_params,
            stats: stats.clone(),
            model_identity: None,
            elapsed_ms: elapsed,
        }
        .write_for(path)?;
    }

    info!(
        peptide_rows = n_pep,
        quantified_peptides = n_quantified_peptides,
        protein_group_rows = n_pg,
        quantified_protein_groups = n_quantified_protein_groups,
        elapsed_ms = elapsed,
        "quant: done"
    );
    Ok((n_pep, n_pg))
}

/// Combine several per-run quant tables into a protein-by-run abundance matrix
/// with MaxLFQ (peptide-level: `by_fragment=false`, reads `peptide_quant`) or
/// directLFQ (ion-level: `by_fragment=true`, reads `fragment_quant`). Each
/// protein's feature-by-run intensity matrix is passed to the ratio-alignment
/// core in [`crate::quant_lfq`]. Output is long form: protein_group, run,
/// quantity, n_features. With one input this reduces to the per-run sum.
///
/// `normalize` applies a cross-run size factor to the feature-by-run matrix
/// before rollup (see [`size_factors`]).
pub fn run_lfq_combine(
    inputs: &[String],
    by_fragment: bool,
    normalize: NormalizeMethod,
    out: &str,
) -> Result<u64> {
    use std::collections::BTreeMap;
    let n = inputs.len();
    // protein_group -> feature key -> per-run intensity
    let mut data: BTreeMap<String, BTreeMap<String, Vec<Option<f64>>>> = BTreeMap::new();
    for (ri, path) in inputs.iter().enumerate() {
        let t = Table::read(path)?;
        let pform = t.str("peptidoform")?;
        let z = t.i32("charge")?;
        let pgc = t.str("protein_group")?;
        let q = t.opt_f64("quantity")?;
        let fname = if by_fragment {
            Some(t.str("fragment_name")?)
        } else {
            None
        };
        for i in 0..t.nrows {
            let Some(quantity) = q[i].filter(|v| v.is_finite() && *v > 0.0) else {
                continue;
            };
            let key = match &fname {
                Some(fnm) => format!("{}|{}|{}", pform[i], z[i], fnm[i]),
                None => format!("{}|{}", pform[i], z[i]),
            };
            let slot = &mut data
                .entry(pgc[i].clone())
                .or_default()
                .entry(key)
                .or_insert_with(|| vec![None; n])[ri];
            *slot = Some(slot.map_or(quantity, |previous| previous.max(quantity)));
        }
    }
    // Cross-run normalization: one global size factor per run, estimated from the
    // whole feature-by-run matrix and applied before protein rollup so both the
    // MaxLFQ profile and any downstream ratio inherit the corrected scale.
    let factors = size_factors(&data, n, normalize);
    if normalize != NormalizeMethod::None {
        for feats in data.values_mut() {
            for vec in feats.values_mut() {
                for r in 0..n {
                    if let Some(v) = vec[r] {
                        vec[r] = Some(v / factors[r]);
                    }
                }
            }
        }
    }
    let (mut c_pg, mut c_run, mut c_q, mut c_nf) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (pgname, feats) in &data {
        let mat: Vec<Vec<Option<f64>>> = feats.values().cloned().collect();
        let prof = crate::quant_lfq::lfq_profile(&mat, n);
        for (r, &v) in prof.iter().enumerate() {
            c_pg.push(pgname.clone());
            c_run.push(r as i32);
            c_q.push(v);
            c_nf.push(feats.len() as i32);
        }
    }
    let rows = write_table(
        out,
        vec![
            Col::Str("protein_group".into(), c_pg),
            Col::I32("run".into(), c_run),
            Col::F64("quantity".into(), c_q),
            Col::I32("n_features".into(), c_nf),
        ],
    )?;

    // Peptide- and precursor-level matrices from the SAME normalized features,
    // written as sibling files next to the protein matrix (the protein output is
    // unchanged). Precursor = one (peptidoform, charge); peptide = one stripped
    // base sequence. Both roll their member features up with the same LFQ engine.
    // Purely additive analysis granularity; strictly post-FDR, no identification
    // or FDR change.
    let mut prec: BTreeMap<(String, i32), Vec<Vec<Option<f64>>>> = BTreeMap::new();
    let mut pep: BTreeMap<String, Vec<Vec<Option<f64>>>> = BTreeMap::new();
    for feats in data.values() {
        for (key, vec) in feats {
            // key = "peptidoform|charge" (maxlfq) or "peptidoform|charge|fragment"
            // (directlfq); peptidoform strings never contain '|'.
            let mut it = key.splitn(3, '|');
            let pform = it.next().unwrap_or("").to_string();
            let charge: i32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            prec.entry((pform.clone(), charge))
                .or_default()
                .push(vec.clone());
            pep.entry(base_sequence(&pform))
                .or_default()
                .push(vec.clone());
        }
    }
    // (group key, charge, feature-by-run matrix) for one sibling-matrix level.
    type LevelGroup = (String, i32, Vec<Vec<Option<f64>>>);
    let write_level = |path: String, groups: Vec<LevelGroup>| -> Result<()> {
        let (mut g_key, mut g_z, mut g_run, mut g_q, mut g_nf) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (key, z, mat) in &groups {
            let prof = crate::quant_lfq::lfq_profile(mat, n);
            for (r, &v) in prof.iter().enumerate() {
                g_key.push(key.clone());
                g_z.push(*z);
                g_run.push(r as i32);
                g_q.push(v);
                g_nf.push(mat.len() as i32);
            }
        }
        write_table(
            &path,
            vec![
                Col::Str("group".into(), g_key),
                Col::I32("charge".into(), g_z),
                Col::I32("run".into(), g_run),
                Col::F64("quantity".into(), g_q),
                Col::I32("n_features".into(), g_nf),
            ],
        )?;
        Ok(())
    };
    write_level(
        format!("{out}.precursor.parquet"),
        prec.into_iter().map(|((p, z), m)| (p, z, m)).collect(),
    )?;
    write_level(
        format!("{out}.peptide.parquet"),
        pep.into_iter().map(|(p, m)| (p, -1, m)).collect(),
    )?;

    info!(
        proteins = data.len(),
        runs = n,
        method = if by_fragment { "directlfq" } else { "maxlfq" },
        normalize = ?normalize,
        size_factors = ?factors,
        "quant-lfq: done (+ .peptide/.precursor sibling matrices)"
    );
    Ok(rows)
}

/// Stripped base amino-acid sequence of a peptidoform: drop bracketed or
/// parenthesized modification blocks and any DECOY_ prefix, keep the residues.
/// Used to roll precursors up to peptide-level LFQ groups.
fn base_sequence(peptidoform: &str) -> String {
    let s = peptidoform.strip_prefix("DECOY_").unwrap_or(peptidoform);
    let mut out = String::new();
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = (depth - 1).max(0),
            c if depth == 0 && c.is_ascii_alphabetic() => out.push(c),
            _ => {}
        }
    }
    out
}

/// Per-run size factors for cross-run normalization of the feature-by-run matrix.
///
/// - `MedianRatio` (DESeq-style): for every complete-case feature (present and
///   positive in all runs) take its ratio to a geometric-mean pseudo-reference;
///   the run factor is the median of those ratios. Robust to a minority of
///   genuinely changing features, so a spike-in design's real fold changes are
///   preserved, not flattened.
/// - `Median`: align each run's median log2 intensity to the median of the
///   per-run medians.
/// - `None`: all factors 1.0.
///
/// Medians are taken over sorted values and the matrix is iterated in `BTreeMap`
/// key order, so the result is deterministic (docs/14_build_test_deploy_gotchas.md).
fn size_factors(
    data: &std::collections::BTreeMap<String, std::collections::BTreeMap<String, Vec<Option<f64>>>>,
    n: usize,
    method: NormalizeMethod,
) -> Vec<f64> {
    match method {
        NormalizeMethod::None => vec![1.0; n],
        NormalizeMethod::MedianRatio => {
            let mut lr: Vec<Vec<f64>> = vec![Vec::new(); n];
            for feats in data.values() {
                for vec in feats.values() {
                    if vec.iter().all(|x| x.is_some_and(|v| v > 0.0)) {
                        let logs: Vec<f64> = vec.iter().map(|x| x.unwrap().log2()).collect();
                        let refm = logs.iter().sum::<f64>() / n as f64;
                        for r in 0..n {
                            lr[r].push(logs[r] - refm);
                        }
                    }
                }
            }
            (0..n)
                .map(|r| {
                    if lr[r].is_empty() {
                        1.0
                    } else {
                        2f64.powf(median_sorted(&mut lr[r]))
                    }
                })
                .collect()
        }
        NormalizeMethod::Median => {
            let mut logs: Vec<Vec<f64>> = vec![Vec::new(); n];
            for feats in data.values() {
                for vec in feats.values() {
                    for r in 0..n {
                        if let Some(v) = vec[r] {
                            if v > 0.0 {
                                logs[r].push(v.log2());
                            }
                        }
                    }
                }
            }
            let med: Vec<f64> = (0..n)
                .map(|r| {
                    if logs[r].is_empty() {
                        0.0
                    } else {
                        median_sorted(&mut logs[r])
                    }
                })
                .collect();
            let mut m2 = med.clone();
            let target = if m2.is_empty() {
                0.0
            } else {
                median_sorted(&mut m2)
            };
            (0..n).map(|r| 2f64.powf(med[r] - target)).collect()
        }
    }
}

/// Median of a slice, sorting in place (ascending). Even lengths average the two
/// middle values. Empty slice returns 0.0. Values are assumed finite (no NaN).
fn median_sorted(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len();
    if m % 2 == 1 {
        v[m / 2]
    } else {
        (v[m / 2 - 1] + v[m / 2]) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quant_test_path(name: &str) -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!("mumdia_quant_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        dir.join(format!("{n}_{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn trapezoid_area() {
        // triangle: rt 0,1,2 ; int 0,2,0 -> area = 2
        assert!((trapezoid(&[0.0, 1.0, 2.0], &[0.0, 2.0, 0.0]) - 2.0).abs() < 1e-9);
        // single point -> raw intensity
        assert_eq!(trapezoid(&[5.0], &[7.0]), 7.0);
    }

    #[test]
    fn trapezoid_window_clips_to_range() {
        let rt = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        let it = [0.0f32, 5.0, 10.0, 5.0, 0.0];
        // Whole trace: symmetric triangle, area = 20.
        assert!((trapezoid(&rt, &it) - 20.0).abs() < 1e-9);
        // Restricted to [1,3]: samples (1,5),(2,10),(3,5) -> 7.5 + 7.5 = 15.
        assert!((trapezoid_window(&rt, &it, 1.0, 3.0, false) - 15.0).abs() < 1e-9);
        // Window with a single in-range sample returns that raw intensity.
        assert_eq!(trapezoid_window(&rt, &it, 2.0, 2.0, false), 10.0);
        // Empty window integrates to 0.
        assert_eq!(trapezoid_window(&rt, &it, 10.0, 20.0, false), 0.0);
    }

    #[test]
    fn base_sequence_strips_mods_and_decoy() {
        assert_eq!(base_sequence("PEPTIDEK"), "PEPTIDEK");
        assert_eq!(base_sequence("M[Oxidation]PEC[Carbamidomethyl]K"), "MPECK");
        assert_eq!(base_sequence("DECOY_VAVGDGVAK"), "VAVGDGVAK");
    }

    #[test]
    fn center_envelope_clips_wing_interference() {
        // A clean rise-then-fall peak is left unchanged.
        let clean = [1.0f32, 3.0, 6.0, 3.0, 1.0];
        assert_eq!(center_envelope_1d(&clean), clean.to_vec());
        // Interference bump in the right wing (idx4 rises back to 5.0 after the
        // trough at idx3=2.0): apex idx2, the outward running-min caps it to 2.0.
        let interf = [1.0f32, 4.0, 10.0, 2.0, 5.0, 1.0];
        assert_eq!(
            center_envelope_1d(&interf),
            vec![1.0, 4.0, 10.0, 2.0, 2.0, 1.0]
        );
        // Enabling the envelope removes the bump, so the integrated area shrinks.
        let rt = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let off = trapezoid_window(&rt, &interf, 0.0, 5.0, false);
        let on = trapezoid_window(&rt, &interf, 0.0, 5.0, true);
        assert!(
            on < off,
            "envelope should not increase the area (on={on}, off={off})"
        );
    }

    #[test]
    fn peak_window_bounds_summed_xic_and_rejects_lone_interferent() {
        // Two fragments share an RT grid. Fragment 0 is the real co-eluting peptide
        // peaking at rt=6; fragment 1 is a lone interferent spiking at rt=1, well
        // separated (>= 2 zero scans) so the grace walk cannot bridge to it. The
        // SUMMED XIC apex lands on rt=6 and the window brackets the real peak only,
        // even though the interferent is tall.
        let grid: Vec<f32> = (0..12).map(|k| k as f32).collect();
        let real = vec![
            0.0f32, 0.0, 0.0, 0.0, 2.0, 6.0, 10.0, 6.0, 2.0, 0.0, 0.0, 0.0,
        ];
        let interf = vec![
            0.0f32, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let ch_rt = vec![grid.clone(), grid.clone()];
        let ch_int = vec![real, interf];
        let (lo, hi, _) = peak_window(&[0, 1], &ch_rt, &ch_int, 1.0 / 6.0, 1, None);
        // Apex rt=6 (sum=10); 1/6 threshold ~1.67. Left: idx4(2)>=thr, idx3/idx2=0
        // -> 2 consecutive misses stop at rt=4. Right: symmetric stop at rt=8.
        assert_eq!(lo, 4.0);
        assert_eq!(hi, 8.0);
        // The interferent spike at rt=1 is outside [lo,hi], so its windowed area is 0.
        assert_eq!(trapezoid_window(&ch_rt[1], &ch_int[1], lo, hi, false), 0.0);
    }

    #[test]
    fn peak_window_grace_bridges_single_dip() {
        // Summed profile with a single-scan dip below threshold on the right shoulder,
        // then recovery. grace=1 must bridge the dip; grace=0 must stop at it.
        // apex=10 at idx4; threshold at 1/3 -> 3.33. Right side: 5,1(dip),5,0.
        let grid: Vec<f32> = (0..9).map(|k| k as f32).collect();
        let prof = vec![0.0f32, 0.0, 1.0, 5.0, 10.0, 5.0, 1.0, 5.0, 0.0];
        let ch_rt = vec![grid.clone()];
        let ch_int = vec![prof];
        let (_, hi1, _) = peak_window(&[0], &ch_rt, &ch_int, 1.0 / 3.0, 1, None);
        let (_, hi0, _) = peak_window(&[0], &ch_rt, &ch_int, 1.0 / 3.0, 0, None);
        // grace=1 bridges the idx6 dip (1.0 < 3.33) and includes idx7 (5.0) -> rt 7.
        assert_eq!(hi1, 7.0);
        // grace=0 stops at the first sub-threshold scan -> last above-threshold rt 5.
        assert_eq!(hi0, 5.0);
    }

    #[test]
    fn median_sorted_odd_even_empty() {
        assert_eq!(median_sorted(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_sorted(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median_sorted(&mut []), 0.0);
    }

    #[test]
    fn median_ratio_recovers_global_scale_not_real_changes() {
        use std::collections::BTreeMap;
        // Two runs. Run 1 is a global 2x of run 0 for the bulk (unchanged) features,
        // plus one genuinely-up and one genuinely-down feature. Median-of-ratios must
        // recover the 2x global scale (f[1]/f[0] ~ 2) without being pulled by the two
        // real changes, and after dividing by the factors the bulk ratio -> 1 while
        // the real changes survive.
        let mut feats: BTreeMap<String, Vec<Option<f64>>> = BTreeMap::new();
        for i in 0..8 {
            let a = 100.0 * (i as f64 + 1.0);
            feats.insert(format!("bulk{i:02}"), vec![Some(a), Some(2.0 * a)]);
        }
        feats.insert("up".into(), vec![Some(100.0), Some(800.0)]); // +2 log2 vs global
        feats.insert("down".into(), vec![Some(400.0), Some(200.0)]); // -2 log2 vs global
        let mut data: BTreeMap<String, BTreeMap<String, Vec<Option<f64>>>> = BTreeMap::new();
        data.insert("PG".into(), feats);

        let f = size_factors(&data, 2, NormalizeMethod::MedianRatio);
        assert!(
            (f[1] / f[0] - 2.0).abs() < 0.02,
            "expected ~2x scale, got {f:?}"
        );
        // Bulk normalizes to ratio 1; the up/down real changes are preserved.
        let bulk = (100.0 / f[0], 200.0 / f[1]);
        assert!(
            (bulk.0 / bulk.1 - 1.0).abs() < 1e-9,
            "bulk should flatten to 1"
        );
        let up = (100.0 / f[0]) / (800.0 / f[1]);
        assert!(
            (up - 0.25).abs() < 1e-9,
            "up feature run0/run1 should stay 1:4"
        );
    }

    #[test]
    fn none_leaves_matrix_unnormalized() {
        use std::collections::BTreeMap;
        let mut feats: BTreeMap<String, Vec<Option<f64>>> = BTreeMap::new();
        feats.insert("f".into(), vec![Some(10.0), Some(40.0)]);
        let mut data: BTreeMap<String, BTreeMap<String, Vec<Option<f64>>>> = BTreeMap::new();
        data.insert("PG".into(), feats);
        assert_eq!(
            size_factors(&data, 2, NormalizeMethod::None),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn peak_window_apex_prefers_coelution_over_lone_interferent() {
        // Four real fragments co-elute at idx5 (each modest); one interferent fragment
        // has a lone tall spike at idx1. A plain summed-intensity argmax picks the
        // interferent (20 > 12); the co-elution rule ("-1 for robustness" on fragment
        // count) must pick idx5 (rt=5) where 4 fragments co-elute. This is the B-apex-
        // on-noise failure (candidate 7064964).
        let grid: Vec<f32> = (0..9).map(|k| k as f32).collect();
        let real = vec![0.0f32, 0.0, 0.0, 0.0, 2.0, 3.0, 2.0, 0.0, 0.0];
        let interf = vec![0.0f32, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ch_rt = vec![
            grid.clone(),
            grid.clone(),
            grid.clone(),
            grid.clone(),
            grid.clone(),
        ];
        let ch_int = vec![
            real.clone(),
            real.clone(),
            real.clone(),
            real.clone(),
            interf,
        ];
        let (_, _, apex) = peak_window(&[0, 1, 2, 3, 4], &ch_rt, &ch_int, 1.0 / 6.0, 1, None);
        assert_eq!(
            apex, 5.0,
            "apex must be the 4-fragment co-elution scan, not the lone interferent spike"
        );
    }

    #[test]
    fn identified_apex_anchors_window_against_brighter_off_apex_peak() {
        // All fragments share a bright interference peak at rt=1, so even the
        // legacy co-elution detector selects it. The identification apex at rt=5
        // must instead anchor the descent walk around the identified peak.
        let grid: Vec<f32> = (0..10).map(|k| k as f32).collect();
        let trace = vec![0.0f32, 50.0, 0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 0.0, 0.0];
        let ch_rt = vec![grid.clone(), grid.clone(), grid.clone()];
        let ch_int = vec![trace.clone(), trace.clone(), trace];

        let (_, _, legacy_apex) = peak_window(&[0, 1, 2], &ch_rt, &ch_int, 1.0 / 6.0, 1, None);
        assert_eq!(legacy_apex, 1.0);

        let (lo, hi, anchored_apex) =
            peak_window(&[0, 1, 2], &ch_rt, &ch_int, 1.0 / 6.0, 1, Some(5.1));
        assert_eq!(anchored_apex, 5.0);
        assert!(lo > 1.0 && hi >= 5.0, "anchored window was [{lo}, {hi}]");

        let (_, _, nonfinite_fallback) =
            peak_window(&[0, 1, 2], &ch_rt, &ch_int, 1.0 / 6.0, 1, Some(f64::NAN));
        assert_eq!(nonfinite_fallback, legacy_apex);
    }

    #[test]
    fn fragment_summary_distinguishes_missing_zero_and_positive_traces() {
        assert_eq!(
            summarize_fragment_areas(None, 3),
            (None, 0, "no_fragment_traces")
        );
        assert_eq!(
            summarize_fragment_areas(Some(&[0.0, -1.0, f64::NAN]), 3),
            (None, 0, "no_positive_fragment_area")
        );
        assert_eq!(
            summarize_fragment_areas(Some(&[0.0, 2.0, 5.0, f64::INFINITY]), 3),
            (Some(7.0), 2, "quantified")
        );
    }

    #[test]
    fn protein_rollup_uses_one_maximum_per_base_peptide() {
        let mut groups = ProteinBaseQuant::new();
        add_protein_base_quantity(&mut groups, "PG", 10, Some(12.0));
        add_protein_base_quantity(&mut groups, "PG", 10, Some(20.0));
        add_protein_base_quantity(&mut groups, "PG", 11, Some(5.0));
        add_protein_base_quantity(&mut groups, "PG", 12, None);

        let bases = &groups["PG"];
        assert_eq!(bases.len(), 2, "only quantifiable unique bases count");
        assert_eq!(
            rollup_protein_bases(bases, RollupMethod::Sum, 3),
            (Some(25.0), 2, "quantified")
        );
        assert_eq!(
            rollup_protein_bases(bases, RollupMethod::TopNSum, 1),
            (Some(20.0), 2, "quantified")
        );

        add_protein_base_quantity(&mut groups, "NO_QUANT", 99, None);
        assert_eq!(
            rollup_protein_bases(&groups["NO_QUANT"], RollupMethod::Sum, 3),
            (None, 0, "no_quantifiable_peptide")
        );
    }

    // Also the legacy-artifact regression: this chromatogram table carries no
    // `predicted_intensity` column, as every artifact written before that column existed
    // does. Quant must still read it and quantify from it unchanged.
    #[test]
    fn quant_run_preserves_unquantifiable_ids_and_applied_window_contract() {
        let scored = quant_test_path("scored.parquet");
        let chrom = quant_test_path("chrom.parquet");
        let peptide = quant_test_path("peptide_quant.parquet");
        let protein = quant_test_path("protein_quant.parquet");
        let fragment = quant_test_path("fragment_quant.parquet");
        let bounds = quant_test_path("peak_bounds.parquet");

        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2, 3, 4]),
                Col::U32("base_peptide_id".into(), vec![10, 10, 11, 12]),
                Col::Str(
                    "peptidoform".into(),
                    vec!["PEP1".into(), "PEP2".into(), "PEP3".into(), "PEP4".into()],
                ),
                Col::I32("charge".into(), vec![2, 3, 2, 2]),
                Col::Str("label".into(), vec!["target".into(); 4]),
                Col::Str(
                    "protein_group".into(),
                    vec!["PG".into(), "PG".into(), "EMPTY".into(), "ZERO".into()],
                ),
                Col::F64("peptide_q_value".into(), vec![0.0; 4]),
                Col::F64("apex_rt".into(), vec![5.0; 4]),
                Col::F64("elution_lo".into(), vec![4.0; 4]),
                Col::F64("elution_hi".into(), vec![6.0; 4]),
            ],
        )
        .unwrap();

        let grid: Vec<f32> = (0..10).map(|rt| rt as f32).collect();
        let c1a = vec![0.0f32, 50.0, 0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 0.0, 0.0];
        let c1b = vec![0.0f32, 25.0, 0.0, 0.0, 2.5, 5.0, 2.5, 0.0, 0.0, 0.0];
        let c2 = vec![0.0f32, 0.0, 0.0, 0.0, 10.0, 20.0, 10.0, 0.0, 0.0, 0.0];
        let zero = vec![0.0f32; 10];
        write_table(
            &chrom,
            vec![
                Col::U32("candidate_id".into(), vec![1, 1, 2, 4]),
                Col::Str(
                    "frag_name".into(),
                    vec!["b2".into(), "y3".into(), "b4".into(), "y5".into()],
                ),
                Col::ListF32(
                    "rt".into(),
                    vec![grid.clone(), grid.clone(), grid.clone(), grid],
                ),
                Col::ListF32("intensity".into(), vec![c1a, c1b, c2, zero]),
            ],
        )
        .unwrap();

        let cfg = QuantConfig::default();
        let rows = run(QuantParams {
            psms_scored: &scored,
            chromatograms: &chrom,
            out_peptide: &peptide,
            out_protein: &protein,
            out_fragment: Some(&fragment),
            out_peak_bounds: Some(&bounds),
            cfg: &cfg,
            config_hash: "test",
        })
        .unwrap();
        assert_eq!(rows, (4, 3));

        let pq = Table::read(&peptide).unwrap();
        assert_eq!(pq.u32("base_peptide_id").unwrap(), vec![10, 10, 11, 12]);
        assert_eq!(
            pq.str("quant_status").unwrap(),
            vec![
                "quantified",
                "quantified",
                "no_fragment_traces",
                "no_positive_fragment_area"
            ]
        );
        assert_eq!(pq.i32("n_fragments_used").unwrap(), vec![2, 1, 0, 0]);
        let quantities = pq.opt_f64("quantity").unwrap();
        assert_eq!(quantities[0], Some(22.5));
        assert_eq!(quantities[1], Some(30.0));
        assert_eq!(quantities[2], None);
        assert_eq!(quantities[3], None);
        assert_eq!(
            pq.opt_f64("integration_apex_rt").unwrap(),
            vec![Some(5.0), Some(5.0), None, Some(5.0)]
        );
        assert_eq!(pq.opt_f64("integration_lo_rt").unwrap()[0], Some(4.0));
        assert_eq!(pq.opt_f64("integration_hi_rt").unwrap()[0], Some(6.0));

        let gq = Table::read(&protein).unwrap();
        let names = gq.str("protein_group").unwrap();
        let values = gq.opt_f64("quantity").unwrap();
        let statuses = gq.str("quant_status").unwrap();
        let counts = gq.i32("n_peptides").unwrap();
        let by_group: HashMap<_, _> = names
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, (values[i], statuses[i].clone(), counts[i])))
            .collect();
        assert_eq!(by_group["PG"], (Some(30.0), "quantified".to_string(), 1));
        assert_eq!(
            by_group["EMPTY"],
            (None, "no_quantifiable_peptide".to_string(), 0)
        );
        assert_eq!(
            by_group["ZERO"],
            (None, "no_quantifiable_peptide".to_string(), 0)
        );

        let fq = Table::read(&fragment).unwrap();
        assert_eq!(fq.nrows, 3, "the all-zero ion must not be exported");
        assert!(fq.f64("quantity").unwrap().iter().all(|area| *area > 0.0));
    }

    #[test]
    fn peak_window_never_collapses_to_single_sample() {
        // A sharp ~1-scan summed XIC (both apex shoulders below the 1/6 threshold)
        // would collapse the peak_bounds window to lo==hi; trapezoid_window would then
        // return the raw apex HEIGHT (10) rather than a time-integrated area. Grid step
        // is 2 s so the true triangle area (20) differs from the height (10), proving
        // the widening yields intensity*seconds units consistent with broad peaks.
        let grid = vec![0.0f32, 2.0, 4.0, 6.0, 8.0];
        let spike = vec![0.0f32, 0.0, 10.0, 0.0, 0.0];
        let ch_rt = vec![grid.clone()];
        let ch_int = vec![spike];
        let (lo, hi, apex) = peak_window(&[0], &ch_rt, &ch_int, 1.0 / 6.0, 1, None);
        assert_eq!(apex, 4.0, "apex should be the summed-XIC max rt");
        assert!(hi > lo, "window must have nonzero width: lo={lo} hi={hi}");
        let a = trapezoid_window(&ch_rt[0], &ch_int[0], lo, hi, false);
        assert!(
            (a - 20.0).abs() < 1e-9,
            "expected triangle area 20, not height 10, got {a}"
        );
    }

    #[test]
    fn fixed_window_scan_and_second_forms_select_the_apex_subwindow() {
        // 1 s grid, triangle apex at 5 s.
        let rt: Vec<f32> = (0..11).map(|i| i as f32).collect();
        let it = vec![0.0f32, 0.0, 0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        // +/-1 scan around the apex sample: (4,5),(5,10),(6,5) -> 7.5 + 7.5 = 15.
        assert!((trapezoid_fixed_opts(&rt, &it, 5.0, 1, 0.0, false, None) - 15.0).abs() < 1e-9);
        // +/-1 s is the same three samples on this grid.
        assert!((trapezoid_fixed_opts(&rt, &it, 5.0, 0, 1.0, false, None) - 15.0).abs() < 1e-9);
        // The seconds form overrides the scan count, as its doc comment claims.
        assert!((trapezoid_fixed_opts(&rt, &it, 5.0, 5, 1.0, false, None) - 15.0).abs() < 1e-9);
        // Half-width 0 keeps only the nearest sample, so `trapezoid`'s single-sample rule
        // returns the raw height. This is why the AIF/Astral configs never use 0.
        assert_eq!(
            trapezoid_fixed_opts(&rt, &it, 5.0, 0, 0.0, false, None),
            10.0
        );
        // A window wider than the trace integrates the whole trace (area 20).
        assert!((trapezoid_fixed_opts(&rt, &it, 5.0, 0, 100.0, false, None) - 20.0).abs() < 1e-9);
        // An apex off the sampled range still integrates around the nearest sample.
        assert_eq!(
            trapezoid_fixed_opts(&rt, &it, -50.0, 0, 0.0, false, None),
            0.0
        );
        // Empty trace integrates to 0 rather than panicking on the index math.
        assert_eq!(
            trapezoid_fixed_opts(&[], &[], 5.0, 3, 0.0, false, None),
            0.0
        );
        assert_eq!(fixed_window_indices(&[], 5.0, 3, 0.0), None);
        assert_eq!(fixed_window_indices(&rt, 5.0, 1, 0.0), Some((4, 7)));
        assert_eq!(fixed_window_indices(&rt, 5.0, 0, 1.0), Some((4, 7)));
    }

    #[test]
    fn flank_baseline_uses_the_flank_quantile() {
        let it = vec![1.0f32, 3.0, 100.0, 100.0, 100.0, 5.0, 7.0];
        // Window [2,5) with 2-sample flanks -> flank pool [1,3,5,7].
        // Median: position round(3 * 0.5) = 2 -> 5.
        assert_eq!(flank_baseline(&it, 2, 5, 2, 0.5), 5.0);
        // Lower quartile: position round(3 * 0.25) = 1 -> 3.
        assert_eq!(flank_baseline(&it, 2, 5, 2, 0.25), 3.0);
        // No flank sample exists (the window covers the trace) -> no background.
        assert_eq!(flank_baseline(&it, 0, 7, 3, 0.5), 0.0);

        // End to end: subtracting the median flank level lowers the area by
        // baseline * width and clips at zero.
        let rt: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let plain = trapezoid_fixed_opts(&rt, &it, 3.0, 1, 0.0, false, None);
        let debased = trapezoid_fixed_opts(&rt, &it, 3.0, 1, 0.0, false, Some((2, 0.5)));
        assert!((plain - 200.0).abs() < 1e-9, "got {plain}");
        assert!((debased - 190.0).abs() < 1e-9, "got {debased}");
    }

    #[test]
    fn select_fragment_areas_ranks_by_predicted_intensity() {
        // Fragment 0 has the largest observed area but the smallest library intensity:
        // the interference case `fragment_selection = predicted` exists to avoid.
        let areas = [(100.0f64, 0.1f32), (40.0, 1.0), (30.0, 0.8)];
        assert_eq!(
            select_fragment_areas(Some(&areas), 2, FragmentSelection::ObservedArea),
            (Some(140.0), 2, "quantified")
        );
        assert_eq!(
            select_fragment_areas(Some(&areas), 2, FragmentSelection::Predicted),
            (Some(70.0), 2, "quantified")
        );
        // `observed_area` must stay byte-identical to the legacy summariser.
        let plain: Vec<f64> = areas.iter().map(|a| a.0).collect();
        assert_eq!(
            select_fragment_areas(Some(&areas), 2, FragmentSelection::ObservedArea),
            summarize_fragment_areas(Some(&plain), 2)
        );
        // Both rankings report the same statuses on the degenerate inputs.
        for sel in [
            FragmentSelection::ObservedArea,
            FragmentSelection::Predicted,
        ] {
            assert_eq!(
                select_fragment_areas(None, 3, sel),
                (None, 0, "no_fragment_traces")
            );
            assert_eq!(
                select_fragment_areas(Some(&[(0.0, 1.0)]), 3, sel),
                (None, 0, "no_positive_fragment_area")
            );
            assert_eq!(
                select_fragment_areas(Some(&areas), 0, sel),
                (None, 0, "no_fragments_selected")
            );
            // Non-finite areas are dropped, not summed into a NaN quantity.
            assert_eq!(
                select_fragment_areas(Some(&[(f64::NAN, 1.0), (10.0, 0.5)]), 3, sel),
                (Some(10.0), 1, "quantified")
            );
        }
    }

    /// Scored + chromatogram pair for the fragment-selection tests: one candidate, three
    /// fragments on a 1 s grid with a 5 s apex and a wide 0-10 s elution hint. The
    /// brightest fragment by observed area (`b2`, which also carries a late interferent)
    /// is the dimmest by library intensity.
    fn selection_fixture(with_predicted: bool) -> (String, String) {
        let scored = quant_test_path("sel_scored.parquet");
        let chrom = quant_test_path("sel_chrom.parquet");
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![1]),
                Col::U32("base_peptide_id".into(), vec![10]),
                Col::Str("peptidoform".into(), vec!["PEP1".into()]),
                Col::I32("charge".into(), vec![2]),
                Col::Str("label".into(), vec!["target".into()]),
                Col::Str("protein_group".into(), vec!["PG".into()]),
                Col::F64("peptide_q_value".into(), vec![0.0]),
                Col::F64("apex_rt".into(), vec![5.0]),
                Col::F64("elution_lo".into(), vec![0.0]),
                Col::F64("elution_hi".into(), vec![10.0]),
            ],
        )
        .unwrap();
        let grid: Vec<f32> = (0..11).map(|rt| rt as f32).collect();
        let b2 = vec![
            0.0f32, 0.0, 0.0, 0.0, 50.0, 100.0, 50.0, 0.0, 300.0, 600.0, 300.0,
        ];
        let y3 = vec![0.0f32, 0.0, 0.0, 0.0, 20.0, 40.0, 20.0, 0.0, 0.0, 0.0, 0.0];
        let y5 = vec![0.0f32, 0.0, 0.0, 0.0, 15.0, 30.0, 15.0, 0.0, 0.0, 0.0, 0.0];
        let mut cols = vec![
            Col::U32("candidate_id".into(), vec![1, 1, 1]),
            Col::Str(
                "frag_name".into(),
                vec!["b2".into(), "y3".into(), "y5".into()],
            ),
            Col::ListF32("rt".into(), vec![grid.clone(), grid.clone(), grid]),
            Col::ListF32("intensity".into(), vec![b2, y3, y5]),
        ];
        if with_predicted {
            cols.push(Col::F32("predicted_intensity".into(), vec![0.1, 1.0, 0.8]));
        }
        write_table(&chrom, cols).unwrap();
        (scored, chrom)
    }

    #[test]
    fn fixed_window_and_predicted_selection_use_the_library_fragments_at_the_apex() {
        let (scored, chrom) = selection_fixture(true);
        let peptide = quant_test_path("sel_peptide.parquet");
        let protein = quant_test_path("sel_protein.parquet");
        let cfg = QuantConfig {
            top_n_fragments: 2,
            fragment_selection: FragmentSelection::Predicted,
            fixed_window_s: 1.0,
            ..QuantConfig::default()
        };
        let rows = run(QuantParams {
            psms_scored: &scored,
            chromatograms: &chrom,
            out_peptide: &peptide,
            out_protein: &protein,
            out_fragment: None,
            out_peak_bounds: None,
            cfg: &cfg,
            config_hash: "test",
        })
        .unwrap();
        assert_eq!(rows, (1, 1));

        // +/-1 s of the 5 s apex integrates y3 to 60 and y5 to 45; b2 integrates to 150
        // there but ranks last by library intensity, so `predicted` must exclude it.
        let pq = Table::read(&peptide).unwrap();
        assert_eq!(pq.opt_f64("quantity").unwrap(), vec![Some(105.0)]);
        assert_eq!(pq.i32("n_fragments_used").unwrap(), vec![2]);
        // The reported window must be the one that was integrated, not the 0-10 s hint.
        assert_eq!(pq.opt_f64("integration_apex_rt").unwrap(), vec![Some(5.0)]);
        assert_eq!(pq.opt_f64("integration_lo_rt").unwrap(), vec![Some(4.0)]);
        assert_eq!(pq.opt_f64("integration_hi_rt").unwrap(), vec![Some(6.0)]);
    }

    #[test]
    fn adding_predicted_intensity_does_not_move_a_default_quantity() {
        // Compatibility contract: the column's presence must not change a legacy result,
        // and its absence must not stop one. Same traces, same default config, both ways.
        let cfg = QuantConfig::default();
        let mut out = Vec::new();
        for with_predicted in [false, true] {
            let (scored, chrom) = selection_fixture(with_predicted);
            let peptide = quant_test_path("cmp_peptide.parquet");
            let protein = quant_test_path("cmp_protein.parquet");
            let rows = run(QuantParams {
                psms_scored: &scored,
                chromatograms: &chrom,
                out_peptide: &peptide,
                out_protein: &protein,
                out_fragment: None,
                out_peak_bounds: None,
                cfg: &cfg,
                config_hash: "test",
            })
            .unwrap();
            let pq = Table::read(&peptide).unwrap();
            out.push((
                rows,
                pq.opt_f64("quantity").unwrap(),
                pq.str("quant_status").unwrap(),
                pq.i32("n_fragments_used").unwrap(),
                pq.opt_f64("integration_lo_rt").unwrap(),
                pq.opt_f64("integration_hi_rt").unwrap(),
            ));
        }
        assert_eq!(out[0], out[1]);
        assert!(out[0].1[0].is_some(), "the fixture must be quantifiable");
    }

    #[test]
    fn an_empty_chromatogram_table_preserves_every_identification() {
        // Extraction can accept a candidate and still write no chromatogram for
        // it, and a run can legitimately produce an empty chromatogram table. The
        // identification must survive with an explicit unquantifiable status: an
        // accepted PSM silently vanishing from the peptide table would be a
        // reported identification lost to a quantification detail, which is the
        // opposite of the contract that identification and quantifiability are
        // separate.
        let scored = quant_test_path("empty_scored.parquet");
        let chrom = quant_test_path("empty_chrom.parquet");
        write_table(
            &scored,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2]),
                Col::U32("base_peptide_id".into(), vec![10, 11]),
                Col::Str("peptidoform".into(), vec!["PEP1".into(), "PEP2".into()]),
                Col::I32("charge".into(), vec![2, 2]),
                Col::Str("label".into(), vec!["target".into(); 2]),
                Col::Str("protein_group".into(), vec!["PG".into(), "PG".into()]),
                Col::F64("peptide_q_value".into(), vec![0.0, 0.0]),
                Col::F64("apex_rt".into(), vec![5.0, 5.0]),
                Col::F64("elution_lo".into(), vec![4.0, 4.0]),
                Col::F64("elution_hi".into(), vec![6.0, 6.0]),
            ],
        )
        .unwrap();
        write_table(
            &chrom,
            vec![
                Col::U32("candidate_id".into(), Vec::new()),
                Col::Str("frag_name".into(), Vec::new()),
                Col::ListF32("rt".into(), Vec::new()),
                Col::ListF32("intensity".into(), Vec::new()),
            ],
        )
        .unwrap();

        let peptide = quant_test_path("empty_peptide.parquet");
        let protein = quant_test_path("empty_protein.parquet");
        let cfg = QuantConfig::default();
        let rows = run(QuantParams {
            psms_scored: &scored,
            chromatograms: &chrom,
            out_peptide: &peptide,
            out_protein: &protein,
            out_fragment: None,
            out_peak_bounds: None,
            cfg: &cfg,
            config_hash: "test",
        })
        .unwrap();
        assert_eq!(rows.0, 2, "both identifications must be reported");

        let pq = Table::read(&peptide).unwrap();
        assert_eq!(
            pq.str("quant_status").unwrap(),
            vec!["no_fragment_traces"; 2]
        );
        assert_eq!(pq.opt_f64("quantity").unwrap(), vec![None, None]);
        assert_eq!(pq.i32("n_fragments_used").unwrap(), vec![0, 0]);
        // The protein group has no quantifiable peptide, and must say so rather
        // than reporting a quantity of zero.
        let gq = Table::read(&protein).unwrap();
        assert_eq!(
            gq.str("quant_status").unwrap(),
            vec!["no_quantifiable_peptide"]
        );
        assert_eq!(gq.opt_f64("quantity").unwrap(), vec![None]);
    }

    #[test]
    fn predicted_selection_without_the_column_is_a_clear_error() {
        let (scored, chrom) = selection_fixture(false);
        let cfg = QuantConfig {
            fragment_selection: FragmentSelection::Predicted,
            ..QuantConfig::default()
        };
        let err = run(QuantParams {
            psms_scored: &scored,
            chromatograms: &chrom,
            out_peptide: &quant_test_path("err_peptide.parquet"),
            out_protein: &quant_test_path("err_protein.parquet"),
            out_fragment: None,
            out_peak_bounds: None,
            cfg: &cfg,
            config_hash: "test",
        })
        .unwrap_err()
        .to_string();
        // The message must name the missing column and the way out, because the artifact
        // is silently older rather than malformed.
        assert!(err.contains("predicted_intensity"), "{err}");
        assert!(err.contains("observed_area"), "{err}");
    }
}
