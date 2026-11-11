//! Quantification from the fragments that belong (`quant.q1_fragments`), on scanning-
//! quadrupole spectra (SCIEX ZT Scan DIA).
//!
//! A ZT Scan chromatogram of a candidate interleaves the steps of every cycle: extract
//! samples each fragment in every scan of the windows that cover the precursor, so the trace
//! holds about five samples per cycle, 1.3 ms apart, then a gap of about a second to the next
//! cycle. Integrated as it is, the trapezoid rule weighs each sample by its distance to its
//! neighbours, so the samples inside a cycle (1.3 ms apart) count for almost nothing and the
//! first and last step of each cycle, where the window transmits the least, carry the area;
//! a fixed window of `2h + 1` samples then spans little more than one cycle.
//!
//! This module turns every trace into one value per cycle, with the calibrated Q1 model of
//! `extract.q1_fragments` (recorded in the psms_extracted report): each step `s` of a cycle
//! gets the weight `w_s`, the fragment's expected Q1 profile (the trapezoid transmission
//! summed over the precursor's isotopes, `q1frag::CandQ1`) at the step's target and the row's
//! fitted Q1 centre and width (`q1f_centre`, `q1f_width`), and the cycle's value is the
//! least-squares amplitude `sum(w x) / sum(w^2)` at the mean retention time of the cycle's
//! samples (`quant.q1_step_weights = uniform`: `w = 1`, the mean). Which step and cycle each
//! sample is comes from spectra_ms2 v3 (`q1_cycle`, the window as read).
//!
//! The decision of each fragment (`q1f_frag_class`, library order, which is the
//! chromatograms' row order) then selects: when at least `quant.q1_min_belonging` fragments
//! belong, every other fragment is left out of the quantity (quant's mask). With fewer the
//! candidate keeps every fragment, on the same cycle traces, so one precursor is in the same
//! units in every run whichever way each run went.
//!
//! [`elution_weights`] is `quant.q1_integration = elution_fit`: one Gaussian elution profile
//! fitted to the window samples, and the integration weights that give each fragment its
//! least-squares amplitude under it times the profile's area.
//!
//! Deterministic: every sum runs in sample order, the scans are ordered by retention time and
//! then by scan index, and every lookup takes the nearest scan with ties to the earlier.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use arrow::array::{Array, Int32Array, UInt32Array};
use mumdia_core::config::{Q1FragModel, Q1StepWeights};
use mumdia_io::table::{ListF32, TableFile};

use super::q1frag;

/// What quant reads under `quant.q1_fragments` besides its usual inputs.
#[derive(Clone, Copy, Debug)]
pub struct Q1QuantInputs<'a> {
    /// The run's psms_extracted v6 (`extract.q1_fragments`); its report records the Q1 model.
    pub psms_extracted: &'a str,
    /// The run's spectra_ms2 v3: the retention time, the window as read and the cycle of
    /// every MS2 spectrum.
    pub spectra_ms2: &'a str,
}

/// How one candidate was quantified under `quant.q1_fragments` (`q1_quant` of
/// peptide_quant v4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Q1Status {
    /// The cycle traces of the fragments that belong.
    Belonging,
    /// Every fragment's cycle trace: fewer than `q1_min_belonging` belong.
    Fallback,
    /// The traces as extracted: no Q1 decision for the candidate, its decision does not
    /// match its chromatogram rows, or a sample lies outside the sweep.
    Uncollapsed,
    /// The candidate has no chromatogram row.
    NoTraces,
}

impl Q1Status {
    pub const ALL: [Q1Status; 4] = [
        Q1Status::Belonging,
        Q1Status::Fallback,
        Q1Status::Uncollapsed,
        Q1Status::NoTraces,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Q1Status::Belonging => "belonging",
            Q1Status::Fallback => "fallback",
            Q1Status::Uncollapsed => "uncollapsed",
            Q1Status::NoTraces => "no_traces",
        }
    }
}

/// One candidate's psms_extracted row as the collapse reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct CandRow {
    pub precursor_mz: f64,
    pub charge: i32,
    pub apex_rt: f64,
    /// The fitted shared Q1 centre (steps, against the expected one) and width scale.
    pub centre: f64,
    pub width: f64,
    /// Per library fragment, in library order: 0 absent, 1 belongs, 2 interfered.
    pub class: Vec<f32>,
}

impl CandRow {
    /// The fragments that belong.
    pub fn n_belong(&self) -> usize {
        self.class
            .iter()
            .filter(|&&c| c as u8 == q1frag::BELONGS)
            .count()
    }
}

/// The Q1 model `extract.q1_fragments` decided with, from the `q1_fragments` block of the
/// psms_extracted report.
pub fn model_of_report(psms_extracted: &str) -> Result<Q1FragModel> {
    let report = format!("{psms_extracted}.report.json");
    let r: serde_json::Value = mumdia_io::json::read_json(&report).with_context(|| {
        format!(
            "quant.q1_fragments reads the Q1 model from {report}, the report extract writes \
             next to psms_extracted"
        )
    })?;
    let block = &r["params"]["q1_fragments"];
    if block.is_null() {
        bail!(
            "quant.q1_fragments: {report} records no q1_fragments model; the run was extracted \
             without extract.q1_fragments"
        );
    }
    let m: Q1FragModel = serde_json::from_value(block.clone())
        .with_context(|| format!("quant.q1_fragments: the q1_fragments block of {report}"))?;
    if !(m.step_mz > 0.0 && m.half_width_mz > 0.0 && m.edge_width_mz >= 0.0) {
        bail!(
            "quant.q1_fragments: {report}: step {} m/z, half-width {} and edge {} are not a \
             window",
            m.step_mz,
            m.half_width_mz,
            m.edge_width_mz
        );
    }
    Ok(m)
}

/// The psms_extracted rows of `wanted` (candidate id -> its peak rank), with their Q1
/// decision. A wanted candidate without a row is absent from the map.
pub fn load_rows(
    psms_extracted: &str,
    wanted: &HashMap<u32, i32>,
) -> Result<HashMap<u32, CandRow>> {
    let t = TableFile::open(psms_extracted)?;
    for c in [
        "q1f_frag_class",
        "q1f_centre",
        "q1f_width",
        "peak_rank",
        "precursor_mz",
        "charge",
        "apex_rt",
    ] {
        if !t.has_column(c) {
            bail!(
                "quant.q1_fragments needs the per-fragment Q1 decisions, and {psms_extracted} \
                 has no {c} column: extract the run with extract.q1_fragments"
            );
        }
    }
    let mut out: HashMap<u32, CandRow> = HashMap::new();
    t.for_each_batch(
        Some(&[
            "candidate_id",
            "peak_rank",
            "precursor_mz",
            "charge",
            "apex_rt",
            "q1f_centre",
            "q1f_width",
            "q1f_frag_class",
        ]),
        1 << 16,
        |b| {
            let col = |n: &str| -> Result<&arrow::array::ArrayRef> {
                Ok(b.column(b.schema().index_of(n)?))
            };
            let f64s = |n: &str| -> Result<&arrow::array::Float64Array> {
                col(n)?
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .with_context(|| format!("psms_extracted.{n} is not f64"))
            };
            let cid = col("candidate_id")?
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("psms_extracted.candidate_id is not u32")?;
            let rank = col("peak_rank")?
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("psms_extracted.peak_rank is not i32")?;
            let z = col("charge")?
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("psms_extracted.charge is not i32")?;
            let (pm, apex, centre, width) = (
                f64s("precursor_mz")?,
                f64s("apex_rt")?,
                f64s("q1f_centre")?,
                f64s("q1f_width")?,
            );
            let cls = ListF32::of(col("q1f_frag_class")?, "q1f_frag_class")?;
            for k in 0..b.num_rows() {
                let c = cid.value(k);
                if wanted.get(&c) != Some(&rank.value(k)) {
                    continue;
                }
                out.insert(
                    c,
                    CandRow {
                        precursor_mz: pm.value(k),
                        charge: z.value(k),
                        apex_rt: apex.value(k),
                        centre: centre.value(k),
                        width: width.value(k),
                        class: cls.row_slice(k, "q1f_frag_class")?.to_vec(),
                    },
                );
            }
            Ok(())
        },
    )?;
    Ok(out)
}

/// The fragment rows of `wanted` candidates in a chromatogram table, in table order: name and
/// library m/z, the MS1 pseudo-traces left out. Only the scalar columns are read.
pub fn load_fragment_mz(
    chromatograms: &str,
    wanted: &HashMap<u32, i32>,
) -> Result<HashMap<u32, Vec<(String, f64)>>> {
    let t = TableFile::open(chromatograms)?;
    let mut out: HashMap<u32, Vec<(String, f64)>> = HashMap::new();
    t.for_each_batch(
        Some(&["candidate_id", "frag_name", "frag_mz"]),
        1 << 16,
        |b| {
            let col = |n: &str| -> Result<&arrow::array::ArrayRef> {
                Ok(b.column(b.schema().index_of(n)?))
            };
            let cid = col("candidate_id")?
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("chromatograms.candidate_id is not u32")?;
            let name = col("frag_name")?
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .context("chromatograms.frag_name is not utf8")?;
            let mz = col("frag_mz")?
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .context("chromatograms.frag_mz is not f64")?;
            for k in 0..b.num_rows() {
                let c = cid.value(k);
                if !wanted.contains_key(&c) || name.value(k).starts_with("ms1_") {
                    continue;
                }
                out.entry(c)
                    .or_default()
                    .push((name.value(k).to_string(), mz.value(k)));
            }
            Ok(())
        },
    )?;
    Ok(out)
}

/// The MS2 spectra of a sweep as the collapse reads them: by retention time (then scan
/// index), each with its cycle and its window as read.
pub struct SweepScans {
    rt: Vec<f64>,
    cycle: Vec<u32>,
    target: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// A sample is a scan's when it lies within this of the scan's retention time: 0.45 of
    /// the median spacing of consecutive scans (half a step, less the f32 rounding of the
    /// chromatogram axis).
    tol: f64,
}

impl SweepScans {
    /// The sweep of a spectra_ms2 v3 table. Refused for a table without one.
    pub fn load(spectra_ms2: &str) -> Result<SweepScans> {
        let sweep = crate::spectra::load_q1_sweep(spectra_ms2)?.ok_or_else(|| {
            anyhow::anyhow!(
                "quant.q1_fragments reads the Q1 sweep spectra_ms2 v3 records (q1_cycle), and \
                 {spectra_ms2} has none: convert the run with convert.isolation_half_width"
            )
        })?;
        let t = TableFile::open(spectra_ms2)?;
        let (rt, target, lower, upper) = (
            t.f64("rt_seconds")?,
            t.f64("window_target")?,
            t.f64("window_lower")?,
            t.f64("window_upper")?,
        );
        let scan = &sweep.scan_index;
        if [
            rt.len(),
            target.len(),
            lower.len(),
            upper.len(),
            sweep.cycle.len(),
        ]
        .iter()
        .any(|&n| n != scan.len())
        {
            bail!("quant.q1_fragments: the columns of {spectra_ms2} differ in length");
        }
        let mut order: Vec<usize> = (0..rt.len()).collect();
        order.sort_by(|&a, &b| rt[a].total_cmp(&rt[b]).then(scan[a].cmp(&scan[b])));
        let pick = |v: &[f64]| -> Vec<f64> { order.iter().map(|&i| v[i]).collect() };
        let rt_s = pick(&rt);
        let mut gaps: Vec<f64> = rt_s
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|d| *d > 0.0)
            .collect();
        let med = super::quant_pattern::median(&mut gaps);
        Ok(SweepScans {
            cycle: order.iter().map(|&i| sweep.cycle[i]).collect(),
            target: pick(&target),
            lower: pick(&lower),
            upper: pick(&upper),
            rt: rt_s,
            tol: if med.is_finite() { 0.45 * med } else { 0.0 },
        })
    }

    /// A sweep of the given scans (tests).
    #[cfg(test)]
    pub fn of(scans: &[(f64, u32, f64, f64, f64)]) -> SweepScans {
        let mut v = scans.to_vec();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        let rt: Vec<f64> = v.iter().map(|s| s.0).collect();
        let mut gaps: Vec<f64> = rt
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|d| *d > 0.0)
            .collect();
        let med = super::quant_pattern::median(&mut gaps);
        SweepScans {
            cycle: v.iter().map(|s| s.1).collect(),
            target: v.iter().map(|s| s.2).collect(),
            lower: v.iter().map(|s| s.3).collect(),
            upper: v.iter().map(|s| s.4).collect(),
            rt,
            tol: 0.45 * med,
        }
    }

    /// The scans of the MS2 spectra read.
    pub fn len(&self) -> usize {
        self.rt.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rt.is_empty()
    }

    /// The cycle and the window target of the scan an axis point at `rt` was sampled from:
    /// the nearest scan (ties to the earlier) whose window as read covers the precursor m/z
    /// `pm`, within the sweep's matching tolerance (0.45 of the median scan spacing). `None`
    /// when there is none.
    pub fn locate(&self, rt: f32, pm: f64) -> Option<(u32, f64)> {
        let x = rt as f64;
        let j = self.rt.partition_point(|&r| r < x - self.tol);
        let mut best: Option<(f64, usize)> = None;
        let mut k = j;
        while k < self.rt.len() && self.rt[k] <= x + self.tol {
            if self.lower[k] <= pm && pm <= self.upper[k] {
                let d = (self.rt[k] - x).abs();
                if best.is_none_or(|(bd, _)| d < bd) {
                    best = Some((d, k));
                }
            }
            k += 1;
        }
        best.map(|(_, k)| (self.cycle[k], self.target[k]))
    }

    /// [`SweepScans::locate`] for every point of `axis`, or `None` when a point is not
    /// found or the cycles do not run in order (a cycle that comes back after another).
    pub fn locate_axis(&self, axis: &[f32], pm: f64) -> Option<Vec<(u32, f64)>> {
        let v: Vec<(u32, f64)> = axis
            .iter()
            .map(|&r| self.locate(r, pm))
            .collect::<Option<_>>()?;
        if v.windows(2).any(|w| w[1].0 < w[0].0) {
            return None;
        }
        Some(v)
    }
}

/// The candidate's fragments as the transmission weights read them: the decision's model of
/// the candidate (`q1frag::CandQ1`) with its fragments' library m/z and charges.
pub fn cand_model(m: &Q1FragModel, row: &CandRow, frags: &[(String, f64)]) -> q1frag::CandQ1 {
    let mzs: Vec<f64> = frags.iter().map(|f| f.1).collect();
    let zs: Vec<i32> = frags.iter().map(|f| q1frag::frag_charge_of(&f.0)).collect();
    let ones = vec![1.0; frags.len()];
    q1frag::CandQ1::new(
        m,
        row.precursor_mz,
        row.charge,
        row.apex_rt,
        &mzs,
        &zs,
        &ones,
        &|_| 1.0,
    )
}

/// One trace collapsed to one value per cycle: `located` is the cycle and target of every
/// point of the trace's axis `rt` ([`SweepScans::locate_axis`]), `weight(t)` the weight of a
/// step of target `t`. Each cycle's value is `sum(w x) / sum(w^2)` over its points (0 where
/// every weight is 0), at the mean retention time of its points.
pub fn collapse_trace(
    located: &[(u32, f64)],
    rt: &[f32],
    it: &[f32],
    weight: &dyn Fn(f64) -> f64,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(located.len(), rt.len());
    debug_assert_eq!(rt.len(), it.len());
    let (mut out_rt, mut out_it) = (Vec::new(), Vec::new());
    let mut i = 0usize;
    while i < rt.len() {
        let c = located[i].0;
        let (mut swx, mut sww, mut srt, mut n) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        while i < rt.len() && located[i].0 == c {
            let w = weight(located[i].1);
            swx += w * it[i] as f64;
            sww += w * w;
            srt += rt[i] as f64;
            n += 1;
            i += 1;
        }
        out_rt.push((srt / n as f64) as f32);
        out_it.push(if sww > 0.0 { (swx / sww) as f32 } else { 0.0 });
    }
    (out_rt, out_it)
}

/// The weight of a step of target `t` for fragment `f` of `cand` under `rule`, at the row's
/// fitted centre and width.
pub fn step_weight(
    rule: Q1StepWeights,
    m: &Q1FragModel,
    cand: &q1frag::CandQ1,
    row: &CandRow,
    f: usize,
    t: f64,
) -> f64 {
    match rule {
        Q1StepWeights::Transmission => {
            let w = if row.width.is_finite() && row.width > 0.0 {
                row.width
            } else {
                1.0
            };
            let mu = if row.centre.is_finite() {
                row.centre
            } else {
                0.0
            };
            cand.expected(m, f, t, mu, w)
        }
        Q1StepWeights::Uniform => 1.0,
    }
}

/// `quant.q1_integration = elution_fit`: the integration weights of window samples at
/// retention times `rt` (NaN outside the trace) under one Gaussian elution profile fitted to
/// their summed intensities `y`. The profile is the weighted least-squares parabola of
/// `ln y` (weights `y^2`) over the samples with `y > 0`; each fragment's area is then
/// `sum(k x)` with `k = G sigma sqrt(2 pi) / sum(G^2)`, its least-squares amplitude under
/// the profile `G` times the profile's area. `None` (the trapezoid applies) with fewer than
/// three positive samples, a parabola that does not open downward, a width outside a fifth
/// of the median spacing to three window spans, or a centre more than one spacing outside
/// the window.
pub fn elution_weights(rt: &[f64], y: &[f64]) -> Option<Vec<f64>> {
    debug_assert_eq!(rt.len(), y.len());
    let fin: Vec<usize> = (0..rt.len()).filter(|&i| rt[i].is_finite()).collect();
    if fin.len() < 3 {
        return None;
    }
    let (t_first, t_last) = (rt[fin[0]], rt[fin[fin.len() - 1]]);
    let span = t_last - t_first;
    if span.is_nan() || span <= 0.0 {
        return None;
    }
    let mut gaps: Vec<f64> = fin.windows(2).map(|w| rt[w[1]] - rt[w[0]]).collect();
    let dt = super::quant_pattern::median(&mut gaps);
    let mid = 0.5 * (t_first + t_last);
    // Normal equations of ln y = a + b u + c u^2, u = t - mid, weights y^2.
    let mut s = [0.0f64; 5];
    let mut r = [0.0f64; 3];
    let mut npos = 0usize;
    for &i in &fin {
        if !(y[i] > 0.0 && y[i].is_finite()) {
            continue;
        }
        npos += 1;
        let u = rt[i] - mid;
        let w = y[i] * y[i];
        let l = y[i].ln();
        let mut up = 1.0;
        for sk in s.iter_mut() {
            *sk += w * up;
            up *= u;
        }
        r[0] += w * l;
        r[1] += w * l * u;
        r[2] += w * l * u * u;
    }
    if npos < 3 {
        return None;
    }
    // [s0 s1 s2; s1 s2 s3; s2 s3 s4] [a b c]' = r, by Cramer's rule.
    let det3 = |m: [[f64; 3]; 3]| -> f64 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let a = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]];
    let d = det3(a);
    if d == 0.0 || !d.is_finite() {
        return None;
    }
    let mut ab = a;
    let mut ac = a;
    for ((rb, rc), &rk) in ab.iter_mut().zip(ac.iter_mut()).zip(&r) {
        rb[1] = rk;
        rc[2] = rk;
    }
    let (b, c) = (det3(ab) / d, det3(ac) / d);
    if c.is_nan() || c >= 0.0 || !b.is_finite() {
        return None;
    }
    let sigma = (-0.5 / c).sqrt();
    let u0 = -b / (2.0 * c);
    let t0 = mid + u0;
    let sigma_ok = sigma.is_finite() && sigma >= 0.2 * dt && sigma <= 3.0 * span;
    let centre_ok = t0 >= t_first - dt && t0 <= t_last + dt;
    if !sigma_ok || !centre_ok {
        return None;
    }
    let g: Vec<f64> = rt
        .iter()
        .map(|&t| {
            if t.is_finite() {
                let z = (t - t0) / sigma;
                (-0.5 * z * z).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sgg: f64 = g.iter().map(|x| x * x).sum();
    if sgg.is_nan() || sgg <= 0.0 {
        return None;
    }
    let area = sigma * (2.0 * std::f64::consts::PI).sqrt();
    Some(g.iter().map(|x| x * area / sgg).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mumdia_core::config::{Q1FragTolerances, Q1OffsetTerm};

    fn model() -> Q1FragModel {
        Q1FragModel {
            step_mz: 0.7614,
            half_width_mz: 2.57,
            edge_width_mz: 4.0,
            offset_terms: vec![Q1OffsetTerm {
                var: "precursor_mz".into(),
                grid: vec![400.0, 900.0],
                offset_mz: vec![0.2, -0.05],
            }],
            tolerances: Q1FragTolerances::default(),
        }
    }

    /// A sweep of `cycles` cycles of 1 s whose steps are 1.345 ms and 0.7614 m/z apart, each
    /// window +-1.9 m/z around its target.
    fn sweep(cycles: u32, steps: u32) -> (SweepScans, Vec<(f64, u32, f64)>) {
        let mut v = Vec::new();
        let mut meta = Vec::new();
        for c in 0..cycles {
            for s in 0..steps {
                let rt = 100.0 + c as f64 + s as f64 * 0.001345;
                let t = 500.0 + s as f64 * 0.7614;
                v.push((rt, c, t, t - 1.9, t + 1.9));
                meta.push((rt, c, t));
            }
        }
        (SweepScans::of(&v), meta)
    }

    #[test]
    fn locate_finds_the_covering_scan_of_an_f32_axis_point() {
        let (sw, meta) = sweep(4, 20);
        let pm = 500.0 + 7.3 * 0.7614;
        for &(rt, c, t) in &meta {
            let got = sw.locate(rt as f32, pm);
            if (t - pm).abs() <= 1.9 {
                assert_eq!(got, Some((c, t)), "{rt}");
            } else {
                assert!(got.is_none_or(|(_, tt)| (tt - pm).abs() <= 1.9));
            }
        }
        // A point between two cycles is no scan's.
        assert_eq!(sw.locate(100.5, pm), None);
    }

    #[test]
    fn collapse_gives_the_amplitude_of_a_transmitted_profile() {
        let (sw, meta) = sweep(5, 30);
        let m = model();
        let pm = 500.0 + 12.0 * 0.7614;
        let row = CandRow {
            precursor_mz: pm,
            charge: 2,
            apex_rt: 102.0,
            centre: 0.3,
            width: 1.0,
            class: vec![1.0],
        };
        let frags = vec![("y7".to_string(), 812.4)];
        let cand = cand_model(&m, &row, &frags);
        // The grid of the covering scans, and a fragment whose intensity is its amplitude
        // per cycle times its transmission at the row's centre.
        let amp = [10.0, 40.0, 100.0, 40.0, 10.0];
        let (mut rt, mut it) = (Vec::new(), Vec::new());
        for &(r, c, t) in &meta {
            if (t - pm).abs() <= 1.9 {
                rt.push(r as f32);
                it.push(
                    (amp[c as usize]
                        * step_weight(Q1StepWeights::Transmission, &m, &cand, &row, 0, t))
                        as f32,
                );
            }
        }
        let located = sw.locate_axis(&rt, pm).unwrap();
        let w = |t: f64| step_weight(Q1StepWeights::Transmission, &m, &cand, &row, 0, t);
        let (crt, cit) = collapse_trace(&located, &rt, &it, &w);
        assert_eq!(crt.len(), 5);
        for (c, (&a, &x)) in amp.iter().zip(&cit).enumerate() {
            assert!((x as f64 - a).abs() < 1e-3 * a, "cycle {c}: {x} vs {a}");
            assert!((crt[c] as f64 - (100.0 + c as f64)).abs() < 0.05);
        }
        // Uniform weights: the mean of the steps, which is smaller.
        let (_, uit) = collapse_trace(&located, &rt, &it, &|_| 1.0);
        assert!(uit[2] < cit[2]);
        // Deterministic.
        assert_eq!(collapse_trace(&located, &rt, &it, &w), (crt, cit));
    }

    #[test]
    fn an_axis_with_a_point_off_the_sweep_is_not_located() {
        let (sw, _) = sweep(3, 20);
        let pm = 505.0;
        assert!(sw.locate_axis(&[100.0, 100.5], pm).is_none());
        assert_eq!(sw.locate_axis(&[], pm), Some(Vec::new()));
    }

    #[test]
    fn the_elution_weights_integrate_a_gaussian_exactly() {
        let rt: Vec<f64> = (0..7).map(|i| 10.0 + i as f64).collect();
        let (t0, sigma, amp) = (13.3, 1.2, 50.0);
        let g = |t: f64| amp * (-0.5 * ((t - t0) / sigma).powi(2)).exp();
        let y: Vec<f64> = rt.iter().map(|&t| g(t)).collect();
        let k = elution_weights(&rt, &y).unwrap();
        let area: f64 = k.iter().zip(&y).map(|(a, b)| a * b).sum();
        let want = amp * sigma * (2.0 * std::f64::consts::PI).sqrt();
        assert!((area - want).abs() < 1e-6 * want, "{area} vs {want}");
        // A second fragment of the same shape and a third of its height gets a third.
        let y3: Vec<f64> = y.iter().map(|v| v / 3.0).collect();
        let a3: f64 = k.iter().zip(&y3).map(|(a, b)| a * b).sum();
        assert!((a3 - want / 3.0).abs() < 1e-6 * want);
        // Samples outside the trace weigh 0.
        let mut rt_n = rt.clone();
        rt_n[0] = f64::NAN;
        let kn = elution_weights(&rt_n, &y).unwrap();
        assert_eq!(kn[0], 0.0);
    }

    #[test]
    fn the_elution_fit_gives_way_where_there_is_no_peak() {
        let rt: Vec<f64> = (0..7).map(|i| i as f64).collect();
        // Rising: the parabola of ln y opens upward or the centre is outside the window.
        assert!(elution_weights(&rt, &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]).is_none());
        // Too few positive samples.
        assert!(elution_weights(&rt, &[0.0, 0.0, 5.0, 3.0, 0.0, 0.0, 0.0]).is_none());
        // Flat.
        assert!(elution_weights(&rt, &[5.0; 7]).is_none());
        // Too few samples.
        assert!(elution_weights(&rt[..2], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn the_status_strings_are_the_documented_ones() {
        let s: Vec<&str> = Q1Status::ALL.iter().map(|s| s.as_str()).collect();
        assert_eq!(s, ["belonging", "fallback", "uncollapsed", "no_traces"]);
    }
}

/// `quant` end to end with `quant.q1_fragments` on a synthetic sweep.
#[cfg(test)]
mod run_tests {
    use super::*;
    use crate::stages::quant::{run, ChromTable, QuantParams};
    use mumdia_core::config::{Q1FragTolerances, QuantConfig};
    use mumdia_io::table::{write_table, Col, Table};

    const STEP: f64 = 0.7614;
    const CYCLES: u32 = 9;
    const STEPS: u32 = 30;
    /// Elution amplitude per cycle (the identification apex at cycle 4).
    const G: [f64; 9] = [0.0, 1.0, 5.0, 20.0, 40.0, 20.0, 5.0, 1.0, 0.0];

    fn path(name: &str) -> String {
        let dir = std::env::temp_dir().join(format!("mumdia_q1quant_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name).to_string_lossy().into_owned()
    }

    fn rt_of(c: u32, s: u32) -> f64 {
        100.0 + c as f64 + s as f64 * 0.001345
    }

    fn target(s: u32) -> f64 {
        500.0 + s as f64 * STEP
    }

    fn model() -> Q1FragModel {
        Q1FragModel {
            step_mz: STEP,
            half_width_mz: 2.57,
            edge_width_mz: 4.0,
            offset_terms: Vec::new(),
            tolerances: Q1FragTolerances {
                offset_steps: 1.15,
                centre_steps: 0.69,
                corr2d_min: 0.55,
            },
        }
    }

    struct Fixture {
        scored: String,
        chrom: String,
        psms: String,
        spectra: String,
    }

    /// Two candidates of four fragments on the sweep; in candidate 1 fragment 3 carries
    /// interference and is the only one that does not belong, in candidate 2 fragments 2 and
    /// 3 carry it and do not belong (two belong: fewer than the default three).
    fn fixture(tag: &str) -> Fixture {
        let m = model();
        let names = ["y4", "y5", "y6", "b3"];
        let fmz = [600.31, 713.39, 812.46, 345.18];
        let pred = [1.0f32, 0.8, 0.6, 0.5];
        let cands: [(u32, u32, [f32; 4]); 2] =
            [(1, 12, [1., 1., 1., 2.]), (2, 18, [1., 1., 2., 2.])];
        let (mut c_cid, mut c_name, mut c_mz, mut c_pred, mut c_rt, mut c_int) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        let (mut e_pm, mut e_cls) = (vec![], vec![]);
        for &(cid, s_mid, cls) in &cands {
            let pm = target(s_mid);
            e_pm.push(pm);
            e_cls.push(cls.to_vec());
            let row = CandRow {
                precursor_mz: pm,
                charge: 2,
                apex_rt: rt_of(4, s_mid),
                centre: 0.0,
                width: 1.0,
                class: cls.to_vec(),
            };
            let fr: Vec<(String, f64)> = names
                .iter()
                .zip(fmz)
                .map(|(n, z)| (n.to_string(), z))
                .collect();
            let cand = cand_model(&m, &row, &fr);
            let grid: Vec<(u32, u32)> = (0..CYCLES)
                .flat_map(|c| (0..STEPS).map(move |s| (c, s)))
                .filter(|&(_, s)| (target(s) - pm).abs() <= 1.9)
                .collect();
            for f in 0..4 {
                c_cid.push(cid);
                c_name.push(names[f].to_string());
                c_mz.push(fmz[f]);
                c_pred.push(pred[f]);
                c_rt.push(
                    grid.iter()
                        .map(|&(c, s)| rt_of(c, s) as f32)
                        .collect::<Vec<_>>(),
                );
                c_int.push(
                    grid.iter()
                        .map(|&(c, s)| {
                            let w = step_weight(
                                Q1StepWeights::Transmission,
                                &m,
                                &cand,
                                &row,
                                f,
                                target(s),
                            );
                            let own = 100.0 * pred[f] as f64 * G[c as usize] * w;
                            let interfered = if cls[f] == 2.0 { 5000.0 } else { 0.0 };
                            (own + interfered) as f32
                        })
                        .collect::<Vec<_>>(),
                );
            }
        }
        let fx = Fixture {
            scored: path(&format!("{tag}_scored.parquet")),
            chrom: path(&format!("{tag}_chrom.parquet")),
            psms: path(&format!("{tag}_psms.parquet")),
            spectra: path(&format!("{tag}_spectra.parquet")),
        };
        write_table(
            &fx.scored,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2]),
                Col::U32("base_peptide_id".into(), vec![10, 11]),
                Col::Str("peptidoform".into(), vec!["PEPA".into(), "PEPB".into()]),
                Col::I32("charge".into(), vec![2, 2]),
                Col::Str("label".into(), vec!["target".into(); 2]),
                Col::Str("protein_group".into(), vec!["PG".into(); 2]),
                Col::F64("peptide_q_value".into(), vec![0.0; 2]),
                Col::F64("apex_rt".into(), vec![rt_of(4, 12), rt_of(4, 18)]),
            ],
        )
        .unwrap();
        write_table(
            &fx.chrom,
            vec![
                Col::U32("candidate_id".into(), c_cid),
                Col::Str("frag_name".into(), c_name),
                Col::F64("frag_mz".into(), c_mz),
                Col::F32("predicted_intensity".into(), c_pred),
                Col::ListF32("rt".into(), c_rt),
                Col::ListF32("intensity".into(), c_int),
            ],
        )
        .unwrap();
        write_table(
            &fx.psms,
            vec![
                Col::U32("candidate_id".into(), vec![1, 2]),
                Col::I32("peak_rank".into(), vec![0, 0]),
                Col::F64("apex_rt".into(), vec![rt_of(4, 12), rt_of(4, 18)]),
                Col::F64("precursor_mz".into(), e_pm),
                Col::I32("charge".into(), vec![2, 2]),
                Col::F64("q1f_centre".into(), vec![0.0, 0.0]),
                Col::F64("q1f_width".into(), vec![1.0, 1.0]),
                Col::ListF32("q1f_frag_class".into(), e_cls),
            ],
        )
        .unwrap();
        std::fs::write(
            format!("{}.report.json", fx.psms),
            serde_json::to_string(&serde_json::json!({"params": {"q1_fragments": model()}}))
                .unwrap(),
        )
        .unwrap();
        let (mut si, mut rt, mut tg, mut lo, mut hi, mut cy, mut st) =
            (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        for c in 0..CYCLES {
            for s in 0..STEPS {
                si.push(c * STEPS + s);
                rt.push(rt_of(c, s));
                tg.push(target(s));
                lo.push(target(s) - 1.9);
                hi.push(target(s) + 1.9);
                cy.push(c);
                st.push(s);
            }
        }
        write_table(
            &fx.spectra,
            vec![
                Col::U32("scan_index".into(), si),
                Col::F64("rt_seconds".into(), rt),
                Col::F64("window_target".into(), tg),
                Col::F64("window_lower".into(), lo),
                Col::F64("window_upper".into(), hi),
                Col::U32("q1_cycle".into(), cy),
                Col::U32("q1_step".into(), st),
            ],
        )
        .unwrap();
        fx
    }

    fn quant(fx: &Fixture, cfg: &QuantConfig, tag: &str) -> (Table, serde_json::Value) {
        let pep = path(&format!("{tag}_pep.parquet"));
        let prot = path(&format!("{tag}_prot.parquet"));
        run(QuantParams {
            psms_scored: &fx.scored,
            chromatograms: &[ChromTable::whole(&fx.chrom)],
            out_peptide: &pep,
            out_protein: &prot,
            out_fragment: None,
            out_peak_bounds: None,
            out_quant_fragments: None,
            cfg,
            config_hash: "test",
            cross_run: None,
            export_cross_run: false,
            first_pass_ids: None,
            q1: cfg.q1_fragments.then_some(Q1QuantInputs {
                psms_extracted: &fx.psms,
                spectra_ms2: &fx.spectra,
            }),
        })
        .unwrap();
        let rep: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{pep}.report.json")).unwrap())
                .unwrap();
        (Table::read(&pep).unwrap(), rep)
    }

    /// The quantity is the trapezoid over the cycles of the clean fragments' amplitudes: the
    /// collapse recovers each cycle's amplitude, and the interfered fragment of candidate 1 is
    /// out of the top 3. Candidate 2 has two fragments that belong and keeps all four.
    #[test]
    fn q1_fragments_quantifies_cycle_traces_of_the_fragments_that_belong() {
        let fx = fixture("e2e");
        let cfg = QuantConfig {
            q1_fragments: true,
            ..QuantConfig::default()
        };
        let (t, rep) = quant(&fx, &cfg, "on");
        let q = t.opt_f64("quantity").unwrap();
        let st = t.str("q1_quant").unwrap();
        assert_eq!(st, ["belonging", "fallback"]);
        assert_eq!(rep["schema_version"], 4);
        // Window: the profile centre over cycles 1..=7 (+-3 around the apex cycle 4), the
        // trapezoid of G there with 1 s spacing.
        let trap: f64 = G[1..=7].iter().sum::<f64>() - 0.5 * (G[1] + G[7]);
        let want = 100.0 * (1.0 + 0.8 + 0.6) * trap;
        let got = q[0].unwrap();
        assert!((got - want).abs() < 1e-3 * want, "{got} vs {want}");
        // Candidate 2 keeps its interfered fragment 2 among the top 3: inflated.
        assert!(q[1].unwrap() > 1.5 * want, "{:?}", q[1]);
        let block = &rep["params"]["q1_fragments"];
        assert_eq!(block["candidates"]["belonging"], 1);
        assert_eq!(block["candidates"]["fallback"], 1);
        assert_eq!(block["masked_fragments"], 1);
        // Uniform step weights: the mean of the steps, smaller than the amplitude.
        let uni = QuantConfig {
            q1_step_weights: Q1StepWeights::Uniform,
            ..cfg.clone()
        };
        let (tu, _) = quant(&fx, &uni, "uni");
        assert!(tu.opt_f64("quantity").unwrap()[0].unwrap() < got);
        // The elution fit integrates the same cycles under a fitted Gaussian.
        let ef = QuantConfig {
            q1_integration: mumdia_core::config::Q1Integration::ElutionFit,
            ..cfg.clone()
        };
        let (te, repe) = quant(&fx, &ef, "ef");
        let ge = te.opt_f64("quantity").unwrap()[0].unwrap();
        assert!(ge > 0.5 * want && ge < 2.0 * want, "{ge} vs {want}");
        assert!(
            repe["params"]["q1_fragments"]["elution_fitted"]
                .as_u64()
                .unwrap()
                >= 1
        );
        // Deterministic.
        let (t2, _) = quant(&fx, &cfg, "on2");
        assert_eq!(t2.opt_f64("quantity").unwrap(), q);
    }

    /// Off, the same tables quantify the interleaved traces as they are, in peptide_quant v2
    /// without the column.
    #[test]
    fn without_the_key_nothing_is_collapsed() {
        let fx = fixture("off");
        let cfg = QuantConfig::default();
        let (t, rep) = quant(&fx, &cfg, "off");
        assert!(t.str("q1_quant").is_err());
        assert_eq!(rep["schema_version"], 2);
        assert!(rep["params"]["q1_fragments"].is_null());
    }

    /// The key without its inputs is refused, naming them.
    #[test]
    fn q1_fragments_needs_its_inputs() {
        let fx = fixture("need");
        let cfg = QuantConfig {
            q1_fragments: true,
            ..QuantConfig::default()
        };
        let err = run(QuantParams {
            psms_scored: &fx.scored,
            chromatograms: &[ChromTable::whole(&fx.chrom)],
            out_peptide: &path("need_pep.parquet"),
            out_protein: &path("need_prot.parquet"),
            out_fragment: None,
            out_peak_bounds: None,
            out_quant_fragments: None,
            cfg: &cfg,
            config_hash: "test",
            cross_run: None,
            export_cross_run: false,
            first_pass_ids: None,
            q1: None,
        })
        .unwrap_err();
        assert!(format!("{err:#}").contains("--psms-extracted"), "{err:#}");
    }
}
