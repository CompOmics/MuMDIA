//! MaxLFQ / directLFQ cross-run label-free quantification
//! (docs/12_quant_lfq_align_mbr_report_audit.md, beyond-MVP). These are
//! inherently MULTI-RUN methods: they reconstruct a
//! per-sample protein abundance profile from the pairwise median log-ratios of
//! the features shared between samples (MaxLFQ: peptide-level; directLFQ:
//! ion/fragment-level), so with a single run there are no cross-sample ratios and
//! the profile reduces to the per-sample summed intensity. The shared core is
//! [`lfq_profile`]; the two methods differ only in the feature granularity the
//! caller passes.
//!
//! The `quant-lfq` command lives here too: [`run_lfq_combine_grouped`] reads the per-run
//! quant tables, divides each run by its size factor and writes the protein-by-run matrix
//! with its precursor and peptide siblings; `run-experiment` runs it on every multi-run
//! experiment (`lfq_maxlfq.parquet`, `lfq_size_factors.tsv`).

use anyhow::{Context as _, Result};
use mumdia_core::config::NormalizeMethod;
use mumdia_io::table::{write_table, Col, TableFile};
use rayon::prelude::*;
use tracing::{info, warn};

/// Solve the reduced normal system `L x = c` for a connected component with the
/// first variable fixed at 0 (removing the Laplacian's rank deficiency). Dense
/// Gaussian elimination with partial pivoting; components are small.
fn solve_fixed(l: &[Vec<f64>], c: &[f64]) -> Vec<f64> {
    let n = l.len();
    if n <= 1 {
        return vec![0.0; n];
    }
    let m = n - 1; // free variables are indices 1..n
    let mut a = vec![vec![0.0f64; m + 1]; m]; // augmented [A | b]
    for i in 0..m {
        for j in 0..m {
            a[i][j] = l[i + 1][j + 1];
        }
        a[i][m] = c[i + 1];
    }
    // forward elimination with partial pivot
    for col in 0..m {
        let mut piv = col;
        for r in (col + 1)..m {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(col, piv);
        let d = a[col][col];
        if d.abs() < 1e-12 {
            continue;
        }
        let pivot_row = a[col].clone();
        for row in a[col + 1..m].iter_mut() {
            let f = row[col] / d;
            if f == 0.0 {
                continue;
            }
            for (value, &pivot_value) in row[col..=m].iter_mut().zip(pivot_row[col..=m].iter()) {
                *value -= f * pivot_value;
            }
        }
    }
    // back substitution
    let mut x = vec![0.0f64; n]; // x[0] fixed at 0
    for i in (0..m).rev() {
        let mut s = a[i][m];
        for j in (i + 1)..m {
            s -= a[i][j] * x[j + 1];
        }
        let d = a[i][i];
        x[i + 1] = if d.abs() < 1e-12 { 0.0 } else { s / d };
    }
    x
}

/// MaxLFQ-style per-sample profile. `mat[feature][sample]` holds the intensity
/// (positive) or `None` if the feature is missing in that sample. Returns a
/// per-sample abundance (length `n_samples`). Each connected component of the
/// feature-sharing graph is fit by least squares to the pairwise median log
/// ratios, then anchored so the component preserves its measured total intensity;
/// isolated samples fall back to their column sum.
///
/// The rows are taken as anything that borrows as `[Option<f64>]`, so a caller that
/// already owns the per-feature vectors (the cross-run combine, which rolls the same
/// features up at protein, precursor and peptide level) passes references rather than
/// three clones of every feature vector. Read-only: the arithmetic and the iteration
/// order over rows are unchanged.
pub fn lfq_profile<R: AsRef<[Option<f64>]>>(mat: &[R], n_samples: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n_samples];
    if n_samples == 0 {
        return out;
    }
    let mut colsum = vec![0.0f64; n_samples];
    let mut colcount = vec![0usize; n_samples];
    for row in mat {
        let row = row.as_ref();
        for s in 0..n_samples {
            if let Some(v) = row.get(s).copied().flatten() {
                if v > 0.0 {
                    colsum[s] += v;
                    colcount[s] += 1;
                }
            }
        }
    }
    if n_samples == 1 {
        return colsum;
    }

    // Pairwise median log-ratios (edges of the sample graph).
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for a in 0..n_samples {
        for b in (a + 1)..n_samples {
            let mut d = Vec::new();
            for row in mat {
                let row = row.as_ref();
                if let (Some(va), Some(vb)) =
                    (row.get(a).copied().flatten(), row.get(b).copied().flatten())
                {
                    if va > 0.0 && vb > 0.0 {
                        d.push(va.ln() - vb.ln());
                    }
                }
            }
            if !d.is_empty() {
                edges.push((a, b, crate::stats::median_or(&mut d, 0.0)));
            }
        }
    }

    // Connected components via union-find.
    let mut uf: Vec<usize> = (0..n_samples).collect();
    fn find(uf: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while uf[r] != r {
            r = uf[r];
        }
        let mut c = x;
        while uf[c] != c {
            let nx = uf[c];
            uf[c] = r;
            c = nx;
        }
        r
    }
    for &(a, b, _) in &edges {
        let ra = find(&mut uf, a);
        let rb = find(&mut uf, b);
        uf[ra] = rb;
    }
    let mut comps: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for s in 0..n_samples {
        let r = find(&mut uf, s);
        comps.entry(r).or_default().push(s);
    }

    for members in comps.values() {
        if members.len() == 1 {
            let s = members[0];
            out[s] = colsum[s];
            continue;
        }
        let idx: std::collections::HashMap<usize, usize> =
            members.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let n = members.len();
        let mut l = vec![vec![0.0f64; n]; n];
        let mut c = vec![0.0f64; n];
        for &(a, b, r) in &edges {
            let (ia, ib) = match (idx.get(&a), idx.get(&b)) {
                (Some(&ia), Some(&ib)) => (ia, ib),
                _ => continue,
            };
            l[ia][ia] += 1.0;
            l[ib][ib] += 1.0;
            l[ia][ib] -= 1.0;
            l[ib][ia] -= 1.0;
            c[ia] += r;
            c[ib] -= r;
        }
        let x = solve_fixed(&l, &c); // log-abundance, x[0] = 0
                                     // Anchor so the component's exp-profile preserves its measured total.
        let measured: f64 = members.iter().map(|&s| colsum[s]).sum();
        let expsum: f64 = x.iter().map(|v| v.exp()).sum();
        let scale = if expsum > 0.0 { measured / expsum } else { 0.0 };
        for (i, &s) in members.iter().enumerate() {
            out[s] = if colcount[s] > 0 {
                x[i].exp() * scale
            } else {
                0.0
            };
        }
    }
    out
}

/// The cross-run precursor matrix `quant-lfq` writes next to the protein matrix at `lfq`.
pub fn precursor_lfq_path(lfq: &str) -> String {
    format!("{lfq}.precursor.parquet")
}

/// [`precursor_lfq_path`] when that file exists (an experiment from before the sibling
/// matrices has none).
pub fn precursor_lfq_sibling(lfq: &str) -> Option<String> {
    let p = precursor_lfq_path(lfq);
    std::path::Path::new(&p).is_file().then_some(p)
}

/// The size factor of each run of a cross-run LFQ, as [`run_lfq_combine_grouped`] applied
/// it, and the number of features it was estimated from.
#[derive(Clone, Debug, PartialEq)]
pub struct SizeFactor {
    pub factor: f64,
    /// Complete-case features of the run's group (`median_ratio`), the run's own features
    /// (`median`), or 0 when no factor was estimated (`none`, a group of one run, a run
    /// with no quantified feature, or no complete feature).
    pub n_features: usize,
}

/// Combine several per-run quant tables into a protein-by-run abundance matrix
/// with MaxLFQ (peptide-level: `by_fragment=false`, reads `peptide_quant`) or
/// directLFQ (ion-level: `by_fragment=true`, reads `fragment_quant`). Each
/// protein's feature-by-run intensity matrix is passed to the ratio-alignment
/// core [`lfq_profile`]. Output is long form: protein_group, run,
/// quantity, n_features. With one input this reduces to the per-run sum.
///
/// `normalize` applies a cross-run size factor to the feature-by-run matrix
/// before rollup (see `size_factors_counted`). `groups` holds one group index per input
/// (`quant.normalisation_scope = condition`): each group's factors are those of `normalize`
/// over that group's runs alone, so a group of one run keeps factor 1. `None` estimates one
/// set over every run. `quantity_column` names the intensity column of
/// the inputs: `quantity`, or `quantity_normalised` of the `peptide_quant_normalised`
/// tables of the cross-run normalisation (`stages::quant_norm`), which are then read
/// with `NormalizeMethod::None`. Returns the row count and each run's factor.
pub fn run_lfq_combine_grouped(
    inputs: &[String],
    by_fragment: bool,
    normalize: NormalizeMethod,
    groups: Option<&[usize]>,
    quantity_column: &str,
    out: &str,
) -> Result<(u64, Vec<SizeFactor>)> {
    use std::collections::BTreeMap;
    let n = inputs.len();
    // protein_group -> feature key -> per-run intensity
    let mut data: BTreeMap<String, BTreeMap<String, Vec<Option<f64>>>> = BTreeMap::new();
    for (ri, path) in inputs.iter().enumerate() {
        let t = TableFile::open(path)?;
        let pform = t.str("peptidoform")?;
        let z = t.i32("charge")?;
        let pgc = t.str("protein_group")?;
        let q = t.opt_f64(quantity_column)?;
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
            // Same reason as `add_protein_base_quantity`: `entry` would clone the protein
            // group name on every row, and a run's quant table repeats each group many
            // times over.
            if !data.contains_key(&pgc[i]) {
                data.insert(pgc[i].clone(), BTreeMap::new());
            }
            let slot = &mut data
                .get_mut(&pgc[i])
                .expect("the group was just inserted")
                .entry(key)
                .or_insert_with(|| vec![None; n])[ri];
            *slot = Some(slot.map_or(quantity, |previous| previous.max(quantity)));
        }
    }
    // Cross-run normalization: one global size factor per run, estimated from the
    // whole feature-by-run matrix and applied before protein rollup so both the
    // MaxLFQ profile and any downstream ratio inherit the corrected scale.
    if let Some(g) = groups {
        anyhow::ensure!(
            g.len() == n,
            "quant-lfq: {} condition labels for {n} input tables",
            g.len()
        );
    }
    let sized = match groups {
        Some(g) => grouped_size_factors(&data, g, normalize),
        None => size_factors_counted(&data, &(0..n).collect::<Vec<_>>(), normalize),
    };
    let factors: Vec<f64> = sized.iter().map(|s| s.factor).collect();
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
    // The rollup levels below read the SAME feature vectors, so each level borrows them
    // from `data` rather than cloning them: with the protein level and the two sibling
    // levels this held three or four copies of every feature vector at once. The groups'
    // profiles are independent, so they are solved in parallel; `collect` keeps the group
    // order, so the table is the serial loop's byte for byte.
    let c_q: Vec<f64> = data
        .par_iter()
        .flat_map_iter(|(_, feats)| lfq_profile(&feats.values().collect::<Vec<_>>(), n))
        .collect();
    let (mut c_pg, mut c_run, mut c_nf) = (Vec::new(), Vec::new(), Vec::new());
    for (pgname, feats) in &data {
        for r in 0..n {
            c_pg.push(pgname.clone());
            c_run.push(r as i32);
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
    let mut prec: BTreeMap<(String, i32), Vec<&Vec<Option<f64>>>> = BTreeMap::new();
    let mut pep: BTreeMap<String, Vec<&Vec<Option<f64>>>> = BTreeMap::new();
    for feats in data.values() {
        for (key, vec) in feats {
            // key = "peptidoform|charge" (maxlfq) or "peptidoform|charge|fragment"
            // (directlfq); peptidoform strings never contain '|'.
            let mut it = key.splitn(3, '|');
            let pform = it.next().unwrap_or("");
            let charge: i32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            prec.entry((pform.to_string(), charge))
                .or_default()
                .push(vec);
            pep.entry(base_sequence(pform)).or_default().push(vec);
        }
    }
    // (group key, charge, feature-by-run matrix) for one sibling-matrix level. The matrix
    // borrows its rows from `data`.
    type LevelGroup<'a> = (String, i32, Vec<&'a Vec<Option<f64>>>);
    let write_level = |path: String, groups: Vec<LevelGroup>| -> Result<()> {
        let g_q: Vec<f64> = groups
            .par_iter()
            .flat_map_iter(|(_, _, mat)| lfq_profile(mat, n))
            .collect();
        let (mut g_key, mut g_z, mut g_run, mut g_nf) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (key, z, mat) in &groups {
            for r in 0..n {
                g_key.push(key.clone());
                g_z.push(*z);
                g_run.push(r as i32);
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
        precursor_lfq_path(out),
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
    Ok((rows, sized))
}

/// The file name of the size-factor table `run-experiment` writes at the experiment root.
pub const LFQ_SIZE_FACTORS_TSV: &str = "lfq_size_factors.tsv";

/// Write the size factor of every run of a cross-run LFQ as a tab-separated table: `run`,
/// `condition` (empty when the run has none), `size_factor` (what the run's feature
/// intensities were divided by; 1 when none was applied) and `n_features` (the features it
/// was estimated from). One row per run, in run order.
pub fn write_size_factors_tsv(
    path: &str,
    runs: &[String],
    conditions: &[Option<String>],
    factors: &[SizeFactor],
) -> Result<()> {
    use std::io::Write;
    anyhow::ensure!(
        runs.len() == factors.len() && conditions.len() == runs.len(),
        "{path}: {} runs, {} conditions and {} size factors",
        runs.len(),
        conditions.len(),
        factors.len()
    );
    let mut w = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("creating {path}"))?,
    );
    writeln!(w, "run\tcondition\tsize_factor\tn_features")?;
    for ((run, cond), f) in runs.iter().zip(conditions).zip(factors) {
        writeln!(
            w,
            "{run}\t{}\t{}\t{}",
            cond.as_deref().unwrap_or(""),
            f.factor,
            f.n_features
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Stripped base amino-acid sequence of a peptidoform: drop bracketed or
/// parenthesized modification blocks and any DECOY_ prefix, keep the residues.
/// Used to roll precursors up to peptide-level LFQ groups, and for the stripped sequences of
/// the TSV report.
pub(crate) fn base_sequence(peptidoform: &str) -> String {
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
///   positive in every run that holds any quantified feature) take its ratio to a
///   geometric-mean pseudo-reference; the run factor is the median of those ratios.
///   Robust to a minority of genuinely changing features, so a spike-in design's real
///   fold changes are preserved, not flattened. A run with no quantified feature keeps
///   factor 1.
/// - `Median`: align each run's median log2 intensity to the median of the
///   per-run medians.
/// - `None`: all factors 1.0.
///
/// Medians are taken over sorted values and the matrix is iterated in `BTreeMap`
/// key order, so the result is deterministic (docs/14_build_test_deploy_gotchas.md).
#[cfg(test)]
fn size_factors(
    data: &std::collections::BTreeMap<String, std::collections::BTreeMap<String, Vec<Option<f64>>>>,
    n: usize,
    method: NormalizeMethod,
) -> Vec<f64> {
    size_factors_counted(data, &(0..n).collect::<Vec<_>>(), method)
        .into_iter()
        .map(|s| s.factor)
        .collect()
}

/// Below this many complete-case features a median-ratio size factor is little more than a
/// guess, and `quant-lfq` warns.
const MIN_SIZE_FACTOR_FEATURES: usize = 20;

/// [`size_factors`] over the runs `runs` of the matrix alone, one entry per entry of `runs`,
/// with the number of features each factor was estimated from. Over every run (`0..n`) it
/// is `size_factors` exactly: the same features in the same order, summed in the same order.
fn size_factors_counted(
    data: &std::collections::BTreeMap<String, std::collections::BTreeMap<String, Vec<Option<f64>>>>,
    runs: &[usize],
    method: NormalizeMethod,
) -> Vec<SizeFactor> {
    let n = runs.len();
    match method {
        NormalizeMethod::None => vec![
            SizeFactor {
                factor: 1.0,
                n_features: 0
            };
            n
        ],
        NormalizeMethod::MedianRatio => {
            // A run with no quantified feature (a blank well, a failed injection) leaves no
            // feature complete over every run, which switched normalisation off for all of
            // them. It keeps factor 1 now, and the others are fitted over the rest.
            let positive = |v: &Option<f64>| v.is_some_and(|v| v > 0.0);
            let held: Vec<usize> = (0..n)
                .filter(|&k| {
                    data.values()
                        .any(|feats| feats.values().any(|vec| positive(&vec[runs[k]])))
                })
                .collect();
            let mut lr: Vec<Vec<f64>> = vec![Vec::new(); n];
            for feats in data.values() {
                for vec in feats.values() {
                    if held.iter().all(|&k| positive(&vec[runs[k]])) {
                        let logs: Vec<f64> =
                            held.iter().map(|&k| vec[runs[k]].unwrap().log2()).collect();
                        let refm = logs.iter().sum::<f64>() / held.len() as f64;
                        for (&k, l) in held.iter().zip(&logs) {
                            lr[k].push(l - refm);
                        }
                    }
                }
            }
            if held.len() < n {
                let empty: Vec<usize> = (0..n)
                    .filter(|k| !held.contains(k))
                    .map(|k| runs[k])
                    .collect();
                warn!(
                    runs = ?empty,
                    "quant-lfq: these runs (0-based, in input order) hold no quantified \
                     feature; they keep size factor 1, and the other runs' factors are \
                     estimated without them"
                );
            }
            let complete = held.first().map_or(0, |&k| lr[k].len());
            if held.len() > 1 && complete < MIN_SIZE_FACTOR_FEATURES {
                warn!(
                    features = complete,
                    runs = ?held.iter().map(|&k| runs[k]).collect::<Vec<_>>(),
                    "quant-lfq: the size factors of these runs rest on fewer than \
                     {MIN_SIZE_FACTOR_FEATURES} features quantified in all of them"
                );
            }
            (0..n)
                .map(|k| SizeFactor {
                    n_features: lr[k].len(),
                    factor: if lr[k].is_empty() {
                        1.0
                    } else {
                        2f64.powf(crate::stats::median_or(&mut lr[k], 0.0))
                    },
                })
                .collect()
        }
        NormalizeMethod::Median => {
            let mut logs: Vec<Vec<f64>> = vec![Vec::new(); n];
            for feats in data.values() {
                for vec in feats.values() {
                    for (k, &r) in runs.iter().enumerate() {
                        if let Some(v) = vec[r] {
                            if v > 0.0 {
                                logs[k].push(v.log2());
                            }
                        }
                    }
                }
            }
            let counts: Vec<usize> = logs.iter().map(Vec::len).collect();
            let med: Vec<f64> = (0..n)
                .map(|k| {
                    if logs[k].is_empty() {
                        0.0
                    } else {
                        crate::stats::median_or(&mut logs[k], 0.0)
                    }
                })
                .collect();
            let mut m2 = med.clone();
            let target = crate::stats::median_or(&mut m2, 0.0);
            (0..n)
                .map(|k| SizeFactor {
                    factor: 2f64.powf(med[k] - target),
                    n_features: counts[k],
                })
                .collect()
        }
    }
}

/// Size factors estimated separately within each group of runs (`groups[r]` is run `r`'s
/// group), scattered back to run order. A group of one run has nothing to be normalised
/// against and keeps factor 1 (`n_features` 0).
fn grouped_size_factors(
    data: &std::collections::BTreeMap<String, std::collections::BTreeMap<String, Vec<Option<f64>>>>,
    groups: &[usize],
    method: NormalizeMethod,
) -> Vec<SizeFactor> {
    let mut members: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (r, &g) in groups.iter().enumerate() {
        members.entry(g).or_default().push(r);
    }
    let mut out = vec![
        SizeFactor {
            factor: 1.0,
            n_features: 0
        };
        groups.len()
    ];
    for runs in members.values() {
        if runs.len() < 2 {
            continue;
        }
        for (k, f) in size_factors_counted(data, runs, method)
            .into_iter()
            .enumerate()
        {
            out[runs[k]] = f;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ratios(v: &[f64]) -> Vec<f64> {
        let b = v[0];
        v.iter().map(|x| x / b).collect()
    }

    #[test]
    fn single_sample_is_column_sum() {
        let mat = vec![vec![Some(3.0)], vec![Some(4.0)], vec![None]];
        assert_eq!(lfq_profile(&mat, 1), vec![7.0]);
    }

    #[test]
    fn recovers_known_profile_complete() {
        // 3 samples with true abundances 1:2:4; each of 4 peptides scales it.
        let true_prof = [1.0, 2.0, 4.0];
        let pep_factor = [10.0, 5.0, 20.0, 3.0];
        let mat: Vec<Vec<Option<f64>>> = pep_factor
            .iter()
            .map(|f| true_prof.iter().map(|p| Some(f * p)).collect())
            .collect();
        let q = lfq_profile(&mat, 3);
        let r = ratios(&q);
        assert!((r[1] - 2.0).abs() < 1e-6, "r1={}", r[1]);
        assert!((r[2] - 4.0).abs() < 1e-6, "r2={}", r[2]);
    }

    #[test]
    fn recovers_profile_with_missing_values() {
        // Same profile, but each peptide observed in only a subset of samples.
        let true_prof = [1.0, 2.0, 4.0];
        let mat = vec![
            vec![Some(10.0), Some(20.0), None],      // pep in s0,s1
            vec![None, Some(10.0), Some(20.0)],      // pep in s1,s2
            vec![Some(3.0), None, Some(12.0)],       // pep in s0,s2
            vec![Some(5.0), Some(10.0), Some(20.0)], // pep in all
        ];
        let _ = true_prof;
        let q = lfq_profile(&mat, 3);
        let r = ratios(&q);
        assert!((r[1] - 2.0).abs() < 1e-6, "r1={}", r[1]);
        assert!((r[2] - 4.0).abs() < 1e-6, "r2={}", r[2]);
    }

    #[test]
    fn total_intensity_preserved() {
        let mat = vec![
            vec![Some(10.0), Some(20.0), Some(40.0)],
            vec![Some(5.0), Some(10.0), Some(20.0)],
        ];
        let q = lfq_profile(&mat, 3);
        let measured: f64 = 10.0 + 20.0 + 40.0 + 5.0 + 10.0 + 20.0;
        let got: f64 = q.iter().sum();
        assert!(
            (got - measured).abs() < 1e-6,
            "sum={} want={}",
            got,
            measured
        );
    }

    #[test]
    fn borrowed_rows_give_the_same_profile_as_owned_rows() {
        // The cross-run combine passes `&Vec<Option<f64>>` rows so the same feature
        // vector is not cloned once per rollup level. The profile must be bit-identical
        // to the owned form it replaced.
        let owned = vec![
            vec![Some(10.0), Some(20.0), None],
            vec![None, Some(10.0), Some(20.0)],
            vec![Some(3.0), None, Some(12.0)],
            vec![Some(5.0), Some(10.0), Some(20.0)],
        ];
        let borrowed: Vec<&Vec<Option<f64>>> = owned.iter().collect();
        assert_eq!(lfq_profile(&owned, 3), lfq_profile(&borrowed, 3));
        let slices: Vec<&[Option<f64>]> = owned.iter().map(|v| v.as_slice()).collect();
        assert_eq!(lfq_profile(&owned, 3), lfq_profile(&slices, 3));
    }

    #[test]
    fn disconnected_sample_falls_back_to_sum() {
        // s2 shares no peptide with s0/s1 -> its own column sum.
        let mat = vec![
            vec![Some(10.0), Some(20.0), None],
            vec![Some(5.0), Some(10.0), None],
            vec![None, None, Some(7.0)],
        ];
        let q = lfq_profile(&mat, 3);
        assert!((q[2] - 7.0).abs() < 1e-6, "q2={}", q[2]);
    }

    #[test]
    fn base_sequence_strips_mods_and_decoy() {
        assert_eq!(base_sequence("PEPTIDEK"), "PEPTIDEK");
        assert_eq!(base_sequence("M[Oxidation]PEC[Carbamidomethyl]K"), "MPECK");
        assert_eq!(base_sequence("DECOY_VAVGDGVAK"), "VAVGDGVAK");
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

    /// A run with no quantified feature (a blank well) keeps factor 1, and the 2x between
    /// the other two runs survives it: their factors are those of the two runs alone. It
    /// used to leave no complete feature, so every run kept factor 1.
    #[test]
    fn an_empty_run_keeps_factor_one_and_the_others_are_normalised() {
        use std::collections::BTreeMap;
        let mut feats: BTreeMap<String, Vec<Option<f64>>> = BTreeMap::new();
        for i in 0..8 {
            let a = 100.0 * (i as f64 + 1.0);
            feats.insert(format!("f{i}"), vec![Some(a), Some(2.0 * a), None]);
        }
        let mut data: BTreeMap<String, BTreeMap<String, Vec<Option<f64>>>> = BTreeMap::new();
        data.insert("PG".into(), feats);
        let m = NormalizeMethod::MedianRatio;
        let f = size_factors_counted(&data, &[0, 1, 2], m);
        assert!((f[1].factor / f[0].factor - 2.0).abs() < 1e-12, "{f:?}");
        assert_eq!((f[0].n_features, f[1].n_features), (8, 8));
        assert_eq!(
            f[2],
            SizeFactor {
                factor: 1.0,
                n_features: 0
            }
        );
        assert_eq!(f[..2], size_factors_counted(&data, &[0, 1], m)[..]);
    }

    /// A four-run design, two conditions of two replicates, where condition B has a real
    /// 4x more of every feature and run 1 a 2x loading error within A. Per condition, the
    /// factors are the factors of each condition's runs alone (geometric mean 1 within a
    /// condition), so the loading error goes and the difference between conditions stays as
    /// measured; over all runs that difference is taken for loading and removed. A condition of one
    /// run keeps factor 1, and the ungrouped kernel equals the subset kernel over `0..n`.
    #[test]
    fn size_factors_within_conditions_keep_the_difference_between_them() {
        use std::collections::BTreeMap;
        let mut feats: BTreeMap<String, Vec<Option<f64>>> = BTreeMap::new();
        for i in 0..9 {
            let a = 100.0 * (i as f64 + 1.0);
            feats.insert(
                format!("f{i}"),
                vec![Some(a), Some(2.0 * a), Some(4.0 * a), Some(4.0 * a)],
            );
        }
        let mut data: BTreeMap<String, BTreeMap<String, Vec<Option<f64>>>> = BTreeMap::new();
        data.insert("PG".into(), feats);
        let m = NormalizeMethod::MedianRatio;
        let by_cond = grouped_size_factors(&data, &[0, 0, 1, 1], m);
        let f: Vec<f64> = by_cond.iter().map(|s| s.factor).collect();
        assert!((f[1] / f[0] - 2.0).abs() < 1e-9, "{f:?}");
        assert!(
            (f[0] * f[1] - 1.0).abs() < 1e-9,
            "geometric mean 1 within A: {f:?}"
        );
        assert!(
            (f[2] - 1.0).abs() < 1e-9 && (f[3] - 1.0).abs() < 1e-9,
            "{f:?}"
        );
        assert!(by_cond.iter().all(|s| s.n_features == 9));
        // Normalised, run 0 and run 1 agree, and B over A is the measured ratio of B to A's
        // geometric mean, 400 / sqrt(100 x 200).
        let v = |r: usize, x: f64| x / f[r];
        assert!((v(1, 200.0) / v(0, 100.0) - 1.0).abs() < 1e-9);
        assert!((v(2, 400.0) / v(0, 100.0) - 400.0 / 20_000f64.sqrt()).abs() < 1e-9);
        // Over all runs the 4x between conditions is removed as if it were loading.
        let all = size_factors(&data, 4, m);
        assert!(
            ((400.0 / all[2]) / (100.0 / all[0]) - 4.0).abs() > 1.0,
            "{all:?}"
        );
        // The subset kernel over every run is the ungrouped kernel, bit for bit.
        let every: Vec<f64> = size_factors_counted(&data, &[0, 1, 2, 3], m)
            .iter()
            .map(|s| s.factor)
            .collect();
        assert_eq!(every, all);
        // A condition with one run keeps 1; `none` gives 1 everywhere.
        let lone = grouped_size_factors(&data, &[0, 0, 0, 1], m);
        assert_eq!(
            lone[3],
            SizeFactor {
                factor: 1.0,
                n_features: 0
            }
        );
        assert!(lone[..3].iter().all(|s| s.n_features == 9));
        let none = grouped_size_factors(&data, &[0, 0, 1, 1], NormalizeMethod::None);
        assert!(none.iter().all(|s| s.factor == 1.0));
        // A feature missing in run 3 leaves condition A's complete cases untouched.
        data.get_mut("PG")
            .unwrap()
            .insert("gap".into(), vec![Some(1.0), Some(9.0), Some(5.0), None]);
        let gap = grouped_size_factors(&data, &[0, 0, 1, 1], m);
        assert_eq!(gap[0].n_features, 10);
        assert_eq!(gap[2].n_features, 9);
    }

    /// End to end through the parquet tables: the grouped combine writes the same LFQ
    /// table as the ungrouped one when every run is its own group under `none`, and refuses
    /// a group list that does not match the inputs.
    #[test]
    fn grouped_lfq_combine_checks_its_groups() {
        let dir = std::env::temp_dir().join(format!("mumdia_lfq_groups_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut inputs = Vec::new();
        for (r, scale) in [1.0f64, 2.0, 8.0].iter().enumerate() {
            let path = dir
                .join(format!("pq{r}.parquet"))
                .to_string_lossy()
                .to_string();
            write_table(
                &path,
                vec![
                    Col::Str("peptidoform".into(), vec!["PEPTIDEK".into(), "AAAK".into()]),
                    Col::I32("charge".into(), vec![2, 2]),
                    Col::Str("protein_group".into(), vec!["P1".into(), "P1".into()]),
                    Col::F64("quantity".into(), vec![100.0 * scale, 50.0 * scale]),
                ],
            )
            .unwrap();
            inputs.push(path);
        }
        let out = dir.join("lfq.parquet").to_string_lossy().to_string();
        let (_, f) = run_lfq_combine_grouped(
            &inputs,
            false,
            NormalizeMethod::MedianRatio,
            Some(&[0, 0, 1]),
            "quantity",
            &out,
        )
        .unwrap();
        assert!((f[1].factor / f[0].factor - 2.0).abs() < 1e-9, "{f:?}");
        assert_eq!(f[2].factor, 1.0);
        assert!(run_lfq_combine_grouped(
            &inputs,
            false,
            NormalizeMethod::MedianRatio,
            Some(&[0, 1]),
            "quantity",
            &out,
        )
        .is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
