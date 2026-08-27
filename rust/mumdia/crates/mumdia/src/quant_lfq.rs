//! MaxLFQ / directLFQ cross-run label-free quantification
//! (docs/12_quant_lfq_align_mbr_report_audit.md, beyond-MVP). These are
//! inherently MULTI-RUN methods: they reconstruct a
//! per-sample protein abundance profile from the pairwise median log-ratios of
//! the features shared between samples (MaxLFQ: peptide-level; directLFQ:
//! ion/fragment-level), so with a single run there are no cross-sample ratios and
//! the profile reduces to the per-sample summed intensity. The shared core is
//! [`lfq_profile`]; `maxlfq`/`directlfq` differ only in the feature granularity
//! the caller passes. Validated on synthetic multi-sample matrices (unit tests);
//! real-data validation requires >=2 runs, which the MVP single-run pipeline does
//! not yet orchestrate.

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

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
pub fn lfq_profile(mat: &[Vec<Option<f64>>], n_samples: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n_samples];
    if n_samples == 0 {
        return out;
    }
    let mut colsum = vec![0.0f64; n_samples];
    let mut colcount = vec![0usize; n_samples];
    for row in mat {
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
                if let (Some(va), Some(vb)) =
                    (row.get(a).copied().flatten(), row.get(b).copied().flatten())
                {
                    if va > 0.0 && vb > 0.0 {
                        d.push(va.ln() - vb.ln());
                    }
                }
            }
            if !d.is_empty() {
                edges.push((a, b, median(&mut d)));
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

/// MaxLFQ (Cox et al. 2014): peptide-level cross-sample profile.
pub fn maxlfq(peptides_by_sample: &[Vec<Option<f64>>], n_samples: usize) -> Vec<f64> {
    lfq_profile(peptides_by_sample, n_samples)
}

/// directLFQ (Ammar et al. 2023) core: the same ratio-alignment applied at the
/// ion/fragment level rather than the peptide level (the caller passes an
/// ion-by-sample matrix).
pub fn directlfq(ions_by_sample: &[Vec<Option<f64>>], n_samples: usize) -> Vec<f64> {
    lfq_profile(ions_by_sample, n_samples)
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
        assert_eq!(maxlfq(&mat, 1), vec![7.0]);
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
        let q = maxlfq(&mat, 3);
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
        let q = maxlfq(&mat, 3);
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
        let q = maxlfq(&mat, 3);
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
    fn disconnected_sample_falls_back_to_sum() {
        // s2 shares no peptide with s0/s1 -> its own column sum.
        let mat = vec![
            vec![Some(10.0), Some(20.0), None],
            vec![Some(5.0), Some(10.0), None],
            vec![None, None, Some(7.0)],
        ];
        let q = maxlfq(&mat, 3);
        assert!((q[2] - 7.0).abs() < 1e-6, "q2={}", q[2]);
    }
}
