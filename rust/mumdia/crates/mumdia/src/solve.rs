//! Deterministic pure-Rust non-negative least squares for spectrum-centric fragment
//! demixing (docs/11_compete_rescore_fdr.md).
//!
//! Solves `min_{x >= 0} ||A x - b||^2` where `A` (row-major `m x n`) is the co-isolated
//! candidate design matrix (rows = observed fragment channels, columns = candidates,
//! `A[m,p]` = candidate p's predicted intensity for the fragment matching channel m) and
//! `b` is the observed intensity vector. The solution `x_p` is candidate p's
//! interference-corrected abundance; a shared channel is apportioned in the proportion
//! that jointly minimizes the residual over all channels (the CHIMERYS coefficient idea,
//! reimplemented clean-room).
//!
//! Determinism + clean-room constraints (CLAUDE.md): the passive-set least squares is the
//! RIDGE-regularized normal equations `(A_P^T A_P + lambda I) z = A_P^T b` solved by a
//! hand-rolled Cholesky. Ridge makes the Gram strictly positive definite, so the solve is
//! unique and bit-stable even under the ~98% column collinearity of wide-window DIA (the
//! deliberate divergence from CHIMERYS's LASSO+AICc). All reductions run in fixed index
//! order; no BLAS/LAPACK (the Windows/OneDrive build bars C deps); no randomness.

/// Reusable buffers for [`nnls_with`], plus the full normal equations for one problem.
///
/// The active-set loop is allocation-heavy by nature: every move needs a passive-column
/// list, a solution vector, a Gram, a right-hand side and a Cholesky factor. Holding
/// them here lets a caller that solves many small problems (the demix path solves one
/// per candidate) allocate once instead of once per move.
///
/// `ata` and `atb` hold `A^T A` and `A^T b` for the current problem. They are computed
/// once per [`nnls_with`] call and indexed per move, replacing a rebuild of the passive
/// Gram on every move. Every entry is summed over rows in ascending order, exactly as
/// the per-move accumulation did, so the values are bit-identical. The precompute is
/// `O(m n^2)`, i.e. at most the cost of the one move where all `n` columns are passive,
/// and callers bound `n` by `extract.demix_max_candidates`.
#[derive(Default)]
pub struct NnlsScratch {
    ata: Vec<f64>,
    atb: Vec<f64>,
    passive: Vec<bool>,
    pcols: Vec<usize>,
    ax: Vec<f64>,
    resid: Vec<f64>,
    w: Vec<f64>,
    z: Vec<f64>,
    zp: Vec<f64>,
    g: Vec<f64>,
    rhs: Vec<f64>,
    l: Vec<f64>,
    y: Vec<f64>,
}

impl NnlsScratch {
    pub fn new() -> NnlsScratch {
        NnlsScratch::default()
    }

    /// Resize every buffer for an `m x n` problem and fill the normal equations.
    fn prepare(&mut self, a: &[f64], m: usize, n: usize, b: &[f64]) {
        fill(&mut self.ata, n * n, 0.0);
        fill(&mut self.atb, n, 0.0);
        fill(&mut self.passive, n, false);
        fill(&mut self.ax, m, 0.0);
        fill(&mut self.resid, m, 0.0);
        fill(&mut self.w, n, 0.0);
        fill(&mut self.z, n, 0.0);
        self.pcols.clear();
        // Row-major accumulation in ascending row order: the same terms in the same
        // order the per-move passive Gram used.
        for i in 0..m {
            let row = &a[i * n..i * n + n];
            let bi = b[i];
            for r in 0..n {
                let air = row[r];
                self.atb[r] += air * bi;
                let g_row = &mut self.ata[r * n..r * n + n];
                for (c, gc) in g_row.iter_mut().enumerate() {
                    *gc += air * row[c];
                }
            }
        }
    }
}

/// Resize `v` to `len` and set every element to `val`.
fn fill<T: Copy>(v: &mut Vec<T>, len: usize, val: T) {
    v.clear();
    v.resize(len, val);
}

/// Solve `(G + lambda I) z = rhs` for a symmetric positive-definite `G` (row-major
/// `k x k`) via Cholesky `G = L L^T` plus forward/back substitution. Writes the result
/// into `out`. Returns `false` only if a non-positive pivot survives the ridge (should
/// not happen with lambda > 0), in which case `out` is left zeroed.
fn cholesky_solve_into(
    g: &[f64],
    k: usize,
    rhs: &[f64],
    lambda: f64,
    l: &mut Vec<f64>,
    y: &mut Vec<f64>,
    out: &mut Vec<f64>,
) -> bool {
    fill(l, k * k, 0.0);
    fill(y, k, 0.0);
    fill(out, k, 0.0);
    // L is lower-triangular, row-major k x k.
    for i in 0..k {
        for j in 0..=i {
            let mut s = g[i * k + j];
            if i == j {
                s += lambda;
            }
            for p in 0..j {
                s -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                if s <= 0.0 {
                    fill(out, k, 0.0);
                    return false;
                }
                l[i * k + j] = s.sqrt();
            } else {
                l[i * k + j] = s / l[j * k + j];
            }
        }
    }
    // Forward solve L y = rhs.
    for i in 0..k {
        let mut s = rhs[i];
        for p in 0..i {
            s -= l[i * k + p] * y[p];
        }
        y[i] = s / l[i * k + i];
    }
    // Back solve L^T z = y.
    for ii in (0..k).rev() {
        let mut s = y[ii];
        for p in (ii + 1)..k {
            s -= l[p * k + ii] * out[p];
        }
        out[ii] = s / l[ii * k + ii];
    }
    true
}

/// Non-negative least squares (Lawson-Hanson active set), ridge-regularized on the
/// passive solve. `a` is row-major `m x n`, `b` has length `m`. Returns `x` (length `n`)
/// with `x_j >= 0`. `lambda` is the ridge added to the passive Gram diagonal (> 0 keeps
/// it PD and deterministic under collinearity). `max_iter` caps the active-set moves.
///
/// Allocates fresh scratch; use [`nnls_with`] to reuse buffers across many solves.
pub fn nnls(a: &[f64], m: usize, n: usize, b: &[f64], lambda: f64, max_iter: usize) -> Vec<f64> {
    let mut s = NnlsScratch::new();
    nnls_with(&mut s, a, m, n, b, lambda, max_iter)
}

/// [`nnls`] reusing `s`'s buffers. Bit-identical to `nnls` for the same inputs; the
/// scratch carries no state between calls beyond allocated capacity.
pub fn nnls_with(
    s: &mut NnlsScratch,
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    lambda: f64,
    max_iter: usize,
) -> Vec<f64> {
    let mut x = vec![0.0f64; n];
    if n == 0 || m == 0 {
        return x;
    }
    s.prepare(a, m, n, b);
    // Positive-gradient tolerance: below this a column is not worth activating.
    let tol = 1e-9;
    let mut moves = 0usize;
    loop {
        // w = A^T (b - A x), computed through the explicit residual.
        //
        // Deliberately NOT rewritten as `A^T b - (A^T A) x` using the precomputed normal
        // equations. Those forms are algebraically equal but not bit-identical, and `w`
        // decides which column enters the passive set: a one-ULP difference near `tol`,
        // or between two nearly-equal gradients, would select a different column and
        // change the apportionment -- i.e. change extracted intensities and therefore
        // identifications. The precompute is used only where it is exactly equivalent
        // (the passive Gram and right-hand side below). This loop is O(m n) per move,
        // far below the O(m k^2) Gram rebuild it does not replace.
        for (i, ax) in s.ax.iter_mut().enumerate() {
            let row = &a[i * n..i * n + n];
            let mut acc = 0.0;
            for (j, &rj) in row.iter().enumerate() {
                acc += rj * x[j];
            }
            *ax = acc;
        }
        for ((r, &bi), &ax) in s.resid.iter_mut().zip(b).zip(s.ax.iter()) {
            *r = bi - ax;
        }
        for o in s.w.iter_mut() {
            *o = 0.0;
        }
        for (i, &vi) in s.resid.iter().enumerate() {
            let row = &a[i * n..i * n + n];
            for (wj, &rj) in s.w.iter_mut().zip(row) {
                *wj += rj * vi;
            }
        }
        // Most positive gradient over the active set; tie-break to the lowest index
        // (strict `>` keeps the first-seen maximum -> deterministic).
        let mut best: Option<usize> = None;
        let mut best_w = tol;
        for j in 0..n {
            if !s.passive[j] && s.w[j] > best_w {
                best_w = s.w[j];
                best = Some(j);
            }
        }
        let t = match best {
            Some(t) => t,
            None => break,
        };
        s.passive[t] = true;
        // Inner loop: least squares on the passive set, backtrack while any coefficient
        // is non-positive.
        loop {
            moves += 1;
            if moves > max_iter {
                return x;
            }
            s.pcols.clear();
            s.pcols.extend((0..n).filter(|&j| s.passive[j]));
            let k = s.pcols.len();
            // Gather the passive sub-Gram and right-hand side out of the precomputed
            // normal equations instead of re-accumulating them over all m rows.
            fill(&mut s.g, k * k, 0.0);
            fill(&mut s.rhs, k, 0.0);
            for r in 0..k {
                let jr = s.pcols[r];
                s.rhs[r] = s.atb[jr];
                for c in 0..k {
                    s.g[r * k + c] = s.ata[jr * n + s.pcols[c]];
                }
            }
            if !cholesky_solve_into(&s.g, k, &s.rhs, lambda, &mut s.l, &mut s.y, &mut s.zp) {
                // Ridge failed to keep the Gram PD: same fallback as before, a zero
                // passive solution, which the sign test below then rejects.
                fill(&mut s.zp, k, 0.0);
            }
            fill(&mut s.z, n, 0.0);
            for r in 0..k {
                s.z[s.pcols[r]] = s.zp[r];
            }
            if s.pcols.iter().all(|&j| s.z[j] > 0.0) {
                x.copy_from_slice(&s.z);
                break;
            }
            // Step to the nearest passive coefficient hitting zero.
            let mut alpha = f64::INFINITY;
            for &j in &s.pcols {
                if s.z[j] <= 0.0 {
                    let d = x[j] - s.z[j];
                    if d > 0.0 {
                        alpha = alpha.min(x[j] / d);
                    }
                }
            }
            if !alpha.is_finite() {
                x.copy_from_slice(&s.z);
                break;
            }
            for (xj, &zj) in x.iter_mut().zip(s.z.iter()) {
                *xj += alpha * (zj - *xj);
            }
            for (xj, pj) in x.iter_mut().zip(s.passive.iter_mut()) {
                if *pj && *xj <= 1e-12 {
                    *pj = false;
                    *xj = 0.0;
                }
            }
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_recovers_b() {
        // A = I3, b = [1,2,3] -> x = b (all positive).
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        let x = nnls(&a, 3, 3, &b, 1e-9, 100);
        for (xi, bi) in x.iter().zip(&b) {
            assert!((xi - bi).abs() < 1e-4, "{xi} vs {bi}");
        }
    }

    #[test]
    fn negative_solution_is_clamped_to_zero() {
        // A = [[1]], b = [-5] -> unconstrained x = -5, but NNLS clamps to 0.
        let x = nnls(&[1.0], 1, 1, &[-5.0], 1e-9, 100);
        assert!(x[0].abs() < 1e-9, "{}", x[0]);
    }

    #[test]
    fn two_column_apportionment() {
        // Two candidates. A col0 = [2,0,1] (its unique + a shared channel), col1 = [0,3,1].
        // b = 2*col0 + 1*col1 = [4,3,3]. NNLS should recover x ~ [2,1].
        let a = vec![2.0, 0.0, 0.0, 3.0, 1.0, 1.0];
        let b = vec![4.0, 3.0, 3.0];
        let x = nnls(&a, 3, 2, &b, 1e-9, 100);
        assert!((x[0] - 2.0).abs() < 1e-3, "x0={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-3, "x1={}", x[1]);
    }

    #[test]
    fn collinear_columns_are_deterministic_under_ridge() {
        // Two near-identical (collinear) columns; ridge shares the mass smoothly and the
        // result is bit-stable across runs.
        let a = vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0001];
        let b = vec![2.0, 4.0, 2.0];
        let x1 = nnls(&a, 3, 2, &b, 1e-3, 100);
        let x2 = nnls(&a, 3, 2, &b, 1e-3, 100);
        assert_eq!(x1, x2);
        assert!(x1[0] >= 0.0 && x1[1] >= 0.0);
    }

    #[test]
    fn zero_problem_is_zero() {
        let x = nnls(&[], 0, 0, &[], 1e-9, 10);
        assert!(x.is_empty());
    }

    /// The pre-precompute implementation, kept verbatim as the equivalence reference:
    /// per-move Gram accumulation over all rows, fresh allocations throughout.
    mod reference {
        fn cholesky_solve(g: &[f64], k: usize, rhs: &[f64], lambda: f64) -> Option<Vec<f64>> {
            let mut l = vec![0.0f64; k * k];
            for i in 0..k {
                for j in 0..=i {
                    let mut s = g[i * k + j];
                    if i == j {
                        s += lambda;
                    }
                    for p in 0..j {
                        s -= l[i * k + p] * l[j * k + p];
                    }
                    if i == j {
                        if s <= 0.0 {
                            return None;
                        }
                        l[i * k + j] = s.sqrt();
                    } else {
                        l[i * k + j] = s / l[j * k + j];
                    }
                }
            }
            let mut y = vec![0.0f64; k];
            for i in 0..k {
                let mut s = rhs[i];
                for p in 0..i {
                    s -= l[i * k + p] * y[p];
                }
                y[i] = s / l[i * k + i];
            }
            let mut z = vec![0.0f64; k];
            for ii in (0..k).rev() {
                let mut s = y[ii];
                for p in (ii + 1)..k {
                    s -= l[p * k + ii] * z[p];
                }
                z[ii] = s / l[ii * k + ii];
            }
            Some(z)
        }

        fn solve_passive(
            a: &[f64],
            m: usize,
            n: usize,
            b: &[f64],
            pcols: &[usize],
            lambda: f64,
        ) -> Vec<f64> {
            let k = pcols.len();
            if k == 0 {
                return Vec::new();
            }
            let mut g = vec![0.0f64; k * k];
            let mut rhs = vec![0.0f64; k];
            for i in 0..m {
                let row = &a[i * n..i * n + n];
                let bi = b[i];
                for (r, &jr) in pcols.iter().enumerate() {
                    let air = row[jr];
                    rhs[r] += air * bi;
                    for (c, &jc) in pcols.iter().enumerate() {
                        g[r * k + c] += air * row[jc];
                    }
                }
            }
            cholesky_solve(&g, k, &rhs, lambda).unwrap_or_else(|| vec![0.0; k])
        }

        pub fn nnls(
            a: &[f64],
            m: usize,
            n: usize,
            b: &[f64],
            lambda: f64,
            max_iter: usize,
        ) -> Vec<f64> {
            let mut x = vec![0.0f64; n];
            if n == 0 || m == 0 {
                return x;
            }
            let mut passive = vec![false; n];
            let tol = 1e-9;
            let a_times = |v: &[f64], out: &mut [f64]| {
                for i in 0..m {
                    let row = &a[i * n..i * n + n];
                    let mut s = 0.0;
                    for j in 0..n {
                        s += row[j] * v[j];
                    }
                    out[i] = s;
                }
            };
            let at_times = |v: &[f64], out: &mut [f64]| {
                for o in out.iter_mut() {
                    *o = 0.0;
                }
                for i in 0..m {
                    let row = &a[i * n..i * n + n];
                    let vi = v[i];
                    for j in 0..n {
                        out[j] += row[j] * vi;
                    }
                }
            };
            let mut ax = vec![0.0f64; m];
            let mut resid = vec![0.0f64; m];
            let mut w = vec![0.0f64; n];
            let mut moves = 0usize;
            loop {
                a_times(&x, &mut ax);
                for i in 0..m {
                    resid[i] = b[i] - ax[i];
                }
                at_times(&resid, &mut w);
                let mut best: Option<usize> = None;
                let mut best_w = tol;
                for j in 0..n {
                    if !passive[j] && w[j] > best_w {
                        best_w = w[j];
                        best = Some(j);
                    }
                }
                let t = match best {
                    Some(t) => t,
                    None => break,
                };
                passive[t] = true;
                loop {
                    moves += 1;
                    if moves > max_iter {
                        return x;
                    }
                    let pcols: Vec<usize> = (0..n).filter(|&j| passive[j]).collect();
                    let zp = solve_passive(a, m, n, b, &pcols, lambda);
                    let mut z = vec![0.0f64; n];
                    for (k, &j) in pcols.iter().enumerate() {
                        z[j] = zp[k];
                    }
                    if pcols.iter().all(|&j| z[j] > 0.0) {
                        x.copy_from_slice(&z);
                        break;
                    }
                    let mut alpha = f64::INFINITY;
                    for &j in &pcols {
                        if z[j] <= 0.0 {
                            let d = x[j] - z[j];
                            if d > 0.0 {
                                alpha = alpha.min(x[j] / d);
                            }
                        }
                    }
                    if !alpha.is_finite() {
                        x.copy_from_slice(&z);
                        break;
                    }
                    for j in 0..n {
                        x[j] += alpha * (z[j] - x[j]);
                    }
                    for j in 0..n {
                        if passive[j] && x[j] <= 1e-12 {
                            passive[j] = false;
                            x[j] = 0.0;
                        }
                    }
                }
            }
            x
        }
    }

    /// Deterministic xorshift64*, so the sweep below is reproducible without a dep.
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    #[test]
    fn precomputed_normal_equations_are_bit_identical_to_per_move_rebuild() {
        // The precompute must not perturb a single bit: `nnls` feeds fragment
        // apportionment, so any drift here changes extracted intensities and therefore
        // identifications. Sweep shapes, sparsity and collinearity, including the
        // degenerate cases (all-zero column, duplicated columns, negative b).
        let mut rng = Rng(0x5EED_1234_ABCD_0001);
        let mut scratch = NnlsScratch::new();
        let mut cases = 0;
        for &(m, n) in &[
            (1, 1),
            (3, 2),
            (5, 5),
            (8, 3),
            (3, 8),
            (12, 7),
            (20, 16),
            (16, 20),
            (40, 12),
        ] {
            for trial in 0..12 {
                let sparsity = 0.15 * (trial % 5) as f64;
                let mut a = vec![0.0f64; m * n];
                for v in a.iter_mut() {
                    *v = if rng.next_f64() < sparsity {
                        0.0
                    } else {
                        rng.next_f64() * 10.0
                    };
                }
                // Every third trial duplicates a column to force collinearity, and every
                // fifth zeroes one entirely.
                if trial % 3 == 0 && n >= 2 {
                    for i in 0..m {
                        a[i * n + 1] = a[i * n];
                    }
                }
                if trial % 5 == 4 {
                    for i in 0..m {
                        a[i * n + (n - 1)] = 0.0;
                    }
                }
                let mut b = vec![0.0f64; m];
                for (i, v) in b.iter_mut().enumerate() {
                    *v = rng.next_f64() * 10.0 - if i % 4 == 0 { 5.0 } else { 0.0 };
                }
                for &lambda in &[1e-9, 1e-3, 1.0] {
                    let want = reference::nnls(&a, m, n, &b, lambda, 200 * n);
                    let got = nnls_with(&mut scratch, &a, m, n, &b, lambda, 200 * n);
                    assert_eq!(
                        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "diverged at m={m} n={n} trial={trial} lambda={lambda}\nwant={want:?}\ngot={got:?}"
                    );
                    cases += 1;
                }
            }
        }
        assert!(cases >= 300, "sweep too small: {cases}");
    }
}
