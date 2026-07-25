//! Deterministic pure-Rust non-negative least squares for spectrum-centric fragment
//! demixing (fragment_competition_strategies.md, family I3 / D2-D3).
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

/// Solve `(G + lambda I) z = rhs` for a symmetric positive-definite `G` (row-major
/// `k x k`) via Cholesky `G = L L^T` plus forward/back substitution. Returns `None`
/// only if a non-positive pivot survives the ridge (should not happen with lambda > 0).
fn cholesky_solve(g: &[f64], k: usize, rhs: &[f64], lambda: f64) -> Option<Vec<f64>> {
    // L is lower-triangular, row-major k x k.
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
    // Forward solve L y = rhs.
    let mut y = vec![0.0f64; k];
    for i in 0..k {
        let mut s = rhs[i];
        for p in 0..i {
            s -= l[i * k + p] * y[p];
        }
        y[i] = s / l[i * k + i];
    }
    // Back solve L^T z = y.
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

/// Least squares of `A` restricted to the passive column set `pcols`, ridge-regularized.
/// Returns the coefficients aligned to `pcols` (one per passive column).
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

/// Non-negative least squares (Lawson-Hanson active set), ridge-regularized on the
/// passive solve. `a` is row-major `m x n`, `b` has length `m`. Returns `x` (length `n`)
/// with `x_j >= 0`. `lambda` is the ridge added to the passive Gram diagonal (> 0 keeps
/// it PD and deterministic under collinearity). `max_iter` caps the active-set moves.
pub fn nnls(a: &[f64], m: usize, n: usize, b: &[f64], lambda: f64, max_iter: usize) -> Vec<f64> {
    let mut x = vec![0.0f64; n];
    if n == 0 || m == 0 {
        return x;
    }
    let mut passive = vec![false; n];
    // Positive-gradient tolerance: below this a column is not worth activating.
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
        // Most positive gradient over the active set; tie-break to the lowest index
        // (strict `>` keeps the first-seen maximum -> deterministic).
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
        // Inner loop: least squares on the passive set, backtrack while any coefficient
        // is non-positive.
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
            // Step to the nearest passive coefficient hitting zero.
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
}
