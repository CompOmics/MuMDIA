//! Shared numeric kernels (PLAN.md Section 7 "Numerics"): population-variance
//! Pearson with a zero-variance guard returning 0, cosine similarity, spectral
//! angle. One implementation, unit-tested (PLAN.md Section 5 improvement 7).

/// Population Pearson correlation; returns 0 on zero variance (documented guard).
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let na = n as f64;
    let ma = a[..n].iter().sum::<f64>() / na;
    let mb = b[..n].iter().sum::<f64>() / na;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 0.0 || vb <= 0.0 {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// Cosine similarity; returns 0 if either vector is all-zero.
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Spectral angle in [0,1] (1 = identical), derived from cosine.
pub fn spectral_angle(a: &[f64], b: &[f64]) -> f64 {
    let c = cosine(a, b).clamp(-1.0, 1.0);
    1.0 - 2.0 * c.acos() / std::f64::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_perfect_and_flat() {
        assert!((pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]) - 1.0).abs() < 1e-9);
        assert_eq!(pearson(&[1.0, 1.0, 1.0], &[2.0, 4.0, 6.0]), 0.0);
    }

    #[test]
    fn cosine_and_angle() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!((spectral_angle(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }
}
