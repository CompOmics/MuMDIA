/// Top-k value selection with zero-padding, matching MuMDIA's Numba compute_top_nb.

/// Sort descending and return the first m values. Pad with zeros if fewer than m elements.
pub fn compute_top_impl(data: &[f64], m: usize) -> Vec<f64> {
    let n = data.len();
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)); // descending

    let mut result = vec![0.0; m];
    for i in 0..m {
        if i < n {
            result[i] = sorted[i];
        }
        // else already 0.0 from initialization
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_top_basic() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let result = compute_top_impl(&data, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 9.0).abs() < 1e-12);
        assert!((result[1] - 6.0).abs() < 1e-12);
        assert!((result[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_top_padding() {
        let data = vec![1.0, 2.0];
        let result = compute_top_impl(&data, 5);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 2.0).abs() < 1e-12);
        assert!((result[1] - 1.0).abs() < 1e-12);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 0.0);
        assert_eq!(result[4], 0.0);
    }

    #[test]
    fn test_compute_top_empty() {
        let data: Vec<f64> = vec![];
        let result = compute_top_impl(&data, 3);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }
}
