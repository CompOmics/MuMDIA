"""Validate Rust (mumdia_rs) functions match Python reference outputs."""

import json
import os

import numpy as np
import pytest

try:
    import mumdia_rs

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "rust_reference_data.json")


@pytest.fixture(scope="module")
def reference_data():
    with open(REFERENCE_PATH) as f:
        return json.load(f)


def cases_for(reference_data, fn_name):
    return [c for c in reference_data if c["fn"] == fn_name]


@pytest.mark.skipif(not RUST_AVAILABLE, reason="mumdia_rs not installed")
class TestRustPercentile:
    def test_percentile(self, reference_data):
        for case in cases_for(reference_data, "percentile"):
            data = np.array(case["data"], dtype=np.float64)
            result = mumdia_rs.percentile(data, case["q"])
            assert (
                abs(result - case["expected"]) < 1e-12
            ), f"percentile [{case['label']}]: got {result}, expected {case['expected']}"

    def test_compute_percentiles(self, reference_data):
        for case in cases_for(reference_data, "compute_percentiles"):
            data = np.array(case["data"], dtype=np.float64)
            qs = np.array(case["qs"], dtype=np.float64)
            result = mumdia_rs.compute_percentiles(data, qs)
            np.testing.assert_allclose(
                result,
                case["expected"],
                atol=1e-12,
                err_msg=f"compute_percentiles [{case['label']}]",
            )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="mumdia_rs not installed")
class TestRustTopK:
    def test_compute_top(self, reference_data):
        for case in cases_for(reference_data, "compute_top"):
            data = np.array(case["data"], dtype=np.float64)
            result = mumdia_rs.compute_top(data, case["m"])
            np.testing.assert_allclose(
                result,
                case["expected"],
                atol=1e-12,
                err_msg=f"compute_top [{case['label']}]",
            )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="mumdia_rs not installed")
class TestRustCorrelation:
    def test_pearson_1d(self, reference_data):
        for case in cases_for(reference_data, "pearson_1d"):
            a = np.array(case["a"], dtype=np.float64)
            b = np.array(case["b"], dtype=np.float64)
            result = mumdia_rs.pearson_1d(a, b)
            assert (
                abs(result - case["expected"]) < 1e-10
            ), f"pearson_1d [{case['label']}]: got {result}, expected {case['expected']}"

    def test_compute_correlations(self, reference_data):
        for case in cases_for(reference_data, "compute_correlations"):
            matrix = np.array(case["matrix"], dtype=np.float64)
            preds = np.array(case["preds"], dtype=np.float64)
            result = mumdia_rs.compute_correlations(matrix, preds)
            np.testing.assert_allclose(
                result,
                case["expected"],
                atol=1e-10,
                err_msg=f"compute_correlations [{case['label']}]",
            )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="mumdia_rs not installed")
class TestRustProperties:
    """Property-based tests that don't need reference data."""

    def test_percentile_monotonic(self):
        data = np.random.randn(100)
        qs = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.float64)
        result = mumdia_rs.compute_percentiles(data, qs)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1], f"Not monotonic at index {i}"

    def test_pearson_range(self):
        for _ in range(20):
            a = np.random.randn(50)
            b = np.random.randn(50)
            r = mumdia_rs.pearson_1d(a, b)
            assert -1.0 - 1e-10 <= r <= 1.0 + 1e-10, f"Pearson r={r} out of range"

    def test_compute_top_descending(self):
        data = np.random.randn(100)
        result = mumdia_rs.compute_top(data, 10)
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1], f"Not descending at index {i}"

    def test_compute_correlations_range(self):
        matrix = np.random.rand(10, 5)
        preds = np.random.rand(5)
        result = mumdia_rs.compute_correlations(matrix, preds)
        for i, r in enumerate(result):
            assert (
                -1.0 - 1e-10 <= r <= 1.0 + 1e-10
            ), f"Row {i}: correlation {r} out of range"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="mumdia_rs not installed")
class TestRustPrefilter:
    def test_prefilter_window_candidates(self):
        spectrum_idx, candidate_idx, matched_counts = (
            mumdia_rs.prefilter_window_candidates(
                np.array([500.25, 503.25], dtype=np.float64),
                np.array(
                    [
                        204.13,
                        317.22,
                        430.31,
                        533.40,
                        646.48,
                        759.57,
                        150.0,
                        250.0,
                        350.0,
                    ],
                    dtype=np.float64,
                ),
                np.array([0, 6], dtype=np.uint64),
                np.array([6, 3], dtype=np.uint64),
                np.array([499.0], dtype=np.float64),
                np.array([501.0], dtype=np.float64),
                np.array(
                    [204.1305, 317.2195, 430.3108, 646.4820, 1000.0], dtype=np.float64
                ),
                np.array([0], dtype=np.uint64),
                np.array([5], dtype=np.uint64),
                20.0,
                3,
                0.08,
            )
        )

        assert spectrum_idx.tolist() == [0]
        assert candidate_idx.tolist() == [0]
        assert matched_counts.tolist() == [4]
