"""Generate reference input/output pairs from Python implementations for Rust validation."""

import json
import os
import sys

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_generators.features_fragment_intensity import compute_correlations
from mumdia import (
    compute_percentiles_nb,
    compute_top_nb,
    corr_np_nb_new,
    numba_percentile,
)


def generate():
    cases = []
    np.random.seed(42)

    # --- percentile ---
    for label, data in [
        ("empty", np.array([], dtype=np.float64)),
        ("single", np.array([42.0])),
        ("constant", np.array([3.0, 3.0, 3.0, 3.0])),
        ("small", np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
        ("random_100", np.random.randn(100)),
    ]:
        for q in [0.0, 25.0, 50.0, 75.0, 100.0]:
            result = float(numba_percentile(data, q))
            cases.append(
                {
                    "fn": "percentile",
                    "label": f"{label}_q{q}",
                    "data": data.tolist(),
                    "q": q,
                    "expected": result,
                }
            )

    # --- compute_percentiles ---
    qs = np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype=np.float64)
    for label, data in [
        ("empty", np.array([], dtype=np.float64)),
        ("single", np.array([42.0])),
        ("five", np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
        ("random_100", np.random.randn(100)),
    ]:
        result = compute_percentiles_nb(data, qs)
        cases.append(
            {
                "fn": "compute_percentiles",
                "label": label,
                "data": data.tolist(),
                "qs": qs.tolist(),
                "expected": result.tolist(),
            }
        )

    # --- compute_top ---
    for label, data, m in [
        ("basic", np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0]), 3),
        ("padded", np.array([1.0, 2.0]), 5),
        ("empty", np.array([], dtype=np.float64), 3),
        ("exact", np.array([5.0, 3.0, 1.0]), 3),
        ("random", np.random.randn(50), 10),
    ]:
        result = compute_top_nb(data, m)
        cases.append(
            {
                "fn": "compute_top",
                "label": label,
                "data": data.tolist(),
                "m": m,
                "expected": result.tolist(),
            }
        )

    # --- pearson_1d ---
    for label, a, b in [
        (
            "perfect",
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        ),
        (
            "anti",
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        ),
        ("zero_var_a", np.array([3.0, 3.0, 3.0]), np.array([1.0, 2.0, 3.0])),
        ("zero_var_b", np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0])),
        ("random", np.random.randn(50), np.random.randn(50)),
    ]:
        result = float(corr_np_nb_new(a, b))
        cases.append(
            {
                "fn": "pearson_1d",
                "label": label,
                "a": a.tolist(),
                "b": b.tolist(),
                "expected": result,
            }
        )

    # --- compute_correlations (2D matrix vs 1D predictions) ---
    for label, n_psms, n_frags in [
        ("tiny", 1, 3),
        ("small", 5, 10),
        ("medium", 20, 15),
        ("typical", 50, 30),
    ]:
        matrix = np.random.rand(n_psms, n_frags).astype(np.float64)
        preds = np.random.rand(n_frags).astype(np.float64)
        result = compute_correlations(matrix, preds)
        cases.append(
            {
                "fn": "compute_correlations",
                "label": label,
                "matrix": matrix.tolist(),
                "preds": preds.tolist(),
                "n_psms": n_psms,
                "n_frags": n_frags,
                "expected": result.tolist(),
            }
        )

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rust_reference_data.json"
    )
    with open(output_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Generated {len(cases)} reference test cases -> {output_path}")


if __name__ == "__main__":
    generate()
