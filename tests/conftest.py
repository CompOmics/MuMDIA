"""
Test configuration and fixtures for MuMDIA test suite.

This module provides common test data, fixtures, and utilities used across
all test modules in the MuMDIA project.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import polars as pl
import pytest

from data_structures import CorrelationResults, PickleConfig, SpectraData


# Test data fixtures
@pytest.fixture
def sample_psm_data():
    """Create sample PSM data for testing."""
    return pl.DataFrame(
        {
            "psm_id": [1, 2, 3, 4],
            "peptide": ["PEPTIDE", "SEQUENCE", "TESTING", "EXAMPLE"],
            "charge": [2, 3, 2, 2],
            "rt": [10.5, 15.2, 20.1, 25.8],
            "fragment_intensity": [1000.0, 1500.0, 800.0, 1200.0],
            "scannr": ["scan_1", "scan_2", "scan_3", "scan_4"],
            "calcmass": [1000.5, 1200.3, 900.2, 1100.1],
            "is_decoy": [False, False, True, False],
        }
    )


@pytest.fixture
def sample_fragment_data():
    """Create sample fragment data for testing."""
    return pl.DataFrame(
        {
            "psm_id": [1, 1, 2, 2, 3, 3],
            "fragment_type": ["b", "y", "b", "y", "b", "y"],
            "fragment_ordinals": [1, 1, 2, 2, 1, 1],
            "fragment_charge": [1, 1, 1, 1, 1, 1],
            "fragment_intensity": [500.0, 800.0, 600.0, 900.0, 400.0, 700.0],
            "fragment_name": ["b1/1", "y1/1", "b2/1", "y2/1", "b1/1", "y1/1"],
            "peptide": [
                "PEPTIDE",
                "PEPTIDE",
                "SEQUENCE",
                "SEQUENCE",
                "TESTING",
                "TESTING",
            ],
            "charge": [2, 2, 3, 3, 2, 2],
            "rt": [10.5, 10.5, 15.2, 15.2, 20.1, 20.1],
            "scannr": ["scan_1", "scan_1", "scan_2", "scan_2", "scan_3", "scan_3"],
            "rt_max_peptide_sub": [10.5, 10.5, 15.2, 15.2, 20.1, 20.1],
        }
    )


@pytest.fixture
def sample_ms2pip_predictions():
    """Create sample MS2PIP predictions for testing."""
    return {
        "b1/1": 0.1,
        "b2/1": 0.2,
        "y1/1": 0.15,
        "y2/1": 0.25,
        "b3/1": 0.08,
        "y3/1": 0.12,
    }


@pytest.fixture
def sample_ms2_dict():
    """Create sample MS2 spectrum data for testing."""
    return {
        "scan_1": {
            "mz": np.array([100.0, 200.0, 300.0, 400.0]),
            "intensity": np.array([1000.0, 1500.0, 800.0, 1200.0]),
        },
        "scan_2": {
            "mz": np.array([150.0, 250.0, 350.0, 450.0]),
            "intensity": np.array([900.0, 1300.0, 700.0, 1100.0]),
        },
        "scan_3": {
            "mz": np.array([120.0, 220.0, 320.0, 420.0]),
            "intensity": np.array([800.0, 1200.0, 600.0, 1000.0]),
        },
    }


@pytest.fixture
def sample_correlation_results():
    """Create sample correlation results for testing."""
    return CorrelationResults(
        correlations=np.array([0.8, 0.6, 0.7]),
        correlations_count=np.array([10, 8, 9]),
        sum_pred_frag_intens=np.array([0.5]),
        correlation_matrix_psm_ids=np.array([0.9, 0.7, 0.8]),
        correlation_matrix_frag_ids=np.array([0.85, 0.65, 0.75]),
        correlation_matrix_psm_ids_ignore_zeros=np.array([]),
        correlation_matrix_psm_ids_ignore_zeros_counts=np.array([]),
        correlation_matrix_psm_ids_missing=np.array([]),
        correlation_matrix_psm_ids_missing_zeros_counts=np.array([]),
        correlation_matrix_frag_ids_ignore_zeros=np.array([]),
        correlation_matrix_frag_ids_ignore_zeros_counts=np.array([]),
        correlation_matrix_frag_ids_missing=np.array([]),
        correlation_matrix_frag_ids_missing_zeros_counts=np.array([]),
        most_intens_cor=0.85,
        most_intens_cos=0.90,
        mse_avg_pred_intens=0.1,
        mse_avg_pred_intens_total=0.12,
    )


@pytest.fixture
def sample_pickle_config():
    """Create sample pickle configuration for testing."""
    return PickleConfig(
        write_deeplc=False,
        write_ms2pip=False,
        write_correlation=False,
        read_deeplc=False,
        read_ms2pip=False,
        read_correlation=False,
    )


@pytest.fixture
def sample_spectra_data():
    """Create sample spectra data for testing."""
    return SpectraData(
        ms1_dict={
            "scan_1": {
                "mz": np.array([500.0, 600.0]),
                "intensity": np.array([2000.0, 1800.0]),
            }
        },
        ms2_to_ms1_dict={"scan_1": "scan_1", "scan_2": "scan_1"},
        ms2_dict={
            "scan_1": {
                "mz": np.array([100.0, 200.0]),
                "intensity": np.array([1000.0, 1500.0]),
            },
            "scan_2": {
                "mz": np.array([150.0, 250.0]),
                "intensity": np.array([900.0, 1300.0]),
            },
        },
    )


# Test utilities
def assert_correlation_results_equal(
    result1: CorrelationResults, result2: CorrelationResults, rtol=1e-5
):
    """Assert that two CorrelationResults are approximately equal."""
    np.testing.assert_allclose(result1.correlations, result2.correlations, rtol=rtol)
    np.testing.assert_array_equal(
        result1.correlations_count, result2.correlations_count
    )
    np.testing.assert_allclose(
        result1.sum_pred_frag_intens, result2.sum_pred_frag_intens, rtol=rtol
    )
    assert abs(result1.most_intens_cor - result2.most_intens_cor) < rtol
    assert abs(result1.most_intens_cos - result2.most_intens_cos) < rtol


def create_test_intensity_matrix(n_psms: int = 3, n_fragments: int = 5) -> np.ndarray:
    """Create a test intensity matrix with known properties."""
    np.random.seed(42)  # For reproducible tests
    return np.random.rand(n_psms, n_fragments) * 1000


def create_test_predictions(fragment_names: list) -> np.ndarray:
    """Create test predictions that correlate with specific patterns."""
    np.random.seed(42)
    return np.random.rand(len(fragment_names))
