"""
Test suite for data structures module.

This module tests the dataclasses and their functionality used throughout
the MuMDIA pipeline.
"""

from dataclasses import fields

import numpy as np
import pytest

from data_structures import CorrelationResults, PickleConfig, SpectraData


class TestCorrelationResults:
    """Test the CorrelationResults dataclass."""

    def test_correlation_results_creation(self):
        """Test basic creation of CorrelationResults."""
        result = CorrelationResults(
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

        assert isinstance(result, CorrelationResults)
        assert len(result.correlations) == 3
        assert result.most_intens_cor == 0.85
        assert result.most_intens_cos == 0.90

    def test_correlation_results_fields(self):
        """Test that all expected fields are present."""
        expected_fields = {
            "correlations",
            "correlations_count",
            "sum_pred_frag_intens",
            "correlation_matrix_psm_ids",
            "correlation_matrix_frag_ids",
            "correlation_matrix_psm_ids_ignore_zeros",
            "correlation_matrix_psm_ids_ignore_zeros_counts",
            "correlation_matrix_psm_ids_missing",
            "correlation_matrix_psm_ids_missing_zeros_counts",
            "correlation_matrix_frag_ids_ignore_zeros",
            "correlation_matrix_frag_ids_ignore_zeros_counts",
            "correlation_matrix_frag_ids_missing",
            "correlation_matrix_frag_ids_missing_zeros_counts",
            "most_intens_cor",
            "most_intens_cos",
            "mse_avg_pred_intens",
            "mse_avg_pred_intens_total",
        }

        actual_fields = {field.name for field in fields(CorrelationResults)}
        assert actual_fields == expected_fields

    def test_correlation_results_numpy_arrays(self):
        """Test that numpy arrays are handled correctly."""
        correlations = np.array([0.1, 0.2, 0.3])
        counts = np.array([5, 6, 7])

        result = CorrelationResults(
            correlations=correlations,
            correlations_count=counts,
            sum_pred_frag_intens=np.array([0.5]),
            correlation_matrix_psm_ids=np.array([]),
            correlation_matrix_frag_ids=np.array([]),
            correlation_matrix_psm_ids_ignore_zeros=np.array([]),
            correlation_matrix_psm_ids_ignore_zeros_counts=np.array([]),
            correlation_matrix_psm_ids_missing=np.array([]),
            correlation_matrix_psm_ids_missing_zeros_counts=np.array([]),
            correlation_matrix_frag_ids_ignore_zeros=np.array([]),
            correlation_matrix_frag_ids_ignore_zeros_counts=np.array([]),
            correlation_matrix_frag_ids_missing=np.array([]),
            correlation_matrix_frag_ids_missing_zeros_counts=np.array([]),
            most_intens_cor=0.0,
            most_intens_cos=0.0,
            mse_avg_pred_intens=0.0,
            mse_avg_pred_intens_total=0.0,
        )

        np.testing.assert_array_equal(result.correlations, correlations)
        np.testing.assert_array_equal(result.correlations_count, counts)


class TestPickleConfig:
    """Test the PickleConfig dataclass."""

    def test_pickle_config_defaults(self):
        """Test default values of PickleConfig."""
        config = PickleConfig()

        assert config.write_deeplc is False
        assert config.write_ms2pip is False
        assert config.write_correlation is False
        assert config.read_deeplc is False
        assert config.read_ms2pip is False
        assert config.read_correlation is False

    def test_pickle_config_custom_values(self):
        """Test custom values for PickleConfig."""
        config = PickleConfig(
            write_deeplc=True, read_ms2pip=True, write_correlation=True
        )

        assert config.write_deeplc is True
        assert config.read_ms2pip is True
        assert config.write_correlation is True
        assert config.read_deeplc is False  # Still default

    def test_pickle_config_all_true(self):
        """Test all flags set to True."""
        config = PickleConfig(
            write_deeplc=True,
            write_ms2pip=True,
            write_correlation=True,
            read_deeplc=True,
            read_ms2pip=True,
            read_correlation=True,
        )

        assert all(
            [
                config.write_deeplc,
                config.write_ms2pip,
                config.write_correlation,
                config.read_deeplc,
                config.read_ms2pip,
                config.read_correlation,
            ]
        )

    def test_pickle_config_fields(self):
        """Test that all expected fields are present."""
        expected_fields = {
            "write_deeplc",
            "write_ms2pip",
            "write_correlation",
            "read_deeplc",
            "read_ms2pip",
            "read_correlation",
        }

        actual_fields = {field.name for field in fields(PickleConfig)}
        assert actual_fields == expected_fields


class TestSpectraData:
    """Test the SpectraData dataclass."""

    def test_spectra_data_defaults(self):
        """Test default values of SpectraData."""
        spectra = SpectraData()

        assert isinstance(spectra.ms1_dict, dict)
        assert isinstance(spectra.ms2_to_ms1_dict, dict)
        assert isinstance(spectra.ms2_dict, dict)
        assert len(spectra.ms1_dict) == 0
        assert len(spectra.ms2_to_ms1_dict) == 0
        assert len(spectra.ms2_dict) == 0

    def test_spectra_data_with_data(self):
        """Test SpectraData with actual data."""
        ms1_data = {
            "scan_1": {"mz": np.array([100.0]), "intensity": np.array([1000.0])}
        }
        ms2_data = {
            "scan_2": {"mz": np.array([200.0]), "intensity": np.array([2000.0])}
        }
        mapping = {"scan_2": "scan_1"}

        spectra = SpectraData(
            ms1_dict=ms1_data, ms2_dict=ms2_data, ms2_to_ms1_dict=mapping
        )

        assert spectra.ms1_dict == ms1_data
        assert spectra.ms2_dict == ms2_data
        assert spectra.ms2_to_ms1_dict == mapping

    def test_spectra_data_mutable_defaults(self):
        """Test that default dicts are independent instances."""
        spectra1 = SpectraData()
        spectra2 = SpectraData()

        spectra1.ms1_dict["test"] = "value"

        # spectra2 should not be affected
        assert "test" not in spectra2.ms1_dict

    def test_spectra_data_fields(self):
        """Test that all expected fields are present."""
        expected_fields = {"ms1_dict", "ms2_to_ms1_dict", "ms2_dict"}

        actual_fields = {field.name for field in fields(SpectraData)}
        assert actual_fields == expected_fields


class TestDataStructureIntegration:
    """Test integration between different data structures."""

    def test_pickle_config_with_correlation_results(
        self, sample_correlation_results, sample_pickle_config
    ):
        """Test using PickleConfig with correlation operations."""
        # This would typically be used in a function that processes correlations
        config = sample_pickle_config
        results = sample_correlation_results

        # Test that both objects work together
        assert isinstance(config, PickleConfig)
        assert isinstance(results, CorrelationResults)

        # Example usage pattern
        if config.write_correlation:
            # Would write results to pickle
            pass

        if config.read_correlation:
            # Would read results from pickle
            pass

    def test_spectra_data_with_correlation_workflow(self, sample_spectra_data):
        """Test SpectraData in a typical correlation workflow."""
        spectra = sample_spectra_data

        # Verify data structure supports typical operations
        assert "scan_1" in spectra.ms2_dict
        assert "scan_1" in spectra.ms2_to_ms1_dict

        # Test accessing spectrum data
        ms2_spectrum = spectra.ms2_dict["scan_1"]
        assert "mz" in ms2_spectrum
        assert "intensity" in ms2_spectrum

        # Test MS2 to MS1 mapping
        ms1_scan = spectra.ms2_to_ms1_dict["scan_1"]
        assert ms1_scan in spectra.ms1_dict


class TestDataStructureValidation:
    """Test validation and error handling for data structures."""

    def test_correlation_results_with_nan_values(self):
        """Test CorrelationResults handles NaN values appropriately."""
        result = CorrelationResults(
            correlations=np.array([0.8, np.nan, 0.7]),
            correlations_count=np.array([10, 0, 9]),
            sum_pred_frag_intens=np.array([0.5]),
            correlation_matrix_psm_ids=np.array([]),
            correlation_matrix_frag_ids=np.array([]),
            correlation_matrix_psm_ids_ignore_zeros=np.array([]),
            correlation_matrix_psm_ids_ignore_zeros_counts=np.array([]),
            correlation_matrix_psm_ids_missing=np.array([]),
            correlation_matrix_psm_ids_missing_zeros_counts=np.array([]),
            correlation_matrix_frag_ids_ignore_zeros=np.array([]),
            correlation_matrix_frag_ids_ignore_zeros_counts=np.array([]),
            correlation_matrix_frag_ids_missing=np.array([]),
            correlation_matrix_frag_ids_missing_zeros_counts=np.array([]),
            most_intens_cor=np.nan,
            most_intens_cos=0.90,
            mse_avg_pred_intens=0.1,
            mse_avg_pred_intens_total=0.12,
        )

        assert np.isnan(result.correlations[1])
        assert np.isnan(result.most_intens_cor)
        assert not np.isnan(result.most_intens_cos)

    def test_spectra_data_empty_arrays(self):
        """Test SpectraData with empty numpy arrays."""
        spectra = SpectraData(
            ms1_dict={"scan_1": {"mz": np.array([]), "intensity": np.array([])}},
            ms2_dict={"scan_2": {"mz": np.array([]), "intensity": np.array([])}},
            ms2_to_ms1_dict={},
        )

        assert len(spectra.ms1_dict["scan_1"]["mz"]) == 0
        assert len(spectra.ms2_dict["scan_2"]["intensity"]) == 0
