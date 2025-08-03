"""
Test suite for retention time feature generation.

This module tests the retention time prediction features,
including DeepLC integration, error calculations, and filtering.
"""

import pytest
import polars as pl
import numpy as np
from feature_generators.features_retention_time import add_retention_time_features


class TestAddRetentionTimeFeatures:
    """Test suite for retention time feature functionality."""

    @pytest.fixture
    def sample_psms(self):
        """Sample PSM data for testing."""
        return pl.DataFrame(
            {
                "peptide": ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3", "PEPTIDE4"],
                "rt": [10.5, 15.2, 20.8, 25.3],
                "charge": [2, 3, 2, 4],
                "score": [0.9, 0.8, 0.85, 0.75],
            }
        )

    @pytest.fixture
    def sample_deeplc_predictions(self):
        """Sample DeepLC predictions for testing."""
        return pl.DataFrame(
            {
                "peptide": ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3", "PEPTIDE5"],
                "rt_predictions": [10.8, 14.9, 22.1, 30.5],
            }
        )

    @pytest.mark.unit
    def test_add_retention_time_features_basic(
        self, sample_psms, sample_deeplc_predictions
    ):
        """Test basic retention time feature addition."""
        result = add_retention_time_features(
            sample_psms,
            sample_deeplc_predictions,
            filter_rel_rt_error=0.5,  # Lenient filtering for this test
        )

        # Check that predictions are joined
        assert "rt_predictions" in result.columns
        assert "rt_prediction_error_abs" in result.columns
        assert "rt_prediction_error_abs_relative" in result.columns

        # PEPTIDE4 should be filtered out (no prediction), others should remain
        assert result.height == 3

        # Check specific values for PEPTIDE1
        peptide1_row = result.filter(pl.col("peptide") == "PEPTIDE1")
        assert peptide1_row.height == 1
        assert peptide1_row["rt_predictions"][0] == 10.8

        # Check absolute error calculation
        expected_abs_error = abs(10.5 - 10.8)  # 0.3
        assert (
            abs(peptide1_row["rt_prediction_error_abs"][0] - expected_abs_error) < 1e-6
        )

    @pytest.mark.unit
    def test_add_retention_time_features_relative_error_calculation(
        self, sample_psms, sample_deeplc_predictions
    ):
        """Test relative error calculation accuracy."""
        result = add_retention_time_features(
            sample_psms,
            sample_deeplc_predictions,
            filter_rel_rt_error=1.0,  # No filtering
        )

        max_rt = sample_psms["rt"].max()  # 25.3

        # Check relative error for PEPTIDE2
        peptide2_row = result.filter(pl.col("peptide") == "PEPTIDE2")
        assert peptide2_row.height == 1

        abs_error = abs(15.2 - 14.9)  # 0.3
        expected_rel_error = abs_error / max_rt  # 0.3 / 25.3

        actual_rel_error = peptide2_row["rt_prediction_error_abs_relative"][0]
        assert abs(actual_rel_error - expected_rel_error) < 1e-6

    @pytest.mark.unit
    def test_add_retention_time_features_filtering_strict(
        self, sample_psms, sample_deeplc_predictions
    ):
        """Test strict filtering based on relative RT error."""
        result = add_retention_time_features(
            sample_psms,
            sample_deeplc_predictions,
            filter_rel_rt_error=0.01,  # Very strict filtering
        )

        # Most peptides should be filtered out due to strict threshold
        # Only peptides with very good RT predictions should remain
        max_rt = sample_psms["rt"].max()

        for row in result.iter_rows(named=True):
            rel_error = row["rt_prediction_error_abs_relative"]
            assert rel_error < 0.01

    @pytest.mark.unit
    def test_add_retention_time_features_no_predictions(self):
        """Test behavior when no DeepLC predictions are available."""
        psms = pl.DataFrame({"peptide": ["PEPTIDE1", "PEPTIDE2"], "rt": [10.5, 15.2]})

        predictions = pl.DataFrame(
            {
                "peptide": ["PEPTIDE3", "PEPTIDE4"],  # No matching peptides
                "rt_predictions": [20.0, 25.0],
            }
        )

        result = add_retention_time_features(psms, predictions, filter_rel_rt_error=0.2)

        # Should result in empty DataFrame after filtering
        assert result.height == 0
        assert "rt_predictions" in result.columns

    @pytest.mark.unit
    def test_add_retention_time_features_disable_absolute_error(
        self, sample_psms, sample_deeplc_predictions
    ):
        """Test disabling absolute error calculation."""
        result = add_retention_time_features(
            sample_psms,
            sample_deeplc_predictions,
            rt_prediction_error_abs=False,
            filter_rel_rt_error=1.0,
        )

        assert "rt_prediction_error_abs" not in result.columns
        assert "rt_prediction_error_abs_relative" in result.columns

    @pytest.mark.unit
    def test_add_retention_time_features_disable_relative_error(
        self, sample_psms, sample_deeplc_predictions
    ):
        """Test disabling relative error calculation."""
        result = add_retention_time_features(
            sample_psms,
            sample_deeplc_predictions,
            rt_prediction_error_abs_relative=False,
            filter_rel_rt_error=0.5,
        )

        assert "rt_prediction_error_abs" in result.columns
        assert "rt_prediction_error_abs_relative" not in result.columns

        # Should filter based on absolute error when relative is disabled
        for row in result.iter_rows(named=True):
            abs_error = row["rt_prediction_error_abs"]
            assert abs_error < 0.5

    @pytest.mark.unit
    def test_add_retention_time_features_disable_both_errors(
        self, sample_psms, sample_deeplc_predictions
    ):
        """Test disabling both error calculations."""
        result = add_retention_time_features(
            sample_psms,
            sample_deeplc_predictions,
            rt_prediction_error_abs=False,
            rt_prediction_error_abs_relative=False,
            filter_rel_rt_error=0.2,
        )

        assert "rt_prediction_error_abs" not in result.columns
        assert "rt_prediction_error_abs_relative" not in result.columns

        # All original PSMs should remain since no filtering can occur without error columns
        assert result.height == 4  # PEPTIDE1, PEPTIDE2, PEPTIDE3, PEPTIDE4

    @pytest.mark.unit
    def test_add_retention_time_features_perfect_predictions(self):
        """Test with perfect RT predictions (zero error)."""
        psms = pl.DataFrame({"peptide": ["PEPTIDE1", "PEPTIDE2"], "rt": [10.0, 20.0]})

        predictions = pl.DataFrame(
            {
                "peptide": ["PEPTIDE1", "PEPTIDE2"],
                "rt_predictions": [10.0, 20.0],  # Perfect predictions
            }
        )

        result = add_retention_time_features(
            psms, predictions, filter_rel_rt_error=0.01
        )

        assert result.height == 2
        assert all(result["rt_prediction_error_abs"] == 0.0)
        assert all(result["rt_prediction_error_abs_relative"] == 0.0)

    @pytest.mark.unit
    def test_add_retention_time_features_empty_input(self):
        """Test with empty input DataFrames."""
        empty_psms = pl.DataFrame(
            {"peptide": [], "rt": []}, schema={"peptide": pl.Utf8, "rt": pl.Float64}
        )

        empty_predictions = pl.DataFrame(
            {"peptide": [], "rt_predictions": []},
            schema={"peptide": pl.Utf8, "rt_predictions": pl.Float64},
        )

        result = add_retention_time_features(empty_psms, empty_predictions)

        assert result.height == 0
        assert "rt_predictions" in result.columns
        assert "rt_prediction_error_abs" in result.columns
        assert "rt_prediction_error_abs_relative" in result.columns


class TestRetentionTimeFeaturesIntegration:
    """Integration tests for retention time features."""

    @pytest.mark.integration
    def test_realistic_retention_time_workflow(self):
        """Test realistic retention time prediction workflow."""
        # Generate realistic retention time data
        np.random.seed(42)

        n_peptides = 100
        peptides = [f"PEPTIDE{i:03d}" for i in range(n_peptides)]

        # Generate realistic RT values (typically 0-60 minutes)
        rt_values = np.random.uniform(5, 55, n_peptides)

        # Generate predictions with some error
        prediction_errors = np.random.normal(0, 2, n_peptides)  # Mean=0, SD=2 minutes
        rt_predictions = rt_values + prediction_errors

        psms = pl.DataFrame(
            {
                "peptide": peptides,
                "rt": rt_values,
                "charge": np.random.choice([2, 3, 4], n_peptides),
                "score": np.random.uniform(0.6, 0.95, n_peptides),
            }
        )

        # Only provide predictions for 80% of peptides (realistic scenario)
        n_predictions = int(0.8 * n_peptides)
        predictions = pl.DataFrame(
            {
                "peptide": peptides[:n_predictions],
                "rt_predictions": rt_predictions[:n_predictions],
            }
        )

        # Test with moderate filtering
        result = add_retention_time_features(
            psms, predictions, filter_rel_rt_error=0.15  # 15% relative error threshold
        )

        # Should have fewer peptides than input due to filtering
        assert result.height < psms.height
        assert (
            result.height <= n_predictions
        )  # Can't have more than predictions available

        # All remaining peptides should meet the RT error threshold
        max_rt = psms["rt"].max()
        for row in result.iter_rows(named=True):
            rel_error = row["rt_prediction_error_abs_relative"]
            assert rel_error < 0.15

    @pytest.mark.integration
    def test_retention_time_features_with_missing_data(self):
        """Test handling of missing retention time data."""
        psms = pl.DataFrame(
            {
                "peptide": ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"],
                "rt": [10.0, None, 20.0],  # Missing RT value
                "charge": [2, 3, 2],
            }
        )

        predictions = pl.DataFrame(
            {
                "peptide": ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"],
                "rt_predictions": [10.5, 15.0, 19.5],
            }
        )

        # Should handle None values gracefully
        result = add_retention_time_features(psms, predictions, filter_rel_rt_error=0.2)

        # Rows with missing RT should be handled appropriately
        assert "rt_predictions" in result.columns

    @pytest.mark.performance
    def test_retention_time_features_performance(self):
        """Test performance with larger datasets."""
        # Generate larger dataset
        n_peptides = 10000

        peptides = [f"PEPTIDE{i:04d}" for i in range(n_peptides)]
        rt_values = np.random.uniform(0, 60, n_peptides)
        rt_predictions = rt_values + np.random.normal(0, 1, n_peptides)

        psms = pl.DataFrame(
            {
                "peptide": peptides,
                "rt": rt_values,
                "charge": np.random.choice([2, 3, 4], n_peptides),
            }
        )

        predictions = pl.DataFrame(
            {"peptide": peptides, "rt_predictions": rt_predictions}
        )

        # This should complete quickly with Polars
        result = add_retention_time_features(psms, predictions, filter_rel_rt_error=0.1)

        # Basic sanity checks
        assert result.height <= psms.height
        assert "rt_predictions" in result.columns
        assert "rt_prediction_error_abs" in result.columns


class TestRetentionTimeFeaturesEdgeCases:
    """Edge case tests for retention time features."""

    @pytest.mark.unit
    def test_retention_time_features_zero_rt_values(self):
        """Test with zero retention time values."""
        psms = pl.DataFrame({"peptide": ["PEPTIDE1", "PEPTIDE2"], "rt": [0.0, 10.0]})

        predictions = pl.DataFrame(
            {"peptide": ["PEPTIDE1", "PEPTIDE2"], "rt_predictions": [1.0, 9.0]}
        )

        result = add_retention_time_features(psms, predictions, filter_rel_rt_error=1.0)

        assert result.height == 2
        # Should handle zero RT values without division errors
        assert "rt_prediction_error_abs_relative" in result.columns

    @pytest.mark.unit
    def test_retention_time_features_negative_rt_values(self):
        """Test with negative retention time values (edge case)."""
        psms = pl.DataFrame({"peptide": ["PEPTIDE1", "PEPTIDE2"], "rt": [-1.0, 10.0]})

        predictions = pl.DataFrame(
            {"peptide": ["PEPTIDE1", "PEPTIDE2"], "rt_predictions": [0.0, 11.0]}
        )

        result = add_retention_time_features(psms, predictions, filter_rel_rt_error=1.0)

        # Should handle negative values gracefully
        assert result.height == 2
        assert "rt_prediction_error_abs" in result.columns

    @pytest.mark.unit
    def test_retention_time_features_very_large_rt_values(self):
        """Test with very large retention time values."""
        psms = pl.DataFrame(
            {"peptide": ["PEPTIDE1", "PEPTIDE2"], "rt": [1000.0, 2000.0]}
        )

        predictions = pl.DataFrame(
            {"peptide": ["PEPTIDE1", "PEPTIDE2"], "rt_predictions": [1010.0, 1990.0]}
        )

        result = add_retention_time_features(psms, predictions, filter_rel_rt_error=0.1)

        # Should handle large values without overflow
        assert "rt_prediction_error_abs_relative" in result.columns

        # Check that relative errors are calculated correctly
        for row in result.iter_rows(named=True):
            assert row["rt_prediction_error_abs_relative"] < 0.1

    @pytest.mark.unit
    def test_retention_time_features_duplicate_peptides(self):
        """Test with duplicate peptide entries."""
        psms = pl.DataFrame(
            {
                "peptide": ["PEPTIDE1", "PEPTIDE1", "PEPTIDE2"],
                "rt": [10.0, 10.5, 20.0],
                "charge": [2, 3, 2],
            }
        )

        predictions = pl.DataFrame(
            {"peptide": ["PEPTIDE1", "PEPTIDE2"], "rt_predictions": [10.2, 19.8]}
        )

        result = add_retention_time_features(psms, predictions, filter_rel_rt_error=0.2)

        # Should handle duplicate peptides by joining on all matches
        assert result.height >= 2  # Should have at least 2 rows

        # Both PEPTIDE1 entries should get the same prediction
        peptide1_rows = result.filter(pl.col("peptide") == "PEPTIDE1")
        assert all(peptide1_rows["rt_predictions"] == 10.2)
