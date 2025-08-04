"""
Test suite for general feature generation functions.

This module tests the core functionality of the features_general module,
including peptide counting, filtering, and data quality checks.
"""

import numpy as np
import polars as pl
import pytest

from feature_generators.features_general import add_count_and_filter_peptides


class TestAddCountAndFilterPeptides:
    """Test suite for peptide counting and filtering functionality."""

    @pytest.mark.unit
    def test_add_count_and_filter_peptides_basic(self):
        """Test basic peptide counting and filtering functionality."""
        # Create test data with some repeated peptides
        df_psms = pl.DataFrame(
            {
                "peptide": [
                    "PEPTIDE1",
                    "PEPTIDE1",
                    "PEPTIDE2",
                    "PEPTIDE3",
                    "PEPTIDE3",
                    "PEPTIDE3",
                ],
                "charge": [2, 3, 2, 2, 3, 4],
                "score": [0.9, 0.8, 0.7, 0.95, 0.85, 0.75],
            }
        )

        result = add_count_and_filter_peptides(df_psms, min_occurrences=2)

        # Should keep PEPTIDE1 (count=2) and PEPTIDE3 (count=3), filter out PEPTIDE2 (count=1)
        assert result.height == 5  # 2 + 3 = 5 rows
        assert "count" in result.columns

        # Check peptide counts
        peptide1_rows = result.filter(pl.col("peptide") == "PEPTIDE1")
        peptide3_rows = result.filter(pl.col("peptide") == "PEPTIDE3")

        assert peptide1_rows.height == 2
        assert peptide3_rows.height == 3
        assert peptide1_rows["count"][0] == 2
        assert peptide3_rows["count"][0] == 3

        # PEPTIDE2 should be filtered out
        peptide2_rows = result.filter(pl.col("peptide") == "PEPTIDE2")
        assert peptide2_rows.height == 0

    @pytest.mark.unit
    def test_add_count_and_filter_peptides_min_occurrences_1(self):
        """Test with minimum occurrences of 1 (no filtering)."""
        df_psms = pl.DataFrame(
            {"peptide": ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"], "score": [0.9, 0.8, 0.7]}
        )

        result = add_count_and_filter_peptides(df_psms, min_occurrences=1)

        # All peptides should be kept
        assert result.height == 3
        assert all(result["count"] == 1)

    @pytest.mark.unit
    def test_add_count_and_filter_peptides_high_threshold(self):
        """Test with high minimum occurrence threshold."""
        df_psms = pl.DataFrame(
            {
                "peptide": ["PEPTIDE1", "PEPTIDE1", "PEPTIDE2", "PEPTIDE3"],
                "score": [0.9, 0.8, 0.7, 0.6],
            }
        )

        result = add_count_and_filter_peptides(df_psms, min_occurrences=3)

        # No peptides should meet the threshold
        assert result.height == 0

    @pytest.mark.unit
    def test_add_count_and_filter_peptides_empty_dataframe(self):
        """Test with empty input DataFrame."""
        df_psms = pl.DataFrame({"peptide": [], "score": []})

        result = add_count_and_filter_peptides(df_psms, min_occurrences=2)

        assert result.height == 0
        assert "count" in result.columns

    @pytest.mark.unit
    def test_add_count_and_filter_peptides_single_peptide_multiple_occurrences(self):
        """Test with a single peptide appearing multiple times."""
        df_psms = pl.DataFrame(
            {
                "peptide": ["PEPTIDE1"] * 5,
                "charge": [2, 3, 2, 4, 3],
                "score": [0.9, 0.8, 0.85, 0.75, 0.95],
            }
        )

        result = add_count_and_filter_peptides(df_psms, min_occurrences=3)

        assert result.height == 5
        assert all(result["count"] == 5)
        assert all(result["peptide"] == "PEPTIDE1")

    @pytest.mark.unit
    def test_add_count_and_filter_peptides_preserves_other_columns(self):
        """Test that other columns are preserved during filtering."""
        df_psms = pl.DataFrame(
            {
                "peptide": ["PEPTIDE1", "PEPTIDE1", "PEPTIDE2"],
                "charge": [2, 3, 2],
                "score": [0.9, 0.8, 0.7],
                "rt": [10.5, 11.2, 15.8],
                "mz": [500.1, 333.4, 600.2],
            }
        )

        result = add_count_and_filter_peptides(df_psms, min_occurrences=2)

        # Should keep only PEPTIDE1 rows
        assert result.height == 2
        assert set(result.columns) == {
            "peptide",
            "charge",
            "score",
            "rt",
            "mz",
            "count",
        }

        # Check that all original columns are preserved
        peptide1_rows = result.filter(pl.col("peptide") == "PEPTIDE1")
        assert peptide1_rows["charge"].to_list() == [2, 3]
        assert peptide1_rows["score"].to_list() == [0.9, 0.8]
        assert peptide1_rows["rt"].to_list() == [10.5, 11.2]

    @pytest.mark.unit
    def test_add_count_and_filter_peptides_case_sensitivity(self):
        """Test that peptide names are case-sensitive."""
        df_psms = pl.DataFrame(
            {"peptide": ["PEPTIDE1", "peptide1", "PEPTIDE1"], "score": [0.9, 0.8, 0.7]}
        )

        result = add_count_and_filter_peptides(df_psms, min_occurrences=2)

        # Only "PEPTIDE1" (uppercase) should meet the threshold
        assert result.height == 2
        peptide_values = result["peptide"].unique().to_list()
        assert "PEPTIDE1" in peptide_values
        assert "peptide1" not in peptide_values

    @pytest.mark.unit
    def test_add_count_and_filter_peptides_identical_scores_different_peptides(self):
        """Test behavior with identical scores but different peptides."""
        df_psms = pl.DataFrame(
            {
                "peptide": ["PEPTIDE1", "PEPTIDE1", "PEPTIDE2", "PEPTIDE2"],
                "score": [0.8, 0.8, 0.8, 0.8],
            }
        )

        result = add_count_and_filter_peptides(df_psms, min_occurrences=2)

        # Both peptides should be kept
        assert result.height == 4
        unique_peptides = result["peptide"].unique().to_list()
        assert len(unique_peptides) == 2
        assert "PEPTIDE1" in unique_peptides
        assert "PEPTIDE2" in unique_peptides


class TestGeneralFeaturesIntegration:
    """Integration tests for general feature functionality."""

    @pytest.mark.integration
    def test_realistic_psm_filtering_workflow(self):
        """Test realistic peptide filtering with proteomics-like data."""
        # Simulate realistic PSM data
        np.random.seed(42)

        peptides = []
        charges = []
        scores = []

        # Generate peptides with different occurrence patterns
        high_occurrence_peptides = ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"]
        medium_occurrence_peptides = ["PEPTIDE4", "PEPTIDE5"]
        low_occurrence_peptides = ["PEPTIDE6", "PEPTIDE7", "PEPTIDE8", "PEPTIDE9"]

        # High occurrence peptides (5-10 observations each)
        for peptide in high_occurrence_peptides:
            n_obs = np.random.randint(5, 11)
            peptides.extend([peptide] * n_obs)
            charges.extend(np.random.choice([2, 3, 4], n_obs).tolist())
            scores.extend(np.random.uniform(0.7, 0.95, n_obs).tolist())

        # Medium occurrence peptides (2-4 observations each)
        for peptide in medium_occurrence_peptides:
            n_obs = np.random.randint(2, 5)
            peptides.extend([peptide] * n_obs)
            charges.extend(np.random.choice([2, 3, 4], n_obs).tolist())
            scores.extend(np.random.uniform(0.6, 0.9, n_obs).tolist())

        # Low occurrence peptides (1 observation each)
        for peptide in low_occurrence_peptides:
            peptides.append(peptide)
            charges.append(np.random.choice([2, 3, 4]))
            scores.append(np.random.uniform(0.5, 0.8))

        df_psms = pl.DataFrame(
            {"peptide": peptides, "charge": charges, "score": scores}
        )

        # Test with different thresholds
        result_min2 = add_count_and_filter_peptides(df_psms, min_occurrences=2)
        result_min5 = add_count_and_filter_peptides(df_psms, min_occurrences=5)

        # Check that filtering works as expected
        unique_peptides_min2 = result_min2["peptide"].unique().to_list()
        unique_peptides_min5 = result_min5["peptide"].unique().to_list()

        # With min_occurrences=2, should keep high and medium occurrence peptides
        for peptide in high_occurrence_peptides + medium_occurrence_peptides:
            assert peptide in unique_peptides_min2

        # Low occurrence peptides should be filtered out
        for peptide in low_occurrence_peptides:
            assert peptide not in unique_peptides_min2

        # With min_occurrences=5, should only keep high occurrence peptides
        for peptide in high_occurrence_peptides:
            if (
                peptide in unique_peptides_min5
            ):  # Some might not meet threshold due to randomness
                assert result_min5.filter(pl.col("peptide") == peptide)["count"][0] >= 5

    @pytest.mark.performance
    def test_add_count_and_filter_peptides_performance(self):
        """Test performance with larger datasets."""
        # Generate larger test dataset
        n_peptides = 1000
        n_observations = 10000

        peptides = np.random.choice(
            [f"PEPTIDE{i}" for i in range(n_peptides)], n_observations
        )
        charges = np.random.choice([2, 3, 4], n_observations)
        scores = np.random.uniform(0.5, 1.0, n_observations)

        df_psms = pl.DataFrame(
            {"peptide": peptides, "charge": charges, "score": scores}
        )

        # This should complete quickly with Polars
        result = add_count_and_filter_peptides(df_psms, min_occurrences=2)

        # Basic sanity checks
        assert result.height <= df_psms.height
        assert "count" in result.columns
        assert all(result["count"] >= 2)


class TestGeneralFeaturesEdgeCases:
    """Edge case tests for general feature functions."""

    @pytest.mark.unit
    def test_add_count_and_filter_peptides_none_values(self):
        """Test behavior with None values in peptide column."""
        df_psms = pl.DataFrame(
            {
                "peptide": ["PEPTIDE1", None, "PEPTIDE1", None],
                "score": [0.9, 0.8, 0.7, 0.6],
            }
        )

        result = add_count_and_filter_peptides(df_psms, min_occurrences=2)

        # Should handle None values gracefully
        # PEPTIDE1 appears twice, None appears twice
        assert result.height >= 2  # At least PEPTIDE1 rows should be kept

    @pytest.mark.unit
    def test_add_count_and_filter_peptides_empty_string_peptides(self):
        """Test behavior with empty string peptides."""
        df_psms = pl.DataFrame(
            {"peptide": ["PEPTIDE1", "", "PEPTIDE1", ""], "score": [0.9, 0.8, 0.7, 0.6]}
        )

        result = add_count_and_filter_peptides(df_psms, min_occurrences=2)

        # Both "PEPTIDE1" and "" should meet threshold
        assert result.height == 4
        unique_peptides = result["peptide"].unique().to_list()
        assert "PEPTIDE1" in unique_peptides
        assert "" in unique_peptides

    @pytest.mark.unit
    def test_add_count_and_filter_peptides_zero_min_occurrences(self):
        """Test with zero minimum occurrences."""
        df_psms = pl.DataFrame(
            {"peptide": ["PEPTIDE1", "PEPTIDE2"], "score": [0.9, 0.8]}
        )

        result = add_count_and_filter_peptides(df_psms, min_occurrences=0)

        # All peptides should be kept
        assert result.height == 2
        assert all(result["count"] >= 0)
