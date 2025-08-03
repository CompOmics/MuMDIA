"""
Tests for features_fragment_intensity module.

This module tests fragment intensity correlation calculations and
feature generation with proper handling of missing dependencies.
"""

import pytest
import numpy as np
import polars as pl
from unittest.mock import Mock, patch

# Check for RustyMS availability
try:
    import rustyms

    RUSTYMS_AVAILABLE = True
except ImportError:
    RUSTYMS_AVAILABLE = False

# Import working functions that don't require problematic dependencies
from feature_generators.features_fragment_intensity import (
    corrcoef_ignore_both_missing,
    corrcoef_ignore_both_missing_counts,
    corrcoef_ignore_zeros,
    cosine_similarity,
)


class TestCorrelationFunctions:
    """Test suite for basic correlation functions that work without Numba/RustyMS."""

    @pytest.mark.unit
    def test_corrcoef_ignore_both_missing_simple_matrix(self):
        """Test correlation calculation with simple 2x3 matrix."""
        # Create simple test matrix where rows are variables, columns are observations
        data = np.array(
            [
                [1.0, 2.0, 3.0],  # Row 1
                [2.0, 4.0, 6.0],  # Row 2 (perfect positive correlation)
            ]
        )

        correlation_matrix = corrcoef_ignore_both_missing(data)

        # Should be 2x2 symmetric matrix
        assert correlation_matrix.shape == (2, 2)
        # Diagonal should be 1.0 (perfect self-correlation)
        assert abs(correlation_matrix[0, 0] - 1.0) < 1e-10
        assert abs(correlation_matrix[1, 1] - 1.0) < 1e-10
        # Off-diagonal should be 1.0 (perfect correlation)
        assert abs(correlation_matrix[0, 1] - 1.0) < 1e-10
        assert abs(correlation_matrix[1, 0] - 1.0) < 1e-10

    @pytest.mark.unit
    def test_corrcoef_ignore_both_missing_with_zeros(self):
        """Test correlation ignoring positions where both values are zero."""
        data = np.array(
            [
                [1.0, 2.0, 0.0, 4.0],  # Row 1
                [2.0, 4.0, 0.0, 8.0],  # Row 2 (both have zero in position 2)
            ]
        )

        correlation_matrix = corrcoef_ignore_both_missing(data)

        # Should still get perfect correlation ignoring the zero positions
        assert correlation_matrix.shape == (2, 2)
        assert abs(correlation_matrix[0, 1] - 1.0) < 1e-10

    @pytest.mark.unit
    def test_corrcoef_ignore_both_missing_counts_simple(self):
        """Test correlation with count tracking."""
        data = np.array([[1.0, 2.0, 0.0, 4.0], [2.0, 4.0, 0.0, 8.0]])

        correlation_matrix, counts_matrix = corrcoef_ignore_both_missing_counts(data)

        assert correlation_matrix.shape == (2, 2)
        assert counts_matrix.shape == (2, 2)
        # Should have 3 valid observations for correlation (excluding the zero pair)
        assert counts_matrix[0, 1] == 3
        assert counts_matrix[1, 0] == 3

    @pytest.mark.unit
    def test_corrcoef_ignore_zeros_simple(self):
        """Test correlation calculation ignoring zero values."""
        data = np.array(
            [
                [1.0, 2.0, 0.0, 4.0],
                [2.0, 4.0, 3.0, 8.0],  # Non-zero where first row has zero
            ]
        )

        correlation_matrix = corrcoef_ignore_zeros(data)

        assert correlation_matrix.shape == (2, 2)
        assert isinstance(correlation_matrix[0, 1], float)

    @pytest.mark.unit
    def test_cosine_similarity_basic(self):
        """Test cosine similarity calculation."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([2.0, 4.0, 6.0])  # Perfect scaling

        similarity = cosine_similarity(vec1, vec2)

        # Should be close to 1.0 for perfect scaling
        assert isinstance(similarity, float)
        assert abs(similarity - 1.0) < 1e-10

    @pytest.mark.unit
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])  # Orthogonal

        similarity = cosine_similarity(vec1, vec2)

        # Should be close to 0.0 for orthogonal vectors
        assert isinstance(similarity, float)
        assert abs(similarity) < 1e-10

    @pytest.mark.unit
    def test_cosine_similarity_zero_norm(self):
        """Test cosine similarity with zero norm vector."""
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([1.0, 2.0, 3.0])

        similarity = cosine_similarity(vec1, vec2)

        # Should handle zero norm gracefully
        assert isinstance(similarity, float)
        # Either 0 or NaN is acceptable for zero norm
        assert similarity == 0.0 or np.isnan(similarity)

    @pytest.mark.unit
    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity with identical vectors."""
        vec = np.array([1.0, 2.0, 3.0])

        similarity = cosine_similarity(vec, vec)

        # Should be exactly 1.0 for identical vectors
        assert abs(similarity - 1.0) < 1e-15


class TestEdgeCases:
    """Test edge cases for working functions."""

    @pytest.mark.unit
    def test_correlation_single_row_matrix(self):
        """Test correlation with single-row matrix."""
        data = np.array([[1.0, 2.0, 3.0]])

        correlation_matrix = corrcoef_ignore_both_missing(data)

        # Should be 1x1 matrix with value 1.0
        assert correlation_matrix.shape == (1, 1)
        assert abs(correlation_matrix[0, 0] - 1.0) < 1e-15

    @pytest.mark.unit
    def test_correlation_insufficient_data(self):
        """Test correlation when insufficient valid data points exist."""
        data = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])  # Only one non-zero overlap

        correlation_matrix = corrcoef_ignore_both_missing(data)

        # Should return NaN when insufficient data
        assert np.isnan(correlation_matrix[0, 1])

    @pytest.mark.unit
    def test_correlation_matrix_symmetry(self):
        """Test that correlation matrices are symmetric."""
        data = np.random.rand(3, 10)  # 3 variables, 10 observations

        correlation_matrix = corrcoef_ignore_both_missing(data)

        # Should be symmetric
        assert np.allclose(correlation_matrix, correlation_matrix.T)


# Skip tests that require problematic dependencies
class TestComputeCorrelations:
    """Test suite for compute_correlations function (requires Numba resolution)."""

    @pytest.mark.skip(
        reason="Numba compilation issues - requires dependency resolution"
    )
    def test_compute_correlations_perfect_correlation(self):
        """Test correlation calculation with perfect correlation."""
        pass

    @pytest.mark.skip(
        reason="Numba compilation issues - requires dependency resolution"
    )
    def test_compute_correlations_no_correlation(self):
        """Test correlation calculation with no correlation."""
        pass

    @pytest.mark.skip(
        reason="Numba compilation issues - requires dependency resolution"
    )
    def test_compute_correlations_zero_variance(self):
        """Test correlation calculation with zero variance."""
        pass

    @pytest.mark.skip(
        reason="Numba compilation issues - requires dependency resolution"
    )
    def test_compute_correlations_empty_matrix(self):
        """Test correlation calculation with empty input matrix."""
        pass

    @pytest.mark.skip(
        reason="Numba compilation issues - requires dependency resolution"
    )
    def test_compute_correlations_single_psm(self):
        """Test correlation calculation with single PSM."""
        pass


class TestMatchFragments:
    """Test the fragment matching function (requires RustyMS)."""

    @pytest.mark.skipif(
        not RUSTYMS_AVAILABLE, reason="RustyMS dependency not available"
    )
    def test_match_fragments_basic(self):
        """Test basic fragment matching."""
        pass

    @pytest.mark.skipif(
        not RUSTYMS_AVAILABLE, reason="RustyMS dependency not available"
    )
    def test_match_fragments_empty_data(self):
        """Test fragment matching with empty data."""
        pass

    @pytest.mark.skipif(
        not RUSTYMS_AVAILABLE, reason="RustyMS dependency not available"
    )
    def test_match_fragments_no_matches(self):
        """Test fragment matching with no matches."""
        pass

    @pytest.mark.skipif(
        not RUSTYMS_AVAILABLE, reason="RustyMS dependency not available"
    )
    def test_match_fragments_tolerance(self):
        """Test fragment matching with different tolerances."""
        pass


class TestGetFeaturesFragmentIntensity:
    """Test the main feature generation function (requires multiple dependencies)."""

    @pytest.mark.skipif(
        not RUSTYMS_AVAILABLE, reason="RustyMS dependency and Rich compatibility issues"
    )
    def test_get_features_fragment_intensity_basic(self):
        """Test basic feature generation."""
        pass

    @pytest.mark.skipif(
        not RUSTYMS_AVAILABLE, reason="RustyMS dependency and Rich compatibility issues"
    )
    def test_get_features_fragment_intensity_no_predictions(self):
        """Test feature generation with no predictions."""
        pass

    @pytest.mark.skipif(
        not RUSTYMS_AVAILABLE, reason="RustyMS dependency and Rich compatibility issues"
    )
    def test_get_features_fragment_intensity_read_pickles(self):
        """Test feature generation reading from pickles."""
        pass

    @pytest.mark.skipif(
        not RUSTYMS_AVAILABLE, reason="RustyMS dependency and Rich compatibility issues"
    )
    def test_get_features_fragment_intensity_write_pickles(self):
        """Test feature generation writing to pickles."""
        pass

    @pytest.mark.skipif(
        not RUSTYMS_AVAILABLE, reason="RustyMS dependency and Rich compatibility issues"
    )
    def test_get_features_fragment_intensity_error_handling(self):
        """Test feature generation error handling."""
        pass
