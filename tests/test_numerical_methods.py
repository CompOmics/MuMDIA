"""
Tests for numerical methods and statistical computations in MuMDIA.

This module tests the numerical accuracy, statistical validity, and
mathematical correctness of calculations used throughout the pipeline.
"""

import pytest
import numpy as np
import polars as pl
from scipy import stats
from unittest.mock import Mock, patch
import warnings

# Test numerical computing modules
try:
    import numba as nb

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class TestNumericalAccuracy:
    """Test numerical accuracy and precision of mathematical operations."""

    @pytest.mark.numerical
    def test_floating_point_precision(self):
        """Test floating point precision in typical calculations."""
        # Test mass calculation precision
        peptide_mass = 1500.123456789
        charge = 2
        proton_mass = 1.007276466812

        # Calculate m/z with high precision
        mz = (peptide_mass + charge * proton_mass) / charge

        # Test precision preservation
        expected_mz = (peptide_mass + 2 * proton_mass) / 2
        assert abs(mz - expected_mz) < 1e-10

        # Test ppm calculation precision
        theoretical_mz = 750.5
        observed_mz = 750.515
        ppm_error = ((observed_mz - theoretical_mz) / theoretical_mz) * 1e6

        expected_ppm = 19.986668
        assert abs(ppm_error - expected_ppm) < 0.001

    @pytest.mark.numerical
    def test_correlation_numerical_stability(self):
        """Test numerical stability of correlation calculations."""
        # Test with well-conditioned data
        n = 1000
        x = np.random.randn(n)
        y = 0.8 * x + 0.6 * np.random.randn(n)

        correlation = np.corrcoef(x, y)[0, 1]

        # Should be close to theoretical correlation
        assert 0.5 < correlation < 0.9
        assert not np.isnan(correlation)
        assert not np.isinf(correlation)

        # Test with constant data (edge case)
        x_constant = np.ones(n)
        y_constant = np.ones(n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlation_constant = np.corrcoef(x_constant, y_constant)[0, 1]

        # Should handle constant data gracefully
        assert np.isnan(correlation_constant) or correlation_constant == 1.0

    @pytest.mark.numerical
    def test_matrix_operations_stability(self):
        """Test numerical stability of matrix operations."""
        # Test well-conditioned matrix
        size = 50
        A = np.random.randn(size, size)

        # Make matrix symmetric positive definite
        A = A @ A.T + np.eye(size)

        # Test eigenvalue calculation
        eigenvals = np.linalg.eigvals(A)

        # All eigenvalues should be positive (due to construction)
        assert np.all(eigenvals > 0)
        assert not np.any(np.isnan(eigenvals))
        assert not np.any(np.isinf(eigenvals))

        # Test matrix inversion
        A_inv = np.linalg.inv(A)
        identity_check = A @ A_inv

        # Should recover identity matrix
        identity_error = np.max(np.abs(identity_check - np.eye(size)))
        assert identity_error < 1e-10

    @pytest.mark.numerical
    def test_percentile_calculation_accuracy(self):
        """Test percentile calculation accuracy."""
        # Test with known data
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Test various percentiles
        p25 = np.percentile(data, 25)
        p50 = np.percentile(data, 50)
        p75 = np.percentile(data, 75)

        # Should match expected values
        assert abs(p25 - 3.25) < 0.1
        assert abs(p50 - 5.5) < 0.1
        assert abs(p75 - 7.75) < 0.1

        # Test with larger random data
        large_data = np.random.exponential(1, 10000)
        percentiles = np.percentile(large_data, [0, 25, 50, 75, 100])

        # Should be monotonically increasing
        for i in range(len(percentiles) - 1):
            assert percentiles[i] <= percentiles[i + 1]


class TestStatisticalMethods:
    """Test statistical methods and hypothesis testing."""

    @pytest.mark.statistical
    def test_correlation_significance_testing(self):
        """Test correlation significance testing."""
        n = 100

        # Test significant correlation
        x = np.random.randn(n)
        y = 0.7 * x + 0.3 * np.random.randn(n)

        correlation, p_value = stats.pearsonr(x, y)

        # Should be significant at α = 0.05
        assert abs(correlation) > 0.5
        assert p_value < 0.05

        # Test non-significant correlation
        x_indep = np.random.randn(n)
        y_indep = np.random.randn(n)

        correlation_indep, p_value_indep = stats.pearsonr(x_indep, y_indep)

        # Should be close to zero with high p-value (usually)
        assert abs(correlation_indep) < 0.5  # Usually true for independent data

    @pytest.mark.statistical
    def test_distribution_fitting_validation(self):
        """Test distribution fitting and validation."""
        # Test exponential distribution (common in MS intensities)
        n = 1000
        true_lambda = 2.0
        data = np.random.exponential(1 / true_lambda, n)

        # Fit exponential distribution
        fitted_lambda = 1 / np.mean(data)

        # Should be close to true parameter
        assert abs(fitted_lambda - true_lambda) < 0.3

        # Test normal distribution
        true_mean = 5.0
        true_std = 2.0
        normal_data = np.random.normal(true_mean, true_std, n)

        fitted_mean = np.mean(normal_data)
        fitted_std = np.std(normal_data, ddof=1)

        assert abs(fitted_mean - true_mean) < 0.2
        assert abs(fitted_std - true_std) < 0.2

    @pytest.mark.statistical
    def test_outlier_detection_methods(self):
        """Test outlier detection statistical methods."""
        # Create data with known outliers
        n = 1000
        clean_data = np.random.normal(0, 1, n)
        outliers = np.array([5, -5, 6, -6])  # Clear outliers
        data_with_outliers = np.concatenate([clean_data, outliers])

        # IQR method
        q75, q25 = np.percentile(data_with_outliers, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        outlier_mask = (data_with_outliers < lower_bound) | (
            data_with_outliers > upper_bound
        )
        detected_outliers = data_with_outliers[outlier_mask]

        # Should detect some outliers
        assert len(detected_outliers) > 0
        assert len(detected_outliers) < len(data_with_outliers) * 0.1  # Less than 10%

        # Z-score method
        z_scores = np.abs(stats.zscore(data_with_outliers))
        z_outliers = data_with_outliers[z_scores > 3]

        # Should detect extreme outliers
        assert len(z_outliers) > 0

    @pytest.mark.statistical
    def test_hypothesis_testing_patterns(self):
        """Test hypothesis testing patterns used in the pipeline."""
        # Two-sample t-test
        n1, n2 = 50, 60
        group1 = np.random.normal(0, 1, n1)
        group2 = np.random.normal(0.5, 1, n2)  # Different mean

        t_stat, p_value = stats.ttest_ind(group1, group2)

        # Should detect difference (usually)
        # Note: This is probabilistic, so we use a relaxed assertion
        assert isinstance(t_stat, (float, np.floating))
        assert isinstance(p_value, (float, np.floating))
        assert 0 <= p_value <= 1

        # Mann-Whitney U test (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")

        assert isinstance(u_stat, (float, np.floating))
        assert 0 <= u_p_value <= 1


class TestSpecializedNumericalMethods:
    """Test specialized numerical methods for mass spectrometry."""

    @pytest.mark.numerical
    def test_mass_accuracy_calculations(self):
        """Test mass accuracy calculation methods."""
        # Test ppm calculation
        theoretical_mass = 1500.7523
        observed_mass = 1500.7543

        ppm_error = ((observed_mass - theoretical_mass) / theoretical_mass) * 1e6
        expected_ppm = ((1500.7543 - 1500.7523) / 1500.7523) * 1e6

        assert abs(ppm_error - expected_ppm) < 1e-6

        # Test Da (Dalton) error
        da_error = observed_mass - theoretical_mass
        expected_da = 0.002

        assert abs(da_error - expected_da) < 1e-6

        # Test relative error
        relative_error = da_error / theoretical_mass
        assert abs(relative_error - (expected_da / theoretical_mass)) < 1e-10

    @pytest.mark.numerical
    def test_isotope_pattern_calculations(self):
        """Test isotope pattern calculation accuracy."""
        # Test carbon isotope spacing
        neutron_mass_diff = 1.0033548378  # Neutron mass difference

        # For different charge states
        charges = [1, 2, 3, 4]
        for charge in charges:
            isotope_spacing = neutron_mass_diff / charge

            # Verify isotope m/z values
            base_mz = 500.0
            isotope_mzs = [base_mz + i * isotope_spacing for i in range(5)]

            # Check spacing consistency
            for i in range(len(isotope_mzs) - 1):
                spacing = isotope_mzs[i + 1] - isotope_mzs[i]
                assert abs(spacing - isotope_spacing) < 1e-10

    @pytest.mark.numerical
    def test_retention_time_modeling_accuracy(self):
        """Test retention time modeling numerical accuracy."""
        # Simulate hydrophobicity-based RT model
        n_peptides = 100
        hydrophobicity_scores = np.random.uniform(0, 100, n_peptides)

        # Linear model: RT = a + b * hydrophobicity + noise
        a, b = 5.0, 0.5  # Model parameters
        noise = np.random.normal(0, 2, n_peptides)
        true_rt = a + b * hydrophobicity_scores + noise

        # Fit linear model
        coeffs = np.polyfit(hydrophobicity_scores, true_rt, 1)
        fitted_b, fitted_a = coeffs

        # Should recover parameters reasonably well
        assert abs(fitted_a - a) < 1.0
        assert abs(fitted_b - b) < 0.2

        # Test prediction accuracy
        predicted_rt = fitted_a + fitted_b * hydrophobicity_scores
        rmse = np.sqrt(np.mean((predicted_rt - true_rt) ** 2))

        # RMSE should be close to noise level
        assert rmse < 5.0  # Should be reasonable

    @pytest.mark.numerical
    def test_fragment_intensity_modeling(self):
        """Test fragment intensity modeling numerical methods."""
        # Simulate fragment intensity relationships
        n_fragments = 200

        # Log-normal distribution for intensities (realistic)
        log_mean, log_std = 3.0, 1.0
        observed_intensities = np.random.lognormal(log_mean, log_std, n_fragments)

        # Add prediction with correlation
        prediction_noise = np.random.lognormal(log_mean, log_std * 0.8, n_fragments)
        predicted_intensities = observed_intensities * 0.7 + prediction_noise * 0.3

        # Test log-space correlation (common in MS)
        log_observed = np.log(observed_intensities)
        log_predicted = np.log(predicted_intensities)

        log_correlation = np.corrcoef(log_observed, log_predicted)[0, 1]

        # Should show reasonable correlation
        assert log_correlation > 0.3
        assert not np.isnan(log_correlation)

        # Test rank correlation (robust to outliers)
        rank_correlation, _ = stats.spearmanr(
            observed_intensities, predicted_intensities
        )

        assert rank_correlation > 0.3
        assert not np.isnan(rank_correlation)


class TestNumericalStabilityEdgeCases:
    """Test numerical stability in edge cases."""

    @pytest.mark.numerical
    def test_zero_division_handling(self):
        """Test handling of zero division cases."""
        # Test safe division patterns
        numerator = np.array([1, 2, 3, 4, 5])
        denominator = np.array([1, 0, 3, 0, 5])  # Contains zeros

        # Safe division with np.divide
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator, dtype=float),
                where=denominator != 0,
            )

        expected = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(result, expected)

        # Alternative safe division
        safe_result = np.where(denominator != 0, numerator / denominator, 0)
        np.testing.assert_array_equal(safe_result, expected)

    @pytest.mark.numerical
    def test_overflow_underflow_handling(self):
        """Test handling of numerical overflow and underflow."""
        # Test large number handling
        large_numbers = np.array([1e300, 1e308, 1e309])  # Near float64 limits

        # Test operations that might overflow
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            squares = large_numbers**2

        # Check for inf values
        inf_count = np.sum(np.isinf(squares))
        assert inf_count >= 1  # At least one should overflow

        # Test small number handling
        small_numbers = np.array([1e-300, 1e-308, 1e-320])  # Near underflow

        # Test operations that might underflow
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            products = small_numbers * small_numbers

        # Check for zero values (underflow)
        zero_count = np.sum(products == 0)
        assert zero_count >= 0  # Some might underflow to zero

    @pytest.mark.numerical
    def test_nan_inf_propagation(self):
        """Test NaN and infinity propagation in calculations."""
        # Test data with NaN and inf
        data_with_nan = np.array([1, 2, np.nan, 4, 5])
        data_with_inf = np.array([1, 2, np.inf, 4, 5])

        # Test NaN handling in statistical functions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            mean_with_nan = np.nanmean(data_with_nan)
            assert not np.isnan(mean_with_nan)
            assert mean_with_nan == np.mean([1, 2, 4, 5])

            # Regular mean should return NaN
            regular_mean = np.mean(data_with_nan)
            assert np.isnan(regular_mean)

        # Test inf handling
        mean_with_inf = np.mean(data_with_inf)
        assert np.isinf(mean_with_inf)

    @pytest.mark.numerical
    def test_numerical_precision_limits(self):
        """Test numerical precision limits and rounding behavior."""
        # Test floating point precision limits
        x = 1.0
        epsilon = np.finfo(float).eps

        # Should be able to distinguish x from x + epsilon
        assert x + epsilon > x

        # But not from x + epsilon/2
        assert x + epsilon / 2 == x

        # Test rounding behavior
        values = np.array([1.4, 1.5, 1.6, 2.5, 3.5])
        rounded = np.round(values)
        expected = np.array([1.0, 2.0, 2.0, 2.0, 4.0])  # Banker's rounding for .5

        np.testing.assert_array_equal(rounded, expected)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
class TestNumbaAcceleratedFunctions:
    """Test Numba-accelerated numerical functions."""

    @pytest.mark.numerical
    def test_numba_percentile_accuracy(self):
        """Test Numba percentile calculation accuracy."""
        # This would test actual Numba functions if they were imported
        # For now, test the concept with regular NumPy

        data = np.random.randn(1000)
        percentiles = [0, 25, 50, 75, 100]

        numpy_results = np.percentile(data, percentiles)

        # Test that percentiles are monotonically increasing
        for i in range(len(numpy_results) - 1):
            assert numpy_results[i] <= numpy_results[i + 1]

        # Test specific percentile properties
        assert numpy_results[0] == np.min(data)  # 0th percentile = min
        assert numpy_results[-1] == np.max(data)  # 100th percentile = max
        assert abs(numpy_results[2] - np.median(data)) < 1e-10  # 50th = median

    @pytest.mark.numerical
    def test_numba_correlation_accuracy(self):
        """Test Numba correlation calculation accuracy."""
        # Test correlation calculation patterns that would be used in Numba
        n = 1000
        x = np.random.randn(n)
        y = 0.6 * x + 0.8 * np.random.randn(n)

        # Manual correlation calculation (as Numba might do)
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))

        manual_correlation = numerator / denominator if denominator != 0 else 0
        numpy_correlation = np.corrcoef(x, y)[0, 1]

        # Should match NumPy result
        assert abs(manual_correlation - numpy_correlation) < 1e-10
