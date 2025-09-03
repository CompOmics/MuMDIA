"""
DIA-NN Feature Generator for Proteomics Analysis

This module implements the DIANNFeatureGenerator class for calculating comprehensive
features from MS/MS proteomics data for use in machine learning models. The features
are based on fragment elution profiles, correlations, and spectral library predictions.

Author: Generated from Jupyter notebook analysis
Date: August 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.signal import savgol_filter
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration parameters for feature calculation."""

    # Mass tolerance settings
    fragment_mass_tolerance: float = 13.0  # ppm
    precursor_mass_tolerance: float = 50.0  # ppm

    # RT tolerance settings
    rt_tolerance: float = 5.0  # minutes or seconds

    # Smoothing settings
    savgol_window_length: int = 3
    savgol_polyorder: int = 1

    # Feature-specific settings
    top_n_fragments: int = 6
    top_n_fragments_extended: int = 12
    isotope_mass_c13: float = 1.00335
    c13_isotope_list: List[int] = None
    ms1_accuracy_factors: List[float] = None
    ms2_accuracy_factors: List[float] = None
    
    # Parallelization settings
    n_jobs: int = -1  # -1 means use all available CPU cores

    def __post_init__(self):
        """Set default values for list parameters."""
        if self.c13_isotope_list is None:
            self.c13_isotope_list = [1, 2, 3]
        if self.ms1_accuracy_factors is None:
            self.ms1_accuracy_factors = [1.0, 0.45, 0.2]
        if self.ms2_accuracy_factors is None:
            self.ms2_accuracy_factors = [1.0, 0.45, 0.2]


class DIANNFeatureGenerator:
    """
    Comprehensive feature generator for DIA-NN proteomics data.

    This class calculates a wide range of features from MS/MS data including:
    - Ion co-elution features (MS2 level)
    - Ion co-elution features (MS1 level)
    - Isotopologue co-elution features
    - Total signal features
    - Fragment intensity features
    - Mass accuracy features
    - Retention time features
    - Elution profile shape features
    - Library characteristics features

    The implementation addresses common issues in proteomics feature engineering:
    - Robust error handling and input validation
    - Configurable parameters instead of hard-coded values
    - Consistent NaN handling
    - Performance optimizations
    - Comprehensive logging and documentation
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the feature generator.

        Parameters
        ----------
        config : FeatureConfig, optional
            Configuration object with parameters. If None, uses defaults.
        """
        self.config = config if config is not None else FeatureConfig()
        self._validate_config()
        self._setup_parallelization()

    def _setup_parallelization(self):
        """Set up parallelization parameters."""
        import os
        if self.config.n_jobs == -1:
            self.n_workers = os.cpu_count()
        elif self.config.n_jobs > 0:
            self.n_workers = min(self.config.n_jobs, os.cpu_count())
        else:
            self.n_workers = 1
        
        logger.info(f"Using {self.n_workers} workers for parallel feature calculation")

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.fragment_mass_tolerance <= 0:
            raise ValueError("Fragment mass tolerance must be positive")
        if self.config.rt_tolerance <= 0:
            raise ValueError("RT tolerance must be positive")
        if self.config.top_n_fragments <= 0:
            raise ValueError("Number of top fragments must be positive")

    def _validate_fragments_input(self, fragments: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean fragment input data.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data

        Returns
        -------
        pd.DataFrame
            Validated and cleaned fragment data

        Raises
        ------
        ValueError
            If required columns are missing or data is invalid
        """
        required_cols = ["fragment_names", "rt", "fragment_intensity"]
        missing_cols = set(required_cols) - set(fragments.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if fragments.empty:
            raise ValueError("Fragment data is empty")

        # Clean data
        fragments = fragments.copy()
        fragments["rt"] = pd.to_numeric(fragments["rt"], errors="coerce")
        fragments["fragment_intensity"] = pd.to_numeric(
            fragments["fragment_intensity"], errors="coerce"
        )

        # Remove invalid data
        initial_size = len(fragments)
        fragments = fragments.dropna(subset=["rt", "fragment_intensity"])

        if fragments.empty:
            raise ValueError("No valid fragment data after cleaning")

        if len(fragments) < initial_size:
            logger.warning(f"Removed {initial_size - len(fragments)} invalid rows")

        return fragments

    def _search_sorted_with_tolerance(
        self, arr: np.ndarray, target: float, tolerance: float
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Find closest value in sorted array within tolerance.

        Parameters
        ----------
        arr : np.ndarray
            Sorted array to search
        target : float
            Target value
        tolerance : float
            Maximum allowed difference

        Returns
        -------
        Tuple[Optional[int], Optional[float]]
            Index and value of closest match, or (None, None) if no match
        """
        if len(arr) == 0:
            return None, None

        arr = np.asarray(arr)
        if not np.all(np.diff(arr) >= 0):
            arr = np.sort(arr)

        idx = np.searchsorted(arr, target)
        candidates = []

        # Check current position and neighbors
        for check_idx in [idx - 1, idx, idx + 1]:
            if 0 <= check_idx < len(arr):
                candidates.append((check_idx, arr[check_idx]))

        if not candidates:
            return None, None

        # Find closest within tolerance
        best_idx, best_val = min(candidates, key=lambda x: abs(x[1] - target))
        return (
            (best_idx, best_val)
            if abs(best_val - target) <= tolerance
            else (None, None)
        )

    def find_top_n_fragments(self, fragments: pd.DataFrame, n: int = None) -> List[str]:
        """
        Find top n fragments by maximum intensity.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        n : int, optional
            Number of fragments to return. Uses config default if None.

        Returns
        -------
        List[str]
            List of fragment names sorted by intensity (descending)
        """
        if n is None:
            n = self.config.top_n_fragments

        fragments = self._validate_fragments_input(fragments)

        if "fragment_names" not in fragments.columns:
            raise ValueError("Missing 'fragment_names' column")

        top_fragments = (
            fragments.groupby("fragment_names")["fragment_intensity"]
            .max()
            .nlargest(n)
            .index.tolist()
        )

        return top_fragments

    def find_best_fragment(self, fragments: pd.DataFrame) -> str:
        """
        Find the "best" fragment based on correlation with other top fragments.

        The best fragment is defined as the fragment from the top 6 most intense
        fragments that maximizes the sum of Pearson correlations with other fragments.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data

        Returns
        -------
        str
            Name of the best fragment

        Raises
        ------
        ValueError
            If no best fragment can be determined
        """
        fragments = self._validate_fragments_input(fragments)
        top_fragments = self.find_top_n_fragments(
            fragments, self.config.top_n_fragments
        )

        if len(top_fragments) < 2:
            logger.warning("Less than 2 fragments available, using first fragment")
            return top_fragments[0] if top_fragments else None

        # Filter to top fragments only
        filtered_fragments = fragments[fragments["fragment_names"].isin(top_fragments)]

        # Create pivot table for correlation calculation
        try:
            pivot_table = filtered_fragments.pivot_table(
                index="rt",
                columns="fragment_names",
                values="fragment_intensity",
                aggfunc="mean",  # Handle duplicates
            )
        except Exception as e:
            raise ValueError(f"Failed to create pivot table: {e}")

        if pivot_table.empty:
            raise ValueError("No data available for correlation calculation")

        # Calculate correlations (excluding self-correlation)
        corr_matrix = pivot_table.corr()

        # Set diagonal to 0 to exclude self-correlations
        np.fill_diagonal(corr_matrix.values, 0)

        # Find fragment with highest sum of correlations
        corr_sums = corr_matrix.sum(axis=1)
        best_fragment = corr_sums.idxmax()

        if pd.isna(best_fragment):
            raise ValueError("Could not determine best fragment")

        return best_fragment

    def _apply_savgol_smoothing(
        self,
        intensity: np.ndarray,
        window_length: Optional[int] = None,
        polyorder: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing with robust parameter handling.

        Parameters
        ----------
        intensity : np.ndarray
            Intensity values to smooth
        window_length : int, optional
            Window length for smoothing
        polyorder : int, optional
            Polynomial order for smoothing

        Returns
        -------
        np.ndarray
            Smoothed intensity values
        """
        if window_length is None:
            window_length = self.config.savgol_window_length
        if polyorder is None:
            polyorder = self.config.savgol_polyorder

        if len(intensity) < 3:
            return intensity.copy()

        # Ensure odd window length
        wl = min(window_length, len(intensity))
        if wl % 2 == 0:
            wl -= 1
        wl = max(3, wl)  # Minimum window length of 3

        # Ensure polyorder is less than window length
        po = min(polyorder, wl - 1)

        try:
            return savgol_filter(intensity, window_length=wl, polyorder=po)
        except Exception as e:
            logger.warning(f"Smoothing failed: {e}, returning original data")
            return intensity.copy()

    def calculate_pearson_correlations(
        self,
        fragments: pd.DataFrame,
        best_fragment: Optional[str] = None,
        use_all_rt: bool = False,
    ) -> Tuple[pd.Series, np.ndarray, Dict[str, float]]:
        """
        Calculate Pearson correlations between fragment elution profiles.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        best_fragment : str, optional
            Name of best fragment. If None, will be determined automatically.
        use_all_rt : bool, default=False
            Whether to use all RT points (filling missing with 0) or only overlapping points

        Returns
        -------
        Tuple[pd.Series, np.ndarray, Dict[str, float]]
            Best fragment trace, smoothed trace, and correlation dictionary
        """
        fragments = self._validate_fragments_input(fragments)

        if best_fragment is None:
            best_fragment = self.find_best_fragment(fragments)

        # Create pivot table
        pivot_table = fragments.pivot_table(
            index="rt",
            columns="fragment_names",
            values="fragment_intensity",
            aggfunc="mean",
        )

        if best_fragment not in pivot_table.columns:
            raise ValueError(f"Best fragment '{best_fragment}' not found in data")

        # Get best fragment trace and smooth it
        best_trace = pivot_table[best_fragment].dropna()
        if len(best_trace) == 0:
            raise ValueError("Best fragment has no valid data points")

        smoothed_best_trace = self._apply_savgol_smoothing(best_trace.values)

        # Calculate correlations
        correlations = {}

        if use_all_rt:
            # Fill missing values with 0 and use all RT points
            pivot_filled = pivot_table.fillna(0.0)
            best_trace_filled = pivot_filled[best_fragment]
            smoothed_best_filled = self._apply_savgol_smoothing(
                best_trace_filled.values
            )

            for frag in pivot_table.columns:
                frag_trace = pivot_filled[frag]
                if len(frag_trace) < 2:
                    correlations[frag] = np.nan
                    continue

                try:
                    corr = pd.Series(
                        smoothed_best_filled, index=pivot_filled.index
                    ).corr(frag_trace)
                    correlations[frag] = corr if not pd.isna(corr) else 0.0
                except Exception:
                    correlations[frag] = 0.0
        else:
            # Use only overlapping RT points
            best_smoothed_series = pd.Series(
                smoothed_best_trace, index=best_trace.index
            )

            for frag in pivot_table.columns:
                frag_trace = pivot_table[frag].dropna()
                common_index = best_trace.index.intersection(frag_trace.index)

                if len(common_index) < 2:
                    correlations[frag] = np.nan
                    continue

                try:
                    corr = best_smoothed_series.loc[common_index].corr(
                        frag_trace.loc[common_index]
                    )
                    correlations[frag] = corr if not pd.isna(corr) else 0.0
                except Exception:
                    correlations[frag] = 0.0

        return best_trace, smoothed_best_trace, correlations

    def build_elution_profile(
        self,
        target_mz: float,
        ms1_dict: Dict[str, Dict[str, Any]],
        tolerance_ppm: Optional[float] = None,
        acc_factor: float = 1.0,
    ) -> Dict[float, float]:
        """
        Build elution profile from MS1 data for a given m/z.

        Parameters
        ----------
        target_mz : float
            Target m/z value
        ms1_dict : dict
            MS1 data dictionary
        tolerance_ppm : float, optional
            Mass tolerance in ppm. Uses config default if None.
        acc_factor : float, default=1.0
            Accuracy factor to multiply tolerance

        Returns
        -------
        Dict[float, float]
            Dictionary mapping RT to intensity
        """
        if tolerance_ppm is None:
            tolerance_ppm = self.config.precursor_mass_tolerance

        elution_profile = {}
        tol_mz = target_mz * tolerance_ppm / 1e6 * acc_factor

        for scan, scan_dict in ms1_dict.items():
            mzs = scan_dict.get("mz", [])
            intensities = scan_dict.get("intensity", [])
            rt = scan_dict.get("retention_time", None)

            if rt is None or len(mzs) == 0 or len(intensities) == 0:
                continue

            # Convert RT from seconds to minutes if needed
            if isinstance(rt, (int, float)) and rt > 1000:  # Likely in seconds
                rt = rt / 60

            best_idx, best_val = self._search_sorted_with_tolerance(
                mzs, target_mz, tol_mz
            )

            if best_idx is not None:
                elution_profile[rt] = intensities[best_idx]

        return elution_profile

    # Feature Group 1: Ion Co-elution (MS2 level)
    def feature_pearson_correlations_top_n(
        self, fragments: pd.DataFrame, n: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate Pearson correlations of top n fragments with best fragment.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        n : int, optional
            Number of top fragments. Uses config default if None.

        Returns
        -------
        np.ndarray
            Array of correlation coefficients
        """
        if n is None:
            n = self.config.top_n_fragments_extended

        fragments = self._validate_fragments_input(fragments)

        try:
            _, _, correlations = self.calculate_pearson_correlations(fragments)
            top_fragments = self.find_top_n_fragments(fragments, n)

            result = np.array(
                [correlations.get(frag, np.nan) for frag in top_fragments]
            )

            return result
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return np.full(n, np.nan)

    def feature_sum_correlations_mass_accuracy(
        self, fragments: pd.DataFrame
    ) -> np.ndarray:
        """
        Sum of correlations for top 6 fragments at different mass accuracies.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data

        Returns
        -------
        np.ndarray
            Array of correlation sums for different mass accuracy factors
        """
        if "ppm_error" not in fragments.columns:
            logger.warning("ppm_error column missing, skipping mass accuracy filtering")
            return np.array([np.nan] * len(self.config.ms2_accuracy_factors))

        results = []
        for factor in self.config.ms2_accuracy_factors:
            try:
                # Filter by mass accuracy
                filtered_fragments = fragments[
                    fragments["ppm_error"]
                    <= self.config.fragment_mass_tolerance * factor
                ]

                if filtered_fragments.empty:
                    results.append(np.nan)
                    continue

                corrs = self.feature_pearson_correlations_top_n(
                    filtered_fragments, self.config.top_n_fragments
                )
                results.append(np.nansum(corrs))

            except Exception as e:
                logger.error(f"Error with mass accuracy factor {factor}: {e}")
                results.append(np.nan)

        return np.array(results)

    def feature_remaining_fragments_correlations(
        self, fragments: pd.DataFrame
    ) -> np.ndarray:
        """
        Sum of correlations for remaining fragments (non-normalized and normalized).

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data

        Returns
        -------
        np.ndarray
            Array with [non_normalized_sum, normalized_sum]
        """
        try:
            _, _, correlations = self.calculate_pearson_correlations(fragments)
            top_fragments = self.find_top_n_fragments(
                fragments, self.config.top_n_fragments_extended
            )

            remaining_corrs = np.array(
                [
                    correlations.get(frag, np.nan)
                    for frag in correlations.keys()
                    if frag not in top_fragments
                ]
            )

            # Remove NaNs
            valid_corrs = remaining_corrs[~np.isnan(remaining_corrs)]

            if len(valid_corrs) == 0:
                return np.array([0.0, 0.0])

            non_normalized = np.sum(valid_corrs)
            normalized = non_normalized / len(valid_corrs)

            return np.array([non_normalized, normalized])

        except Exception as e:
            logger.error(f"Error calculating remaining fragment correlations: {e}")
            return np.array([np.nan, np.nan])

    def feature_best_b_fragments_correlation(
        self, fragments: pd.DataFrame, n: int = 3
    ) -> float:
        """
        Sum of correlations for top n b-fragments.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        n : int, default=3
            Number of top b-fragments to consider

        Returns
        -------
        float
            Sum of correlations for best b-fragments
        """
        try:
            _, _, correlations = self.calculate_pearson_correlations(fragments)

            # Find b-fragments
            b_fragments = [frag for frag in correlations.keys() if frag.startswith("b")]

            if len(b_fragments) == 0:
                return np.nan

            # Sort by correlation and take top n
            b_fragments_sorted = sorted(
                b_fragments, key=lambda x: correlations.get(x, -np.inf), reverse=True
            )[:n]

            corrs = [correlations.get(frag, 0.0) for frag in b_fragments_sorted]
            return np.nansum(corrs)

        except Exception as e:
            logger.error(f"Error calculating b-fragment correlations: {e}")
            return np.nan

    def feature_precursor_best_fragment_correlation(
        self,
        precursor: pd.DataFrame,
        fragments: pd.DataFrame,
        ms1_dict: Dict[str, Dict[str, Any]],
    ) -> float:
        """
        Correlation between precursor and best fragment elution profiles.

        Parameters
        ----------
        precursor : pd.DataFrame
            Precursor information
        fragments : pd.DataFrame
            Fragment data
        ms1_dict : dict
            MS1 data dictionary

        Returns
        -------
        float
            Correlation coefficient
        """
        try:
            # Calculate precursor m/z
            precursor_mz = precursor["calcmass"].iloc[0] / precursor["charge"].iloc[0]

            # Build precursor elution profile
            elution_profile = self.build_elution_profile(precursor_mz, ms1_dict)

            if not elution_profile:
                return np.nan

            # Get best fragment profile
            best_trace, smoothed_best_trace, _ = self.calculate_pearson_correlations(
                fragments
            )

            # Align profiles
            smoothed_best_df = pd.DataFrame(
                {"rt": best_trace.index, "smoothed_intensity": smoothed_best_trace}
            )

            elution_df = pd.DataFrame(
                list(elution_profile.items()), columns=["rt", "intensity"]
            )

            # Merge with tolerance
            merged = pd.merge_asof(
                smoothed_best_df.sort_values("rt"),
                elution_df.sort_values("rt"),
                on="rt",
                direction="nearest",
                tolerance=self.config.rt_tolerance,
            )

            # Calculate correlation
            merged_clean = merged.dropna(subset=["smoothed_intensity", "intensity"])

            if len(merged_clean) < 2:
                return np.nan

            return merged_clean["smoothed_intensity"].corr(merged_clean["intensity"])

        except Exception as e:
            logger.error(f"Error calculating precursor-fragment correlation: {e}")
            return np.nan

    # Feature Group 2: MS1 Level Co-elution
    def feature_ms1_accuracy_correlations(
        self,
        precursor: pd.DataFrame,
        fragments: pd.DataFrame,
        ms1_dict: Dict[str, Dict[str, Any]],
    ) -> np.ndarray:
        """
        Correlations at different MS1 mass accuracies.

        Parameters
        ----------
        precursor : pd.DataFrame
            Precursor information
        fragments : pd.DataFrame
            Fragment data
        ms1_dict : dict
            MS1 data dictionary

        Returns
        -------
        np.ndarray
            Array of correlations for different accuracy factors
        """
        results = []

        for acc_factor in self.config.ms1_accuracy_factors:
            try:
                corr = self._calculate_precursor_fragment_correlation_with_accuracy(
                    precursor, fragments, ms1_dict, acc_factor
                )
                results.append(corr)
            except Exception as e:
                logger.error(f"Error with MS1 accuracy factor {acc_factor}: {e}")
                results.append(np.nan)

        return np.array(results)

    def _calculate_precursor_fragment_correlation_with_accuracy(
        self,
        precursor: pd.DataFrame,
        fragments: pd.DataFrame,
        ms1_dict: Dict[str, Dict[str, Any]],
        acc_factor: float,
    ) -> float:
        """Helper method for MS1 correlation calculation."""
        precursor_mz = precursor["calcmass"].iloc[0] / precursor["charge"].iloc[0]
        elution_profile = self.build_elution_profile(
            precursor_mz, ms1_dict, acc_factor=acc_factor
        )

        if not elution_profile:
            return np.nan

        best_trace, smoothed_best_trace, _ = self.calculate_pearson_correlations(
            fragments
        )

        smoothed_best_df = pd.DataFrame(
            {"rt": best_trace.index, "smoothed_intensity": smoothed_best_trace}
        )

        elution_df = pd.DataFrame(
            list(elution_profile.items()), columns=["rt", "intensity"]
        )

        merged = pd.merge_asof(
            smoothed_best_df.sort_values("rt"),
            elution_df.sort_values("rt"),
            on="rt",
            direction="nearest",
            tolerance=self.config.rt_tolerance,
        )

        merged_clean = merged.dropna(subset=["smoothed_intensity", "intensity"])

        if len(merged_clean) < 2:
            return np.nan

        return merged_clean["smoothed_intensity"].corr(merged_clean["intensity"])

    # Feature Group 3: Isotopologue Co-elution
    def feature_c13_isotope_correlations(
        self,
        precursor: pd.DataFrame,
        fragments: pd.DataFrame,
        ms1_dict: Dict[str, Dict[str, Any]],
    ) -> np.ndarray:
        """
        Correlations with C13 isotope elution profiles.

        Parameters
        ----------
        precursor : pd.DataFrame
            Precursor information
        fragments : pd.DataFrame
            Fragment data
        ms1_dict : dict
            MS1 data dictionary

        Returns
        -------
        np.ndarray
            Array of correlations for C13 isotopes
        """
        try:
            best_trace, smoothed_best_trace, _ = self.calculate_pearson_correlations(
                fragments
            )

            smoothed_best_df = pd.DataFrame(
                {"rt": best_trace.index, "smoothed_intensity": smoothed_best_trace}
            )

            base_precursor_mz = (
                precursor["calcmass"].iloc[0] / precursor["charge"].iloc[0]
            )
            charge = precursor["charge"].iloc[0]

            correlations = []

            for c13_count in self.config.c13_isotope_list:
                isotope_mz = base_precursor_mz + (
                    c13_count * self.config.isotope_mass_c13 / charge
                )

                isotope_profile = self.build_elution_profile(isotope_mz, ms1_dict)

                if not isotope_profile:
                    correlations.append(np.nan)
                    continue

                isotope_df = pd.DataFrame(
                    list(isotope_profile.items()), columns=["rt", "isotope_intensity"]
                )

                merged = pd.merge_asof(
                    smoothed_best_df.sort_values("rt"),
                    isotope_df.sort_values("rt"),
                    on="rt",
                    direction="nearest",
                    tolerance=self.config.rt_tolerance,
                )

                merged_clean = merged.dropna(
                    subset=["smoothed_intensity", "isotope_intensity"]
                )

                if len(merged_clean) < 2:
                    correlations.append(np.nan)
                    continue

                corr = merged_clean["smoothed_intensity"].corr(
                    merged_clean["isotope_intensity"]
                )
                correlations.append(corr if not pd.isna(corr) else 0.0)

            return np.array(correlations)

        except Exception as e:
            logger.error(f"Error calculating C13 isotope correlations: {e}")
            return np.full(len(self.config.c13_isotope_list), np.nan)

    def feature_c13_subtracted_correlations(
        self, fragments: pd.DataFrame, ms2dict: Dict[str, Dict[str, Any]]
    ) -> np.ndarray:
        """
        Correlations for fragments with C13 mass subtracted.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        ms2dict : dict
            MS2 spectral data dictionary

        Returns
        -------
        np.ndarray
            Array of correlations for C13-subtracted fragments
        """
        try:
            best_trace, smoothed_best_trace, _ = self.calculate_pearson_correlations(
                fragments
            )

            smoothed_best_df = pd.DataFrame(
                {"rt": best_trace.index, "smoothed_intensity": smoothed_best_trace}
            )

            top_fragments = self.find_top_n_fragments(
                fragments, self.config.top_n_fragments
            )
            correlations = []

            for frag_name in top_fragments:
                try:
                    corr = self._calculate_c13_subtracted_correlation(
                        frag_name, fragments, smoothed_best_df, ms2dict
                    )
                    correlations.append(corr)
                except Exception as e:
                    logger.warning(f"Error processing fragment {frag_name}: {e}")
                    correlations.append(np.nan)

            # Pad with NaN if needed
            while len(correlations) < self.config.top_n_fragments:
                correlations.append(np.nan)

            return np.array(correlations[: self.config.top_n_fragments])

        except Exception as e:
            logger.error(f"Error calculating C13 subtracted correlations: {e}")
            return np.full(self.config.top_n_fragments, np.nan)

    def _calculate_c13_subtracted_correlation(
        self,
        frag_name: str,
        fragments: pd.DataFrame,
        smoothed_best_df: pd.DataFrame,
        ms2dict: Dict[str, Dict[str, Any]],
    ) -> float:
        """Helper method for C13 subtracted correlation calculation."""
        frag_data = fragments[fragments["fragment_names"] == frag_name]

        if frag_data.empty:
            return np.nan

        # Calculate target m/z with C13 mass subtracted
        original_mz = frag_data["fragment_mz_calculated"].iloc[0]
        charge = frag_data["fragment_charge"].iloc[0]
        c13_subtracted_mz = original_mz - (self.config.isotope_mass_c13 / charge)

        # Extract elution profile
        c13_subtracted_profile = {}

        for _, row in frag_data.iterrows():
            rt = row["rt"]
            scan = row["scannr"]

            scan_data = ms2dict.get(scan, {})
            mzs = scan_data.get("mz", [])
            intensities = scan_data.get("intensity", [])

            if len(mzs) == 0 or len(intensities) == 0:
                continue

            best_idx, best_val = self._search_sorted_with_tolerance(
                mzs, c13_subtracted_mz, self.config.fragment_mass_tolerance
            )

            if best_idx is not None:
                c13_subtracted_profile[rt] = intensities[best_idx]

        if not c13_subtracted_profile:
            return np.nan

        # Calculate correlation
        c13_df = pd.DataFrame(
            list(c13_subtracted_profile.items()),
            columns=["rt", "c13_subtracted_intensity"],
        )

        merged = pd.merge_asof(
            smoothed_best_df.sort_values("rt"),
            c13_df.sort_values("rt"),
            on="rt",
            direction="nearest",
            tolerance=self.config.rt_tolerance,
        )

        merged_clean = merged.dropna(
            subset=["smoothed_intensity", "c13_subtracted_intensity"]
        )

        if len(merged_clean) < 2:
            return np.nan

        corr = merged_clean["smoothed_intensity"].corr(
            merged_clean["c13_subtracted_intensity"]
        )

        return corr if not pd.isna(corr) else 0.0

    def feature_sum_c13_subtracted_correlations(
        self, fragments: pd.DataFrame, ms2dict: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Sum of C13 subtracted correlations.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        ms2dict : dict
            MS2 spectral data dictionary

        Returns
        -------
        float
            Sum of correlations
        """
        correlations = self.feature_c13_subtracted_correlations(fragments, ms2dict)
        return np.nansum(correlations)

    # Feature Group 4: Total Signal
    def feature_weighted_auc(self, fragments: pd.DataFrame) -> float:
        """
        Natural log of weighted AUC for top fragments.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data

        Returns
        -------
        float
            Natural log of weighted AUC
        """
        try:
            _, _, correlations = self.calculate_pearson_correlations(fragments)
            top_fragments = self.find_top_n_fragments(
                fragments, self.config.top_n_fragments
            )

            aucs = []
            corrs = []

            for frag in top_fragments:
                frag_data = fragments[fragments["fragment_names"] == frag]
                if frag_data.empty:
                    continue

                frag_data_sorted = frag_data.sort_values("rt")
                rt = frag_data_sorted["rt"].values
                intensity = frag_data_sorted["fragment_intensity"].values

                if len(rt) > 1:
                    auc = np.trapz(intensity, rt)
                    aucs.append(auc)
                    corrs.append(correlations.get(frag, 0.0))

            if not aucs:
                return np.nan

            aucs = np.array(aucs)
            corrs = np.nan_to_num(corrs, nan=0.0)

            weighted_aucs = aucs * corrs
            total_weighted_auc = np.sum(weighted_aucs)

            return np.log(total_weighted_auc) if total_weighted_auc > 0 else np.nan

        except Exception as e:
            logger.error(f"Error calculating weighted AUC: {e}")
            return np.nan

    # Feature Group 5: Fragment Intensities
    def feature_relative_intensities_top_6(self, fragments: pd.DataFrame) -> np.ndarray:
        """
        Relative intensities of top 6 fragments.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data

        Returns
        -------
        np.ndarray
            Array of relative intensities
        """
        try:
            top_fragments = self.find_top_n_fragments(
                fragments, self.config.top_n_fragments
            )

            intensities = []
            for frag in top_fragments:
                frag_data = fragments[fragments["fragment_names"] == frag]
                if not frag_data.empty:
                    max_intensity = frag_data["fragment_intensity"].max()
                    intensities.append(max_intensity)
                else:
                    intensities.append(0.0)

            # Pad with zeros if needed
            while len(intensities) < self.config.top_n_fragments:
                intensities.append(0.0)

            intensities = np.array(intensities[: self.config.top_n_fragments])

            # Normalize by maximum
            max_intensity = np.max(intensities)
            if max_intensity > 0:
                intensities = intensities / max_intensity

            return intensities

        except Exception as e:
            logger.error(f"Error calculating relative intensities: {e}")
            return np.zeros(self.config.top_n_fragments)

    # Feature Group 6: Mass Accuracy
    def feature_weighted_mass_accuracy(self, fragments: pd.DataFrame) -> np.ndarray:
        """
        Mass accuracy weighted by correlations.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data

        Returns
        -------
        np.ndarray
            Array of weighted mass accuracies
        """
        try:
            if "ppm_error" not in fragments.columns:
                logger.warning("ppm_error column missing")
                return np.full(self.config.top_n_fragments, np.nan)

            # Find apex RT
            apex_idx = fragments["fragment_intensity"].idxmax()
            apex_rt = fragments.loc[apex_idx, "rt"]

            # Get fragments at apex
            fragments_at_apex = fragments[fragments["rt"] == apex_rt]
            top_6_apex = fragments_at_apex.nlargest(
                self.config.top_n_fragments, "fragment_intensity"
            )

            if top_6_apex.empty:
                return np.full(self.config.top_n_fragments, np.nan)

            # Get correlations
            _, _, correlations = self.calculate_pearson_correlations(fragments)

            # Calculate weighted mass accuracies
            weighted_mass_accs = []
            for _, row in top_6_apex.iterrows():
                frag_name = row["fragment_names"]
                ppm_error = row["ppm_error"]
                correlation = correlations.get(frag_name, 0.0)

                if pd.isna(correlation):
                    correlation = 0.0

                weighted_mass_accs.append(ppm_error * correlation)

            # Pad with NaN if needed
            while len(weighted_mass_accs) < self.config.top_n_fragments:
                weighted_mass_accs.append(np.nan)

            return np.array(weighted_mass_accs[: self.config.top_n_fragments])

        except Exception as e:
            logger.error(f"Error calculating weighted mass accuracy: {e}")
            return np.full(self.config.top_n_fragments, np.nan)

    # Feature Group 7: Retention Time
    def feature_rt_apex(self, fragments: pd.DataFrame) -> float:
        """
        Retention time at intensity apex.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data

        Returns
        -------
        float
            RT at apex
        """
        try:
            fragments = self._validate_fragments_input(fragments)
            apex_idx = fragments["fragment_intensity"].idxmax()
            return fragments.loc[apex_idx, "rt"]
        except Exception as e:
            logger.error(f"Error calculating RT apex: {e}")
            return np.nan

    def feature_rt_prediction_difference(
        self, fragments: pd.DataFrame, rt_predictions: pd.DataFrame
    ) -> float:
        """
        Square root of RT prediction difference.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        rt_predictions : pd.DataFrame
            RT prediction data

        Returns
        -------
        float
            Square root of absolute RT difference
        """
        try:
            apex_rt = self.feature_rt_apex(fragments)

            if pd.isna(apex_rt):
                return np.nan

            peptide = fragments["peptide"].iloc[0]

            if (
                "peptide" not in rt_predictions.columns
                or "rt_predictions" not in rt_predictions.columns
            ):
                logger.warning("Required columns missing in RT predictions")
                return np.nan

            pred_data = rt_predictions[rt_predictions["peptide"] == peptide]

            if pred_data.empty:
                logger.warning(f"No RT prediction found for peptide {peptide}")
                return np.nan

            pred_rt = pred_data["rt_predictions"].iloc[0]

            return np.sqrt(abs(apex_rt - pred_rt))

        except Exception as e:
            logger.error(f"Error calculating RT prediction difference: {e}")
            return np.nan

    # Feature Group 8: Elution Profile Shape
    def feature_scanning_window_splits(
        self, fragments: pd.DataFrame, splits: int = 5
    ) -> np.ndarray:
        """
        Relative intensities in scanning window splits.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        splits : int, default=5
            Number of splits

        Returns
        -------
        np.ndarray
            Array of relative intensities for each split
        """
        try:
            best_fragment = self.find_best_fragment(fragments)
            frag_data = fragments[fragments["fragment_names"] == best_fragment]

            if frag_data.empty:
                return np.zeros(splits)

            rt_min = frag_data["rt"].min()
            rt_max = frag_data["rt"].max()
            rt_range = rt_max - rt_min

            if rt_range == 0:
                # All data at same RT
                result = np.zeros(splits)
                result[0] = 1.0  # Put all intensity in first bin
                return result

            split_size = rt_range / splits
            split_intensities = []

            for i in range(splits):
                split_min = rt_min + i * split_size
                split_max = rt_min + (i + 1) * split_size

                # Include max value in last split
                if i == splits - 1:
                    mask = (frag_data["rt"] >= split_min) & (
                        frag_data["rt"] <= split_max
                    )
                else:
                    mask = (frag_data["rt"] >= split_min) & (
                        frag_data["rt"] < split_max
                    )

                split_intensity = frag_data[mask]["fragment_intensity"].sum()
                split_intensities.append(split_intensity)

            # Normalize
            total_intensity = sum(split_intensities)
            if total_intensity > 0:
                split_intensities = [x / total_intensity for x in split_intensities]
            else:
                split_intensities = [1.0 / splits] * splits  # Equal distribution

            return np.array(split_intensities)

        except Exception as e:
            logger.error(f"Error calculating scanning window splits: {e}")
            return np.zeros(splits)

    # Feature Group 10: Library Characteristics
    def feature_relative_predicted_intensities(
        self,
        fragments: pd.DataFrame,
        predictions: Dict[str, Dict[str, float]],
        top_n: int = 12,
    ) -> np.ndarray:
        """
        Relative predicted intensities for fragments 2 to top_n.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        predictions : dict
            MS2PiP predictions
        top_n : int, default=12
            Number of top fragments to consider

        Returns
        -------
        np.ndarray
            Array of relative predicted intensities (excluding top fragment)
        """
        try:
            peptide = fragments["peptide"].iloc[0]
            charge = fragments["charge"].iloc[0]
            proforma = f"{peptide}/{charge}"

            if proforma not in predictions:
                logger.warning(f"No predictions found for {proforma}")
                return np.full(top_n - 1, np.nan)

            pred_ints = predictions[proforma]

            # Convert to Series and sort
            pred_series = pd.Series(pred_ints).sort_values(ascending=False)

            if len(pred_series) < 2:
                return np.full(top_n - 1, np.nan)

            # Take top_n and normalize by maximum
            top_preds = pred_series.head(top_n)
            max_intensity = top_preds.max()

            if max_intensity > 0:
                normalized = top_preds / max_intensity
            else:
                normalized = top_preds

            # Return all except first (exclude top fragment)
            result = normalized.iloc[1:].values

            # Pad with NaN if needed
            if len(result) < top_n - 1:
                padded = np.full(top_n - 1, np.nan)
                padded[: len(result)] = result
                result = padded

            return result[: top_n - 1]

        except Exception as e:
            logger.error(f"Error calculating relative predicted intensities: {e}")
            return np.full(top_n - 1, np.nan)

    def feature_precursor_mz(self, precursor: pd.DataFrame) -> float:
        """Calculate precursor m/z."""
        try:
            return precursor["calcmass"].iloc[0] / precursor["charge"].iloc[0]
        except Exception as e:
            logger.error(f"Error calculating precursor m/z: {e}")
            return np.nan

    def feature_precursor_charge(self, precursor: pd.DataFrame) -> int:
        """Get precursor charge."""
        try:
            return int(precursor["charge"].iloc[0])
        except Exception as e:
            logger.error(f"Error getting precursor charge: {e}")
            return -1

    def feature_precursor_length(self, precursor: pd.DataFrame) -> int:
        """Calculate precursor peptide length."""
        try:
            return len(precursor["stripped_peptide"].iloc[0])
        except Exception as e:
            logger.error(f"Error calculating precursor length: {e}")
            return -1

    def feature_library_fragment_count(
        self, precursor: pd.DataFrame, predictions: Dict[str, Dict[str, float]]
    ) -> int:
        """Count library fragments with intensity > 0."""
        try:
            peptide = precursor["peptide"].iloc[0]
            charge = precursor["charge"].iloc[0]
            proforma = f"{peptide}/{charge}"

            if proforma not in predictions:
                return 0

            pred_ints = predictions[proforma]
            return len([v for v in pred_ints.values() if v > 0])

        except Exception as e:
            logger.error(f"Error counting library fragments: {e}")
            return 0

    def calculate_all_features(
        self,
        precursor: pd.DataFrame,
        fragments: pd.DataFrame,
        ms1_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        ms2dict: Optional[Dict[str, Dict[str, Any]]] = None,
        rt_predictions: Optional[pd.DataFrame] = None,
        intensity_predictions: Optional[Dict[str, Dict[str, float]]] = None,
        parallel: bool = True,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate all available features.

        Parameters
        ----------
        precursor : pd.DataFrame
            Precursor information
        fragments : pd.DataFrame
            Fragment data
        ms1_dict : dict, optional
            MS1 data dictionary
        ms2dict : dict, optional
            MS2 data dictionary
        rt_predictions : pd.DataFrame, optional
            RT prediction data
        intensity_predictions : dict, optional
            Intensity prediction data
        parallel : bool, default True
            Whether to use parallel processing for feature calculation

        Returns
        -------
        Dict[str, Union[float, np.ndarray]]
            Dictionary of calculated features
        """
        if parallel and self.n_workers > 1:
            return self._calculate_all_features_parallel(
                precursor, fragments, ms1_dict, ms2dict, rt_predictions, intensity_predictions
            )
        else:
            return self._calculate_all_features_sequential(
                precursor, fragments, ms1_dict, ms2dict, rt_predictions, intensity_predictions
            )

    def _calculate_all_features_parallel(
        self,
        precursor: pd.DataFrame,
        fragments: pd.DataFrame,
        ms1_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        ms2dict: Optional[Dict[str, Dict[str, Any]]] = None,
        rt_predictions: Optional[pd.DataFrame] = None,
        intensity_predictions: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate all features using parallel processing.
        """
        features = {}
        
        # Define feature calculation tasks
        tasks = []
        
        # Group 1: Ion co-elution (MS2 level) - can be parallelized
        tasks.extend([
            ("pearson_correlations_top_12", self.feature_pearson_correlations_top_n, (fragments,)),
            ("sum_correlations_mass_accuracy", self.feature_sum_correlations_mass_accuracy, (fragments,)),
            ("remaining_fragments_correlations", self.feature_remaining_fragments_correlations, (fragments,)),
            ("best_b_fragments_correlation", self.feature_best_b_fragments_correlation, (fragments,)),
        ])
        
        if ms1_dict is not None:
            tasks.append(("precursor_best_fragment_correlation", 
                         self.feature_precursor_best_fragment_correlation, 
                         (precursor, fragments, ms1_dict)))
            
            # Group 2: MS1 level co-elution
            tasks.append(("ms1_accuracy_correlations", 
                         self.feature_ms1_accuracy_correlations, 
                         (precursor, fragments, ms1_dict)))
            
            # Group 3: Isotopologue co-elution
            tasks.append(("c13_isotope_correlations", 
                         self.feature_c13_isotope_correlations, 
                         (precursor, fragments, ms1_dict)))
        
        if ms2dict is not None:
            tasks.extend([
                ("c13_subtracted_correlations", self.feature_c13_subtracted_correlations, (fragments, ms2dict)),
                ("sum_c13_subtracted_correlations", self.feature_sum_c13_subtracted_correlations, (fragments, ms2dict)),
            ])
        
        # Group 4-10: Other features
        tasks.extend([
            ("weighted_auc", self.feature_weighted_auc, (fragments,)),
            ("relative_intensities_top_6", self.feature_relative_intensities_top_6, (fragments,)),
            ("weighted_mass_accuracy", self.feature_weighted_mass_accuracy, (fragments,)),
            ("rt_apex", self.feature_rt_apex, (fragments,)),
            ("scanning_window_splits", self.feature_scanning_window_splits, (fragments,)),
            ("precursor_mz", self.feature_precursor_mz, (precursor,)),
            ("precursor_charge", self.feature_precursor_charge, (precursor,)),
            ("precursor_length", self.feature_precursor_length, (precursor,)),
        ])
        
        if rt_predictions is not None:
            tasks.append(("rt_prediction_difference", 
                         self.feature_rt_prediction_difference, 
                         (fragments, rt_predictions)))
        
        if intensity_predictions is not None:
            tasks.extend([
                ("relative_predicted_intensities", 
                 self.feature_relative_predicted_intensities, 
                 (fragments, intensity_predictions)),
                ("library_fragment_count", 
                 self.feature_library_fragment_count, 
                 (precursor, intensity_predictions)),
            ])
        
        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_name = {}
            for feature_name, func, args in tasks:
                future = executor.submit(self._safe_feature_calculation, func, args)
                future_to_name[future] = feature_name
            
            # Collect results as they complete
            for future in as_completed(future_to_name):
                feature_name = future_to_name[future]
                try:
                    result = future.result()
                    if result is not None:
                        features[feature_name] = result
                except Exception as e:
                    logger.error(f"Error calculating {feature_name}: {e}")
        
        logger.info(f"Calculated {len(features)} feature groups in parallel")
        return features

    def _safe_feature_calculation(self, func, args):
        """Safely execute feature calculation with error handling."""
        try:
            return func(*args)
        except Exception as e:
            logger.error(f"Error in feature calculation: {e}")
            return None

    def _calculate_all_features_sequential(
        self,
        precursor: pd.DataFrame,
        fragments: pd.DataFrame,
        ms1_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        ms2dict: Optional[Dict[str, Dict[str, Any]]] = None,
        rt_predictions: Optional[pd.DataFrame] = None,
        intensity_predictions: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate all available features.

        Parameters
        ----------
        precursor : pd.DataFrame
            Precursor information
        fragments : pd.DataFrame
            Fragment data
        ms1_dict : dict, optional
            MS1 data dictionary
        ms2dict : dict, optional
            MS2 data dictionary
        rt_predictions : pd.DataFrame, optional
            RT prediction data
        intensity_predictions : dict, optional
            Intensity prediction data

        Returns
        -------
        Dict[str, Union[float, np.ndarray]]
            Dictionary of calculated features
        """
        features = {}

        logger.info("Calculating DIA-NN features...")

        # Group 1: Ion co-elution (MS2 level)
        try:
            features["pearson_correlations_top_12"] = (
                self.feature_pearson_correlations_top_n(fragments)
            )
            features["sum_correlations_mass_accuracy"] = (
                self.feature_sum_correlations_mass_accuracy(fragments)
            )
            features["remaining_fragments_correlations"] = (
                self.feature_remaining_fragments_correlations(fragments)
            )
            features["best_b_fragments_correlation"] = (
                self.feature_best_b_fragments_correlation(fragments)
            )

            if ms1_dict is not None:
                features["precursor_best_fragment_correlation"] = (
                    self.feature_precursor_best_fragment_correlation(
                        precursor, fragments, ms1_dict
                    )
                )
        except Exception as e:
            logger.error(f"Error in group 1 features: {e}")

        # Group 2: MS1 level co-elution
        if ms1_dict is not None:
            try:
                features["ms1_accuracy_correlations"] = (
                    self.feature_ms1_accuracy_correlations(
                        precursor, fragments, ms1_dict
                    )
                )
            except Exception as e:
                logger.error(f"Error in group 2 features: {e}")

        # Group 3: Isotopologue co-elution
        if ms1_dict is not None:
            try:
                features["c13_isotope_correlations"] = (
                    self.feature_c13_isotope_correlations(
                        precursor, fragments, ms1_dict
                    )
                )
            except Exception as e:
                logger.error(f"Error in group 3.1 features: {e}")

        if ms2dict is not None:
            try:
                features["c13_subtracted_correlations"] = (
                    self.feature_c13_subtracted_correlations(fragments, ms2dict)
                )
                features["sum_c13_subtracted_correlations"] = (
                    self.feature_sum_c13_subtracted_correlations(fragments, ms2dict)
                )
            except Exception as e:
                logger.error(f"Error in group 3.3-3.4 features: {e}")

        # Group 4: Total signal
        try:
            features["weighted_auc"] = self.feature_weighted_auc(fragments)
        except Exception as e:
            logger.error(f"Error in group 4 features: {e}")

        # Group 5: Fragment intensities
        try:
            features["relative_intensities_top_6"] = (
                self.feature_relative_intensities_top_6(fragments)
            )
        except Exception as e:
            logger.error(f"Error in group 5 features: {e}")

        # Group 6: Mass accuracy
        try:
            features["weighted_mass_accuracy"] = self.feature_weighted_mass_accuracy(
                fragments
            )
        except Exception as e:
            logger.error(f"Error in group 6 features: {e}")

        # Group 7: Retention time
        try:
            features["rt_apex"] = self.feature_rt_apex(fragments)
            if rt_predictions is not None:
                features["rt_prediction_difference"] = (
                    self.feature_rt_prediction_difference(fragments, rt_predictions)
                )
        except Exception as e:
            logger.error(f"Error in group 7 features: {e}")

        # Group 8: Elution profile shape
        try:
            features["scanning_window_splits"] = self.feature_scanning_window_splits(
                fragments
            )
        except Exception as e:
            logger.error(f"Error in group 8 features: {e}")

        # Group 10: Library characteristics
        try:
            features["precursor_mz"] = self.feature_precursor_mz(precursor)
            features["precursor_charge"] = self.feature_precursor_charge(precursor)
            features["precursor_length"] = self.feature_precursor_length(precursor)

            if intensity_predictions is not None:
                features["relative_predicted_intensities"] = (
                    self.feature_relative_predicted_intensities(
                        fragments, intensity_predictions
                    )
                )
                features["library_fragment_count"] = (
                    self.feature_library_fragment_count(
                        precursor, intensity_predictions
                    )
                )
        except Exception as e:
            logger.error(f"Error in group 10 features: {e}")

        logger.info(f"Calculated {len(features)} feature groups")
        return features


def main():
    """Example usage of the DIANNFeatureGenerator."""

    # Create configuration
    config = FeatureConfig(
        fragment_mass_tolerance=15.0, rt_tolerance=3.0, top_n_fragments=6
    )

    # Initialize generator
    generator = DIANNFeatureGenerator(config)

    # Example data (replace with actual data loading)
    print("DIANNFeatureGenerator initialized successfully!")
    print(f"Configuration: {config}")

    # Example feature calculation would go here with real data
    # features = generator.calculate_all_features(precursor, fragments, ...)

    return generator


if __name__ == "__main__":
    main()
