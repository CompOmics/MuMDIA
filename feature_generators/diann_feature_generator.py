"""
DIA-NN Feature Generator for Proteomics Analysis

This module implements the DIANNFeatureGenerator class for calculating comprehensive
features from MS/MS proteomics data for use in machine learning models. The features
are based on fragment elution profiles, correlations, and spectral library predictions.

Author: Generated from Jupyter notebook analysis
Date: August 2025
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Silence DIA-NN feature generator logging — too verbose for per-peptidoform calls
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


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

    # Feature toggles
    enable_ms1_features: bool = (
        False  # MS1-based features (groups 2-3); slow, disabled by default
    )

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
        Initialize the feature generator with built-in optimizations.

        Parameters
        ----------
        config : FeatureConfig, optional
            Configuration object with parameters. If None, uses defaults.
        """
        self.config = config if config is not None else FeatureConfig()
        self._validate_config()
        self._setup_parallelization()

        # Initialize optimization caches
        self._cache = {}
        self._pivot_cache = {}
        self._correlation_cache = {}

        # Pre-processed MS1 data (set via prepare_ms1_dict)
        self._ms1_prepared = (
            None  # list of (rt, mz_array, intensity_array) sorted by RT
        )

        logger.info("Initialized DIANNFeatureGenerator with built-in optimizations")

    def prepare_ms1_dict(self, ms1_dict: Dict[str, Dict[str, Any]]) -> None:
        """Pre-convert ms1_dict to sorted numpy arrays for fast elution profile building.

        Call once before processing peptidoforms. Converts each scan's mz/intensity
        lists to numpy arrays and sorts the scan list by RT. This avoids repeated
        np.asarray + sort checks in build_elution_profile (~20ms -> ~2ms per call).
        """
        prepared = []
        for scan_dict in ms1_dict.values():
            mzs = scan_dict.get("mz", [])
            intensities = scan_dict.get("intensity", [])
            rt = scan_dict.get("retention_time", None)
            if rt is None or len(mzs) == 0:
                continue
            # Convert RT from seconds to minutes if needed
            if isinstance(rt, (int, float)) and rt > 1000:
                rt = rt / 60
            mz_arr = np.asarray(mzs)
            int_arr = np.asarray(intensities)
            # Ensure sorted by m/z
            if len(mz_arr) > 1 and mz_arr[0] > mz_arr[-1]:
                order = np.argsort(mz_arr)
                mz_arr = mz_arr[order]
                int_arr = int_arr[order]
            prepared.append((rt, mz_arr, int_arr))
        # Sort by RT for potential windowed access
        prepared.sort(key=lambda x: x[0])
        self._ms1_prepared = prepared

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

    def _get_cache_key(self, fragments: pd.DataFrame) -> str:
        """Generate cache key for fragments DataFrame."""
        try:
            rt_min, rt_max = fragments["rt"].min(), fragments["rt"].max()
            frag_names = sorted(fragments["fragment_names"].unique())
            return f"{len(fragments)}_{rt_min:.3f}_{rt_max:.3f}_{hash(tuple(frag_names[:10]))}"
        except Exception:
            return str(hash(str(fragments.shape)))

    def _get_or_create_pivot_table(self, fragments: pd.DataFrame) -> pd.DataFrame:
        """Get or create pivot table with caching."""
        cache_key = f"pivot_{self._get_cache_key(fragments)}"

        if cache_key in self._pivot_cache:
            return self._pivot_cache[cache_key]

        # Create pivot table efficiently
        pivot_table = fragments.pivot_table(
            index="rt",
            columns="fragment_names",
            values="fragment_intensity",
            aggfunc="mean",
        )

        # Cache the result
        self._pivot_cache[cache_key] = pivot_table
        return pivot_table

    def clear_cache(self):
        """Clear all caches to free memory."""
        self._cache.clear()
        self._pivot_cache.clear()
        self._correlation_cache.clear()
        logger.debug("Cleared all caches")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        return {
            "main_cache_size": len(self._cache),
            "pivot_cache_size": len(self._pivot_cache),
            "correlation_cache_size": len(self._correlation_cache),
        }

    def _validate_fragments_input(self, fragments: pd.DataFrame) -> pd.DataFrame:
        """
        Optimized fragment validation with caching.

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
        if fragments.empty:
            raise ValueError("Fragment data is empty")

        required_cols = ["fragment_names", "rt", "fragment_intensity"]
        missing_cols = set(required_cols) - set(fragments.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check if already cleaned (cache check)
        cache_key = f"validated_{self._get_cache_key(fragments)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Clean data efficiently
        fragments = fragments.copy()

        # Vectorized numeric conversion
        fragments["rt"] = pd.to_numeric(fragments["rt"], errors="coerce")
        fragments["fragment_intensity"] = pd.to_numeric(
            fragments["fragment_intensity"], errors="coerce"
        )

        # Remove invalid data in one operation
        initial_size = len(fragments)
        valid_mask = fragments["rt"].notna() & fragments["fragment_intensity"].notna()
        fragments = fragments[valid_mask]

        if fragments.empty:
            raise ValueError("No valid fragment data after cleaning")

        if len(fragments) < initial_size:
            logger.debug(f"Removed {initial_size - len(fragments)} invalid rows")

        # Cache the result
        self._cache[cache_key] = fragments
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
        Optimized top fragment finding with caching.

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

        cache_key = f"top_frags_{n}_{self._get_cache_key(fragments)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        fragments = self._validate_fragments_input(fragments)

        if "fragment_names" not in fragments.columns:
            raise ValueError("Missing 'fragment_names' column")

        # Vectorized groupby operation
        top_fragments = (
            fragments.groupby("fragment_names", sort=False)["fragment_intensity"]
            .max()
            .nlargest(n)
            .index.tolist()
        )

        # Cache the result
        self._cache[cache_key] = top_fragments
        return top_fragments

    def find_best_fragment(self, fragments: pd.DataFrame) -> str:
        """
        Optimized best fragment finding with caching and vectorized correlation calculation.

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
        cache_key = f"best_frag_{self._get_cache_key(fragments)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        fragments = self._validate_fragments_input(fragments)
        top_fragments = self.find_top_n_fragments(
            fragments, self.config.top_n_fragments
        )

        if len(top_fragments) < 2:
            logger.warning("Less than 2 fragments available, using first fragment")
            best = top_fragments[0] if top_fragments else None
            self._cache[cache_key] = best
            return best

        # Filter to top fragments efficiently
        top_fragment_set = set(top_fragments)
        filtered_fragments = fragments[
            fragments["fragment_names"].isin(top_fragment_set)
        ]

        # Create pivot table
        pivot_table = self._get_or_create_pivot_table(filtered_fragments)

        if pivot_table.empty:
            best = top_fragments[0]
            self._cache[cache_key] = best
            return best

        # Calculate correlations efficiently using numpy
        correlation_sums = {}

        # Convert to numpy for faster computation
        pivot_values = pivot_table.values
        fragment_names = pivot_table.columns.tolist()

        # Only compute correlations for top fragments
        top_indices = [
            i for i, name in enumerate(fragment_names) if name in top_fragments
        ]

        if len(top_indices) < 2:
            best = top_fragments[0]
            self._cache[cache_key] = best
            return best

        # Compute correlation matrix for top fragments only
        top_values = pivot_values[:, top_indices]
        correlation_matrix = np.corrcoef(top_values.T, rowvar=True)
        correlation_matrix = np.nan_to_num(correlation_matrix, 0.0)

        # Sum correlations for each fragment
        for i, global_idx in enumerate(top_indices):
            frag_name = fragment_names[global_idx]
            correlation_sums[frag_name] = (
                np.sum(correlation_matrix[i]) - 1.0
            )  # Subtract self-correlation

        # Find best fragment
        if correlation_sums:
            best = max(correlation_sums.items(), key=lambda x: x[1])[0]
        else:
            best = top_fragments[0]

        # Cache the result
        self._cache[cache_key] = best
        return best

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
        visualize: bool = False,
    ) -> Tuple[pd.Series, np.ndarray, Dict[str, float]]:
        """
        Optimized Pearson correlation calculation with caching.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        best_fragment : str, optional
            Name of best fragment. If None, will be determined automatically.
        use_all_rt : bool, default=False
            Whether to use all RT points (filling missing with 0) or only overlapping points
        visualize : bool, default=False
            If True, render diagnostic plots

        Returns
        -------
        Tuple[pd.Series, np.ndarray, Dict[str, float]]
            Best fragment trace, smoothed trace, and correlation dictionary
        """
        cache_key = f"correlations_{self._get_cache_key(fragments)}_{best_fragment}_{use_all_rt}"

        if not visualize and cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]

        fragments = self._validate_fragments_input(fragments)

        if best_fragment is None:
            best_fragment = self.find_best_fragment(fragments)

        # Get or create pivot table
        pivot_table = self._get_or_create_pivot_table(fragments)

        if best_fragment not in pivot_table.columns:
            raise ValueError(f"Best fragment '{best_fragment}' not found in data")

        # Get best fragment trace and smooth it
        best_trace = pivot_table[best_fragment].dropna()
        if len(best_trace) == 0:
            raise ValueError("Best fragment has no valid data points")

        smoothed_best_trace = self._apply_savgol_smoothing(best_trace.values)

        # Calculate correlations efficiently
        correlations = {}

        if use_all_rt:
            # Fill missing values with 0 and use all RT points
            pivot_filled = pivot_table.fillna(0.0)
            best_trace_filled = pivot_filled[best_fragment]
            smoothed_best_filled = self._apply_savgol_smoothing(
                best_trace_filled.values
            )

            # Vectorized correlation calculation
            smoothed_series = pd.Series(smoothed_best_filled, index=pivot_filled.index)

            # Batch correlation calculation
            for frag in pivot_table.columns:
                frag_trace = pivot_filled[frag]
                if len(frag_trace) < 2:
                    correlations[frag] = np.nan
                    continue

                # Use pandas built-in correlation which is optimized
                corr = smoothed_series.corr(frag_trace)
                correlations[frag] = corr if not pd.isna(corr) else 0.0
        else:
            # Use only overlapping RT points
            best_smoothed_series = pd.Series(
                smoothed_best_trace, index=best_trace.index
            )

            # Pre-compute common indices for efficiency
            best_index_set = set(best_trace.index)

            for frag in pivot_table.columns:
                frag_trace = pivot_table[frag].dropna()

                # Fast intersection using set operations
                frag_index_set = set(frag_trace.index)
                common_indices = best_index_set & frag_index_set

                if len(common_indices) < 2:
                    correlations[frag] = np.nan
                    continue

                # Convert back to sorted index for pandas
                common_index = sorted(common_indices)

                # Use optimized pandas correlation
                corr = best_smoothed_series.loc[common_index].corr(
                    frag_trace.loc[common_index]
                )
                correlations[frag] = corr if not pd.isna(corr) else 0.0

        # For visualization alignment
        vis_index = best_trace.index
        smoothed_for_vis = smoothed_best_trace

        # -------------------- Visualization (optional) --------------------
        if visualize:
            # Prepare correlation series
            corr_s = pd.Series(correlations).dropna().sort_values(ascending=False)

            # Choose top-6 by correlation for overlay (fallback to available)
            top_k = 6
            top_frags = corr_s.index[:top_k].tolist()

            # Build a matrix of fragment traces aligned to vis_index
            # (If use_all_rt=False, traces outside vis_index become NaN -> fill 0)
            aligned = (
                pivot_table.reindex(index=vis_index)
                .reindex(columns=top_frags)
                .fillna(0.0)
            )

            # Normalize each trace to [0,1] for fair overlay
            def _minmax(x: pd.Series) -> pd.Series:
                x = x.astype(float)
                rng = float(x.max() - x.min())
                return (x - x.min()) / (rng if rng > 0 else 1.0)

            aligned_norm = aligned.apply(_minmax, axis=0)
            smoothed_norm = (smoothed_for_vis - np.min(smoothed_for_vis)) / (
                (np.max(smoothed_for_vis) - np.min(smoothed_for_vis)) or 1.0
            )

            # Create figure with three panels
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # (i) Best vs. smoothed best
            axes[0].plot(
                best_trace.index, best_trace.values, label="Best fragment (raw)"
            )
            axes[0].plot(vis_index, smoothed_for_vis, label="Best fragment (smoothed)")
            axes[0].set_title("Best fragment trace")
            axes[0].set_xlabel("RT")
            axes[0].set_ylabel("Intensity")
            axes[0].legend()

            # (ii) Overlay of top-6 normalized traces vs. smoothed best
            for col in aligned_norm.columns:
                axes[1].plot(vis_index, aligned_norm[col].values, alpha=0.8, label=col)
            axes[1].plot(
                vis_index,
                smoothed_norm,
                linestyle="--",
                linewidth=2,
                label="Smoothed best (norm)",
            )
            axes[1].set_title("Top-6 fragment traces (normalized)")
            axes[1].set_xlabel("RT")
            axes[1].set_ylabel("Relative intensity")
            axes[1].legend(fontsize=8, ncol=1, loc="upper right")

            # (iii) Correlation bar chart
            axes[2].barh(corr_s.index[::-1], corr_s.values[::-1])
            axes[2].set_xlim(-1.0, 1.0)
            axes[2].set_title("Pearson correlations vs. smoothed best")
            axes[2].set_xlabel("r")
            axes[2].set_ylabel("Fragment")

            plt.tight_layout()
            plt.show()

            # -----------------------------------------------------------------

        result = (best_trace, smoothed_best_trace, correlations)

        # Cache if not visualizing
        if not visualize:
            self._correlation_cache[cache_key] = result

        return result

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

        tol_mz = target_mz * tolerance_ppm / 1e6 * acc_factor

        # Fast path: use pre-processed arrays (avoids np.asarray + sort per scan)
        if self._ms1_prepared is not None:
            elution_profile = {}
            for rt, mz_arr, int_arr in self._ms1_prepared:
                idx = np.searchsorted(mz_arr, target_mz)
                best_idx = None
                best_diff = tol_mz
                for check_idx in (idx - 1, idx, idx + 1):
                    if 0 <= check_idx < len(mz_arr):
                        diff = abs(mz_arr[check_idx] - target_mz)
                        if diff < best_diff:
                            best_diff = diff
                            best_idx = check_idx
                if best_idx is not None:
                    elution_profile[rt] = int_arr[best_idx]
            return elution_profile

        # Slow fallback: original dict-based path
        elution_profile = {}
        for scan, scan_dict in ms1_dict.items():
            mzs = scan_dict.get("mz", [])
            intensities = scan_dict.get("intensity", [])
            rt = scan_dict.get("retention_time", None)

            if rt is None or len(mzs) == 0 or len(intensities) == 0:
                continue

            if isinstance(rt, (int, float)) and rt > 1000:
                rt = rt / 60

            best_idx, best_val = self._search_sorted_with_tolerance(
                mzs, target_mz, tol_mz
            )

            if best_idx is not None:
                elution_profile[rt] = intensities[best_idx]

        return elution_profile

    # Feature Group 1: Ion Co-elution (MS2 level)

    def feature_pearson_correlations_top_n(
        self,
        fragments: pd.DataFrame,
        n: Optional[int] = None,
        visualize: bool = False,
    ) -> np.ndarray:
        """
        Calculate Pearson correlations of the top-n fragments with the smoothed
        elution profile of the best fragment. Optionally visualize the results.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data. Must contain at least:
            ['rt', 'fragment_names', 'fragment_intensity'].
        n : int, optional
            Number of top fragments. Uses `self.config.top_n_fragments_extended`
            if None.
        visualize : bool, default=False
            If True, render:
            (i) a bar chart of the top-n correlations, and
            (ii) an overlay of the top-n (normalized) traces vs. the smoothed best.

        Returns
        -------
        np.ndarray
            Array (length n) with Pearson correlation coefficients for the selected
            top-n fragments (NaN-padded if fewer than n are available).
        """
        if n is None:
            n = self.config.top_n_fragments_extended

        fragments = self._validate_fragments_input(fragments)

        try:
            # Avoid double plotting here: get correlations without visualization,
            # then perform visualization specific to this "top-n" feature below.
            (
                best_trace,
                smoothed_best_trace,
                correlations,
            ) = self.calculate_pearson_correlations(fragments, visualize=False)

            # Determine the top-n fragments according to your internal criterion
            top_fragments = self.find_top_n_fragments(fragments, n)

            # Build the fixed-size result vector (NaN-padded)
            result = np.full(n, np.nan, dtype=float)
            for i, frag in enumerate(top_fragments[:n]):
                if frag in correlations:
                    result[i] = correlations[frag]

            # -------------------- Visualization (optional) --------------------
            if visualize:
                # 1) Bar chart of correlations for the chosen top-n (in the chosen order)
                labels = top_fragments[:n]
                vals = [correlations.get(f, np.nan) for f in labels]

                # 2) Overlay of normalized traces for the top-n vs. smoothed best
                #    Align everything to the best_trace index (RT grid).
                pivot = (
                    fragments.pivot_table(
                        index="rt",
                        columns="fragment_names",
                        values="fragment_intensity",
                        aggfunc="mean",
                    )
                    .reindex(best_trace.index)  # align rows to best fragment RTs
                    .reindex(columns=labels)  # keep only selected fragments
                    .fillna(0.0)
                )

                # Per-fragment min–max normalization for display
                def _minmax(x: pd.Series) -> pd.Series:
                    x = x.astype(float)
                    rng = float(x.max() - x.min())
                    return (x - x.min()) / (rng if rng > 0 else 1.0)

                pivot_norm = pivot.apply(_minmax, axis=0)

                # Normalize smoothed best for overlay
                sb = np.asarray(smoothed_best_trace, float)
                sb_norm = (sb - sb.min()) / (
                    sb.max() - sb.min() if sb.max() > sb.min() else 1.0
                )

                fig, axes = plt.subplots(1, 2, figsize=(13, 4))

                # (i) Bar chart of correlations
                axes[0].barh(labels[::-1], np.asarray(vals, dtype=float)[::-1])
                axes[0].set_xlim(-1.05, 1.05)
                axes[0].axvline(0.0, linestyle="--", linewidth=1)
                axes[0].set_title(
                    f"Top-{len(labels)} fragment correlations (Pearson r)"
                )
                axes[0].set_xlabel("r")
                axes[0].set_ylabel("Fragment")

                # (ii) Overlay of normalized traces vs. smoothed best
                for col in pivot_norm.columns:
                    axes[1].plot(
                        best_trace.index, pivot_norm[col].values, alpha=0.8, label=col
                    )
                axes[1].plot(
                    best_trace.index,
                    sb_norm,
                    linestyle="--",
                    linewidth=2,
                    label="Smoothed best (norm)",
                )
                axes[1].set_title("Top-n fragment chromatograms (normalized)")
                axes[1].set_xlabel("RT")
                axes[1].set_ylabel("Relative intensity")
                axes[1].legend(fontsize=8, ncol=1, loc="upper right")

                plt.tight_layout()
                plt.show()
            # -----------------------------------------------------------------

            return result

        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return np.full(n, np.nan, dtype=float)

    def feature_sum_correlations_mass_accuracy(
        self,
        fragments: pd.DataFrame,
        visualize: bool = False,  # <-- NEW toggle
    ) -> np.ndarray:
        """
        Sum of correlations for top-N fragments at different mass-accuracy factors,
        with an optional visualization.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data. Must include a 'ppm_error' column to enable filtering.
        visualize : bool, default=False
            If True, renders:
            (i) a line/bar chart of correlation sums vs. mass-accuracy factor, and
            (ii) a boxplot of per-factor correlation distributions (top-N).

        Returns
        -------
        np.ndarray
            Array of correlation sums for each mass-accuracy factor in
            `self.config.ms2_accuracy_factors` (NaN for failures/empty).
        """
        # --- Configuration guards ---
        if not hasattr(self, "config") or not hasattr(
            self.config, "ms2_accuracy_factors"
        ):
            raise AttributeError(
                "self.config.ms2_accuracy_factors is required but not found."
            )
        if not hasattr(self.config, "fragment_mass_tolerance"):
            raise AttributeError(
                "self.config.fragment_mass_tolerance is required but not found."
            )
        if not hasattr(self.config, "top_n_fragments"):
            raise AttributeError(
                "self.config.top_n_fragments is required but not found."
            )

        if "ppm_error" not in fragments.columns:
            logger.warning("ppm_error column missing, skipping mass accuracy filtering")
            # Preserve output shape
            out = np.array([np.nan] * len(self.config.ms2_accuracy_factors))
            if visualize:
                plt.figure(figsize=(7, 3))
                plt.plot(self.config.ms2_accuracy_factors, out, marker="o")
                plt.title(
                    "Correlation sums vs. mass-accuracy factor (ppm_error missing)"
                )
                plt.xlabel("Mass-accuracy factor")
                plt.ylabel("Sum of correlations (top-N)")
                plt.tight_layout()
                plt.show()
            return out

        # --- Main computation loop ---
        factors = list(self.config.ms2_accuracy_factors)
        results: list[float] = []
        per_factor_corrs: list[np.ndarray] = []  # for visualization (boxplot)

        # CRITICAL FIX: Determine the best fragment from the FULL dataset first
        # to ensure we use the same reference fragment for all mass accuracy factors
        try:
            best_fragment = self.find_best_fragment(fragments)
        except Exception as e:
            logger.error(f"Error finding best fragment: {e}")
            # Return all NaN if we can't find a best fragment
            results_arr = np.array([np.nan] * len(factors))
            if visualize:
                plt.figure(figsize=(7, 3))
                plt.plot(factors, results_arr, marker="o")
                plt.title(
                    "Correlation sums vs. mass-accuracy factor (no best fragment)"
                )
                plt.xlabel("Mass-accuracy factor")
                plt.ylabel("Sum of correlations (top-N)")
                plt.tight_layout()
                plt.show()
            return results_arr

        for factor in factors:
            try:
                # Filter by (absolute) ppm tolerance scaled by factor
                tol_ppm = float(self.config.fragment_mass_tolerance) * float(factor)
                filtered = fragments[fragments["ppm_error"] <= tol_ppm]

                if filtered.empty:
                    results.append(np.nan)
                    per_factor_corrs.append(np.array([np.nan]))
                    continue

                # CRITICAL FIX: Use the pre-determined best fragment, but fall back if it's not available
                # Check if the best fragment survived the filtering
                fragment_names_in_filtered = filtered["fragment_names"].unique()

                if best_fragment in fragment_names_in_filtered:
                    # Best case: use the original best fragment
                    reference_fragment = best_fragment
                else:
                    # Fallback: find the best fragment from the filtered data
                    # This maintains consistency within this mass accuracy level
                    try:
                        reference_fragment = self.find_best_fragment(filtered)
                        logger.warning(
                            f"Best fragment '{best_fragment}' not found in filtered data "
                            f"for factor {factor}, using '{reference_fragment}' instead"
                        )
                    except Exception as e:
                        logger.error(
                            f"Could not find any best fragment in filtered data for factor {factor}: {e}"
                        )
                        results.append(np.nan)
                        per_factor_corrs.append(np.array([np.nan]))
                        continue

                # Calculate correlations using the determined reference fragment
                (
                    _,
                    _,
                    correlations,
                ) = self.calculate_pearson_correlations(  # TODO: Use all RT???
                    filtered, best_fragment=reference_fragment
                )

                # Get top N fragments from the filtered data
                top_fragments = self.find_top_n_fragments(
                    filtered, self.config.top_n_fragments
                )

                # Calculate correlations for top N fragments, padding with NaN as needed
                corrs = np.full(self.config.top_n_fragments, np.nan)
                for i, frag in enumerate(top_fragments[: self.config.top_n_fragments]):
                    if frag in correlations:
                        corrs[i] = correlations[frag]

                # Store individual correlations for visualization and the sum for the feature
                per_factor_corrs.append(np.asarray(corrs, dtype=float))
                results.append(np.nansum(corrs))

            except Exception as e:
                logger.error(f"Error with mass accuracy factor {factor}: {e}")
                results.append(np.nan)
                per_factor_corrs.append(np.array([np.nan]))

        results_arr = np.asarray(results, dtype=float)

        # --- Optional visualization ---
        if visualize:
            # Figure 1: correlation sums vs factor
            fig, ax = plt.subplots(1, 2, figsize=(13, 4))

            # Left: line/marker plot (handles NaNs gracefully)
            ax[0].plot(factors, results_arr, marker="o")
            ax[0].set_title(
                "Sum of fragment–reference correlations\nacross mass-accuracy factors"
            )
            ax[0].set_xlabel("Mass-accuracy factor (× base tolerance)")
            ax[0].set_ylabel("Sum of correlations (top-N)")
            ax[0].grid(True, alpha=0.3)

            # Right: boxplot of per-factor correlation distributions (top-N each)
            # Prepare data: replace pure-NaN arrays with [np.nan] to keep boxplot aligned
            box_data = [
                vals if np.any(np.isfinite(vals)) else np.array([np.nan])
                for vals in per_factor_corrs
            ]
            ax[1].boxplot(box_data, labels=[str(f) for f in factors], showmeans=True)
            ax[1].set_title("Distribution of per-fragment correlations (top-N)")
            ax[1].set_xlabel("Mass-accuracy factor")
            ax[1].set_ylabel("Pearson r")
            ax[1].set_ylim(-1.05, 1.05)

            fig.suptitle(
                f"Base fragment mass tolerance = {self.config.fragment_mass_tolerance} ppm; "
                f"Top-N = {self.config.top_n_fragments}",
                y=1.03,
                fontsize=10,
            )
            plt.tight_layout()
            plt.show()

        return results_arr

    def feature_remaining_fragments_correlations(
        self,
        fragments: pd.DataFrame,
        visualize: bool = False,  # <-- NEW toggle
    ) -> np.ndarray:
        """
        Sum of correlations for remaining fragments (non-normalized and normalized).

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        visualize : bool, default=False
            If True, render diagnostic plots for the remaining-fragment correlations:
            (i) sorted bar plot of r for the remaining fragments,
            (ii) histogram (density) with mean and zero-reference.

        Returns
        -------
        np.ndarray
            Array with [non_normalized_sum, normalized_sum]
        """
        try:
            # Compute correlations for all fragments against the smoothed best trace
            # (do not visualize here; this function controls plotting)
            _, _, correlations = self.calculate_pearson_correlations(
                fragments, visualize=False
            )

            # Determine the "top" set to exclude, using the extended N from config
            top_fragments = self.find_top_n_fragments(
                fragments, self.config.top_n_fragments
            )

            # Collect correlations for fragments not in the top set
            remaining_names = [f for f in correlations.keys() if f not in top_fragments]
            remaining_corrs = np.array(
                [correlations.get(f, np.nan) for f in remaining_names], dtype=float
            )

            # Remove NaNs
            valid_mask = np.isfinite(remaining_corrs)
            valid_corrs = remaining_corrs[valid_mask]
            valid_names = [n for n, m in zip(remaining_names, valid_mask) if m]

            if valid_corrs.size == 0:
                # Nothing to visualize either
                if visualize:
                    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
                    ax.set_title("No remaining fragments with valid correlations")
                    ax.set_axis_off()
                    plt.tight_layout()
                    plt.show()
                return np.array([0.0, 0.0], dtype=float)

            non_normalized = float(np.sum(valid_corrs))
            normalized = float(non_normalized / valid_corrs.size)

            # -------------------- Visualization (optional) --------------------
            if visualize:
                # Sort for bar plot readability
                order = np.argsort(valid_corrs)  # ascending
                corr_sorted = valid_corrs[order]
                names_sorted = [valid_names[i] for i in order]

                fig, axes = plt.subplots(1, 2, figsize=(13, 4))

                # (i) Sorted bar plot of remaining correlations
                axes[0].barh(names_sorted, corr_sorted)
                axes[0].set_xlim(-1.05, 1.05)
                axes[0].axvline(0.0, linestyle="--", linewidth=1)
                axes[0].set_title("Remaining fragments: Pearson r (sorted)")
                axes[0].set_xlabel("r")
                axes[0].set_ylabel("Fragment")

                # (ii) Histogram with density; indicate mean and zero
                axes[1].hist(valid_corrs, bins=20, density=True)
                axes[1].axvline(0.0, linestyle="--", linewidth=1, label="r = 0")
                axes[1].axvline(
                    normalized,
                    linestyle=":",
                    linewidth=2,
                    label=f"Mean r = {normalized:.3f}",
                )
                axes[1].set_xlim(-1.05, 1.05)
                axes[1].set_title("Distribution of remaining-fragment correlations")
                axes[1].set_xlabel("r")
                axes[1].set_ylabel("Density")
                axes[1].legend()

                fig.suptitle(
                    f"Excluded top-{self.config.top_n_fragments}; "
                    f"Remaining: {valid_corrs.size} fragments\n"
                    f"Sum = {non_normalized:.3f}, Mean = {normalized:.3f}",
                    y=1.03,
                    fontsize=10,
                )
                plt.tight_layout()
                plt.show()
            # -----------------------------------------------------------------

            return np.array([non_normalized, normalized], dtype=float)

        except Exception as e:
            logger.error(f"Error calculating remaining fragment correlations: {e}")
            return np.array([np.nan, np.nan], dtype=float)

    def feature_best_b_fragments_correlation(
        self,
        fragments: pd.DataFrame,
        n: int = 3,
        visualize: bool = False,  # <-- NEW
    ) -> float:
        """
        Sum of correlations for top-n b-fragments, with optional visualization.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        n : int, default=3
            Number of top b-fragments to consider
        visualize : bool, default=False
            If True, render diagnostic plots of b-fragment correlations.

        Returns
        -------
        float
            Sum of correlations for best b-fragments
        """
        try:
            _, _, correlations = self.calculate_pearson_correlations(
                fragments, visualize=False
            )

            # Select only b-fragments
            b_fragments = [frag for frag in correlations.keys() if frag.startswith("b")]

            if len(b_fragments) == 0:
                if visualize:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.set_title("No b-fragments found")
                    ax.set_axis_off()
                    plt.show()
                return np.nan

            # Sort by correlation (descending) and take top-n
            b_fragments_sorted = sorted(
                b_fragments, key=lambda x: correlations.get(x, -np.inf), reverse=True
            )
            top_b = b_fragments_sorted[:n]
            corrs = [correlations.get(f, 0.0) for f in top_b]
            corr_sum = float(np.nansum(corrs))

            # -------------------- Visualization --------------------
            if visualize:
                # Build sorted correlation series for all b-fragments
                corr_s = pd.Series(
                    {f: correlations.get(f, np.nan) for f in b_fragments}
                )
                corr_s = corr_s.sort_values(ascending=True)  # for barh plot

                fig, axes = plt.subplots(1, 2, figsize=(13, 4))

                # (i) Barh plot of all b-fragment correlations
                colors = [
                    "tab:blue" if f not in top_b else "tab:orange" for f in corr_s.index
                ]
                axes[0].barh(corr_s.index, corr_s.values, color=colors)
                axes[0].axvline(0.0, linestyle="--", linewidth=1, color="black")
                axes[0].set_xlim(-1.05, 1.05)
                axes[0].set_title(f"b-fragment correlations (top-{n} in orange)")
                axes[0].set_xlabel("Pearson r")

                # (ii) Histogram of correlation distribution
                vals = corr_s.values[np.isfinite(corr_s.values)]
                axes[1].hist(vals, bins=20, density=True, alpha=0.7)
                axes[1].axvline(
                    0.0, linestyle="--", linewidth=1, color="black", label="r = 0"
                )
                axes[1].axvline(
                    np.mean(corrs),
                    linestyle=":",
                    linewidth=2,
                    color="tab:orange",
                    label=f"Mean of top-{n} = {np.nanmean(corrs):.3f}",
                )
                axes[1].set_xlim(-1.05, 1.05)
                axes[1].set_title("Distribution of b-fragment correlations")
                axes[1].set_xlabel("r")
                axes[1].set_ylabel("Density")
                axes[1].legend()

                fig.suptitle(
                    f"Top-{n} b-fragment correlation sum = {corr_sum:.3f}",
                    y=1.02,
                    fontsize=11,
                )
                plt.tight_layout()
                plt.show()
            # -------------------------------------------------------

            return corr_sum

        except Exception as e:
            logger.error(f"Error calculating b-fragment correlations: {e}")
            return np.nan

    def feature_precursor_best_fragment_correlation(
        self,
        precursor: pd.DataFrame,
        fragments: pd.DataFrame,
        ms1_dict: Dict[str, Dict[str, Any]],
        visualize: bool = False,  # <-- NEW toggle
    ) -> float:
        """
        Correlation between precursor (MS1) and best fragment (MS2) elution profiles.
        Optionally visualizes the aligned traces and the point-wise relationship.

        Parameters
        ----------
        precursor : pd.DataFrame
            Precursor information; must contain 'calcmass' and 'charge' columns.
        fragments : pd.DataFrame
            Fragment data used to locate and smooth the best-fragment chromatogram.
        ms1_dict : dict
            MS1 data dictionary consumed by `self.build_elution_profile(...)`.
        visualize : bool, default=False
            If True, render:
            (i) overlay of aligned precursor and smoothed-best-fragment elution profiles (min–max normalized),
            (ii) scatter of paired intensities used for the Pearson correlation.

        Returns
        -------
        float
            Pearson correlation coefficient between the aligned precursor intensity
            and the smoothed best-fragment intensity (NaN if insufficient data).
        """
        try:
            # --- Compute precursor m/z ---
            precursor_mz = (
                precursor["calcmass"].iloc[0] / precursor["charge"].iloc[0]
                + 1.007276466879
            )

            # --- Build precursor (MS1) elution profile ---
            elution_profile = self.build_elution_profile(precursor_mz, ms1_dict)
            if not elution_profile:
                if visualize:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.set_title("No MS1 elution profile available")
                    ax.set_axis_off()
                    plt.tight_layout()
                    plt.show()
                return np.nan

            # --- Best-fragment (MS2) trace and its smoothed version ---
            best_trace, smoothed_best_trace, _ = self.calculate_pearson_correlations(
                fragments, visualize=False
            )

            # --- Align profiles (ASOF merge on RT within tolerance) ---
            smoothed_best_df = pd.DataFrame(
                {
                    "rt": best_trace.index.astype(float),
                    "smoothed_intensity": smoothed_best_trace,
                }
            ).sort_values("rt")

            elution_df = (
                pd.DataFrame(list(elution_profile.items()), columns=["rt", "intensity"])
                .astype({"rt": float})
                .sort_values("rt")
            )

            merged = pd.merge_asof(
                smoothed_best_df,
                elution_df,
                on="rt",
                direction="nearest",
                tolerance=self.config.rt_tolerance,
            )
            merged_clean = merged.dropna(subset=["smoothed_intensity", "intensity"])

            if len(merged_clean) < 2:
                if visualize:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.set_title("Insufficient overlap after RT alignment")
                    ax.set_xlabel("RT")
                    ax.set_ylabel("Intensity")
                    plt.tight_layout()
                    plt.show()
                return np.nan

            # --- Pearson correlation ---
            r = float(
                merged_clean["smoothed_intensity"].corr(merged_clean["intensity"])
            )

            # -------------------- Visualization (optional) --------------------
            if visualize:
                # Min–max normalization for visual comparison of shapes
                def _minmax(x: pd.Series) -> np.ndarray:
                    x = x.to_numpy(dtype=float)
                    xmin, xmax = np.nanmin(x), np.nanmax(x)
                    return (
                        (x - xmin) / (xmax - xmin) if xmax > xmin else np.zeros_like(x)
                    )

                rt_aligned = merged_clean["rt"].to_numpy(dtype=float)
                ms2_norm = _minmax(merged_clean["smoothed_intensity"])
                ms1_norm = _minmax(merged_clean["intensity"])

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # (i) Overlay of normalized elution profiles
                axes[0].plot(
                    rt_aligned, ms2_norm, label="Best fragment (smoothed, norm.)"
                )
                axes[0].plot(rt_aligned, ms1_norm, label="Precursor (MS1, norm.)")
                axes[0].set_title(f"Aligned elution profiles (r = {r:.3f})")
                axes[0].set_xlabel("RT")
                axes[0].set_ylabel("Relative intensity")
                axes[0].legend()

                # (ii) Scatter of paired intensities used for correlation
                axes[1].scatter(
                    merged_clean["smoothed_intensity"].to_numpy(dtype=float),
                    merged_clean["intensity"].to_numpy(dtype=float),
                    alpha=0.7,
                )
                axes[1].set_title("Point-wise relationship (paired samples)")
                axes[1].set_xlabel("Best fragment (smoothed) intensity")
                axes[1].set_ylabel("Precursor (MS1) intensity")

                plt.tight_layout()
                plt.show()
            # -----------------------------------------------------------------

            return r

        except Exception as e:
            logger.error(f"Error calculating precursor-fragment correlation: {e}")
            return np.nan

    # Feature Group 2: MS1 Level Co-elution
    def feature_ms1_accuracy_correlations(
        self,
        precursor: pd.DataFrame,
        fragments: pd.DataFrame,
        ms1_dict: Dict[str, Dict[str, Any]],
        visualize: bool = False,  # <-- NEW: overall summary visualization
        visualize_per_factor: bool = False,  # <-- NEW: delegate per-factor plots to helper
    ) -> np.ndarray:
        """
        Correlations between precursor (MS1) and best-fragment (MS2) elution profiles
        computed across a sweep of MS1 mass-accuracy factors. Optionally visualizes:
        (i) a summary line plot of correlation vs. accuracy factor, and
        (ii) per-factor diagnostic plots (if `visualize_per_factor=True`, via helper).

        Parameters
        ----------
        precursor : pd.DataFrame
            Precursor information; must contain 'calcmass' and 'charge'.
        fragments : pd.DataFrame
            Fragment data for best-fragment profiling.
        ms1_dict : dict
            MS1 data dictionary to be consumed by `self.build_elution_profile(...)`.
        visualize : bool, default=False
            Draw a summary plot of correlation vs. MS1 accuracy factor.
        visualize_per_factor : bool, default=False
            If True, pass `visualize=True` down to the helper to render an overlay
            and scatter for *each* factor.

        Returns
        -------
        np.ndarray
            Array of Pearson correlations (one per MS1 accuracy factor in
            `self.config.ms1_accuracy_factors`), NaN where unavailable.
        """
        if not hasattr(self, "config") or not hasattr(
            self.config, "ms1_accuracy_factors"
        ):
            raise AttributeError(
                "self.config.ms1_accuracy_factors is required but missing."
            )

        factors: List[float] = list(self.config.ms1_accuracy_factors)
        results: List[float] = []

        for acc_factor in factors:
            try:
                corr = self._calculate_precursor_fragment_correlation_with_accuracy(
                    precursor=precursor,
                    fragments=fragments,
                    ms1_dict=ms1_dict,
                    acc_factor=acc_factor,
                    visualize=visualize_per_factor,  # pass through if requested
                )
                results.append(corr)
            except Exception as e:
                logger.error(f"Error with MS1 accuracy factor {acc_factor}: {e}")
                results.append(np.nan)

        results_arr = np.asarray(results, dtype=float)

        # ---- Summary visualization (optional) ----
        if visualize:
            plt.figure(figsize=(7, 4))
            plt.plot(factors, results_arr, marker="o")
            plt.title("MS1–MS2 correlation vs. MS1 mass-accuracy factor")
            plt.xlabel("MS1 mass-accuracy factor (× base tolerance)")
            plt.ylabel("Pearson correlation (r)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return results_arr

    def _calculate_precursor_fragment_correlation_with_accuracy(
        self,
        precursor: pd.DataFrame,
        fragments: pd.DataFrame,
        ms1_dict: Dict[str, Dict[str, Any]],
        acc_factor: float,
        visualize: bool = False,  # <-- NEW: per-factor diagnostic visualization
    ) -> float:
        """
        Compute the Pearson correlation between the precursor (MS1) elution profile
        extracted at a given MS1 mass-accuracy factor and the smoothed best-fragment
        (MS2) elution profile. Optionally visualizes the aligned traces and the
        paired-intensity scatter for this specific factor.

        Parameters
        ----------
        precursor : pd.DataFrame
            Must contain 'calcmass' and 'charge'.
        fragments : pd.DataFrame
            Fragment data used to obtain the best-fragment chromatogram and smoothing.
        ms1_dict : dict
            MS1 data dictionary used by `self.build_elution_profile`.
        acc_factor : float
            Multiplicative factor applied to the base MS1 mass tolerance.
        visualize : bool, default=False
            Draw two panels for this factor:
            (i) overlay of normalized MS1 vs. smoothed MS2 elution profiles,
            (ii) scatter of paired intensities used for the correlation.

        Returns
        -------
        float
            Pearson correlation coefficient (NaN if insufficient overlap or data).
        """
        # Precursor m/z and MS1 elution profile at this accuracy factor
        precursor_mz = (
            precursor["calcmass"].iloc[0] / precursor["charge"].iloc[0] + 1.007276466879
        )
        elution_profile = self.build_elution_profile(
            precursor_mz, ms1_dict, acc_factor=acc_factor
        )
        if not elution_profile:
            if visualize:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.set_title(f"No MS1 elution profile at factor={acc_factor:g}")
                ax.set_axis_off()
                plt.tight_layout()
                plt.show()
            return np.nan

        # Best-fragment (MS2) trace and smoothing
        best_trace, smoothed_best_trace, _ = self.calculate_pearson_correlations(
            fragments, visualize=False
        )

        # Align on RT using asof-merge with configured tolerance
        smoothed_best_df = pd.DataFrame(
            {
                "rt": best_trace.index.astype(float),
                "smoothed_intensity": smoothed_best_trace,
            }
        ).sort_values("rt")

        elution_df = (
            pd.DataFrame(list(elution_profile.items()), columns=["rt", "intensity"])
            .astype({"rt": float})
            .sort_values("rt")
        )

        merged = pd.merge_asof(
            smoothed_best_df,
            elution_df,
            on="rt",
            direction="nearest",
            tolerance=self.config.rt_tolerance,
        )
        merged_clean = merged.dropna(subset=["smoothed_intensity", "intensity"])

        if len(merged_clean) < 2:
            if visualize:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.set_title(
                    f"Insufficient overlap after RT alignment (factor={acc_factor:g})"
                )
                ax.set_xlabel("RT")
                ax.set_ylabel("Intensity")
                plt.tight_layout()
                plt.show()
            return np.nan

        # Pearson correlation at this accuracy factor
        r = float(merged_clean["smoothed_intensity"].corr(merged_clean["intensity"]))

        # ---- Per-factor visualization (optional) ----
        if visualize:
            # Min–max normalization for visual comparability
            def _minmax(arr: pd.Series) -> np.ndarray:
                x = arr.to_numpy(dtype=float)
                xmin, xmax = np.nanmin(x), np.nanmax(x)
                return (x - xmin) / (xmax - xmin) if xmax > xmin else np.zeros_like(x)

            rt_aligned = merged_clean["rt"].to_numpy(dtype=float)
            ms2_norm = _minmax(merged_clean["smoothed_intensity"])
            ms1_norm = _minmax(merged_clean["intensity"])

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # (i) Overlay of normalized elution profiles
            axes[0].plot(
                rt_aligned, ms2_norm, label="MS2 best fragment (smoothed, norm.)"
            )
            axes[0].plot(
                rt_aligned, ms1_norm, label=f"MS1 (norm.) @ factor={acc_factor:g}"
            )
            axes[0].set_title(f"Aligned elution profiles (r = {r:.3f})")
            axes[0].set_xlabel("RT")
            axes[0].set_ylabel("Relative intensity")
            axes[0].legend()

            # (ii) Scatter of paired intensities
            axes[1].scatter(
                merged_clean["smoothed_intensity"].to_numpy(dtype=float),
                merged_clean["intensity"].to_numpy(dtype=float),
                alpha=0.7,
            )
            axes[1].set_title("Point-wise relationship (paired samples)")
            axes[1].set_xlabel("Best fragment (smoothed) intensity")
            axes[1].set_ylabel("Precursor (MS1) intensity")

            plt.tight_layout()
            plt.show()

        return r

    # Feature Group 3: Isotopologue Co-elution

    def feature_c13_isotope_correlations(
        self,
        precursor: pd.DataFrame,
        fragments: pd.DataFrame,
        ms1_dict: Dict[str, Dict[str, Any]],
        visualize: bool = False,  # <-- NEW: summary visualization
        visualize_per_isotope: bool = False,  # <-- NEW: per-isotope diagnostics
    ) -> np.ndarray:
        """
        Correlations between the smoothed best-fragment (MS2) elution profile and
        MS1 elution profiles extracted at precursor + k*C13/z for k in c13_isotope_list.
        Optionally visualizes a summary bar plot and per-isotope diagnostic overlays.

        Parameters
        ----------
        precursor : pd.DataFrame
            Precursor information (must contain 'calcmass' and 'charge').
        fragments : pd.DataFrame
            Fragment data used to determine and smooth the best-fragment elution profile.
        ms1_dict : dict
            MS1 data dictionary consumed by `self.build_elution_profile`.
        visualize : bool, default=False
            If True, render a summary bar chart of correlation vs. C13 count.
        visualize_per_isotope : bool, default=False
            If True, render per-isotope overlays:
            (i) normalized smoothed best-fragment vs. isotope elution profile,
            (ii) scatter of paired intensities used for the correlation.

        Returns
        -------
        np.ndarray
            Array of Pearson correlations, aligned with `self.config.c13_isotope_list`.
            NaN where no valid alignment is available.
        """
        try:
            # --- Best-fragment (MS2) reference: trace and smoothing ---
            best_trace, smoothed_best_trace, _ = self.calculate_pearson_correlations(
                fragments, visualize=False
            )

            smoothed_best_df = pd.DataFrame(
                {
                    "rt": best_trace.index.astype(float),
                    "smoothed_intensity": smoothed_best_trace,
                }
            ).sort_values("rt")

            # --- Precursor m/z and charge ---
            base_precursor_mz = (
                float(precursor["calcmass"].iloc[0])
                / float(precursor["charge"].iloc[0])
                + 1.007276466879
            )
            charge = int(precursor["charge"].iloc[0])

            # --- Iterate over requested C13 isotope counts ---
            counts: List[int] = list(self.config.c13_isotope_list)
            corrs: List[float] = []

            for c13_count in counts:
                try:
                    isotope_mz = base_precursor_mz + (
                        float(c13_count)
                        * float(self.config.isotope_mass_c13)
                        / float(charge)
                    )

                    # Build isotope elution profile at this m/z
                    isotope_profile = self.build_elution_profile(isotope_mz, ms1_dict)
                    if not isotope_profile:
                        corrs.append(np.nan)
                        if visualize_per_isotope:
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.set_title(
                                f"No isotope elution profile (C13 count = {c13_count})"
                            )
                            ax.set_axis_off()
                            plt.tight_layout()
                            plt.show()
                        continue

                    isotope_df = (
                        pd.DataFrame(
                            list(isotope_profile.items()),
                            columns=["rt", "isotope_intensity"],
                        )
                        .astype({"rt": float})
                        .sort_values("rt")
                    )

                    # Align to MS2 smoothed reference using asof with RT tolerance
                    merged = pd.merge_asof(
                        smoothed_best_df,
                        isotope_df,
                        on="rt",
                        direction="nearest",
                        tolerance=self.config.rt_tolerance,
                    )
                    merged_clean = merged.dropna(
                        subset=["smoothed_intensity", "isotope_intensity"]
                    )

                    if len(merged_clean) < 2:
                        corrs.append(np.nan)
                        if visualize_per_isotope:
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.set_title(
                                f"Insufficient overlap after RT alignment (C13 = {c13_count})"
                            )
                            ax.set_xlabel("RT")
                            ax.set_ylabel("Intensity")
                            plt.tight_layout()
                            plt.show()
                        continue

                    # Pearson correlation
                    r = float(
                        merged_clean["smoothed_intensity"].corr(
                            merged_clean["isotope_intensity"]
                        )
                    )
                    corrs.append(r if np.isfinite(r) else np.nan)

                    # ----- Per-isotope visualization (optional) -----
                    if visualize_per_isotope:
                        # Min–max normalization for shape comparison
                        def _minmax(series: pd.Series) -> np.ndarray:
                            x = series.to_numpy(dtype=float)
                            xmin, xmax = np.nanmin(x), np.nanmax(x)
                            return (
                                (x - xmin) / (xmax - xmin)
                                if xmax > xmin
                                else np.zeros_like(x)
                            )

                        rt_aligned = merged_clean["rt"].to_numpy(dtype=float)
                        ms2_norm = _minmax(merged_clean["smoothed_intensity"])
                        iso_norm = _minmax(merged_clean["isotope_intensity"])

                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                        # (i) Overlaid normalized elution profiles
                        axes[0].plot(
                            rt_aligned, ms2_norm, label="MS2 best (smoothed, norm.)"
                        )
                        axes[0].plot(
                            rt_aligned,
                            iso_norm,
                            label=f"MS1 isotope (C13={c13_count}, norm.)",
                        )
                        axes[0].set_title(
                            f"Aligned profiles — r = {r:.3f} (C13 = {c13_count})"
                        )
                        axes[0].set_xlabel("RT")
                        axes[0].set_ylabel("Relative intensity")
                        axes[0].legend()

                        # (ii) Scatter of paired intensities
                        axes[1].scatter(
                            merged_clean["smoothed_intensity"].to_numpy(dtype=float),
                            merged_clean["isotope_intensity"].to_numpy(dtype=float),
                            alpha=0.7,
                        )
                        axes[1].set_title("Point-wise relationship (paired samples)")
                        axes[1].set_xlabel("Best fragment (smoothed) intensity")
                        axes[1].set_ylabel("Isotope (MS1) intensity")

                        plt.tight_layout()
                        plt.show()

                except Exception as e_iso:
                    logger.error(
                        f"C13={c13_count}: error calculating isotope correlation: {e_iso}"
                    )
                    corrs.append(np.nan)

            corrs_arr = np.asarray(corrs, dtype=float)

            # ----- Summary visualization (optional) -----
            if visualize:
                plt.figure(figsize=(7, 4))
                plt.bar([str(k) for k in counts], corrs_arr)
                plt.ylim(-1.05, 1.05)
                plt.axhline(0.0, linestyle="--", linewidth=1)
                plt.title("Correlation with C13 isotope elution profiles")
                plt.xlabel("C13 atoms (k)")
                plt.ylabel("Pearson r (MS2 smoothed vs. MS1 isotope)")
                plt.tight_layout()
                plt.show()

            return corrs_arr

        except Exception as e:
            logger.error(f"Error calculating C13 isotope correlations: {e}")
            return np.full(len(self.config.c13_isotope_list), np.nan, dtype=float)

    def feature_c13_subtracted_correlations(
        self,
        fragments: pd.DataFrame,
        ms2dict: Dict[str, Dict[str, Any]],
        visualize: bool = False,  # <-- NEW: summary bar plot
        visualize_per_fragment: bool = False,  # <-- NEW: per-fragment diagnostics
    ) -> np.ndarray:
        """
        Correlations between the smoothed best-fragment (MS2) elution profile and
        elution profiles reconstructed at (fragment_mz - C13/z) for the top-N fragments.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        ms2dict : dict
            MS2 spectral data dict
        visualize : bool, default=False
            If True, render a summary bar chart of per-fragment correlations.
        visualize_per_fragment : bool, default=False
            If True, for each fragment render:
            (i) normalized smoothed best vs. C13-subtracted elution profile, and
            (ii) scatter of paired intensities used for the correlation.

        Returns
        -------
        np.ndarray
            Correlations for the top-N fragments (NaN-padded).
        """
        try:
            # Reference MS2 profile (best fragment, smoothed)
            best_trace, smoothed_best_trace, _ = self.calculate_pearson_correlations(
                fragments, visualize=False
            )
            smoothed_best_df = pd.DataFrame(
                {
                    "rt": best_trace.index.astype(float),
                    "smoothed_intensity": smoothed_best_trace,
                }
            )

            # Select top-N fragments
            top_fragments = self.find_top_n_fragments(
                fragments, self.config.top_n_fragments
            )
            correlations: List[float] = []
            labels: List[str] = []

            for frag_name in top_fragments:
                try:
                    corr = self._calculate_c13_subtracted_correlation(
                        frag_name,
                        fragments,
                        smoothed_best_df,
                        ms2dict,
                        visualize=visualize_per_fragment,  # pass through
                    )
                    correlations.append(corr)
                    labels.append(frag_name)
                except Exception as e:
                    logger.warning(f"Error processing fragment {frag_name}: {e}")
                    correlations.append(np.nan)
                    labels.append(frag_name)

            # Pad to fixed length if needed
            while len(correlations) < self.config.top_n_fragments:
                correlations.append(np.nan)
                labels.append(f"pad{len(correlations)}")

            corr_arr = np.asarray(
                correlations[: self.config.top_n_fragments], dtype=float
            )

            # ----- Summary visualization (optional) -----
            if visualize:
                plt.figure(figsize=(8, 4))
                plt.bar(labels[: self.config.top_n_fragments], corr_arr)
                plt.ylim(-1.05, 1.05)
                plt.axhline(0.0, linestyle="--", linewidth=1)
                plt.title("C13-subtracted correlations (per fragment)")
                plt.xlabel("Fragment")
                plt.ylabel("Pearson r")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.show()

            return corr_arr

        except Exception as e:
            logger.error(f"Error calculating C13 subtracted correlations: {e}")
            return np.full(self.config.top_n_fragments, np.nan, dtype=float)

    def feature_sum_c13_subtracted_correlations(
        self,
        fragments: pd.DataFrame,
        ms2dict: Dict[str, Dict[str, Any]],
        visualize: bool = False,  # <-- NEW: summary with annotation
    ) -> float:
        """
        Sum of C13-subtracted correlations for the top fragments.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        ms2dict : dict
            MS2 spectral data dictionary
        visualize : bool, default=False
            If True, display the bar chart of per-fragment correlations with the sum annotated.

        Returns
        -------
        float
            Sum of C13-subtracted correlations (NaN-safe).
        """
        corr_arr = self.feature_c13_subtracted_correlations(
            fragments, ms2dict, visualize=visualize, visualize_per_fragment=False
        )
        total = float(np.nansum(corr_arr))

        if visualize:
            # If the per-fragment chart was already shown, add a compact gauge for the sum.
            plt.figure(figsize=(4, 3))
            plt.bar(["Σ corr"], [total])
            plt.ylim(min(0.0, total) - 0.1, max(0.0, total) + 0.1)
            plt.title("Sum of C13-subtracted correlations")
            plt.tight_layout()
            plt.show()

        return total

    def _calculate_c13_subtracted_correlation(
        self,
        frag_name: str,
        fragments: pd.DataFrame,
        smoothed_best_df: pd.DataFrame,
        ms2dict: Dict[str, Dict[str, Any]],
        visualize: bool = False,  # <-- NEW: per-fragment plots
    ) -> float:
        """
        Helper: correlation between smoothed best-fragment intensity and the
        C13-subtracted elution profile for a specific fragment.

        Parameters
        ----------
        frag_name : str
            Target fragment name
        fragments : pd.DataFrame
            Fragment data
        smoothed_best_df : pd.DataFrame
            DataFrame with columns ['rt','smoothed_intensity']
        ms2dict : dict
            MS2 spectra by scan: {scan: {"mz": [...], "intensity": [...]} }
        visualize : bool, default=False
            If True, render:
            (i) normalized smoothed best vs. C13-subtracted elution profile,
            (ii) scatter of paired intensities used for the correlation.

        Returns
        -------
        float
            Pearson correlation coefficient (0.0 if computed NaN, NaN on failure).
        """
        frag_data = fragments[fragments["fragment_names"] == frag_name]
        if frag_data.empty:
            return np.nan

        # Target m/z with one C13 mass subtracted
        original_mz = float(frag_data["fragment_mz_calculated"].iloc[0])
        charge = float(frag_data["fragment_charge"].iloc[0])
        c13_subtracted_mz = original_mz - (self.config.isotope_mass_c13 / charge)

        # Build elution profile at the adjusted m/z using MS2 scans
        c13_subtracted_profile: Dict[float, float] = {}
        for _, row in frag_data.iterrows():
            rt = float(row["rt"])
            scan = row["scannr"]
            scan_data = ms2dict.get(scan, {})
            mzs = scan_data.get("mz", [])
            intensities = scan_data.get("intensity", [])
            if len(mzs) == 0 or len(intensities) == 0:
                continue

            best_idx, _best_val = self._search_sorted_with_tolerance(
                mzs, c13_subtracted_mz, self.config.fragment_mass_tolerance
            )
            if best_idx is not None:
                c13_subtracted_profile[rt] = float(intensities[best_idx])

        if not c13_subtracted_profile:
            if visualize:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.set_title(f"{frag_name}: no C13-subtracted profile")
                ax.set_axis_off()
                plt.tight_layout()
                plt.show()
            return np.nan

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
            if visualize:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.set_title(f"{frag_name}: insufficient overlap after RT alignment")
                ax.set_xlabel("RT")
                ax.set_ylabel("Intensity")
                plt.tight_layout()
                plt.show()
            return np.nan

        r = merged_clean["smoothed_intensity"].corr(
            merged_clean["c13_subtracted_intensity"]
        )
        r = float(r) if pd.notna(r) else 0.0

        # ----- Per-fragment visualization (optional) -----
        if visualize:

            def _minmax(series: pd.Series) -> np.ndarray:
                x = series.to_numpy(dtype=float)
                xmin, xmax = np.nanmin(x), np.nanmax(x)
                return (x - xmin) / (xmax - xmin) if xmax > xmin else np.zeros_like(x)

            rt_aligned = merged_clean["rt"].to_numpy(dtype=float)
            ms2_norm = _minmax(merged_clean["smoothed_intensity"])
            c13_norm = _minmax(merged_clean["c13_subtracted_intensity"])

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # (i) Overlaid normalized elution profiles
            axes[0].plot(rt_aligned, ms2_norm, label="MS2 best (smoothed, norm.)")
            axes[0].plot(
                rt_aligned, c13_norm, label=f"{frag_name} (C13-subtracted, norm.)"
            )
            axes[0].set_title(f"{frag_name} — aligned profiles (r = {r:.3f})")
            axes[0].set_xlabel("RT")
            axes[0].set_ylabel("Relative intensity")
            axes[0].legend()

            # (ii) Scatter of paired intensities
            axes[1].scatter(
                merged_clean["smoothed_intensity"].to_numpy(dtype=float),
                merged_clean["c13_subtracted_intensity"].to_numpy(dtype=float),
                alpha=0.7,
            )
            axes[1].set_title("Point-wise relationship (paired samples)")
            axes[1].set_xlabel("Best fragment (smoothed) intensity")
            axes[1].set_ylabel("C13-subtracted intensity")

            plt.tight_layout()
            plt.show()

        return r

    def feature_weighted_auc(
        self,
        fragments: pd.DataFrame,
        visualize: bool = False,  # <-- NEW toggle
    ) -> float:
        """
        Natural log of weighted AUC for top fragments.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        visualize : bool, default=False
            If True, render diagnostic plots:
            (i) overlay of top-N fragment chromatograms (RT–intensity),
            (ii) bar charts of AUC, correlation, and AUC×corr (weighted AUC) per fragment.

        Returns
        -------
        float
            Natural log of sum_i (AUC_i * corr_i) over the selected top-N fragments.
            Returns NaN if the sum is non-positive or if no valid AUCs are found.
        """
        try:
            # Correlations of each fragment vs. smoothed best
            _, _, correlations = self.calculate_pearson_correlations(
                fragments, visualize=False
            )

            # Select top-N fragments according to your internal criterion
            top_fragments = self.find_top_n_fragments(
                fragments, self.config.top_n_fragments
            )

            aucs: list[float] = []
            corrs: list[float] = []
            used_names: list[str] = []

            # Compute AUC per fragment over RT (sorted), and pair with its correlation
            for frag in top_fragments:
                frag_data = fragments[fragments["fragment_names"] == frag]
                if frag_data.empty:
                    continue

                frag_data_sorted = frag_data.sort_values("rt")
                rt = frag_data_sorted["rt"].to_numpy(dtype=float)
                intensity = frag_data_sorted["fragment_intensity"].to_numpy(dtype=float)

                if rt.size > 1:
                    auc = float(np.trapz(intensity, rt))
                    aucs.append(auc)
                    corrs.append(float(correlations.get(frag, 0.0)))
                    used_names.append(frag)

            if not aucs:
                # Nothing to compute or visualize
                if visualize:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.set_title("No valid AUCs for selected fragments")
                    ax.set_axis_off()
                    plt.tight_layout()
                    plt.show()
                return np.nan

            aucs_arr = np.asarray(aucs, dtype=float)
            corrs_arr = np.asarray(corrs, dtype=float)
            # Replace NaN correlations by 0 (neutral); negatives will reduce the weighted sum
            corrs_arr = np.nan_to_num(corrs_arr, nan=0.0)

            weighted_aucs = aucs_arr * corrs_arr
            total_weighted_auc = float(np.sum(weighted_aucs))
            result = np.log(total_weighted_auc) if total_weighted_auc > 0 else np.nan

            # -------------------- Visualization (optional) --------------------
            if visualize:
                # (i) Overlay of top-N chromatograms (raw scale)
                fig, axes = plt.subplots(1, 2, figsize=(14, 4))

                # Gather traces for selected fragments
                for frag in used_names:
                    fd = fragments[fragments["fragment_names"] == frag].sort_values(
                        "rt"
                    )
                    axes[0].plot(
                        fd["rt"].to_numpy(),
                        fd["fragment_intensity"].to_numpy(),
                        label=frag,
                        alpha=0.85,
                    )
                axes[0].set_title("Top-N fragment chromatograms (RT vs intensity)")
                axes[0].set_xlabel("RT")
                axes[0].set_ylabel("Intensity")
                axes[0].legend(fontsize=8, ncol=1, loc="best")

                # (ii) Per-fragment bars: AUC, corr, and weighted AUC
                x = np.arange(len(used_names))
                width = 0.28
                axes[1].bar(x - width, aucs_arr, width, label="AUC")
                axes[1].bar(x, corrs_arr, width, label="corr (vs. smoothed best)")
                axes[1].bar(x + width, weighted_aucs, width, label="AUC × corr")
                axes[1].set_xticks(x, used_names, rotation=45, ha="right")
                axes[1].set_title(
                    f"Weighted AUC components (Σ AUC×corr = {total_weighted_auc:.3g}; ln = {result if np.isfinite(result) else float('nan'):.3g})"
                )
                axes[1].set_ylabel("Value")
                axes[1].legend()

                plt.tight_layout()
                plt.show()
            # -----------------------------------------------------------------

            return result

        except Exception as e:
            logger.error(f"Error calculating weighted AUC: {e}")
            return np.nan

    """
    TODO: MISSING: Cosine similarity measure (itself and to
    power 3) between the predicted and
    measured intensities of the top 6 fragments
    weighted by the squared values of the
    smoothed “best” fragment elution curve at
    the respective time points
    """

    # Feature Group 5: Fragment Intensities
    def feature_relative_intensities_top_6(
        self,
        fragments: pd.DataFrame,
        visualize: bool = False,  # <-- NEW
    ) -> np.ndarray:
        """
        Relative intensities of the top-N fragments (default = 6).
        Optionally visualize the per-fragment contributions.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data
        visualize : bool, default=False
            If True, render:
            (i) bar chart of relative intensities (normalized),
            (ii) pie chart of raw max-intensity proportions.

        Returns
        -------
        np.ndarray
            Array of relative intensities of length `self.config.top_n_fragments`,
            normalized so that the maximum fragment = 1.0.
        """
        try:
            n = self.config.top_n_fragments
            top_fragments = self.find_top_n_fragments(fragments, n)

            raw_intensities: list[float] = []
            for frag in top_fragments:
                frag_data = fragments[fragments["fragment_names"] == frag]
                if not frag_data.empty:
                    raw_intensities.append(float(frag_data["fragment_intensity"].max()))
                else:
                    raw_intensities.append(0.0)

            # Pad with zeros if fewer than n fragments available
            while len(raw_intensities) < n:
                raw_intensities.append(0.0)

            raw_intensities = raw_intensities[:n]
            intensities = np.asarray(raw_intensities, dtype=float)

            # Normalize by maximum
            max_intensity = float(np.max(intensities))
            if max_intensity > 0:
                intensities = intensities / max_intensity

            # -------------------- Visualization (optional) --------------------
            if visualize:
                frag_labels = top_fragments[:n] + [
                    f"pad{i+1}" for i in range(n - len(top_fragments))
                ]

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # (i) Bar chart of normalized intensities
                axes[0].bar(frag_labels, intensities)
                axes[0].set_ylim(0, 1.05)
                axes[0].set_title(
                    "Relative intensities of top-N fragments (normalized)"
                )
                axes[0].set_xlabel("Fragment")
                axes[0].set_ylabel("Relative intensity (max=1)")
                axes[0].tick_params(axis="x", rotation=45)

                # (ii) Pie chart of raw intensities (proportions)
                total = np.sum(raw_intensities)
                if total > 0:
                    axes[1].pie(
                        raw_intensities,
                        labels=frag_labels,
                        autopct="%1.1f%%",
                        startangle=90,
                    )
                    axes[1].set_title("Raw max-intensity contributions")
                else:
                    axes[1].text(0.5, 0.5, "No signal", ha="center", va="center")
                    axes[1].set_axis_off()

                plt.tight_layout()
                plt.show()
            # -----------------------------------------------------------------

            return intensities

        except Exception as e:
            logger.error(f"Error calculating relative intensities: {e}")
            return np.zeros(self.config.top_n_fragments, dtype=float)

    # Feature Group 6: Mass Accuracy
    def feature_weighted_mass_accuracy(
        self,
        fragments: pd.DataFrame,
        visualize: bool = False,  # <-- NEW toggle
    ) -> np.ndarray:
        """
        Mass accuracy at the chromatographic apex, weighted by fragment–reference correlations.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data; must contain columns:
            ['rt', 'fragment_names', 'fragment_intensity', 'ppm_error'].
        visualize : bool, default=False
            If True, render diagnostic plots:
            (i) bar chart per fragment of ppm_error, correlation, and ppm_error×correlation,
            (ii) scatter of correlation vs. ppm_error with the weighted product highlighted.

        Returns
        -------
        np.ndarray
            Array (length = self.config.top_n_fragments) of weighted mass accuracies
            for the top-N most intense fragments at the apex RT. NaN-padded as needed.
        """
        try:
            n = int(self.config.top_n_fragments)

            if "ppm_error" not in fragments.columns:
                logger.warning("ppm_error column missing")
                return np.full(n, np.nan, dtype=float)

            # --- 1) Identify apex RT based on overall max fragment intensity ---
            apex_idx = fragments["fragment_intensity"].idxmax()
            if pd.isna(apex_idx):
                return np.full(n, np.nan, dtype=float)
            apex_rt = float(fragments.loc[apex_idx, "rt"])

            # --- 2) Select top-N fragments at the apex by intensity ---
            fragments_at_apex = fragments[fragments["rt"] == apex_rt]
            if fragments_at_apex.empty:
                return np.full(n, np.nan, dtype=float)

            top_apex = fragments_at_apex.nlargest(n, "fragment_intensity")

            # --- 3) Correlations vs. smoothed best-fragment trace (across RT) ---
            _, _, correlations = self.calculate_pearson_correlations(
                fragments, visualize=False
            )

            # --- 4) Compute weighted mass accuracies (ppm_error * correlation) ---
            names, ppms, corrs, weighted = [], [], [], []

            for _, row in top_apex.iterrows():
                frag_name = row["fragment_names"]
                ppm_error = float(row["ppm_error"])
                r = correlations.get(frag_name, 0.0)
                r = 0.0 if pd.isna(r) else float(r)

                names.append(frag_name)
                ppms.append(ppm_error)
                corrs.append(r)
                weighted.append(ppm_error * r)

            # Pad to fixed length if fewer than n fragments at apex
            while len(weighted) < n:
                names.append(f"pad{len(weighted)+1}")
                ppms.append(np.nan)
                corrs.append(np.nan)
                weighted.append(np.nan)

            weighted_arr = np.asarray(weighted[:n], dtype=float)

            # -------------------- Visualization (optional) --------------------
            if visualize:
                labels = names[:n]
                ppms_arr = np.asarray(ppms[:n], dtype=float)
                corrs_arr = np.asarray(corrs[:n], dtype=float)

                fig, axes = plt.subplots(1, 2, figsize=(14, 4))

                # (i) Bar chart per fragment
                x = np.arange(n)
                width = 0.28
                axes[0].bar(x - width, ppms_arr, width, label="ppm_error")
                axes[0].bar(x, corrs_arr, width, label="correlation (r)")
                axes[0].bar(x + width, weighted_arr, width, label="ppm_error × r")
                axes[0].set_xticks(x, labels, rotation=45, ha="right")
                axes[0].set_title(
                    f"Apex RT = {apex_rt:.4f} — mass accuracy & weighting"
                )
                axes[0].set_ylabel("Value")
                axes[0].legend()

                # (ii) Scatter: correlation vs. ppm error (color encodes product magnitude)
                axes[1].scatter(corrs_arr, ppms_arr, s=60)
                for i, lbl in enumerate(labels):
                    if np.isfinite(corrs_arr[i]) and np.isfinite(ppms_arr[i]):
                        axes[1].annotate(
                            lbl,
                            (corrs_arr[i], ppms_arr[i]),
                            fontsize=8,
                            xytext=(3, 3),
                            textcoords="offset points",
                        )
                axes[1].axvline(0.0, linestyle="--", linewidth=1)
                axes[1].set_xlabel("Correlation r (fragment vs. smoothed best)")
                axes[1].set_ylabel("ppm_error at apex")
                axes[1].set_title("Apex mass accuracy vs. chromatographic consistency")

                plt.tight_layout()
                plt.show()
            # -----------------------------------------------------------------

            return weighted_arr

        except Exception as e:
            logger.error(f"Error calculating weighted mass accuracy: {e}")
            return np.full(self.config.top_n_fragments, np.nan, dtype=float)

    # Feature Group 7: Retention Time
    def feature_rt_apex(
        self,
        fragments: pd.DataFrame,
        visualize: bool = False,  # <-- NEW
        top_k: int = 6,  # visual overlay of up to top_k fragment traces
    ) -> float:
        """
        Retention time at intensity apex.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data with at least ['rt','fragment_names','fragment_intensity'].
        visualize : bool, default=False
            If True, overlays up to `top_k` fragment chromatograms and marks the apex RT.
        top_k : int, default=6
            Maximum number of fragment traces to overlay when visualize=True.

        Returns
        -------
        float
            RT at the global intensity apex across all fragment rows.
        """
        try:
            fragments = self._validate_fragments_input(fragments)
            if fragments.empty:
                return np.nan

            # Locate the single row with the highest intensity; take its RT
            apex_idx = fragments["fragment_intensity"].idxmax()
            apex_rt = float(fragments.loc[apex_idx, "rt"])

            if visualize:
                # Build pivot of RT × fragment_names to plot chromatograms
                piv = (
                    fragments.pivot_table(
                        index="rt",
                        columns="fragment_names",
                        values="fragment_intensity",
                        aggfunc="mean",
                    )
                    .sort_index()
                    .fillna(0.0)
                )

                # Choose up to top_k fragments by their max intensity over RT
                frag_max = piv.max(axis=0).sort_values(ascending=False)
                chosen = frag_max.index[:top_k]

                plt.figure(figsize=(8, 4))
                for col in chosen:
                    plt.plot(piv.index.values, piv[col].values, label=col, alpha=0.85)

                # Mark the apex RT
                plt.axvline(
                    apex_rt,
                    linestyle="--",
                    linewidth=2,
                    label=f"Apex RT = {apex_rt:.4f}",
                )
                plt.title("Fragment chromatograms with apex RT")
                plt.xlabel("RT")
                plt.ylabel("Intensity")
                plt.legend(fontsize=8, ncol=1, loc="best")
                plt.tight_layout()
                plt.show()

            return apex_rt

        except Exception as e:
            logger.error(f"Error calculating RT apex: {e}")
            return np.nan

    # --------- Feature: sqrt absolute difference to RT prediction --------- #
    def feature_rt_prediction_difference(
        self,
        fragments: pd.DataFrame,
        rt_predictions: pd.DataFrame,
        visualize: bool = False,  # <-- NEW
        top_k: int = 6,  # visual overlay of up to top_k fragment traces
    ) -> float:
        """
        Square root of absolute difference between observed apex RT and predicted RT.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data with at least ['rt','fragment_names','fragment_intensity','peptide'].
        rt_predictions : pd.DataFrame
            Must include columns ['peptide','rt_predictions'].
        visualize : bool, default=False
            If True, overlays top-k fragment chromatograms with vertical lines for
            apex RT and predicted RT, and displays |ΔRT| and sqrt(|ΔRT|).
        top_k : int, default=6
            Maximum number of fragment traces to overlay when visualize=True.

        Returns
        -------
        float
            sqrt(|apex_rt - predicted_rt|), or NaN if unavailable.
        """
        try:
            apex_rt = self.feature_rt_apex(fragments, visualize=False)

            if pd.isna(apex_rt):
                return np.nan

            if "peptide" not in fragments.columns:
                logger.warning("Column 'peptide' missing in fragments")
                return np.nan
            peptide = fragments["peptide"].iloc[0]

            required_cols = {"peptide", "rt_predictions"}
            if not required_cols.issubset(set(rt_predictions.columns)):
                logger.warning("Required columns missing in RT predictions")
                return np.nan

            pred_row = rt_predictions[rt_predictions["peptide"] == peptide]
            if pred_row.empty:
                logger.warning(f"No RT prediction found for peptide {peptide}")
                return np.nan
            pred_rt = float(pred_row["rt_predictions"].iloc[0])

            diff = abs(apex_rt - pred_rt)
            result = float(np.sqrt(diff))

            if visualize:
                # Build pivot of RT × fragment_names for chromatograms
                frags_val = self._validate_fragments_input(fragments)
                piv = (
                    frags_val.pivot_table(
                        index="rt",
                        columns="fragment_names",
                        values="fragment_intensity",
                        aggfunc="mean",
                    )
                    .sort_index()
                    .fillna(0.0)
                )

                # Choose up to top_k fragments by their max intensity
                frag_max = piv.max(axis=0).sort_values(ascending=False)
                chosen = frag_max.index[:top_k]

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # (i) Overlay chromatograms with apex & prediction
                for col in chosen:
                    axes[0].plot(
                        piv.index.values, piv[col].values, label=col, alpha=0.85
                    )
                axes[0].axvline(
                    apex_rt,
                    linestyle="--",
                    linewidth=2,
                    label=f"Apex RT = {apex_rt:.4f}",
                )
                axes[0].axvline(
                    pred_rt,
                    linestyle=":",
                    linewidth=2,
                    label=f"Pred RT = {pred_rt:.4f}",
                )
                axes[0].set_title("Top fragment chromatograms with RT markers")
                axes[0].set_xlabel("RT")
                axes[0].set_ylabel("Intensity")
                axes[0].legend(fontsize=8, ncol=1, loc="best")

                # (ii) Simple gauge of |ΔRT| and sqrt(|ΔRT|)
                axes[1].bar(["|ΔRT|", "sqrt(|ΔRT|)"], [diff, result])
                axes[1].set_title(f"RT difference for peptide: {peptide}")
                axes[1].set_ylabel("Time units")

                plt.tight_layout()
                plt.show()

            return result

        except Exception as e:
            logger.error(f"Error calculating RT prediction difference: {e}")
            return np.nan

    # Feature Group 8: Elution Profile Shape
    def feature_scanning_window_splits(
        self,
        fragments: pd.DataFrame,
        splits: int = 5,
        visualize: bool = False,  # <-- NEW toggle
    ) -> np.ndarray:
        """
        Relative intensities in scanning-window splits for the best fragment.

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data. Must include ['rt', 'fragment_names', 'fragment_intensity'].
        splits : int, default=5
            Number of equal RT segments to divide the best-fragment window into.
        visualize : bool, default=False
            If True, render:
            (i) the best-fragment chromatogram with shaded RT segments and percentages,
            (ii) a bar chart of the relative intensities per segment.

        Returns
        -------
        np.ndarray
            Array of length `splits` with relative intensities per RT segment,
            summing to 1.0 (or all zeros on error).
        """
        try:
            if splits <= 0:
                raise ValueError("`splits` must be a positive integer.")

            best_fragment = self.find_best_fragment(fragments)
            frag_data = fragments[fragments["fragment_names"] == best_fragment]

            if frag_data.empty:
                if visualize:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.set_title("Best fragment not found or empty")
                    ax.set_axis_off()
                    plt.tight_layout()
                    plt.show()
                return np.zeros(splits, dtype=float)

            # Sort by RT to ensure proper plotting/segmenting
            frag_data = frag_data.sort_values("rt")
            rt_min = float(frag_data["rt"].min())
            rt_max = float(frag_data["rt"].max())
            rt_range = rt_max - rt_min

            # Degenerate window (all points at the same RT)
            if rt_range == 0.0:
                result = np.zeros(splits, dtype=float)
                result[0] = 1.0  # put all signal in the first bin by convention
                if visualize:
                    # Minimal plot
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                    # (i) chromatogram (single vertical line)
                    axes[0].plot(
                        frag_data["rt"], frag_data["fragment_intensity"], marker="o"
                    )
                    axes[0].set_title(
                        f"Best fragment: {best_fragment}\n(degenerate RT window)"
                    )
                    axes[0].set_xlabel("RT")
                    axes[0].set_ylabel("Intensity")
                    axes[0].axvline(rt_min, linestyle="--", linewidth=2)

                    # (ii) bar chart of relative intensities
                    axes[1].bar([str(i + 1) for i in range(splits)], result)
                    axes[1].set_ylim(0, 1.05)
                    axes[1].set_title("Relative intensities per segment")
                    axes[1].set_xlabel("Segment")
                    axes[1].set_ylabel("Relative intensity")

                    plt.tight_layout()
                    plt.show()
                return result

            # Compute equal-width segment boundaries
            bounds = np.linspace(rt_min, rt_max, splits + 1)
            split_intensities = []

            # Sum intensity within each segment
            for i in range(splits):
                left = bounds[i]
                right = bounds[i + 1]
                if i == splits - 1:
                    mask = (frag_data["rt"] >= left) & (frag_data["rt"] <= right)
                else:
                    mask = (frag_data["rt"] >= left) & (frag_data["rt"] < right)
                split_intensities.append(
                    float(frag_data.loc[mask, "fragment_intensity"].sum())
                )

            # Normalize to relative intensities
            total_intensity = float(np.sum(split_intensities))
            if total_intensity > 0.0:
                rel = np.asarray(split_intensities, dtype=float) / total_intensity
            else:
                # Fallback: equal distribution if there is no signal at all
                rel = np.full(splits, 1.0 / splits, dtype=float)

            # -------------------- Visualization (optional) --------------------
            if visualize:
                # (i) Chromatogram with shaded segments and percentage labels
                fig, axes = plt.subplots(1, 2, figsize=(13, 4))

                axes[0].plot(
                    frag_data["rt"].to_numpy(dtype=float),
                    frag_data["fragment_intensity"].to_numpy(dtype=float),
                    label=f"Best fragment: {best_fragment}",
                    alpha=0.9,
                )
                ymax = (
                    float(frag_data["fragment_intensity"].max())
                    if not frag_data.empty
                    else 1.0
                )
                for i in range(splits):
                    axes[0].axvspan(bounds[i], bounds[i + 1], color="gray", alpha=0.1)
                    # place text at 90% of max intensity within segment center
                    x_mid = 0.5 * (bounds[i] + bounds[i + 1])
                    axes[0].text(
                        x_mid,
                        0.9 * ymax,
                        f"{rel[i]*100:.1f}%",
                        ha="center",
                        va="top",
                        fontsize=9,
                    )
                axes[0].set_title("Best-fragment chromatogram with RT segments")
                axes[0].set_xlabel("RT")
                axes[0].set_ylabel("Intensity")
                axes[0].legend(fontsize=8, loc="best")

                # (ii) Bar chart of relative intensities
                axes[1].bar([str(i + 1) for i in range(splits)], rel)
                axes[1].set_ylim(0, 1.05)
                axes[1].set_title("Relative intensities per segment")
                axes[1].set_xlabel("Segment (1..N)")
                axes[1].set_ylabel("Relative intensity (sum = 1)")

                plt.tight_layout()
                plt.show()
            # -----------------------------------------------------------------

            return rel

        except Exception as e:
            logger.error(f"Error calculating scanning window splits: {e}")
            return np.zeros(splits, dtype=float)

    # Feature Group 10: Library Characteristics
    def feature_relative_predicted_intensities(
        self,
        fragments: pd.DataFrame,
        predictions: Dict[str, Dict[str, float]],
        top_n: int = 12,
        visualize: bool = False,  # <-- NEW toggle
    ) -> np.ndarray:
        """
        Relative predicted intensities for fragments 2..top_n (top-1 excluded).

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data; must contain ['peptide','charge'].
        predictions : dict
            Mapping { "PEPTIDE/charge": {frag_key: predicted_intensity, ...}, ... }.
        top_n : int, default=12
            Number of highest predicted fragments to consider (first is excluded in the output).
        visualize : bool, default=False
            If True, render:
            (i) bar chart of normalized predictions for top_n (top highlighted),
            (ii) bar chart of the returned vector (positions 2..top_n).

        Returns
        -------
        np.ndarray
            Array of length (top_n-1) with normalized predicted intensities
            for ranks 2..top_n (NaN-padded if fewer are available).
        """
        try:
            if top_n < 2:
                raise ValueError(
                    "`top_n` must be ≥ 2 since the top-1 is excluded from the output."
                )

            peptide = fragments["peptide"].iloc[0]
            charge = fragments["charge"].iloc[0]
            proforma = f"{peptide}/{charge}"

            if proforma not in predictions:
                logger.warning(f"No predictions found for {proforma}")
                result = np.full(top_n - 1, np.nan, dtype=float)
                if visualize:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.set_title(f"No MS2 predictions for {proforma}")
                    ax.set_axis_off()
                    plt.tight_layout()
                    plt.show()
                return result

            # Convert prediction dict to a Series and sort descending
            pred_series = pd.Series(predictions[proforma], dtype=float).sort_values(
                ascending=False
            )

            # Require at least two items (since we exclude the top one from the return vector)
            if pred_series.size < 2:
                return np.full(top_n - 1, np.nan, dtype=float)

            # Take the top_n (or fewer if not available)
            top_preds = pred_series.head(top_n)
            max_intensity = float(top_preds.max()) if top_preds.size > 0 else 0.0

            # Normalize by maximum among the selected top_n
            if max_intensity > 0.0:
                normalized = top_preds / max_intensity
            else:
                normalized = top_preds * 0.0  # all zeros if max is zero

            # Exclude the top-1 and construct a fixed-length (top_n-1) vector
            vec = normalized.iloc[1:].to_numpy(dtype=float)  # ranks 2..top_n
            if vec.size < (top_n - 1):
                padded = np.full(top_n - 1, np.nan, dtype=float)
                padded[: vec.size] = vec
                vec = padded
            result = vec[: (top_n - 1)]

            # -------------------- Visualization (optional) --------------------
            if visualize:
                # Labels: use fragment keys if available, else rank labels
                labels_all = list(top_preds.index)
                labels_excl_top = labels_all[1:]

                fig, axes = plt.subplots(1, 2, figsize=(14, 4))

                # (i) All top_n normalized predictions (top highlighted)
                colors = ["tab:orange"] + ["tab:blue"] * (len(normalized) - 1)
                axes[0].bar(labels_all, normalized.values, color=colors)
                axes[0].set_ylim(0, 1.05)
                axes[0].set_title(
                    f"Top-{len(normalized)} predicted intensities (normalized)"
                )
                axes[0].set_xlabel("Fragment")
                axes[0].set_ylabel("Relative predicted intensity (max = 1)")
                axes[0].tick_params(axis="x", rotation=45, labelsize=8)

                # (ii) Returned vector (ranks 2..top_n)
                axes[1].bar(labels_excl_top[: (top_n - 1)], result)
                axes[1].set_ylim(0, 1.05)
                axes[1].set_title(f"Relative predicted intensities (ranks 2..{top_n})")
                axes[1].set_xlabel("Fragment")
                axes[1].set_ylabel("Relative predicted intensity")
                axes[1].tick_params(axis="x", rotation=45, labelsize=8)

                plt.tight_layout()
                plt.show()
            # -----------------------------------------------------------------

            return result

        except Exception as e:
            logger.error(f"Error calculating relative predicted intensities: {e}")
            return np.full(top_n - 1, np.nan, dtype=float)

    def feature_cos_pred_obs_weighted(
        self,
        fragments: pd.DataFrame,
        intensity_predictions: Dict[str, Dict[str, float]],
        top_n: int = 6,
        use_all_rt: bool = False,
        visualize: bool = False,
    ) -> np.ndarray:
        """
        Weighted cosine similarity between predicted and observed fragment intensities.

        This feature calculates cosine similarity averaged over RT with weights equal to
        b(t)^2 where b(t) is the smoothed elution profile of the 'best' fragment.
        Returns both the similarity (S) and its cubic power (S^3).

        Parameters
        ----------
        fragments : pd.DataFrame
            Fragment data containing columns:
            ['fragment_names', 'fragment_type', 'fragment_ordinals', 'fragment_charge',
             'fragment_intensity', 'rt', 'peptide', 'charge']
        intensity_predictions : dict
            MS2PiP predictions mapping peptide/charge to fragment predictions
        top_n : int, default=6
            Number of top fragments to use for cosine similarity calculation
        use_all_rt : bool, default=False
            Whether to use all RT points (filling missing with 0) or only overlapping points
        visualize : bool, default=False
            If True, render diagnostic plots

        Returns
        -------
        np.ndarray
            Array containing [cosine_similarity, cosine_similarity_cubed]
        """
        try:
            # Get peptide identifier for predictions lookup
            if fragments.empty:
                return np.array([0.0, 0.0])

            peptide = fragments["peptide"].iloc[0]
            charge = fragments["charge"].iloc[0]
            proforma = f"{peptide}/{charge}"

            if proforma not in intensity_predictions:
                logger.warning(f"No intensity predictions found for {proforma}")
                return np.array([0.0, 0.0])

            precursor_preds = intensity_predictions[proforma]

            # 1) Select top-N fragment names
            used_fragments = self.find_top_n_fragments(fragments, n=top_n)

            if len(used_fragments) == 0:
                return np.array([0.0, 0.0])

            # 2) Get best trace and smoothed trace
            best_trace, smoothed_best_trace, _ = self.calculate_pearson_correlations(
                fragments, use_all_rt=use_all_rt
            )

            if best_trace is None or len(best_trace) == 0:
                return np.array([0.0, 0.0])

            # Align everything to the RT grid of the best fragment
            rt_index = best_trace.index.to_numpy()
            b = np.asarray(smoothed_best_trace, dtype=np.float64)

            if b.shape[0] != rt_index.shape[0]:
                logger.error("Best trace and RT index length mismatch")
                return np.array([0.0, 0.0])

            # 3) Pivot to (rt × fragment) and align to best_trace.index
            piv = (
                fragments.pivot_table(
                    index="rt",
                    columns="fragment_names",
                    values="fragment_intensity",
                    aggfunc="sum",
                )
                .reindex(rt_index)
                .fillna(0.0)
            )

            # Restrict to the chosen fragments, in that order
            piv = piv.reindex(columns=used_fragments).fillna(0.0)

            # Measured matrix (K x T)
            measured = piv.to_numpy(dtype=np.float64).T  # shape (K, T)
            K, T = measured.shape

            if T == 0 or K == 0:
                return np.array([0.0, 0.0])

            # 4) Build predicted vector for the same K fragments
            predicted_vals = []

            # Get metadata for fragments to construct prediction keys
            meta = (
                fragments[fragments["fragment_names"].isin(used_fragments)]
                .sort_values("rt")
                .drop_duplicates(subset=["fragment_names"], keep="first")
                .set_index("fragment_names")
            )

            for frag_name in used_fragments:
                if frag_name in meta.index:
                    key = f"{meta.loc[frag_name, 'fragment_type']}{int(meta.loc[frag_name, 'fragment_ordinals'])}/{int(meta.loc[frag_name, 'fragment_charge'])}"
                    predicted_vals.append(float(precursor_preds.get(key, 0.0)))
                else:
                    predicted_vals.append(0.0)

            predicted = np.asarray(predicted_vals, dtype=np.float64)  # shape (K,)

            # Guard against all-zero predictions
            pred_norm = float(np.linalg.norm(predicted))
            if pred_norm == 0.0:
                return np.array([0.0, 0.0])

            # 5) Calculate cosine similarity per RT point
            cosine_by_rt = np.zeros(T, dtype=np.float64)
            for j in range(T):
                obs = measured[:, j]
                obs_norm = float(np.linalg.norm(obs))
                if obs_norm == 0.0:
                    cosine_by_rt[j] = 0.0
                else:
                    cosine_by_rt[j] = float(
                        np.dot(obs, predicted) / (obs_norm * pred_norm)
                    )

            # Numerical safety - clip to [0, 1]
            np.clip(cosine_by_rt, 0.0, 1.0, out=cosine_by_rt)

            # 6) Weighted average with b(t)^2
            w = b**2
            w_sum = float(w.sum())

            if w_sum == 0.0:
                return np.array([0.0, 0.0])

            S = float(np.sum(w * cosine_by_rt) / w_sum)
            S_cubed = S**3

            # Optional visualization
            if visualize:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                # Plot 1: Best fragment trace and weights
                axes[0, 0].plot(
                    rt_index, best_trace.values, "b-", label="Best fragment", alpha=0.7
                )
                axes[0, 0].plot(
                    rt_index, smoothed_best_trace, "r-", label="Smoothed", linewidth=2
                )
                axes[0, 0].set_xlabel("RT")
                axes[0, 0].set_ylabel("Intensity")
                axes[0, 0].set_title("Best Fragment Elution Profile")
                axes[0, 0].legend()

                # Plot 2: Weights (b^2)
                axes[0, 1].plot(rt_index, w, "g-", linewidth=2)
                axes[0, 1].set_xlabel("RT")
                axes[0, 1].set_ylabel("Weight (b²)")
                axes[0, 1].set_title("Weighting Function")

                # Plot 3: Per-RT cosine similarities
                axes[1, 0].plot(
                    rt_index, cosine_by_rt, "purple", marker="o", markersize=3
                )
                axes[1, 0].axhline(
                    y=S, color="red", linestyle="--", label=f"Weighted avg: {S:.3f}"
                )
                axes[1, 0].set_xlabel("RT")
                axes[1, 0].set_ylabel("Cosine Similarity")
                axes[1, 0].set_title("Per-RT Cosine Similarities")
                axes[1, 0].legend()
                axes[1, 0].set_ylim(0, 1)

                # Plot 4: Predicted vs observed intensities (relative scale)
                x_pos = np.arange(len(used_fragments))
                pred_arr = np.asarray(predicted, dtype=float)
                obs_max = measured.max(axis=1).astype(
                    float
                )  # Max observed intensity per fragment

                # Normalize to [0,1] by their respective maxima to make scales comparable
                pred_max = pred_arr.max() if pred_arr.size > 0 else 0.0
                obs_max_val = obs_max.max() if obs_max.size > 0 else 0.0

                pred_rel = (
                    pred_arr / pred_max if pred_max > 0 else np.zeros_like(pred_arr)
                )
                obs_rel = (
                    obs_max / obs_max_val if obs_max_val > 0 else np.zeros_like(obs_max)
                )

                axes[1, 1].bar(
                    x_pos - 0.2, pred_rel, 0.4, label="Predicted (rel.)", alpha=0.7
                )
                axes[1, 1].bar(
                    x_pos + 0.2, obs_rel, 0.4, label="Observed (max, rel.)", alpha=0.7
                )
                axes[1, 1].set_xlabel("Fragment")
                axes[1, 1].set_ylabel("Relative intensity (0..1)")
                axes[1, 1].set_title("Predicted vs Observed (relative)")
                axes[1, 1].set_xticks(x_pos)
                axes[1, 1].set_xticklabels(used_fragments, rotation=45)
                axes[1, 1].set_ylim(0, 1.05)
                axes[1, 1].legend()

                plt.tight_layout()
                plt.suptitle(
                    f"Cosine Similarity Analysis (S={S:.3f}, S³={S_cubed:.3f})", y=1.02
                )
                plt.show()

            return np.array([S, S_cubed])

        except Exception as e:
            logger.error(f"Error calculating weighted cosine similarity: {e}")
            return np.array([np.nan, np.nan])

    def feature_precursor_mz(self, precursor: pd.DataFrame) -> float:
        """Calculate precursor m/z."""
        try:
            return (
                precursor["calcmass"].iloc[0] / precursor["charge"].iloc[0] + 1.007276
            )
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
                precursor,
                fragments,
                ms1_dict,
                ms2dict,
                rt_predictions,
                intensity_predictions,
            )
        else:
            return self._calculate_all_features_sequential(
                precursor,
                fragments,
                ms1_dict,
                ms2dict,
                rt_predictions,
                intensity_predictions,
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
        tasks.extend(
            [
                (
                    "pearson_correlations_top_12",
                    self.feature_pearson_correlations_top_n,
                    (fragments,),
                ),
                (
                    "sum_correlations_mass_accuracy",
                    self.feature_sum_correlations_mass_accuracy,
                    (fragments,),
                ),
                (
                    "remaining_fragments_correlations",
                    self.feature_remaining_fragments_correlations,
                    (fragments,),
                ),
                (
                    "best_b_fragments_correlation",
                    self.feature_best_b_fragments_correlation,
                    (fragments,),
                ),
            ]
        )

        if ms1_dict is not None:
            tasks.append(
                (
                    "precursor_best_fragment_correlation",
                    self.feature_precursor_best_fragment_correlation,
                    (precursor, fragments, ms1_dict),
                )
            )

            # Group 2: MS1 level co-elution
            tasks.append(
                (
                    "ms1_accuracy_correlations",
                    self.feature_ms1_accuracy_correlations,
                    (precursor, fragments, ms1_dict),
                )
            )

            # Group 3: Isotopologue co-elution
            tasks.append(
                (
                    "c13_isotope_correlations",
                    self.feature_c13_isotope_correlations,
                    (precursor, fragments, ms1_dict),
                )
            )

        if ms2dict is not None:
            tasks.extend(
                [
                    (
                        "c13_subtracted_correlations",
                        self.feature_c13_subtracted_correlations,
                        (fragments, ms2dict),
                    ),
                    (
                        "sum_c13_subtracted_correlations",
                        self.feature_sum_c13_subtracted_correlations,
                        (fragments, ms2dict),
                    ),
                ]
            )

        # Group 4-10: Other features
        tasks.extend(
            [
                ("weighted_auc", self.feature_weighted_auc, (fragments,)),
                (
                    "relative_intensities_top_6",
                    self.feature_relative_intensities_top_6,
                    (fragments,),
                ),
                (
                    "weighted_mass_accuracy",
                    self.feature_weighted_mass_accuracy,
                    (fragments,),
                ),
                ("rt_apex", self.feature_rt_apex, (fragments,)),
                (
                    "scanning_window_splits",
                    self.feature_scanning_window_splits,
                    (fragments,),
                ),
                ("precursor_mz", self.feature_precursor_mz, (precursor,)),
                ("precursor_charge", self.feature_precursor_charge, (precursor,)),
                ("precursor_length", self.feature_precursor_length, (precursor,)),
            ]
        )

        if rt_predictions is not None:
            tasks.append(
                (
                    "rt_prediction_difference",
                    self.feature_rt_prediction_difference,
                    (fragments, rt_predictions),
                )
            )

        if intensity_predictions is not None:
            tasks.extend(
                [
                    (
                        "relative_predicted_intensities",
                        self.feature_relative_predicted_intensities,
                        (fragments, intensity_predictions),
                    ),
                    (
                        "library_fragment_count",
                        self.feature_library_fragment_count,
                        (precursor, intensity_predictions),
                    ),
                ]
            )

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

        logger.debug(f"Calculated {len(features)} feature groups in parallel")
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

        logger.debug("Calculating DIA-NN features...")

        # Group 1: Ion co-elution (MS2 level)
        try:
            features[
                "pearson_correlations_top_12"
            ] = self.feature_pearson_correlations_top_n(fragments)
            features[
                "sum_correlations_mass_accuracy"
            ] = self.feature_sum_correlations_mass_accuracy(fragments)
            features[
                "remaining_fragments_correlations"
            ] = self.feature_remaining_fragments_correlations(fragments)
            features[
                "best_b_fragments_correlation"
            ] = self.feature_best_b_fragments_correlation(fragments)

            if ms1_dict is not None:
                features[
                    "precursor_best_fragment_correlation"
                ] = self.feature_precursor_best_fragment_correlation(
                    precursor, fragments, ms1_dict
                )
        except Exception as e:
            logger.error(f"Error in group 1 features: {e}")

        # Group 2: MS1 level co-elution
        if ms1_dict is not None:
            try:
                features[
                    "ms1_accuracy_correlations"
                ] = self.feature_ms1_accuracy_correlations(
                    precursor, fragments, ms1_dict
                )
            except Exception as e:
                logger.error(f"Error in group 2 features: {e}")

        # Group 3: Isotopologue co-elution
        if ms1_dict is not None:
            try:
                features[
                    "c13_isotope_correlations"
                ] = self.feature_c13_isotope_correlations(
                    precursor, fragments, ms1_dict
                )
            except Exception as e:
                logger.error(f"Error in group 3.1 features: {e}")

        if ms2dict is not None:
            try:
                features[
                    "c13_subtracted_correlations"
                ] = self.feature_c13_subtracted_correlations(fragments, ms2dict)
                features[
                    "sum_c13_subtracted_correlations"
                ] = self.feature_sum_c13_subtracted_correlations(fragments, ms2dict)
            except Exception as e:
                logger.error(f"Error in group 3.3-3.4 features: {e}")

        # Group 4: Total signal
        try:
            features["weighted_auc"] = self.feature_weighted_auc(fragments)
        except Exception as e:
            logger.error(f"Error in group 4 features: {e}")

        # Group 5: Fragment intensities
        try:
            features[
                "relative_intensities_top_6"
            ] = self.feature_relative_intensities_top_6(fragments)
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
                features[
                    "rt_prediction_difference"
                ] = self.feature_rt_prediction_difference(fragments, rt_predictions)
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
                features[
                    "relative_predicted_intensities"
                ] = self.feature_relative_predicted_intensities(
                    fragments, intensity_predictions
                )
                features[
                    "library_fragment_count"
                ] = self.feature_library_fragment_count(
                    precursor, intensity_predictions
                )
        except Exception as e:
            logger.error(f"Error in group 10 features: {e}")

        logger.debug(f"Calculated {len(features)} feature groups")
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
