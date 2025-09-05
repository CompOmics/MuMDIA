"""
Test script for DIANNFeatureGenerator

This script demonstrates how to use the DIANNFeatureGenerator class
with real data from the MuMDIA pipeline.
"""

import pandas as pd
import pickle
import numpy as np
from diann_feature_generator import DIANNFeatureGenerator, FeatureConfig
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import os
import psutil
import warnings


# Suppress RuntimeWarnings from numpy
np.seterr(all="ignore")
# Suppress RuntimeWarnings Degrees of freedom
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Global variables for multiprocessing workers (to avoid serializing large data repeatedly)
_global_data = None
_global_config = None


def _init_worker(data_dict, config_dict):
    """Initialize worker process with shared data."""
    global _global_data, _global_config
    _global_data = data_dict
    _global_config = config_dict


def process_single_precursor_efficient(precursor_id):
    """
    Efficient worker function for processing a single precursor in parallel.
    Uses global data to avoid serialization overhead.

    Parameters
    ----------
    precursor_id : str
        String identifier for the precursor

    Returns
    -------
    tuple or None
        (precursor_id, feature_row) if successful, None if failed
    """
    global _global_data, _global_config

    try:
        # Recreate configuration object
        config = FeatureConfig(**_global_config)
        generator = DIANNFeatureGenerator(config)

        # Extract precursor and fragment data
        precursor = _global_data["df_psms"][
            _global_data["df_psms"]["precursor"] == precursor_id
        ]
        if precursor.empty:
            return None

        precursor_fragments = _global_data["df_fragment"][
            _global_data["df_fragment"]["psm_id"].isin(precursor["psm_id"])
        ]

        if precursor_fragments.empty:
            return None

        # Calculate PPM error if not present
        if "ppm_error" not in precursor_fragments.columns:
            precursor_fragments = precursor_fragments.copy()
            precursor_fragments["ppm_error"] = (
                abs(
                    precursor_fragments["fragment_mz_calculated"]
                    - precursor_fragments["fragment_mz_experimental"]
                )
                / precursor_fragments["fragment_mz_calculated"]
                * 1e6
            )

        # Calculate features with parallel=False to avoid nested parallelization
        features = generator.calculate_all_features(
            precursor=precursor,
            fragments=precursor_fragments,
            ms1_dict=_global_data["ms1_dict"],
            ms2dict=_global_data["ms2dict"],
            rt_predictions=_global_data["deeplc_preds"],
            intensity_predictions=_global_data["ms2pip_preds"],
            parallel=False,  # Disable internal parallelization
        )

        if not features:
            return None

        # Create feature row with PSM information
        feature_row = {
            "PSMId": precursor["psm_id"].iloc[0],
            "Label": 1,  # Assuming all are positive for now
            "ScanNr": (
                precursor["scannr"].iloc[0] if "scannr" in precursor.columns else 0
            ),
            "Peptide": precursor["peptide"].iloc[0],
            "Proteins": (
                precursor.get("proteins", ["unknown"]).iloc[0]
                if "proteins" in precursor.columns
                else "unknown"
            ),
        }

        # Add calculated features with flattening for arrays
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, np.ndarray):
                # Flatten arrays into individual columns
                if feature_value.size == 1:
                    feature_row[feature_name] = float(feature_value.item())
                else:
                    for j, val in enumerate(feature_value):
                        feature_row[f"{feature_name}_{j+1}"] = (
                            float(val) if not np.isnan(val) else 0.0
                        )
            else:
                # Convert to float, replace NaN with 0
                if pd.isna(feature_value):
                    feature_row[feature_name] = 0.0
                else:
                    feature_row[feature_name] = float(feature_value)

        return (precursor_id, feature_row)

    except Exception:
        return None


def process_single_precursor(args):
    """
    Worker function for processing a single precursor in parallel.

    Parameters
    ----------
    args : tuple
        (precursor_id, data_dict, config_dict) where:
        - precursor_id: string identifier for the precursor
        - data_dict: dictionary containing all loaded data
        - config_dict: configuration parameters as dictionary

    Returns
    -------
    tuple or None
        (precursor_id, feature_row) if successful, None if failed
    """
    precursor_id, data_dict, config_dict = args

    try:
        # Recreate configuration object
        config = FeatureConfig(**config_dict)
        generator = DIANNFeatureGenerator(config)

        # Extract precursor and fragment data
        precursor = data_dict["df_psms"][
            data_dict["df_psms"]["precursor"] == precursor_id
        ]
        if precursor.empty:
            return None

        precursor_fragments = data_dict["df_fragment"][
            data_dict["df_fragment"]["psm_id"].isin(precursor["psm_id"])
        ]

        if precursor_fragments.empty:
            return None

        # Calculate PPM error if not present
        if "ppm_error" not in precursor_fragments.columns:
            precursor_fragments = precursor_fragments.copy()
            precursor_fragments["ppm_error"] = (
                abs(
                    precursor_fragments["fragment_mz_calculated"]
                    - precursor_fragments["fragment_mz_experimental"]
                )
                / precursor_fragments["fragment_mz_calculated"]
                * 1e6
            )

        # Calculate features with parallel=False to avoid nested parallelization
        features = generator.calculate_all_features(
            precursor=precursor,
            fragments=precursor_fragments,
            ms1_dict=data_dict["ms1_dict"],
            ms2dict=data_dict["ms2dict"],
            rt_predictions=data_dict["deeplc_preds"],
            intensity_predictions=data_dict["ms2pip_preds"],
            parallel=False,  # Disable internal parallelization
        )

        if not features:
            return None

        # Create feature row with PSM information
        feature_row = {
            "PSMId": precursor["psm_id"].iloc[0],
            "Label": 1,  # Assuming all are positive for now
            "ScanNr": (
                precursor["scannr"].iloc[0] if "scannr" in precursor.columns else 0
            ),
            "Peptide": precursor["peptide"].iloc[0],
            "Proteins": (
                precursor.get("protein", ["unknown"]).iloc[0]
                if "protein" in precursor.columns
                else "unknown"
            ),
        }

        # Add calculated features with flattening for arrays
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, np.ndarray):
                # Flatten arrays into individual columns
                if feature_value.size == 1:
                    feature_row[feature_name] = float(feature_value.item())
                else:
                    for j, val in enumerate(feature_value):
                        feature_row[f"{feature_name}_{j+1}"] = (
                            float(val) if not np.isnan(val) else 0.0
                        )
            else:
                # Convert to float, replace NaN with 0
                if pd.isna(feature_value):
                    feature_row[feature_name] = 0.0
                else:
                    feature_row[feature_name] = float(feature_value)

        return (precursor_id, feature_row)

    except Exception:
        return None


def load_test_data():
    """Load test data from the MuMDIA pipeline."""
    try:
        # Load predictions
        ms2pip_preds = pickle.load(
            open(
                "/home/robbe/MuMDIA/results/config_playing/ms2pip_predictions.pkl", "rb"
            )
        )
        deeplc_preds = pickle.load(
            open(
                "/home/robbe/MuMDIA/results/config_playing/predictions_deeplc.pkl", "rb"
            )
        )

        # Load PSM and fragment data
        df_psms = pd.read_csv(
            "/home/robbe/MuMDIA/results/config_playing/df_psms.tsv", sep="\t"
        )
        df_psms["precursor"] = df_psms["peptide"] + "/" + df_psms["charge"].astype(str)

        df_fragment = pd.read_csv(
            "/home/robbe/MuMDIA/results/config_playing/df_fragment.tsv", sep="\t"
        )
        df_fragment["fragment_names"] = df_fragment["fragment_type"].astype(
            str
        ) + df_fragment["fragment_ordinals"].astype("Int64").astype(str)

        # Load spectral dictionaries
        ms2dict = pickle.load(open("/home/robbe/MuMDIA/debug/ms2dict.pkl", "rb"))
        ms1_dict = pickle.load(open("/home/robbe/MuMDIA/debug/ms1_dict.pkl", "rb"))

        # Convert DeepLC predictions to pandas if needed
        if hasattr(deeplc_preds, "to_pandas"):
            deeplc_preds = deeplc_preds.to_pandas()

        return {
            "ms2pip_preds": ms2pip_preds,
            "deeplc_preds": deeplc_preds,
            "df_psms": df_psms,
            "df_fragment": df_fragment,
            "ms2dict": ms2dict,
            "ms1_dict": ms1_dict,
        }

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None


def test_single_precursor(data, precursor_id="GYINSLGALTGGQALQQAK/2"):
    """Test feature generation for a single precursor."""

    logger.info(f"Testing feature generation for precursor: {precursor_id}")

    # Extract precursor and fragment data
    precursor = data["df_psms"][data["df_psms"]["precursor"] == precursor_id]
    if precursor.empty:
        logger.error(f"Precursor {precursor_id} not found")
        return None

    precursor_fragments = data["df_fragment"][
        data["df_fragment"]["psm_id"].isin(precursor["psm_id"])
    ]

    if precursor_fragments.empty:
        logger.error(f"No fragments found for precursor {precursor_id}")
        return None

    # Calculate PPM error if not present
    if "ppm_error" not in precursor_fragments.columns:
        precursor_fragments = precursor_fragments.copy()
        precursor_fragments["ppm_error"] = (
            abs(
                precursor_fragments["fragment_mz_calculated"]
                - precursor_fragments["fragment_mz_experimental"]
            )
            / precursor_fragments["fragment_mz_calculated"]
            * 1e6
        )

    logger.info(f"Found {len(precursor_fragments)} fragment observations")

    # Create configuration
    config = FeatureConfig(
        fragment_mass_tolerance=13.0,
        precursor_mass_tolerance=50.0,
        rt_tolerance=5.0,
        top_n_fragments=6,
        top_n_fragments_extended=12,
    )

    # Initialize feature generator
    generator = DIANNFeatureGenerator(config)

    # Calculate all features
    features = generator.calculate_all_features(
        precursor=precursor,
        fragments=precursor_fragments,
        ms1_dict=data["ms1_dict"],
        ms2dict=data["ms2dict"],
        rt_predictions=data["deeplc_preds"],
        intensity_predictions=data["ms2pip_preds"],
    )

    return features


def print_feature_summary(features):
    """Print a summary of calculated features."""
    print("\n" + "=" * 50)
    print("FEATURE CALCULATION SUMMARY")
    print("=" * 50)

    for feature_name, feature_value in features.items():
        print(f"\n{feature_name}:")
        if isinstance(feature_value, np.ndarray):
            if feature_value.size <= 10:
                print(f"  Values: {feature_value}")
            else:
                print(f"  Shape: {feature_value.shape}")
                print(f"  First 5: {feature_value[:5]}")
                print(f"  Last 5: {feature_value[-5:]}")
            print(
                f"  Stats: mean={np.nanmean(feature_value):.4f}, "
                f"std={np.nanstd(feature_value):.4f}"
            )
        else:
            print(f"  Value: {feature_value}")

    print("\n" + "=" * 50)


def test_multiple_precursors(data, max_precursors=5):
    """Test feature generation for multiple precursors."""

    logger.info(f"Testing feature generation for up to {max_precursors} precursors")

    # Get unique precursors
    unique_precursors = data["df_psms"]["precursor"].unique()[:max_precursors]

    results = {}

    for precursor_id in unique_precursors:
        try:
            logger.info(f"Processing {precursor_id}")
            features = test_single_precursor(data, precursor_id)
            if features:
                results[precursor_id] = features
            else:
                logger.warning(f"Failed to calculate features for {precursor_id}")
        except Exception as e:
            logger.error(f"Error processing {precursor_id}: {e}")

    return results


def validate_features(features):
    """Validate calculated features."""
    issues = []

    for feature_name, feature_value in features.items():
        if isinstance(feature_value, np.ndarray):
            if np.all(np.isnan(feature_value)):
                issues.append(f"{feature_name}: All values are NaN")
            elif np.any(np.isinf(feature_value)):
                issues.append(f"{feature_name}: Contains infinite values")
        elif pd.isna(feature_value):
            issues.append(f"{feature_name}: Value is NaN")
        elif np.isinf(feature_value):
            issues.append(f"{feature_name}: Value is infinite")

    return issues


def benchmark_performance(data, num_tests=10):
    """Benchmark feature calculation performance."""
    import time

    logger.info(f"Benchmarking performance with {num_tests} tests")

    # Get test precursors
    test_precursors = data["df_psms"]["precursor"].unique()[:num_tests]

    times = []

    for precursor_id in test_precursors:
        start_time = time.time()

        try:
            features = test_single_precursor(data, precursor_id)
            if features:
                elapsed = time.time() - start_time
                times.append(elapsed)
                logger.info(f"{precursor_id}: {elapsed:.3f}s")
        except Exception as e:
            logger.error(f"Error benchmarking {precursor_id}: {e}")

    if times:
        print("\nPerformance Summary:")
        print(f"  Average time: {np.mean(times):.3f}s")
        print(f"  Min time: {np.min(times):.3f}s")
        print(f"  Max time: {np.max(times):.3f}s")
        print(f"  Total time: {np.sum(times):.3f}s")


def output_features_to_file(
    output_path, n_jobs=-1, chunk_size=100, detailed_monitoring=False
):
    """
    Calculate features for all precursors and output in Percolator format using multiprocessing.

    Parameters
    ----------
    output_path : str
        Path to output .pin file
    n_jobs : int, default -1
        Number of parallel processes to use. -1 means use all CPU cores.
    chunk_size : int, default 100
        Number of precursors to process in each chunk
    detailed_monitoring : bool, default False
        Whether to enable detailed system monitoring (may cause issues on some systems)
    """
    import time

    # Temporarily disable logging
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.CRITICAL)

    try:
        print("Loading data for feature calculation...")
        start_time = time.time()
        data = load_test_data()

        if data is None:
            print("Failed to load data for feature calculation.")
            return

        # Show memory usage
        memory_info = psutil.virtual_memory()
        print(
            f"Memory usage: {memory_info.used / 1024**3:.1f}GB / {memory_info.total / 1024**3:.1f}GB ({memory_info.percent:.1f}%)"
        )

        # Create configuration
        config = FeatureConfig(
            fragment_mass_tolerance=13.0,
            precursor_mass_tolerance=50.0,
            rt_tolerance=5.0,
            top_n_fragments=6,
            top_n_fragments_extended=12,
            n_jobs=1,  # Disable internal parallelization since we're parallelizing at precursor level
        )

        # Convert config to dict for serialization
        config_dict = {
            "fragment_mass_tolerance": config.fragment_mass_tolerance,
            "precursor_mass_tolerance": config.precursor_mass_tolerance,
            "rt_tolerance": config.rt_tolerance,
            "top_n_fragments": config.top_n_fragments,
            "top_n_fragments_extended": config.top_n_fragments_extended,
            "savgol_window_length": config.savgol_window_length,
            "savgol_polyorder": config.savgol_polyorder,
            "isotope_mass_c13": config.isotope_mass_c13,
            "c13_isotope_list": config.c13_isotope_list,
            "ms1_accuracy_factors": config.ms1_accuracy_factors,
            "ms2_accuracy_factors": config.ms2_accuracy_factors,
            "n_jobs": 1,
        }

        # Get all unique precursors
        unique_precursors = data["df_psms"]["precursor"].unique()
        print(f"Processing {len(unique_precursors)} precursors...")

        # Determine number of workers - use a conservative limit for better efficiency
        if n_jobs == -1:
            # Use a reasonable number of workers based on system characteristics
            # Too many processes can cause overhead and memory issues
            physical_cores = psutil.cpu_count(logical=False) or 8
            n_workers = min(physical_cores, 64)  # Cap at 64 for efficiency
        else:
            n_workers = min(n_jobs, 64)  # Cap user setting at 64

        print(f"Using {n_workers} processes for parallel computation")
        print(
            f"  (Physical cores: {psutil.cpu_count(logical=False)}, Logical cores: {psutil.cpu_count()})"
        )
        print(f"  Chunk size: {chunk_size}")

        # Monitor CPU usage before starting
        cpu_percent_before = psutil.cpu_percent(interval=1)
        print(f"  CPU usage before: {cpu_percent_before:.1f}%")

        # Get list of precursor IDs only (avoid serializing large data)
        precursor_ids = unique_precursors.tolist()

        all_features = []
        failed_count = 0

        # Process in parallel using multiprocessing with efficient data sharing
        print("Starting parallel processing...")
        process_start = time.time()

        with Pool(
            processes=n_workers, initializer=_init_worker, initargs=(data, config_dict)
        ) as pool:
            # Use imap for progress tracking with efficient worker function
            results = []
            processed_count = 0

            # Start CPU monitoring in background
            import threading

            stop_monitoring = threading.Event()
            cpu_usage_samples = []

            def monitor_cpu():
                import time

                while not stop_monitoring.is_set():
                    try:
                        cpu_percent = psutil.cpu_percent(interval=0.5)
                        cpu_usage_samples.append(cpu_percent)
                    except Exception:
                        # If CPU monitoring fails, just skip this sample
                        time.sleep(0.5)
                        continue

            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.start()

            for result in tqdm(
                pool.imap(
                    process_single_precursor_efficient,
                    precursor_ids,
                    chunksize=chunk_size,
                ),
                total=len(precursor_ids),
                desc="Calculating features",
            ):
                results.append(result)
                processed_count += 1

            # Stop monitoring
            stop_monitoring.set()
            monitor_thread.join(timeout=1)

        process_time = time.time() - process_start

        # Calculate CPU usage statistics
        if cpu_usage_samples:
            avg_cpu = sum(cpu_usage_samples) / len(cpu_usage_samples)
            max_cpu = max(cpu_usage_samples)
            print(f"Parallel processing completed in {process_time:.2f} seconds")
            print(
                f"CPU usage during processing: avg={avg_cpu:.1f}%, max={max_cpu:.1f}%"
            )
            print(f"Estimated speedup vs single core: {avg_cpu/100 * n_workers:.1f}x")
        else:
            print(f"Parallel processing completed in {process_time:.2f} seconds")

        # Collect successful results
        for result in results:
            if result is not None:
                precursor_id, feature_row = result
                all_features.append(feature_row)
            else:
                failed_count += 1

        print(
            f"Successfully processed {len(all_features)} precursors, {failed_count} failed"
        )

        if not all_features:
            print("No features calculated. Cannot create output file.")
            return

        # Convert to DataFrame
        feature_df = pd.DataFrame(all_features)

        # Write Percolator format file
        print(f"Writing features to {output_path}...")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Percolator .pin format: tab-separated with specific column order
        # Standard columns: PSMId, Label, ScanNr, followed by features, then Peptide, Proteins
        standard_cols = ["PSMId", "Label", "ScanNr"]
        feature_cols = [
            col
            for col in feature_df.columns
            if col not in ["PSMId", "Label", "ScanNr", "Peptide", "Proteins"]
        ]
        end_cols = ["Peptide", "Proteins"]

        # Reorder columns
        column_order = standard_cols + sorted(feature_cols) + end_cols
        feature_df = feature_df.reindex(columns=column_order)

        # Write to file
        feature_df.to_csv(output_path, sep="\t", index=False, na_rep="0.0")

        print(f"Successfully wrote {len(feature_df)} feature vectors to {output_path}")
        print(f"Feature columns: {len(feature_cols)}")
        print(f"Sample feature names: {sorted(feature_cols)[:10]}")

    except Exception as e:
        print(f"Error in feature calculation: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Restore original logging level
        logging.getLogger().setLevel(original_level)


def output_features_to_file_sequential(output_path, n=100):
    """
    Calculate features for all precursors and output in Percolator format (sequential version).

    Parameters
    ----------
    output_path : str
        Path to output .pin file
    n : int
        Number of precursors to process (for testing)
    """
    # Temporarily disable logging
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.CRITICAL)

    try:
        print("Loading data for feature calculation...")
        data = load_test_data()

        if data is None:
            print("Failed to load data for feature calculation.")
            return

        # Create configuration
        config = FeatureConfig(
            fragment_mass_tolerance=13.0,
            precursor_mass_tolerance=50.0,
            rt_tolerance=5.0,
            top_n_fragments=6,
            top_n_fragments_extended=12,
        )

        # Initialize feature generator
        generator = DIANNFeatureGenerator(config)

        # Get all unique precursors
        unique_precursors = data["df_psms"]["precursor"].unique()[:n]
        print(f"Processing {len(unique_precursors)} precursors...")

        all_features = []
        failed_count = 0

        # Use tqdm for progress bar
        for i, precursor_id in enumerate(
            tqdm(unique_precursors, desc="Calculating features")
        ):
            try:
                # Extract precursor and fragment data
                precursor = data["df_psms"][
                    data["df_psms"]["precursor"] == precursor_id
                ]
                if precursor.empty:
                    failed_count += 1
                    continue

                precursor_fragments = data["df_fragment"][
                    data["df_fragment"]["psm_id"].isin(precursor["psm_id"])
                ]

                if precursor_fragments.empty:
                    failed_count += 1
                    continue

                # Calculate PPM error if not present
                if "ppm_error" not in precursor_fragments.columns:
                    precursor_fragments = precursor_fragments.copy()
                    precursor_fragments["ppm_error"] = (
                        abs(
                            precursor_fragments["fragment_mz_calculated"]
                            - precursor_fragments["fragment_mz_experimental"]
                        )
                        / precursor_fragments["fragment_mz_calculated"]
                        * 1e6
                    )

                # Calculate features
                features = generator.calculate_all_features(
                    precursor=precursor,
                    fragments=precursor_fragments,
                    ms1_dict=data["ms1_dict"],
                    ms2dict=data["ms2dict"],
                    rt_predictions=data["deeplc_preds"],
                    intensity_predictions=data["ms2pip_preds"],
                )

                if features:
                    # Create feature row with PSM information
                    feature_row = {
                        "PSMId": precursor["psm_id"].iloc[0],
                        "Label": 1,  # Assuming all are positive for now
                        "ScanNr": (
                            precursor["scannr"].iloc[0]
                            if "scannr" in precursor.columns
                            else 0
                        ),
                        "Peptide": precursor["peptide"].iloc[0],
                        "Proteins": (
                            precursor.get("protein", ["unknown"]).iloc[0]
                            if "protein" in precursor.columns
                            else "unknown"
                        ),
                    }

                    # Add calculated features with flattening for arrays
                    for feature_name, feature_value in features.items():
                        if isinstance(feature_value, np.ndarray):
                            # Flatten arrays into individual columns
                            if feature_value.size == 1:
                                feature_row[feature_name] = float(feature_value.item())
                            else:
                                for j, val in enumerate(feature_value):
                                    feature_row[f"{feature_name}_{j+1}"] = (
                                        float(val) if not np.isnan(val) else 0.0
                                    )
                        else:
                            # Convert to float, replace NaN with 0
                            if pd.isna(feature_value):
                                feature_row[feature_name] = 0.0
                            else:
                                feature_row[feature_name] = float(feature_value)

                    all_features.append(feature_row)
                else:
                    failed_count += 1

            except Exception:
                failed_count += 1
                continue

        print(
            f"Successfully processed {len(all_features)} precursors, {failed_count} failed"
        )

        if not all_features:
            print("No features calculated. Cannot create output file.")
            return

        # Convert to DataFrame
        feature_df = pd.DataFrame(all_features)

        # Write Percolator format file
        print(f"Writing features to {output_path}...")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Percolator .pin format: tab-separated with specific column order
        # Standard columns: PSMId, Label, ScanNr, followed by features, then Peptide, Proteins
        standard_cols = ["PSMId", "Label", "ScanNr"]
        feature_cols = [
            col
            for col in feature_df.columns
            if col not in ["PSMId", "Label", "ScanNr", "Peptide", "Proteins"]
        ]
        end_cols = ["Peptide", "Proteins"]

        # Reorder columns
        column_order = standard_cols + sorted(feature_cols) + end_cols
        feature_df = feature_df.reindex(columns=column_order)

        # Write to file
        feature_df.to_csv(output_path, sep="\t", index=False, na_rep="0.0")

        print(f"Successfully wrote {len(feature_df)} feature vectors to {output_path}")
        print(f"Feature columns: {len(feature_cols)}")
        print(f"Sample feature names: {sorted(feature_cols)[:10]}")

    except Exception as e:
        print(f"Error in feature calculation: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Restore original logging level
        logging.getLogger().setLevel(original_level)


def benchmark_parallel_vs_sequential(n_precursors=100):
    """
    Benchmark parallel vs sequential processing performance.

    Parameters
    ----------
    n_precursors : int
        Number of precursors to process for comparison
    """
    import time

    print(f"\n{'='*60}")
    print(f"PERFORMANCE BENCHMARK: {n_precursors} precursors")
    print(f"{'='*60}")

    # Load data once
    print("Loading test data...")
    data = load_test_data()
    if data is None:
        print("Failed to load data.")
        return

    unique_precursors = data["df_psms"]["precursor"].unique()[:n_precursors]
    print(f"Benchmarking with {len(unique_precursors)} precursors")

    # Test sequential processing
    print("\n1. Sequential Processing...")
    start_time = time.time()

    config = FeatureConfig(
        fragment_mass_tolerance=13.0,
        precursor_mass_tolerance=50.0,
        rt_tolerance=5.0,
        top_n_fragments=6,
        top_n_fragments_extended=12,
    )

    generator = DIANNFeatureGenerator(config)

    sequential_results = []
    for precursor_id in tqdm(unique_precursors, desc="Sequential"):
        try:
            precursor = data["df_psms"][data["df_psms"]["precursor"] == precursor_id]
            if precursor.empty:
                continue

            precursor_fragments = data["df_fragment"][
                data["df_fragment"]["psm_id"].isin(precursor["psm_id"])
            ]
            if precursor_fragments.empty:
                continue

            features = generator.calculate_all_features(
                precursor=precursor,
                fragments=precursor_fragments,
                ms1_dict=data["ms1_dict"],
                ms2dict=data["ms2dict"],
                rt_predictions=data["deeplc_preds"],
                intensity_predictions=data["ms2pip_preds"],
                parallel=False,
            )

            if features:
                sequential_results.append(features)

        except Exception:
            continue

    sequential_time = time.time() - start_time

    # Test parallel processing
    print("\n2. Parallel Processing...")
    start_time = time.time()

    config_dict = {
        "fragment_mass_tolerance": 13.0,
        "precursor_mass_tolerance": 50.0,
        "rt_tolerance": 5.0,
        "top_n_fragments": 6,
        "top_n_fragments_extended": 12,
        "savgol_window_length": 3,
        "savgol_polyorder": 1,
        "isotope_mass_c13": 1.00335,
        "c13_isotope_list": [1, 2, 3],
        "ms1_accuracy_factors": [1.0, 0.45, 0.2],
        "ms2_accuracy_factors": [1.0, 0.45, 0.2],
        "n_jobs": 1,
    }

    process_args = [
        (precursor_id, data, config_dict) for precursor_id in unique_precursors
    ]

    parallel_results = []
    n_workers = cpu_count()

    with Pool(processes=n_workers) as pool:
        results = []
        for result in tqdm(
            pool.imap(process_single_precursor, process_args, chunksize=10),
            total=len(process_args),
            desc="Parallel",
        ):
            results.append(result)

    parallel_results = [r for r in results if r is not None]
    parallel_time = time.time() - start_time

    # Results
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Sequential processing:")
    print(f"  Time: {sequential_time:.2f} seconds")
    print(f"  Successful: {len(sequential_results)} precursors")
    print(f"  Speed: {len(sequential_results)/sequential_time:.2f} precursors/second")

    print(f"\nParallel processing ({n_workers} cores):")
    print(f"  Time: {parallel_time:.2f} seconds")
    print(f"  Successful: {len(parallel_results)} precursors")
    print(f"  Speed: {len(parallel_results)/parallel_time:.2f} precursors/second")

    if parallel_time > 0 and sequential_time > 0:
        speedup = sequential_time / parallel_time
        print(f"\nSpeedup: {speedup:.2f}x")
        efficiency = speedup / n_workers * 100
        print(f"Efficiency: {efficiency:.1f}%")

    print(f"{'='*60}")


def main_benchmark():
    """Run performance benchmarks."""
    print("DIA-NN Feature Generator - Performance Benchmark")

    # Small benchmark first
    benchmark_parallel_vs_sequential(n_precursors=50)

    # Larger benchmark if desired
    user_input = input("\nRun larger benchmark with 200 precursors? (y/n): ")
    if user_input.lower() == "y":
        benchmark_parallel_vs_sequential(n_precursors=200)


def main():
    """Main test function."""

    print("Loading test data...")
    data = load_test_data()

    if data is None:
        print("Failed to load test data. Please check file paths.")
        return

    print("Data loaded successfully!")

    print(
        "\n Calculating all features for all precursors and outputting to outfile.pin..."
    )

    # Output features to file - using parallel version by default
    print("Using parallel processing for maximum speed...")
    output_features_to_file(
        "debug/outfile.pin", n_jobs=-1, chunk_size=50, detailed_monitoring=False
    )


def main_sequential():
    """Main test function using sequential processing."""

    print("Loading test data...")
    data = load_test_data()

    if data is None:
        print("Failed to load test data. Please check file paths.")
        return

    print("Data loaded successfully!")

    # Test single precursor
    print("\n1. Testing single precursor...")
    features = test_single_precursor(data)

    if features:
        print_feature_summary(features)

        # Validate features
        issues = validate_features(features)
        if issues:
            print("\nValidation Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nAll features passed validation!")

    # Test multiple precursors
    print("\n2. Testing multiple precursors...")
    multiple_results = test_multiple_precursors(data, max_precursors=3)

    print(f"Successfully processed {len(multiple_results)} precursors")

    # Show feature consistency across precursors
    if len(multiple_results) > 1:
        print("\n3. Feature consistency check...")
        feature_names = set()
        for result in multiple_results.values():
            feature_names.update(result.keys())

        print("Common features across all precursors:")
        common_features = feature_names
        for result in multiple_results.values():
            common_features = common_features.intersection(result.keys())

        print(f"  {len(common_features)} common features: {sorted(common_features)}")

    # Performance benchmark
    print("\n4. Performance benchmark...")
    benchmark_performance(data, num_tests=5)

    print("\nTest completed successfully!")

    print(
        "\n Calculating all features for all precursors and outputting to outfile.pin (sequential)..."
    )

    # Output features to file - using sequential version
    print("Using sequential processing...")
    output_features_to_file_sequential("debug/outfile_sequential.pin", n=100)


if __name__ == "__main__":
    import sys

    # Simple command line interface
    if len(sys.argv) > 1:
        if sys.argv[1] == "--sequential":
            print("Running in sequential mode...")
            main_sequential()
        elif sys.argv[1] == "--parallel":
            print("Running in parallel mode...")
            main()
        elif sys.argv[1] == "--benchmark":
            print("Running performance benchmark...")
            main_benchmark()
        elif sys.argv[1] == "--help":
            print(
                "Usage: python test_diann_features.py [--parallel|--sequential|--benchmark|--help]"
            )
            print("  --parallel   : Use parallel processing (default)")
            print("  --sequential : Use sequential processing")
            print(
                "  --benchmark  : Run performance comparison between parallel and sequential"
            )
            print("  --help       : Show this help message")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for available options")
    else:
        print("Running in parallel mode (default)...")
        print("For other options, use: python test_diann_features.py --help")
        main()
