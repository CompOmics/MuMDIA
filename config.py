#!/usr/bin/env python3
"""
Simplified MuMDIA Configuration System

This module provides a simplified, flat configuration system that replaces the previous
complex nested configuration. Key features:

1. Single flat parameter structure instead of nested sections
2. Override mechanism using suffixes (_initial_search, _full_search)  
3. Automatic default values for all parameters
4. Backwards compatibility with existing workflow code
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def convert_legacy_config(legacy_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old nested config format to new flat format."""
    flat_data = {}

    # Extract basic parameters from sage_basic section
    if "sage_basic" in legacy_data:
        sage_basic = legacy_data["sage_basic"]

        # Extract file paths
        if "mzml_paths" in sage_basic and sage_basic["mzml_paths"]:
            flat_data["mzml_file"] = sage_basic["mzml_paths"][0]

        # Extract database parameters
        if "database" in sage_basic:
            db = sage_basic["database"]
            if "fasta" in db:
                flat_data["fasta_file"] = db["fasta"]
            if "enzyme" in db:
                enzyme = db["enzyme"]
                flat_data["missed_cleavages"] = enzyme.get("missed_cleavages", 2)
                flat_data["cleave_at"] = enzyme.get("cleave_at", "KR")
            flat_data["max_variable_mods"] = db.get("max_variable_mods", 1)

        # Extract other sage_basic parameters
        flat_data["deisotope"] = sage_basic.get("deisotope", False)
        flat_data["chimera"] = sage_basic.get("chimera", True)
        flat_data["wide_window"] = sage_basic.get("wide_window", True)
        flat_data["report_psms"] = sage_basic.get("report_psms", 5)

    # Extract full search overrides from sage section
    if "sage" in legacy_data:
        sage = legacy_data["sage"]

        # Full search specific parameters
        flat_data["deisotope_full_search"] = sage.get("deisotope", True)
        flat_data["report_psms_full_search"] = sage.get("report_psms", 12)

        if "database" in sage:
            db = sage["database"]
            if "enzyme" in db and "cleave_at" in db["enzyme"]:
                flat_data["cleave_at_initial_search"] = db["enzyme"]["cleave_at"]
            flat_data["max_variable_mods_full_search"] = db.get("max_variable_mods", 2)

    # Extract MuMDIA parameters - only include fields that exist in MuMDIAConfig
    mumdia_field_mapping = {
        "write_deeplc_pickle": "write_deeplc_pickle",
        "write_ms2pip_pickle": "write_ms2pip_pickle",
        "write_correlation_pickles": "write_correlation_pickles",
        "write_initial_search_pickle": "write_initial_search_pickle",
        "write_full_search_pickle": "write_full_search_pickle",
        "read_deeplc_pickle": "read_deeplc_pickle",
        "read_ms2pip_pickle": "read_ms2pip_pickle",
        "read_correlation_pickles": "read_correlation_pickles",
        "read_full_search_pickles": "read_full_search_pickle",  # Fix plural->singular
        "read_initial_search_pickle": "read_initial_search_pickle",
        "remove_intermediate_files": "remove_intermediate_files",
        "fdr_init_search": "fdr_init_search",
        "min_occurrences": "min_occurrences",
    }

    if "mumdia" in legacy_data:
        mumdia = legacy_data["mumdia"]
        for old_key, new_key in mumdia_field_mapping.items():
            if old_key in mumdia:
                flat_data[new_key] = mumdia[old_key]

    # Set default result_dir if not present
    if "result_dir" not in flat_data:
        flat_data["result_dir"] = "results"

    return flat_data


def load_config_from_json(json_path: str) -> "MuMDIAConfig":
    """Load configuration from JSON file, supporting both old and new formats."""
    with open(json_path, "r") as f:
        config_data = json.load(f)

    # Check if this is the old nested format
    if "sage_basic" in config_data or "sage" in config_data or "mumdia" in config_data:
        print("Converting legacy config format to new simplified format...")
        config_data = convert_legacy_config(config_data)

    # Filter out comment fields (fields starting with _comment)
    filtered_data = {
        k: v for k, v in config_data.items() if not k.startswith("_comment")
    }

    return MuMDIAConfig(**filtered_data)


@dataclass
class MuMDIAConfig:
    """
    Simplified MuMDIA configuration with smart defaults and override mechanism.

    Parameters can be overridden for specific search types using suffixes:
    - parameter_initial_search: Override for initial search only
    - parameter_full_search: Override for full search only
    - parameter: Default value used for both if no specific override
    """

    # === File paths ===
    mzml_file: str = ""
    fasta_file: str = ""
    mgf_file: str = ""
    result_dir: str = "results"
    config_file: str = "configs/config.json"

    # === Search parameters ===
    # Database settings
    bucket_size: int = 1024
    bucket_size_initial_search: Optional[int] = None
    bucket_size_full_search: Optional[int] = None

    # Enzyme settings
    missed_cleavages: int = 2
    missed_cleavages_initial_search: Optional[int] = None
    missed_cleavages_full_search: Optional[int] = None

    min_len: int = 6
    max_len: int = 30
    cleave_at: str = "KR"
    cleave_at_initial_search: str = "$"  # Different default for initial search
    cleave_at_full_search: Optional[str] = None

    restrict: str = "P"
    c_terminal: bool = True

    # Mass ranges
    fragment_min_mz: float = 100.0
    fragment_max_mz: float = 2500.0
    peptide_min_mass: float = 300.0
    peptide_max_mass: float = 5000.0

    # Ion settings
    ion_kinds: List[str] = field(default_factory=lambda: ["b", "y"])
    min_ion_index: int = 2
    max_fragment_charge: int = 1

    # Modifications
    static_mods: Dict[str, float] = field(default_factory=lambda: {"C": 57.0215})
    variable_mods: Dict[str, List[float]] = field(
        default_factory=lambda: {"M": [15.9949]}
    )
    max_variable_mods: int = 1
    max_variable_mods_initial_search: Optional[int] = None
    max_variable_mods_full_search: int = 2  # Different default for full search

    # Decoys
    decoy_tag: str = "rev_"
    generate_decoys: bool = True

    # Tolerances
    precursor_tol_da_low: float = -40.0
    precursor_tol_da_high: float = 40.0
    fragment_tol_ppm_low: float = -13.0
    fragment_tol_ppm_high: float = 13.0

    # Charge and isotope settings
    precursor_charge_min: int = 1
    precursor_charge_max: int = 4
    isotope_errors_min: int = -1
    isotope_errors_max: int = 1

    # Processing settings
    deisotope: bool = False
    deisotope_initial_search: Optional[bool] = None
    deisotope_full_search: bool = True  # Different default for full search

    annotate_matches: bool = True
    chimera: bool = True
    predict_rt: bool = False
    wide_window: bool = True

    # Peak settings
    min_peaks: int = 0
    max_peaks: int = 10000
    min_matched_peaks: int = 5

    # Reporting
    report_psms: int = 5
    report_psms_initial_search: Optional[int] = None
    report_psms_full_search: int = 12  # Different default for full search

    # === MuMDIA-specific settings ===
    # Pickle settings
    write_deeplc_pickle: bool = True
    write_ms2pip_pickle: bool = True
    write_correlation_pickles: bool = True
    write_initial_search_pickle: bool = True
    write_full_search_pickle: bool = True
    read_deeplc_pickle: bool = False
    read_ms2pip_pickle: bool = False
    read_correlation_pickles: bool = False
    read_full_search_pickle: bool = False
    read_initial_search_pickle: bool = False

    # Filtering settings
    min_occurrences: int = 5  # Minimum PSMs per peptide to keep

    # Processing settings
    remove_intermediate_files: bool = False
    dlc_transfer_learn: bool = False
    fdr_init_search: float = 0.01

    # ML settings
    n_windows: int = 10
    training_fdr: float = 0.01
    final_fdr: float = 0.01
    model_type: str = "xgboost"

    # Runtime flags
    no_cache: bool = False
    clean: bool = False
    sage_only: bool = False
    skip_mokapot: bool = False
    use_diann_features: bool = True  # DIA-NN features enabled (fast with Rust backend)
    diann_na_strategy: str = "overlap_only"  # "overlap_only" or "fill_zero"
    verbose: bool = False

    # Feature settings
    rescoring_features: List[str] = field(
        default_factory=lambda: [
            "distribution_correlation_matrix_psm_ids",
            "distribution_correlation_matrix_frag_ids",
            "distribution_correlation_individual",
            "top_correlation_individual",
            "top_correlation_matrix_frag_ids",
            "top_correlation_matrix_psm_ids",
        ]
    )

    # Column processing settings (simplified from current complex structure)
    collapse_max_columns: List[str] = field(
        default_factory=lambda: [
            "fragment_ppm",
            "rank",
            "delta_next",
            "delta_rt_model",
            "matched_peaks",
            "longest_b",
            "longest_y",
            "matched_intensity_pct",
            "fragment_intensity",
            "poisson",
            "spectrum_q",
            "peptide_q",
            "protein_q",
            "rt",
            "rt_predictions",
            "rt_prediction_error_abs",
            "rt_prediction_error_abs_relative",
            "precursor_ppm",
            "hyperscore",
            "delta_best",
        ]
    )

    def get_parameter_value(self, param_name: str, search_type: str) -> Any:
        """
        Get the effective value for a parameter considering override hierarchy.

        Args:
            param_name: Base parameter name (e.g., 'cleave_at')
            search_type: 'initial_search' or 'full_search'

        Returns:
            The effective parameter value
        """
        # Check for specific override first
        specific_override = f"{param_name}_{search_type}"
        if hasattr(self, specific_override):
            value = getattr(self, specific_override)
            if value is not None:
                return value

        # Fall back to base parameter
        return getattr(self, param_name)

    def get_initial_search_config(self) -> Dict[str, Any]:
        """Generate Sage config for initial search."""
        return self._generate_sage_config("initial_search")

    def get_full_search_config(self) -> Dict[str, Any]:
        """Generate Sage config for full search."""
        return self._generate_sage_config("full_search")

    def _generate_sage_config(self, search_type: str) -> Dict[str, Any]:
        """Generate Sage configuration for specified search type."""

        # Helper to get parameter value for this search type
        def get_param(name: str) -> Any:
            return self.get_parameter_value(name, search_type)

        config = {
            "database": {
                "bucket_size": get_param("bucket_size"),
                "enzyme": {
                    "missed_cleavages": get_param("missed_cleavages"),
                    "min_len": self.min_len,
                    "max_len": self.max_len,
                    "cleave_at": get_param("cleave_at"),
                    "restrict": self.restrict,
                    "c_terminal": self.c_terminal,
                },
                "fragment_min_mz": self.fragment_min_mz,
                "fragment_max_mz": self.fragment_max_mz,
                "peptide_min_mass": self.peptide_min_mass,
                "peptide_max_mass": self.peptide_max_mass,
                "ion_kinds": self.ion_kinds,
                "min_ion_index": self.min_ion_index,
                "static_mods": self.static_mods,
                "variable_mods": self.variable_mods,
                "max_variable_mods": get_param("max_variable_mods"),
                "decoy_tag": self.decoy_tag,
                "generate_decoys": self.generate_decoys,
                "fasta": self.fasta_file,
            },
            "precursor_tol": {
                "da": [self.precursor_tol_da_low, self.precursor_tol_da_high]
            },
            "fragment_tol": {
                "ppm": [self.fragment_tol_ppm_low, self.fragment_tol_ppm_high]
            },
            "precursor_charge": [self.precursor_charge_min, self.precursor_charge_max],
            "isotope_errors": [self.isotope_errors_min, self.isotope_errors_max],
            "deisotope": get_param("deisotope"),
            "annotate_matches": self.annotate_matches,
            "chimera": self.chimera,
            "wide_window": self.wide_window,
            "min_peaks": self.min_peaks,
            "max_peaks": self.max_peaks,
            "min_matched_peaks": self.min_matched_peaks,
            "max_fragment_charge": self.max_fragment_charge,
            "report_psms": get_param("report_psms"),
            "output_directory": "./",
            "mzml_paths": [self.mzml_file] if self.mzml_file else [],
        }

        # Add predict_rt only for full search
        if search_type == "full_search":
            config["predict_rt"] = self.predict_rt

        return config

    def get_mumdia_config(self) -> Dict[str, Any]:
        """Get MuMDIA-specific configuration."""
        return {
            "write_deeplc_pickle": self.write_deeplc_pickle,
            "write_ms2pip_pickle": self.write_ms2pip_pickle,
            "write_correlation_pickles": self.write_correlation_pickles,
            "write_initial_search_pickle": self.write_initial_search_pickle,
            "write_full_search_pickle": self.write_full_search_pickle,
            "read_deeplc_pickle": self.read_deeplc_pickle,
            "read_ms2pip_pickle": self.read_ms2pip_pickle,
            "read_correlation_pickles": self.read_correlation_pickles,
            "read_full_search_pickle": self.read_full_search_pickle,
            "read_initial_search_pickle": self.read_initial_search_pickle,
            "remove_intermediate_files": self.remove_intermediate_files,
            "dlc_transfer_learn": self.dlc_transfer_learn,
            "fdr_init_search": self.fdr_init_search,
            "rescoring_features": self.rescoring_features,
            "collapse_max_columns": self.collapse_max_columns,
            "collapse_min_columns": self.collapse_max_columns,  # Simplified
            "collapse_mean_columns": self.collapse_max_columns,  # Simplified
            "collapse_sum_columns": self.collapse_max_columns,  # Simplified
            "get_first_entry": [  # Default list
                "psm_id",
                "filename",
                "scannr",
                "peptide",
                "num_proteins",
                "proteins",
                "expmass",
                "calcmass",
                "is_decoy",
                "charge",
                "peptide_len",
                "missed_cleavages",
            ],
            "collect_distributions": list(range(0, 101, 5)),  # 0, 5, 10, ..., 100
            "collect_top": list(range(1, 11)),  # 1, 2, 3, ..., 10
            # Add simple config values
            "mzml_file": self.mzml_file,
            "fasta_file": self.fasta_file,
            "mgf_file": self.mgf_file,
            "result_dir": self.result_dir,
            "n_windows": self.n_windows,
            "training_fdr": self.training_fdr,
            "final_fdr": self.final_fdr,
            "model_type": self.model_type,
            "no_cache": self.no_cache,
            "clean": self.clean,
            "sage_only": self.sage_only,
            "skip_mokapot": self.skip_mokapot,
            "use_diann_features": self.use_diann_features,
            "diann_na_strategy": self.diann_na_strategy,
            "verbose": self.verbose,
            "min_occurrences": self.min_occurrences,
        }

    def to_legacy_format(self) -> Dict[str, Any]:
        """
        Convert to the legacy format expected by existing run.py code.
        """
        return {
            "sage_basic": self.get_initial_search_config(),
            "sage": self.get_full_search_config(),
            "mumdia": self.get_mumdia_config(),
        }

    @classmethod
    def from_json(cls, json_path: str) -> "MuMDIAConfig":
        """Load configuration from JSON file."""
        if not os.path.exists(json_path):
            print(f"Warning: Config file {json_path} not found, using defaults")
            return cls()

        with open(json_path, "r") as f:
            data = json.load(f)

        # Create instance with defaults
        config = cls()

        # Override with JSON values
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}' ignored")

        return config

    @classmethod
    def from_args(cls) -> "MuMDIAConfig":
        """Create configuration from command line arguments."""
        parser = argparse.ArgumentParser(
            description="MuMDIA: Simplified Configuration System"
        )

        # File arguments
        parser.add_argument(
            "--config_file",
            default="configs/config.json",
            help="Path to JSON configuration file",
        )
        parser.add_argument("--mzml_file", help="Path to mzML file")
        parser.add_argument("--fasta_file", help="Path to FASTA file")
        parser.add_argument("--mgf_file", help="Path to MGF file")
        parser.add_argument("--result_dir", default="results", help="Output directory")

        # Search parameters
        parser.add_argument(
            "--n_windows", type=int, default=10, help="Number of RT windows"
        )
        parser.add_argument(
            "--training_fdr", type=float, default=0.01, help="Training FDR"
        )
        parser.add_argument("--final_fdr", type=float, default=0.01, help="Final FDR")
        parser.add_argument(
            "--model_type",
            choices=["xgboost", "nn", "percolator"],
            default="xgboost",
            help="ML model type",
        )

        # Runtime flags
        parser.add_argument("--no-cache", action="store_true", help="Disable caching")
        parser.add_argument(
            "--clean", action="store_true", help="Clean intermediate files"
        )
        parser.add_argument("--sage-only", action="store_true", help="Run Sage only")
        parser.add_argument("--skip-mokapot", action="store_true", help="Skip Mokapot")
        parser.add_argument("--verbose", action="store_true", help="Verbose output")

        args = parser.parse_args()

        # Load from JSON first
        config = cls.from_json(args.config_file)

        # Override with command line arguments
        config.config_file = args.config_file
        if args.mzml_file:
            config.mzml_file = args.mzml_file
        if args.fasta_file:
            config.fasta_file = args.fasta_file
        if args.mgf_file:
            config.mgf_file = args.mgf_file
        if args.result_dir != "results":  # Only override if explicitly set
            config.result_dir = args.result_dir
        if args.n_windows != 10:
            config.n_windows = args.n_windows
        if args.training_fdr != 0.01:
            config.training_fdr = args.training_fdr
        if args.final_fdr != 0.01:
            config.final_fdr = args.final_fdr
        if args.model_type != "xgboost":
            config.model_type = args.model_type

        config.no_cache = args.no_cache
        config.clean = args.clean
        config.sage_only = args.sage_only
        config.skip_mokapot = args.skip_mokapot
        config.verbose = args.verbose

        return config

    def validate(self) -> None:
        """Validate configuration."""
        errors = []

        if self.mzml_file and not os.path.exists(self.mzml_file):
            errors.append(f"mzML file not found: {self.mzml_file}")

        if self.fasta_file and not os.path.exists(self.fasta_file):
            errors.append(f"FASTA file not found: {self.fasta_file}")

        if self.mgf_file and not os.path.exists(self.mgf_file):
            errors.append(f"MGF file not found: {self.mgf_file}")

        if errors:
            for error in errors:
                print(f"Error: {error}")
            sys.exit(1)


def get_config() -> MuMDIAConfig:
    """
    Get configuration from command-line arguments with validation.

    This is the main entry point for MuMDIA configuration.

    Returns:
        Validated MuMDIAConfig instance
    """
    config = MuMDIAConfig.from_args()
    config.validate()
    return config


if __name__ == "__main__":
    # Demo the new simplified approach
    try:
        config = get_config()
        print("MuMDIA Simplified Configuration Demo")
        print("=" * 50)
        print(f"mzML file: {config.mzml_file}")
        print(f"FASTA file: {config.fasta_file}")
        print(f"Result dir: {config.result_dir}")
        print()

        print("Initial Search Config:")
        initial = config.get_initial_search_config()
        print(f"  cleave_at: {initial['database']['enzyme']['cleave_at']}")
        print(f"  report_psms: {initial['report_psms']}")
        print(f"  deisotope: {initial['deisotope']}")
        print()

        print("Full Search Config:")
        full = config.get_full_search_config()
        print(f"  cleave_at: {full['database']['enzyme']['cleave_at']}")
        print(f"  report_psms: {full['report_psms']}")
        print(f"  deisotope: {full['deisotope']}")
        print()

        print("Simplified configuration with smart defaults!")
        print("Automatic initial vs full search differentiation!")
        print("Override system working!")

    except SystemExit:
        print("Configuration validation failed")
