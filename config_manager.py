"""
Simplified configuration management for MuMDIA.

This module provides a clean, unified way to handle configuration from
JSON files, command line arguments, and defaults.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class MuMDIAConfig:
    """
    Simplified configuration class for MuMDIA.

    This replaces the complex argument parsing and config merging logic
    with a clean, type-safe dataclass approach.
    """

    # Required files
    mzml_file: str = ""
    fasta_file: str = ""
    mgf_file: str = ""

    # Output configuration
    result_dir: str = "results"
    search_config: str = "configs/config.json"

    # Processing parameters
    n_windows: int = 10
    training_fdr: float = 0.05
    final_fdr: float = 0.01
    model_type: str = "xgboost"  # choices: xgboost, nn, percolator

    # Behavioral flags
    no_cache: bool = False
    clean: bool = False
    sage_only: bool = False
    skip_mokapot: bool = False
    verbose: bool = False

    @classmethod
    def from_json(cls, config_path: str) -> "MuMDIAConfig":
        """Load configuration from JSON file with sensible defaults."""
        config = cls()  # Start with defaults

        if Path(config_path).exists():
            with open(config_path) as f:
                json_data = json.load(f)

            # Update with JSON values
            if "mumdia" in json_data:
                for key, value in json_data["mumdia"].items():
                    if hasattr(config, key):
                        setattr(config, key, value)

            # Store the full JSON config for sage/mumdia sections
            config.sage_basic = json_data.get("sage_basic", {})
            config.sage = json_data.get("sage", {})
            config.mumdia = json_data.get("mumdia", {})

        return config

    @classmethod
    def from_args(cls, args: Optional[argparse.Namespace] = None) -> "MuMDIAConfig":
        """Create config from command line arguments."""
        if args is None:
            args = parse_arguments()

        # Start with JSON config if provided
        config = cls.from_json(args.config_file)

        # Override with any explicitly provided CLI arguments
        for key, value in vars(args).items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)

        return config

    def save(self, path: str) -> None:
        """Save current configuration to JSON file."""
        output = {
            "mumdia": {
                key: getattr(self, key)
                for key in [
                    "mzml_file",
                    "mzml_dir",
                    "fasta_file",
                    "result_dir",
                    "remove_intermediate_files",
                    "write_initial_search",
                    "read_initial_search",
                    "write_full_search",
                    "read_full_search",
                    "write_deeplc",
                    "read_deeplc",
                    "write_ms2pip",
                    "read_ms2pip",
                    "write_correlation",
                    "read_correlation",
                    "dlc_transfer_learn",
                    "fdr_init_search",
                ]
            },
            "sage_basic": self.sage_basic,
            "sage": self.sage,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)


def parse_arguments() -> argparse.Namespace:
    """Simplified argument parser with only the essential arguments."""
    parser = argparse.ArgumentParser(
        description="MuMDIA: Multi-modal Data-Independent Acquisition Analysis"
    )

    # Essential paths
    parser.add_argument("--mzml_file", help="Path to mzML file")
    parser.add_argument("--fasta_file", help="Path to FASTA file")
    parser.add_argument("--result_dir", help="Results directory")
    parser.add_argument("--config_file", help="Configuration JSON file")

    # Common flags
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable all caching (force recomputation)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove intermediate files after processing",
    )

    # Advanced overrides
    parser.add_argument(
        "--fdr",
        type=float,
        dest="fdr_init_search",
        help="FDR threshold for initial search",
    )

    return parser.parse_args()


def get_config() -> MuMDIAConfig:
    """
    One-liner to get fully configured MuMDIA settings.

    This replaces all the complex config parsing logic.
    """
    args = parse_arguments()
    config = MuMDIAConfig.from_args(args)

    # Handle special flags
    if args.no_cache:
        config.read_initial_search = False
        config.read_full_search = False
        config.read_deeplc = False
        config.read_ms2pip = False
        config.read_correlation = False

    if args.clean:
        config.remove_intermediate_files = True

    return config
