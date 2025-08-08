#!/usr/bin/env python3
"""
Simplified Configuration Management for MuMDIA

This module provides a clean, dataclass-based approach to configuration management,
replacing the complex argument parsing and config merging logic in the original run.py.

The MuMDIAConfig class centralizes all configuration options with sensible defaults
and provides simple methods for loading from JSON files or command-line arguments.
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


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
    def from_json(cls, json_path: str) -> "MuMDIAConfig":
        """
        Load configuration from a JSON file.

        Args:
            json_path: Path to the JSON configuration file

        Returns:
            MuMDIAConfig instance with values from the JSON file
        """
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            # Create config with JSON data, using defaults for missing keys
            config = cls()
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            return config

        except FileNotFoundError:
            print(f"Warning: Config file {json_path} not found, using defaults")
            return cls()
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {json_path}: {e}")
            sys.exit(1)

    @classmethod
    def from_args(cls, args: Optional[argparse.Namespace] = None) -> "MuMDIAConfig":
        """
        Create configuration from command-line arguments.

        Args:
            args: Parsed arguments (if None, will parse sys.argv)

        Returns:
            MuMDIAConfig instance with values from command-line arguments
        """
        if args is None:
            parser = cls._create_parser()
            args = parser.parse_args()

        # Start with config from file if provided
        if hasattr(args, "config_file") and args.config_file:
            config = cls.from_json(args.config_file)
        else:
            config = cls()

        # Override with command-line arguments
        for key, value in vars(args).items():
            if hasattr(config, key) and value is not None:
                # Handle boolean flags properly
                if key in ["no_cache", "clean", "sage_only", "skip_mokapot", "verbose"]:
                    setattr(config, key, bool(value))
                else:
                    setattr(config, key, value)

        return config

    @staticmethod
    def _create_parser() -> argparse.ArgumentParser:
        """Create the argument parser with simplified options."""
        parser = argparse.ArgumentParser(
            description="MuMDIA: Multi-modal Data-Independent Acquisition pipeline"
        )

        # Configuration file
        parser.add_argument("--config_file", help="Path to JSON configuration file")

        # Required files
        parser.add_argument("--mzml_file", help="Path to mzML file")
        parser.add_argument("--fasta_file", help="Path to FASTA file")
        parser.add_argument("--mgf_file", help="Path to MGF file (optional)")

        # Output and processing
        parser.add_argument("--result_dir", default="results", help="Output directory")
        parser.add_argument(
            "--search_config", default="configs/config.json", help="Sage config file"
        )
        parser.add_argument(
            "--n_windows", type=int, default=10, help="Number of RT windows"
        )
        parser.add_argument(
            "--training_fdr", type=float, default=0.05, help="Training FDR threshold"
        )
        parser.add_argument(
            "--final_fdr", type=float, default=0.01, help="Final FDR threshold"
        )
        parser.add_argument(
            "--model_type",
            choices=["xgboost", "nn", "percolator"],
            default="xgboost",
            help="ML model type",
        )

        # Boolean flags
        parser.add_argument(
            "--no-cache", action="store_true", help="Force recomputation"
        )
        parser.add_argument(
            "--clean", action="store_true", help="Clean intermediate files"
        )
        parser.add_argument("--sage-only", action="store_true", help="Run Sage only")
        parser.add_argument("--skip-mokapot", action="store_true", help="Skip Mokapot")
        parser.add_argument("--verbose", action="store_true", help="Verbose logging")

        return parser

    def save(self, path: str) -> None:
        """
        Save configuration to a JSON file.

        Args:
            path: Path where to save the configuration
        """
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def validate(self) -> None:
        """
        Validate the configuration and exit if invalid.
        """
        if not self.mzml_file:
            print("Error: mzml_file is required")
            sys.exit(1)
        if not self.fasta_file:
            print("Error: fasta_file is required")
            sys.exit(1)

        # Check file existence
        if not Path(self.mzml_file).exists():
            print(f"Error: mzML file not found: {self.mzml_file}")
            sys.exit(1)
        if not Path(self.fasta_file).exists():
            print(f"Error: FASTA file not found: {self.fasta_file}")
            sys.exit(1)


def get_config() -> MuMDIAConfig:
    """
    Get configuration from command-line arguments with validation.

    This is the main entry point for configuration management.

    Returns:
        Validated MuMDIAConfig instance
    """
    config = MuMDIAConfig.from_args()
    config.validate()
    return config
