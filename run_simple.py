#!/usr/bin/env python3
"""
MuMDIA (Multi-modal Data-Independent Acquisition) - Simplified Main Workflow

This demonstrates the new simplified configuration approach for MuMDIA.
This is much cleaner than the original run.py with complex argument parsing.

Usage:
    python run_simple.py --mzml_file data.mzML --fasta_file proteins.fasta --result_dir results/
    python run_simple.py --config_file my_config.json
    python run_simple.py --no-cache  # Force recomputation
"""

import os

os.environ["POLARS_MAX_THREADS"] = "1"

from pathlib import Path

from config_manager_clean import get_config
from data_structures import PickleConfig
from utilities.logger import log_info


def main():
    """
    Main execution function using the simplified configuration approach.

    Compare this to the complex original run.py:
    - No complex argument parsing (100+ lines reduced to 1 line: get_config())
    - No manual config merging and override logic
    - Clean, readable configuration access
    - Automatic config file saving for reference
    """
    # Get configuration - this replaces ~100 lines of complex parsing!
    config = get_config()

    log_info(f"Starting MuMDIA pipeline with config:")
    log_info(f"  mzML file: {config.mzml_file}")
    log_info(f"  FASTA file: {config.fasta_file}")
    log_info(f"  Result directory: {config.result_dir}")
    log_info(f"  Windows: {config.n_windows}")
    log_info(f"  Training FDR: {config.training_fdr}")
    log_info(f"  Final FDR: {config.final_fdr}")
    log_info(f"  Model type: {config.model_type}")
    log_info(f"  No cache: {config.no_cache}")
    log_info(f"  Clean: {config.clean}")
    log_info(f"  Sage only: {config.sage_only}")

    # Create result directory
    result_dir = Path(config.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save the effective configuration for reference
    config.save(str(result_dir / "effective_config.json"))
    log_info(f"Saved effective configuration to {result_dir / 'effective_config.json'}")

    # Configure pickle settings based on cache preference
    pickle_config = PickleConfig()

    # Now you can continue with the actual MuMDIA workflow
    # The key difference is that configuration is clean and simple!

    if config.sage_only:
        log_info("Running Sage-only workflow...")
        # Add Sage-only logic here
    else:
        log_info("Running full MuMDIA workflow...")
        # Add full workflow logic here

    if config.clean:
        log_info("Cleaning up intermediate files...")
        # Add cleanup logic here

    log_info("MuMDIA pipeline completed successfully!")


if __name__ == "__main__":
    main()
