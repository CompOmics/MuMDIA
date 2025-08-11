#!/usr/bin/env python3
"""
Simple demo of the new MuMDIA configuration system

This shows how much cleaner configuration management can be.
"""

import sys

sys.path.append(".")

from config_manager_clean import get_config


def main():
    """Demo the simplified configuration approach."""
    try:
        # Get configuration - this replaces ~100 lines of complex parsing!
        config = get_config()

        print("MuMDIA Configuration Demo")
        print("=" * 40)
        print(f"mzML file: {config.mzml_file}")
        print(f"FASTA file: {config.fasta_file}")
        print(f"Result directory: {config.result_dir}")
        print(f"Windows: {config.n_windows}")
        print(f"Training FDR: {config.training_fdr}")
        print(f"Final FDR: {config.final_fdr}")
        print(f"Model type: {config.model_type}")
        print(f"No cache: {config.no_cache}")
        print(f"Clean: {config.clean}")
        print(f"Sage only: {config.sage_only}")
        print(f"Skip Mokapot: {config.skip_mokapot}")
        print(f"Verbose: {config.verbose}")
        print()
        print("Configuration loaded successfully!")
        print("This replaces the complex 100+ line argument parsing in run.py!")

    except SystemExit:
        # This happens when validation fails (missing required files)
        print("\nTo test with actual files, try:")
        print(
            "  python config_demo.py --mzml_file LFQ_Orbitrap_AIF_Ecoli_01.mzML --fasta_file fasta/ecoli_22032024.fasta"
        )


if __name__ == "__main__":
    main()
