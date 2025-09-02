#!/usr/bin/env python3
"""
MuMDIA Configuration System

Clean, type-safe configuration that handles complex nested JSON structure
while providing a simple interface.

Usage:
    from config import get_config
    config = get_config()
    print(config.mzml_file, config.n_windows)
"""

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import sys


@dataclass
class DatabaseConfig:
    """Sage database configuration section."""
    bucket_size: int = 1024
    enzyme: Dict[str, Any] = field(default_factory=lambda: {
        "missed_cleavages": 2,
        "min_len": 6,
        "max_len": 30,
        "cleave_at": "KR",
        "restrict": "P",
        "c_terminal": True
    })
    fragment_min_mz: float = 100.0
    fragment_max_mz: float = 2500.0
    peptide_min_mass: float = 300.0
    peptide_max_mass: float = 5000.0
    ion_kinds: List[str] = field(default_factory=lambda: ["b", "y"])
    min_ion_index: int = 2
    static_mods: Dict[str, float] = field(default_factory=lambda: {"C": 57.0215})
    variable_mods: Dict[str, List[float]] = field(default_factory=lambda: {"M": [15.9949]})
    max_variable_mods: int = 1
    decoy_tag: str = "rev_"
    generate_decoys: bool = True
    fasta: str = ""


@dataclass
class SageConfig:
    """Complete Sage configuration."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    precursor_tol: Dict[str, List[float]] = field(default_factory=lambda: {"da": [-40, 40]})
    fragment_tol: Dict[str, List[float]] = field(default_factory=lambda: {"ppm": [-13, 13]})
    precursor_charge: List[int] = field(default_factory=lambda: [1, 4])
    isotope_errors: List[int] = field(default_factory=lambda: [-1, 1])
    deisotope: bool = False
    annotate_matches: bool = True
    chimera: bool = True
    wide_window: bool = True
    min_peaks: int = 0
    max_peaks: int = 10000
    min_matched_peaks: int = 5
    max_fragment_charge: int = 1
    report_psms: int = 5
    output_directory: str = "./"
    mzml_paths: List[str] = field(default_factory=list)


@dataclass
class MuMDIASettings:
    """MuMDIA-specific settings - complete with all existing options."""
    # Pickle settings
    write_deeplc_pickle: bool = True
    write_ms2pip_pickle: bool = True
    write_correlation_pickles: bool = True
    write_initial_search_pickle: bool = True
    write_full_search_pickle: bool = True
    read_deeplc_pickle: bool = True
    read_ms2pip_pickle: bool = True
    read_correlation_pickles: bool = True
    read_full_search_pickles: bool = True
    read_initial_search_pickle: bool = True
    
    # Processing settings
    remove_intermediate_files: bool = False
    dlc_transfer_learn: bool = False
    fdr_init_search: float = 0.01
    
    # Feature settings
    rescoring_features: List[str] = field(default_factory=lambda: [
        "distribution_correlation_matrix_psm_ids",
        "distribution_correlation_matrix_frag_ids",
        "distribution_correlation_individual",
        "top_correlation_individual",
        "top_correlation_matrix_frag_ids",
        "top_correlation_matrix_psm_ids"
    ])
    
    # Column processing settings  
    collapse_max_columns: List[str] = field(default_factory=lambda: [
        "fragment_ppm", "rank", "delta_next", "delta_rt_model",
        "matched_peaks", "longest_b", "longest_y", "matched_intensity_pct",
        "fragment_intensity", "poisson", "spectrum_q", "peptide_q", "protein_q",
        "rt", "rt_predictions", "rt_prediction_error_abs", 
        "rt_prediction_error_abs_relative", "precursor_ppm", "hyperscore", "delta_best"
    ])
    
    collapse_min_columns: List[str] = field(default_factory=lambda: [
        "fragment_ppm", "rank", "delta_next", "delta_rt_model",
        "matched_peaks", "longest_b", "longest_y", "matched_intensity_pct",
        "fragment_intensity", "poisson", "spectrum_q", "peptide_q", "protein_q",
        "rt", "rt_predictions", "rt_prediction_error_abs", 
        "rt_prediction_error_abs_relative", "precursor_ppm", "hyperscore", "delta_best"
    ])
    
    collapse_mean_columns: List[str] = field(default_factory=lambda: [
        "fragment_ppm", "rank", "delta_next", "delta_rt_model",
        "matched_peaks", "longest_b", "longest_y", "matched_intensity_pct",
        "fragment_intensity", "poisson", "spectrum_q", "peptide_q", "protein_q",
        "rt", "rt_predictions", "rt_prediction_error_abs", 
        "rt_prediction_error_abs_relative", "precursor_ppm", "hyperscore", "delta_best"
    ])
    
    collapse_sum_columns: List[str] = field(default_factory=lambda: [
        "hyperscore", "delta_rt_model", "matched_peaks", "longest_b", "longest_y",
        "matched_intensity_pct", "fragment_intensity", "rt", "rt_predictions",
        "rt_prediction_error_abs", "rt_prediction_error_abs_relative", "precursor_ppm",
        "fragment_ppm", "delta_next", "rank", "delta_best"
    ])
    
    get_first_entry: List[str] = field(default_factory=lambda: [
        "psm_id", "filename", "scannr", "peptide", "num_proteins", "proteins",
        "expmass", "calcmass", "is_decoy", "charge", "peptide_len", "missed_cleavages"
    ])
    
    collect_distributions: List[int] = field(default_factory=lambda: [
        0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
    ])
    
    collect_top: List[int] = field(default_factory=lambda: [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ])


@dataclass 
class Config:
    """
    MuMDIA configuration that handles all complexity internally.
    
    This provides a single source of truth while maintaining compatibility
    with the existing complex JSON structure.
    """
    
    # === Core Required Files ===
    mzml_file: str = ""
    fasta_file: str = ""
    mgf_file: str = ""
    
    # === Processing Parameters ===
    result_dir: str = "results"
    n_windows: int = 10
    training_fdr: float = 0.05
    final_fdr: float = 0.01
    model_type: str = "xgboost"
    
    # === Behavioral Flags ===
    no_cache: bool = False
    clean: bool = False
    sage_only: bool = False
    skip_mokapot: bool = False
    verbose: bool = False
    
    # === Complex Nested Configurations ===
    sage_basic: SageConfig = field(default_factory=SageConfig)
    sage: SageConfig = field(default_factory=SageConfig)
    mumdia: MuMDIASettings = field(default_factory=MuMDIASettings)
    
    # === Internal ===
    _config_file: str = "configs/config.json"
    
    @classmethod
    def from_json(cls, json_path: str) -> "Config":
        """Load from complex nested JSON while providing simple interface."""
        config = cls()
        
        if not Path(json_path).exists():
            print(f"Warning: Config file {json_path} not found, using defaults")
            return config
            
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            # Handle the nested structure
            if "mumdia" in data:
                # Extract simple values from mumdia section
                mumdia_data = data["mumdia"]
                for key, value in mumdia_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                
                # Set the complex mumdia settings
                config.mumdia = MuMDIASettings(**{
                    k: v for k, v in mumdia_data.items() 
                    if k in MuMDIASettings.__annotations__
                })
            
            # Handle sage configurations
            if "sage_basic" in data:
                config.sage_basic = cls._parse_sage_config(data["sage_basic"])
            if "sage" in data:
                config.sage = cls._parse_sage_config(data["sage"])
            
            # Extract file paths from sage configs if not set
            if not config.mzml_file:
                paths = config.sage.mzml_paths or config.sage_basic.mzml_paths
                if paths:
                    config.mzml_file = paths[0]
                    
            if not config.fasta_file:
                fasta = config.sage.database.fasta or config.sage_basic.database.fasta
                if fasta:
                    config.fasta_file = fasta
                    
        except Exception as e:
            print(f"Error loading config from {json_path}: {e}")
            
        return config
    
    @classmethod
    def _parse_sage_config(cls, sage_data: Dict[str, Any]) -> SageConfig:
        """Parse nested sage configuration."""
        sage_config = SageConfig()
        
        # Handle database section
        if "database" in sage_data:
            db_data = sage_data["database"]
            sage_config.database = DatabaseConfig(**{
                k: v for k, v in db_data.items() 
                if k in DatabaseConfig.__annotations__
            })
        
        # Handle other sage settings
        for key, value in sage_data.items():
            if key != "database" and hasattr(sage_config, key):
                setattr(sage_config, key, value)
                
        return sage_config
    
    @classmethod
    def from_args(cls, args: Optional[argparse.Namespace] = None) -> "Config":
        """Create from command line arguments with config file support."""
        if args is None:
            parser = cls._create_parser()
            args = parser.parse_args()
        
        # Start with config file if provided
        config_file = getattr(args, 'config_file', 'configs/config.json')
        config = cls.from_json(config_file)
        config._config_file = config_file
        
        # Override with CLI arguments
        for key, value in vars(args).items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        
        # Handle special flags
        if config.no_cache:
            config.mumdia.read_deeplc_pickle = False
            config.mumdia.read_ms2pip_pickle = False
            config.mumdia.read_correlation_pickles = False
            config.mumdia.read_initial_search_pickle = False
            config.mumdia.read_full_search_pickles = False
            
        if config.clean:
            config.mumdia.remove_intermediate_files = True
        
        return config
    
    @staticmethod
    def _create_parser() -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="MuMDIA: Unified Configuration System"
        )
        
        # Config file
        parser.add_argument("--config_file", default="configs/config.json",
                          help="Path to JSON configuration file")
        
        # Required files  
        parser.add_argument("--mzml_file", help="Path to mzML file")
        parser.add_argument("--fasta_file", help="Path to FASTA file")
        parser.add_argument("--mgf_file", help="Path to MGF file")
        
        # Processing
        parser.add_argument("--result_dir", default="results", help="Output directory")
        parser.add_argument("--n_windows", type=int, default=10, help="Number of RT windows")
        parser.add_argument("--training_fdr", type=float, default=0.05, help="Training FDR")
        parser.add_argument("--final_fdr", type=float, default=0.01, help="Final FDR")
        parser.add_argument("--model_type", choices=["xgboost", "nn", "percolator"],
                          default="xgboost", help="ML model type")
        
        # Flags
        parser.add_argument("--no-cache", action="store_true", help="Disable caching")
        parser.add_argument("--clean", action="store_true", help="Clean intermediate files")
        parser.add_argument("--sage-only", action="store_true", help="Run Sage only")
        parser.add_argument("--skip-mokapot", action="store_true", help="Skip Mokapot")
        parser.add_argument("--verbose", action="store_true", help="Verbose output")
        
        return parser
    
    def save(self, path: str) -> None:
        """Save in the complex nested JSON format for backwards compatibility."""
        # Sync file paths into sage configs
        if self.mzml_file:
            self.sage_basic.mzml_paths = [self.mzml_file]
            self.sage.mzml_paths = [self.mzml_file]
        if self.fasta_file:
            self.sage_basic.database.fasta = self.fasta_file
            self.sage.database.fasta = self.fasta_file
        
        # Create the complex nested structure
        output = {
            "sage_basic": asdict(self.sage_basic),
            "sage": asdict(self.sage),
            "mumdia": asdict(self.mumdia)
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
    
    def validate(self) -> None:
        """Validate configuration."""
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
    
    def get_effective_config_path(self) -> str:
        """Get the path where effective config will be saved."""
        return os.path.join(self.result_dir, "effective_config.json")
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """
        Convert to the legacy format expected by existing run.py code.
        
        Returns:
            Dictionary in the format expected by the legacy mumdia workflow
        """
        # Sync file paths into sage configs first
        if self.mzml_file:
            self.sage_basic.mzml_paths = [self.mzml_file]
            self.sage.mzml_paths = [self.mzml_file]
        if self.fasta_file:
            self.sage_basic.database.fasta = self.fasta_file
            self.sage.database.fasta = self.fasta_file
        
        # Create the full legacy structure
        legacy_config = {
            "sage_basic": asdict(self.sage_basic),
            "sage": asdict(self.sage),
            "mumdia": asdict(self.mumdia)
        }
        
        # Ensure mumdia section has the simple config values too
        legacy_config["mumdia"].update({
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
            "verbose": self.verbose
        })
        
        return legacy_config
    
    def get_mumdia_args_dict(self) -> Dict[str, Any]:
        """
        Get the mumdia args dictionary expected by existing code.
        
        This method provides compatibility with the existing workflow
        that expects args_dict = config["mumdia"]
        """
        return self.to_legacy_format()["mumdia"]


def get_config() -> Config:
    """
    Get configuration from command-line arguments with validation.
    
    This is the main entry point for MuMDIA configuration.
    
    Returns:
        Validated Config instance
    """
    config = Config.from_args()
    config.validate()
    return config


if __name__ == "__main__":
    # Demo the unified approach
    try:
        config = get_config()
        print("MuMDIA Configuration Demo")
        print("=" * 50)
        print(f"mzML file: {config.mzml_file}")
        print(f"FASTA file: {config.fasta_file}")
        print(f"Result dir: {config.result_dir}")
        print(f"Windows: {config.n_windows}")
        print(f"No cache: {config.no_cache}")
        print()
        print("✅ Complex nested config handled transparently!")
        print("✅ Simple interface for common operations!")
        print("✅ Full backwards compatibility!")
        
    except SystemExit:
        print("❌ Configuration validation failed")
