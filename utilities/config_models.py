"""
Typed configuration models for MuMDIA using dataclasses.

These models provide IDE hints and basic validation while keeping
runtime dependencies minimal (no pydantic required).

If stronger validation is desired later, a pydantic equivalent can be
introduced with the same attributes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnzymeConfig:
    """Enzyme digestion parameters for Sage search."""

    missed_cleavages: int = 2
    min_len: int = 6
    max_len: int = 30
    cleave_at: str = "KR"
    restrict: Optional[str] = None
    c_terminal: bool = True


@dataclass
class ToleranceConfig:
    """Mass tolerance settings (Da or ppm) for precursor/fragment matching."""

    da: Optional[List[int]] = None
    ppm: Optional[List[int]] = None


@dataclass
class DatabaseConfig:
    """Protein database and search space parameters for Sage."""

    fasta: str = ""
    bucket_size: int = 1024
    enzyme: EnzymeConfig = field(default_factory=EnzymeConfig)
    fragment_min_mz: float = 100.0
    fragment_max_mz: float = 2500.0
    peptide_min_mass: float = 300.0
    peptide_max_mass: float = 5000.0
    ion_kinds: List[str] = field(default_factory=lambda: ["b", "y"])
    min_ion_index: int = 2
    static_mods: Dict[str, float] = field(default_factory=dict)
    fixed_mods: Dict[str, float] = field(default_factory=dict)
    variable_mods: Dict[str, List[float]] = field(default_factory=dict)
    max_variable_mods: int = 1
    decoy_tag: str = "rev_"
    generate_decoys: bool = True

    def validate(self) -> None:
        """Validate database configuration fields."""
        if not isinstance(self.fasta, str):
            raise ValueError("database.fasta must be a string")
        # Allow empty strings during initial config creation


@dataclass
class SageSection:
    """Complete Sage search engine configuration for one search stage."""

    mzml_paths: List[str] = field(default_factory=list)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    precursor_tol: ToleranceConfig = field(default_factory=ToleranceConfig)
    fragment_tol: ToleranceConfig = field(default_factory=ToleranceConfig)
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
    predict_rt: Optional[bool] = None

    def validate(self) -> None:
        """Validate Sage section fields and cascade to database validation."""
        if not isinstance(self.mzml_paths, list):
            raise ValueError("sage section requires mzml_paths to be a list")
        # Allow empty lists during initial config creation
        self.database.validate()


@dataclass
class MuMDIASettings:
    """MuMDIA pipeline settings: file paths, pickle caching flags, and processing options."""

    # Core paths
    mzml_file: str = "mzml_files/LFQ_Orbitrap_AIF_Ecoli_01.mzML"
    mzml_dir: str = "mzml_files"
    fasta_file: str = "fasta/unmodified_peptides.fasta"
    result_dir: str = "results"
    config_file: str = "configs/config.json"

    # Flags & options
    remove_intermediate_files: bool = False

    write_initial_search_pickle: bool = False
    read_initial_search_pickle: bool = True

    write_deeplc_pickle: bool = False
    write_ms2pip_pickle: bool = False
    read_deeplc_pickle: bool = True
    read_ms2pip_pickle: bool = True

    write_correlation_pickles: bool = False
    read_correlation_pickles: bool = True

    dlc_transfer_learn: bool = True

    write_full_search_pickle: bool = False
    read_full_search_pickle: bool = True

    fdr_init_search: float = 0.05

    coefficient_bounds: int = 1

    def validate(self) -> None:
        """Validate MuMDIA settings (FDR must be non-negative)."""
        # Basic sanity checks aligning with argparse defaults
        if (
            not isinstance(self.fdr_init_search, (int, float))
            or self.fdr_init_search < 0
        ):
            raise ValueError("fdr_init_search must be a non-negative number")
        # Allow empty strings during initial config creation - they'll be filled with defaults


@dataclass
class ConfigModel:
    """Top-level configuration aggregating MuMDIA settings and two Sage search stages."""

    mumdia: MuMDIASettings = field(default_factory=MuMDIASettings)
    sage_basic: SageSection = field(default_factory=SageSection)
    sage: SageSection = field(default_factory=SageSection)

    def validate(self) -> None:
        """Cascade validation through all configuration sections."""
        self.mumdia.validate()
        self.sage_basic.validate()
        self.sage.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigModel":
        """Construct ConfigModel from dict, filtering unknown fields and building nested objects."""
        # Extract sections with safe fallbacks
        mumdia_raw: Dict[str, Any] = data.get("mumdia", {}) or {}
        sage_basic_raw: Dict[str, Any] = data.get("sage_basic", {}) or {}
        sage_raw: Dict[str, Any] = data.get("sage", {}) or {}

        # Build nested objects; tolerate missing keys and filter unknown fields
        mumdia_defaults = asdict(MuMDIASettings())
        mumdia_filtered = {k: v for k, v in mumdia_raw.items() if k in mumdia_defaults}
        mumdia = MuMDIASettings(**{**mumdia_defaults, **mumdia_filtered})

        # Helper function to build database config
        def build_database_config(db_data: Dict[str, Any]) -> DatabaseConfig:
            defaults = asdict(DatabaseConfig())
            enzyme_data = db_data.get("enzyme", {})
            enzyme = EnzymeConfig(**{**asdict(EnzymeConfig()), **enzyme_data})

            # Filter out unknown fields and handle enzyme separately
            filtered_data = {
                k: v for k, v in db_data.items() if k in defaults and k != "enzyme"
            }
            return DatabaseConfig(**{**defaults, **filtered_data, "enzyme": enzyme})

        # Helper function to build tolerance config
        def build_tolerance_config(tol_data: Dict[str, Any]) -> ToleranceConfig:
            return ToleranceConfig(da=tol_data.get("da"), ppm=tol_data.get("ppm"))

        # Helper function to build sage section
        def build_sage_section(section_data: Dict[str, Any]) -> SageSection:
            defaults = asdict(SageSection())
            database = build_database_config(section_data.get("database", {}))
            precursor_tol = build_tolerance_config(
                section_data.get("precursor_tol", {})
            )
            fragment_tol = build_tolerance_config(section_data.get("fragment_tol", {}))

            # Filter out complex nested objects and handle them separately
            filtered_data = {
                k: v
                for k, v in section_data.items()
                if k in defaults
                and k not in ["database", "precursor_tol", "fragment_tol"]
            }

            return SageSection(
                **{
                    **defaults,
                    **filtered_data,
                    "database": database,
                    "precursor_tol": precursor_tol,
                    "fragment_tol": fragment_tol,
                }
            )

        # Build sage sections
        sage_basic = build_sage_section(sage_basic_raw)
        sage = build_sage_section(sage_raw)

        return cls(mumdia=mumdia, sage_basic=sage_basic, sage=sage)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to flat dict with mumdia/sage_basic/sage sections."""
        return {
            "mumdia": asdict(self.mumdia),
            "sage_basic": asdict(self.sage_basic),
            "sage": asdict(self.sage),
        }
