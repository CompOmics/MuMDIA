"""
Typed configuration models for MuMDIA using dataclasses.

These models provide IDE hints and basic validation while keeping
runtime dependencies minimal (no pydantic required).

If stronger validation is desired later, a pydantic equivalent can be
introduced with the same attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class DatabaseConfig:
    fasta: str = ""

    def validate(self) -> None:
        if not isinstance(self.fasta, str) or self.fasta.strip() == "":
            raise ValueError("database.fasta must be a non-empty string")


@dataclass
class SageSection:
    mzml_paths: List[str] = field(default_factory=list)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    def validate(self) -> None:
        if not isinstance(self.mzml_paths, list) or any(
            not isinstance(x, str) or x.strip() == "" for x in self.mzml_paths
        ):
            raise ValueError("sage section requires non-empty list of mzml_paths")
        self.database.validate()


@dataclass
class MuMDIASettings:
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

    def validate(self) -> None:
        # Basic sanity checks aligning with argparse defaults
        if (
            not isinstance(self.fdr_init_search, (int, float))
            or self.fdr_init_search < 0
        ):
            raise ValueError("fdr_init_search must be a non-negative number")
        for path_attr in (
            "mzml_file",
            "mzml_dir",
            "fasta_file",
            "result_dir",
            "config_file",
        ):
            val = getattr(self, path_attr, "")
            if not isinstance(val, str) or val.strip() == "":
                raise ValueError(f"{path_attr} must be a non-empty string")


@dataclass
class ConfigModel:
    mumdia: MuMDIASettings = field(default_factory=MuMDIASettings)
    sage_basic: SageSection = field(default_factory=SageSection)
    sage: SageSection = field(default_factory=SageSection)

    def validate(self) -> None:
        self.mumdia.validate()
        self.sage_basic.validate()
        self.sage.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigModel":
        # Extract sections with safe fallbacks
        mumdia_raw: Dict[str, Any] = data.get("mumdia", {}) or {}
        sage_basic_raw: Dict[str, Any] = data.get("sage_basic", {}) or {}
        sage_raw: Dict[str, Any] = data.get("sage", {}) or {}

        # Build nested objects; tolerate missing keys
        mumdia = MuMDIASettings(**{**asdict(MuMDIASettings()), **mumdia_raw})

        # Database
        sb_db = DatabaseConfig(
            **({**asdict(DatabaseConfig()), **sage_basic_raw.get("database", {})})
        )
        s_db = DatabaseConfig(
            **({**asdict(DatabaseConfig()), **sage_raw.get("database", {})})
        )

        # Sections
        sage_basic = SageSection(
            mzml_paths=sage_basic_raw.get("mzml_paths", []) or [],
            database=sb_db,
        )
        sage = SageSection(
            mzml_paths=sage_raw.get("mzml_paths", []) or [],
            database=s_db,
        )

        return cls(mumdia=mumdia, sage_basic=sage_basic, sage=sage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mumdia": asdict(self.mumdia),
            "sage_basic": {
                "mzml_paths": list(self.sage_basic.mzml_paths),
                "database": asdict(self.sage_basic.database),
            },
            "sage": {
                "mzml_paths": list(self.sage.mzml_paths),
                "database": asdict(self.sage.database),
            },
        }
