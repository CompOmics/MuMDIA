"""Derive an effective search-space manifest from a MuMDIA library.

Sensitivity-plan backlog item P0.1 (search-space parity). The script is
non-invasive: it reads an existing MuMDIA library-precursors Parquet, derives a
machine-readable manifest in the shape of sensitivity_plan spec 02 section 3,
and optionally compares that manifest to a declared manifest (--compare) or a
DIA-NN report (--diann). With --fail-on-mismatch it exits nonzero when the
effective search spaces differ in a way that would invalidate a benchmark
comparison.

Only fields that a precursor library actually encodes are derived. Fields that a
library cannot encode (enzyme, missed cleavages, fragment m/z window, FDR
thresholds, fixed-vs-variable mod classification) are written as the literal
"unknown_from_library" with an explanatory note, so a reader never mistakes an
absent value for a real setting.

The output is deterministic: no wall-clock timestamps, sorted keys where order is
not semantically meaningful, and a SHA-256 content hash of the input library for
provenance.

Interpreter: C:/Users/robbi/anaconda3/envs/py312_mumdia/python.exe
(pyarrow, pandas, numpy, pyyaml, stdlib tomllib/hashlib).

Example
-------
python search_space_manifest.py \
    --library-precursors C:/proteobench/lib/lib_precursors_ft.parquet \
    --out C:/proteobench/accept/ecoli_search_space.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

UNKNOWN = "unknown_from_library"

# Bracketed UniMod-style modification token, e.g. "[Carbamidomethyl]".
_MOD_RE = re.compile(r"\[[^\]]*\]")
_MOD_TOKEN_RE = re.compile(r"\[([^\]]*)\]")
_DECOY_RE = re.compile(r"^DECOY_")


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in fixed-size chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def resolve_engine_version(script_path: Path) -> str:
    """Resolve the MuMDIA engine version from the Cargo manifests.

    mumdia-core declares ``version.workspace = true``, so the literal version
    lives in the workspace root manifest. Falls back to "unknown".
    """
    repo_root = script_path.resolve().parents[1]
    core = repo_root / "rust" / "mumdia" / "crates" / "mumdia-core" / "Cargo.toml"
    workspace = repo_root / "rust" / "mumdia" / "Cargo.toml"
    try:
        with core.open("rb") as fh:
            core_cfg = tomllib.load(fh)
        ver = core_cfg.get("package", {}).get("version")
        if isinstance(ver, str):
            return ver
        with workspace.open("rb") as fh:
            ws_cfg = tomllib.load(fh)
        ver = ws_cfg.get("workspace", {}).get("package", {}).get("version")
        if isinstance(ver, str):
            return ver
    except (OSError, tomllib.TOMLDecodeError):
        pass
    return "unknown"


def _norm_col(name: str) -> str:
    """Normalize a column name to lowercase alphanumerics for matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first column whose normalized name matches a candidate."""
    norm = {_norm_col(c): c for c in columns}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    return None


# --------------------------------------------------------------------------- #
# Manifest derivation
# --------------------------------------------------------------------------- #
def derive_manifest(lib_path: Path, engine_version: str) -> dict[str, Any]:
    """Derive an effective search-space manifest from a library-precursors table."""
    needed = ["peptidoform", "charge", "precursor_mz", "label", "protein"]
    schema_names = pq.read_schema(lib_path).names
    read_cols = [c for c in needed if c in schema_names]
    tbl = pq.read_table(lib_path, columns=read_cols)
    n_rows = tbl.num_rows

    # Charge and m/z ranges are the same for targets and decoys, so compute
    # them over all precursors for a faithful picture of the extraction space.
    charge_arr = tbl.column("charge")
    mz_arr = tbl.column("precursor_mz")
    charge_mm = pc.min_max(charge_arr)
    mz_mm = pc.min_max(mz_arr)
    min_charge = charge_mm["min"].as_py()
    max_charge = charge_mm["max"].as_py()
    min_mz = mz_mm["min"].as_py()
    max_mz = mz_mm["max"].as_py()

    # Charge histogram over all precursors, sorted by charge.
    vc = pc.value_counts(charge_arr)
    charge_hist = {
        int(struct["values"].as_py()): int(struct["counts"].as_py()) for struct in vc
    }
    charge_hist = dict(sorted(charge_hist.items()))

    # Target / decoy split.
    label_arr = tbl.column("label")
    is_target = pc.equal(label_arr, "target")
    n_target = int(pc.sum(pc.cast(is_target, pa.int64())).as_py())
    n_decoy = n_rows - n_target

    # String-level statistics are computed over the target subset (the real
    # search space); decoys are artificial reverse/scramble sequences.
    targets = tbl.filter(is_target).select(
        [c for c in ["peptidoform", "protein"] if c in read_cols]
    )
    tdf = targets.to_pandas()
    pforms = tdf["peptidoform"].astype("string")

    stripped = pforms.str.replace(_MOD_RE, "", regex=True).str.replace(
        _DECOY_RE, "", regex=True
    )
    lengths = stripped.str.len().to_numpy(dtype="int64")
    n_distinct_stripped = int(stripped.nunique())

    has_mod = pforms.str.contains("[", regex=False)
    n_with_mod = int(has_mod.sum())
    observed_mods: set[str] = set()
    for pf in pforms[has_mod]:
        observed_mods.update(_MOD_TOKEN_RE.findall(pf))
    observed_mods_sorted = sorted(observed_mods)

    # Distinct proteins over targets, splitting group strings on ';'.
    prot_series = tdf["protein"].astype("string") if "protein" in tdf else None
    if prot_series is not None:
        prot_set: set[str] = set()
        for grp in prot_series.dropna().unique():
            for pid in str(grp).split(";"):
                pid = pid.strip()
                if pid:
                    prot_set.add(pid)
        n_distinct_proteins: int | str = len(prot_set)
    else:
        n_distinct_proteins = UNKNOWN

    length_note = (
        "peptide lengths derived from stripped target sequences; enzyme, "
        "specificity, and missed cleavages are search-time settings not stored "
        "in a precursor library"
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "provenance": {
            "source_library": str(lib_path),
            "source_sha256": sha256_file(lib_path),
            "source_bytes": lib_path.stat().st_size,
            "n_rows": n_rows,
            "engine": "mumdia",
            "engine_version": engine_version,
            "generator": "search_space_manifest.py",
            "manifest_schema_version": 1,
        },
        "digestion": {
            "enzyme": UNKNOWN,
            "specificity": UNKNOWN,
            "missed_cleavages": UNKNOWN,
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "note": length_note,
        },
        "precursors": {
            "min_charge": int(min_charge),
            "max_charge": int(max_charge),
            "min_mz": round(float(min_mz), 4),
            "max_mz": round(float(max_mz), 4),
        },
        "fragments": {
            "ion_series": UNKNOWN,
            "min_mz": UNKNOWN,
            "max_mz": UNKNOWN,
            "note": "fragment ion series and m/z window require the fragment "
            "table (lib_fragments), not provided here",
        },
        "modifications": {
            "fixed": UNKNOWN,
            "variable": UNKNOWN,
            "max_variable_modifications": UNKNOWN,
            "observed_mod_tokens": observed_mods_sorted,
            "n_peptidoforms_with_mod": n_with_mod,
            "note": "fixed vs variable classification cannot be inferred from a "
            "library; observed bracketed tokens are listed as-is",
        },
        "fdr": {
            "precursor": UNKNOWN,
            "peptide": UNKNOWN,
            "protein": UNKNOWN,
            "note": "FDR thresholds are a search-time setting, not stored in the "
            "library",
        },
        "derived": {
            "n_precursors": n_rows,
            "n_target_precursors": n_target,
            "n_decoy_precursors": n_decoy,
            "n_distinct_stripped_sequences": n_distinct_stripped,
            "n_distinct_proteins": n_distinct_proteins,
            "charge_histogram": charge_hist,
            "peptide_length": {
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "median": float(np.median(lengths)),
            },
        },
    }
    return manifest


# --------------------------------------------------------------------------- #
# MuMDIA target precursor keys (for --diann overlap)
# --------------------------------------------------------------------------- #
def mumdia_target_keys(lib_path: Path) -> tuple[set[tuple[str, int]], dict[str, Any]]:
    """Return {(stripped_seq_upper, charge)} for target precursors plus ranges."""
    tbl = pq.read_table(
        lib_path, columns=["peptidoform", "charge", "precursor_mz", "label"]
    )
    is_target = pc.equal(tbl.column("label"), "target")
    tdf = tbl.filter(is_target).to_pandas()
    stripped = (
        tdf["peptidoform"]
        .astype("string")
        .str.replace(_MOD_RE, "", regex=True)
        .str.replace(_DECOY_RE, "", regex=True)
        .str.upper()
    )
    charges = tdf["charge"].astype("int64")
    keys = set(zip(stripped.tolist(), charges.tolist()))
    ranges = {
        "charge_range": [int(charges.min()), int(charges.max())],
        "mz_range": [
            round(float(tdf["precursor_mz"].min()), 4),
            round(float(tdf["precursor_mz"].max()), 4),
        ],
    }
    return keys, ranges


# --------------------------------------------------------------------------- #
# Reference (DIA-NN) loading
# --------------------------------------------------------------------------- #
def load_reference_keys(path: Path) -> tuple[set[tuple[str, int]], dict[str, Any]]:
    """Load (stripped_seq_upper, charge) keys and ranges from a DIA-NN report.

    Accepts Parquet or a tab-separated report. Column names are matched
    case-insensitively against the usual DIA-NN header set.
    """
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        cols = pq.read_schema(path).names
    else:
        import pandas as pd

        cols = list(pd.read_csv(path, sep="\t", nrows=0).columns)

    stripped_col = _find_col(
        cols, ["strippedsequence", "peptide", "sequence", "pepseq"]
    )
    charge_col = _find_col(cols, ["precursorcharge", "charge", "z"])
    mz_col = _find_col(cols, ["precursormz", "mz"])
    if stripped_col is None or charge_col is None:
        raise ValueError(
            f"could not locate stripped-sequence and charge columns in {path}; "
            f"available columns: {cols}"
        )

    read_cols = [stripped_col, charge_col] + ([mz_col] if mz_col else [])
    if suffix in {".parquet", ".pq"}:
        df = pq.read_table(path, columns=read_cols).to_pandas()
    else:
        import pandas as pd

        df = pd.read_csv(path, sep="\t", usecols=read_cols)

    df = df.dropna(subset=[stripped_col, charge_col])
    stripped = df[stripped_col].astype("string").str.upper()
    charges = df[charge_col].astype("int64")
    keys = set(zip(stripped.tolist(), charges.tolist()))
    ranges: dict[str, Any] = {
        "charge_range": [int(charges.min()), int(charges.max())],
    }
    if mz_col:
        ranges["mz_range"] = [
            round(float(df[mz_col].min()), 4),
            round(float(df[mz_col].max()), 4),
        ]
    return keys, ranges


# --------------------------------------------------------------------------- #
# Comparison logic
# --------------------------------------------------------------------------- #
def compare_overlap(
    mumdia_keys: set[tuple[str, int]],
    mumdia_ranges: dict[str, Any],
    ref_keys: set[tuple[str, int]],
    ref_ranges: dict[str, Any],
    ref_label: str,
    mz_tol: float,
) -> dict[str, Any]:
    """Compute (stripped_seq, charge) precursor-key overlap and range mismatches."""
    shared = mumdia_keys & ref_keys
    only_mum = mumdia_keys - ref_keys
    only_ref = ref_keys - mumdia_keys
    n_ref = len(ref_keys)
    n_mum = len(mumdia_keys)

    charge_mismatch = mumdia_ranges["charge_range"] != ref_ranges.get("charge_range")
    mz_mismatch = False
    if "mz_range" in ref_ranges and "mz_range" in mumdia_ranges:
        m = mumdia_ranges["mz_range"]
        r = ref_ranges["mz_range"]
        mz_mismatch = abs(m[0] - r[0]) > mz_tol or abs(m[1] - r[1]) > mz_tol

    return {
        "reference": ref_label,
        "n_mumdia_target_keys": n_mum,
        "n_reference_keys": n_ref,
        "n_shared": len(shared),
        "n_only_in_mumdia": len(only_mum),
        "n_only_in_reference": len(only_ref),
        "shared_fraction_of_reference": (len(shared) / n_ref) if n_ref else 0.0,
        "shared_fraction_of_mumdia": (len(shared) / n_mum) if n_mum else 0.0,
        "charge_range_mumdia": mumdia_ranges["charge_range"],
        "charge_range_reference": ref_ranges.get("charge_range"),
        "mz_range_mumdia": mumdia_ranges.get("mz_range"),
        "mz_range_reference": ref_ranges.get("mz_range"),
        "charge_range_mismatch": charge_mismatch,
        "mz_range_mismatch": mz_mismatch,
    }


def compare_declared(
    derived: dict[str, Any], other: dict[str, Any], mz_tol: float
) -> dict[str, Any]:
    """Compare a derived manifest against a declared manifest, field by field."""
    diffs: list[dict[str, Any]] = []

    def cmp(section: str, field: str, material_if_diff: bool, tol: float = 0.0) -> None:
        a = derived.get(section, {}).get(field)
        b = other.get(section, {}).get(field)
        if a is None or b is None:
            return
        if a == UNKNOWN or b == UNKNOWN:
            diffs.append(
                {
                    "field": f"{section}.{field}",
                    "derived": a,
                    "declared": b,
                    "status": "not_comparable",
                    "material": False,
                }
            )
            return
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and tol > 0.0:
            differs = abs(float(a) - float(b)) > tol
        else:
            differs = a != b
        diffs.append(
            {
                "field": f"{section}.{field}",
                "derived": a,
                "declared": b,
                "status": "mismatch" if differs else "match",
                "material": bool(differs and material_if_diff),
            }
        )

    cmp("digestion", "enzyme", True)
    cmp("digestion", "specificity", True)
    cmp("digestion", "missed_cleavages", True)
    cmp("digestion", "min_length", True)
    cmp("digestion", "max_length", True)
    cmp("precursors", "min_charge", True)
    cmp("precursors", "max_charge", True)
    cmp("precursors", "min_mz", True, tol=mz_tol)
    cmp("precursors", "max_mz", True, tol=mz_tol)
    cmp("modifications", "max_variable_modifications", True)
    cmp("fdr", "precursor", True)
    cmp("fdr", "peptide", True)
    cmp("fdr", "protein", True)

    return {
        "reference": "declared_manifest",
        "fields": diffs,
        "n_material_mismatches": sum(1 for d in diffs if d["material"]),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_overlap(cmp: dict[str, Any], min_shared_frac: float) -> bool:
    """Print an overlap diff. Return True if it constitutes a material mismatch."""
    print(f"\n=== search-space overlap vs {cmp['reference']} ===")
    print(f"  MuMDIA target keys : {cmp['n_mumdia_target_keys']}")
    print(f"  reference keys     : {cmp['n_reference_keys']}")
    print(f"  shared             : {cmp['n_shared']}")
    print(f"  only in MuMDIA     : {cmp['n_only_in_mumdia']}")
    print(f"  only in reference  : {cmp['n_only_in_reference']}")
    print(
        f"  shared / reference : {cmp['shared_fraction_of_reference']:.4f} "
        f"(threshold {min_shared_frac:.4f})"
    )
    print(f"  shared / MuMDIA    : {cmp['shared_fraction_of_mumdia']:.4f}")
    print(
        f"  charge range       : MuMDIA {cmp['charge_range_mumdia']} vs "
        f"reference {cmp['charge_range_reference']}"
        + ("  MISMATCH" if cmp["charge_range_mismatch"] else "")
    )
    print(
        f"  m/z range          : MuMDIA {cmp['mz_range_mumdia']} vs "
        f"reference {cmp['mz_range_reference']}"
        + ("  MISMATCH" if cmp["mz_range_mismatch"] else "")
    )
    low_overlap = cmp["shared_fraction_of_reference"] < min_shared_frac
    if low_overlap:
        print("  -> shared fraction below threshold")
    return low_overlap or cmp["charge_range_mismatch"] or cmp["mz_range_mismatch"]


def print_declared(cmp: dict[str, Any]) -> bool:
    """Print a declared-manifest diff. Return True if any material mismatch."""
    print("\n=== declared-manifest comparison ===")
    for d in cmp["fields"]:
        flag = {
            "match": "ok",
            "mismatch": "MISMATCH",
            "not_comparable": "n/a",
        }[d["status"]]
        material = "  (material)" if d["material"] else ""
        print(
            f"  {d['field']:<38} derived={d['derived']!r:<24} "
            f"declared={d['declared']!r:<24} {flag}{material}"
        )
    print(f"  material mismatches: {cmp['n_material_mismatches']}")
    return cmp["n_material_mismatches"] > 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Derive and validate a MuMDIA search-space manifest (P0.1)."
    )
    p.add_argument(
        "--library-precursors",
        required=True,
        type=Path,
        help="MuMDIA library-precursors Parquet.",
    )
    p.add_argument("--out", type=Path, help="Output manifest YAML (default: stdout).")
    p.add_argument(
        "--compare", type=Path, help="Declared manifest YAML to compare against."
    )
    p.add_argument(
        "--diann", type=Path, help="DIA-NN report (Parquet or TSV) to compare against."
    )
    p.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit nonzero on a material search-space mismatch.",
    )
    p.add_argument(
        "--min-shared-frac",
        type=float,
        default=0.9,
        help="Minimum shared fraction of reference precursors (default 0.9).",
    )
    p.add_argument(
        "--mz-range-tol",
        type=float,
        default=5.0,
        help="Tolerance (Th) for treating m/z-range endpoints as equal (default 5.0).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.library_precursors.exists():
        print(f"error: library not found: {args.library_precursors}", file=sys.stderr)
        return 1

    engine_version = resolve_engine_version(Path(__file__))
    manifest = derive_manifest(args.library_precursors, engine_version)

    yaml_text = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml_text, encoding="utf-8")
        print(f"wrote manifest: {args.out}")
    else:
        print(yaml_text)

    material_mismatch = False

    if args.compare or args.diann:
        mum_keys, mum_ranges = mumdia_target_keys(args.library_precursors)

        if args.diann:
            ref_keys, ref_ranges = load_reference_keys(args.diann)
            ocmp = compare_overlap(
                mum_keys,
                mum_ranges,
                ref_keys,
                ref_ranges,
                str(args.diann),
                args.mz_range_tol,
            )
            material_mismatch |= print_overlap(ocmp, args.min_shared_frac)

        if args.compare:
            with args.compare.open("rb") as fh:
                other = yaml.safe_load(fh)
            dcmp = compare_declared(manifest, other, args.mz_range_tol)
            material_mismatch |= print_declared(dcmp)

    if material_mismatch and args.fail_on_mismatch:
        print("\nFAIL: material search-space mismatch detected.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
