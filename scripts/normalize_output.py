"""Normalize a MuMDIA scored table to the common benchmark schema.

Sensitivity-plan backlog item P0.2 (normalized output converter). The script is
non-invasive: it reads existing MuMDIA artifacts (a scored PSM table, and
optionally the psms and peptide-quant tables) and writes a single Parquet in the
common schema of sensitivity_plan spec 02 section 4:

    run_id, precursor_id, stripped_sequence, modified_sequence, charge,
    protein_ids, is_decoy, is_entrapment, precursor_mz, apex_rt, score, q_value,
    pep, quantity, engine, engine_version

Two export modes support the spec requirement to keep both "all top-scoring
candidates before FDR" and "final reported candidates":
    --all-candidates  keep every row (default)
    --reported-only   keep rows with q_value <= --q

The output is deterministic: rows are sorted by (run_id, precursor_id), and
precursor keys are reproducible because precursor_id is the stable MuMDIA
candidate_id.

Interpreter: C:/Users/robbi/anaconda3/envs/py312_mumdia/python.exe
(pyarrow, pandas, numpy, stdlib tomllib).

Example
-------
python normalize_output.py \
    --scored C:/proteobench/out_ecoli/scored.parquet \
    --psms   C:/proteobench/out_ecoli/psms.parquet \
    --out    C:/proteobench/accept/ecoli_normalized.parquet \
    --run-id ecoli
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Bracketed UniMod-style modification token and the decoy prefix.
_MOD_RE = re.compile(r"\[[^\]]*\]")
_DECOY_RE = re.compile(r"^DECOY_")

# Preference order for the discriminant score. The final rescore "score" is
# preferred over the pre-rescore "prelim_score" so that the reported score and
# q_value come from the same model; override with --score-col.
_SCORE_PREF = ["score", "prelim_score"]
# Candidate column names for a posterior error probability, if present.
_PEP_PREF = ["pep", "posterior_error_prob", "PEP", "posterior_error_probability"]

# Common schema, in the fixed spec-02 section-4 order.
_OUT_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("precursor_id", pa.uint32()),
        ("stripped_sequence", pa.string()),
        ("modified_sequence", pa.string()),
        ("charge", pa.int32()),
        ("protein_ids", pa.string()),
        ("is_decoy", pa.bool_()),
        ("is_entrapment", pa.bool_()),
        ("precursor_mz", pa.float64()),
        ("apex_rt", pa.float64()),
        ("score", pa.float64()),
        ("q_value", pa.float64()),
        ("pep", pa.float64()),
        ("quantity", pa.float64()),
        ("engine", pa.string()),
        ("engine_version", pa.string()),
    ]
)


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


def pick_column(columns: list[str], prefs: list[str]) -> str | None:
    """Return the first preferred column that is present, else None."""
    present = set(columns)
    for name in prefs:
        if name in present:
            return name
    return None


def normalize(
    scored: pd.DataFrame,
    psms: pd.DataFrame | None,
    pep_quant: pd.DataFrame | None,
    run_id: str,
    entrapment_substr: str,
    score_col: str | None,
    engine_version: str,
) -> pd.DataFrame:
    """Map MuMDIA columns onto the common normalized schema."""
    n = len(scored)
    out = pd.DataFrame(index=scored.index)

    out["run_id"] = run_id
    out["precursor_id"] = scored["candidate_id"].astype("uint32")

    pforms = scored["peptidoform"].astype("string")
    out["stripped_sequence"] = pforms.str.replace(_MOD_RE, "", regex=True).str.replace(
        _DECOY_RE, "", regex=True
    )
    out["modified_sequence"] = pforms
    out["charge"] = scored["charge"].astype("int32")
    out["protein_ids"] = scored["protein"].astype("string")
    out["is_decoy"] = (scored["label"].astype("string") == "decoy").astype(bool)
    out["is_entrapment"] = (
        scored["protein"].astype("string").str.contains(entrapment_substr, regex=False)
    ).fillna(False).astype(bool)

    # precursor_mz and apex_rt live in the psms table (not in scored); take them
    # from a join when available, otherwise from scored if it carries them, else
    # leave null.
    out["precursor_mz"] = _fill_from_join(
        scored, psms, "precursor_mz", "candidate_id"
    )
    out["apex_rt"] = _fill_from_join(scored, psms, "apex_rt", "candidate_id")

    if score_col is None:
        score_col = pick_column(list(scored.columns), _SCORE_PREF)
    if score_col is None:
        out["score"] = np.nan
    else:
        if score_col not in scored.columns:
            raise ValueError(f"score column {score_col!r} not in scored table")
        out["score"] = pd.to_numeric(scored[score_col], errors="coerce")

    out["q_value"] = pd.to_numeric(scored["q_value"], errors="coerce")

    pep_col = pick_column(list(scored.columns), _PEP_PREF)
    out["pep"] = (
        pd.to_numeric(scored[pep_col], errors="coerce")
        if pep_col
        else pd.Series(np.nan, index=scored.index)
    )

    out["quantity"] = _fill_from_join(
        scored, pep_quant, "quantity", "candidate_id"
    )

    out["engine"] = "mumdia"
    out["engine_version"] = engine_version

    assert len(out) == n
    return out


def _fill_from_join(
    scored: pd.DataFrame,
    other: pd.DataFrame | None,
    value_col: str,
    key: str,
) -> pd.Series:
    """Return value_col aligned to scored: from `other` via join, or from scored,
    else all-null. The join collapses duplicate keys in `other` to the first."""
    if other is not None and value_col in other.columns and key in other.columns:
        lut = other.drop_duplicates(subset=[key]).set_index(key)[value_col]
        return pd.to_numeric(scored[key].map(lut), errors="coerce")
    if value_col in scored.columns:
        return pd.to_numeric(scored[value_col], errors="coerce")
    return pd.Series(np.nan, index=scored.index)


def to_arrow(df: pd.DataFrame) -> pa.Table:
    """Build an Arrow table with the fixed common schema (NaN -> null)."""
    arrays = []
    for field in _OUT_SCHEMA:
        arrays.append(pa.array(df[field.name], type=field.type, from_pandas=True))
    return pa.Table.from_arrays(arrays, schema=_OUT_SCHEMA)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Normalize a MuMDIA scored table to the common schema (P0.2)."
    )
    p.add_argument("--scored", required=True, type=Path, help="MuMDIA scored Parquet.")
    p.add_argument(
        "--psms", type=Path, help="MuMDIA psms Parquet (for apex_rt, precursor_mz)."
    )
    p.add_argument(
        "--peptide-quant",
        type=Path,
        help="MuMDIA peptide-quant Parquet (for quantity).",
    )
    p.add_argument("--out", type=Path, help="Output normalized Parquet.")
    p.add_argument("--run-id", default="run", help="Run identifier (default 'run').")
    p.add_argument(
        "--entrapment-substr",
        default="_HUMAN",
        help="Substring flagging a protein as entrapment (default '_HUMAN').",
    )
    p.add_argument(
        "--score-col",
        default=None,
        help="Score column to use (default: first of 'score', 'prelim_score').",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--all-candidates",
        action="store_true",
        help="Keep all candidates (default).",
    )
    mode.add_argument(
        "--reported-only",
        action="store_true",
        help="Keep only rows with q_value <= --q.",
    )
    p.add_argument(
        "--q", type=float, default=0.01, help="q-value cutoff for --reported-only."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.scored.exists():
        print(f"error: scored table not found: {args.scored}", file=sys.stderr)
        return 1

    scored = pq.read_table(args.scored).to_pandas()
    psms = pq.read_table(args.psms).to_pandas() if args.psms else None
    pep_quant = (
        pq.read_table(args.peptide_quant).to_pandas() if args.peptide_quant else None
    )

    engine_version = resolve_engine_version(Path(__file__))
    out = normalize(
        scored,
        psms,
        pep_quant,
        run_id=args.run_id,
        entrapment_substr=args.entrapment_substr,
        score_col=args.score_col,
        engine_version=engine_version,
    )

    n_total = len(out)
    if args.reported_only:
        out = out[out["q_value"] <= args.q].copy()

    # Deterministic ordering: precursor_id is the stable candidate_id.
    out = out.sort_values(["run_id", "precursor_id"], kind="stable").reset_index(
        drop=True
    )

    table = to_arrow(out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, args.out)
        print(f"wrote normalized table: {args.out}")

    n_dec = int(out["is_decoy"].sum())
    n_ent = int(out["is_entrapment"].sum())
    mode = "reported-only" if args.reported_only else "all-candidates"
    print(
        f"rows: {len(out)} (of {n_total} scored; mode={mode}); "
        f"targets={len(out) - n_dec}, decoys={n_dec}, entrapment={n_ent}"
    )
    print(f"columns: {', '.join(f.name for f in _OUT_SCHEMA)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
