"""Shared helpers of the bench/tims scripts: the reference precursor table and sequence keys.

The reference is an independent identification of the same run, read from a precursor-level
parquet report with one row per precursor. The scripts read these columns by name; each
script names the subset it needs:

  Modified.Sequence    peptidoform in UniMod notation, e.g. `AM(UniMod:35)C(UniMod:4)K`
  Stripped.Sequence    plain amino-acid sequence
  Precursor.Id         precursor identifier (any string unique per peptidoform and charge)
  Precursor.Charge     precursor charge
  Precursor.Mz         precursor m/z
  RT                   apex retention time, in MINUTES
  IM                   apex 1/K0 (V s cm^-2); absent or null for a run without ion mobility
  Predicted.IM         the reference's own predicted 1/K0
  Q.Value              run-level precursor q
  Decoy                1 for a decoy row, 0 for a target; absent means all targets
  Precursor.Quantity   precursor abundance, for the abundance quintiles
  Protein.Group        protein group string
  PG.Q.Value           protein-group q

Analysis only: nothing here is imported by the engine or a worker.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.normpath(os.path.join(_HERE, "..", "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
# The UniMod -> ProForma-lite mapping of the library importer, which covers every entry of
# the engine's modification catalogue. Reused so that the reference peptidoform is spelled
# exactly as an imported library spells it.
from import_diann_lib import to_proforma  # noqa: E402

# A bracketed or parenthesised modification, in either notation.
_MOD = re.compile(r"\[[^\]]*\]|\([^)]*\)")
# The importer drops N-terminal modifications, so its mapping has no terminal form. The
# reference may still carry an N-terminal acetyl; spell it the ProForma way.
_NTERM_ACETYL = "(UniMod:1)"


def strip(pf: str) -> str:
    """Plain sequence of a MuMDIA or reference peptidoform: no modifications, no `DECOY_`."""
    return _MOD.sub("", str(pf).removeprefix("DECOY_")).replace("-", "").replace(".", "")


def reference_pf(modseq: str) -> str:
    """Reference `Modified.Sequence` (UniMod notation) to the MuMDIA ProForma-lite peptidoform."""
    s = str(modseq)
    if s.startswith(_NTERM_ACETYL):
        s = "[Acetyl]-" + s[len(_NTERM_ACETYL):]
    return to_proforma(s)


def il(seqs) -> set:
    """Set of sequences with I and L merged (the two are isobaric)."""
    return {s.replace("I", "L") for s in seqs}


def key(seq: pd.Series, z: pd.Series) -> pd.Series:
    """Precursor key: the I/L-merged plain sequence plus the charge, as `SEQ/z`."""
    return seq.str.replace("I", "L") + "/" + z.astype(int).astype(str)


def read_reference(path: str, required: list[str], optional: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read the named reference columns. A missing required column is an error that names it.
    A missing optional column is filled: `Decoy` with 0 (all targets), any other with NaN."""
    have = set(pq.read_schema(path).names)
    missing = [c for c in required if c not in have]
    if missing:
        raise SystemExit(f"{path}: reference table lacks column(s) {missing}")
    cols = list(required) + [c for c in optional if c in have]
    d = pd.read_parquet(path, columns=cols)
    for c in optional:
        if c not in d:
            d[c] = 0 if c == "Decoy" else np.nan
    return d


def _selfcheck() -> None:
    assert strip("DECOY_PEPM[Oxidation]C[Carbamidomethyl]K") == "PEPMCK"
    assert strip("[Acetyl]-AM(UniMod:35)K") == "AMK"
    assert reference_pf("(UniMod:1)AM(UniMod:35)C(UniMod:4)K") == "[Acetyl]-AM[Oxidation]C[Carbamidomethyl]K"
    assert reference_pf("AC(UniMod:4)K") == "AC[Carbamidomethyl]K"
    assert il(["LIK", "IAK"]) == {"LLK", "LAK"}
    assert key(pd.Series(["PIK"]), pd.Series([2])).tolist() == ["PLK/2"]


_selfcheck()
