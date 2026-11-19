"""Merge an empirical spectral library with a predicted one into one search library (#320).

An empirical library (spectra and retention times observed in earlier runs) covers the
peptides that were identified there; a library predicted from a FASTA covers the whole
digest. This helper searches both as one library: every entry of the predicted library, with
the empirical spectrum and retention time where the empirical library has the entry.

Inputs are two MuMDIA libraries (precursor and fragment tables): an empirical library
imported with `import_diann_lib.py` (a library DIA-NN wrote from a search, such as its
`report-lib.parquet`) or written otherwise in the library schema, and a predicted library
(imported, or built with `mumdia predict-frag` from a FASTA). Only their targets are read: the
decoys of the merged target set are built again, because a decoy population must be paired
with exactly the targets it is searched with.

Merge rules:

- Entries are keyed on `(peptidoform, charge)`, the ProForma peptidoform both libraries
  write. An entry in both takes the spectrum (fragments), retention time and 1/K0 of the
  library `--precedence` names, `empirical` by default; `predicted` keeps the predicted
  entry and uses the empirical library only for the entries the prediction lacks.
- An entry only in the empirical library is kept (`--empirical-only drop` removes it, so the
  search space stays the predicted library's); an entry only in the predicted library is
  kept.
- Protein: an entry the predicted library has takes the predicted library's protein, which
  names the proteins of the FASTA the search is meant for; an empirical-only entry keeps its
  own. `remap_library.py` maps the merged library to one FASTA when the two disagree. The
  annotation columns follow the protein; a library without them gives empty cells.
- Retention time: the two libraries' `predicted_irt` are on different scales (observed
  minutes or an empirical iRT against a predictor's own scale). With `--rt calibrate` (the
  default) the empirical values are mapped onto the predicted scale by a monotone curve fitted
  on the shared entries: the pairs sorted by empirical iRT, cut into equal-count bins (80, or
  fewer below 40 pairs a bin), the median of each side per bin, the predicted side made
  non-decreasing, linear in between and linearly extrapolated past the outer bins (the
  curve of the engine's second-pass consensus, docs/12). `--rt keep` writes the empirical
  values unchanged, for two libraries already on one scale. The fit is reported with the
  median absolute residual of the shared entries. A search that re-predicts the library's
  retention times (`rt_im_train.library_irt = auto` with a DeepLC interpreter, the default)
  replaces every entry's value anyway; `rt_im_train.library_irt = library` keeps them.
- Intensities are max-normalised per precursor in both halves. `base_peptide_id` is
  recomputed from the stripped sequence and `peptidoform_id` from the peptidoform, over the
  merged targets, so a peptide's forms from either library share one competition group.
  Rows are sorted by precursor m/z with `candidate_id` the contiguous range the engine
  requires; `n_fragments` and the fragment `cardinality` (how many merged targets share each
  0.01 Da fragment bin, as the importer counts it) are recomputed.

Decoys (`--decoys`):

- `diann` (default, needs `--diann <exe>`): reversed sequences (`make_reverse_decoys.py`)
  whose spectra DIA-NN predicts (`predict_decoys.py build`), the default decoy rule for an
  imported library since 2026-10-09: decoys with copied intensities made the reported FDR
  optimistic, so this helper never writes them.
- `none`: write the merged targets only, and add the decoys with those two steps by hand.

`--decoy-fragments match` (the default) then trims each decoy to the fragment count of its
target, keeping its most intense predicted fragments, so a decoy of an empirical entry with
eight observed fragments is not a twelve-fragment prediction: the fragment count of the two
members of a pair stays equal, as it is in a library whose two halves come from one
predictor. `all` keeps every predicted decoy fragment.

FDR caveat (measured, docs/13 "merge_libraries"): on the E. coli + human entrapment run the
merged library reported 5.2% more precursors at a precursor q of 1% than the predicted
library alone, at a measured entrapment FDP of 1.43% instead of 1.00%, and 0.8% fewer at a
measured FDP of 1%. An empirical library is the accepted list of an earlier search and holds
that search's false identifications with observed spectra, which their predicted decoys do
not match. Compare a merged library's counts at a measured FDP, not at the reported q.

Usage:
  python merge_libraries.py --empirical-precursors E_prec --empirical-fragments E_frag
      --predicted-precursors P_prec --predicted-fragments P_frag
      --out-precursors OUT_prec --out-fragments OUT_frag
      [--precedence empirical|predicted] [--empirical-only keep|drop]
      [--rt calibrate|keep] [--rt-min-shared 50]
      [--decoys diann --diann <exe> [--threads N] [--keep-unimod 4,35,...] | --decoys none]
      [--decoy-fragments match|all] [--work-dir DIR] [--summary out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _lib_io import (ANNOTATION_COLUMNS, MZ_BIN_MAX, embed_annotations,  # noqa: E402
                     has_annotation_columns, mz_bins, narrow_table, retire_side_table,
                     stripped, write_engine_parquet)

# The precursor columns every library has; others are optional and kept when present.
_REQUIRED = ("candidate_id", "peptidoform", "charge", "precursor_mz", "predicted_irt", "label",
             "protein")


def read_targets(prec_path, frag_path, what):
    """The target precursors of a library, with their annotations in columns, and the target
    fragments; each precursor row keeps its own `candidate_id` (`src_id`)."""
    prec = pd.read_parquet(prec_path)
    missing = [c for c in _REQUIRED if c not in prec.columns]
    if missing:
        raise SystemExit(f"{what} library {prec_path}: not a MuMDIA precursor table, missing "
                         f"{missing}")
    prec, _ = embed_annotations(prec, prec_path)
    n_all = len(prec)
    prec = prec[prec["label"].astype(str) == "target"].copy()
    if prec.empty:
        raise SystemExit(f"{what} library {prec_path}: no target precursors")
    dup = prec.duplicated(["peptidoform", "charge"])
    if dup.any():
        raise SystemExit(f"{what} library {prec_path}: {int(dup.sum())} target rows repeat a "
                         f"(peptidoform, charge) key")
    prec = prec.rename(columns={"candidate_id": "src_id"})
    want = np.zeros(int(prec["src_id"].max()) + 1, dtype=bool)
    want[prec["src_id"].to_numpy().astype(np.int64)] = True
    parts = []
    pf = pq.ParquetFile(frag_path)
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i)
        cid = t.column("candidate_id").to_numpy().astype(np.int64)
        inside = cid < want.size
        sel = np.zeros(cid.size, dtype=bool)
        sel[inside] = want[cid[inside]]
        if sel.any():
            parts.append(t.filter(pa.array(sel)))
    pf.close()
    frag = pa.concat_tables(parts).to_pandas() if parts else pd.DataFrame()
    print(f"{what}: {len(prec)} target precursors of {n_all} rows, {len(frag)} target "
          f"fragments ({prec_path})", flush=True)
    return prec.reset_index(drop=True), frag.rename(columns={"candidate_id": "src_id"})


def monotone_curve(x, y, bins=80, min_per_bin=40):
    """(knots_x, knots_y) of the monotone binned-median curve mapping x onto y (docs/12,
    second-pass consensus): pairs sorted by x, equal-count bins, per-bin medians, y made
    non-decreasing."""
    order = np.argsort(x, kind="mergesort")
    x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
    n_bins = int(max(2, min(bins, len(x) // min_per_bin)))
    edges = np.linspace(0, len(x), n_bins + 1).round().astype(int)
    kx, ky = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            kx.append(float(np.median(x[a:b])))
            ky.append(float(np.median(y[a:b])))
    kx, ky = np.asarray(kx), np.maximum.accumulate(np.asarray(ky))
    # Equal knots (repeated x medians) would divide by zero in the extrapolation.
    keep = np.r_[True, np.diff(kx) > 0]
    return kx[keep], ky[keep]


def apply_curve(kx, ky, x):
    """The curve at x: linear between knots, linear past the outer knots with the slope of
    the outer segment."""
    x = np.asarray(x, float)
    if len(kx) == 1:
        return x - kx[0] + ky[0]
    out = np.interp(x, kx, ky)
    lo, hi = x < kx[0], x > kx[-1]
    s_lo = (ky[1] - ky[0]) / (kx[1] - kx[0])
    s_hi = (ky[-1] - ky[-2]) / (kx[-1] - kx[-2])
    out[lo] = ky[0] + (x[lo] - kx[0]) * s_lo
    out[hi] = ky[-1] + (x[hi] - kx[-1]) * s_hi
    return out


def cardinality(frag_mz, frag_cid):
    """Per fragment, how many distinct precursors have a fragment in its 0.01 Da bin."""
    b = mz_bins(frag_mz)
    pairs = pd.DataFrame({"c": np.asarray(frag_cid, np.int64), "b": b}).drop_duplicates()
    card = np.bincount(pairs["b"].to_numpy(), minlength=MZ_BIN_MAX)
    return card[b].astype(np.int32)


def merge_targets(emp, emp_frag, pred, pred_frag, precedence="empirical", empirical_only="keep",
                  rt="calibrate", min_shared=50, log=print):
    """The merged target tables (precursors, fragments) and a summary dict."""
    if precedence not in ("empirical", "predicted"):
        raise ValueError(precedence)
    key_e = list(zip(emp["peptidoform"].astype(str), emp["charge"].astype(int)))
    key_p = list(zip(pred["peptidoform"].astype(str), pred["charge"].astype(int)))
    pos_p = {k: i for i, k in enumerate(key_p)}
    in_pred = np.array([k in pos_p for k in key_e], dtype=bool)
    shared_e = np.flatnonzero(in_pred)
    shared_p = np.array([pos_p[key_e[i]] for i in shared_e], dtype=np.int64)
    only_e = np.flatnonzero(~in_pred)

    # Retention time of the empirical rows on the predicted scale.
    emp_rt = emp["predicted_irt"].astype(float).to_numpy()
    fit = {"mode": rt, "shared": int(len(shared_e))}
    if rt == "calibrate":
        if len(shared_e) < min_shared:
            raise SystemExit(f"--rt calibrate needs at least {min_shared} entries in both "
                             f"libraries (--rt-min-shared), {len(shared_e)} found; use --rt keep "
                             f"for libraries on one scale")
        x = emp_rt[shared_e]
        y = pred["predicted_irt"].astype(float).to_numpy()[shared_p]
        kx, ky = monotone_curve(x, y)
        emp_rt = apply_curve(kx, ky, emp_rt)
        resid = np.abs(emp_rt[shared_e] - y)
        fit.update(knots=int(len(kx)), median_abs_residual=float(np.median(resid)),
                   p95_abs_residual=float(np.percentile(resid, 95)),
                   empirical_range=[float(x.min()), float(x.max())],
                   predicted_range=[float(y.min()), float(y.max())])
        log(f"rt: empirical iRT mapped onto the predicted scale by a {len(kx)}-knot monotone "
            f"curve fitted on {len(shared_e)} shared entries; median |residual| "
            f"{fit['median_abs_residual']:.3g}, p95 {fit['p95_abs_residual']:.3g} (predicted "
            f"range {fit['predicted_range'][0]:.4g} to {fit['predicted_range'][1]:.4g})")
    elif rt != "keep":
        raise ValueError(rt)
    emp = emp.assign(predicted_irt=emp_rt.astype(np.float32))

    # Which source each merged entry takes its spectrum, RT and 1/K0 from.
    take_emp_shared = precedence == "empirical"
    use_p = np.ones(len(pred), dtype=bool)
    if take_emp_shared:
        use_p[shared_p] = False
    use_e = np.zeros(len(emp), dtype=bool)
    if take_emp_shared:
        use_e[shared_e] = True
    if empirical_only == "keep":
        use_e[only_e] = True
    elif empirical_only != "drop":
        raise ValueError(empirical_only)

    # Columns: the union, every row of the other library's columns null or empty.
    ann = has_annotation_columns(emp.columns) or has_annotation_columns(pred.columns)
    cols = ["src_id", "peptidoform", "charge", "precursor_mz", "predicted_irt", "label",
            "protein"]
    has_im = "predicted_im" in emp.columns or "predicted_im" in pred.columns
    if has_im:
        cols.append("predicted_im")
    if ann:
        cols += list(ANNOTATION_COLUMNS)

    def part(df, sel, source):
        out = pd.DataFrame(index=np.arange(int(sel.sum())))
        sub = df.loc[sel].reset_index(drop=True)
        for c in cols:
            if c in sub.columns:
                col = sub[c]
                # Categorical annotation columns (`_lib_io.annotate_precursors`) as plain
                # values, so the two halves concatenate and shared rows can be overwritten.
                out[c] = (col.astype(object).to_numpy() if isinstance(col.dtype, pd.CategoricalDtype)
                          else col.to_numpy())
            elif c == "predicted_im":
                out[c] = np.full(len(sub), np.nan)
            else:
                out[c] = ""
        out["source"] = source
        return out

    pe = part(emp, use_e, "empirical")
    pp = part(pred, use_p, "predicted")
    # An entry in both keeps the predicted library's protein (the search's FASTA).
    if take_emp_shared and len(shared_e):
        prot_cols = ["protein"] + (list(ANNOTATION_COLUMNS) if ann else [])
        emp_sel = np.flatnonzero(use_e)
        row_of = {int(i): j for j, i in enumerate(emp_sel)}
        rows = np.array([row_of[int(i)] for i in shared_e], dtype=np.int64)
        for c in prot_cols:
            if c in pred.columns:
                pe.loc[rows, c] = pred[c].to_numpy()[shared_p]
        pe.loc[rows, "source"] = "empirical (shared)"
    merged = pd.concat([pe, pp], ignore_index=True)
    merged["label"] = "target"
    for c in ("protein", *([*ANNOTATION_COLUMNS] if ann else [])):
        merged[c] = merged[c].fillna("").astype(str)
    merged.loc[merged["protein"].str.strip() == "", "protein"] = "UNASSIGNED"

    # Fragments of the chosen rows, max-normalised per precursor, on new contiguous ids.
    merged = merged.sort_values("precursor_mz", kind="mergesort").reset_index(drop=True)
    merged["candidate_id"] = np.arange(len(merged), dtype=np.uint32)
    frags = []
    for source, ff in (("empirical", emp_frag), ("predicted", pred_frag)):
        sel = merged["source"].str.startswith(source).to_numpy()
        lut = pd.Series(merged["candidate_id"].to_numpy()[sel],
                        index=merged["src_id"].to_numpy()[sel].astype(np.int64))
        f = ff[ff["src_id"].isin(lut.index)].copy()
        f["candidate_id"] = lut.loc[f["src_id"].astype(np.int64)].to_numpy().astype(np.uint32)
        frags.append(f.drop(columns=["src_id"]))
    fcols = ["candidate_id", "mz", "predicted_intensity", "name", "ion_type", "ordinal",
             "frag_charge"]
    for f in frags:
        miss = [c for c in fcols if c not in f.columns]
        if miss:
            raise SystemExit(f"a fragment table lacks {miss}")
    frag = pd.concat([f[fcols] for f in frags], ignore_index=True)
    frag = frag.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)
    mx = frag.groupby("candidate_id")["predicted_intensity"].transform("max")
    frag["predicted_intensity"] = np.where(mx > 0, frag["predicted_intensity"] / mx,
                                           0.0).astype(np.float32)
    frag["cardinality"] = cardinality(frag["mz"], frag["candidate_id"])
    nfrag = np.bincount(frag["candidate_id"].to_numpy().astype(np.int64), minlength=len(merged))
    if (nfrag == 0).any():
        raise SystemExit(f"{int((nfrag == 0).sum())} merged precursors have no fragments")
    merged["n_fragments"] = nfrag.astype(np.int32)

    seq = merged["peptidoform"].map(stripped)
    merged["base_peptide_id"] = pd.factorize(seq)[0].astype(np.uint32)
    merged["peptidoform_id"] = pd.factorize(merged["peptidoform"])[0].astype(np.uint32)
    summary = {
        "precedence": precedence,
        "empirical_only": empirical_only,
        "empirical_targets": int(len(emp)),
        "predicted_targets": int(len(pred)),
        "shared": int(len(shared_e)),
        "empirical_only_entries": int(len(only_e)),
        "predicted_only_entries": int(len(pred) - len(shared_e)),
        "merged_targets": int(len(merged)),
        "targets_with_empirical_spectrum": int(merged["source"].str.startswith("empirical").sum()),
        "fragments": int(len(frag)),
        "rt": fit,
    }
    log(f"merge: {summary['shared']} entries in both libraries ({precedence} spectrum and RT), "
        f"{summary['empirical_only_entries']} only empirical"
        f"{' (dropped)' if empirical_only == 'drop' else ''}, "
        f"{summary['predicted_only_entries']} only predicted; {len(merged)} targets, "
        f"{summary['targets_with_empirical_spectrum']} with an empirical spectrum, "
        f"{len(frag)} fragments")
    order = ["candidate_id", "peptidoform_id", "base_peptide_id", "peptidoform", "charge",
             "precursor_mz", "predicted_irt", "label", "protein", "n_fragments"]
    if has_im:
        merged["predicted_im"] = pd.to_numeric(merged["predicted_im"], errors="coerce")
        order.append("predicted_im")
    if ann:
        order += list(ANNOTATION_COLUMNS)
    out = merged[order].copy()
    out["charge"] = out["charge"].astype(np.int32)
    out["precursor_mz"] = out["precursor_mz"].astype(np.float64)
    out["predicted_irt"] = out["predicted_irt"].astype(np.float32)
    frag["ordinal"] = frag["ordinal"].astype(np.int32)
    frag["frag_charge"] = frag["frag_charge"].astype(np.int32)
    frag["mz"] = frag["mz"].astype(np.float64)
    return out, frag, summary


def match_decoy_fragments(prec_path, frag_path):
    """Trim every decoy to the fragment count of its target (pair key `(peptidoform_id,
    charge)`), keeping its most intense fragments, in place. Returns the rows removed."""
    prec = pd.read_parquet(prec_path)
    t = prec[prec["label"] == "target"][["peptidoform_id", "charge", "n_fragments"]]
    d = prec[prec["label"] == "decoy"][["peptidoform_id", "charge", "candidate_id"]]
    pr = d.merge(t, on=["peptidoform_id", "charge"], how="left")
    if pr["n_fragments"].isna().any():
        raise SystemExit("decoys without a target on (peptidoform_id, charge)")
    limit = np.full(len(prec), np.iinfo(np.int32).max, dtype=np.int64)
    limit[pr["candidate_id"].to_numpy().astype(np.int64)] = pr["n_fragments"].to_numpy()
    frag = pq.read_table(frag_path).to_pandas()
    n0 = len(frag)
    frag["_rank"] = frag.groupby("candidate_id")["predicted_intensity"].rank(
        method="first", ascending=False)
    frag = frag[frag["_rank"] <= limit[frag["candidate_id"].to_numpy().astype(np.int64)]]
    frag = frag.drop(columns=["_rank"]).sort_values(["candidate_id"], kind="mergesort")
    counts = np.bincount(frag["candidate_id"].to_numpy().astype(np.int64), minlength=len(prec))
    prec["n_fragments"] = counts[prec["candidate_id"].to_numpy().astype(np.int64)].astype(np.int32)
    schema = pq.read_schema(frag_path)
    pq.write_table(narrow_table(pa.Table.from_pandas(frag, preserve_index=False)
                                .select(schema.names).cast(schema)),
                   frag_path, compression="snappy")
    write_engine_parquet(prec, prec_path)
    return n0 - len(frag)


def run(cmd):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--empirical-precursors", required=True)
    ap.add_argument("--empirical-fragments", required=True)
    ap.add_argument("--predicted-precursors", required=True)
    ap.add_argument("--predicted-fragments", required=True)
    ap.add_argument("--out-precursors", required=True)
    ap.add_argument("--out-fragments", required=True)
    ap.add_argument("--precedence", choices=["empirical", "predicted"], default="empirical",
                    help="which library an entry in both takes its spectrum, RT and 1/K0 from")
    ap.add_argument("--empirical-only", choices=["keep", "drop"], default="keep",
                    help="entries only the empirical library has")
    ap.add_argument("--rt", choices=["calibrate", "keep"], default="calibrate",
                    help="map the empirical iRT onto the predicted scale, or keep it")
    ap.add_argument("--rt-min-shared", type=int, default=50,
                    help="fewest shared entries the calibration is fitted on")
    ap.add_argument("--decoys", choices=["diann", "none"], default="diann")
    ap.add_argument("--diann", help="the DIA-NN executable that predicts the decoy spectra")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--keep-unimod", help="forwarded to predict_decoys.py")
    ap.add_argument("--decoy-fragments", choices=["match", "all"], default="match",
                    help="trim each decoy to its target's fragment count (default) or keep all")
    ap.add_argument("--work-dir", help="scratch directory (default: merge_work/ beside the output)")
    ap.add_argument("--summary", help="also write the counts as JSON here")
    a = ap.parse_args(argv)
    if a.decoys == "diann" and not a.diann:
        raise SystemExit("--decoys diann needs --diann <exe>; --decoys none writes the merged "
                         "targets only")
    work = a.work_dir or os.path.join(os.path.dirname(os.path.abspath(a.out_precursors)),
                                      "merge_work")
    os.makedirs(work, exist_ok=True)

    emp, emp_frag = read_targets(a.empirical_precursors, a.empirical_fragments, "empirical")
    pred, pred_frag = read_targets(a.predicted_precursors, a.predicted_fragments, "predicted")
    prec, frag, summary = merge_targets(emp, emp_frag, pred, pred_frag, a.precedence,
                                        a.empirical_only, a.rt, a.rt_min_shared)
    del emp, emp_frag, pred, pred_frag

    if a.decoys == "none":
        write_engine_parquet(prec, a.out_precursors)
        write_engine_parquet(frag, a.out_fragments)
        print("decoys: none written; add them with make_reverse_decoys.py and "
              "predict_decoys.py build before a search")
    else:
        tp, tf = os.path.join(work, "merged_targets_prec.parquet"), os.path.join(
            work, "merged_targets_frag.parquet")
        write_engine_parquet(prec, tp)
        write_engine_parquet(frag, tf)
        rp, rf = os.path.join(work, "merged_rev_prec.parquet"), os.path.join(
            work, "merged_rev_frag.parquet")
        run([sys.executable, os.path.join(HERE, "make_reverse_decoys.py"), tp, tf, rp, rf])
        cmd = [sys.executable, os.path.join(HERE, "predict_decoys.py"), "build", rp, rf,
               a.out_precursors, a.out_fragments, "--diann", a.diann, "--threads", a.threads,
               "--work", os.path.join(work, "decoy_prediction")]
        if a.keep_unimod:
            cmd += ["--keep-unimod", a.keep_unimod]
        run(cmd)
        if a.decoy_fragments == "match":
            removed = match_decoy_fragments(a.out_precursors, a.out_fragments)
            summary["decoy_fragments_trimmed"] = int(removed)
            print(f"decoys: {removed} predicted decoy fragments beyond their target's count "
                  f"removed (--decoy-fragments match)")
        out = pd.read_parquet(a.out_precursors, columns=["label"])
        summary["out_targets"] = int((out["label"] == "target").sum())
        summary["out_decoys"] = int((out["label"] == "decoy").sum())
    retire_side_table(a.predicted_precursors, a.out_precursors)
    if a.summary:
        with open(a.summary, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    print(f"DONE -> {a.out_precursors} / {a.out_fragments}")


if __name__ == "__main__":
    main(sys.argv[1:])
