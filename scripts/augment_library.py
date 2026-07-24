"""Augment an imported spectral library with the tryptic FASTA peptides it is
MISSING, so those peptides enter the search space (a completeness fix).

Motivation: an imported DIA-NN library can lack peptides that are perfectly real
in the sample. On the LFQ_Orbitrap_AIF_Ecoli_01 benchmark, every peptide DIA-NN
reported at 1% FDR but absent from the MuMDIA search database was an N-terminal
methionine-excision peptide. MuMDIA's digest now supports Met excision
(`digest.n_term_met_excision`, default on), so digesting the FASTA and adding the
peptides the imported library lacks recovers exactly those.

Pipeline (reusing the engine's own stages so peptidoform strings are byte
identical to what the search consumes):

    mumdia digest       FASTA           -> peptides           (Met excision on)
    mumdia peptidoforms peptides        -> peptidoforms       (mods + charges)
    set-diff vs imported target bases   -> missing peptidoforms
    mumdia predict-frag missing         -> missing target lib (native spectra+iRT)
    merge imported targets + missing    -> merged target lib
    make_shift_decoys.py merged targets -> augmented library  (paired shift decoys)

The RT axis of the predicted entries need not match the imported DIA-NN iRT axis:
the per-run DeepLC fine-tune (`rt_im_train.finetune_deeplc`) re-predicts iRT for
the whole library, putting every entry on one axis before extraction.

Usage:
  python augment_library.py \
    --fasta fasta/ecoli.fasta \
    --imported-precursors lib/lib_precursors.parquet \
    --imported-fragments  lib/lib_fragments.parquet \
    --out-precursors lib/lib_precursors_aug.parquet \
    --out-fragments  lib/lib_fragments_aug.parquet \
    --mumdia-bin /path/to/mumdia.exe \
    --config config.local-diann-lib.json \
    --work-dir <scratch> \
    [--match-level base_sequence|peptidoform_charge] \
    [--decoy-strategy shift|reverse]
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

STRIP_MODS = re.compile(r"\[[^\]]*\]")


def stripped(peptidoform: str) -> str:
    """Base amino-acid sequence: drop a DECOY_ prefix and bracketed mod blocks."""
    s = peptidoform[6:] if peptidoform.startswith("DECOY_") else peptidoform
    return STRIP_MODS.sub("", s)


def run_stage(mumdia_bin, subcmd, args, config):
    cmd = [mumdia_bin, subcmd] + args
    if config:
        cmd += ["--config", config]
    print(f"  $ {subcmd} {' '.join(args)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
        raise SystemExit(f"{subcmd} failed (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--imported-precursors", required=True)
    ap.add_argument("--imported-fragments", required=True)
    ap.add_argument("--out-precursors", required=True)
    ap.add_argument("--out-fragments", required=True)
    ap.add_argument("--mumdia-bin", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument(
        "--match-level",
        choices=["base_sequence", "peptidoform_charge"],
        default="base_sequence",
        help="base_sequence (default): add peptides whose stripped sequence is absent. "
        "peptidoform_charge: also add absent modforms/charges of present sequences "
        "(changes the FDR population; benchmark-gated).",
    )
    ap.add_argument(
        "--decoy-strategy",
        choices=["shift", "reverse"],
        default="shift",
        help="must match the strategy that built the imported library's decoys.",
    )
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    W = lambda n: os.path.join(args.work_dir, n)

    # 1-2. Digest FASTA + expand peptidoforms via the engine's own stages.
    print("[1/6] digest + peptidoforms (Met excision on)", flush=True)
    run_stage(args.mumdia_bin, "digest", ["--fasta", args.fasta, "--out", W("aug_pep.parquet")], args.config)
    run_stage(args.mumdia_bin, "peptidoforms", ["--peptides", W("aug_pep.parquet"), "--out", W("aug_pforms.parquet")], args.config)

    # 3. Set-diff against the imported library's target population.
    print("[2/6] diff against imported library", flush=True)
    imp = pd.read_parquet(args.imported_precursors)
    imp_t = imp[imp.label == "target"].copy()
    present_bases = set(imp_t.peptidoform.map(stripped))
    present_forms = set(zip(imp_t.peptidoform, imp_t.charge.astype(int)))

    pf = pd.read_parquet(W("aug_pforms.parquet"))
    pf = pf[pf.label == "target"].copy()
    base = pf.peptide.map(stripped) if "peptide" in pf.columns else pf.peptidoform.map(stripped)
    if args.match_level == "base_sequence":
        keep = ~base.isin(present_bases)
    else:
        keep = [(p, int(c)) not in present_forms for p, c in zip(pf.peptidoform, pf.charge)]
    missing = pf[np.asarray(keep)].copy()
    print(f"      imported targets: {len(imp_t)} ({len(present_bases)} base sequences)")
    print(f"      FASTA peptidoforms: {len(pf)}; MISSING to add: {len(missing)} "
          f"({missing.peptide.map(stripped).nunique() if 'peptide' in missing.columns else '?'} base sequences)")
    if missing.empty:
        raise SystemExit("nothing missing; imported library already covers the FASTA digest")

    # predict-frag input schema: id, base_peptide_id, peptidoform, charge, label, protein.
    missing_in = missing[["id", "base_peptide_id", "peptidoform", "charge", "label", "protein"]].copy()
    missing_in.to_parquet(W("aug_missing_pforms.parquet"), index=False)

    # 4. Predict spectra + iRT for the missing set (native; fine-tune re-predicts RT).
    print("[3/6] predict-frag on missing set", flush=True)
    run_stage(args.mumdia_bin, "predict-frag",
              ["--peptidoforms", W("aug_missing_pforms.parquet"),
               "--out-precursors", W("aug_missing_prec.parquet"),
               "--out-fragments", W("aug_missing_frag.parquet"),
               "--work-dir", args.work_dir], args.config)
    mprec = pd.read_parquet(W("aug_missing_prec.parquet"))
    mfrag = pd.read_parquet(W("aug_missing_frag.parquet"))

    # 5. Make ids disjoint from the imported library; keep sibling/base linkage.
    print("[4/6] merge imported targets + missing targets", flush=True)
    pfid_off = int(imp_t.peptidoform_id.max()) + 1
    base_off = int(imp_t.base_peptide_id.max()) + 1
    cid_off = int(imp_t.candidate_id.max()) + 1
    mprec["peptidoform_id"] = mprec["peptidoform_id"].astype(np.int64) + pfid_off
    mprec["base_peptide_id"] = mprec["base_peptide_id"].astype(np.int64) + base_off
    # candidate_id only needs to be unique before make_shift_decoys re-densifies.
    mprec["candidate_id"] = mprec["candidate_id"].astype(np.int64) + cid_off
    mfrag["candidate_id"] = mfrag["candidate_id"].astype(np.int64) + cid_off

    # Per-precursor max-normalize the predicted intensities of the added entries to
    # the imported library's base-peak=1 convention (DIA-NN Relative.Intensity).
    mx = mfrag.groupby("candidate_id")["predicted_intensity"].transform("max")
    mfrag["predicted_intensity"] = np.where(mx > 0, mfrag["predicted_intensity"] / mx, 0.0)

    # 6. Concatenate target halves, align columns, hand off to the decoy builder.
    imp_tfrag = pd.read_parquet(args.imported_fragments)
    imp_tfrag = imp_tfrag[imp_tfrag.candidate_id.isin(set(imp_t.candidate_id))].copy()
    pcols = list(imp_t.columns)
    fcols = list(imp_tfrag.columns)
    merged_prec = pd.concat([imp_t[pcols], mprec[pcols]], ignore_index=True)
    merged_frag = pd.concat([imp_tfrag[fcols], mfrag[fcols]], ignore_index=True)
    # recompute n_fragments from the emitted fragment rows.
    nfrag = merged_frag.groupby("candidate_id").size()
    merged_prec["n_fragments"] = merged_prec["candidate_id"].map(nfrag).fillna(0).astype(np.int32)
    merged_prec.to_parquet(W("aug_merged_prec.parquet"), index=False)
    merged_frag.to_parquet(W("aug_merged_frag.parquet"), index=False)
    print(f"      merged target precursors: {len(merged_prec)} (added {len(mprec)})")

    # 7. Build paired decoys + densify via the existing decoy builder.
    print("[5/6] build paired decoys (%s)" % args.decoy_strategy, flush=True)
    script = "make_shift_decoys.py" if args.decoy_strategy == "shift" else "make_reverse_decoys.py"
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, script),
                        W("aug_merged_prec.parquet"), W("aug_merged_frag.parquet"),
                        args.out_precursors, args.out_fragments], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        sys.stderr.write(r.stderr + "\n")
        raise SystemExit("decoy build failed")

    # 8. Validate the emitted library.
    print("[6/6] validate", flush=True)
    op = pd.read_parquet(args.out_precursors)
    n = len(op)
    cid = op.candidate_id.to_numpy()
    assert np.array_equal(np.sort(cid), np.arange(n)), "candidate_id not contiguous 0..N-1"
    assert op.precursor_mz.is_monotonic_increasing, "precursor_mz not ascending after sort"
    labs = set(op.label)
    assert "target" in labs and "decoy" in labs, f"both labels required, got {labs}"
    print(f"      rows={n} targets={int((op.label=='target').sum())} decoys={int((op.label=='decoy').sum())}")
    # Sequence-level target/decoy overlap is only a defect for sequence-rewrite
    # decoys (reverse/scramble). Shift decoys deliberately keep the target
    # sequence and separate the null in fragment-m/z space, so overlap is expected.
    if args.decoy_strategy == "reverse":
        tset = set(op[op.label == "target"].peptidoform.map(stripped))
        dset = set(op[op.label == "decoy"].peptidoform.map(stripped))
        assert not (tset & dset), f"reverse decoys overlap targets on {len(tset & dset)} sequences"
    print(f"DONE -> {args.out_precursors} / {args.out_fragments}")


if __name__ == "__main__":
    main()
