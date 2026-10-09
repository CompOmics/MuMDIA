"""Give the reverse decoys of an imported DIA-NN library DIA-NN-predicted spectra.

`make_reverse_decoys.py` builds each decoy as its target's reversed sequence and COPIES the
target's fragment intensities ion for ion onto the reversed b/y series. A copied pattern is
not what a predictor would produce for that sequence, so the classifier learns to separate
decoys from real signal by spectral shape alone, and the reported FDR goes optimistic. On the
ProteoBench Astral entrapment module (FDRBench paired entrapment, three HeLa runs) copied
decoys gave a paired entrapment FDP of 1.68% at a reported precursor q of 1%; the same
decoys with DIA-NN-predicted spectra gave 0.85% (valid at every q cut) and more precursors at
a matched FDP. This tool keeps the decoy sequences, their pairing, precursor m/z and iRT,
and replaces only the decoy fragments with DIA-NN's prediction for those sequences. It is
the default step for an imported library since 2026-10-09.

  python predict_decoys.py build <in_prec> <in_frag> <out_prec> <out_frag> --diann <diann>
        [--threads N] [--work DIR] [--keep-unimod 4,35,...] [--diann-arg ARG ...]
      the whole chain: write the decoys as a DIA-NN library TSV, let DIA-NN predict them,
      re-export the prediction as Parquet when DIA-NN writes a .speclib, import it with
      import_diann_lib.py, and merge. <in_*> is make_reverse_decoys.py's output.
  python predict_decoys.py tsv <in_prec> <in_frag> <out.tsv>
  python predict_decoys.py merge <in_prec> <in_frag> <pred_prec> <pred_frag> <out_prec> <out_frag>
      the two halves, for a DIA-NN run made by hand (`diann --lib out.tsv --predictor
      --gen-spec-lib --out-lib X`, then import_diann_lib.py on X's Parquet).

Only reverse-sequence decoys are accepted: a shift decoy (`make_shift_decoys.py`) carries its
target's own sequence, so predicting it would return the target's spectrum. A decoy DIA-NN
does not predict is dropped together with its target, so the populations stay paired. The
merge streams both fragment tables one parquet row group at a time; resident memory is the
two precursor tables plus one row group.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _lib_io import narrow_table, sort_fragments_by_candidate, write_engine_parquet  # noqa: E402
from _unimod import MODIFICATIONS  # noqa: E402

TO_UNIMOD = {name: uid for name, uid, _ in MODIFICATIONS}
MODS = re.compile(r"\[([^\]]+)\]")


def strip_decoy(pform):
    return pform[6:] if pform.startswith("DECOY_") else pform


def to_diann(pform):
    """MuMDIA peptidoform (ProForma names) -> DIA-NN modified sequence (UniMod ids)."""
    return "_" + MODS.sub(lambda m: f"[UniMod:{TO_UNIMOD[m.group(1)]}]", strip_decoy(pform)) + "_"


def load_precursors(path):
    p = pd.read_parquet(path)
    for col in ("candidate_id", "peptidoform_id", "peptidoform", "charge", "label"):
        if col not in p.columns:
            sys.exit(f"{path}: missing column {col!r}")
    return p


def pairs(prec):
    """Decoy rows with their target's peptidoform, paired on (peptidoform_id, charge)."""
    t = prec[prec.label == "target"][["peptidoform_id", "charge", "candidate_id", "peptidoform"]]
    d = prec[prec.label == "decoy"][["peptidoform_id", "charge", "candidate_id", "peptidoform"]]
    return d.merge(t, on=["peptidoform_id", "charge"], how="left", suffixes=("", "_target"))


def check_reverse(prec):
    pr = pairs(prec)
    unpaired = int(pr.candidate_id_target.isna().sum())
    if unpaired:
        sys.exit(f"{unpaired} decoys have no target with the same (peptidoform_id, charge); "
                 "expected make_reverse_decoys.py output")
    same = (pr.peptidoform.map(strip_decoy) == pr.peptidoform_target).sum()
    if same:
        sys.exit(f"{same} decoys carry their target's own sequence (shift decoys?). Predicting "
                 "them would return the target's spectrum; build reverse decoys with "
                 "make_reverse_decoys.py first")
    return pr


def tsv(in_prec, in_frag, out):
    prec = load_precursors(in_prec)
    check_reverse(prec)
    dec = prec[prec.label == "decoy"].drop_duplicates(["peptidoform", "charge"])
    # DIA-NN wants at least one fragment per entry; it replaces them with its prediction.
    want = np.zeros(int(prec.candidate_id.max()) + 1, dtype=bool)
    want[dec.candidate_id.to_numpy()] = True
    first = np.full(want.size, np.nan)
    pf = pq.ParquetFile(in_frag)
    for i in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(i, columns=["candidate_id", "mz"])
        cid = rg.column("candidate_id").to_numpy().astype(np.int64)
        mz = rg.column("mz").to_numpy()
        sel = want[cid] & np.isnan(first[cid])
        first[cid[sel]] = mz[sel]  # duplicates in one group: any of them is a valid placeholder
    with open(out, "w") as f:
        f.write("ModifiedPeptide\tStrippedPeptide\tPrecursorCharge\tPrecursorMz\tFragmentMz\t"
                "RelativeIntensity\tFragmentType\tFragmentSeriesNumber\tFragmentCharge\t"
                "ProteinId\tiRT\n")
        for cid, pform, z, mz in zip(dec.candidate_id, dec.peptidoform, dec.charge, dec.precursor_mz):
            frag_mz = first[cid] if np.isfinite(first[cid]) else mz
            stripped = MODS.sub("", strip_decoy(pform))
            f.write(f"{to_diann(pform)}\t{stripped}\t{z}\t{mz:.6f}\t{frag_mz:.6f}\t1\ty\t1\t1\t"
                    f"DECOYSET\t0\n")
    print(f"tsv: {len(dec)} decoy precursors -> {out}")


def merge(in_prec, in_frag, pred_prec, pred_frag, out_prec, out_frag):
    prec = load_precursors(in_prec)
    pr = check_reverse(prec)
    pred = pd.read_parquet(pred_prec, columns=["candidate_id", "peptidoform", "charge"])
    pred = pred.assign(peptidoform=pred.peptidoform.map(strip_decoy)).drop_duplicates(
        ["peptidoform", "charge"])
    pr = pr.assign(key=pr.peptidoform.map(strip_decoy)).merge(
        pred.rename(columns={"candidate_id": "pred_id", "peptidoform": "key"}),
        on=["key", "charge"], how="left")
    missing = pr.pred_id.isna()
    drop = set(pr.candidate_id[missing]) | set(pr.candidate_id_target[missing].astype(np.int64))
    print(f"merge: {int((~missing).sum())} decoys predicted, {int(missing.sum())} not predicted "
          f"(dropped with their targets)")

    keep = prec[~prec.candidate_id.isin(drop)].sort_values("precursor_mz", kind="mergesort")
    keep = keep.reset_index(drop=True)
    o2n = np.full(int(prec.candidate_id.max()) + 1, -1, dtype=np.int64)
    o2n[keep.candidate_id.to_numpy().astype(np.int64)] = np.arange(len(keep))
    is_target = np.zeros(o2n.size, dtype=bool)
    is_target[keep.candidate_id[keep.label == "target"].to_numpy().astype(np.int64)] = True

    ok = pr[~missing]
    src = ok.pred_id.to_numpy().astype(np.int64)
    dst = o2n[ok.candidate_id.to_numpy().astype(np.int64)]
    # A prediction is normally used by one decoy; two decoys with the same sequence and
    # charge share it, so extra destinations are kept apart and written as copies.
    order = np.argsort(src, kind="mergesort")
    src, dst = src[order], dst[order]
    first_of = np.r_[True, src[1:] != src[:-1]]
    p2n = np.full(int(pred.candidate_id.max()) + 1, -1, dtype=np.int64)
    p2n[src[first_of]] = dst[first_of]
    extra = {}
    for s, d in zip(src[~first_of], dst[~first_of]):
        extra.setdefault(int(s), []).append(int(d))

    schema = pq.read_schema(in_frag)
    counts = np.zeros(len(keep), dtype=np.int64)
    writer = pq.ParquetWriter(out_frag, narrow_table(schema.empty_table()).schema,
                              compression="snappy")
    try:
        pf = pq.ParquetFile(in_frag)
        for i in range(pf.metadata.num_row_groups):
            rg = pf.read_row_group(i)
            cid = rg.column("candidate_id").to_numpy().astype(np.int64)
            sel = is_target[cid]
            if sel.any():
                part = rg.filter(pa.array(sel))
                new = o2n[cid[sel]]
                counts += np.bincount(new, minlength=counts.size)
                part = part.set_column(0, "candidate_id", pa.array(new.astype(np.uint32)))
                writer.write_table(narrow_table(part.select(schema.names).cast(schema)))
        pf = pq.ParquetFile(pred_frag)
        for i in range(pf.metadata.num_row_groups):
            rg = pf.read_row_group(i)
            cid = rg.column("candidate_id").to_numpy().astype(np.int64)
            inside = cid < p2n.size
            new = np.full(cid.size, -1, dtype=np.int64)
            new[inside] = p2n[cid[inside]]
            sel = new >= 0
            if sel.any():
                part = rg.filter(pa.array(sel))
                counts += np.bincount(new[sel], minlength=counts.size)
                part = part.set_column(0, "candidate_id", pa.array(new[sel].astype(np.uint32)))
                writer.write_table(narrow_table(part.select(schema.names).cast(schema)))
            if extra:
                hit = np.isin(cid, np.fromiter(extra, dtype=np.int64))
                for j in np.nonzero(hit)[0]:
                    row = rg.slice(int(j), 1)
                    for d in extra[int(cid[j])]:
                        counts[d] += 1
                        row2 = row.set_column(0, "candidate_id", pa.array([d], pa.uint32()))
                        writer.write_table(narrow_table(row2.select(schema.names).cast(schema)))
    finally:
        writer.close()

    if (counts == 0).any():
        sys.exit(f"{int((counts == 0).sum())} precursors have no fragments after the merge")
    keep["candidate_id"] = np.arange(len(keep), dtype=np.uint32)
    if "n_fragments" in keep.columns:
        keep["n_fragments"] = counts.astype(np.int32)
    write_engine_parquet(keep, out_prec)
    n = sort_fragments_by_candidate(out_frag)
    print(f"merge: {len(keep)} precursors ({int((keep.label == 'decoy').sum())} decoys), "
          f"{n} fragments, dropped {len(drop)} precursors of unpredicted pairs")


def find_parquet(work, stem):
    for cand in (os.path.join(work, f"{stem}.parquet"),
                 *sorted(glob.glob(os.path.join(work, f"{stem}*.parquet")))):
        if os.path.isfile(cand):
            return cand
    return None


def run(cmd):
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build(a):
    work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out_prec)), "decoy_prediction")
    os.makedirs(work, exist_ok=True)
    stem = "decoys"
    lib_tsv = os.path.join(work, f"{stem}.tsv")
    tsv(a.in_prec, a.in_frag, lib_tsv)
    run([a.diann, "--lib", lib_tsv, "--predictor", "--gen-spec-lib", "--out-lib",
         os.path.join(work, stem), "--threads", str(a.threads), *a.diann_arg])
    parquet = find_parquet(work, stem)
    if parquet is None:
        speclib = sorted(glob.glob(os.path.join(work, f"{stem}*.speclib")))
        if not speclib:
            sys.exit(f"DIA-NN wrote neither a Parquet nor a .speclib library in {work}")
        parquet = os.path.join(work, f"{stem}.parquet")
        reexport = [a.diann, "--lib", speclib[0], "--gen-spec-lib", "--out-lib", parquet]
        try:
            run(reexport + ["--threads", str(a.threads)])
        except subprocess.CalledProcessError as e:
            # Seen once on a 5.4M-decoy library: DIA-NN 2.7 died with an access violation
            # right after "Initialising library", and the same command succeeded on retry.
            print(f"re-export failed ({e.returncode}); retrying once with fewer threads", flush=True)
            run(reexport + ["--threads", str(max(1, min(8, a.threads)))])
        if not os.path.isfile(parquet):
            sys.exit(f"the re-export produced no {parquet}; DIA-NN 2.x is needed")
    pp, pfr = os.path.join(work, "pred_precursors.parquet"), os.path.join(work, "pred_fragments.parquet")
    imp = [sys.executable, os.path.join(HERE, "import_diann_lib.py"), parquet, pp, pfr]
    if a.keep_unimod:
        imp += ["--keep-unimod", a.keep_unimod]
    run(imp)
    merge(a.in_prec, a.in_frag, pp, pfr, a.out_prec, a.out_frag)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("tsv", "merge"):
        args = sys.argv[2:]
        want = 3 if sys.argv[1] == "tsv" else 6
        if len(args) != want:
            sys.exit(__doc__)
        (tsv if sys.argv[1] == "tsv" else merge)(*args)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    for name in ("in_prec", "in_frag", "out_prec", "out_frag"):
        b.add_argument(name)
    b.add_argument("--diann", required=True, help="the DIA-NN executable")
    b.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    b.add_argument("--work", help="scratch directory (default: decoy_prediction/ beside out_prec)")
    b.add_argument("--keep-unimod", help="forwarded to import_diann_lib.py")
    b.add_argument("--diann-arg", action="append", default=[],
                   help="extra argument for the DIA-NN prediction run (repeatable)")
    a = ap.parse_args()
    build(a)


if __name__ == "__main__":
    main()
