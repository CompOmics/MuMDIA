"""Build a fragment-shift decoy library from the target half of an existing
library (clean-room realization of the DIA-NN terminal-shift decoy idea,
docs/13_sidecars.md).

Each decoy copies a target's predicted intensities and iRT and keeps the target
precursor m/z (so the decoy co-isolates in the same window / RT and experiences
the same interference, i.e. a valid null for the false-target rate), but shifts
b-ion m/z by -DELTA and y-ion m/z by +DELTA (net precursor shift zero). DELTA is
one CH2 (14.01565 Da), a documented, self-chosen shift (no borrowed map).

Usage: python make_shift_decoys.py <in_precursors> <in_fragments> <out_precursors> <out_fragments>
"""
import sys
import numpy as np
import pandas as pd

DELTA = 14.015650  # CH2, neutral-mass shift


def main():
    inp, inf, outp, outf = sys.argv[1:5]
    prec = pd.read_parquet(inp)
    frag = pd.read_parquet(inf)

    # Keep only targets as the basis.
    tprec = prec[prec.label == "target"].copy()
    tids = set(tprec.candidate_id)
    tfrag = frag[frag.candidate_id.isin(tids)].copy()

    # Decoy precursors: same precursor m/z + iRT, new id space offset.
    off = int(tprec.candidate_id.max()) + 1
    dprec = tprec.copy()
    dprec["candidate_id"] = dprec["candidate_id"] + off
    dprec["label"] = "decoy"
    dprec["protein"] = "DECOY_" + dprec["protein"].astype(str)
    dprec["peptidoform"] = "DECOY_" + dprec["peptidoform"].astype(str)

    # Decoy fragments: copy intensities; shift b down / y up by DELTA/charge.
    dfrag = tfrag.copy()
    dfrag["candidate_id"] = dfrag["candidate_id"] + off
    z = dfrag["frag_charge"].to_numpy()
    shift = DELTA / np.where(z < 1, 1, z)
    is_b = dfrag["ion_type"].to_numpy() == "b"
    dfrag["mz"] = dfrag["mz"].to_numpy() + np.where(is_b, -shift, shift)

    # Concatenate, re-sort precursors by precursor m/z, reassign contiguous ids.
    allp = pd.concat([tprec, dprec], ignore_index=True)
    allp = allp.sort_values("precursor_mz", kind="mergesort").reset_index(drop=True)
    old2new = {old: new for new, old in enumerate(allp["candidate_id"].tolist())}
    allp["candidate_id"] = np.arange(len(allp), dtype=np.uint32)

    allf = pd.concat([tfrag, dfrag], ignore_index=True)
    allf["candidate_id"] = allf["candidate_id"].map(old2new).astype(np.uint32)
    # fragments must be grouped by candidate_id order for the index build.
    allf = allf.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)

    # enforce dtypes matching the Rust schema
    allp["candidate_id"] = allp["candidate_id"].astype(np.uint32)
    allp["peptidoform_id"] = allp["peptidoform_id"].astype(np.uint32)
    allp["base_peptide_id"] = allp["base_peptide_id"].astype(np.uint32)
    allp["charge"] = allp["charge"].astype(np.int32)
    allp["predicted_irt"] = allp["predicted_irt"].astype(np.float32)
    allp["n_fragments"] = allp["n_fragments"].astype(np.int32)
    allf["predicted_intensity"] = allf["predicted_intensity"].astype(np.float32)
    allf["ordinal"] = allf["ordinal"].astype(np.int32)
    allf["frag_charge"] = allf["frag_charge"].astype(np.int32)

    allp.to_parquet(outp, index=False)
    allf.to_parquet(outf, index=False)
    print(f"targets={len(tprec)} decoys={len(dprec)} total_prec={len(allp)} total_frag={len(allf)}")


if __name__ == "__main__":
    main()
