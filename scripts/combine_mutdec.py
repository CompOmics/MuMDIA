"""Combine a target library with a DIA-NN-predicted MUTATION-decoy library into a
single MuMDIA library (contiguous candidate_id sorted by precursor m/z). Decoys are
labeled 'decoy' and DECOY_-prefixed. Unlike shift decoys (which reuse target
intensities on shifted m/z), these decoys have DIA-NN-predicted fragments for a
mutated sequence, so they experience the same chimeric interference as targets.

Usage: python combine_mutdec.py <tgt_prec> <tgt_frag> <dec_prec> <dec_frag> <out_prec> <out_frag>
"""
import sys
import numpy as np
import pandas as pd


def main():
    tp, tf, dp, df_, op, of = sys.argv[1:7]
    tprec = pd.read_parquet(tp); tfrag = pd.read_parquet(tf)
    dprec = pd.read_parquet(dp); dfrag = pd.read_parquet(df_)
    # Match the decoy (length, charge) distribution to the targets so no feature
    # (e.g. peptide_length) trivially separates target from decoy -> valid null.
    import re
    sstrip = lambda s: re.sub(r"\[[^\]]*\]", "", str(s).replace("DECOY_", ""))
    plen = lambda s: len(re.sub(r"\[[^\]]*\]", "", str(s)))
    tprec = tprec.copy(); dprec = dprec.copy().reset_index(drop=True)
    # Drop decoys whose stripped sequence also occurs among targets (shared peptides
    # would be present in the sample -> false decoys).
    tset = set(tprec.peptidoform.map(sstrip))
    before = len(dprec)
    dprec = dprec[~dprec.peptidoform.map(sstrip).isin(tset)].reset_index(drop=True)
    if before != len(dprec):
        print(f"dropped {before - len(dprec)} decoys sharing a target stripped sequence")
    tprec["_len"] = tprec.peptidoform.map(plen); dprec["_len"] = dprec.peptidoform.map(plen)
    rng = np.random.RandomState(0)
    tc = tprec.groupby(["_len", "charge"]).size()
    keep = []
    for (L, z), n in tc.items():
        pool = dprec.index[(dprec._len == L) & (dprec.charge == z)].to_numpy()
        if pool.size:
            keep.append(rng.choice(pool, min(int(n), pool.size), replace=False))
    dprec = dprec.loc[np.concatenate(keep)].drop(columns="_len").reset_index(drop=True)
    tprec = tprec.drop(columns="_len")
    dfrag = dfrag[dfrag["candidate_id"].isin(set(dprec["candidate_id"]))].reset_index(drop=True)
    tprec["label"] = "target"
    dprec["label"] = "decoy"
    dprec["peptidoform"] = "DECOY_" + dprec["peptidoform"].astype(str)
    dprec["protein"] = "DECOY_" + dprec["protein"].astype(str)

    # offset decoy candidate_ids so target/decoy ids are disjoint before merge
    off = int(tprec["candidate_id"].max()) + 1
    dprec = dprec.copy(); dfrag = dfrag.copy()
    dprec["candidate_id"] = dprec["candidate_id"].astype(np.int64) + off
    dfrag["candidate_id"] = dfrag["candidate_id"].astype(np.int64) + off

    allp = pd.concat([tprec, dprec], ignore_index=True)
    allp = allp.sort_values("precursor_mz", kind="mergesort").reset_index(drop=True)
    old2new = {old: new for new, old in enumerate(allp["candidate_id"].tolist())}
    allp["candidate_id"] = np.arange(len(allp), dtype=np.uint32)

    allf = pd.concat([tfrag, dfrag], ignore_index=True)
    allf["candidate_id"] = allf["candidate_id"].map(old2new).astype(np.uint32)
    allf = allf.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)

    allp["candidate_id"] = allp["candidate_id"].astype(np.uint32)
    allp["peptidoform_id"] = allp["peptidoform_id"].astype(np.uint32)
    allp["base_peptide_id"] = allp["base_peptide_id"].astype(np.uint32)
    allp["charge"] = allp["charge"].astype(np.int32)
    allp["predicted_irt"] = allp["predicted_irt"].astype(np.float32)
    allp["n_fragments"] = allp["n_fragments"].astype(np.int32)
    allf["predicted_intensity"] = allf["predicted_intensity"].astype(np.float32)
    allf["ordinal"] = allf["ordinal"].astype(np.int32)
    allf["frag_charge"] = allf["frag_charge"].astype(np.int32)

    allp.to_parquet(op, index=False); allf.to_parquet(of, index=False)
    nt = int((allp.label == "target").sum()); nd = int((allp.label == "decoy").sum())
    print(f"targets={nt} decoys={nd} total_prec={len(allp)} total_frag={len(allf)}")


if __name__ == "__main__":
    main()
