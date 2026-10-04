"""ms2_repredict.py <weights.pth> <out_fragments.parquet>: the HYE fragment library with only predicted_intensity replaced
by the fine-tuned peptdeep MS2 model (ms2_finetune.py). Every candidate, target and decoy, keeps its 12 fragments (m/z,
name, cardinality, row order); the precursor table is unchanged. Same model call as scripts/peptdeep_worker.py (generic,
timsTOF, NCE 35, b/y at charge 1 and 2), per-precursor max-normalised as the engine stores it."""
import sys, time, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, "/home/robbe/MuMDIA/scripts")
from peptdeep_worker import parse_peptidoform


def main():
    WTS, OUT = sys.argv[1:3]
    import torch; torch.set_num_threads(4)
    from peptdeep.pretrained_models import ModelManager
    W = "/public/local/ProteoBench/HYE_diaPASEF_mumdia"
    LF = f"{W}/l1d/run/fragment_library_fragments.parquet"
    mm = ModelManager(device="cpu"); mm.load_installed_models("generic"); mm.ms2_model.load(WTS)
    fr = pq.read_table(LF, columns=["candidate_id", "ion_type", "ordinal", "frag_charge"])
    cid = fr["candidate_id"].to_numpy(); assert np.all(np.diff(cid.astype(np.int64)) >= 0), "fragments not sorted by candidate"
    ion = (fr["ion_type"].to_numpy(zero_copy_only=False) == "y").astype(np.int8); ordn = fr["ordinal"].to_numpy(); z = fr["frag_charge"].to_numpy()
    del fr
    off = np.searchsorted(cid, np.arange(int(cid.max()) + 2))
    pre = pq.read_table(f"{W}/l1d/run/r0/fragment_library_precursors_refit.parquet", columns=["candidate_id", "peptidoform", "charge"]).to_pandas()
    pre = pre.sort_values("candidate_id").reset_index(drop=True)
    new = np.full(len(cid), np.nan, np.float32)
    COLS = ["b_z1", "b_z2", "y_z1", "y_z2"]; t0 = time.time(); CH = 200_000
    for s in range(0, len(pre), CH):
        b = pre.iloc[s:s + CH].copy(); p = b.peptidoform.map(parse_peptidoform); b = b[p.notna()]; p = p[p.notna()]
        df = pd.DataFrame({"mumdia_id": b.candidate_id.to_numpy(np.uint32), "sequence": p.str[0].values, "mods": p.str[1].values,
                           "mod_sites": p.str[2].values, "charge": b.charge.to_numpy(np.int32), "nce": np.float32(35.0), "instrument": "timsTOF"})
        out = mm.predict_all(df, predict_items=["ms2"], frag_types=COLS, multiprocessing=True, process_num=48)
        P, I = out["precursor_df"], out["fragment_intensity_df"][COLS].to_numpy(np.float32)
        for c, st, naa in zip(P.mumdia_id.to_numpy(), P.frag_start_idx.to_numpy(), P.nAA.to_numpy()):
            a, e = off[c], off[c + 1]
            if a == e: continue
            o, y, zz = ordn[a:e], ion[a:e], z[a:e]
            row = st + np.where(y == 1, naa - 1 - o, o - 1); col = y * 2 + (zz - 1)
            v = I[row, col]; m = v.max()
            new[a:e] = v / m if m > 0 else v
        print(f"{min(s + CH, len(pre))}/{len(pre)} precursors, {time.time() - t0:.0f} s", flush=True)
    miss = np.isnan(new); print("fragments without a new prediction (kept as before):", int(miss.sum()))
    t = pq.read_table(LF)
    old = t["predicted_intensity"].to_numpy(); new[miss] = old[miss]
    t = t.set_column(t.schema.get_field_index("predicted_intensity"), t.schema.field("predicted_intensity"),
                     pa.array(new.astype(old.dtype), t.schema.field("predicted_intensity").type))
    pq.write_table(t, OUT, compression="snappy")
    print("wrote", OUT, "| median |new - old|", float(np.median(np.abs(new - old))))


if __name__ == "__main__":
    main()
