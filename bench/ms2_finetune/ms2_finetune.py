"""ms2_finetune.py: does fine-tuning peptdeep's MS2 model on HYE's own confident IDs bring the library's fragment pattern
closer to the observed one? (docs/TIMS_ROADMAP_bis.md section 13; prototype, outside the engine.)

Training targets: precursors accepted at pooled q <= 0.001 in any of the six eng_repick runs; observed pattern = mean over
those runs of the max-normalised pass-1 fixed-window fragment areas (q_rp_crwbg_w25_s0/r*/fragment_quant.parquet). Only
the 12 library fragments are observed; every other b/y position is set to 0. Split by stripped sequence, 80/20.
Evaluation on the held-out 20%, as frag_headroom.py: cosine of square-rooted vectors over the fragments positive in runs 0
and 1 (accepted in both), predicted against run 0, and run 1 against run 0 (the ceiling). Model: generic, timsTOF, NCE 35,
as the library build. Usage: ms2_finetune.py <out_dir> [epochs] [lr]"""
import os, sys, zlib, numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "/home/robbe/MuMDIA/scripts")
from peptdeep_worker import parse_peptidoform
OUT = sys.argv[1]; EPOCH = int(sys.argv[2]) if len(sys.argv) > 2 else 10; LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
os.makedirs(OUT, exist_ok=True)
import torch; torch.set_num_threads(48)
from peptdeep.pretrained_models import ModelManager
W = "/public/local/ProteoBench/HYE_diaPASEF_mumdia"; Q = f"{W}/q_rp_crwbg_w25_s0"
NCE, INSTR = 35.0, "timsTOF"

conf = {}
for i in range(6):
    s = pq.read_table(f"{W}/eng_repick/r{i}/scored.parquet", columns=["candidate_id", "label", "q_value"]).to_pandas()
    conf[i] = set(s.candidate_id[(s.label == "target") & (s.q_value <= 0.001)])
acc01 = [set(pq.read_table(f"{W}/eng_repick/r{i}/scored.parquet", columns=["candidate_id", "label", "q_value"]).to_pandas()
             .query("label == 'target' and q_value <= 0.01").candidate_id) for i in (0, 1)]
ids = sorted(set().union(*conf.values()))
lib = pq.read_table(f"{W}/l1d/run/fragment_library_fragments.parquet", columns=["candidate_id", "name", "ion_type", "ordinal", "frag_charge"],
                    filters=[("candidate_id", "in", ids)]).to_pandas()
pre = pq.read_table(f"{W}/l1d/run/r0/fragment_library_precursors_refit.parquet", columns=["candidate_id", "peptidoform", "charge"],
                    filters=[("candidate_id", "in", ids)]).to_pandas()
obs = []
for i in range(6):
    f = pq.read_table(f"{Q}/r{i}/fragment_quant.parquet", columns=["candidate_id", "fragment_name", "quantity"],
                      filters=[("candidate_id", "in", ids)]).to_pandas()
    f = f[f.candidate_id.isin(conf[i])]
    f["norm"] = f.quantity.clip(lower=0) / f.groupby("candidate_id").quantity.transform("max").replace(0, np.nan)
    obs.append(f.assign(run=i))
obs = pd.concat(obs)
runs01 = obs[obs.run.isin([0, 1])].pivot_table(index=["candidate_id", "fragment_name"], columns="run", values="quantity")
cons = obs.groupby(["candidate_id", "fragment_name"]).norm.mean().rename("obs").reset_index()
lib = lib.merge(cons.rename(columns={"fragment_name": "name"}), on=["candidate_id", "name"], how="left").fillna({"obs": 0.0})
print(f"training precursors (q <= 0.001 in any run): {len(ids)}, fragment rows {len(lib)}", flush=True)

pre["p"] = pre.peptidoform.map(parse_peptidoform); pre = pre[pre.p.notna()].copy()
pre["sequence"] = pre.p.str[0]; pre["mods"] = pre.p.str[1]; pre["mod_sites"] = pre.p.str[2]
pre["nAA"] = pre.sequence.str.len(); pre["nce"] = NCE; pre["instrument"] = INSTR
pre["test"] = pre.sequence.map(lambda s: zlib.crc32(s.encode()) % 5 == 0)

mm = ModelManager(device="cpu"); mm.load_installed_models("generic")
cols = list(mm.ms2_model.charged_frag_types)
COL = {("b", 1): "b_z1", ("b", 2): "b_z2", ("y", 1): "y_z1", ("y", 2): "y_z2"}

def frame(df):
    """peptdeep precursor frame with frag_start/stop and the observed intensity matrix (library fragments, 0 elsewhere)."""
    df = df.sort_values("candidate_id").reset_index(drop=True)
    n = (df.nAA - 1).to_numpy(); stop = np.cumsum(n); start = stop - n
    df["frag_start_idx"], df["frag_stop_idx"] = start, stop
    M = pd.DataFrame(0.0, index=np.arange(int(stop[-1])), columns=cols)
    pos = dict(zip(df.candidate_id, zip(start, df.nAA)))
    L = lib[lib.candidate_id.isin(pos)]
    for c, it, o, z, v in zip(L.candidate_id, L.ion_type, L.ordinal, L.frag_charge, L.obs):
        st, naa = pos[c]; row = st + (o - 1 if it == "b" else naa - 1 - o)
        M.iat[int(row), cols.index(COL[(it, int(z))])] = v
    return df, M

def predict_lib(df):
    """predicted intensity of each library fragment of df's precursors -> {(cid, name): value}"""
    out = mm.predict_all(df[["sequence", "mods", "mod_sites", "charge", "nce", "instrument"]].assign(mumdia_id=df.candidate_id.values),
                         predict_items=["ms2"], frag_types=["b_z1", "b_z2", "y_z1", "y_z2"], multiprocessing=False)
    p, I = out["precursor_df"], out["fragment_intensity_df"]
    pos = dict(zip(p.mumdia_id, zip(p.frag_start_idx, p.nAA)))
    L = lib[lib.candidate_id.isin(pos)]; res = {}
    for c, nm, it, o, z in zip(L.candidate_id, L.name, L.ion_type, L.ordinal, L.frag_charge):
        st, naa = pos[c]; res[(c, nm)] = float(I[COL[(it, int(z))]].iat[int(st + (o - 1 if it == "b" else naa - 1 - o))])
    return res

def evaluate(tag, df):
    ev = df[df.candidate_id.isin(acc01[0] & acc01[1])]
    pr = predict_lib(ev)
    r = runs01.reset_index(); r = r[r.candidate_id.isin(set(ev.candidate_id))]
    r = r[(r[0] > 0) & (r[1] > 0)]; r["pred"] = [pr.get((c, n), np.nan) for c, n in zip(r.candidate_id, r.fragment_name)]
    cs = []
    for c, g in r.groupby("candidate_id"):
        if len(g) < 4: continue
        a0, a1, p = np.sqrt(g[0].to_numpy()), np.sqrt(g[1].to_numpy()), np.sqrt(np.clip(g.pred.to_numpy(), 0, None))
        cs.append((np.dot(p, a0) / (np.linalg.norm(p) * np.linalg.norm(a0) + 1e-30), np.dot(a1, a0) / (np.linalg.norm(a1) * np.linalg.norm(a0) + 1e-30)))
    cs = np.array(cs)
    print(f"{tag}: held-out precursors {len(cs)}; median cosine prediction vs run 0 {np.median(cs[:, 0]):.4f}, run 1 vs run 0 {np.median(cs[:, 1]):.4f}; "
          f"run-to-run better in {np.mean(cs[:, 1] > cs[:, 0]):.3f}", flush=True)

test = pre[pre.test]; train = pre[~pre.test]
evaluate("base model", test)
tr_df, tr_I = frame(train)
print(f"training on {len(tr_df)} precursors, {EPOCH} epochs, lr {LR}", flush=True)
mm.ms2_model.train(tr_df, tr_I, epoch=EPOCH, lr=LR, batch_size=512, verbose=True)
mm.ms2_model.save(f"{OUT}/ms2_finetuned.pth")
evaluate(f"fine-tuned ({EPOCH} epochs, lr {LR})", test)
