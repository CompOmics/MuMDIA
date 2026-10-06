"""Second-pass search (`mbr.strategy = second_pass`, run-experiment): docs/TIMS_QUANT_ROADMAP.md 4m, 4n.

The pass-1 identifications become an empirical library that every run is searched with again; the second pass is
the reported result. The engine form of arm D of section 4n, ported from the prototype `quant_diag/sp_lib.py` and
`sp_eval.py`.

  mbr_second_pass.py build  <spec.json>
  mbr_second_pass.py report <spec.json>

build spec: scored (the pooled pass-1 scored table), lib_precursors / lib_fragments (the library pass 1 searched),
runs (one {repick, seed, windows[, fragquant]} per run: its re-picked psms_extracted, seed PSMs, pass-1 run windows
and, for empirical intensities, pass-1 fragment areas), out, lib_q, min_rt_hw, min_im_hw, intensity
(predicted | empirical), decoys (reverse_nc | native).

build writes to <out>: lib_precursors.parquet, lib_fragments.parquet (contiguous candidate_id by precursor_mz),
id_map.parquet (old -> new candidate_id), r<i>/run_windows.parquet (ID windows), r<i>/run_windows_quant.parquet (same
centres, pass-1 half-widths, for the quant traces), r<i>/seed_psms.parquet (remapped) with the run's
.masscal.json beside it, summary.json.

- Library: targets with experiment-wide precursor_q <= lib_q (minimum over the precursor's rows). Support runs: runs
  with run_psm_q <= 0.01; a precursor with none takes its best-scoring row's run.
- Expected RT per run: mbr_worker's binned-median maps through run 0, median over the support runs. Expected 1/K0: the
  median of the support runs' re-picked apex_im, each moved onto run 0 by its median offset, then onto the run.
- Half-widths per run: p99 of the held-out residual (each confident row predicted from the other support runs only),
  floored by min_rt_hw / min_im_hw.
- Intensities: predicted (the library's, max-normalised), or empirical (pass-1 areas over the support runs, median over
  >= 3 runs, mean over 1-2, averaged with the prediction under 2 runs).
- Decoys: one per target. reverse_nc: interior reversed, both termini kept, re-scrambled on a collision with any
  search-library target or another new decoy (I = L), dropped with its target if it still collides. native: the
  target's paired decoy of the search library. Either way the decoy carries the target's ion names, intensities, RT
  and 1/K0 windows, and takes its ids and seed row from the native paired decoy. The construction invariant (every
  library-side value equal within a pair) is asserted.

report spec: scored2 (the pooled second-pass scored table, new ids), scored1 (pass 1), id_map, report_q, out.
report writes <out>: the second-pass table with every target whose pass-1 precursor_q exceeds report_q set to q 1.0
in every q column (DIA-NN's Lib.Q.Value filter), is_transferred (accepted at pooled q_value <= 0.01 in pass 2 but not
in pass 1 for that run) and transfer_q (NaN). Per-run counts go to stdout.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib_io import write_engine_parquet, write_engine_table  # noqa: E402
from make_reverse_decoys import (MAX_TRIES, _cumulative_masses, _fragment_mz, parse,  # noqa: E402
                                 reverse_keep_cterm, scramble, splitmix, stable_seed, stripped,
                                 to_pform, valid)
from mbr_worker import binned_map  # noqa: E402

Q_RUN, SHRINK_RUNS, P_HW = 0.01, 2, 99
Q_COLS = ("q_value", "run_psm_q", "experiment_psm_q", "precursor_q", "peptide_q_value", "pg_q_value")


def build(spec):
    out = spec["out"]
    runs = spec["runs"]
    N = len(runs)
    lib_q = float(spec["lib_q"])
    min_rt_hw, min_im_hw = float(spec.get("min_rt_hw", 0.0)), float(spec.get("min_im_hw", 0.0))
    intensity, decoys = spec.get("intensity", "predicted"), spec.get("decoys", "reverse_nc")
    os.makedirs(out, exist_ok=True)
    log = {}

    # ---- inclusion and support ----
    s = pq.read_table(spec["scored"], filters=[("label", "=", "target")],
                      columns=["candidate_id", "source", "apex_rt", "score", "run_psm_q", "precursor_q"]).to_pandas()
    s["candidate_id"] = s.candidate_id.astype(np.int64)
    pqm = s.groupby("candidate_id").precursor_q.min()
    lib_t = np.sort(pqm.index[pqm <= lib_q].to_numpy())
    s = s[s.candidate_id.isin(lib_t)].sort_values(["candidate_id", "source", "score"], ascending=[True, True, False])
    s = s.drop_duplicates(["candidate_id", "source"])
    sup = s[s.run_psm_q <= Q_RUN]
    best = s[~s.candidate_id.isin(sup.candidate_id)].sort_values("score", ascending=False).drop_duplicates("candidate_id")
    sup = pd.concat([sup, best])[["candidate_id", "source", "apex_rt"]]
    nsup = sup.groupby("candidate_id").size()
    log.update(lib_q=lib_q, targets=int(len(lib_t)), no_confident_run=int(len(best)),
               support_runs={int(k): int(v) for k, v in nsup.value_counts().sort_index().items()})
    print(f"library targets {len(lib_t):,} (precursor_q <= {lib_q}); without a confident run {len(best)}; "
          f"support runs {log['support_runs']}", flush=True)

    im = []
    for i, r in enumerate(runs):
        e = pq.read_table(r["repick"], columns=["candidate_id", "peak_rank", "apex_im"]).to_pandas()
        e = e[(e.peak_rank == 0) & e.candidate_id.isin(lib_t)].drop_duplicates("candidate_id")
        im.append(pd.DataFrame({"candidate_id": e.candidate_id.astype(np.int64), "source": i, "apex_im": e.apex_im}))
    sup = sup.merge(pd.concat(im), on=["candidate_id", "source"], how="left")

    # ---- RT and 1/K0 onto run 0 and back ----
    anc = {i: g.set_index("candidate_id") for i, g in sup.groupby("source")}
    to_ref, from_ref, off = {}, {}, {}
    for i in range(N):
        sh = anc[i].index.intersection(anc[0].index)
        x, y = anc[i].apex_rt.reindex(sh).to_numpy(), anc[0].apex_rt.reindex(sh).to_numpy()
        to_ref[i] = binned_map(x, y) if i else (lambda q: q)
        from_ref[i] = binned_map(y, x) if i else (lambda q: q)
        off[i] = float(np.nanmedian(anc[i].apex_im.reindex(sh).to_numpy() - anc[0].apex_im.reindex(sh).to_numpy())) if i else 0.0
    sup["rt_ref"] = np.nan
    for i in range(N):
        m = sup.source == i
        sup.loc[m, "rt_ref"] = to_ref[i](sup.apex_rt[m].to_numpy())
        sup.loc[m, "im_ref"] = sup.apex_im[m].to_numpy() - off[i]
    g = sup.groupby("candidate_id")
    ref = pd.DataFrame({"rt_ref": g.rt_ref.median(), "im_ref": g.im_ref.median(), "n": g.size()}).reindex(lib_t)

    hw_rt, hw_im = {}, {}
    for i in range(N):
        own = sup[sup.source == i].set_index("candidate_id")
        oth = sup[(sup.source != i) & sup.candidate_id.isin(own.index)].groupby("candidate_id")
        prt, pim = oth.rt_ref.median(), oth.im_ref.median()
        r_rt = np.abs(own.apex_rt.reindex(prt.index).to_numpy() - from_ref[i](prt.to_numpy()))
        r_im = np.abs(own.apex_im.reindex(pim.index).to_numpy() - (pim.to_numpy() + off[i]))
        if not np.isfinite(r_rt).any():  # one run: no held-out residual, the floors are the half-widths
            print(f"  run {i}: no held-out rows; half-widths from the floors {min_rt_hw} s / {min_im_hw}", flush=True)
            hw_rt[i], hw_im[i] = min_rt_hw, min_im_hw
            continue
        hw_rt[i] = max(float(np.nanpercentile(r_rt, P_HW)), min_rt_hw)
        hw_im[i] = max(float(np.nanpercentile(r_im, P_HW)), min_im_hw)
        print(f"  run {i}: held-out |dRT| median {np.nanmedian(r_rt):.2f} p95 {np.nanpercentile(r_rt, 95):.2f} "
              f"p99 {np.nanpercentile(r_rt, 99):.2f} s; |d1/K0| median {np.nanmedian(r_im):.4f} p99 "
              f"{np.nanpercentile(r_im, 99):.4f} ({len(prt):,} rows); 1/K0 offset {off[i]:+.4f}; "
              f"half-widths {hw_rt[i]:.2f} s / {hw_im[i]:.4f}", flush=True)
    log.update(hw_rt=hw_rt, hw_im=hw_im, im_offset=off)

    # ---- decoy pairing ----
    lp = pq.read_table(spec["lib_precursors"]).to_pandas()
    lp["candidate_id"] = lp.candidate_id.astype(np.int64)
    lpi = lp.set_index("candidate_id", drop=False)
    tp = lpi.loc[lib_t]
    dec = lp[(lp.label == "decoy") & lp.base_peptide_id.isin(set(tp.base_peptide_id))]
    by_rev = {(b, z, p): c for b, z, p, c in zip(dec.base_peptide_id, dec.charge, dec.peptidoform.str.replace("DECOY_", ""), dec.candidate_id)}
    by_mz = {}
    for b, z, mz, c in zip(dec.base_peptide_id, dec.charge, dec.precursor_mz.round(6), dec.candidate_id):
        by_mz.setdefault((b, z, mz), []).append(c)
    pair, how = {}, {"reversed": 0, "mass": 0, "none": 0}
    for c, b, z, p, mz in zip(tp.candidate_id, tp.base_peptide_id, tp.charge, tp.peptidoform, tp.precursor_mz.round(6)):
        d = by_rev.get((b, z, to_pform(reverse_keep_cterm(parse(p)))))
        if d is not None:
            pair[c] = d
            how["reversed"] += 1
        elif len(by_mz.get((b, z, mz), [])) == 1:  # scrambled decoy: the one decoy of equal mass in the group
            pair[c] = by_mz[(b, z, mz)][0]
            how["mass"] += 1
        else:
            how["none"] += 1
    assert len(set(pair.values())) == len(pair), "two targets share a decoy"
    log["pairing"] = how
    print(f"decoy pairing: {how}", flush=True)
    lib_t = np.array([c for c in lib_t if c in pair])  # an unpaired target leaves with its (missing) decoy

    new_toks = {}
    if decoys == "reverse_nc":
        norm = lambda t: stripped(t).replace("I", "L")  # noqa: E731
        tgt_seq = set(lp.peptidoform[lp.label == "target"].str.replace(r"\[[^\]]*\]", "", regex=True).str.replace("I", "L"))
        owner, nd_ = {}, {"reversed": 0, "scrambled": 0, "dropped": 0}
        for pf, cid in zip(lpi.peptidoform.reindex(lib_t), lib_t):
            t = parse(pf)
            src_ = norm(t)
            cand = t[:1] + t[1:-1][::-1] + t[-1:]
            bad = lambda x: norm(x) in tgt_seq or owner.get(norm(x), src_) != src_  # noqa: E731
            gen, k = splitmix(stable_seed(src_) ^ 0xD1CE), 0
            while bad(cand) and k < MAX_TRIES:
                cand = t[:1] + scramble(t[1:], gen)
                k += 1
            if bad(cand):
                nd_["dropped"] += 1
                continue
            nd_["scrambled" if k else "reversed"] += 1
            new_toks[cid] = cand
            owner.setdefault(norm(cand), src_)
        lib_t = np.array([c for c in lib_t if c in new_toks])
        assert not ({norm(t) for t in new_toks.values()} & tgt_seq), "a new decoy equals a target"
        log["new_decoys"] = nd_
        print(f"new decoys (both termini kept): {nd_}", flush=True)
    ref = ref.reindex(lib_t)

    # ---- intensities ----
    lf = pq.read_table(spec["lib_fragments"], filters=[("candidate_id", "in", pa.array(lib_t, pa.uint32()))]).to_pandas()
    lf["candidate_id"] = lf.candidate_id.astype(np.int64)
    lf = lf.sort_values("candidate_id", kind="mergesort")
    assert lf.groupby("candidate_id").size().reindex(lib_t).notna().all(), "library target without fragments"
    lf["pred"] = lf.predicted_intensity / lf.groupby("candidate_id").predicted_intensity.transform("max").replace(0, 1)
    if intensity == "predicted":
        lf["inten"] = lf.pred
    else:
        emp = []
        supc = sup[sup.candidate_id.isin(lib_t)]
        for i, r in enumerate(runs):
            f = pq.read_table(r["fragquant"], columns=["candidate_id", "fragment_name", "quantity"]).to_pandas()
            f["candidate_id"] = f.candidate_id.astype(np.int64)
            f = f[f.candidate_id.isin(set(supc.candidate_id[supc.source == i]))]
            x = lf[["candidate_id", "name"]][lf.candidate_id.isin(set(f.candidate_id))].merge(
                f.rename(columns={"fragment_name": "name"}), on=["candidate_id", "name"], how="left").fillna({"quantity": 0.0})
            mx = x.groupby("candidate_id").quantity.transform("max")
            x["v"] = np.where(mx > 0, x.quantity / mx.where(mx > 0, 1), 0.0)
            emp.append(x.assign(source=i)[["candidate_id", "name", "v", "source"]])
        emp = pd.concat(emp)
        ne = emp.groupby("candidate_id").source.nunique()
        ge = emp.groupby(["candidate_id", "name"]).v
        agg = pd.DataFrame({"med": ge.median(), "mean": ge.mean()}).reset_index()
        agg["n"] = agg.candidate_id.map(ne)
        agg["emp"] = np.where(agg.n >= 3, agg.med, agg["mean"])
        lf = lf.merge(agg[["candidate_id", "name", "emp", "n"]], on=["candidate_id", "name"], how="left")
        lf["n"] = lf.n.fillna(0)
        lf["inten"] = np.where(lf.n == 0, lf.pred, np.where(lf.n < SHRINK_RUNS, 0.5 * (lf.emp + lf.pred), lf.emp))
        mx = lf.groupby("candidate_id").inten.transform("max")
        lf["inten"] = np.where(mx > 0, lf.inten / mx.where(mx > 0, 1), 0.0)
        log["fragment_runs"] = {int(k): int(v) for k, v in ne.reindex(lib_t).fillna(0).astype(int).value_counts().sort_index().items()}
    log["intensity"] = intensity
    log["all_zero_fragments"] = int((lf.inten == 0).sum())
    print(f"intensities {intensity}; fragments at intensity 0: {log['all_zero_fragments']:,} of {len(lf):,}", flush=True)

    # ---- decoy fragments: target ion names, m/z on the decoy sequence ----
    dids = np.array([pair[c] for c in lib_t])
    dtoks = [new_toks[c] for c in lib_t] if new_toks else [parse(p) for p in lpi.peptidoform.reindex(dids)]
    assert all(valid(t) for t in dtoks)
    row = {c: k for k, c in enumerate(lib_t)}
    ttoks = [parse(p) for p in lpi.peptidoform.reindex(lib_t)]
    cum_t, len_t = _cumulative_masses(ttoks, max(map(len, ttoks)))
    cum_d, len_d = _cumulative_masses(dtoks, max(map(len, dtoks)))
    rows = lf.candidate_id.map(row).to_numpy()
    isb, ordn, z = (lf.ion_type == "b").to_numpy(), lf.ordinal.to_numpy().astype(np.int64), lf.frag_charge.to_numpy().astype(np.float64)
    tmz = _fragment_mz(cum_t, len_t, rows, isb, ordn, z)
    err = 1e6 * np.abs(tmz - lf.mz.to_numpy()) / lf.mz.to_numpy()
    print(f"calculator vs library target m/z: median {np.median(err):.3f} p99 {np.percentile(err, 99):.3f} max {err.max():.3f} ppm", flush=True)
    assert np.percentile(err, 99) < 5.0, "m/z calculator inconsistent with the library"
    dmz = _fragment_mz(cum_d, len_d, rows, isb, ordn, z)
    if not new_toks:  # cross-check on the native decoy's own fragments where it has the same ion
        nd = pq.read_table(spec["lib_fragments"], filters=[("candidate_id", "in", pa.array(dids, pa.uint32()))],
                           columns=["candidate_id", "name", "mz"]).to_pandas()
        chk = pd.DataFrame({"candidate_id": np.array([pair[c] for c in lf.candidate_id]), "name": lf.name.to_numpy(),
                            "dmz": dmz}).merge(nd, on=["candidate_id", "name"])
        derr = 1e6 * np.abs(chk.dmz - chk.mz) / chk.mz
        print(f"decoy m/z vs native decoy fragments ({len(chk):,} shared ions): p99 {np.percentile(derr, 99):.3f} "
              f"max {derr.max():.3f} ppm", flush=True)
        assert np.percentile(derr, 99) < 5.0

    # ---- tables ----
    T = tp.loc[lib_t].copy()
    T["predicted_irt"] = ref.rt_ref.to_numpy().astype(np.float32)
    T["predicted_im"] = np.where(np.isfinite(ref.im_ref), ref.im_ref, T.predicted_im)
    D = T.copy()
    dsrc = lpi.loc[dids]
    for col in ("candidate_id", "peptidoform_id", "peptidoform", "label", "protein"):
        D[col] = dsrc[col].to_numpy()
    if new_toks:
        D["peptidoform"] = [to_pform(new_toks[c]) for c in lib_t]
    assert np.abs(dsrc.precursor_mz.to_numpy() - T.precursor_mz.to_numpy()).max() < 1e-6, "pair precursor m/z differs"
    P = pd.concat([T, D], ignore_index=True).sort_values(["precursor_mz", "candidate_id"], kind="mergesort").reset_index(drop=True)
    o2n = pd.Series(np.arange(len(P)), index=P.candidate_id.to_numpy())
    idmap = pd.DataFrame({"old_candidate_id": P.candidate_id.astype(np.uint32), "new_candidate_id": np.arange(len(P), dtype=np.uint32),
                          "label": P.label})
    P["candidate_id"] = np.arange(len(P), dtype=np.uint32)
    P["n_fragments"] = 12
    for c, dt in [("peptidoform_id", "uint32"), ("base_peptide_id", "uint32"), ("charge", "int32"), ("predicted_irt", "float32"),
                  ("n_fragments", "int32")]:
        P[c] = P[c].astype(dt)
    P = P[lp.columns]
    write_engine_parquet(P, f"{out}/lib_precursors.parquet")
    write_engine_parquet(idmap, f"{out}/id_map.parquet")

    fcols = ["candidate_id", "mz", "predicted_intensity", "name", "ion_type", "ordinal", "frag_charge", "cardinality"]
    FT = lf.assign(predicted_intensity=lf.inten.astype(np.float32))
    FD = FT.assign(candidate_id=np.array([pair[c] for c in FT.candidate_id]), mz=dmz)
    F = pd.concat([FT, FD])[fcols]
    F["candidate_id"] = o2n.reindex(F.candidate_id.to_numpy()).to_numpy().astype(np.uint32)
    F = F.sort_values("candidate_id", kind="mergesort")
    for c, dt in [("predicted_intensity", "float32"), ("ordinal", "int32"), ("frag_charge", "int32"), ("cardinality", "int32")]:
        F[c] = F[c].astype(dt)
    write_engine_parquet(F, f"{out}/lib_fragments.parquet")

    # ---- per-run windows and seeds ----
    for i, r in enumerate(runs):
        os.makedirs(f"{out}/r{i}", exist_ok=True)
        rt = from_ref[i](ref.rt_ref.to_numpy())
        imc = ref.im_ref.to_numpy() + off[i]
        if not np.isfinite(imc).all():  # no re-picked 1/K0 anywhere: pass-1 window centre of this run
            rw1 = pq.read_table(r["windows"], columns=["candidate_id", "im_pred_cal"],
                                filters=[("candidate_id", "in", pa.array(lib_t, pa.uint32()))]).to_pandas().set_index("candidate_id").im_pred_cal
            imc = np.where(np.isfinite(imc), imc, rw1.reindex(lib_t).to_numpy())
        tw = pd.DataFrame({"old": np.r_[lib_t, dids], "rt": np.r_[rt, rt], "im": np.r_[imc, imc]})
        tw["candidate_id"] = o2n.reindex(tw.old.to_numpy()).to_numpy()
        tw = tw.sort_values("candidate_id")
        rw = pa.table({"candidate_id": pa.array(tw.candidate_id.to_numpy(), pa.uint32()),
                       "rt_pred_cal": tw.rt.to_numpy(), "rt_lo": tw.rt.to_numpy() - hw_rt[i], "rt_hi": tw.rt.to_numpy() + hw_rt[i],
                       "im_pred_cal": tw.im.to_numpy(), "im_lo": tw.im.to_numpy() - hw_im[i], "im_hi": tw.im.to_numpy() + hw_im[i]})
        write_engine_table(rw, f"{out}/r{i}/run_windows.parquet")
        # Quant traces: same centres, pass-1 half-widths. Extract and retrace build traces only inside the window,
        # and quant needs up to 7 + flank scans either side of the apex, more than the p99 ID window holds.
        p1 = pq.read_table(r["windows"], columns=["rt_lo", "rt_hi", "im_lo", "im_hi"]).to_pandas()
        qrt, qim = float(np.median(p1.rt_hi - p1.rt_lo) / 2), float(np.nanmedian(p1.im_hi - p1.im_lo) / 2)
        log.setdefault("quant_hw", {})[i] = [qrt, qim]
        write_engine_table(pa.table({"candidate_id": rw["candidate_id"], "rt_pred_cal": rw["rt_pred_cal"],
                                     "rt_lo": tw.rt.to_numpy() - qrt, "rt_hi": tw.rt.to_numpy() + qrt, "im_pred_cal": rw["im_pred_cal"],
                                     "im_lo": tw.im.to_numpy() - qim, "im_hi": tw.im.to_numpy() + qim}),
                           f"{out}/r{i}/run_windows_quant.parquet")
        sd = pq.read_table(r["seed"]).to_pandas()
        sd = sd[sd.candidate_id.astype(np.int64).isin(o2n.index)].copy()
        sd["candidate_id"] = o2n.reindex(sd.candidate_id.astype(np.int64).to_numpy()).to_numpy().astype(np.uint32)
        write_engine_parquet(sd.sort_values("candidate_id"), f"{out}/r{i}/seed_psms.parquet")
        # The run's mass calibration, beside the remapped seed where extract and retrace look for it.
        if os.path.exists(r["seed"] + ".masscal.json"):
            shutil.copyfile(r["seed"] + ".masscal.json", f"{out}/r{i}/seed_psms.parquet.masscal.json")
        log.setdefault("seed_rows", {})[i] = {"target": int((sd.label == "target").sum()), "decoy": int((sd.label == "decoy").sum())}
        log.setdefault("im_fallback", {})[i] = int(np.isnan(ref.im_ref.to_numpy()).sum())

    # ---- construction invariant: every library-side value equal within a pair ----
    P2 = pq.read_table(f"{out}/lib_precursors.parquet").to_pandas()
    F2 = pq.read_table(f"{out}/lib_fragments.parquet").to_pandas()
    assert (P2.candidate_id.to_numpy() == np.arange(len(P2))).all() and (np.diff(P2.precursor_mz) >= 0).all()
    nt, nd_ = o2n.reindex(lib_t).to_numpy(), o2n.reindex(dids).to_numpy()
    pt, pd_ = P2.iloc[nt].reset_index(drop=True), P2.iloc[nd_].reset_index(drop=True)
    assert (pt.label == "target").all() and (pd_.label == "decoy").all()
    for c in ("base_peptide_id", "charge", "precursor_mz", "predicted_irt", "n_fragments"):
        assert (pt[c].to_numpy() == pd_[c].to_numpy()).all(), c
    assert np.array_equal(pt.predicted_im.to_numpy(), pd_.predicted_im.to_numpy(), equal_nan=True)
    ft = F2.set_index("candidate_id").loc[nt].reset_index()
    fd = F2.set_index("candidate_id").loc[nd_].reset_index()
    assert len(ft) == len(fd)
    for c in ("predicted_intensity", "name", "ion_type", "ordinal", "frag_charge", "cardinality"):
        assert (ft[c].to_numpy() == fd[c].to_numpy()).all(), c
    for i in range(N):
        rw = pq.read_table(f"{out}/r{i}/run_windows.parquet").to_pandas().set_index("candidate_id")
        assert np.array_equal(rw.loc[nt].to_numpy(), rw.loc[nd_].to_numpy(), equal_nan=True), f"run {i} windows differ in a pair"
    print(f"construction invariant holds: {len(lib_t):,} pairs; precursors {len(P2):,}, fragments {len(F2):,}", flush=True)
    log.update(pairs=int(len(lib_t)), precursors=int(len(P2)), fragments=int(len(F2)))
    with open(f"{out}/summary.json", "w") as fh:
        json.dump(log, fh, indent=1, default=float)


def report(spec):
    q = 0.01
    report_q = float(spec["report_q"])
    idm = pq.read_table(spec["id_map"]).to_pandas()
    new2old = pd.Series(idm.old_candidate_id.astype(np.int64).to_numpy(), index=idm.new_candidate_id.astype(np.int64).to_numpy())
    libt = set(idm.old_candidate_id[idm.label == "target"].astype(np.int64))
    p1 = pq.read_table(spec["scored1"], columns=["candidate_id", "source", "q_value", "precursor_q"],
                       filters=[("label", "=", "target")]).to_pandas()
    acc1 = {i: set(g.candidate_id.astype(np.int64)) for i, g in p1[p1.q_value <= q].groupby("source")}
    t2 = pq.read_table(spec["scored2"])
    pqm = p1.groupby("candidate_id").precursor_q.min()
    libt = libt & set(pqm.index[pqm <= report_q].astype(np.int64))
    old = new2old.reindex(t2["candidate_id"].to_numpy().astype(np.int64)).to_numpy()
    out_ = (t2["label"].to_numpy() == "target") & ~np.isin(old, list(libt))
    for c in Q_COLS:
        if c in t2.column_names:
            t2 = t2.set_column(t2.schema.get_field_index(c), c, pa.array(np.where(out_, 1.0, t2[c].to_numpy())))
    print(f"report: targets limited to pass-1 precursor_q <= {report_q}: {len(libt):,} precursors; "
          f"{int(out_.sum()):,} target rows set to q 1.0", flush=True)
    src = t2["source"].to_numpy()
    acc2 = (t2["label"].to_numpy() == "target") & (t2["q_value"].to_numpy() <= q)
    tr = np.zeros(len(src), bool)
    print("run | pass-1 accepted | pass-2 accepted | new | dropped, in library | dropped, outside | net", flush=True)
    for i in np.unique(src):
        m = src == i
        a2, a1 = set(old[m & acc2]), acc1.get(int(i), set())
        tr[m & acc2] = ~np.isin(old[m & acc2], list(a1))
        print(f"r{i} | {len(a1):,} | {len(a2):,} | {len(a2 - a1):,} | {len((a1 - a2) & libt):,} | {len(a1 - libt):,} | "
              f"{len(a2) - len(a1):+,}", flush=True)
    for c, v in (("is_transferred", pa.array(tr)), ("transfer_q", pa.array(np.full(len(src), np.nan)))):
        if c in t2.column_names:
            t2 = t2.set_column(t2.schema.get_field_index(c), c, v)
        else:
            t2 = t2.append_column(c, v)
    write_engine_table(t2, spec["out"])


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("build", "report"):
        sys.exit(__doc__)
    with open(sys.argv[2], encoding="utf-8") as fh:
        spec = json.load(fh)
    {"build": build, "report": report}[sys.argv[1]](spec)
