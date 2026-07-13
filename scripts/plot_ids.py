"""Per-peptide diagnostic plots for MuMDIA, drawn from the SEARCH ENGINE's OWN output
(not a Python re-extraction), so they can be used to evaluate the engine:
  - XICs   = the engine's emitted chromatograms (extract.rs; zero-filled window grid
             when emit_window_grid is on, else matched scans).
  - apex   = the engine's apex_rt from psms_extracted (fragment-count + top-3).
  - bounds = the engine's emitted elution_lo/elution_hi (features.rs), i.e. the window
             over which the engine computed its trace features. Read from the features
             parquet, NOT recomputed in Python.
  Panel A: fragment XICs + summed + engine apex + engine feature bounds + predicted RT
           and search window; DIA-NN [RT.Start,RT.Stop] shaded in 'diann' mode.
  Panel B: apex MS2 mirror, observed (engine, at apex) vs predicted library.

Modes (chrom source):
  mumdia -> engine gate-on output (chrom_port / psms_port), all MuMDIA IDs.
  diann  -> engine gate-off grid output (chrom_portng_grid / psms_portng_grid), DIA-NN
            IDs MuMDIA misses at the SEQUENCE level (a peptide identified under a
            different charge/modform is NOT counted as missed).
Env: FRAC (bound fraction, default 1/3 = features default), CHUNK/NCHUNK, OUTSUB.
Usage: python plot_ids.py <mumdia|diann> [limit]
"""
import os
import re
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "c:/Users/robbi/OneDrive - UGent/MuMDIA_NG"
Q = 0.01
FRAC = float(os.environ.get("FRAC", str(1.0 / 3.0)))
CHUNK = int(os.environ.get("CHUNK", "0"))
NCHUNK = int(os.environ.get("NCHUNK", "1"))
OUTSUB = os.environ.get("OUTSUB", "")


def to_pf(s):
    return str(s).replace("(UniMod:4)", "[Carbamidomethyl]").replace("(UniMod:35)", "[Oxidation]")


def strip_seq(pf):
    """Stripped amino-acid sequence (drop mods + DECOY_ prefix) for sequence-level ID matching."""
    s = re.sub(r"\[[^\]]*\]", "", str(pf))
    s = re.sub(r"\([^\)]*\)", "", s)
    return s.replace("DECOY_", "")


def parse_frag(name):
    m = re.match(r"([by])(\d+)(?:\^(\d+))?", str(name))
    return (m.group(1), int(m.group(2)), int(m.group(3) or 1)) if m else ("?", 0, 1)


def frag_color(ion, ordinal, maxord):
    cmap = plt.cm.Blues if ion == "b" else plt.cm.Reds if ion == "y" else plt.cm.Greys
    return cmap(0.45 + 0.5 * (ordinal / max(maxord, 1)))


def smooth(v):
    if len(v) < 3:
        return v.astype(float)
    o = v.astype(float).copy()
    o[1:-1] = 0.5 * v[1:-1] + 0.25 * v[:-2] + 0.25 * v[2:]
    o[0] = 2 / 3 * v[0] + 1 / 3 * v[1]
    o[-1] = 2 / 3 * v[-1] + 1 / 3 * v[-2]
    return o


def peak_bounds(axis, prof, ai, frac):
    """Match features.rs peak_bounds: descend from apex to prof < frac*peak."""
    if axis.size < 3:
        return (axis[0], axis[-1]) if axis.size else (0.0, 0.0)
    peak = prof[ai] if prof[ai] > 0 else prof.max()
    if peak <= 0:
        return axis[0], axis[-1]
    thr = frac * peak
    lo = ai
    while lo > 0 and prof[lo - 1] >= thr:
        lo -= 1
    hi = ai
    while hi + 1 < axis.size and prof[hi + 1] >= thr:
        hi += 1
    return axis[lo], axis[hi]


def plot_one(cid, pf, charge, qv, apex_rt, chrom_rows, lib_rows, outpath,
             dn_win=None, pred_rt=None, half=None, elo=None, ehi=None):
    # chrom_rows: list of (name, rt_array, int_array) = ENGINE chromatograms.
    # Build a shared union axis for the mirror.
    axis = np.unique(np.concatenate([r[1] for r in chrom_rows])) if chrom_rows else np.array([])
    names = [r[0] for r in chrom_rows]
    predmap = {r[0]: r[1] for r in lib_rows}
    mat = np.zeros((len(chrom_rows), axis.size))
    for i, (_, rt, it) in enumerate(chrom_rows):
        idx = np.clip(np.searchsorted(axis, rt), 0, axis.size - 1) if axis.size else []
        mat[i, idx] = it
    ai = int(np.argmin(np.abs(axis - apex_rt))) if axis.size else 0
    # elution bounds: use the ENGINE's emitted feature bounds (features.rs), not recomputed
    lo_rt, hi_rt = (elo, ehi) if (elo is not None and ehi is not None and ehi > elo) else (apex_rt, apex_rt)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    maxord = max([parse_frag(n)[1] for n in (names + list(predmap))] + [1])
    apex_col = mat[:, ai] if axis.size else np.zeros(len(names))
    athr = 0.05 * apex_col.max() if apex_col.max() > 0 else 0.0
    obs_apex = apex_col > athr
    summ = mat.sum(axis=0)
    for i, name in sorted(enumerate(names), key=lambda kv: parse_frag(kv[1])):
        if not obs_apex[i]:
            continue
        ion, ordi, _ = parse_frag(name)
        axA.plot(chrom_rows[i][1], chrom_rows[i][2], "-", lw=1.2, color=frag_color(ion, ordi, maxord), label=name)
    if axis.size:
        axA.plot(axis, summ, "-", lw=2.0, color="0.4", alpha=0.5, label="summed", zorder=1)
        axA.axvspan(lo_rt, hi_rt, color="0.85", alpha=0.4, zorder=0, label="feature bounds (engine)")
        axA.axvline(lo_rt, color="0.5", ls=":", lw=1); axA.axvline(hi_rt, color="0.5", ls=":", lw=1)
    axA.axvline(apex_rt, color="k", ls="--", lw=1.3, label="engine apex")
    if pred_rt is not None and np.isfinite(pred_rt):
        axA.axvline(pred_rt, color="tab:purple", ls=":", lw=1.4, label="pred RT")
        if half:
            axA.axvline(pred_rt - half, color="tab:orange", ls=":", lw=1.0, alpha=0.7)
            axA.axvline(pred_rt + half, color="tab:orange", ls=":", lw=1.0, alpha=0.7, label=f"search +/-{half:.0f}s")
    if dn_win is not None and np.isfinite(dn_win[0]):
        axA.axvspan(dn_win[0], dn_win[1], color="tab:green", alpha=0.12, zorder=0)
        axA.axvline(dn_win[0], color="tab:green", lw=1, alpha=0.6)
        axA.axvline(dn_win[1], color="tab:green", lw=1, alpha=0.6, label="DIA-NN RT")
    all_names = list(predmap)
    missing = [n for n in all_names if n not in set(np.array(names)[obs_apex]) if len(names)]
    axA.set_xlabel("retention time (s)"); axA.set_ylabel("fragment intensity")
    axA.set_title(f"XIC (engine)  {int(obs_apex.sum())}/{len(all_names)} frags at apex"
                  + (f"\nmissing: {', '.join(missing[:8])}" if missing else ""), fontsize=9)
    axA.legend(fontsize=6, ncol=2, loc="upper right")

    obs = {names[i]: (mat[i, ai] if axis.size else 0.0) for i in range(len(names))}
    order = sorted(all_names, key=lambda n: (parse_frag(n)[0], parse_frag(n)[1]))
    pmax = max(predmap.values()) if predmap else 1.0
    omax = max(obs.values()) if obs and max(obs.values()) > 0 else 1.0
    for i, name in enumerate(order):
        ion, ordi, _ = parse_frag(name); c = frag_color(ion, ordi, maxord)
        axB.bar(i, obs.get(name, 0) / omax, 0.8, color=c)
        axB.bar(i, -(predmap.get(name, 0) / pmax), 0.8, color=c, alpha=0.55,
                hatch="///" if obs.get(name, 0) <= athr else None)
    axB.axhline(0, color="k", lw=0.8)
    axB.set_xticks(range(len(order))); axB.set_xticklabels(order, rotation=90, fontsize=6)
    axB.set_ylim(-1.15, 1.15); axB.set_ylabel("observed (up) / predicted (down)")
    axB.set_title("apex MS2: observed vs predicted", fontsize=9)
    fig.suptitle(f"{pf}  (z={charge})  q={qv:.4f}  apex={apex_rt:.0f}s  cid={cid}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=90); plt.close(fig)


def main():
    mode = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    lp = pd.read_parquet(f"{D}/out_dl/lib_precursors_ft.parquet",
                         columns=["candidate_id", "peptidoform", "charge", "label"])
    lf = pd.read_parquet(f"{D}/out_dl/lib_fragments_ft.parquet",
                         columns=["candidate_id", "name", "predicted_intensity"])
    scored = pd.read_parquet(f"{D}/out_dl/scored_port.parquet",
                             columns=["candidate_id", "peptidoform", "charge", "label", "q_value"])
    mm = scored[(scored.label == "target") & (scored.q_value <= Q)]
    mm_ids = set(mm.candidate_id)
    mm_seqs = set(mm.peptidoform.map(strip_seq))  # sequence-level IDs (any charge/modform)
    _cal = json.load(open(f"{D}/out_dl/cal_ftgate.json"))
    HALF = _cal["w_rt"] / _cal.get("multiplier", 1.0) * 1.10

    if mode == "mumdia":
        outdir = f"{D}/out_dl/plots_mumdia{OUTSUB}"
        chrom_path = f"{D}/out_dl/chrom_port.parquet"
        psms = pd.read_parquet(f"{D}/out_dl/psms_port.parquet", columns=["candidate_id", "apex_rt", "rt_pred_cal"])
        feat = pd.read_parquet(f"{D}/out_dl/feat_port.parquet", columns=["candidate_id", "elution_lo", "elution_hi"])
        t = mm[["candidate_id", "peptidoform", "charge", "q_value"]]
        targets = t.set_index("candidate_id"); dn_bounds = {}
    else:
        outdir = f"{D}/out_dl/plots_diann_not_mumdia{OUTSUB}"
        # Small pre-subset of the 2 GB gate-off grid to just the truly-missed candidates
        # (scratchpad/subset_grid.py), so parallel workers don't each decode a ~4 GB row group.
        chrom_path = f"{D}/out_dl/chrom_missed_grid.parquet"
        psms = pd.read_parquet(f"{D}/out_dl/psms_portng_grid.parquet", columns=["candidate_id", "apex_rt", "rt_pred_cal"])
        feat = pd.read_parquet(f"{D}/out_dl/feat_portng_grid.parquet", columns=["candidate_id", "elution_lo", "elution_hi"])
        dn = pd.read_csv(f"{D}/out_diann/report.tsv", sep="\t")
        dn = dn[dn["Q.Value"] <= Q].copy().rename(columns={"RT.Start": "rs", "RT.Stop": "re",
                                                           "Precursor.Charge": "pz", "Q.Value": "qv"})
        dn["pf"] = dn["Modified.Sequence"].map(to_pf); dn["key"] = list(zip(dn.pf, dn.pz.astype(int)))
        lpt = lp[lp.label == "target"].copy(); lpt["key"] = list(zip(lpt.peptidoform, lpt.charge.astype(int)))
        k2c = dict(zip(lpt.key, lpt.candidate_id))
        dn = dn.drop_duplicates("key"); dn["cid"] = dn.key.map(k2c); dn = dn[dn.cid.notna()]
        dn["cid"] = dn.cid.astype(int)
        # Exclude at SEQUENCE level: a peptide MuMDIA identified under any charge/modform is
        # NOT a miss (compete keeps one charge per base peptide). Excluding only the exact
        # candidate_id would leave sibling charges (e.g. z3 when z2 was identified) as false misses.
        dn["seq"] = dn.pf.map(strip_seq)
        dn = dn[~dn.cid.isin(mm_ids) & ~dn.seq.isin(mm_seqs)]
        dn_bounds = {int(r.cid): (r.rs * 60, r.re * 60) for r in dn.itertuples()}
        targets = dn.set_index("cid")[["pf", "pz", "qv"]]; targets.columns = ["peptidoform", "charge", "q_value"]

    os.makedirs(outdir, exist_ok=True)
    apex = dict(zip(psms.candidate_id, psms.apex_rt))
    rtcal = dict(zip(psms.candidate_id, psms.rt_pred_cal))
    ebounds = {r.candidate_id: (r.elution_lo, r.elution_hi) for r in feat.itertuples()}
    cids = [c for i, c in enumerate(targets.index) if i % NCHUNK == CHUNK]
    if limit:
        cids = cids[:limit]
    cidset = set(int(c) for c in cids)
    # Load only this chunk's candidates (predicate pushdown) so a 2 GB grid parquet does
    # not blow up memory with 8 parallel workers each reading the whole file.
    chrom = pd.read_parquet(chrom_path, filters=[("candidate_id", "in", cidset)])
    lf = lf[lf.candidate_id.isin(cidset)].sort_values(["candidate_id", "predicted_intensity"], ascending=[True, False])
    chrom_by = {c: [(r.frag_name, np.asarray(r.rt, float), np.asarray(r.intensity, float))
                    for r in g.itertuples()] for c, g in chrom.groupby("candidate_id")}
    lib_by = {c: list(zip(g.name.head(12), g.predicted_intensity.head(12))) for c, g in lf.groupby("candidate_id")}
    print(f"mode={mode} chunk={CHUNK}/{NCHUNK} n={len(cids)} -> {outdir}", flush=True)
    done = 0
    for cid in cids:
        cr = chrom_by.get(cid); lr = lib_by.get(cid)
        if not cr or not lr or cid not in apex:
            continue
        row = targets.loc[cid]
        pf = str(row["peptidoform"]); z = int(row["charge"]); qv = float(row["q_value"])
        fn = re.sub(r"[^A-Za-z0-9]", "_", pf)[:40] + f"_z{z}_{cid}.png"
        outpath = os.path.join(outdir, fn)
        if os.path.exists(outpath):
            done += 1; continue
        eb = ebounds.get(cid, (None, None))
        plot_one(cid, pf, z, qv, apex[cid], cr, lr, outpath,
                 dn_win=dn_bounds.get(cid), pred_rt=rtcal.get(cid), half=HALF,
                 elo=eb[0], ehi=eb[1])
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(cids)}", flush=True)
    print(f"wrote/kept {done} -> {outdir}")


if __name__ == "__main__":
    main()
