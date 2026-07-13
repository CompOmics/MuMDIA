"""Build a per-precursor / per-peptide LOSS FUNNEL attributing every DIA-NN
identification to the earliest MuMDIA stage where we lose it, on the current-best
E. coli pipeline (DIA-NN predicted library + shift decoys + GPU-fine-tuned
multitask DeepLC iRT + rich features).

Because the MuMDIA library IS DIA-NN's predicted library, library/fragment
prediction is (near) 100% coverage; the funnel isolates the DOWNSTREAM losses:
  NOT_IN_LIB -> NO_EVIDENCE (not extracted even gate-off)
             -> GATE_LOSS   (extracted gate-off, rejected by the extract gate)
             -> FDR_LOSS    (extracted with gate, ranked below 1% FDR)
             -> FOUND       (target q <= 0.01, gate-on)

Two chains feed it: gate-ON (reported IDs) and gate-OFF (maximal extraction, to
separate "no evidence at all" from "gate rejected it"). Diagnostics from DIA-NN
(RT, quantity, CScore, Ms1.Profile.Corr, Spectrum.Similarity) and from our own
features (frag_corr, coelution, ref_corr, rt_error, in-window) are attached per
row for the downstream deep-dive.

Usage: python build_funnel.py <out_dir> <gate_tag> <nogate_tag> <out_prefix>
  e.g. python build_funnel.py out_dl ftgate ftnogate out_dl/funnel
"""
import re
import sys
import numpy as np
import pandas as pd

Q = 0.01


def dn_to_pf(modseq):
    s = str(modseq).replace("(UniMod:4)", "[Carbamidomethyl]").replace("(UniMod:35)", "[Oxidation]")
    return s


def unmapped(modseq):
    return "(UniMod:" in dn_to_pf(modseq)


def strip_pf(s):
    return re.sub(r"\[[^\]]*\]", "", str(s).replace("DECOY_", ""))


def main():
    D, gate, nogate, outpref = sys.argv[1:5]

    # ---- DIA-NN reference (precursor level); run from project root ----
    dn = pd.read_csv("out_diann/report.tsv", sep="\t")
    dn = dn[dn["Q.Value"] <= Q].copy()
    dn = dn.rename(columns={"Precursor.Quantity": "dn_quantity", "CScore": "dn_cscore",
                            "Ms1.Profile.Corr": "dn_ms1corr", "Spectrum.Similarity": "dn_specsim",
                            "Mass.Evidence": "dn_massev"})
    dn["pf"] = dn["Modified.Sequence"].map(dn_to_pf)
    dn["mod_unmapped"] = dn["Modified.Sequence"].map(unmapped)
    dn["charge"] = dn["Precursor.Charge"].astype(int)
    dn["rt_sec"] = dn["RT"] * 60.0
    dn["rt_start_sec"] = dn["RT.Start"] * 60.0
    dn["rt_stop_sec"] = dn["RT.Stop"] * 60.0
    dn["key"] = list(zip(dn["pf"], dn["charge"]))
    dn["stripped"] = dn["Stripped.Sequence"]

    # ---- our combined ft library (targets): key -> candidate_id ----
    lib = pd.read_parquet(f"{D}/lib_precursors_ft.parquet",
                          columns=["candidate_id", "peptidoform", "charge", "label", "predicted_irt", "protein"])
    libt = lib[lib.label == "target"].copy()
    libt["key"] = list(zip(libt["peptidoform"], libt["charge"].astype(int)))
    key2cand = dict(zip(libt["key"], libt["candidate_id"]))

    # ---- seed (confident) ----
    seed = pd.read_parquet(f"{D}/seed_{gate}.parquet", columns=["candidate_id", "label", "spectrum_q"])
    seed_conf = set(seed[(seed.label == "target") & (seed.spectrum_q <= Q)].candidate_id)

    # ---- run windows (RT) ----
    rw = pd.read_parquet(f"{D}/rw_{gate}.parquet", columns=["candidate_id", "rt_lo", "rt_hi"])
    cand2lo = dict(zip(rw.candidate_id, rw.rt_lo))
    cand2hi = dict(zip(rw.candidate_id, rw.rt_hi))

    # ---- extraction: gate-off (any evidence) and gate-on ----
    px_ng = pd.read_parquet(f"{D}/psms_{nogate}.parquet",
                            columns=["candidate_id", "apex_rt", "n_matched_fragments", "coelution_run"])
    extracted_ng = set(px_ng.candidate_id)
    cand2apex = dict(zip(px_ng.candidate_id, px_ng.apex_rt))
    px_g = pd.read_parquet(f"{D}/psms_{gate}.parquet", columns=["candidate_id"])
    extracted_g = set(px_g.candidate_id)

    # ---- features gate-off (gate quantities for the GATE_LOSS deep-dive) ----
    fcols = ["candidate_id", "frag_corr", "frag_cosine", "n_matched_fragments", "coelution_run",
             "coelution_mean", "matched_fraction", "ref_corr", "profile_cos", "rt_error_abs",
             "log_apex_intensity", "isotope_corr", "log_sn", "seed_score"]
    feat_ng = pd.read_parquet(f"{D}/feat_{nogate}.parquet", columns=fcols).set_index("candidate_id")

    # ---- scored q-values: gate-on (reported) and gate-off ----
    sc_g = pd.read_parquet(f"{D}/scored_{gate}.parquet", columns=["candidate_id", "label", "q_value", "peptide_q_value"])
    cand2q_g = dict(zip(sc_g[sc_g.label == "target"].candidate_id, sc_g[sc_g.label == "target"].q_value))
    sc_ng = pd.read_parquet(f"{D}/scored_{nogate}.parquet", columns=["candidate_id", "label", "q_value"])
    cand2q_ng = dict(zip(sc_ng[sc_ng.label == "target"].candidate_id, sc_ng[sc_ng.label == "target"].q_value))

    # ---- assemble per-precursor rows ----
    rows = []
    for r in dn.itertuples(index=False):
        cand = key2cand.get(r.key)
        in_lib = cand is not None
        ext_ng = in_lib and (cand in extracted_ng)
        ext_g = in_lib and (cand in extracted_g)
        qg = cand2q_g.get(cand, np.nan) if in_lib else np.nan
        qng = cand2q_ng.get(cand, np.nan) if in_lib else np.nan
        found = in_lib and (qg <= Q)
        lo = cand2lo.get(cand, np.nan) if in_lib else np.nan
        hi = cand2hi.get(cand, np.nan) if in_lib else np.nan
        in_window = in_lib and not np.isnan(lo) and (lo <= r.rt_sec <= hi)
        f = feat_ng.loc[cand] if (in_lib and cand in feat_ng.index) else None

        if not in_lib:
            bucket = "1_NOT_IN_LIB"
        elif not ext_ng:
            bucket = "2_NO_EVIDENCE"
        elif not ext_g:
            bucket = "3_GATE_LOSS"
        elif not found:
            bucket = "4_FDR_LOSS"
        else:
            bucket = "5_FOUND"

        rows.append(dict(
            stripped=r.stripped, pf=r.pf, charge=r.charge, mod_unmapped=r.mod_unmapped,
            bucket=bucket, candidate_id=(int(cand) if in_lib else -1),
            in_lib=in_lib, seeded=(in_lib and cand in seed_conf),
            extracted_nogate=ext_ng, extracted_gate=ext_g,
            q_gate=qg, q_nogate=qng, found=found,
            rt_in_window=in_window, our_apex_rt=(cand2apex.get(cand, np.nan) if ext_ng else np.nan),
            rt_lo=lo, rt_hi=hi,
            dn_rt_sec=r.rt_sec, dn_rt_start_sec=r.rt_start_sec, dn_rt_stop_sec=r.rt_stop_sec,
            dn_quantity=r.dn_quantity, dn_cscore=r.dn_cscore, dn_ms1corr=r.dn_ms1corr,
            dn_specsim=r.dn_specsim, dn_massev=r.dn_massev,
            our_frag_corr=(float(f.frag_corr) if f is not None else np.nan),
            our_frag_cosine=(float(f.frag_cosine) if f is not None else np.nan),
            our_n_matched=(float(f.n_matched_fragments) if f is not None else np.nan),
            our_coelution_run=(float(f.coelution_run) if f is not None else np.nan),
            our_coelution_mean=(float(f.coelution_mean) if f is not None else np.nan),
            our_matched_fraction=(float(f.matched_fraction) if f is not None else np.nan),
            our_ref_corr=(float(f.ref_corr) if f is not None else np.nan),
            our_profile_cos=(float(f.profile_cos) if f is not None else np.nan),
            our_rt_error_abs=(float(f.rt_error_abs) if f is not None else np.nan),
            our_log_apex=(float(f.log_apex_intensity) if f is not None else np.nan),
            our_log_sn=(float(f.log_sn) if f is not None else np.nan),
        ))
    pp = pd.DataFrame(rows)
    pp.to_csv(f"{outpref}_per_precursor.csv", index=False)

    # ---- collapse to peptide level: best (max) bucket reached across precursors ----
    order = {"1_NOT_IN_LIB": 1, "2_NO_EVIDENCE": 2, "3_GATE_LOSS": 3, "4_FDR_LOSS": 4, "5_FOUND": 5}
    pp["brank"] = pp.bucket.map(order)
    pep = pp.loc[pp.groupby("stripped")["brank"].idxmax()].copy()  # best stage reached per peptide
    pep.to_csv(f"{outpref}_per_peptide.csv", index=False)

    # ---- summary ----
    with open(f"{outpref}_summary.txt", "w") as fh:
        def out(*a):
            print(*a); print(*a, file=fh)
        out("=== FUNNEL (precursor level, DIA-NN @1% =", len(pp), "precursors) ===")
        vc = pp.bucket.value_counts().sort_index()
        cum = 0
        for b in sorted(order, key=order.get):
            n = int(vc.get(b, 0)); cum += n
            out(f"  {b:16s} {n:6d}  ({100*n/len(pp):5.1f}%)   cum {cum}")
        out("")
        out("=== FUNNEL (peptide level, DIA-NN @1% =", len(pep), "peptides) ===")
        vcp = pep.bucket.value_counts().sort_index()
        for b in sorted(order, key=order.get):
            n = int(vcp.get(b, 0))
            out(f"  {b:16s} {n:6d}  ({100*n/len(pep):5.1f}%)")
        out("")
        out("=== per-bucket DIA-NN abundance / quality (precursor medians) ===")
        for b in sorted(order, key=order.get):
            g = pp[pp.bucket == b]
            if len(g) == 0:
                continue
            out(f"  {b:16s} n={len(g):5d}  dn_quant_med={g.dn_quantity.median():.3e}  "
                f"cscore_med={g.dn_cscore.median():.3f}  ms1corr_med={g.dn_ms1corr.median():.3f}  "
                f"specsim_med={g.dn_specsim.median():.3f}  rt_in_window={100*g.rt_in_window.mean():.0f}%  "
                f"seeded={100*g.seeded.mean():.0f}%")
        out("")
        out("=== GATE_LOSS sub-diagnosis (why the gate rejected; gate needs frag_corr>=0.5, presence>=3, coel>=2) ===")
        gl = pp[pp.bucket == "3_GATE_LOSS"]
        if len(gl):
            out(f"  n={len(gl)}  frag_corr<0.5: {100*(gl.our_frag_corr<0.5).mean():.0f}%  "
                f"n_matched<3: {100*(gl.our_n_matched<3).mean():.0f}%  "
                f"coelution_run<2: {100*(gl.our_coelution_run<2).mean():.0f}%")
            out(f"  frag_corr med={gl.our_frag_corr.median():.3f}  BUT DIA-NN specsim med={gl.dn_specsim.median():.3f} "
                f"(high specsim + low our frag_corr => our frag_corr computation, not missing signal)")
        out("")
        out("=== FDR_LOSS: would gate-off scoring have found them? ===")
        fl = pp[pp.bucket == "4_FDR_LOSS"]
        if len(fl):
            out(f"  n={len(fl)}  q_gate med={fl.q_gate.median():.3f}  "
                f"would be <=1% at gate-off q: {100*(fl.q_nogate<=Q).mean():.0f}%  "
                f"rt_in_window={100*fl.rt_in_window.mean():.0f}%")
        print("\nwrote", f"{outpref}_per_precursor.csv", f"{outpref}_per_peptide.csv", f"{outpref}_summary.txt")


if __name__ == "__main__":
    main()
