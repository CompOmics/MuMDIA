"""MBR transfer worker (Stage D3, rescuable tier: M1 anchors, M2 expected RT, M3
transfer score, M4 decoy-transfer FDR). Extends the validated /c/proteobench
prototype into a reusable sidecar.

Reads the experiment-wide scored_combined (candidate_id, source, label, q_value, ...)
plus per-run psms (candidate_id, apex_rt). For each precursor confident in
>= min_anchor_runs OTHER runs but sub-threshold in a target run where it WAS
extracted (rescuable), predicts its RT in the target run from the median of the
other runs' binned-median-aligned apex RTs, and accepts the transfer if the observed
apex sits within a data-driven window. False-transfer FDR is estimated with a
permuted-RT decoy-transfer null (transfer to a shuffled precursor's expected RT);
the transfer q-value is standard target/decoy competition on the RT residual.

Contract:
  mbr_worker.py <scored_combined> <psms_csv> <out_transferred> [options]
    <psms_csv> = comma-separated per-run psms.parquet paths in `source` order.
  options: --q-anchor 0.01 --min-anchor-runs 2 --q-transfer 0.01 --seed 0
           (RT window is learned from the anchors, not fixed.)

Output <out_transferred>.parquet: one row per ACCEPTED transfer
  (candidate_id, source, peptidoform, charge, protein_group, label, expected_rt,
   observed_rt, rt_delta, transfer_q).
Also prints a validation summary (accepted counts per run, empirical decoy fraction).
"""
import argparse
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa


def binned_map(x, y, nb=80):
    """Monotone binned-median calibration x -> y."""
    o = np.argsort(x)
    xs, ys = np.asarray(x)[o], np.asarray(y)[o]
    e = np.linspace(0, len(xs), nb + 1).astype(int)
    cx, cy = [], []
    for b in range(nb):
        lo, hi = e[b], e[b + 1]
        if hi > lo:
            cx.append(np.median(xs[lo:hi]))
            cy.append(np.median(ys[lo:hi]))
    cx, cy = np.array(cx), np.array(cy)
    return lambda q: np.interp(q, cx, cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scored")
    ap.add_argument("psms_csv")
    ap.add_argument("out")
    ap.add_argument("--q-anchor", type=float, default=0.01)
    ap.add_argument("--min-anchor-runs", type=int, default=2)
    ap.add_argument("--q-transfer", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-scored", default=None,
                    help="also write an augmented scored table: the input scored_combined "
                         "with each accepted transfer's (candidate_id, source) row q_value set "
                         "to its transfer_q and an is_transferred flag added, so quant/report "
                         "(with quant.q_filter=psm_q) pick up the transfers.")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    psms_paths = a.psms_csv.split(",")
    n_runs = len(psms_paths)
    sc = pq.read_table(a.scored, columns=["candidate_id", "source", "label", "q_value",
                                          "peptidoform", "charge", "protein_group"]).to_pandas()
    # meta per candidate_id (peptidoform/charge/protein_group/label) from any row
    meta = sc.drop_duplicates("candidate_id").set_index("candidate_id")[
        ["peptidoform", "charge", "protein_group", "label"]]

    # confident set per run (targets AND decoys tracked so we can measure the empirical
    # decoy fraction among accepted transfers). Anchors use targets only (a decoy anchor
    # would inject a random cross-run RT pair); decoys ride the same transfer test.
    conf_t = {i: set(sc[(sc.source == i) & (sc.label == "target") & (sc.q_value <= a.q_anchor)].candidate_id)
              for i in range(n_runs)}

    # per-run apex RT (all extracted candidates) + confident-target apex (for maps)
    rt_all, rt_anchor = {}, {}
    for i, p in enumerate(psms_paths):
        d = pq.read_table(p, columns=["candidate_id", "apex_rt"]).to_pandas()
        m = dict(zip(d.candidate_id, d.apex_rt))
        rt_all[i] = m
        rt_anchor[i] = {c: m[c] for c in conf_t[i] if c in m}

    REF = 0
    to_ref = {i: (binned_map([rt_anchor[i][c] for c in rt_anchor[i] if c in rt_anchor[REF]],
                             [rt_anchor[REF][c] for c in rt_anchor[i] if c in rt_anchor[REF]])
                  if i != REF and sum(c in rt_anchor[REF] for c in rt_anchor[i]) >= 200
                  else (lambda q: q)) for i in range(n_runs)}
    from_ref = {i: (binned_map([rt_anchor[REF][c] for c in rt_anchor[REF] if c in rt_anchor[i]],
                               [rt_anchor[i][c] for c in rt_anchor[REF] if c in rt_anchor[i]])
                    if i != REF and sum(c in rt_anchor[i] for c in rt_anchor[REF]) >= 200
                    else (lambda q: q)) for i in range(n_runs)}

    support_t = {}
    allc = set().union(*conf_t.values())
    for c in allc:
        support_t[c] = sum(c in conf_t[i] for i in range(n_runs))

    # build transfer candidates (rescuable: confident-elsewhere, sub-threshold here,
    # extracted here). Predict RT; also a permuted-RT decoy prediction for the null.
    rows = {"candidate_id": [], "source": [], "expected_rt": [], "observed_rt": [],
            "rt_delta": []}
    decoy_delta = []                      # permuted-RT null residuals (per target)
    for i in range(n_runs):
        cand, preds, obs = [], [], []
        for c in allc:
            other = support_t[c] - (1 if c in conf_t[i] else 0)
            if other < a.min_anchor_runs or c in conf_t[i] or c not in rt_all[i]:
                continue
            js = [j for j in range(n_runs) if j != i and c in rt_anchor[j]]
            if len(js) < a.min_anchor_runs:
                continue
            pred = from_ref[i](np.median([to_ref[j](rt_anchor[j][c]) for j in js]))
            cand.append(c); preds.append(pred); obs.append(rt_all[i][c])
        if not cand:
            continue
        preds = np.array(preds); obs = np.array(obs)
        d = np.abs(obs - preds)
        # permuted-RT decoy: each candidate gets a shuffled candidate's predicted RT
        shuf = rng.permutation(len(cand))
        dd = np.abs(obs - preds[shuf])
        for k, c in enumerate(cand):
            rows["candidate_id"].append(c); rows["source"].append(i)
            rows["expected_rt"].append(float(preds[k])); rows["observed_rt"].append(float(obs[k]))
            rows["rt_delta"].append(float(d[k]))
        decoy_delta.extend(dd.tolist())

    target_delta = np.array(rows["rt_delta"])
    decoy_delta = np.array(decoy_delta)
    if len(target_delta) == 0:
        print("MBR: no transfer candidates"); pa_write_empty(a.out); return

    # transfer q via target/decoy competition on rt_delta (smaller = better). At a
    # threshold delta, FDR = (#decoy <= delta) / (#target <= delta). q = running min.
    order = np.argsort(target_delta)
    dt = np.sort(target_delta)
    dd = np.sort(decoy_delta)
    dec_cum = np.searchsorted(dd, dt, side="right")          # decoys within each delta
    tgt_cum = np.arange(1, len(dt) + 1)
    fdr = dec_cum / tgt_cum
    q_sorted = np.minimum.accumulate(fdr[::-1])[::-1]         # monotone q from the tail
    q = np.empty_like(q_sorted); q[order] = q_sorted          # map back to row order

    accept = q <= a.q_transfer
    cid = np.array(rows["candidate_id"]); src = np.array(rows["source"])
    lab = meta.reindex(cid)["label"].values
    n_acc = int(accept.sum())
    acc_dec = int(((lab == "decoy") & accept).sum())
    print(f"MBR transfer: candidates={len(cid)} accepted@q<={a.q_transfer}={n_acc} "
          f"(target={n_acc-acc_dec}, decoy={acc_dec}, empirical decoy-frac="
          f"{acc_dec/max(1,n_acc)*100:.2f}%)")
    delta_star = dt[q_sorted <= a.q_transfer].max() if (q_sorted <= a.q_transfer).any() else 0.0
    print(f"  RT window at q<={a.q_transfer}: {delta_star:.1f}s")
    for i in range(n_runs):
        m = accept & (src == i) & (lab == "target")
        print(f"  run {i}: +{int(m.sum())} target transfers")

    out = pa.table({
        "candidate_id": pa.array(cid[accept], pa.uint32()),
        "source": pa.array(src[accept], pa.uint32()),
        "peptidoform": pa.array(meta.reindex(cid[accept])["peptidoform"].values),
        "charge": pa.array(meta.reindex(cid[accept])["charge"].values.astype("int32")),
        "protein_group": pa.array(meta.reindex(cid[accept])["protein_group"].values),
        "label": pa.array(lab[accept]),
        "expected_rt": pa.array(np.array(rows["expected_rt"])[accept], pa.float64()),
        "observed_rt": pa.array(np.array(rows["observed_rt"])[accept], pa.float64()),
        "rt_delta": pa.array(target_delta[accept], pa.float64()),
        "transfer_q": pa.array(q[accept], pa.float64()),
    })
    pq.write_table(out, a.out)
    print(f"wrote {a.out} ({out.num_rows} accepted transfers)")

    # M5: augmented scored table. Lower the accepted transfers' PSM q_value to their
    # transfer_q on the matching (candidate_id, source) row and flag them, so a
    # downstream quant/report with quant.q_filter=psm_q includes the transfers.
    if a.out_scored:
        full = pq.read_table(a.scored).to_pandas()
        acc = {(int(c), int(s)): float(qq) for c, s, qq in
               zip(cid[accept], src[accept], q[accept])}
        key = list(zip(full.candidate_id.astype(int), full.source.astype(int)))
        newq = full.q_value.to_numpy().copy()
        is_tr = np.zeros(len(full), dtype=bool)
        for i, k in enumerate(key):
            if k in acc:
                newq[i] = min(newq[i], acc[k])
                is_tr[i] = True
        full["q_value"] = newq
        full["is_transferred"] = is_tr
        pq.write_table(pa.Table.from_pandas(full, preserve_index=False), a.out_scored)
        print(f"wrote {a.out_scored} (augmented scored; {int(is_tr.sum())} rows flagged transferred)")


def pa_write_empty(path):
    pq.write_table(pa.table({"candidate_id": pa.array([], pa.uint32())}), path)


if __name__ == "__main__":
    main()
