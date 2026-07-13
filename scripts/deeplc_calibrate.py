"""Per-run DeepLC calibration (PLAN.md Stage B / Section 5 improvement): calibrate
the DeepLC multitask model to this run's observed RTs (from confident seed PSMs),
then write calibrated predicted_irt into the library precursor table.

Targets are predicted+calibrated; shift-decoys inherit their target's calibrated
RT (their peptidoform is `DECOY_<target>`).

Usage: python deeplc_calibrate.py <lib_precursors_in> <seed_psms> <lib_precursors_out>
"""
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def agg(a):
    a = np.asarray(a, dtype=np.float64)
    return a.mean(axis=1) if a.ndim == 2 else a


def main():
    lib_in, seed_path, lib_out = sys.argv[1:4]
    import deeplc
    from psm_utils import PSM, PSMList

    lib = pq.read_table(lib_in)
    pform = lib.column("peptidoform").to_pylist()
    label = lib.column("label").to_pylist()

    # Reference: confident target seed PSMs (peptidoform + observed RT).
    seed = pq.read_table(seed_path).to_pydict()
    ref = {}
    for i in range(len(seed["peptidoform"])):
        if seed["label"][i] == "target" and seed["spectrum_q"][i] <= 0.01:
            ref[seed["peptidoform"][i]] = seed["observed_rt"][i]
    ref_psms = PSMList(psm_list=[
        PSM(peptidoform=pf, retention_time=rt, spectrum_id=str(k))
        for k, (pf, rt) in enumerate(ref.items())
    ])
    print(f"calibration reference: {len(ref_psms)} confident seed peptides")

    # Unique target peptidoforms to predict+calibrate.
    uniq = []
    seen = set()
    for pf, lb in zip(pform, label):
        if lb == "target" and pf not in seen:
            seen.add(pf)
            uniq.append(pf)
    print(f"predicting {len(uniq)} unique target peptidoforms")

    cal = {}
    chunk = 200_000
    for s in range(0, len(uniq), chunk):
        batch = uniq[s:s + chunk]
        preds = agg(deeplc.predict_and_calibrate(batch, ref_psms))
        for pf, v in zip(batch, preds):
            cal[pf] = float(v)
        print(f"  {min(s + chunk, len(uniq))}/{len(uniq)}")

    # Assign: targets from cal; decoys from their DECOY_-stripped target.
    new_irt = np.empty(len(pform), dtype=np.float32)
    miss = 0
    for i, (pf, lb) in enumerate(zip(pform, label)):
        key = pf[6:] if pf.startswith("DECOY_") else pf
        if key in cal:
            new_irt[i] = cal[key]
        else:
            new_irt[i] = 0.0
            miss += 1
    print(f"assigned iRT; {miss} rows without a prediction")

    idx = lib.schema.get_field_index("predicted_irt")
    lib = lib.set_column(idx, "predicted_irt", pa.array(new_irt, pa.float32()))
    pq.write_table(lib, lib_out)
    print(f"wrote calibrated library: {lib_out}")


if __name__ == "__main__":
    main()
