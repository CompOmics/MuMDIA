"""AlphaPeptDeep MS2 sidecar worker (the file contract in docs/13_sidecars.md).

Usage:
    python peptdeep_worker.py <input.parquet> <output.parquet> \
        [model] [nce] [instrument] [processes]

Environment:
    MUMDIA_PEPTDEEP_DEVICE = auto    auto|cuda|cpu, as `nn_rescore_worker.py` reads
                                     MUMDIA_NN_DEVICE. auto uses the GPU when torch
                                     can see one; cuda errors rather than silently
                                     falling back on a CPU-only torch build, so a
                                     device comparison cannot quietly measure the
                                     wrong thing.

Input parquet columns:  id (uint32), peptidoform (ProForma string), charge (int)
Output parquet columns: id (uint32), ion_type (str 'b'/'y'), ordinal (int),
                        frag_charge (int, 1 or 2), intensity (float, linear)

Same five-column contract as `ms2pip_worker.py`, deliberately: the engine enumerates
its own fragments and asks a predictor only for an intensity per
`(ion_type, ordinal, frag_charge)` triple, so the two predictors are interchangeable
at this boundary.

Run with an env that has peptdeep + pyarrow. The first run downloads AlphaPeptDeep's
pretrained models into `~/peptdeep/pretrained_models`.

What this worker does NOT do:

- predict retention time or ion mobility. AlphaPeptDeep can, but MuMDIA's RT comes
  from DeepLC and its calibration machinery is built around that;
- emit a, c, x, z or neutral-loss ions. `mumdia_core::mass::IonType` is `B | Y`, so
  the engine never generates those fragments and an intensity for one has nowhere to
  go. `frag_types` below is fixed for that reason, not by preference.

Both charge states are always requested. This is the one thing that must not become
configurable: with charge-1-only predictions the engine fills the charge-2 fragments
from its native heuristic and max-normalises each charge group separately, and the
measured consequence on a real run was 0 confident PSMs at 1% (78.6% of kept
fragments were heuristics; see `predict_frag.ms2pip_model` in config.rs). The engine
decides WHICH fragment charges exist, through `charge2_from_precursor_charge`; this
worker's job is to have an intensity ready for whichever it asks about.
"""
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# The four series the engine can use, in output order. Not configurable: see above.
FRAG_TYPES = ["b_z1", "b_z2", "y_z1", "y_z2"]
# (column, ion code 0=b 1=y, fragment charge)
SERIES = (("b_z1", 0, 1), ("b_z2", 0, 2), ("y_z1", 1, 1), ("y_z2", 1, 2))
ION_CODES = ("b", "y")

OUT_SCHEMA = pa.schema(
    [
        ("id", pa.uint32()),
        ("ion_type", pa.string()),
        ("ordinal", pa.int32()),
        ("frag_charge", pa.int32()),
        ("intensity", pa.float32()),
    ]
)


def parse_peptidoform(pform):
    """ProForma as MuMDIA writes it -> AlphaPeptDeep's (sequence, mods, mod_sites).

    MuMDIA's two producers, the native digest and `import_diann_lib.py`, both write
    modifications as UniMod NAMES in brackets (`PEC[Carbamidomethyl]TIDE`,
    `[Acetyl]-PEPK`), which is also alphabase's own `Name@Residue` convention, so the
    mapping is mechanical. Sites are 1-based residue positions, 0 for the N terminus
    and -1 for the C terminus, as alphabase expects.

    Returns `(sequence, mods, mod_sites)` or `None` for anything unsupported: a mass
    delta (`[+57.021464]`), or a name alphabase does not know at that residue. The
    engine drops a candidate no predictor covered, together with its target/decoy
    pair, and fails the run if more than 2% of the library goes that way, so refusing
    here is a reported outcome rather than a silent substitution.
    """
    seq, mods, sites = [], [], []
    i, n = 0, len(pform)

    # Optional N-terminal group, `[Mod]-`.
    if i < n and pform[i] == "[":
        close = pform.find("]", i)
        if close < 0 or close + 1 >= n or pform[close + 1] != "-":
            return None
        mods.append(f"{pform[i + 1:close]}@Any_N-term")
        sites.append(0)
        i = close + 2

    while i < n:
        c = pform[i]
        if c == "-":
            # C-terminal group, `-[Mod]`, only valid at the very end.
            if i + 1 >= n or pform[i + 1] != "[":
                return None
            close = pform.find("]", i + 1)
            if close < 0 or close != n - 1:
                return None
            mods.append(f"{pform[i + 2:close]}@Any_C-term")
            sites.append(-1)
            break
        if not c.isalpha():
            return None
        seq.append(c.upper())
        i += 1
        if i < n and pform[i] == "[":
            close = pform.find("]", i)
            if close < 0:
                return None
            name = pform[i + 1 : close]
            # A mass delta carries no identity, and guessing one from the mass would
            # pick the wrong isobaric modification silently. Name it instead.
            if name.lstrip("+-").replace(".", "", 1).isdigit():
                return None
            mods.append(f"{name}@{seq[-1]}")
            sites.append(len(seq))
            i = close + 1

    if not seq:
        return None
    return "".join(seq), ";".join(mods), ";".join(str(s) for s in sites)


def check_vocabulary(mods, known):
    """Every modification in `mods` that alphabase does not define."""
    out = set()
    for spec in mods:
        for m in spec.split(";"):
            if m and m not in known:
                out.add(m)
    return out


def fragment_rows(precursor_df, intensity_df):
    """Flatten one prediction into the five aligned output arrays.

    `precursor_df` carries `mumdia_id`, `nAA` and the half-open slice
    `[frag_start_idx, frag_stop_idx)` into `intensity_df`, whose columns are the four
    `FRAG_TYPES`. Within a precursor's slice, row `i` is the cleavage after residue
    `i + 1`, so it holds b(i+1) and y(nAA-1-i). Verified against
    AlphaPeptDeep's own `fragment_mz_df`: for a 9-mer, row 0 is b1 / y8 and row 7 is
    b8 / y1.

    Zero intensities are kept rather than dropped. They cost space and say nothing on
    their own, but the engine decides whether a model predicted the doubly charged
    series by looking for ANY charge-2 key; a precursor whose charge-2 predictions
    happened to be all zero would, if dropped, be scored as though the model were
    charge-1-only and get native heuristic values instead.
    """
    ids, ions, ords, chgs, ints = [], [], [], [], []
    vals = {c: intensity_df[c].to_numpy(dtype=np.float32, copy=False) for c in FRAG_TYPES}
    for rid, n_aa, start, stop in zip(
        precursor_df["mumdia_id"].to_numpy(),
        precursor_df["nAA"].to_numpy(),
        precursor_df["frag_start_idx"].to_numpy(),
        precursor_df["frag_stop_idx"].to_numpy(),
    ):
        width = int(stop) - int(start)
        if width <= 0:
            continue
        pos = np.arange(width, dtype=np.int32)
        b_ord = pos + 1
        y_ord = np.int32(n_aa) - 1 - pos
        for col, code, charge in SERIES:
            arr = vals[col][int(start) : int(stop)]
            ids.append(np.full(width, rid, dtype=np.uint32))
            ions.append(np.full(width, code, dtype=np.int8))
            ords.append(b_ord if code == 0 else y_ord)
            chgs.append(np.full(width, charge, dtype=np.int32))
            ints.append(arr.astype(np.float32, copy=False))
    if not ids:
        return None
    return (
        np.concatenate(ids),
        np.concatenate(ions),
        np.concatenate(ords),
        np.concatenate(chgs),
        np.concatenate(ints),
    )


def to_table(parts):
    ids, ions, ords, chgs, ints = parts
    ion_type = pa.DictionaryArray.from_arrays(
        pa.array(ions, pa.int8()), pa.array(list(ION_CODES), pa.string())
    ).dictionary_decode()
    return pa.table(
        {
            "id": pa.array(ids, pa.uint32()),
            "ion_type": ion_type,
            "ordinal": pa.array(ords, pa.int32()),
            "frag_charge": pa.array(chgs, pa.int32()),
            "intensity": pa.array(ints, pa.float32()),
        },
        schema=OUT_SCHEMA,
    )


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "generic"
    nce = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    instrument = sys.argv[5] if len(sys.argv) > 5 else "Lumos"
    cpu = os.cpu_count() or 1
    procs = int(sys.argv[6]) if len(sys.argv) > 6 else min(8, cpu)
    procs = max(1, min(procs, cpu))

    import pandas as pd
    import torch
    from alphabase.constants.modification import MOD_DF
    from peptdeep.pretrained_models import ModelManager
    from peptdeep.settings import global_settings

    # AlphaPeptDeep maps an unknown instrument onto its default rather than failing,
    # which would predict a different spectrum than the configuration asked for and
    # say nothing. Check it against the vocabulary the installed version defines.
    vocab = global_settings["model_mgr"]["instrument_group"]
    if instrument not in vocab:
        raise SystemExit(
            f"peptdeep_worker: unknown instrument {instrument!r}. This AlphaPeptDeep "
            f"knows: {', '.join(sorted(set(vocab)))}"
        )

    want = os.environ.get("MUMDIA_PEPTDEEP_DEVICE", "auto").strip().lower()
    if want not in ("auto", "cuda", "cpu"):
        raise SystemExit(
            f"MUMDIA_PEPTDEEP_DEVICE must be auto, cuda or cpu (got {want!r})"
        )
    if want == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "MUMDIA_PEPTDEEP_DEVICE=cuda but torch reports no CUDA device; this torch "
            f"build is {torch.__version__} -- install a CUDA build to use the GPU"
        )
    # AlphaPeptDeep spells its own request "gpu" and falls back to CPU on its own.
    device = "cpu" if want == "cpu" else "gpu"

    tbl = pq.read_table(in_path)
    ids = tbl.column("id").to_pylist()
    pforms = tbl.column("peptidoform").to_pylist()
    charges = tbl.column("charge").to_pylist()

    parsed, kept_ids, kept_charges, unsupported = [], [], [], []
    for rid, pform, charge in zip(ids, pforms, charges):
        p = parse_peptidoform(pform)
        if p is None:
            unsupported.append(pform)
        else:
            parsed.append(p)
            kept_ids.append(rid)
            kept_charges.append(charge)
    if unsupported:
        show = ", ".join(sorted(set(unsupported))[:5])
        print(
            f"peptdeep_worker: {len(unsupported)} of {len(ids)} peptidoforms are not "
            f"expressible for AlphaPeptDeep and are left unpredicted (e.g. {show}). "
            "The engine drops each with its pair and fails above 2%.",
            flush=True,
        )
    if not parsed:
        pq.write_table(pa.table([], schema=OUT_SCHEMA), out_path, compression="snappy")
        print("peptdeep_worker: nothing to predict")
        return

    known = set(MOD_DF.index)
    missing = check_vocabulary([m for _, m, _ in parsed], known)
    if missing:
        raise SystemExit(
            "peptdeep_worker: alphabase does not define these modifications, so their "
            f"intensities would be wrong rather than absent: {', '.join(sorted(missing))}"
        )

    mm = ModelManager(device=device)
    mm.load_installed_models(model)
    resolved = str(getattr(mm.ms2_model, "device", "unknown"))
    # Said out loud because the difference is an order of magnitude in wall time on a
    # whole-proteome library, and `auto` on a CPU-only torch build looks identical to
    # a working GPU from the outside.
    print(
        f"peptdeep_worker: peptdeep model={model} nce={nce} instrument={instrument} "
        f"device={want} -> {resolved} (torch {torch.__version__})",
        flush=True,
    )
    if want == "auto" and not resolved.startswith("cuda"):
        print(
            "peptdeep_worker: running on CPU. On a whole-proteome library this is the "
            "slow path; a CUDA torch build in this environment would use the GPU.",
            flush=True,
        )

    writer = pq.ParquetWriter(out_path, OUT_SCHEMA, compression="snappy")
    # Rows per prediction call. Streamed rather than concatenated at the end: a whole
    # proteome is tens of millions of precursors and hundreds of millions of fragment
    # rows, and holding all of them as arrays before the first byte is written is how
    # the MS2PIP worker once reached 13 GB of Python objects.
    chunk = 200_000
    n_rows = 0
    try:
        for start in range(0, len(parsed), chunk):
            end = min(start + chunk, len(parsed))
            block = parsed[start:end]
            df = pd.DataFrame(
                {
                    "mumdia_id": np.asarray(kept_ids[start:end], dtype=np.uint32),
                    "sequence": [p[0] for p in block],
                    "mods": [p[1] for p in block],
                    "mod_sites": [p[2] for p in block],
                    "charge": np.asarray(kept_charges[start:end], dtype=np.int32),
                    "nce": np.float32(nce),
                    "instrument": instrument,
                }
            )
            out = mm.predict_all(
                df,
                predict_items=["ms2"],
                frag_types=FRAG_TYPES,
                # AlphaPeptDeep may reorder the frame it is given (it batches by peptide
                # length), so `mumdia_id` is carried THROUGH the call and read back from
                # the returned precursor frame rather than zipped against the input.
                multiprocessing=(resolved == "cpu" and (end - start) >= 3000),
                process_num=procs,
            )
            part = fragment_rows(out["precursor_df"], out["fragment_intensity_df"])
            if part is not None:
                writer.write_table(to_table(part))
                n_rows += part[0].size
            print(
                f"peptdeep_worker: {end}/{len(parsed)} peptidoforms predicted",
                flush=True,
            )
    finally:
        writer.close()
    print(f"peptdeep_worker: {len(parsed)} peptidoforms -> {n_rows} fragment rows")


if __name__ == "__main__":
    main()
