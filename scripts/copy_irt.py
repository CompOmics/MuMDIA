"""Copy calibrated predicted_irt from a source precursor table into a target
precursor table, matching by peptidoform. Avoids re-running DeepLC calibration
when only the fragment set changed.

Usage: python copy_irt.py <src_precursors> <dst_precursors>
"""
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    src_path, dst_path = sys.argv[1], sys.argv[2]
    src = pq.read_table(src_path).to_pydict()
    irt_by_pf = {}
    for pf, irt in zip(src["peptidoform"], src["predicted_irt"]):
        irt_by_pf[pf] = irt

    dst = pq.read_table(dst_path)
    pforms = dst.column("peptidoform").to_pylist()
    new = np.array([irt_by_pf.get(pf, 0.0) for pf in pforms], dtype=np.float32)
    miss = int((new == 0.0).sum())
    idx = dst.schema.get_field_index("predicted_irt")
    dst = dst.set_column(idx, "predicted_irt", pa.array(new, pa.float32()))
    pq.write_table(dst, dst_path)
    print(f"copied iRT for {len(pforms)} rows ({miss} unmatched)")


if __name__ == "__main__":
    main()
