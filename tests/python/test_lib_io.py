"""The shared library writer: engine encoding for tables of any size and any chunk layout."""
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import _lib_io as L  # noqa: E402


def test_sliced_large_string_chunk_narrows_to_string_with_rebased_offsets():
    # A chunk that is a slice of a larger array keeps the parent's offsets; the cast to
    # `string` rejects such a slice once the parent's offsets pass 2 GiB. Re-basing is what
    # makes the cast see only the slice, so the offsets must start at zero afterwards.
    parent = pa.array([f"pep{i}" for i in range(1000)], pa.large_string())
    table = pa.table({"peptidoform": pa.chunked_array([parent.slice(600, 300)]), "x": pa.array(range(300))})
    narrowed = L.narrow_table(table)
    assert narrowed.schema.field("peptidoform").type == pa.string()
    chunk = narrowed.column("peptidoform").chunks[0]
    assert chunk.offset == 0
    assert chunk[0].as_py() == "pep600" and chunk[299].as_py() == "pep899"
    assert narrowed.column("x").to_pylist() == list(range(300))


def test_to_engine_table_slices_large_frames(monkeypatch):
    monkeypatch.setattr(L, "SLICE_ROWS", 100)
    df = pd.DataFrame({"peptidoform": [f"A{i}" for i in range(350)], "protein": [""] * 350, "mz": range(350)})
    table = L.to_engine_table(df)
    assert table.num_rows == 350
    assert table.schema.field("peptidoform").type == pa.string()
    assert table.column("peptidoform").num_chunks == 4
    assert table.column("peptidoform")[349].as_py() == "A349"
    assert table.column("mz").to_pylist() == list(range(350))


def test_to_engine_table_small_frame_is_one_chunk():
    df = pd.DataFrame({"peptidoform": ["PEPTIDEK"], "protein": ["P1"]})
    table = L.to_engine_table(df)
    assert table.num_rows == 1 and table.column("peptidoform").num_chunks == 1
    assert table.schema.field("protein").type == pa.string()


def test_sort_fragments_by_candidate_orders_rows_and_keeps_each_candidates_order(tmp_path):
    import numpy as np
    import pyarrow.parquet as pq

    # 500 fragments over 100 candidates, shuffled, written in row groups of 50 so several
    # groups span the whole id range before the sort.
    rng = np.random.default_rng(3)
    cid = np.repeat(np.arange(100, dtype=np.uint32), 5)
    order = rng.permutation(len(cid))
    cid = cid[order]
    seq = np.arange(len(cid), dtype=np.int64)[order]  # original row id, to check stability
    table = pa.table({"candidate_id": pa.array(cid, pa.uint32()), "mz": pa.array(seq.astype(float) * 0.5), "row": pa.array(seq)})
    p = tmp_path / "frag.parquet"
    pq.write_table(table, p, row_group_size=50)
    n = L.sort_fragments_by_candidate(p, buckets=7)
    assert n == 500
    out = pq.read_table(p)
    got = out.column("candidate_id").to_numpy()
    assert list(got) == sorted(got)
    # Stable within a candidate: rows keep their pre-sort relative order.
    rows = out.column("row").to_numpy()
    for c in range(100):
        r = rows[got == c]
        pre = seq[cid == c]
        assert list(r) == list(pre)
    assert out.schema == table.schema
    # Row-group statistics are monotonic, which is what the engine's range load keys on.
    pf = pq.ParquetFile(p)
    idx = pf.schema_arrow.get_field_index("candidate_id")
    mx = [pf.metadata.row_group(i).column(idx).statistics.max for i in range(pf.num_row_groups)]
    mn = [pf.metadata.row_group(i).column(idx).statistics.min for i in range(pf.num_row_groups)]
    assert all(mx[i] <= mn[i + 1] for i in range(len(mx) - 1))
