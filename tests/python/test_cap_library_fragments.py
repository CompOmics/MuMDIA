"""Contract tests for `scripts/cap_library_fragments.py`.

The helper exists to bring a fragment table under the engine's 2**32 - 1 row limit by
keeping each precursor's most intense fragments. What must hold: the N most intense of
every candidate survive, in their stored order, across row-group boundaries; candidate ids
are untouched, so the precursor table still pairs with the output; and the output is a
parquet the engine reads.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from conftest import assert_engine_readable_parquet, run_worker, run_worker_ok


def _fragments(path, cids, intensities, row_group_size):
    n = len(cids)
    t = pa.table(
        {
            "candidate_id": pa.array(cids, pa.uint32()),
            "mz": pa.array(np.linspace(200.0, 900.0, n), pa.float64()),
            "predicted_intensity": pa.array(intensities, pa.float32()),
            "name": pa.array([f"y{i}" for i in range(n)], pa.large_string()),
            "ion_type": pa.array(["y"] * n, pa.string()),
            "ordinal": pa.array(np.arange(n) % 20 + 1, pa.int32()),
            "frag_charge": pa.array([1] * n, pa.int32()),
        }
    )
    pq.write_table(t, path, row_group_size=row_group_size)
    return t


def test_keeps_the_top_n_per_candidate_in_stored_order_across_row_groups(work):
    rng = np.random.default_rng(7)
    # Candidates with 1, 3, 25 and 8 fragments; row groups of 5 rows cut through them.
    counts = [1, 3, 25, 8]
    cids = np.repeat(np.arange(len(counts)), counts)
    inten = rng.random(len(cids)).astype(np.float32)
    src, dst = work / "frag.parquet", work / "capped.parquet"
    full = _fragments(src, cids, inten, row_group_size=5)
    run_worker_ok("cap_library_fragments.py", str(src), str(dst), "--top", "4")

    out = pq.read_table(dst)
    assert out.column("candidate_id").to_pylist() == sorted(out.column("candidate_id").to_pylist())
    full_df = full.to_pandas()
    out_df = out.to_pandas()
    for c, k in enumerate(counts):
        mine = full_df[full_df.candidate_id == c]
        want = mine.sort_values("predicted_intensity", ascending=False, kind="stable").head(4)
        got = out_df[out_df.candidate_id == c]
        # The top 4 (or all, when fewer), in the order they were stored.
        assert list(got["name"]) == list(mine[mine.index.isin(want.index)]["name"]), c
        assert len(got) == min(k, 4)
    assert_engine_readable_parquet(dst)


def test_refuses_a_table_not_ordered_by_candidate(work):
    src, dst = work / "frag.parquet", work / "capped.parquet"
    _fragments(src, [0, 1, 0, 1], [1.0, 1.0, 1.0, 1.0], row_group_size=4)
    rc, out, err = run_worker("cap_library_fragments.py", str(src), str(dst))
    assert rc != 0
    assert "not ordered by candidate_id" in out + err
    assert not dst.exists()
