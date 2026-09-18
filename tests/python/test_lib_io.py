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
