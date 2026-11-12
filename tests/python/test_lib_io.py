"""The shared library writer: engine encoding for tables of any size and any chunk layout."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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


def test_uniprot_fields_follows_the_engines_header_rule():
    """The importer and the engine (`protein_annotations::uniprot_fields`) split an
    identifier the same way: `db|ACCESSION|ENTRY_NAME`, prefixed or not, gives its two
    fields; anything else is its own accession."""
    from _lib_io import uniprot_fields

    assert uniprot_fields("sp|P02768|ALBU_HUMAN") == ("P02768", "ALBU_HUMAN")
    assert uniprot_fields("tr|A0A024R161|A0A024R161_HUMAN") == ("A0A024R161", "A0A024R161_HUMAN")
    assert uniprot_fields("CON_sp|P02769|ALBU_BOVIN") == ("P02769", "ALBU_BOVIN")
    assert uniprot_fields("CON__P13645") == ("CON__P13645", "")
    assert uniprot_fields(" P02768 ") == ("P02768", "")


def test_protein_annotations_round_trip_sorted_and_retire(tmp_path):
    """Rows ascend by key whatever order they are given in and read back as written;
    `retire_side_table` removes a side table beside an output library that is not the
    input's own."""
    from _lib_io import (annotations_path, read_protein_annotations, retire_side_table,
                         write_protein_annotations)

    a = tmp_path / "a"
    a.mkdir()
    path = annotations_path(a / "lib_precursors.parquet")
    n = write_protein_annotations({"B": {"protein_ids": "P2"}, "A": {"genes": "G1"}}, path)
    assert n == 2
    t = pq.read_table(path)
    assert t.column("protein").to_pylist() == ["A", "B"]
    rows = read_protein_annotations(path)
    assert rows["A"]["genes"] == "G1" and rows["B"]["protein_ids"] == "P2"
    assert rows["A"]["protein_ids"] == "" and rows["B"]["description"] == ""
    retire_side_table(a / "p.parquet", a / "q.parquet")
    assert os.path.isfile(path), "the input's own table stays"
    b = tmp_path / "b"
    b.mkdir()
    stale = annotations_path(b / "x.parquet")
    write_protein_annotations({"Z": {"genes": "GZ"}}, stale)
    retire_side_table(a / "lib_precursors.parquet", b / "x.parquet")
    assert not os.path.exists(stale)


def test_annotation_columns_are_cheap_categoricals_written_as_utf8(tmp_path):
    """`annotate_precursors` gives every row its protein's annotation, the engine's join
    (the key's row, else its members'), a decoy its target's; the columns are pandas
    Categoricals and land in the parquet as plain utf8, which the engine reads.
    `annotation_table_from_precursors` reads the same table back from the target rows."""
    import pandas as pd

    from _lib_io import (ANNOTATION_COLUMNS, annotate_precursors,
                         annotation_table_from_precursors, has_annotation_columns,
                         write_engine_parquet)

    table = {"A_HUMAN": {"protein_ids": "P1", "protein_names": "A_HUMAN", "genes": "GA"},
             "B_HUMAN": {"protein_ids": "P2", "protein_names": "B_HUMAN", "genes": ""}}
    df = pd.DataFrame({
        "candidate_id": np.arange(5, dtype=np.uint32),
        "label": ["target", "decoy", "target", "target", "target"],
        "protein": ["A_HUMAN", "DECOY_A_HUMAN", "A_HUMAN;B_HUMAN", "UNASSIGNED", "B_HUMAN"],
    })
    annotate_precursors(df, table)
    assert all(isinstance(df[c].dtype, pd.CategoricalDtype) for c in ANNOTATION_COLUMNS)
    got = list(zip(*(df[c].astype(str) for c in ANNOTATION_COLUMNS)))
    assert got == [("P1", "A_HUMAN", "GA"), ("P1", "A_HUMAN", "GA"),
                   ("P1;P2", "A_HUMAN;B_HUMAN", "GA"), ("", "", ""), ("P2", "B_HUMAN", "")]
    path = tmp_path / "lib_precursors.parquet"
    write_engine_parquet(df, path)
    schema = pq.read_schema(str(path))
    assert all(schema.field(c).type == pa.string() for c in ANNOTATION_COLUMNS)
    back = pd.read_parquet(path)
    assert has_annotation_columns(back.columns)
    assert annotation_table_from_precursors(back) == {
        "A_HUMAN": {"protein_ids": "P1", "protein_names": "A_HUMAN", "genes": "GA"},
        "A_HUMAN;B_HUMAN": {"protein_ids": "P1;P2", "protein_names": "A_HUMAN;B_HUMAN",
                            "genes": "GA"},
        "B_HUMAN": {"protein_ids": "P2", "protein_names": "B_HUMAN", "genes": ""},
    }
    with pytest.raises(SystemExit, match="written together"):
        has_annotation_columns(["protein", "genes"])
    assert not has_annotation_columns(["protein", "label"])


def test_fasta_headers_complete_what_a_table_leaves_empty(tmp_path):
    """`fasta_annotations` reads each header as `mumdia digest` does; `complete_from_fasta`
    fills only the empty fields of a key from the FASTA proteins its members or accessions
    name, by first word, accession or entry name."""
    from _lib_io import complete_from_fasta, fasta_annotations, fasta_index

    fasta = tmp_path / "p.fasta"
    fasta.write_text(
        ">sp|P02768|ALBU_HUMAN Serum albumin OS=Homo sapiens OX=9606 GN=ALB PE=1 SV=2\nK\n"
        ">CON_sp|P02769|ALBU_BOVIN Albumin OS=Bos taurus GN=ALB\nK\n"
        ">tr|Q00001|Q00001_HUMAN Uncharacterized OS=Homo sapiens GN=UNC1\nK\n"
        ">plain description words\nK\n", encoding="utf-8")
    rows = fasta_annotations(fasta)
    assert rows["sp|P02768|ALBU_HUMAN"] == {"protein_ids": "P02768", "protein_names":
                                            "ALBU_HUMAN", "genes": "ALB",
                                            "description": "Serum albumin"}
    assert rows["plain"]["protein_ids"] == "plain" and rows["plain"]["genes"] == ""
    index = fasta_index(rows)
    table = {
        # names given, no gene: the gene is filled, the names kept
        "ALBU_HUMAN": {"protein_ids": "P02768", "protein_names": "MY_NAME", "genes": ""},
        # nothing given: a contaminant id names its FASTA entry by accession
        "CON_sp|P02769|ALBU_BOVIN": {"protein_ids": "", "protein_names": "", "genes": ""},
    }
    keys = ["ALBU_HUMAN", "CON_sp|P02769|ALBU_BOVIN", "Q00001_HUMAN;ALBU_HUMAN", "NOPE"]
    assert complete_from_fasta(table, keys, rows, index) == 3
    assert table["ALBU_HUMAN"] == {"protein_ids": "P02768", "protein_names": "MY_NAME",
                                   "genes": "ALB"}
    assert table["CON_sp|P02769|ALBU_BOVIN"]["genes"] == "ALB"
    assert table["Q00001_HUMAN;ALBU_HUMAN"] == {"protein_ids": "Q00001;P02768",
                                                "protein_names": "Q00001_HUMAN;ALBU_HUMAN",
                                                "genes": "UNC1;ALB"}
    assert "NOPE" not in table
