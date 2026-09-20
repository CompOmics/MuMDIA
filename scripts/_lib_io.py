"""Parquet writing for the library-recipe helpers, in the encoding the engine accepts.

The engine rejects a parquet whose string columns are `large_string` rather than
`string`: `Table::read` fails with `column 'peptidoform' is not utf8`
(CLAUDE.md, "Sidecar and IO contracts"). `pandas.DataFrame.to_parquet` chooses the
width itself, and pandas 3.x with pyarrow 25 chooses `large_string`, so the four
library helpers silently produced libraries the engine refuses to load. The failure
is version-dependent, which is worse than a plain bug: it works for whoever wrote
the script and fails for the next person, at load time, on a file that looks fine.

`write_engine_parquet` narrows the large variants back and pins snappy, so a helper
cannot emit a library the engine will not read.

Shared rather than copied into each helper because it is a correctness contract:
four copies would drift, and the one that drifts is the one that produces an
unreadable library. Python puts a script's own directory on `sys.path`, so
`from _lib_io import write_engine_parquet` resolves no matter where the helper is
invoked from.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq


def narrow_type(ty: pa.DataType) -> pa.DataType:
    """Replace the 64-bit-offset arrow types with the 32-bit ones the engine reads.

    Recursive: a `large_list<large_string>` has to be narrowed at both levels, and
    the fragment tables do carry list columns.
    """
    if pa.types.is_large_string(ty):
        return pa.string()
    if pa.types.is_large_binary(ty):
        return pa.binary()
    if pa.types.is_large_list(ty):
        return pa.list_(narrow_type(ty.value_type))
    if pa.types.is_list(ty):
        return pa.list_(narrow_type(ty.value_type))
    if pa.types.is_struct(ty):
        return pa.struct([pa.field(f.name, narrow_type(f.type), f.nullable) for f in ty])
    return ty


def _rebased(col: pa.ChunkedArray) -> pa.ChunkedArray:
    """`col` with every chunk copied so its offsets start at zero.

    A chunk that is a slice of a larger array keeps the parent's offset buffer, and the
    `large_string -> string` cast rejects the slice when the parent's offsets past the
    slice's end exceed 2 GiB ("input array too large"), however small the slice itself is.
    That is how `to_engine_table` failed on the 8-12-mer library: the decoy builder's m/z
    sort left one 2.86 GB `large_string` array, and every 4M-row slice past the 2 GiB mark
    of it was refused. `concat_arrays` of a single chunk materialises it with its own
    offsets, after which the cast sees only the slice's bytes.
    """
    return pa.chunked_array([pa.concat_arrays([c]) for c in col.chunks], type=col.type)


def narrow_table(table: pa.Table) -> pa.Table:
    """An arrow Table cast to the 32-bit-offset encoding the engine accepts.

    Separate from `to_engine_table` because not every producer starts from pandas:
    `mbr_worker.py` builds its transfers table with `pa.array` calls directly, and a
    string built by `pa.array` from a numpy object array is just as likely to come out
    `large_string` as one pandas chose.
    """
    schema = pa.schema(
        [pa.field(f.name, narrow_type(f.type), f.nullable) for f in table.schema]
    )
    if schema == table.schema:
        return table
    cols = []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        narrowing = field.type != schema.field(i).type
        if narrowing and (pa.types.is_large_string(field.type) or pa.types.is_large_binary(field.type)):
            col = _rebased(col)
        cols.append(col)
    return pa.Table.from_arrays(cols, schema=table.schema).cast(schema)


# Rows converted per slice when a DataFrame becomes an arrow Table. A `string` array holds
# at most 2 GiB of character data (32-bit offsets), and `Table.from_pandas` converts a column
# as ONE array, so past that size pandas 3 / pyarrow 25 produce a `large_string` array that
# `narrow_table` cannot cast back ("Failed casting from large_string to string: input array
# too large"). Measured on the 285M-row 8-12-mer immunopeptidomics precursor table: the
# `peptidoform` column alone is 4.5 GB, and the reverse-decoy builder failed at the final
# write after 70 minutes. Converting in slices keeps every array under the limit; the
# resulting table is chunked, which parquet writes as row groups and the engine reads
# row group by row group.
SLICE_ROWS = 4_000_000


def to_engine_table(df) -> pa.Table:
    """A pandas DataFrame as an arrow Table the engine will accept, of any row count."""
    n = len(df)
    if n <= SLICE_ROWS:
        return narrow_table(pa.Table.from_pandas(df, preserve_index=False))
    parts = []
    for start in range(0, n, SLICE_ROWS):
        part = narrow_table(pa.Table.from_pandas(df.iloc[start:start + SLICE_ROWS], preserve_index=False))
        # A slice whose object column happens to be all-null converts to type `null`; cast
        # it to the first slice's schema so the concatenation is one type per column.
        if parts and part.schema != parts[0].schema:
            part = part.cast(parts[0].schema)
        parts.append(part)
    return pa.concat_tables(parts)


def write_engine_parquet(df, path) -> None:
    """Write `df` to `path` as snappy parquet with 32-bit-offset string columns.

    Use this instead of `df.to_parquet(path, index=False)` anywhere the output is
    read back by the engine.
    """
    pq.write_table(to_engine_table(df), str(path), compression="snappy")


def write_engine_table(table, path) -> None:
    """As `write_engine_parquet`, for a caller that already holds an arrow Table."""
    pq.write_table(narrow_table(table), str(path), compression="snappy")


def sort_fragments_by_candidate(path, buckets: int = 64, tmp_dir=None) -> int:
    """Rewrite the fragment table at `path` ordered by `candidate_id`, in place, streaming.

    The engine reads one candidate-id range of a fragment table through its parquet
    row-group statistics (`Library::load_range_with`), which needs the rows sorted by
    `candidate_id` at row-group granularity. The writers cannot produce that order directly:
    the importer streams fragments in the input's order, and a decoy builder appends decoy
    fragments after the targets while the precursor order interleaves the two. So every
    writer finishes with this pass. Two streaming passes: rows are partitioned into
    `buckets` temporary files by candidate-id range, then each bucket is sorted in memory
    (stable, so a candidate's fragments keep their stored order) and appended to the output.
    Resident set is one bucket; the temporary files are one extra copy on disk.

    Returns the row count. An already-sorted table is rewritten all the same; the cost is one
    read and one write of the table.
    """
    import os
    import shutil
    import tempfile

    import numpy as np
    import pyarrow.compute as pc

    pf = pq.ParquetFile(str(path))
    n_rows = pf.metadata.num_rows
    if n_rows == 0:
        pf.close()
        return 0
    # Candidate-id range from the footer statistics when present, else from a scan.
    hi = 0
    have_stats = True
    for i in range(pf.num_row_groups):
        st = pf.metadata.row_group(i).column(pf.schema_arrow.get_field_index("candidate_id")).statistics
        if st is None or st.max is None:
            have_stats = False
            break
        hi = max(hi, int(st.max))
    if not have_stats:
        for i in range(pf.num_row_groups):
            hi = max(hi, int(pc.max(pf.read_row_group(i, columns=["candidate_id"]).column(0)).as_py()))
    buckets = max(1, min(int(buckets), hi + 1))
    width = (hi + buckets) // buckets  # ids per bucket, so that bucket = id // width
    work = tempfile.mkdtemp(prefix="sort_fragments_", dir=tmp_dir or os.path.dirname(os.path.abspath(str(path))) or None)
    try:
        writers = [None] * buckets
        schema = None
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i)
            if schema is None:
                schema = t.schema
            b = pc.divide(t.column("candidate_id").cast(pa.int64()), width).to_numpy()
            for k in np.unique(b):
                part = t.filter(pa.array(b == k))
                if writers[k] is None:
                    writers[k] = pq.ParquetWriter(os.path.join(work, f"bucket_{k:04d}.parquet"), schema, compression="snappy")
                writers[k].write_table(part)
        for w in writers:
            if w is not None:
                w.close()
        # The input must be closed before it is replaced: Windows refuses to rename over an
        # open file.
        pf.close()
        out = str(path) + ".sorted.tmp"
        writer = pq.ParquetWriter(out, schema, compression="snappy")
        written = 0
        try:
            for k in range(buckets):
                bp = os.path.join(work, f"bucket_{k:04d}.parquet")
                if not os.path.exists(bp):
                    continue
                t = pq.read_table(bp)
                idx = pc.sort_indices(t, sort_keys=[("candidate_id", "ascending")])  # stable
                t = t.take(idx)
                writer.write_table(t)
                written += t.num_rows
                os.remove(bp)
        finally:
            writer.close()
        if written != n_rows:
            os.remove(out)
            raise RuntimeError(f"sort_fragments_by_candidate: wrote {written} of {n_rows} rows")
        os.replace(out, str(path))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return n_rows
