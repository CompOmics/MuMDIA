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
    return table.cast(schema) if schema != table.schema else table


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
