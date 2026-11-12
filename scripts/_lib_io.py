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
    the fragment tables do carry list columns. A dictionary column (a pandas
    `Categorical`, which the protein annotation helpers below build to keep a
    per-row string column cheap in memory) is decoded to its value type: the engine
    reads plain utf8 columns, and parquet dictionary-encodes them on disk anyway.
    """
    if pa.types.is_dictionary(ty):
        return narrow_type(ty.value_type)
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


# ---------------------------------------------------------------------------------------
# Protein annotations: in the library itself, and the side table older libraries have.
#
# A library's `protein` column is the grouping key every later table inherits (the scored
# tables' `protein` and `protein_group`, quant, LFQ) and the one the engine reports. It is
# one string per precursor, `;`-separated for a shared peptide, and for an imported DIA-NN
# library it holds UniProt entry names (`ALBU_HUMAN`), so the accessions and gene names
# were lost at import. The annotations keep them without touching the key: per `protein`
# string, its accessions, entry names and genes. The engine joins them at report time to fill
# `protein_ids`, `protein_names` and `genes`, the last three columns of the TSVs; every
# identification and quantity is unchanged.
#
# Where they live (#299): in three optional columns of the precursor table itself
# (`ANNOTATION_COLUMNS`), which give every row the annotation of its `protein`, so a library
# carries them wherever its two tables are copied. The importer writes them, and every helper
# that writes a library keeps them: the decoy builders copy each target row, columns
# included, onto its decoy, and a library from before the columns existed gets them from the
# side table beside it (`embed_annotations`). That side table, `protein_annotations.parquet`
# (one row per `protein` key), is what the importer wrote beside a library before; the engine
# still reads it beside a library without the columns, and `mumdia digest` still writes one
# for a FASTA, which is how a FASTA-built library and the added peptides of
# `augment_library.py` are annotated.
# ---------------------------------------------------------------------------------------

PROTEIN_ANNOTATIONS = "protein_annotations.parquet"
# The side table's columns, in order; all utf8 and non-null (empty when unknown). The key is
# unique and the rows ascend by it, so the bytes depend only on the content.
PROTEIN_ANNOTATION_COLUMNS = ("protein", "protein_ids", "protein_names", "genes", "description")
# The engine's name and version for it (`artifact::PROTEIN_ANNOTATIONS`).
PROTEIN_ANNOTATIONS_SCHEMA = ("protein_annotations", 1)
# The optional precursor-table columns: on every row, the annotation of that row's `protein`
# (utf8, `;`-separated, empty when unknown). All three or none. A decoy row carries its
# target's values (target and decoy rows stay alike in every column a decoy builder does not
# rewrite); the engine reads the target rows only (`protein_annotations::LIBRARY_COLUMNS`) and
# no stage of the search reads them, so they change no identification or quantity.
ANNOTATION_COLUMNS = ("protein_ids", "protein_names", "genes")


def annotations_path(precursors_path) -> str:
    """Where the protein annotation side table of the library at `precursors_path` lives."""
    import os

    return os.path.join(os.path.dirname(os.path.abspath(str(precursors_path))), PROTEIN_ANNOTATIONS)


def uniprot_fields(token: str):
    """(accession, entry name) of one FASTA header token or protein identifier.

    `db|ACCESSION|ENTRY_NAME` (UniProt `sp|` / `tr|`, and the same with a prefix such as
    `CON_sp|` or `ENTRAP_sp|`) gives its two fields; any other token is its own accession
    with no entry name. The engine's FASTA reader applies the same rule
    (`protein_annotations::uniprot_fields`)."""
    token = token.strip()
    parts = token.split("|")
    if len(parts) >= 3 and parts[1]:
        return parts[1], parts[2]
    return token, ""


def join_unique(values) -> str:
    """`;`-join of the non-empty values, each once, in first-occurrence order."""
    seen = []
    for v in values:
        v = v.strip()
        if v and v not in seen:
            seen.append(v)
    return ";".join(seen)


def _tokens(cell) -> list:
    """The `;`-separated values of one annotation cell."""
    return [t for t in ("" if cell is None else str(cell)).split(";") if t.strip()]


def write_protein_annotations(rows, path) -> int:
    """Write the annotation side table: `rows` maps a protein key to the dict of its columns
    (missing ones empty). Rows ascend by key. Returns the row count."""
    keys = sorted(k for k in rows if k)
    cols = {c: [] for c in PROTEIN_ANNOTATION_COLUMNS}
    for k in keys:
        r = rows[k]
        cols["protein"].append(k)
        for c in PROTEIN_ANNOTATION_COLUMNS[1:]:
            cols[c].append(str(r.get(c, "") or ""))
    schema = pa.schema([pa.field(c, pa.string(), False) for c in PROTEIN_ANNOTATION_COLUMNS])
    table = pa.table({c: pa.array(cols[c], pa.string()) for c in PROTEIN_ANNOTATION_COLUMNS},
                     schema=schema)
    pq.write_table(table, str(path), compression="snappy")
    return len(keys)


def read_protein_annotations(path) -> dict:
    """The annotation side table at `path` as {protein: {column: value}}, or {} when absent."""
    import os

    if not os.path.isfile(str(path)):
        return {}
    t = pq.read_table(str(path)).to_pydict()
    out = {}
    for i, k in enumerate(t.get("protein", [])):
        out[k] = {c: (t[c][i] if c in t and t[c][i] is not None else "")
                  for c in PROTEIN_ANNOTATION_COLUMNS[1:]}
    return out


def has_annotation_columns(columns) -> bool:
    """Whether a precursor table with these column names carries the annotation columns: all
    three, or none. A table with only some of them is refused, as the engine refuses it."""
    present = [c for c in ANNOTATION_COLUMNS if c in columns]
    if present and len(present) < len(ANNOTATION_COLUMNS):
        missing = [c for c in ANNOTATION_COLUMNS if c not in present]
        raise SystemExit(f"the library carries the protein annotation column(s) {present} but "
                         f"not {missing}; the three are written together")
    return bool(present)


def lookup_annotation(table, protein):
    """(protein_ids, protein_names, genes) of one `protein` key in an annotation table
    ({protein: {column: value}}): the key's own row, else the rows of its `;`-separated
    members, each value once, in member order. The engine's join
    (`ProteinAnnotations::lookup`)."""
    row = table.get(protein)
    if row is not None:
        return tuple(join_unique(_tokens(row.get(c, ""))) for c in ANNOTATION_COLUMNS)
    members = [table[m.strip()] for m in str(protein).split(";") if m.strip() in table]
    return tuple(join_unique(t for r in members for t in _tokens(r.get(c, "")))
                 for c in ANNOTATION_COLUMNS)


def annotate_precursors(df, table):
    """`df` (a precursor table) with the `ANNOTATION_COLUMNS` set from the annotation table
    `table` on each row's `protein` (`lookup_annotation`); a decoy row is looked up under its
    target's key, without the `DECOY_` prefix, so it carries its target's values.

    The columns are pandas `Categorical`s, one category per distinct value, so a library of
    hundreds of millions of rows costs a code per row here; the writer decodes them to the
    utf8 columns the engine reads (`narrow_type`), which parquet stores dictionary-encoded.
    `df` is changed in place and returned."""
    import numpy as np
    import pandas as pd

    protein = df["protein"].astype(str)
    if "label" in df.columns:
        decoy = (df["label"].astype(str) == "decoy") & protein.str.startswith("DECOY_")
        if decoy.any():
            protein = protein.where(~decoy, protein.str.slice(6))
    codes, keys = pd.factorize(protein, sort=False)
    cells = [lookup_annotation(table, k) for k in keys]
    for j, c in enumerate(ANNOTATION_COLUMNS):
        values = np.array([cell[j] for cell in cells] or [""], dtype=object)
        cats, inverse = np.unique(values, return_inverse=True)
        df[c] = pd.Categorical.from_codes(inverse[codes], categories=list(cats))
    return df


def annotation_table_from_precursors(df) -> dict:
    """The annotation table ({protein: {column: value}}) a precursor table's
    `ANNOTATION_COLUMNS` amount to, read from its target rows: each key's values in row order,
    each once; a key whose three values are all empty gets no entry. The engine reads the
    same (`protein_annotations::from_library`)."""
    t = df[df["label"].astype(str) == "target"] if "label" in df.columns else df
    sub = t[["protein", *ANNOTATION_COLUMNS]].astype(str).drop_duplicates()
    rows = {}
    for prot, *vals in sub.itertuples(index=False):
        if not prot or not any(v.strip() for v in vals):
            continue
        row = rows.setdefault(prot, {c: "" for c in ANNOTATION_COLUMNS})
        for c, v in zip(ANNOTATION_COLUMNS, vals):
            row[c] = join_unique(_tokens(row[c]) + _tokens(v))
    return rows


def embed_annotations(df, precursors_path):
    """The library read into `df` from `precursors_path`, with its protein annotations in its
    columns: as it is when it carries them; else with columns taken from the side table beside
    it (a library imported before the columns existed), a decoy row taking its target's
    values; else unchanged. Returns (df, how), how being `columns`, `side table` or `none`,
    and says which."""
    if has_annotation_columns(df.columns):
        print("protein annotations: carried in the library's protein_ids, protein_names and "
              "genes columns", flush=True)
        return df, "columns"
    side = read_protein_annotations(annotations_path(precursors_path))
    if side:
        print(f"protein annotations: {len(side)} protein entr(ies) from "
              f"{annotations_path(precursors_path)}, written into the output library's columns",
              flush=True)
        return annotate_precursors(df, side), "side table"
    print("protein annotations: none in the input library (no protein_ids, protein_names and "
          "genes columns, no protein_annotations.parquet beside it); a search will warn",
          flush=True)
    return df, "none"


def retire_side_table(src_precursors, dst_precursors) -> None:
    """After a helper wrote a library at `dst_precursors` from the one at `src_precursors`:
    remove a `protein_annotations.parquet` beside the output that is not the input's own. It
    belongs to another library, and the engine reads it beside a library without annotation
    columns. The output carries its annotations in its columns (`embed_annotations`), so no
    side table is written for it."""
    import os

    src, dst = annotations_path(src_precursors), annotations_path(dst_precursors)
    if os.path.normcase(os.path.abspath(src)) == os.path.normcase(os.path.abspath(dst)):
        return
    if os.path.isfile(dst):
        os.remove(dst)
        print(f"protein annotations: removed {dst}, left by another library (the output "
              f"library carries its annotations in its columns)", flush=True)


def fasta_annotations(path) -> dict:
    """The annotation of every protein of a FASTA, keyed by its header's first word (what
    `mumdia digest` writes as the protein), as `digest` writes them
    (`protein_annotations::from_fasta_header`): the accession and entry name of a
    `db|ACCESSION|ENTRY_NAME` word, else the word as the accession; the gene from `GN=`; the
    description before ` OS=`. The first header of a repeated key wins."""
    rows = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            words = header.split()
            if not words:
                continue
            key = words[0]
            rest = header[len(key):].strip()
            acc, name = uniprot_fields(key)
            gene = next((w[3:] for w in rest.split() if w.startswith("GN=")), "")
            if rest.startswith("OS="):
                desc = ""
            else:
                i = rest.find(" OS=")
                desc = rest if i < 0 else rest[:i].strip()
            rows.setdefault(key, {"protein_ids": acc, "protein_names": name, "genes": gene,
                                  "description": desc})
    return rows


def fasta_index(rows) -> dict:
    """token -> FASTA key: each header's first word, its accession and its entry name, the
    first header to give a token winning it."""
    index = {}
    for key, r in rows.items():
        for token in (key, r["protein_ids"], r["protein_names"]):
            if token:
                index.setdefault(token, key)
    return index


def complete_from_fasta(table, keys, fasta_rows, index) -> int:
    """Fill, for every protein key in `keys`, each annotation the table `table` leaves empty
    from the FASTA proteins its tokens name: the key's `;`-separated members and the
    accessions the table already gives it, each matched as a header's first word, its
    accession or its entry name (a `db|ACCESSION|NAME` token also by its accession). A value
    the table gives is kept as it is. Changes `table` in place; returns how many keys gained
    a value."""
    filled = 0
    for key in keys:
        row = table.get(key, {})
        tokens = _tokens(key) + _tokens(row.get("protein_ids", ""))
        hits = []
        for t in tokens:
            t = t.strip()
            k = index.get(t) or index.get(uniprot_fields(t)[0])
            if k is not None and k not in hits:
                hits.append(k)
        if not hits:
            continue
        gained = False
        for c in ANNOTATION_COLUMNS:
            if row.get(c, ""):
                continue
            v = join_unique(t for k in hits for t in _tokens(fasta_rows[k][c]))
            if v:
                row[c] = v
                gained = True
        if gained:
            table[key] = {c: row.get(c, "") for c in ANNOTATION_COLUMNS}
            filled += 1
    return filled
