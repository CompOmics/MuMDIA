"""Re-map the peptides of a MuMDIA library to the proteins of another FASTA (#319).

A library names, for every precursor, the proteins its peptide was found in: the `protein`
column, `;`-separated for a shared peptide. That column is the grouping key of every later
table (protein groups, protein q-values, quantities), and it is fixed when the library is
built or imported. A library predicted from one FASTA can be searched against another
database, such as the same proteome with contaminants or entrapment proteins added, or a
newer UniProt release, only when its peptides are mapped to that database again. This
helper does that, on any MuMDIA library: an imported one, a FASTA-built one, with or
without its decoys.

Mapping. A target peptide maps to every protein of the FASTA that contains its stripped
sequence as a peptide the engine's digest (`stages/digest.rs`) can produce, with I and L read
as one residue (they are isobaric, so the peptide is the same evidence for either protein):

- the occurrence starts at a protein start or a cut of the cleavage rule, and ends at the
  protein end or a cut. The rule is a preset of `digest.enzyme` (`--enzyme`, default
  `trypsin_p`, the engine's default) or a residue-pair rule (`--cleavage-rule K*,R*,!*P`);
  missed cleavages are not limited, since the library's peptides already chose them.
  `--enzyme any` (or `non_specific`) counts every occurrence, for a non-specific library
  such as an immunopeptidome. A plain substring rule is not the default because short
  peptides then match proteins they are no digestion product of: on a 0.85M-precursor
  E. coli library mapped to a human, yeast and E. coli FASTA, substring matching touched
  28,950 proteins, Trypsin/P 16,854;
- a protein start is position 0, or position 1 of a protein that begins with `M`: the
  N-terminal methionine-excision form (`digest.n_term_met_excision`; `--no-met-excision`
  turns it off);
- a peptidoform with a protein N-terminal modification group (`[Acetyl]-PEPK`, the names
  in `--protein-n-term-mods`, default `Acetyl`) maps only to the proteins it starts
  (`starts_protein`).

The new `protein` string is the matched proteins' FASTA keys joined with `;` in FASTA
order, the key being the header's first word, which is what `mumdia digest` writes
(`--protein-key accession` or `entry_name` use that field of a `db|ACCESSION|ENTRY_NAME`
header instead; a header without the field keeps its first word). The protein groups the
engine forms at rescore time follow from that column, so nothing else has to be rebuilt.

Peptides that the FASTA does not contain: `--unmapped drop` (default) removes them, their
decoys with them, so the library holds the FASTA's search space only; `--unmapped keep`
keeps them under the protein `UNASSIGNED`, the group the engine and the importer give a
peptide without a protein. The counts are printed either way.

Decoys are not mapped: a decoy row takes `DECOY_` plus the new protein of its target, the
target being the row with the same `(peptidoform_id, charge)` (what `make_reverse_decoys.py`
and `predict_decoys.py` pair on) or, failing that, the targets of its `base_peptide_id`
(the engine's picked-competition pairing). A decoy whose target was dropped is dropped.

Protein annotations (`protein_ids`, `protein_names`, `genes`) are rewritten from the new
FASTA's headers (`_lib_io.fasta_annotations`), because the old ones describe the old
proteins. A `protein_annotations.parquet` beside the output that is not the input's own is
removed (`retire_side_table`).

The remaining columns are kept. Rows are only removed, never reordered, so precursors stay
in ascending `precursor_mz`; `candidate_id` is renumbered to the contiguous range the
engine requires, and the fragment table is streamed one row group at a time with its ids
renumbered alike. The fragment `cardinality` column (how many library precursors share
each fragment m/z bin) is left as computed for the input library.

Usage:
  python remap_library.py <in_precursors> <in_fragments> <out_precursors> <out_fragments>
      --fasta <proteins.fasta> [--enzyme trypsin_p|trypsin|lys_c|...|any | --cleavage-rule <rule>]
      [--unmapped drop|keep] [--protein-key header|accession|entry_name]
      [--protein-n-term-mods Acetyl,...] [--no-met-excision] [--summary <out.json>]

The output may be the input (written through a temporary file).

`import_diann_lib.py --remap-fasta <fasta>` runs the same mapping on the imported targets.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from _lib_io import (annotate_precursors, embed_annotations, fasta_annotations,
                     narrow_table, retire_side_table, uniprot_fields, write_engine_parquet)

UNASSIGNED = "UNASSIGNED"
# A bracketed modification, with the dash of a terminal group (`[Acetyl]-PEPK`).
_STRIP_MODS = re.compile(r"-?\[[^\]]*\]-?")
# The ProForma N-terminal group, `[Name]-`, before the first residue.
_NTERM = re.compile(r"^\[([^\]]*)\]-")


def stripped(peptidoform: str) -> str:
    """Residues of a peptidoform: no `DECOY_` marker, no modification, no terminal dash."""
    s = peptidoform[6:] if peptidoform.startswith("DECOY_") else peptidoform
    return _STRIP_MODS.sub("", s)


def n_term_group(peptidoform: str) -> str:
    """The N-terminal modification group of a peptidoform (`Acetyl` of `[Acetyl]-PEPK`),
    or ''."""
    m = _NTERM.match(peptidoform[6:] if peptidoform.startswith("DECOY_") else peptidoform)
    return m.group(1) if m else ""


def il_key(sequence: str) -> str:
    """The sequence with I read as L: two sequences that differ only by I/L have the same
    precursor and fragment masses (the engine's `digest::il_key`)."""
    return sequence.replace("I", "L")


def read_fasta(path):
    """(key, sequence) of every FASTA record, as the engine's `digest::read_fasta` reads
    them: the key is the header's first word, the sequence upper-cased with whitespace
    trimmed from each line."""
    out = []
    key, seq = None, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if key:
                    out.append((key, "".join(seq)))
                words = line[1:].split()
                key, seq = (words[0] if words else ""), []
            else:
                seq.append(line.strip().upper())
    if key:
        out.append((key, "".join(seq)))
    return out


def protein_key(fasta_key: str, kind: str) -> str:
    """The protein name a FASTA record is written under: its header's first word
    (`header`, as `mumdia digest` writes it), or the accession / entry name field of a
    `db|ACCESSION|ENTRY_NAME` word, falling back to the first word when the field is
    absent."""
    if kind == "header":
        return fasta_key
    acc, name = uniprot_fields(fasta_key)
    value = acc if kind == "accession" else name
    return value or fasta_key


# The engine's cleavage presets (`mumdia-core/src/enzymes.json`), name -> rule. A copy,
# because the scripts are shipped without the engine's sources; `test_remap_library.py`
# checks it against the file. `any` is no rule: every occurrence counts.
ENZYME_RULES = {
    "trypsin_p": "K*,R*",
    "trypsin": "K*,R*,!*P",
    "lys_c": "K*",
    "lys_n": "*K",
    "arg_c": "R*,!*P",
    "asp_n": "*D",
    "glu_c": "E*",
    "chymotrypsin": "F*,Y*,W*,!*P",
    "non_specific": "**",
}


class CleavageRule:
    """A residue-pair cleavage rule (`K*,R*,!*P`), as the engine's `cleavage::CleavageRule`
    parses and applies it: a bond is cut when a positive pair matches and no `!` pair does,
    `*` matching any residue."""

    def __init__(self, text):
        self.text = text
        self.cut, self.never = [], []
        for raw in str(text).split(","):
            tok = raw.strip().upper()
            if not tok:
                continue
            neg = tok.startswith("!")
            pair = tok[1:] if neg else tok
            if len(pair) != 2 or any(not (c == "*" or "A" <= c <= "Z") for c in pair):
                raise SystemExit(f"cleavage rule pair {tok!r} must be two characters (residue "
                                 f"before and after the cut, '*' for any), optionally after '!'")
            side = tuple(None if c == "*" else c for c in pair)
            (self.never if neg else self.cut).append(side)
        if not self.cut:
            raise SystemExit(f"cleavage rule {text!r} has no pair that cuts")

    @staticmethod
    def _matches(pair, before, after):
        return (pair[0] is None or pair[0] == before) and (pair[1] is None or pair[1] == after)

    def cuts(self, before, after):
        return (any(self._matches(p, before, after) for p in self.cut)
                and not any(self._matches(p, before, after) for p in self.never))

    def sites(self, seq):
        """The engine's `CleavageRule::sites`: cut positions with both protein ends."""
        out = [0] + [i for i in range(1, len(seq)) if self.cuts(seq[i - 1], seq[i])]
        if out[-1] != len(seq):
            out.append(len(seq))
        return out


def cleavage_rule(enzyme="trypsin_p", rule_text=None):
    """The rule an occurrence must satisfy, or None for any occurrence (also for a rule that
    cuts every bond, `**`)."""
    if rule_text:
        rule = CleavageRule(rule_text)
        return None if (not rule.never and (None, None) in rule.cut) else rule
    if enzyme in (None, "any"):
        return None
    if enzyme not in ENZYME_RULES:
        raise SystemExit(f"--enzyme {enzyme!r}: not a preset ({', '.join(ENZYME_RULES)}, any)")
    return None if enzyme == "non_specific" else CleavageRule(ENZYME_RULES[enzyme])


def map_sequences(sequences, records, met_excision=True, rule=None):
    """Where each stripped sequence occurs in the FASTA, up to I/L.

    `sequences`: iterable of stripped sequences. `records`: [(key, sequence)] of the FASTA.
    Returns {sequence: [(record index, starts_protein)]}, the records in FASTA order, each
    once, `starts_protein` true when one of the occurrences in that record starts the
    protein (position 0, or position 1 after an initiator `M` when `met_excision`).

    With a cleavage `rule`, only a specific occurrence counts: it starts at a protein start
    (position 0, or 1 after an initiator `M`) or a cut of the rule, and ends at the protein
    end or a cut, which is what the engine's digest produces at some number of missed
    cleavages.

    Exhaustive and exact: every record is scanned once with a window of `k` residues, `k`
    the length of the shortest sequence (at most 7); a window that is the first `k`
    residues of some sequence is then compared in full. Every occurrence starts with a
    window the index holds, so none is missed.
    """
    seqs = sorted({s for s in sequences if s})
    if not seqs:
        return {}
    keys = {s: il_key(s) for s in seqs}
    k = max(1, min(7, min(len(s) for s in seqs)))
    by_prefix = {}
    for s in seqs:
        by_prefix.setdefault(keys[s][:k], []).append(s)
    hits = {}
    for ri, (_, protein) in enumerate(records):
        p = il_key(protein)
        n = len(p)
        if n < k:
            continue
        get = by_prefix.get
        for pos in range(n - k + 1):
            cands = get(p[pos:pos + k])
            if cands is None:
                continue
            for s in cands:
                if not p.startswith(keys[s], pos):
                    continue
                starts = pos == 0 or (met_excision and pos == 1 and protein[:1] == "M")
                if rule is not None:
                    end = pos + len(s)
                    if not (starts or rule.cuts(protein[pos - 1], protein[pos])):
                        continue
                    if end < n and not rule.cuts(protein[end - 1], protein[end]):
                        continue
                lst = hits.setdefault(s, [])
                if lst and lst[-1][0] == ri:
                    if starts and not lst[-1][1]:
                        lst[-1] = (ri, True)
                else:
                    lst.append((ri, starts))
    return hits


def target_proteins(peptidoforms, records, key_kind="header", n_term_mods=("Acetyl",),
                    met_excision=True, rule=None):
    """The new protein string of each target peptidoform, '' for one the FASTA does not
    contain (or, for a protein N-terminal group, does not contain at a protein start).

    Returns (proteins as a list aligned with `peptidoforms`, the number of distinct
    FASTA records any of them matched)."""
    pforms = [str(p) for p in peptidoforms]
    uniq = pd.unique(pd.Series(pforms, dtype=object))
    seq_of = {p: stripped(p) for p in uniq}
    hits = map_sequences(seq_of.values(), records, met_excision, rule)
    names = [protein_key(key, key_kind) for key, _ in records]
    n_term_mods = set(n_term_mods)
    cache = {}
    used = set()
    for p in uniq:
        occ = hits.get(seq_of[p], [])
        if n_term_group(p) in n_term_mods:
            occ = [o for o in occ if o[1]]
        joined = []
        for ri, _ in occ:
            name = names[ri]
            if name not in joined:
                joined.append(name)
            used.add(ri)
        cache[p] = ";".join(joined)
    return [cache[p] for p in pforms], len(used)


def annotation_table(records_path, key_kind):
    """The FASTA's annotations keyed by the protein name the remap writes."""
    rows = fasta_annotations(records_path)
    if key_kind == "header":
        return rows
    out = {}
    for key, row in rows.items():
        out.setdefault(protein_key(key, key_kind), row)
    return out


def remap_precursors(prec, fasta, unmapped="drop", key_kind="header", n_term_mods=("Acetyl",),
                     met_excision=True, rule=None, log=print):
    """`prec` (a precursor table, targets and optionally decoys) mapped to `fasta`.

    Returns (the remapped table, a boolean keep mask over the input rows, a summary dict).
    The returned table is `prec[keep]` with `protein` and the annotation columns rewritten;
    its `candidate_id` is not yet renumbered."""
    if unmapped not in ("drop", "keep"):
        raise ValueError(f"unmapped must be drop or keep, not {unmapped!r}")
    records = read_fasta(fasta)
    if not records:
        raise SystemExit(f"{fasta}: no FASTA records")
    label = prec["label"].astype(str).to_numpy()
    is_t = label == "target"
    is_d = label == "decoy"
    old_protein = prec["protein"].astype(str).to_numpy()
    t_idx = np.flatnonzero(is_t)
    new_t, n_records = target_proteins(prec["peptidoform"].to_numpy()[t_idx], records, key_kind,
                                       n_term_mods, met_excision, rule)
    new_t = np.asarray(new_t, dtype=object)
    mapped = new_t != ""
    protein = old_protein.astype(object).copy()
    keep = np.ones(len(prec), dtype=bool)
    if unmapped == "drop":
        keep[t_idx[~mapped]] = False
    new_t[~mapped] = UNASSIGNED
    protein[t_idx] = new_t

    # Decoys: their target's new protein, by (peptidoform_id, charge), else by base peptide.
    n_decoy_pair = n_decoy_base = n_decoy_orphan = 0
    d_idx = np.flatnonzero(is_d)
    if len(d_idx):
        pfid = prec["peptidoform_id"].astype(np.int64).to_numpy()
        charge = prec["charge"].astype(np.int64).to_numpy()
        base = prec["base_peptide_id"].astype(np.int64).to_numpy()
        by_pair = {}
        for i in t_idx:
            by_pair.setdefault((pfid[i], charge[i]), i)
        by_base = {}
        for i in t_idx:
            if keep[i]:
                lst = by_base.setdefault(base[i], [])
                for tok in str(protein[i]).split(";"):
                    if tok not in lst:
                        lst.append(tok)
        for i in d_idx:
            t = by_pair.get((pfid[i], charge[i]))
            if t is not None:
                n_decoy_pair += 1
                if keep[t]:
                    protein[i] = "DECOY_" + str(protein[t])
                else:
                    keep[i] = False
                continue
            toks = by_base.get(base[i])
            if toks:
                n_decoy_base += 1
                # A base peptide mapped somewhere and left UNASSIGNED elsewhere (a protein
                # N-terminal form) takes its mapped proteins.
                real = [t for t in toks if t != UNASSIGNED] or toks
                protein[i] = "DECOY_" + ";".join(real)
            else:
                n_decoy_orphan += 1
                keep[i] = False

    out = prec.loc[keep].copy()
    out["protein"] = pd.Series(protein[keep], index=out.index, dtype=object).astype(str)
    if out.empty or not (out["label"].astype(str) == "target").any():
        raise SystemExit(f"no target peptide of the library occurs in {fasta}; nothing to write")
    annotate_precursors(out, annotation_table(fasta, key_kind))

    changed = int((old_protein[t_idx][mapped] != new_t[mapped]).sum())
    shared = int(sum(1 for p in new_t[mapped] if ";" in p))
    summary = {
        "fasta": os.path.abspath(fasta),
        "fasta_records": len(records),
        "fasta_records_matched": int(n_records),
        "protein_key": key_kind,
        "cleavage_rule": rule.text if rule is not None else "any",
        "unmapped": unmapped,
        "targets_in": int(is_t.sum()),
        "targets_mapped": int(mapped.sum()),
        "targets_unmapped": int((~mapped).sum()),
        "targets_protein_changed": changed,
        "targets_shared": shared,
        "decoys_in": int(is_d.sum()),
        "decoys_paired_by_peptidoform": n_decoy_pair,
        "decoys_paired_by_base_peptide": n_decoy_base,
        "decoys_without_target": n_decoy_orphan,
        "rows_out": int(keep.sum()),
        "targets_out": int((out["label"].astype(str) == "target").sum()),
        "decoys_out": int((out["label"].astype(str) == "decoy").sum()),
    }
    log(f"remap: {summary['targets_mapped']} of {summary['targets_in']} target precursors map "
        f"to {summary['fasta_records_matched']} of {len(records)} proteins of {fasta} "
        f"({changed} with a changed protein, {shared} shared between proteins)")
    if summary["targets_in"] and summary["targets_mapped"] < 0.5 * summary["targets_in"]:
        log(f"remap: WARNING: fewer than half of the target precursors map under the cleavage "
            f"rule {summary['cleavage_rule']}; a library of another enzyme needs --enzyme (any "
            f"for a non-specific one)")
    if summary["targets_unmapped"]:
        what = "dropped with their decoys" if unmapped == "drop" else f"kept as {UNASSIGNED}"
        log(f"remap: {summary['targets_unmapped']} target precursors occur in no protein of the "
            f"FASTA; {what} (--unmapped)")
    if summary["decoys_in"]:
        log(f"remap: decoys take their target's proteins ({n_decoy_pair} paired by "
            f"peptidoform and charge, {n_decoy_base} by base peptide; {n_decoy_orphan} "
            f"without a kept target dropped)")
    log(f"remap: {summary['rows_out']} precursors out ({summary['targets_out']} targets, "
        f"{summary['decoys_out']} decoys)")
    return out, keep, summary


def _renumber(prec_in, keep):
    """Old candidate_id -> new contiguous id (-1 for a dropped row), as an array indexed by
    the old id."""
    old = prec_in["candidate_id"].astype(np.int64).to_numpy()
    lut = np.full(int(old.max()) + 1 if len(old) else 0, -1, dtype=np.int64)
    kept = old[keep]
    # Ids follow row order, so a kept row's new id is its rank among the kept rows.
    lut[kept] = np.arange(len(kept), dtype=np.int64)
    return lut


def rewrite_fragments(in_frag, out_frag, lut):
    """Stream the fragment table, keeping the rows of kept candidates with renumbered ids.
    Returns the number of rows written."""
    pf = pq.ParquetFile(in_frag)
    tmp = str(out_frag) + ".remap.tmp"
    schema = narrow_table(pf.schema_arrow.empty_table()).schema
    written = 0
    writer = pq.ParquetWriter(tmp, schema, compression="snappy")
    try:
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            cid = t.column("candidate_id").to_numpy().astype(np.int64)
            new = lut[cid]
            sel = new >= 0
            if not sel.any():
                continue
            t = t.filter(pa.array(sel))
            ci = t.schema.get_field_index("candidate_id")
            t = t.set_column(ci, t.schema.field(ci),
                             pa.array(new[sel].astype(np.uint32)).cast(t.schema.field(ci).type))
            writer.write_table(narrow_table(t).cast(schema))
            written += t.num_rows
    finally:
        writer.close()
        pf.close()
    os.replace(tmp, str(out_frag))
    return written


def remap_files(in_prec, in_frag, out_prec, out_frag, fasta, unmapped="drop",
                key_kind="header", n_term_mods=("Acetyl",), met_excision=True, rule=None,
                summary_path=None, log=print):
    """Remap the library at (in_prec, in_frag) to `fasta` and write it at (out_prec,
    out_frag). Returns the summary dict."""
    prec = pd.read_parquet(in_prec)
    for c in ("candidate_id", "peptidoform", "label", "protein"):
        if c not in prec.columns:
            raise SystemExit(f"{in_prec}: not a MuMDIA precursor table (no {c!r} column)")
    n = len(prec)
    cid = prec["candidate_id"].astype(np.int64).to_numpy()
    if not np.array_equal(cid, np.arange(n)):
        raise SystemExit(f"{in_prec}: candidate_id is not the row-aligned range 0..{n}")
    n_frag_in = pq.read_metadata(in_frag).num_rows
    prec, _ = embed_annotations(prec, in_prec)
    out, keep, summary = remap_precursors(prec, fasta, unmapped, key_kind, n_term_mods,
                                          met_excision, rule, log)
    lut = _renumber(prec, keep)
    out["candidate_id"] = lut[out["candidate_id"].astype(np.int64).to_numpy()].astype(
        prec["candidate_id"].dtype)
    out = out.reset_index(drop=True)
    n_frag = rewrite_fragments(in_frag, out_frag, lut)
    tmp = str(out_prec) + ".remap.tmp"
    write_engine_parquet(out, tmp)
    os.replace(tmp, str(out_prec))
    retire_side_table(in_prec, out_prec)
    summary["fragments_out"] = int(n_frag)
    if "n_fragments" in out.columns and int(out["n_fragments"].sum()) != n_frag:
        # Only a disagreement the input did not already have is a remap error.
        if int(prec["n_fragments"].sum()) == n_frag_in:
            raise RuntimeError(f"fragment rows: n_fragments sums to "
                               f"{int(out['n_fragments'].sum())}, {n_frag} written")
    log(f"remap: {n_frag} fragment rows -> {out_frag}")
    if summary_path:
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return summary


def parse_mods(value):
    return tuple(m.strip() for m in str(value).split(",") if m.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("in_precursors")
    ap.add_argument("in_fragments")
    ap.add_argument("out_precursors")
    ap.add_argument("out_fragments")
    ap.add_argument("--fasta", required=True, help="the FASTA to map the peptides to")
    ap.add_argument("--unmapped", choices=["drop", "keep"], default="drop",
                    help="peptides the FASTA does not contain: drop them with their decoys "
                    "(default) or keep them as UNASSIGNED")
    ap.add_argument("--protein-key", choices=["header", "accession", "entry_name"],
                    default="header", help="the protein name written: the header's first word "
                    "(default, as `mumdia digest`), or its accession or entry name field")
    ap.add_argument("--protein-n-term-mods", default="Acetyl",
                    help="N-terminal modification groups that need a protein N-terminus "
                    "(comma-separated; '' for none)")
    ap.add_argument("--no-met-excision", action="store_true",
                    help="a protein N-terminal group maps at position 0 only, not after an "
                    "initiator Met")
    ap.add_argument("--enzyme", default="trypsin_p",
                    help="only occurrences this cleavage preset can produce count (default "
                    "trypsin_p; trypsin, lys_c, ..., or any for every occurrence)")
    ap.add_argument("--cleavage-rule", default=None,
                    help="a residue-pair rule instead of a preset, e.g. 'K*,R*,!*P'")
    ap.add_argument("--summary", default=None, help="also write the counts as JSON here")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.fasta):
        raise SystemExit(f"--fasta {a.fasta}: no such file")
    remap_files(a.in_precursors, a.in_fragments, a.out_precursors, a.out_fragments, a.fasta,
                a.unmapped, a.protein_key, parse_mods(a.protein_n_term_mods),
                not a.no_met_excision, cleavage_rule(a.enzyme, a.cleavage_rule), a.summary)


if __name__ == "__main__":
    main(sys.argv[1:])
