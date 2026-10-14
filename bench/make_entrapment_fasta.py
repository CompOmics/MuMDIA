#!/usr/bin/env python
"""Build a real + 1:1 foreign entrapment FASTA, and measure the entrapment ratio of the
library built from it (bench/README.md, "FDR validity: entrapment").

Build the FASTA:

  make_entrapment_fasta.py --real fasta/ecoli_22032024.fasta --foreign human.fasta \
      --out ecoli_human_entrap.fasta [--seed 20260828]

Every real entry's identifier is prefixed `REAL_` and a seeded sample of foreign entries,
as many as the real FASTA has, is prefixed `ENTRAP_`. The prefixes, not species suffixes,
carry the classification: a real FASTA can bundle `_HUMAN` contaminants. Declare
`rescore.entrapment_marker = "ENTRAP_"` and `rescore.entrapment_exclude = "REAL_"` (a peptide
shared with a real protein is real).

Measure the ratio, once per built library:

  make_entrapment_fasta.py --ratio-library out_lib/fragment_library_precursors.parquet \
      [--marker ENTRAP_] [--exclude REAL_] [--contaminant KRT --contaminant K1C ...]

`rescore.entrapment_ratio` is N_real / N_entrap over the library's TARGET candidates, with
each target classified as the engine does (`rescore.rs::classify_entrapment`): entrapment
when its protein contains the marker, does not contain the exclude string and contains no
contaminant marker; real otherwise. Pass the marker, exclude and contaminant settings of the
run config. The ratio is a property of the library, not of the FASTA alone: a 1:1 protein
count does not give a 1:1 candidate count, and the digest, the modifications and the charges
all change it. Put the printed value in the run config and pass the same value to
bench/entrapment_fdp.py (which reads it from that config) or to `fs_lib.entrapment_fdp`.
"""
import argparse
import random

DEFAULT_CONTAMINANTS = ("KRT", "K1C", "K2C", "ALBU", "TRYP")


def read_fasta(path):
    entries, head, seq = [], None, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if head is not None:
                    entries.append((head, "".join(seq)))
                head, seq = line[1:], []
            elif line:
                seq.append(line.strip())
    if head is not None:
        entries.append((head, "".join(seq)))
    return entries


def build(real_path, foreign_path, out_path, seed):
    real = read_fasta(real_path)
    foreign = read_fasta(foreign_path)
    if len(foreign) < len(real):
        raise SystemExit(f"{foreign_path} has {len(foreign)} entries, fewer than the {len(real)} real ones")
    pick = random.Random(seed).sample(foreign, len(real))
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        for prefix, entries in (("REAL_", real), ("ENTRAP_", pick)):
            for head, seq in entries:
                fh.write(f">{prefix}{head}\n")
                for i in range(0, len(seq), 60):
                    fh.write(seq[i : i + 60] + "\n")
    print(f"wrote {out_path}: {len(real)} real + {len(pick)} entrapment entries (seed {seed})")


def is_entrapment(protein, marker, exclude, contaminants):
    """The engine's rule for one target's protein string."""
    return (marker in protein
            and (not exclude or exclude not in protein)
            and not any(c in protein for c in contaminants))


def ratio(library, marker, exclude, contaminants):
    """(n_real, n_entrap) over the target candidates of a fragment_library_precursors parquet."""
    import pyarrow.parquet as pq

    t = pq.read_table(library, columns=["label", "protein"])
    n_real = n_entrap = 0
    for label, protein in zip(t.column("label").to_pylist(), t.column("protein").to_pylist()):
        if label != "target":
            continue
        if is_entrapment(str(protein), marker, exclude, contaminants):
            n_entrap += 1
        else:
            n_real += 1
    return n_real, n_entrap


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real", help="FASTA of the sample's own proteome")
    ap.add_argument("--foreign", help="FASTA to sample the entrapment entries from")
    ap.add_argument("--out", help="entrapment FASTA to write")
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--ratio-library", help="fragment_library_precursors.parquet: print the ratio and exit")
    ap.add_argument("--marker", default="ENTRAP_")
    ap.add_argument("--exclude", default="REAL_")
    ap.add_argument("--contaminant", action="append",
                    help=f"contaminant marker, repeatable (default {' '.join(DEFAULT_CONTAMINANTS)})")
    a = ap.parse_args()
    if a.ratio_library:
        cont = tuple(a.contaminant) if a.contaminant else DEFAULT_CONTAMINANTS
        n_real, n_entrap = ratio(a.ratio_library, a.marker, a.exclude, cont)
        if n_entrap == 0:
            raise SystemExit(f"{a.ratio_library}: no target candidate matches marker {a.marker!r}")
        print(f"{a.ratio_library}: {n_real} real / {n_entrap} entrapment target candidates; "
              f"entrapment_ratio = {n_real / n_entrap:.6f}")
        return
    if not (a.real and a.foreign and a.out):
        ap.error("building a FASTA needs --real, --foreign and --out (or pass --ratio-library)")
    build(a.real, a.foreign, a.out, a.seed)


if __name__ == "__main__":
    assert is_entrapment("ENTRAP_sp|P1|X_HUMAN", "ENTRAP_", "REAL_", DEFAULT_CONTAMINANTS)
    assert not is_entrapment("ENTRAP_sp|P1|X_HUMAN;REAL_sp|P2|Y_ECOLI", "ENTRAP_", "REAL_", ())
    assert not is_entrapment("ENTRAP_sp|P04264|K2C1_HUMAN", "ENTRAP_", "REAL_", DEFAULT_CONTAMINANTS)
    main()
