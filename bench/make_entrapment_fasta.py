#!/usr/bin/env python
"""Build a real + 1:1 foreign entrapment FASTA (bench/README.md, "FDR validity: entrapment").

  make_entrapment_fasta.py --real fasta/ecoli_22032024.fasta --foreign human.fasta \
      --out ecoli_human_entrap.fasta [--seed 20260828]

Every real entry's identifier is prefixed `REAL_` and a seeded sample of foreign entries,
as many as the real FASTA has, is prefixed `ENTRAP_`. The prefixes, not species suffixes,
carry the classification: a real FASTA can bundle `_HUMAN` contaminants. Declare
`entrapment_marker = "ENTRAP_"` and `entrapment_exclude = "REAL_"` (a peptide shared with a
real protein is real), and measure `entrapment_ratio` from the built library.
"""
import argparse
import random


def read_fasta(path):
    entries, head, seq = [], None, []
    for line in open(path, encoding="utf-8", errors="replace"):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--foreign", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260828)
    a = ap.parse_args()
    real = read_fasta(a.real)
    foreign = read_fasta(a.foreign)
    pick = random.Random(a.seed).sample(foreign, len(real))
    with open(a.out, "w") as fh:
        for prefix, entries in (("REAL_", real), ("ENTRAP_", pick)):
            for head, seq in entries:
                fh.write(f">{prefix}{head}\n")
                for i in range(0, len(seq), 60):
                    fh.write(seq[i : i + 60] + "\n")
    print(f"wrote {a.out}: {len(real)} real + {len(pick)} entrapment entries (seed {a.seed})")


if __name__ == "__main__":
    main()
