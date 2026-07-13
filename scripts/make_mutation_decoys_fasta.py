"""Generate DIA-NN-style MUTATION decoys as a FASTA for DIA-NN to predict, so the
decoy fragments are predicted by the SAME model as the targets (a valid target-decoy
null that experiences the same chimeric interference). Reimplements the mutation-decoy
concept from DIA-NN (diann.cpp, CC BY 4.0, Demichev et al.); the substitution map is
independently chosen, not copied.

Each interior residue is substituted via a fixed derangement whose images never
contain K or R, so interior tryptic sites are removed and each decoy peptide, written
as a one-peptide protein, digests back to exactly itself (C-term K/R kept). Terminal
residues are preserved. Decoys equal to a real target are re-mutated.

Usage: python make_mutation_decoys_fasta.py <target_lib_precursors.parquet> <out.fasta>
"""
import re
import sys
import pandas as pd

# derangement over the 20 AAs; no image is K or R (keeps interior non-tryptic)
MUT = {'A': 'L', 'R': 'N', 'N': 'D', 'D': 'E', 'C': 'S', 'E': 'G', 'Q': 'A',
       'G': 'V', 'H': 'Y', 'I': 'M', 'L': 'F', 'K': 'Q', 'M': 'T', 'F': 'W',
       'P': 'S', 'S': 'T', 'T': 'V', 'W': 'Y', 'Y': 'H', 'V': 'I'}
STD = set("ACDEFGHIKLMNPQRSTVWY")
strip = lambda pf: re.sub(r"\[[^\]]*\]", "", str(pf).replace("DECOY_", ""))


def mutate(seq):
    if len(seq) < 3:
        return None
    interior = [MUT.get(c, c) for c in seq[1:-1]]
    return seq[0] + "".join(interior) + seq[-1]


def main():
    lib_in, out_fasta = sys.argv[1:3]
    lib = pd.read_parquet(lib_in, columns=["peptidoform"])
    seqs = sorted({strip(p) for p in lib.peptidoform})
    seqs = [s for s in seqs if len(s) >= 5 and all(c in STD for c in s)]
    target_set = set(seqs)
    print(f"unique target sequences: {len(seqs)}", flush=True)

    out, made, collide = [], 0, 0
    for i, s in enumerate(seqs):
        d = mutate(s)
        if d is None:
            continue
        if d in target_set:
            # second-pass: also mutate the first residue to break the collision
            d = MUT.get(s[0], s[0]) + d[1:]
            collide += 1
            if d in target_set:
                continue
        out.append(f">DECOY_{i}\n{d}")
        made += 1
    with open(out_fasta, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {made} decoy proteins ({collide} needed collision fix) -> {out_fasta}")


if __name__ == "__main__":
    main()
