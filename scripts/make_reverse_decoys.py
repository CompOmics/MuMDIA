"""Build reverse-sequence decoys from the target half of a library, with a
guaranteed no-overlap-with-targets invariant (clean-room; docs/13_sidecars.md).

Each target is reversed keeping the C-terminal residue fixed. Reversal preserves
residue composition, so the decoy keeps the target's precursor m/z + iRT
(co-isolates/co-elutes), while fragment m/z are the REAL b/y of the reversed
sequence (recomputed from residue masses) and intensities are copied ion-for-ion.

No-overlap invariant: a reversed sequence whose stripped form collides with ANY
real target stripped sequence (palindrome, or reverse == another target) is
re-scrambled with a stable per-peptide-seeded Fisher-Yates on the interior; if
still colliding after MAX_TRIES, its target/decoy precursor pair is dropped
together. A final assertion enforces
decoy_stripped ∩ target_stripped == {}. The decoy peptidoform is the reversed
sequence itself (label matches its fragments), not DECOY_<target>.

The m/z calculator is validated against the library's own target fragment m/z
before writing (aborts if the residue-mass model is inconsistent).

Usage: python make_reverse_decoys.py <in_prec> <in_frag> <out_prec> <out_frag>

The precursor table is held in memory (one row per target); the fragment table is
streamed one parquet row group at a time and the decoy fragment m/z are computed
from per-decoy cumulative residue masses with array lookups, so a library of
hundreds of millions of fragment rows is a matter of minutes and a few GB rather
than a per-fragment Python loop over a table that must fit in RAM.
"""
import sys, re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
# The engine rejects `large_string` parquet columns ("column 'peptidoform' is not
# utf8"), and `to_parquet` picks the width itself: pandas 3.x chooses the large
# variant, so this helper silently emitted libraries the engine would not load.
from _lib_io import narrow_table, sort_fragments_by_candidate, write_engine_parquet

RES = {
    'G':57.021463735,'A':71.037113805,'S':87.032028435,'P':97.052763875,'V':99.068413945,
    'T':101.047678505,'C':103.009184505,'L':113.084064015,'I':113.084064015,'N':114.042927470,
    'D':115.026943065,'Q':128.058577540,'K':128.094963050,'E':129.042593135,'M':131.040484645,
    'H':137.058911875,'F':147.068413945,'R':156.101111050,'Y':163.063328575,'W':186.079312980,
}
UNIMOD = {'Carbamidomethyl':57.021463735,'Oxidation':15.994914620,'Acetyl':42.010564684,
          'Phospho':79.966331090,'Deamidated':0.984016106,'Methyl':14.015650064,
          'Dimethyl':28.031300128,'Carbamyl':43.005813726}
WATER=18.010564684; PROTON=1.007276466812
TOK=re.compile(r'([A-Z])(\[[^\]]*\])?')
MAX_TRIES=30

def parse(pform):
    """peptidoform (DECOY_ stripped) -> list of (residue, modname|'') tokens."""
    pform = pform.replace('DECOY_','')
    return [(res, mod[1:-1] if mod else '') for res,mod in TOK.findall(pform)]

class UnknownMod(Exception):
    """A modification this script cannot assign a mass to."""


def mod_mass(name):
    """Monoisotopic delta for a modification token, or raise.

    Raising rather than returning 0.0 is the whole point. The previous behaviour was
    `except: d = 0.0`, so any modification outside the eight names in UNIMOD, and any
    bracket content that is not a bare number, silently became a MASSLESS modification.
    The decoy's fragment m/z were then computed for the wrong molecule, so those decoys
    could not match anything, and a decoy that cannot match does not compete: the
    target-decoy null loses exactly the peptides carrying that modification and the
    reported q-values are optimistic for them. Nothing in the output distinguishes such
    a decoy from a good one.

    The sampled calculator check in `main` does not cover this. It compares 500
    precursors at the 99th percentile, so a modification on a small fraction of the
    library keeps p99 under the 5 ppm abort threshold and passes.

    The engine's own parser already treats this as an error
    (`MassError::UnknownModification`, `mumdia-core/src/mass.rs`), and
    `import_diann_lib.py` drops precursors carrying modifications it does not map. This
    is the third implementation of the same decision and was the only one that guessed.
    """
    if not name:
        return 0.0
    d = UNIMOD.get(name)
    if d is not None:
        return d
    try:
        # A numeric delta, as ProForma writes it: `[+79.966331]`, `[-17.026549]`.
        return float(name.lstrip('+'))
    except ValueError:
        raise UnknownMod(name) from None

def tmass(tok): return RES[tok[0]] + mod_mass(tok[1])
def valid(toks):
    """Every residue has a mass AND every modification has a mass.

    The modification half was missing, so a peptidoform with an unmodellable mod was
    called valid and went on to be reversed with that mod silently weighing nothing.
    """
    if len(toks) == 0 or not all(r in RES for r, _ in toks):
        return False
    for _, m in toks:
        try:
            mod_mass(m)
        except UnknownMod:
            return False
    return True


def unknown_mods(toks):
    """The modification names in `toks` that have no mass, for reporting."""
    out = []
    for _, m in toks:
        try:
            mod_mass(m)
        except UnknownMod:
            out.append(m)
    return out
def stripped(toks): return ''.join(r for r,_ in toks)
def to_pform(toks): return ''.join(r + (f'[{m}]' if m else '') for r,m in toks)
def reverse_keep_cterm(t): return t[:-1][::-1] + t[-1:] if len(t) >= 2 else t[:]

def frag_mz(toks, ion, ordinal, z):
    if ion == 'b':
        s = sum(tmass(t) for t in toks[:ordinal])
    else:
        s = sum(tmass(t) for t in toks[len(toks)-ordinal:]) + WATER
    return (s + z*PROTON)/z

def splitmix(seed):
    x = seed & 0xFFFFFFFFFFFFFFFF
    while True:
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = x
        z = ((z ^ (z>>30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z>>27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        yield (z ^ (z>>31)) & 0xFFFFFFFFFFFFFFFF

def stable_seed(s):
    """Process-independent FNV-1a seed (unlike Python's randomized hash())."""
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h

def scramble(toks, gen):
    inter = toks[:-1][:]
    for i in range(len(inter)-1, 0, -1):
        j = next(gen) % (i+1)
        inter[i], inter[j] = inter[j], inter[i]
    return inter + toks[-1:]

def _cumulative_masses(toks_list, width):
    """Per peptide, the cumulative residue+modification mass after k residues, k = 0..L,
    zero-padded to `width` + 1 columns, plus each peptide's length. Feeds the
    vectorised b/y calculator: a b_k ion sums the first k residues, a y_k ion the last k."""
    n = len(toks_list)
    cum = np.zeros((n, width + 1), dtype=np.float64)
    length = np.zeros(n, dtype=np.int64)
    for i, toks in enumerate(toks_list):
        L = len(toks)
        length[i] = L
        acc = 0.0
        for k, tok in enumerate(toks):
            acc += tmass(tok)
            cum[i, k + 1] = acc
    return cum, length


def _fragment_mz(cum, length, rows, ion_is_b, ordinal, z):
    """Vectorised `frag_mz`: `rows` indexes peptides in `cum`, the rest are per-fragment
    arrays. Same arithmetic as the scalar version, including the ordinal clamp."""
    k = np.minimum(ordinal, length[rows])
    total = cum[rows, length[rows]]
    b = cum[rows, k]
    y = total - cum[rows, length[rows] - k] + WATER
    s = np.where(ion_is_b, b, y)
    return (s + z * PROTON) / z


def main():
    inp, inf, outp, outf = sys.argv[1:5]
    prec = pd.read_parquet(inp)
    tprec = prec[prec.label == 'target'].copy().reset_index(drop=True)
    tids = tprec.candidate_id.to_numpy().astype(np.int64)
    tgt_toks = {cid: parse(pf) for cid, pf in zip(tprec.candidate_id, tprec.peptidoform)}
    target_stripped = {stripped(t) for t in tgt_toks.values()}

    # Name the modifications this script cannot model, before anything is written.
    # `valid()` excludes these precursors below, so they get no decoy; without the
    # message the only symptom would be a smaller decoy population than target
    # population, which is easy to miss and biases every q-value that depends on it.
    unmapped = {}
    for cid, t in tgt_toks.items():
        for m in unknown_mods(t):
            unmapped[m] = unmapped.get(m, 0) + 1
    if unmapped:
        listed = ", ".join(f"{k!r} x{v}" for k, v in sorted(unmapped.items()))
        print(f"WARNING: {sum(unmapped.values())} precursors carry a modification with no "
              f"mass in this script's table and will get NO decoy: {listed}", flush=True)
        print("         Add the monoisotopic delta to UNIMOD in this file, or use a "
              "library whose modifications are written as numeric deltas.", flush=True)

    # --- validate m/z calculator against library target fragments ---
    # Up to 500 target precursors from the first row group of the fragment table.
    pf = pq.ParquetFile(inf)
    sample = pf.read_row_group(0).to_pandas()
    sample = sample[sample.candidate_id.isin(set(tids.tolist()))]
    first500 = set(sample.candidate_id.drop_duplicates().head(500).tolist())
    sample = sample[sample.candidate_id.isin(first500)]
    err = []
    for cid, g in sample.groupby('candidate_id', sort=True):
        t = tgt_toks.get(int(cid))
        if t is None or not valid(t):
            continue
        for x in g.itertuples():
            err.append(1e6 * abs(frag_mz(t, x.ion_type, int(x.ordinal), int(x.frag_charge)) - x.mz) / x.mz)
    err = np.array(err)
    if len(err) == 0:
        sys.exit("ABORT: no target fragments found to validate the m/z calculator against")
    p99 = np.percentile(err, 99)
    print(f"calculator vs library: median {np.median(err):.2f} ppm, 99th {p99:.2f} ppm, max {err.max():.2f} ppm", flush=True)
    if p99 > 5.0:
        sys.exit(f"ABORT: m/z calculator inconsistent (99th {p99:.1f} ppm > 5)")

    # --- reversed decoys with no-overlap invariant ---
    rev = {}; palin = 0; scr = 0; drop = 0; invalid = 0
    # Different target base sequences must not collapse onto the same decoy
    # sequence. Repeated charge/modification rows of one base sequence may reuse
    # that sequence, which preserves the precursor-level library structure.
    decoy_owner = {}
    for cid in sorted(tgt_toks):
        t = tgt_toks[cid]
        if not valid(t):
            rev[cid] = None; invalid += 1; continue
        source = stripped(t)
        gen = splitmix(stable_seed(source) ^ 0xD1CE)
        cand = reverse_keep_cterm(t); tries = 0
        def conflicts(sequence):
            owner = decoy_owner.get(sequence)
            return sequence in target_stripped or (owner is not None and owner != source)
        while conflicts(stripped(cand)) and tries < MAX_TRIES:
            if tries == 0: palin += 1
            cand = scramble(t, gen); tries += 1
        if conflicts(stripped(cand)):
            rev[cid] = None; drop += 1
        else:
            if tries > 0: scr += 1
            rev[cid] = cand
            decoy_owner.setdefault(stripped(cand), source)
    print(f"reverse: target-collisions={palin} resolved-by-scramble={scr} dropped={drop} skipped-nonstd={invalid}", flush=True)

    off = int(tprec.candidate_id.max()) + 1
    keep = [cid for cid in tprec.candidate_id if rev[cid] is not None]
    keepset = set(keep)
    # Keep target and decoy populations paired. Retaining a target whose decoy
    # could not be generated biases the null, even when the unresolved set is
    # small, so remove its target precursor and fragments too.
    tprec_out = tprec[tprec.candidate_id.isin(keepset)].copy()
    dprec = tprec_out.copy()
    dprec['candidate_id'] = dprec['candidate_id'] + off
    dprec['label'] = 'decoy'
    dprec['protein'] = 'DECOY_' + dprec['protein'].astype(str)
    dprec['peptidoform'] = dprec['candidate_id'].map(lambda nc: 'DECOY_' + to_pform(rev[nc - off]))

    allp = pd.concat([tprec_out, dprec], ignore_index=True).sort_values('precursor_mz', kind='mergesort').reset_index(drop=True)
    # old candidate_id -> new, as an array: the ids are 0..2*off, so a lookup table
    # replaces the dict and lets every fragment batch be remapped in one indexing op.
    o2n = np.full(2 * off + 1, -1, dtype=np.int64)
    o2n[allp['candidate_id'].to_numpy().astype(np.int64)] = np.arange(len(allp), dtype=np.int64)
    allp['candidate_id'] = np.arange(len(allp), dtype=np.uint32)
    for c, dt in [('peptidoform_id', 'uint32'), ('base_peptide_id', 'uint32'), ('charge', 'int32'), ('predicted_irt', 'float32'), ('n_fragments', 'int32')]:
        if c in allp.columns: allp[c] = allp[c].astype(dt)

    # Per-decoy cumulative masses, indexed by the TARGET's old candidate_id.
    kept_mask = np.zeros(off, dtype=bool)
    kept_mask[np.array(keep, dtype=np.int64)] = True
    rev_toks = [rev[cid] for cid in keep]
    width = max((len(t) for t in rev_toks), default=0)
    cum, length = _cumulative_masses(rev_toks, width)
    row_of = np.full(off, -1, dtype=np.int64)
    row_of[np.array(keep, dtype=np.int64)] = np.arange(len(keep), dtype=np.int64)

    dstr = {stripped(parse(s)) for s in allp[allp.label == 'decoy'].peptidoform}
    ov = dstr & target_stripped
    print(f"FINAL overlap decoy-vs-target stripped = {len(ov)} (must be 0)", flush=True)
    assert len(ov) == 0, f"overlap invariant violated: {len(ov)}"
    paired = {stripped(tgt_toks[cid]): stripped(rev[cid]) for cid in keep}
    assert len(set(paired.values())) == len(paired), "distinct targets share a decoy sequence"

    write_engine_parquet(allp, outp)

    # --- fragments: one row group in, target rows plus their decoy twins out ---
    frag_schema = pa.schema([
        pa.field("candidate_id", pa.uint32(), False),
        pa.field("mz", pa.float64(), False),
        pa.field("predicted_intensity", pa.float32(), False),
        pa.field("name", pa.string(), False),
        pa.field("ion_type", pa.string(), False),
        pa.field("ordinal", pa.int32(), False),
        pa.field("frag_charge", pa.int32(), False),
        pa.field("cardinality", pa.int32(), False),
    ])
    n_out = 0
    writer = pq.ParquetWriter(str(outf), frag_schema, compression="snappy")
    try:
        for rg in range(pf.num_row_groups):
            g = pf.read_row_group(rg).to_pandas()
            old = g['candidate_id'].to_numpy().astype(np.int64)
            sel = (old < off) & kept_mask[np.minimum(old, off - 1)]
            g = g[sel]
            if not len(g):
                continue
            old = g['candidate_id'].to_numpy().astype(np.int64)
            rows = row_of[old]
            ion_is_b = (g['ion_type'].astype(str).to_numpy() == 'b')
            ordinal = g['ordinal'].to_numpy().astype(np.int64)
            z = g['frag_charge'].to_numpy().astype(np.float64)
            dmz = _fragment_mz(cum, length, rows, ion_is_b, ordinal, z)
            tgt_new = o2n[old]
            dec_new = o2n[old + off]
            assert (tgt_new >= 0).all() and (dec_new >= 0).all()
            cid = np.concatenate([tgt_new, dec_new]).astype(np.uint32)
            mz = np.concatenate([g['mz'].to_numpy(dtype=np.float64), dmz])
            def twice(col, dtype):
                v = g[col].to_numpy()
                return np.concatenate([v, v]).astype(dtype)
            table = pa.table({
                "candidate_id": pa.array(cid, pa.uint32()),
                "mz": pa.array(mz, pa.float64()),
                "predicted_intensity": pa.array(twice('predicted_intensity', np.float32), pa.float32()),
                "name": pa.array(twice('name', object).astype(str), pa.string()),
                "ion_type": pa.array(twice('ion_type', object).astype(str), pa.string()),
                "ordinal": pa.array(twice('ordinal', np.int32), pa.int32()),
                "frag_charge": pa.array(twice('frag_charge', np.int32), pa.int32()),
                "cardinality": pa.array(twice('cardinality', np.int32), pa.int32()),
            }, schema=frag_schema)
            writer.write_table(narrow_table(table))
            n_out += table.num_rows
    finally:
        writer.close()
    # Targets were streamed first and decoys after, while the precursor order interleaves
    # them; restore candidate order so the engine can read one id range of the table.
    sort_fragments_by_candidate(outf)

    print(f"targets_in={len(tprec)} targets_out={len(tprec_out)} decoys={len(dprec)} total_prec={len(allp)} total_frag={n_out}", flush=True)


if __name__ == '__main__':
    main()
