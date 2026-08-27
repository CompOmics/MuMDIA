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
"""
import sys, re
import numpy as np
import pandas as pd
# The engine rejects `large_string` parquet columns ("column 'peptidoform' is not
# utf8"), and `to_parquet` picks the width itself: pandas 3.x chooses the large
# variant, so this helper silently emitted libraries the engine would not load.
from _lib_io import write_engine_parquet

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

def mod_mass(name):
    if not name: return 0.0
    d = UNIMOD.get(name)
    if d is None:
        try: d = float(name.lstrip('+'))
        except: d = 0.0
    return d

def tmass(tok): return RES[tok[0]] + mod_mass(tok[1])
def valid(toks): return len(toks) > 0 and all(r in RES for r,_ in toks)
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

def main():
    inp,inf,outp,outf = sys.argv[1:5]
    prec = pd.read_parquet(inp); frag = pd.read_parquet(inf)
    tprec = prec[prec.label=='target'].copy().reset_index(drop=True)
    tids = set(tprec.candidate_id); tfrag = frag[frag.candidate_id.isin(tids)].copy()
    tgt_toks = {cid: parse(pf) for cid,pf in zip(tprec.candidate_id, tprec.peptidoform)}
    target_stripped = {stripped(t) for t in tgt_toks.values()}

    # --- validate m/z calculator against library target fragments ---
    fg = {cid:g for cid,g in tfrag.groupby('candidate_id')}
    err=[]
    for r in tprec.head(500).itertuples():
        t=tgt_toks[r.candidate_id]; g=fg.get(r.candidate_id)
        if g is None or not valid(t): continue
        for x in g.itertuples():
            err.append(1e6*abs(frag_mz(t,x.ion_type,int(x.ordinal),int(x.frag_charge))-x.mz)/x.mz)
    err=np.array(err); p99=np.percentile(err,99)
    print(f"calculator vs library: median {np.median(err):.2f} ppm, 99th {p99:.2f} ppm, max {err.max():.2f} ppm",flush=True)
    if p99>5.0: sys.exit(f"ABORT: m/z calculator inconsistent (99th {p99:.1f} ppm > 5)")

    # --- reversed decoys with no-overlap invariant ---
    rev={}; palin=0; scr=0; drop=0; invalid=0
    # Different target base sequences must not collapse onto the same decoy
    # sequence. Repeated charge/modification rows of one base sequence may reuse
    # that sequence, which preserves the precursor-level library structure.
    decoy_owner={}
    for cid in sorted(tgt_toks):
        t=tgt_toks[cid]
        if not valid(t): rev[cid]=None; invalid+=1; continue
        source=stripped(t)
        gen=splitmix(stable_seed(source) ^ 0xD1CE)
        cand=reverse_keep_cterm(t); tries=0
        def conflicts(sequence):
            owner=decoy_owner.get(sequence)
            return sequence in target_stripped or (owner is not None and owner != source)
        while conflicts(stripped(cand)) and tries<MAX_TRIES:
            if tries==0: palin+=1
            cand=scramble(t,gen); tries+=1
        if conflicts(stripped(cand)): rev[cid]=None; drop+=1
        else:
            if tries>0: scr+=1
            rev[cid]=cand
            decoy_owner.setdefault(stripped(cand), source)
    print(f"reverse: target-collisions={palin} resolved-by-scramble={scr} dropped={drop} skipped-nonstd={invalid}",flush=True)

    off=int(tprec.candidate_id.max())+1
    keep=[cid for cid in tprec.candidate_id if rev[cid] is not None]
    keepset=set(keep)
    # Keep target and decoy populations paired. Retaining a target whose decoy
    # could not be generated biases the null, even when the unresolved set is
    # small, so remove its target precursor and fragments too.
    tprec_out=tprec[tprec.candidate_id.isin(keepset)].copy()
    tfrag_out=tfrag[tfrag.candidate_id.isin(keepset)].copy()
    dprec=tprec_out.copy()
    dprec['candidate_id']=dprec['candidate_id']+off
    dprec['label']='decoy'
    dprec['protein']='DECOY_'+dprec['protein'].astype(str)
    dprec['peptidoform']=dprec['candidate_id'].map(lambda nc: 'DECOY_'+to_pform(rev[nc-off]))

    df=tfrag_out.copy()
    df['candidate_id']=df['candidate_id']+off
    mz=np.empty(len(df))
    ci=(df['candidate_id']-off).to_numpy(); ion=df['ion_type'].to_numpy(); ordn=df['ordinal'].to_numpy(); ch=df['frag_charge'].to_numpy()
    for i in range(len(df)):
        t=rev[ci[i]]; mz[i]=frag_mz(t,ion[i],int(ordn[i]),int(ch[i]))
    df['mz']=mz

    allp=pd.concat([tprec_out,dprec],ignore_index=True).sort_values('precursor_mz',kind='mergesort').reset_index(drop=True)
    o2n={o:n for n,o in enumerate(allp['candidate_id'].tolist())}
    allp['candidate_id']=np.arange(len(allp),dtype=np.uint32)
    allf=pd.concat([tfrag_out,df],ignore_index=True)
    allf['candidate_id']=allf['candidate_id'].map(o2n).astype(np.uint32)
    allf=allf.sort_values('candidate_id',kind='mergesort').reset_index(drop=True)
    for c,dt in [('peptidoform_id','uint32'),('base_peptide_id','uint32'),('charge','int32'),('predicted_irt','float32'),('n_fragments','int32')]:
        if c in allp.columns: allp[c]=allp[c].astype(dt)
    allf['predicted_intensity']=allf['predicted_intensity'].astype('float32'); allf['ordinal']=allf['ordinal'].astype('int32'); allf['frag_charge']=allf['frag_charge'].astype('int32')

    dstr={stripped(parse(s)) for s in allp[allp.label=='decoy'].peptidoform}
    ov=dstr & target_stripped
    print(f"FINAL overlap decoy-vs-target stripped = {len(ov)} (must be 0)",flush=True)
    assert len(ov)==0, f"overlap invariant violated: {len(ov)}"
    paired={stripped(tgt_toks[cid]): stripped(rev[cid]) for cid in keep}
    assert len(set(paired.values())) == len(paired), "distinct targets share a decoy sequence"

    write_engine_parquet(allp, outp); write_engine_parquet(allf, outf)
    print(f"targets_in={len(tprec)} targets_out={len(tprec_out)} decoys={len(dprec)} total_prec={len(allp)} total_frag={len(allf)}",flush=True)

if __name__=='__main__':
    main()
