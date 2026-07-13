import sys, re, numpy as np, pandas as pd

def auc(y, x):
    # rank-based AUC (Mann-Whitney), direction-agnostic
    x = np.nan_to_num(x.astype(float))
    if x.std() == 0: return 0.5
    order = x.argsort()
    ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x)+1)
    # average ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    # simpler: use scipy-free tie-aware rank via pandas
    ranks = pd.Series(x).rank().values
    npos = y.sum(); nneg = len(y)-npos
    if npos==0 or nneg==0: return 0.5
    a = (ranks[y==1].sum() - npos*(npos+1)/2) / (npos*nneg)
    return max(a, 1-a)

def strip(pf): return re.sub(r'\[[^\]]*\]','',str(pf))
dn=pd.read_csv("out_diann/report.tsv",sep="\t",usecols=['Q.Value','Stripped.Sequence'])
diann=set(dn[dn['Q.Value']<=0.01]['Stripped.Sequence'].unique())
c=pd.read_parquet(sys.argv[1]); c=c[c.label=='target'].copy()
c['human']=c.protein.str.contains("_HUMAN")&~c.protein.str.contains("_ECOLI")
c['strip']=c.peptidoform.map(strip)
c['tp']=(~c.human)&c.strip.isin(diann)
lab=c[c.tp|c.human].copy(); y=lab.tp.astype(int).values
skip={'candidate_id','label','base_peptide_id','peptidoform','protein','apex_rt','precursor_mz','prelim_score','human','strip','tp'}
feats=[x for x in c.columns if x not in skip]
print(f"pos(real ecoli in DIA-NN)={int(y.sum())}  neg(human false)={int((y==0).sum())}",flush=True)
rows=[(f, auc(y, pd.to_numeric(lab[f],errors='coerce').values)) for f in feats]
rows.sort(key=lambda z:-z[1])
for f,a in rows: print(f"AUC {f:24s} {a:.3f}",flush=True)
