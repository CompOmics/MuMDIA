# 36. Modification site localisation (`localise`)

Experimental and opt-in (`localise.enabled`, default false). The stage is reporting only: it
reads the final scored table and the spectra and writes `localisation.parquet`, three columns of
`peptides.tsv` and `sites.tsv`. Identifications, every q-value of the scored table, the
quantities and every other artifact are the same with the stage on or off, and a configuration
that does not set the section keeps its config hash. Code: `rust/mumdia/crates/mumdia/src/stages/localise/`.

## 1. The problem

A modification search lists every placement of a variable modification as its own peptidoform.
Positional isomers (the same residues, charge and modification composition, differing only in
where the modification sits) are separate candidates. They do not compete with each other:
under `compete.group_by = peptidoform_charge` each competes only against its own alternative
peaks (`compete.rs`), and shadow demotion exempts its own base peptide (`shadow.rs`). Several
isomers are therefore accepted on one chromatographic peak, each with its own q-value, and the
peptidoform-level q says nothing about which site is right.

Measured on the twelve-run Arabidopsis phospho diaPASEF experiment (section 7): 60,923
accepted phospho precursor rows per run on `run_psm_q` fall into 29,103 isomer groups; 52% of
the rows are not the best-scoring isomer of their group, and 82% of those have their apex within
3 s of the best one. They are mostly the same signal assigned to several sites.

## 2. Method

Clean-room, from published concepts only: site-determining ions (fragments whose m/z differs
between placements), a likelihood over the placement hypotheses, and a decoy amino acid for an
empirical false localisation rate. No external localisation tool's code or constants were
consulted.

1. **Isomer groups.** The scored rows that carry a localised modification (`localise.mods`,
   default `["Phospho"]`) are grouped by (source, base peptide, modification composition,
   label). The composition key is the sorted list of mass deltas, terminal ones included
   (`hypotheses::composition_key`); the prescreen's sibling key, which adds charge and label,
   lives in the same module (`hypotheses::sibling_key`).
2. **Peaks.** Within a group and charge, the accepted rows (`run_psm_q <= localise.q_max`,
   default 0.01) are sorted by apex retention time and cut into peaks: a row joins the current
   peak when its apex lies within `localise.peak_rt_tol_s` (5 s) of the peak's first apex. The
   isomers of a peak share one localisation, computed on its representative (the highest
   score). Isomers further apart are separate peaks, each localised on its own.
3. **Hypotheses.** Every placement of the representative's count of the modification on its
   candidate residues, in lexicographic order (`hypotheses::enumerate`). The candidate residues
   are those `peptidoforms.variable_mods` gives the modification, or, without such an entry (an
   imported library), the residues it is found on in the scored table. Other modifications stay
   where the identification put them, and a residue that carries one is not a candidate.
   Placements on the decoy residue (`localise.decoy_residue`, default `A`) are added; they are
   never searched. A peptide with more than `localise.max_hypotheses` (200) placements is not
   localised. A peptide with a single real placement is `unambiguous`: its site is fixed by the
   sequence.
4. **Ions.** For every hypothesis, the b1 to b(n-1) and y1 to y(n-1) ions at fragment charge 1,
   and 2 for precursors of charge 3 or more (`ions::build`). One ion is one distinct m/z over all
   hypotheses (1e-4 Th), with the set of hypotheses that predict it. An ion every hypothesis
   predicts is common; one only some predict is site-determining. A position's modification
   count that no hypothesis gives it (b2 carrying the site when neither residue is a candidate)
   is an impossible variant: never a fragment of the peptide under any placement.
5. **Evidence.** The spectra are read again (`xic`): the MS2 scans of the isolation windows that
   hold the precursor m/z (on diaPASEF, the slots whose 1/K0 range, widened by 0.02, holds the
   apex 1/K0), pooled in retention-time order, and the `localise.apex_scans` (2) scans either
   side of the one nearest the apex. An ion is present when any peak of those scans lies within
   the fragment tolerance of its calibrated m/z (the seed's `masscal.json` offset and tolerance,
   else `extract.frag_tol_ppm`) and, with 1/K0, within `localise.im_half_width` (0.05) of the
   apex 1/K0. The apex 1/K0 comes from `psms_extracted` (the selected peak), or, when that is
   not given, the intensity-weighted median 1/K0 of the peaks matching the common ions.
6. **Likelihood.** Four ion classes: b and y at charge 1 and 2. Per class, the detection
   probability `d` is the share of the peak's common ions that are present and the
   false-presence probability `f` the share of its impossible variants that are present, each
   shrunk towards the run-wide share (Laplace-smoothed over the target peaks) by
   `localise.prior_strength` (8) pseudo-counts (`model::Counts::shrunk`). Under a hypothesis a
   site-determining ion it predicts is present with probability `d` and one it does not predict
   with probability `f`; the log-likelihood is the sum over the site-determining ions. The
   posterior is the softmax of the log-likelihoods divided by `localise.temperature` (1), with a
   flat prior. A site's probability is the posterior mass of the hypotheses that use it. The
   best hypothesis is the highest posterior, ties to the lexicographically first placement, so
   the answer does not depend on the order the hypotheses were listed in.
7. **Calls.** The peak is `localised` when the best hypothesis uses no decoy residue and every
   one of its sites reaches `localise.threshold` (0.75); `decoy_residue` when the best uses the
   decoy residue; `ambiguous` otherwise.

Why the false presence comes from impossible variants and not from random m/z: in the
prototype a random probe m/z (shifted by a non-integer offset) was present in 0.4% of cases,
while an impossible variant was present in 14% (b ions lacking a site they must carry 25%, y
ions carrying a site they cannot carry 13%). Site-determining m/z are not random m/z: neutral
loss of HPO3 from a modified fragment lands on the unmodified variant, co-isolated forms with
another phospho count produce the other variants, and the fragment ladders of co-eluting
peptides are dense near them. With the random-probe rate the model was overconfident by a
factor of 2 to 20 against the decoy residue (prototype, every arm); with the impossible-variant
rate it agrees with it without any temperature (section 7).

## 3. False localisation rates

Two estimates per peak, over the target peaks with a real choice (at least two placements off
the decoy residue), at the peak's own minimum site probability `p` (`flr_curve`):

- **model**: the mean of `1 - p` over the real localisations at or above `p`;
- **decoy residue**: the decoy-residue wins at or above `p`, times the ratio of candidate to
  decoy-residue positions over those peaks, divided by the real localisations at or above `p`.
  The decoy residue is never searched, so a placement on it is wrong by construction; the ratio
  scales its wins to the number of positions a wrong call could land on.

Both are made monotone in `p` the way q-values are. They are site-level rates over peaks, not
q-values of identifications.

## 4. `isomer_group_q`

A picked target-decoy q over isomer groups (`rescore::grouped_q` on the key (source, base
peptide, composition), the score the group's best row), written to the group's winning row and
1.0 to the others, like the grouped q columns of `rescore` (docs/11). A target and its paired
decoy share base peptide and composition, so they compete. The population is the rows that carry
the localised modification, so it is also a modification-subset FDR, which the pooled q-values
are not (on the Arabidopsis first pass the pooled 1% held a phospho-subset decoy fraction of 1.9
to 2.6%). Count modified peptides on it, one per group: a phosphopeptide with n sites, whatever
the placement.

## 5. Outputs

`localisation.parquet` (artifact `localisation` v1), one row per scored row of the run that
carries a localised modification, in scored-table order:

| column | type | meaning |
|---|---|---|
| `candidate_id`, `source`, `peptidoform`, `charge`, `label`, `protein`, `apex_rt`, `score`, `run_psm_q` | | copied from the scored row |
| `localised_mod` | utf8 | the modification localised |
| `isomer_group_id` | u32 | the row's isomer group (dense within the table, target and decoy groups separate) |
| `isomer_group_q` | f64 | section 4; on the group's winning row, 1.0 on the others |
| `isomer_peak_id` | i32, nullable | the row's peak; null when the row was not attempted |
| `is_peak_representative` | bool | the row the peak was localised on; count peaks on it |
| `n_hypotheses` | u32 | placements scored, decoy residue included |
| `best_site_peptidoform` | utf8, nullable | the best placement, other modifications as on the representative |
| `site_probabilities` | utf8, nullable | `S3:0.9120;T5:0.0710;A6:0.0170`: every candidate position (1-based) with its probability |
| `min_site_prob` | f64, nullable | the smallest site probability of the best placement |
| `peptidoform_probability` | f64, nullable | the posterior of the row's own placement |
| `own_site_prob` | f64, nullable | the smallest site probability of the row's own sites |
| `localised` | bool | the peak is `localised` or `unambiguous` |
| `on_decoy_residue` | bool | the best placement uses the decoy residue |
| `n_sdi`, `n_sdi_observed` | u32 | site-determining ions, and how many were present |
| `delta_loglik` | f64, nullable | best minus second-best log-likelihood |
| `site_flr_model`, `site_flr_decoy` | f64, nullable | section 3, at the peak's `min_site_prob` |
| `status` | utf8 | `localised`, `ambiguous`, `decoy_residue`, `unambiguous`, `not_attempted_q`, `too_many_hypotheses`, `no_spectra` |

The report (`run`, `run-experiment`, or `mumdia report --localisation` / `--experiment-dir`)
appends `site_probability` (the row's `own_site_prob`), `localised` (the row's own placement is
the peak's localised best) and `best_site_peptidoform` to `peptides.tsv`. In an experiment the
three cells come from the run of the row the precursor is reported on, its experiment-wide
winner; experiment-level combination of the runs is not done. `sites.tsv` lists every site of
a localised target peak representative: protein, protein position (from the FASTA of a FASTA
run, or `mumdia report --fasta`; empty when the FASTA does not hold the peptide), residue,
modification, peptide, peptide position, the best site probability over the runs, the false
localisation rates at it, the best placement and the number of runs it was localised in. The
experiment manifest records the per-run tables under `localisation` and the file under `sites`.

## 6. Running it

```text
# inside a search
{"localise": {"enabled": true}}

# on a finished single run
mumdia localise --psms-scored run/psms_scored.parquet \
  --spectra-ms2 run/spectra/spectra_ms2.parquet \
  --mass-cal run/seed_psms.parquet.masscal.json \
  --psms-extracted run/psms_extracted.parquet \
  --out run/localisation.parquet --config search.json
mumdia report --psms-scored run/psms_scored.parquet --localisation run/localisation.parquet \
  --fasta proteins.fasta --out-dir run

# on one run of a finished experiment (after a second pass, its psms_extracted is
# second_pass/<run>/psms_extracted.parquet)
mumdia localise --psms-scored exp/r0/scored.parquet --source 0 \
  --spectra-ms2 exp/r0/spectra/spectra_ms2.parquet \
  --mass-cal exp/r0/seed_psms.parquet.masscal.json \
  --psms-extracted exp/second_pass/r0/psms_extracted.parquet \
  --out exp/r0/localisation.parquet --config search.json
```

The configuration matters for the candidate residues (`peptidoforms.variable_mods`) and the
fallback fragment tolerance. Cost: under one second and about 1 GB per diaPASEF run of the
Arabidopsis set (146,690 MS2 scans, 64.5M peaks), most of it decoding the spectra.

## 7. Validation

### Prototype (P0)

A Python prototype on the Arabidopsis outputs, run before the engine stage, decided three
things.

- **Re-extraction from the spectra is required.** The chromatograms extract writes hold the
  library's top 12 fragments of the searched peptidoform only. Of the site-determining ions of
  a representative's own placement 42% are among them; the site-determining ions of every other
  placement, including those the library does not carry, are not in them at all, and those are
  half the evidence. The stage therefore reads the spectra.
- **The evidence is presence, at the apex.** Of the presence rules tried (any signal in the five
  scans at the apex; at least two scans; correlation with the common-ion profile at 0.3, 0.5,
  0.7; combinations), "any signal" localised the most peaks at a given decoy-residue FLR once
  the false presence came from impossible variants. Graded (three- and four-level) evidence and
  direction-specific false-presence rates were both worse calibrated against the decoy residue.
- **No temperature.** With the impossible-variant rates the model FLR and the decoy-residue FLR
  agreed at temperature 1 (0.75: 9.5% against 8.7%; 0.9: 3.0% against 3.9%; 0.99: 0.2%
  against 0.2%), so the default is 1.

### Arabidopsis phospho diaPASEF, twelve runs (engine)

The twelve-run time course of the Arabidopsis phospho set 2025-17-03B (timsTOF Pro diaPASEF, FASTA library with
Phospho STY and Met oxidation, at most two variable modifications, second-pass experiment),
stage run per run on the final split tables:

| measure | value |
|---|---|
| target peaks, summed over runs | 31,862 |
| with a choice (at least two real placements) | 29,335 |
| localised at 0.75 | 9,513 (32.4%; 27.9% to 37.2% per run) |
| unambiguous (one real placement) | 2,527 |
| decoy-residue wins (best on A) | 3,050 |
| candidate to decoy-residue position ratio | 3.72 |

| threshold | localised | model FLR | decoy-residue FLR |
|---|---|---|---|
| 0.5 | 15,736 (53.6%) | 20.7% | 18.9% |
| 0.75 | 9,513 (32.4%) | 9.3% | 9.0% |
| 0.9 | 5,352 (18.2%) | 3.0% | 3.9% |
| 0.99 | 2,185 (7.4%) | 0.2% | 0.7% |

- **Cross-run consistency.** Of the pairs of runs in which the same peptide, charge and
  composition were localised (main peak per run), 93.5% name the same placement at 0.75
  (29,509 pairs) and 97.3% at 0.9 (16,534 pairs).
- **Co-eluting isomers.** 45,220 accepted rows sit on peaks with more than one accepted isomer;
  the posterior of the row's own placement is below 0.25 for 54.5% of them and at or above 0.75
  for 8.6%.
- **Separated isomers.** 2,625 groups have two or more peaks at one charge in one run; where at
  least two of the peaks were localised, 267 name a different site on each peak and 137 the same
  site on each (a peptide eluting twice, or a peak split by the 5 s rule).
- **Isomer groups.** 28,678 target and 268 decoy groups at `isomer_group_q <= 0.01`, summed over
  runs (0.93%).

The decoy-residue estimate rests on the assumption that a wrong call lands on an alanine as
often as on any other candidate position; it is an estimate, not a ground truth. At 0.75 about
one localised site in eleven is wrong on this data by both estimates, which is why the
threshold is a reviewable default and 0.9 (about 3 to 4%) is the conservative choice.

### Synthetic phosphopeptides with known sites

SYNTHETIC_PLACEHOLDER

## 8. Limits and open work

- Per run. Combining the runs of an experiment into one site probability is not done; the
  experiment report takes each precursor's cells from its winning run.
- No neutral-loss ions. The impossible-variant rate absorbs part of their effect on the
  site-determining ions; scoring them would add evidence.
- Terminal modifications are not localised (the hypotheses place the modification on residues
  only); a modification carried on a residue together with another is not localised.
- Several localised modifications in one peptide: only the first of `localise.mods` it carries
  is localised.
- Separated isomers are localised independently; a placement claimed by one peak is not removed
  from the hypotheses of the other.
- Reporting only. Collapsing co-eluting isomers before FDR, de-duplicating their quantities and
  site-determining-ion features for the rescore are later work, each to be validated by
  entrapment.
