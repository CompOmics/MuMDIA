# 36. Modification site localisation (`localise`)

Experimental and opt-in (`localise.enabled`, default false). The stage is reporting only: it
reads the final scored table and the spectra and writes `localisation.parquet` per run, in
`run-experiment` also `localisation_experiment.parquet` (the runs combined), columns of
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
   default 0.01) are sorted by apex retention time and cut into chromatographic peaks
   (`peaks::cut`, `localise.peak_rule`). Under `elution_bounds` two rows next to each other in
   apex order are on one peak when the apex of either lies inside the other's elution bounds
   (`elution_lo` to `elution_hi` of the scored row) and the summed profile of the fragments
   their two placements share has no valley between the apexes: smoothed over three scans, it
   does not fall below `localise.peak_valley_ratio` (0.5) of the lower apex (`xic::has_valley`;
   0 cuts on the bounds alone). Two isomers that elute apart but overlap are then still two
   peaks, while one broad peak whose isomers were given apexes a few seconds apart is one.
   Under `apex_tolerance`, the rule of the first version, a row joins the current peak when its
   apex lies within `localise.peak_rt_tol_s` (5 s) of the peak's first apex; a scored table
   without elution bounds is always cut this way. The isomers of a peak share one
   localisation, computed on its representative (the highest score). Isomers further apart are
   separate peaks, each localised on its own.
3. **Hypotheses.** Every placement of the representative's count of the modification on its
   candidate slots, in lexicographic order (`hypotheses::enumerate`). A slot is a residue or
   the N-terminal amine, which is a placement of its own and not the first residue's side
   chain. The candidate residues are those of the modification's residue rules in
   `peptidoforms.variable_mods` (`position` `anywhere`); a terminal rule (residue `*`) opens the
   N-terminal slot instead: on every peptide for `peptide_n_term`, and for `protein_n_term` on a
   peptide whose N-terminal form is among the scored rows or that starts a protein of the
   search's FASTA, at its first residue or after the initiator methionine (a decoy follows its
   target through the shared base peptide). Without any rule for the modification (an imported
   library), the candidates are the residues it is found on in the scored table, and the
   N-terminus on the peptides the table carries it there. Acetyl with a protein N-terminal and
   a K rule is thus localised between the N-terminus and the lysines. Other modifications stay
   where the identification put them, and a residue or N-terminus that carries one is not a
   candidate. Placements on the decoy residue (`localise.decoy_residue`, default `A`) are
   added; they are never searched. While the N-terminal slot is open the decoy residue in first
   position is not: a modification there and one on the N-terminus give the same b and y
   ladder, so that decoy could never lose. A peptide with more than `localise.max_hypotheses`
   (200) placements is not localised. A peptide with a single real placement is
   `unambiguous`: its site is fixed by the sequence.
4. **Ions.** For every hypothesis, the b1 to b(n-1) and y1 to y(n-1) ions at fragment charge 1,
   and 2 for precursors of charge 3 or more (`ions::build`). One ion is one distinct m/z over all
   hypotheses (1e-4 Th), with the set of hypotheses that predict it. An ion every hypothesis
   predicts is common; one only some predict is site-determining. A position's modification
   count that no hypothesis gives it (b2 carrying the site when neither residue is a candidate)
   is an impossible variant: never a fragment of the peptide under any placement. With
   `localise.neutral_loss` (Phospho only), every fragment that carries a phosphoserine or
   -threonine also gives a loss ion 97.9769 Da lighter (phosphoric acid, the phosphate group
   plus water), as four classes of their own (`ions::phosphate_loss`); the decoy residue loses
   it as serine and threonine do, so that its placements stay exchangeable with the real ones,
   and a phosphotyrosine does not. The loss of a fragment carrying `c` sites falls exactly on the
   water loss of the same fragment carrying `c - 1`, which a competing placement predicts at
   every position between two candidate sites; such loss ions would count the competitor's
   water loss as evidence against it, so only loss ions without that alias are kept (in
   practice the positions where every placement puts the same number of sites in the fragment,
   which tell serine and threonine from tyrosine). They have no impossible variants, and take
   the false-presence rate of the plain class of the same ion type and charge.
5. **Evidence.** The spectra are read again (`xic`): the MS2 scans of the isolation windows that
   hold the precursor m/z (on diaPASEF, the slots whose 1/K0 range, widened by 0.02, holds the
   apex 1/K0), pooled in retention-time order, and the `localise.apex_scans` (2) scans either
   side of the one nearest the apex. An ion is present when any peak of those scans lies within
   the fragment tolerance of its calibrated m/z (the seed's `masscal.json` offset and tolerance,
   else `extract.frag_tol_ppm`) and, with 1/K0, within `localise.im_half_width` (0.05) of the
   apex 1/K0. The apex 1/K0 comes from `psms_extracted` (the selected peak), or, when that is
   not given, the intensity-weighted median 1/K0 of the peaks matching the common ions.
6. **Likelihood.** Four ion classes (eight with neutral losses): b and y at charge 1 and 2. Per class, the detection
   probability `d` is the share of the peak's common ions that are present and the
   false-presence probability `f` the share of its impossible variants that are present, each
   shrunk towards the run-wide share (Laplace-smoothed over the target peaks) by
   `localise.prior_strength` (64) pseudo-counts (`model::Counts::shrunk`). Under a hypothesis a
   site-determining ion it predicts is present with probability `d` and one it does not predict
   with probability `f`; the log-likelihood is the sum over the site-determining ions. The
   posterior is the softmax of the log-likelihoods divided by the temperature
   (`localise.temperature`, a number or fitted to the data, section 3), with a flat prior. A
   site's probability is the posterior mass of the hypotheses that use it. The
   best hypothesis is the highest posterior, ties to the lexicographically first placement, so
   the answer does not depend on the order the hypotheses were listed in.
7. **Calls.** The peak is `localised` when the best hypothesis uses no decoy residue and every
   one of its sites reaches `localise.threshold` (0.9); `decoy_residue` when the best uses the
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

### The temperature

The likelihood treats the site-determining ions as independent, and they are not (a peptide's
fragments are present or absent together, an interference adds several at once), so the
posterior can be too sharp or too flat, by an amount that depends on the acquisition
(section 7). `localise.temperature = "auto"` fits the divisor of the log-likelihoods without
ground truth (`calibrate`): the temperature at which the model's expected number of wrong real
calls (the sum of `1 - p`) equals the decoy residue's estimate of it (decoy-residue wins times
the position ratio) among the calls at or above 0.75, whatever the reporting threshold, with a
normal prior on `ln T` centred on 1 (standard deviation 0.35) and the Poisson variance of the
decoy wins. It moves the temperature only when the two estimates disagree at 1 by more than
two standard errors: a population with no or one decoy-residue win estimates no error rate,
and fitted on it the temperature fell below 1 on the dilution series, where the model was
already optimistic (section 7).

The population matters more than the objective. A peak whose peptide or modification count is
wrong has no right placement among the hypotheses; its calls land on the decoy residue as often
as on any other position, however confident the model is, and no temperature reconciles the two
estimates over such peaks (on the synthetic phosphopeptides, where two thirds of the peaks with
a choice are not on a synthesised sequence, the temperature fitted over every peak ran to the
top of the grid). Those peaks see few site-determining ions of any placement, so the fit is made
on the quarter of the eligible peaks (targets with a choice of site) with
the largest share of site-determining ions present. The share depends on the evidence and not on
which placement won, so the decoy residue stays exchangeable with the wrong real placements
inside the quarter. In `run-experiment` one temperature is fitted over the pooled runs; a single
run fits its own. The report records the fitted value, the population and the two error
estimates at it (`stats.temperature`). A number in `localise.temperature` is used as given.

## 3b. Combining the runs of an experiment

In `run-experiment` (and `mumdia localise --experiment-dir`) every run is analysed up to its
log-likelihoods first, the temperature is fitted over all of them, the per-run tables are
written, and then, with `localise.combine_runs` (default true), the runs are combined into one
site probability per consensus peak (`experiment`):

1. every run's confident target apexes (`run_psm_q <= 0.01`) are put on the scale of a
   reference run (the one with the most) by the monotone binned-median curves the second pass
   uses for its consensus retention times (`second_pass::alignment`; a run sharing fewer than
   200 precursors with the reference keeps its own scale);
2. the peaks of one precursor (localised modification, the peptidoform without it, its count,
   label and charge) are clustered across runs on that scale: in order of descending
   representative score, a peak joins the first consensus peak that has no peak of its run yet
   and whose anchor's aligned apex lies inside its aligned elution bounds, or whose bounds hold
   its apex, and otherwise starts a consensus peak of its own. Separated isomers therefore stay
   separate consensus peaks, and a run in which the precursor is absent, or elutes elsewhere,
   adds nothing to a consensus peak;
3. the log-likelihoods of the runs whose peak joined are summed hypothesis by hypothesis (in
   source order; a run whose hypothesis list differs, which only a different N-terminal
   permission or candidate set can cause, is left out), and the posterior is their softmax at
   the experiment temperature `localise.experiment_temperature`. Under `"auto"` (the default) it
   is fitted on the consensus peaks as the run temperature is on the run peaks. The runs of an
   experiment repeat each other's interferences, so summed evidence is not independent
   evidence: the right temperature lies between the run temperature (independent runs) and the
   run temperature times the number of runs summed (runs that repeat the same evidence), and
   the prior is centred on their geometric mean, the run temperature times the square root of
   the mean number of runs summed, from where the decoy residue moves it;
4. the calls and both false localisation rates are computed over the consensus peaks, as over the
   peaks of a run.

The per-run tables are the same with the combination on or off.

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

`localisation.parquet` (artifact `localisation` v2; v2 adds the last six columns), one row per
scored row of the run that carries a localised modification, in scored-table order:

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
| `site_probabilities` | utf8, nullable | `S3:0.9120;T5:0.0710;A6:0.0170`: every candidate slot (residue and 1-based position, or `N-term`) with its probability |
| `min_site_prob` | f64, nullable | the smallest site probability of the best placement |
| `peptidoform_probability` | f64, nullable | the posterior of the row's own placement |
| `own_site_prob` | f64, nullable | the smallest site probability of the row's own sites |
| `localised` | bool | the peak is `localised` or `unambiguous` |
| `on_decoy_residue` | bool | the best placement uses the decoy residue |
| `n_sdi`, `n_sdi_observed` | u32 | site-determining ions, and how many were present |
| `delta_loglik` | f64, nullable | best minus second-best log-likelihood |
| `site_flr_model`, `site_flr_decoy` | f64, nullable | section 3, at the peak's `min_site_prob` |
| `status` | utf8 | `localised`, `ambiguous`, `decoy_residue`, `unambiguous`, `not_attempted_q`, `too_many_hypotheses`, `no_spectra` |
| `n_common`, `n_common_observed` | u32 | common ions, and how many were present |
| `peak_rt_lo`, `peak_rt_hi` | f64, nullable | the peak's elution bounds: the smallest `elution_lo` and largest `elution_hi` of its rows |
| `hypothesis_sites` | utf8, nullable | on a representative scored on its evidence: the placements, `S2;T6;A4` (sites of one placement joined by `,`) |
| `hypothesis_loglik` | list of f64 | the log-likelihood of each placement, in that order (empty on the other rows); the posterior at a temperature is their softmax |

`localisation_experiment.parquet` (artifact `localisation_experiment` v1, at the experiment
root, written by `run-experiment` with more than one run and `localise.combine_runs`), one row
per consensus peak (section 3b): `consensus_peak_id`; `localised_mod`, `peptidoform`, `charge`,
`label`, `protein` of the anchor (the highest-scoring run peak); `consensus_rt`, its apex on the
reference run's scale; `n_runs` (runs whose log-likelihoods were summed) and `n_runs_observed`
(runs with a peak in it); `runs`, the run peaks it combines as `source:isomer_peak_id` joined by
`;`; and `n_hypotheses`, `best_site_peptidoform`, `site_probabilities`, `min_site_prob`,
`localised`, `on_decoy_residue`, `delta_loglik`, `site_flr_model`, `site_flr_decoy`, `status`,
`hypothesis_sites` and `hypothesis_loglik` (summed) as in the per-run table. Its report records
both temperatures and the alignment.

The report (`run`, `run-experiment`, or `mumdia report --localisation` / `--experiment-dir`)
appends `site_probability` (the row's `own_site_prob`), `localised` (the row's own placement is
the peak's localised best) and `best_site_peptidoform` to `peptides.tsv`. In an experiment the
three cells come from the run of the row the precursor is reported on, its experiment-wide
winner, and, when the runs were combined, three more from the consensus peak that run's peak
joined: `experiment_site_probability`, `experiment_localised` and
`experiment_best_site_peptidoform`. `sites.tsv` lists every site of a localised target peak
representative: protein, protein position (from the FASTA of a FASTA run, or `mumdia report
--fasta`; empty when the FASTA does not hold the peptide), residue (`N-term` for the N-terminal
slot, numbered at the peptide's first residue), modification, peptide, peptide position, the
best site probability over the runs, the false localisation rates at it, the best placement and
the number of runs it was localised in. With the combination it also lists every site of a
localised target consensus peak, and five more columns: `experiment_site_probability` (the best
over the localised consensus peaks naming the site), `experiment_site_flr_model`,
`experiment_site_flr_decoy`, `experiment_best_site_peptidoform` and `experiment_n_runs`; a site
localised only at the experiment level has the per-run cells empty and `n_runs` 0. The
experiment manifest records the per-run tables under `localisation`, the experiment-level table
under `localisation_experiment` and the file under `sites`.

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

# on every run of a finished experiment, as run-experiment does it (one temperature for the
# runs, then the combination); in place, recorded in the manifest for
# `mumdia report --experiment-dir exp`, or elsewhere with --out-dir
mumdia localise --experiment-dir exp --config search.json
mumdia localise --experiment-dir exp --out-dir trial --sources 0,1,2 --config search.json
```

The configuration matters for the candidate residues and terminal rules
(`peptidoforms.variable_mods`) and the fallback fragment tolerance; `--fasta` tells a
`protein_n_term` modification which peptides start a protein (`run` and `run-experiment` pass
the search's FASTA). Cost: under one second and about 1 GB per diaPASEF run of the
Arabidopsis set (146,690 MS2 scans, 64.5M peaks), most of it decoding the spectra.

## 7. Validation

The prototype, Arabidopsis and synthetic subsections below describe the first version at its
defaults (the 5 s peak rule, temperature 1, threshold 0.75); the last subsection measures the
follow-ups and the current defaults against it.

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
  against 0.2%), so the first version kept 1. The synthetic set below shows the agreement is
  acquisition-dependent; the follow-ups fit it (section 3).

### Arabidopsis phospho diaPASEF, twelve runs (engine)

The twelve-run time course of the Arabidopsis phospho set 2025-17-03B (timsTOF Pro diaPASEF,
FASTA library with Phospho STY and Met oxidation, at most two variable modifications,
second-pass experiment), stage run per run on the final split tables with the first version's
defaults:

| measure | value |
|---|---|
| target peaks, summed over runs | 31,862 |
| with a choice (at least two real placements) | 29,335 |
| localised at 0.75 | 9,613 (32.8%; 27.7% to 37.3% per run) |
| unambiguous (one real placement) | 2,527 |
| decoy-residue wins (best on A) | 2,879 |
| candidate to decoy-residue position ratio | 3.72 |

| threshold | localised | model FLR | decoy-residue FLR |
|---|---|---|---|
| 0.5 | 16,178 (55.1%) | 21.6% | 17.6% |
| 0.75 | 9,613 (32.8%) | 10.4% | 6.7% |
| 0.9 | 4,861 (16.6%) | 3.0% | 2.6% |
| 0.99 | 1,876 (6.4%) | 0.3% | 0.8% |

- **Cross-run consistency.** Of the pairs of runs in which the same peptide, charge and
  composition were localised (main peak per run), 94.8% name the same placement at 0.75
  (30,680 pairs) and 97.8% at 0.9 (16,022 pairs).
- **Co-eluting isomers.** 45,220 accepted rows sit on peaks with more than one accepted isomer;
  the posterior of the row's own placement is below 0.25 for 53.2% of them and at or above 0.75
  for 8.3%.
- **Separated isomers.** 2,625 groups have two or more peaks at one charge in one run; where at
  least two of the peaks were localised, 270 name a different site on each peak and 137 the same
  site on each. 90% of the same-site pairs have the second apex inside the first one's elution
  bounds (median apex gap 6.4 s against 49.4 s for different sites): mostly one peak split by the
  5 s rule.
- **Isomer groups.** 28,678 target and 268 decoy groups at `isomer_group_q <= 0.01`, summed over
  runs (0.93%).

### Synthetic phosphopeptides with known sites

PXD019797 / JPST000859 (Kitata et al., Nat Commun 2021): 166 synthetic phosphopeptides from 61
human proteins (139 mono-, 21 di- and 6 tri-phosphorylated; 117 sequence and phospho-count
groups, 30 of them with two to five positional isomers synthesised and pooled), acquired by DIA
on an Orbitrap Fusion Lumos (10 Th windows, 500 to 1,000 m/z, 160 min gradient). The three
replicates of the pure pool (`PhosphoPooledSynth_DIA..._1903090{1,2,3}.raw`) were searched as one
experiment against a FASTA of the 61 proteins (trypsin, two missed cleavages, Phospho STY and Met
oxidation, at most three variable modifications, charges 2 to 4, AlphaPeptDeep fragments,
DeepLC retention times, `nn_torch`). A call is correct when its placement is one of the
synthesised isomers of that sequence and phospho count; isomers of one sequence were pooled, so
a co-eluting mixture is possible, and the 87 groups with a single synthesised placement are
also scored on their own.

1,207 target peaks fell on a synthesised sequence and phospho count, 1,165 of them with a
choice of site (94 sequences). With the first version's defaults:

| threshold | localised | wrong | true FLR | model FLR | decoy-residue FLR |
|---|---|---|---|---|---|
| 0.5 | 830 (71.2%) | 125 | 15.1% | 12.0% | 30.2% |
| 0.75 | 687 (59.0%) | 69 | 10.0% | 6.2% | 10.9% |
| 0.9 | 479 (41.1%) | 28 | 5.8% | 2.1% | 7.3% |
| 0.99 | 240 (20.6%) | 3 | 1.3% | 0.2% | 1.8% |

Single-placement groups only (690 peaks, 68 sequences): 0.75 400 localised, true FLR 11.3%
(model 5.4%, decoy residue 15.7%); 0.9 293, 6.1% (1.7%, 9.4%); 0.99 150, 0.0%. Share correct
by minimum site probability, all groups: 0.75 to 0.9 80.3% (mean probability 0.84), 0.9 to
0.99 89.5% (0.96), 0.99 and above 98.8% (1.00).

What this says:

- **The decoy-residue FLR tracks the true FLR**: 10.9% against 10.0% at 0.75, 7.3% against 5.8%
  at 0.9, 1.8% against 1.3% at 0.99. It is the estimate to read.
- **The model FLR is overconfident on this data** by a factor of 1.5 to 3 at 0.75 and above, while
  on the Arabidopsis set it was within a factor of 1.6 of the decoy residue in the other
  direction. The model's absolute calibration depends on the acquisition; its ranking does not
  (the share correct rises monotonically with the probability). With 8 pseudo-counts, a temperature of 1.5
  to 2 brought the model FLR to the true FLR here (1.5: 8.3% model against 9.5% true at 0.75)
  and made it conservative on Arabidopsis (1.5: 10.6% against a decoy-residue 4.3%), so no
  single temperature fits both. A per-run fit against the decoy residue would not work on this
  search: most of its accepted phospho peaks are not synthesised peptides, and over all of them
  the decoy-residue rate exceeded the model's at every temperature tried (1 to 3). The first
  version kept 1; the follow-ups fit it on a population that leaves those peaks out.
- **Prior strength.** Against 8 pseudo-counts, 64 localised more peaks at a lower error on both
  sets (synthetic 0.75: 687 at 10.0% true FLR against 642 at 11.7%; 0.9: 479 at 5.8% against 496
  at 8.7%; Arabidopsis 0.75: 9,613 at a decoy-residue 6.7% against 9,513 at 9.0%, cross-run
  agreement 94.8% against 93.5%), and 32 to 1,000 were within noise of each other; 64 is the
  default.
- **Threshold.** About one call in ten is wrong at 0.75 (synthetic 10.0% true, Arabidopsis 6.7%
  decoy residue), one in twenty at 0.9. The first version kept 0.75 for review; the follow-ups
  measure it again with the fitted temperature and move the default to 0.9.

Only 1,165 of the 3,650 target peaks with a choice in this search are on synthesised peptides:
the restricted FASTA accepts phospho forms of other peptides of the 61 proteins (synthesis by-
products, in-source fragments, and false identifications in a search space this small). Their
decoy-residue wins (48% decoy-residue FLR at 0.75 over all 3,650) are why a run-level temperature
fit is not safe. The pool was also measured diluted into a yeast digest (`PhosphoPooledSynth4xdY` and
`20xdY`, three replicates each; yeast is not in the FASTA, so its peptides are interference).
The nine runs searched as one experiment, at 0.75 and 0.9:

| sample | peaks with a choice | localised at 0.75 | true FLR | decoy-residue FLR | localised at 0.9 | true FLR | decoy-residue FLR |
|---|---|---|---|---|---|---|---|
| pure pool | 1,114 | 677 (60.8%) | 8.7% | 7.7% | 473 | 6.1% | 3.7% |
| 4x in yeast | 863 | 544 (63.0%) | 10.3% | 8.0% | 389 | 7.5% | 1.1% |
| 20x in yeast | 614 | 355 (57.8%) | 8.2% | 7.8% | 248 | 3.6% | 0.0% |

The localisation rate and the error at 0.75 hold across a 20-fold dilution into a background
proteome, and the decoy-residue estimate stays within about 2.5 points of the truth at 0.75; at
0.9 it has too few decoy wins (0 to 4 per three runs) to be more than a bound. In the 20x runs a
per-run `isomer_group_q` reached no group at 1% in two of three runs (minimum 0.011): about 230
target groups per run, with the first decoy group early, which is the `1 / T` floor of a small
population, not a fault.

### Follow-ups: peaks, neutral losses, temperature, experiment level, threshold

Every choice was made on the synthetic pure pool (three runs) alone, then confirmed on the 4x
and 20x dilutions (three runs each, from the nine-run experiment above) and on Arabidopsis, each
arm with `mumdia localise --experiment-dir` on the finished searches. One correction came from
the confirmation and is marked below. "At a true FLR of 5%" is the most peaks on synthesised
sequences localised at any threshold whose calls are at most 5% wrong: it measures the ranking,
whatever the temperature. The number of peaks with a choice changes with the peak rule (fewer
when broad peaks are no longer cut in two), so rates are given beside counts.

Pure pool, per run, peaks on synthesised sequences with a choice of site (localised, true FLR,
model FLR, decoy-residue FLR):

| arm | T | with a choice | at 0.75 | at 0.9 | at 0.99 | at a true FLR of 5% / 10% | single-isomer sequences cut into two peaks |
|---|---|---|---|---|---|---|---|
| first version (5 s rule, T 1) | 1 | 1,165 | 687, 10.0%, 6.2%, 10.9% | 479, 5.8%, 2.1%, 7.3% | 240, 1.2% | 382 (32.8%) / 685 (58.8%) | 207 of 426 |
| elution bounds, T 1 | 1 | 777 | 457, 8.5%, 6.0%, 10.0% | 329, 4.0%, 2.2%, 7.6% | 172, 0.6% | 375 (48.3%) / 475 (61.1%) | 77 of 426 |
| fitted temperature, 5 s rule | 1.40 | 1,165 | 601, 8.3%, 8.4%, 7.3% | 388, 5.4%, 3.0%, 5.7% | 163, 0.0% | 368 (31.6%) / 690 (59.2%) | 207 of 426 |
| neutral losses, 5 s rule, T 1 | 1 | 1,165 | 683, 9.5%, 6.5%, 9.0% | 472, 6.6%, 2.1%, 7.5% | 239, 2.5% | 342 (29.4%) / 686 (58.9%) | 207 of 426 |
| current defaults | 1.29 | 777 | 419, 7.2%, 7.6%, 6.9% | 276, 3.6%, 2.6%, 7.5% | 136, 0.0% | 362 (46.6%) / 482 (62.0%) | 77 of 426 |
| current defaults and neutral losses | 1.57 | 777 | 371, 6.7%, 8.5%, 6.7% | 235, 2.6%, 3.1%, 3.5% | 103, 0.0% | 347 (44.7%) / 475 (61.1%) | 77 of 426 |

Confirmation, per run:

| set | arm | T | with a choice | at 0.75 | at 0.9 | at 0.99 | at a true FLR of 5% / 10% | cut in two |
|---|---|---|---|---|---|---|---|---|
| 4x | first version | 1 | 863 | 544, 10.3%, 6.4%, 8.0% | 389, 7.5%, 2.5%, 1.1% | 170, 2.9% | 223 (25.8%) / 502 (58.2%) | 132 of 378 |
| 4x | current defaults | 1 (not fitted) | 613 | 379, 11.6%, 6.4%, 7.8% | 272, 7.4%, 2.4%, 3.1% | 118, 1.7% | 160 (26.1%) / 353 (57.6%) | 13 of 378 |
| 20x | first version | 1 | 614 | 355, 8.2%, 6.4%, 7.8% | 248, 3.6%, 2.3%, 0.0% | 120, 1.7% | 280 (45.6%) / 408 (66.4%) | 66 of 316 |
| 20x | current defaults | 1 (not fitted) | 498 | 296, 8.8%, 6.3%, 3.0% | 204, 3.9%, 2.3%, 0.0% | 102, 1.0% | 226 (45.4%) / 320 (64.3%) | 5 of 316 |

On Arabidopsis, without ground truth: localised, model FLR and decoy-residue FLR, the agreement of
the runs on the placement of the main peak, and the separated peaks of one group and charge in
one run localised to the same and to different sites:

| arm | T | with a choice | at 0.75 | at 0.9 | at 0.99 | runs agree at 0.75 / 0.9 | separated, same / different site |
|---|---|---|---|---|---|---|---|
| first version | 1 | 29,335 | 9,613, 10.4%, 6.7% | 4,861, 3.0%, 2.6% | 1,876, 0.3%, 0.8% | 94.8% / 97.8% | 137 / 270 |
| elution bounds, T 1 | 1 | 28,442 | 9,427, 10.3%, 6.7% | 4,817, 2.9%, 2.5% | 1,871, 0.3%, 0.8% | 94.8% / 97.8% | 23 / 265 |
| neutral losses, 5 s rule, T 1 | 1 | 29,335 | 9,811, 10.2%, 6.9% | 5,084, 2.9%, 2.5% | 1,953, 0.3%, 0.8% | 94.7% / 97.7% | 142 / 267 |
| current defaults | 0.80 | 28,442 | 11,232, 8.9%, 9.8% | 6,209, 2.8%, 3.3% | 2,735, 0.2%, 1.3% | 93.3% / 97.4% | 35 / 350 |

Experiment level at the current defaults (consensus peaks, synthesised ones on the synthetic
sets; "gained" are localised by the combination and by no run, "lost" by at least one run and
not by the combination):

| set | T run / experiment | with a choice | at 0.75 | at 0.9 | at 0.99 | at a true FLR of 5% / 10% | gained / lost (wrong) |
|---|---|---|---|---|---|---|---|
| pure | 1.29 / 2.40 | 327 | 185, 5.9%, 6.3%, 10.7% | 136, 4.4%, 2.3%, 8.7% | 72, 0.0% | 153 (46.8%) / 212 (64.8%) | 6 (1) / 23 (8) |
| 4x | 1 / 1.49 (prior) | 245 | 163, 11.7%, 3.9%, 12.5% | 139, 10.8%, 1.6%, 5.9% | 94, 9.6% | 65 (26.5%) / 125 (51.0%) | 5 (1) / 15 (6) |
| 20x | 1 / 1.54 (prior) | 191 | 132, 6.1%, 4.4%, 3.2% | 111, 6.3%, 1.9%, 0.0% | 65, 1.5% | 102 (53.4%) / 153 (80.1%) | 10 (1) / 10 (4) |
| Arabidopsis | 0.80 / 1.44 (prior) | 8,673 | 3,480, 6.0%, 11.1% | 2,546, 1.9%, 6.7% | 1,524, 0.2%, 2.8% | | 203 / 1,088 |

What the follow-ups show:

- **Peaks (elution bounds).** The rule cut 63% fewer single-isomer sequences of the pure pool
  into two peaks (207 to 77; 132 to 13 and 66 to 5 on the dilutions) and left 23 instead of 137
  same-site pairs of separated peaks on Arabidopsis, while the pairs localised to different
  sites stayed (86 to 84 on the pure pool at temperature 1, 41 to 44, 27 to 30, 270 to 265). The
  share of peaks localisable at a true FLR of 5% rose from 32.8% to 48.3% on the pure pool and
  held on the dilutions (25.8% to 26.1%, 45.6% to 45.4%). A valley ratio of 0.7 kept three more
  different-site pairs and cut 13 more sequences in two; 0, the bounds alone, eleven fewer and
  13 fewer. The default is 0.5.
- **Neutral losses.** Alias-free loss ions localised fewer peaks at a true FLR of 5% on the pure
  pool (342 against 382), were mixed on the dilutions (233 against 223, 267 against 280), and
  localised 2% more at an unchanged decoy-residue rate on Arabidopsis. Every loss ion, aliases
  included, was worse (15.6% true FLR at 0.75 against 10.0%). Off by default.
- **Temperature.** Fitted on every peak with a choice the temperature rose to 3.6 or more (the
  off-target peaks); on the top quarter by share of site-determining ions present it fitted 1.29
  on the pure pool (1.19 fitted on the synthesised peptides alone), where the model's rate at
  0.75 became 7.6% against a true 7.2% (6.0% against 8.5% at temperature 1). On Arabidopsis it
  fitted 0.80 (z 5.2): the model's rate went from 10.3% against a decoy-residue 6.7% to 8.9%
  against 9.8%, with 19% more peaks localised at 0.75 and the runs agreeing a little less (93.3%
  against 94.8%). Correction from the confirmation: without the two-standard-error gate the
  dilutions, with one and no decoy-residue win in their populations, fitted 0.83 and 0.90 and
  sharpened a model that was already optimistic there (6.4% against a true 11.6%); with it they
  stay at 1 (z 1.0 and 1.5). The prior width 0.35 was chosen on the pure pool, where 0.5 let the
  fit jump between 1.1 and 2.3 as the population moved by a few peaks.
- **Experiment level.** One call per consensus peak, at a localisation rate at a fixed true FLR
  equal to a run's or above it (pure 46.8% against 46.6% at 5%, 64.8% against 62.0% at 10%), and
  a true FLR at 0.75 equal to or below a run's (5.9% against 7.2%, 11.7% against 11.6%, 6.1%
  against 8.8%). Summed as independent evidence (experiment temperature 1) the model's rate fell
  to 2.6% against a true 9.1% and a decoy-residue 18.1% on the pure pool; the prior centred on the
  run temperature times the square root of the runs summed (1.96 there) and the fit (2.40) hold
  it at 6.3% against 5.9%. Where the gate holds the prior (4x, Arabidopsis) the model's
  experiment-level rate stays optimistic, and the decoy-residue rate is the one to read. The
  combination drops calls the runs do not repeat: 8 of the 23 the pure pool lost were wrong.
- **Threshold.** With the fitted temperature the site probability is close to calibrated, and
  0.75 left 7.2% (pure), 11.6% (4x) and 8.8% (20x) of the calls on synthesised sequences wrong
  and a decoy-residue rate of 9.8% on Arabidopsis; 0.9 left 3.6%, 7.4%, 3.9% and 3.3%, for 28%
  to 45% fewer localised peaks. The default is 0.9, so that a peak flagged `localised` is wrong
  about once in twenty-five on three of the four sets; 0.75 is one key away, and both rates are
  in every row.
- **Cost.** The localisation of the three pure-pool runs took 1.4 s and 1.0 GB as one
  experiment, that of the twelve Arabidopsis runs 6.1 s and 1.2 GB (first version 6.7 s and
  1.2 GB): the combination and the temperature fit are a small part of reading the spectra.

## 8. Limits and open work

- The experiment-level probability sums correlated evidence. Its temperature is fitted only when
  the decoy residue disagrees with the model by more than two standard errors; otherwise it stays
  at the prior's centre, and the model's experiment-level rate was then optimistic (4x dilution
  3.9% against a true 11.7% at 0.75, Arabidopsis 6.0% against a decoy-residue 11.1%). Read
  `site_flr_decoy` there.
- The fitted temperature needs decoy-residue wins: on small or clean experiments (the
  three-run dilution series: no or one win in the population) it stays at 1.
- Neutral-loss ions (`localise.neutral_loss`) are implemented for Phospho and off: they did not
  localise more peaks at a given true false localisation rate on the synthetic set.
- C-terminal modifications are not localised (the search refuses C-terminal rules); a
  modification carried on a residue or the N-terminus together with another is not localised.
  A modification on the N-terminus and the same one on the first residue give the same
  fragments, so the stage reports them as ambiguous rather than choosing.
- Several localised modifications in one peptide: only the first of `localise.mods` it carries
  is localised.
- Separated isomers are localised independently; a placement claimed by one peak is not removed
  from the hypotheses of the other.
- Reporting only. Collapsing co-eluting isomers before FDR, de-duplicating their quantities and
  site-determining-ion features for the rescore are later work, each to be validated by
  entrapment.
