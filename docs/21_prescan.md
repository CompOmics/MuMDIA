# 21. Prescan: sequence-tag pruning for modification searches

`mumdia prescan` (`stages/prescan.rs`) keeps only the modification-bearing library candidates
whose modification-anchored sequence trimers are actually observed in their own isolation window
and retention-time range. It is optional, per run, and sits between `convert` and library
assembly.

It exists because a modification search multiplies the candidate space by the number of modforms
per peptide, while any single run supports only a small fraction of those hypotheses. Pruning per
file means extraction and rescoring see a search space sized to the evidence rather than to the
enumeration.

## Only modification-bearing candidates are ever pruned

Every library row is tokenised, but `anchored_tris` emits trimers only for positions carrying an
`anchor_mods` modification. A candidate with no anchored modification therefore yields an empty tag
set and is absent from the output: the survivor list is, by construction, exclusively
anchor-bearing candidates. Library assembly then adds the whole unanchored remainder back
unconditionally and applies the survivor list only to the anchored part.

Measured on a 54.8M-row library with three cysteine modifications as anchors:

| | rows | share |
|---|---|---|
| library total | 54,821,556 | |
| eligible (carries an anchor mod) | 41,434,790 | 75.6% |
| never at risk (unmodified, or only non-anchor mods) | 13,386,766 | 24.4% |
| survivors | 4,906,730 | 11.8% of eligible |
| survivors NOT carrying an anchor mod | 0 | |

The consequence is the safety property that makes this stage cheap to experiment with: an ordinary
proteome result cannot be damaged by prescan tuning. Changing `tol_da`, `rt_slack_s` or
`top_peaks` can only add or remove modified hypotheses, because unmodified candidates never reach
the screen's decision at all.

Note that the stage still walks every row rather than pre-filtering to anchor-bearing ones. That
costs about a second on a 54.8M-row library and keeps the eligibility rule in exactly one place,
`anchored_tris`, instead of duplicating it as a string match over peptidoforms that could drift
out of agreement with the configured anchors.

## What it is not

The screen cannot separate a true modified peptide from its decoy, and no tuning will make it.
`anchored_tris` (`prescan.rs:145-161`) emits every trimer in both orientations, and a reverse
decoy preserves both composition and precursor m/z, so a decoy's anchored tag set is identical to
its target's. A decoy therefore survives exactly when its target does. Measured target:decoy
survival ratio is 1.0000 (2,453,365 / 2,453,365 on a 54.8M-row library).

That property is the reason the stage is safe rather than a reason to distrust it: because the
screen is blind to label, target/decoy exchangeability is untouched and downstream q-values remain
valid. Treat it as a compute reduction, never as a discriminator, and do not read a reduced
survivor count as enrichment for real modified peptides.

## The decoy screen is not optional

Both labels go through the identical criterion, each evaluated on its own sequence, its own
precursor m/z and its own RT window.

Screening only targets and then admitting their paired decoys by shared peptide looks equivalent
and is not. Surviving targets are then selected for having observed signal while their decoys are
a signal-blind sample, so the decoy score distribution understates the target one and the
modification's q-values come out anticonservative. On this data that mistake made the apparent
modification-specific FDR improve from ~45% to ~32% under pruning; with a symmetric screen it
returns to ~43%, and the "improvement" was the bias.

`prescan.rs` aborts when survivors come back single-label above a small count, because that means
the screen has become label-dependent and the modification's null is gone.

## Algorithm

1. **Observed tag index.** For each MS2, take the `top_peaks` most intense peaks, sort by m/z, and
   find chained charge-1 residue-mass deltas: peak pairs separated by one residue mass within
   `tol_da`, chained three deep. The resulting trimers are stored per
   `(window_id, floor(rt / rt_bin_s))`. Parallel over spectra.
2. **Candidate screen.** Tokenise each peptidoform into alphabet indices, take the trimers that
   cover an anchored (modified) position in both orientations, and keep the candidate if any of
   them appears in an index cell whose window contains the precursor m/z and whose RT bin overlaps
   `[rt_lo - rt_slack_s, rt_hi + rt_slack_s]`. Parallel over candidates.

Deliberately permissive: raw peak deltas, no deisotoping, no charge deconvolution. A false tag
only fails to prune, while a missed tag discards a real candidate with no way to recover it
downstream.

Anchoring on the modified residue is what makes the evidence specific. A trimer elsewhere in the
peptide is evidence for the backbone, not for the modification, and would let unmodified signal
keep a modified hypothesis alive.

I and L share one alphabet index (`prescan.rs:68-80`). A residue-mass delta cannot separate
isobaric residues, so distinguishing them would invent tags no spectrum can confirm.

Residue and modification masses come from the shared model, `residue_mass` plus `unimod_mass`; a
modified residue's tag mass is backbone + delta. A modification name absent from that model is a
typed error, not a silently skipped residue.

## Configuration

```json
"prescan": {
  "tol_da": 0.005,
  "rt_slack_s": 150.0,
  "rt_bin_s": 25.0,
  "top_peaks": 150,
  "mods": ["C:Carbamidomethyl", "M:Oxidation"],
  "anchor_mods": ["C:Farnesyl", "C:GeranylGeranyl", "C:Hydroxyfarnesyl"]
}
```

- `mods` are the modifications a screened peptidoform may carry. A peptidoform holding anything
  outside `mods` plus `anchor_mods` is dropped rather than screened on a partly understood
  sequence.
- `anchor_mods` are what the screen anchors on. An empty list is a hard error: nothing could ever
  be anchored, so every modified candidate would be discarded.
- `rt_slack_s` widens each candidate's RT window. The window comes from a calibration fitted on a
  different run, and `cal.json` residuals are in-sample and roughly 3x optimistic (see
  docs/08_rt_im_train.md), so size this from out-of-sample RT error. The default of 150 s covers a
  measured p90 of 146 s on Orbitrap DIA. It is deliberately the loose end of the trade.
- `top_peaks` bounds the O(peaks^2) delta search. It is **not** destructive: it affects tag
  construction only, never the spectra artifact that extraction later reads. Contrast
  `--top-peaks-ms2`, which is destructive (docs/04_convert.md).

### `top_peaks` is a sensitive knob

Moving `top_peaks` by **one peak** changed the survivor set by 3.2% (155,404 candidates) on a
50-window Orbitrap DIA run, because the cut sits deep in the peak distribution where consecutive
peaks are near-equal in intensity. Two consequences:

- the peak cut is tie-broken deterministically, intensity descending then m/z ascending
  (`prescan.rs`), so reruns do not differ for no reason;
- do not treat a survivor count as a precise quantity, and do not tune this knob by chasing small
  differences in it.

## Performance

Screening is independent per candidate, so it is a rayon fan-out. Measured on one 50-window
Orbitrap DIA run (32,950 MS2) against a 54.8M-row library, against the Python implementation this
stage replaced:

| step | Python | Rust | speedup |
|---|---|---|---|
| tag index build | 27 s | 0.64 s | 42x |
| candidate screening | 412 s | 5.6 s | 74x |
| total wall | 439 s | 38 s | 11.6x |

Most of the remaining 38 s is reading the library and RT tables, not computing.

## Inputs and output

```
mumdia prescan \
  --ms2 <spectra_ms2.parquet> \
  --isolation-windows <isolation_windows.parquet> \
  --library-precursors <lib_precursors.parquet> \
  --run-windows <run_windows.parquet> \
  --out prescan_survivors.parquet \
  --config <config.json>
```

`--run-windows` is a `run_windows`-shaped table supplying `candidate_id, rt_lo, rt_hi`. It is
joined by `candidate_id` through a dense lookup rather than by row position, so a reordered table
cannot silently mismatch RT bounds.

`prescan_survivors.parquet`: `candidate_id` u32, `label` str, sorted by candidate id so the
artifact is independent of thread count. The sibling `report.json` records `screened`,
`survivors`, `targets`, `decoys`, `target_decoy_ratio` and `index_cells`; check
`target_decoy_ratio` on every run, since a value away from 1.0 means the screen has stopped being
label-blind.

## Where it fits

The survivor list selects which modification-bearing candidates enter a per-run library. Assembly
of that library is out of scope for this stage: it consumes `prescan_survivors.parquet` and must
preserve the library contracts in docs/06_predict_frag_index_matchers.md, in particular contiguous
`candidate_id` and precursors ascending by `precursor_mz`.

A per-run library means a per-run search space. Under an experiment-wide rescore that is still
sound, because `rescore --competed` stamps `source` and computes a per-source `run_psm_q`
(docs/11_compete_rescore_fdr.md), but cross-run quantification will have structural gaps wherever
a precursor is present in one run's library and absent from another's.

For a modification search, pair this stage with `compete.group_by = peptidoform_charge`. Under the
default competition key every modform of a peptide collapses to one winner before rescore, which
discards exactly the candidates this stage was run to retain (docs/11_compete_rescore_fdr.md).
