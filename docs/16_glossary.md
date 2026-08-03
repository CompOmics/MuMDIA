# 16. Glossary

Domain and codebase terms as MuMDIA uses them. Each entry is self-contained.
Assertions about code behavior cite `file:line` against the source tree; read
that source to confirm details. Terms are alphabetical.

---

**AIF (all-ion fragmentation)**. A wide isolation mode in which no quadrupole
window is applied, so every precursor in the scan range is co-fragmented into one
highly chimeric MS2. When `convert` sees an MS2 whose isolation bounds are both
zero, it synthesizes a full-range window `[0, 1e6]` so the scan is treated as
covering all precursors (`rust/mumdia/crates/mumdia/src/stages/convert.rs:143`).
The `LFQ_Orbitrap_AIF_Ecoli_01` benchmark file is AIF.

**apex**. The retention-time point (a scan or a small scan group) where a
candidate's co-eluting fragments are strongest, taken as the PSM's single
representative scan. `extract` locates it as the final step of its acceptance
cascade and emits one apex-level PSM per surviving candidate
(`rust/mumdia/crates/mumdia/src/stages/extract.rs:10`). Because a single fragment
m/z channel is chimeric in DIA, the apex is chosen by co-eluting fragment breadth,
optionally by only the top-K predicted (signature) fragments, rather than by raw
maximum intensity (`rust/mumdia/crates/mumdia-core/src/config.rs:413`).

**candidate_id**. The dense integer key for one library entry (a peptidoform at a
charge). It is a precondition of the fragment index that `candidate_id` be the
contiguous range `0..n` in precursor-row order, and that precursors be sorted
ascending by precursor m/z; `Library::load` checks both and fails loudly if
violated (`rust/mumdia/crates/mumdia/src/index.rs:74`,
`rust/mumdia/crates/mumdia/src/index.rs:135`). This lets the inverted index recover
the isolation-window candidate slice by a binary search over `candidate_id`, since
`candidate_id` order equals precursor-m/z order
(`rust/mumdia/crates/mumdia/src/index.rs:6`).

**chimeric spectrum**. An MS2 in which fragments from several co-isolated,
co-eluting precursors overlap in one spectrum, the normal case in DIA (in
wide-window DIA about 98% of fragment m/z collide within tolerance,
`rust/mumdia/crates/mumdia-core/src/config.rs:127`). Chimeric interference is why
intensity-based apex selection and single-scan gates are unreliable, and why
in-silico decoys can under-model the true false-match population (motivating the
entrapment cross-check, `rust/mumdia/crates/mumdia/src/fdr.rs:60`).

**chromatogram / XIC (extracted ion chromatogram)**. The intensity of one m/z
channel traced across retention time. `extract` emits a per-fragment chromatogram
for each surviving candidate plus MS1 isotope XICs for the precursor
(`rust/mumdia/crates/mumdia/src/stages/extract.rs:12`); `features` and `quant`
read these traces for co-elution, apex-shape, and integration.

**claimed / contested / shared peak**. In chimeric DIA one observed peak often
matches several candidates at once. A shared peak is one matched by more than one
candidate; claiming is the policy (`PeakClaim`) that decides how its intensity is
apportioned so a chimeric candidate cannot borrow a real peptide's peak wholesale,
ranging from `None` (every claimant gets the full peak) through winner-take-all to
co-elution-profile apportionment
(`rust/mumdia/crates/mumdia-core/src/config.rs:132`). With
`emit_contested_features` the extractor records a non-destructive `contested_frac`
per PSM: the fraction of a candidate's matched intensity a co-eluting competitor
claims more strongly (`rust/mumdia/crates/mumdia-core/src/config.rs:454`). All peak
claiming is default-off.

**co-elution**. Agreement in retention-time shape: a candidate's fragments should
rise and fall together over the same scans. `extract` uses a consecutive-scan
co-elution run in its acceptance cascade
(`rust/mumdia/crates/mumdia/src/stages/extract.rs:11`), and the co-elution feature
family scores each fragment's XIC against a predicted-intensity-weighted reference
profile (`rust/mumdia/crates/mumdia/src/stages/features/coelution.rs:3`). It is
temporal agreement, orthogonal to intensity-pattern agreement.

**competition key (`compete.group_by`)**. The grouping over which `compete`
resolves rival candidates, keeping only the highest `prelim_score` member of each
group and deleting the rest before rescore and before any FDR estimate
(`rust/mumdia/crates/mumdia/src/stages/compete.rs:319-340`). The default variant
is named `precursor` but is a misnomer: the key is
`(base_peptide_id, label_code, 0, peak_rank)`
(`rust/mumdia/crates/mumdia/src/stages/compete.rs:88`), and `base_peptide_id` is
the stripped-sequence identity, so every charge and every modification variant of
one peptide collapses to a single winner. `peptidoform_charge`
(`rust/mumdia/crates/mumdia/src/stages/compete.rs:93-98`) keys
`(pform_id, label, charge, peak_rank)` instead and is required, not optional, for
a PTM or modification search, where the default would delete the modified form
whenever an unmodified sibling scored higher. Competition is within label, so it
never eliminates a target against its own decoy. See
`docs/11_compete_rescore_fdr.md`.

**decoy (reverse / scramble / shift)**. A deliberately false library entry used to
estimate the false discovery rate. MuMDIA mints one paired decoy per target at
digest time: `Reverse` reverses the interior residues keeping the C-terminal
residue fixed, `Scramble` runs a seeded deterministic Fisher-Yates shuffle of the
interior (`rust/mumdia/crates/mumdia/src/stages/digest.rs:99`). Native decoys are
collision-checked: a decoy that would collide with a target or an already-emitted
decoy is re-scrambled with independent seeds while keeping the C terminus fixed,
and dropped if no distinct sequence is found
(`rust/mumdia/crates/mumdia/src/stages/digest.rs:137`). `DiannShift` (a
terminal-residue fragment m/z shift) is threaded through config but realized
nowhere and rejected by validation
(`rust/mumdia/crates/mumdia/src/stages/digest.rs:126`,
`rust/mumdia/crates/mumdia-core/src/config.rs:1021`). The default is `Reverse`
(`rust/mumdia/crates/mumdia-core/src/config.rs:20`).

**DeepLC fine-tune**. An optional per-run adaptation of the DeepLC retention-time
model on the confident seed PSMs, run between `search-seed` and RT calibration; it
writes a new `_ft` precursor-library table with replaced `predicted_irt` and
rebinds downstream stages to it; the input file is unchanged. It is opt-in
(`rt_im_train.finetune_deeplc`) and nondeterministic (no fixed torch/numpy seed).
See `iRT`.

**entrapment**. An optional decoy-independent FDR cross-check that spikes a foreign
proteome into the search library and treats those spike-in PSMs as real negatives.
`entrapment_q` estimates `FDR = (ratio * n_entrap + 1) / max(1, n_real)`, the
empirical-null analog of target-decoy q that also feels the chimeric interference
in-silico decoys under-model (`rust/mumdia/crates/mumdia/src/fdr.rs:64`). A PSM is
classified as entrapment when its protein contains `entrapment_marker` (with
exclude/contaminant carve-outs, `rust/mumdia/crates/mumdia/src/stages/rescore.rs:577`).
Opt-in via `RescorerKind::Entrapment`; the default path relies on target-decoy FDR.

**final ID (vs seed)**. The identifications reported to the user, produced by
`rescore` from the extracted, feature-scored, competed PSMs with target-decoy
q-values. This is distinct from the seed search, whose only purpose is calibration
(see `seed`). See `q-value`.

**gate (`min_frag_corr`) and GateMode**. The extraction acceptance gate: a
candidate is rejected when its observed-vs-predicted fragment agreement falls below
`extract.min_frag_corr` (default 0.2; 0 disables,
`rust/mumdia/crates/mumdia-core/src/config.rs:406`,
`rust/mumdia/crates/mumdia-core/src/config.rs:522`). `GateMode` selects which
agreement score is thresholded: `ApexPearson` (single-apex-scan intensity Pearson,
the default), `PeakSpectral` (peak-integrated spectrum), `SpectralEntropy` (Li
similarity on sqrt-transformed intensities), `Coelution` (temporal correlation), or
`Combined` (spectral AND co-elution), at
`rust/mumdia/crates/mumdia-core/src/config.rs:551`. Gating on the rescorer's single
best discriminator regresses end-to-end, so a strong discriminator is the wrong
criterion for a gate.

**hyperscore**. The Sage-style score used by `search-seed`, defined as
`ln(matched!) + ln(1 + summed matched intensity)`
(`rust/mumdia/crates/mumdia/src/stages/search_seed.rs:413`). It ranks candidates in
the broad seed search only; it is not the final rescoring score.

**inverted fragment index**. A peak-major index mapping fragment m/z to the
candidates that predict a fragment there, so extraction work scales with
peak-candidate collisions rather than library size. The `Library` form is a flat
structure-of-arrays globally sorted by fragment m/z and bucketed, with candidates
ordered by precursor m/z within a bucket
(`rust/mumdia/crates/mumdia/src/index.rs:1`); the default `fragindex` matcher is a
separate log-binned CSR implementation of the same idea
(`rust/mumdia/crates/mumdia/src/matchers/fragindex.rs:1`). See `matcher`.

**iRT (indexed retention time)**. A run-independent predicted retention-time
coordinate carried per candidate as `predicted_irt`. `predict-frag` assigns it
natively or from a DeepLC sidecar
(`rust/mumdia/crates/mumdia/src/stages/predict_frag.rs:3`); `rt-im-train` calibrates
iRT onto this run's observed RT (linear or LOESS) and derives per-candidate RT
windows from the residuals (`rust/mumdia/crates/mumdia/src/stages/rt_im_train.rs:1`).
Peptidoforms with no predicted iRT are anchored at 0.0: this is the silent
construction default (`rust/mumdia/crates/mumdia/src/stages/predict_frag.rs:96`),
and the DeepLC path additionally logs a warning when the sidecar returns no iRT
(`rust/mumdia/crates/mumdia/src/stages/predict_frag.rs:298`). See `DeepLC fine-tune`.

**isolation window**. The precursor m/z range the instrument selected for
fragmentation in an MS2, given as `(target, lower, upper)`. `extract` and
`search-seed` use it to restrict candidates to those whose precursor m/z falls
inside the window (`Library::candidate_range`,
`rust/mumdia/crates/mumdia/src/index.rs:238`). In AIF/all-ion scans there is no
quadrupole selection, so a full-range window stands in (see `AIF`).

**manifest**. `manifest.json`, the per-run provenance record `run` writes: the
resolved config and its hash, per-stage model identities, and one
`ArtifactRecord` per output (path, schema, row count, content hash, producing
stage). Provenance is recorded, not required: because every stage reads
path-addressable inputs, no stage depends on the manifest to run
(`rust/mumdia/crates/mumdia-core/src/manifest.rs:1`).

**matcher (`fragindex`)**. The fragment-matcher backend shared by `search-seed` and
`extract`. The default `Fragindex` is a log-space-binned CSR inverted index that is
faster than the older `Bucketed` `Library::page_search` path with essentially
unchanged identifications (`rust/mumdia/crates/mumdia-core/src/config.rs:34`,
`rust/mumdia/crates/mumdia/src/matchers/fragindex.rs:1`). Both apply the same
f32-stored / f64-verify ppm predicate.

**MBR / transfer (match-between-runs)**. A cross-run step (Stage D3) that transfers
an identification confident in one run to a run where it scored below threshold,
gaining sensitivity. In MuMDIA it is a stub with config hooks and a Python sidecar
contract (`mbr_worker.py`); the default is no MBR and it needs at least two runs
(`rust/mumdia/crates/mumdia/src/sidecar.rs:157`,
`rust/mumdia/crates/mumdia-core/src/config.rs:845`). A transferred PSM is flagged
`is_transferred` with its q-value lowered
(`rust/mumdia/crates/mumdia/src/main.rs:249`).

**mokapot**. An opt-in Python rescoring sidecar (`RescorerKind::Mokapot`) run over
the PIN file with a logistic-regression default; the default `rescore.strict=true`
makes failure fatal. Setting strict false explicitly enables native compatibility
fallback
(`rust/mumdia/crates/mumdia-core/src/config.rs:941`,
`rust/mumdia/crates/mumdia/src/stages/rescore.rs:172`). See `percolator_lite`.

**nn_torch**. An opt-in PyTorch semi-supervised MLP rescoring sidecar
(`RescorerKind::NnTorch`, `nn_rescore_worker.py`) over the same PIN contract as
mokapot; a nonlinear Percolator/mokapot-style model with CV folds and iterative
positive re-selection that beats the linear model on the E.coli benchmark and gains
further when the extraction gate is opened
(`rust/mumdia/crates/mumdia-core/src/config.rs:107`).

**peptidoform**. A concrete modified peptide: a stripped sequence plus a specific
placement of fixed and variable modifications and a charge state. `peptidoforms`
expands stripped peptides into peptidoforms and emits each as a ProForma-lite
string with UniMod names (`rust/mumdia/crates/mumdia/src/stages/peptidoforms.rs:1`).
One peptidoform at one charge is one library candidate (see `candidate_id`).

**percolator_lite**. The native default rescorer (`RescorerKind::NativeTda`): a
deterministic Percolator/mokapot-style linear model that standardizes features and,
per CV fold, trains logistic regression on confident targets versus all decoys with
iterative positive-set re-selection, then computes target-decoy q-values
(`rust/mumdia/crates/mumdia/src/rescoring.rs:1`). It is always available and needs
no external dependency.

**pg_q_value**. See `q-value`.

**precursor**. The intact peptide ion selected for fragmentation, i.e. a
peptidoform at a charge, characterized by its precursor m/z. It is the grouping key
for `precursor_q` (peptidoform + charge,
`rust/mumdia/crates/mumdia/src/stages/rescore.rs:388`) and the reporting unit of
`peptides.tsv`, whose rows are precursors, not stripped sequences
(`rust/mumdia/crates/mumdia/src/stages/report.rs:93`). Under the default
`compete.group_by = precursor`, however, competition has already deleted the
charge and modification siblings by stripped peptide before rescore, so the
surviving rows are precursor-shaped but effectively one per base peptide, and
`precursor_q` then counts base peptides rather than precursors. See
`competition key` and `docs/11_compete_rescore_fdr.md`.

**prelim_score**. A cheap preliminary PSM score computed in `features` before
rescoring, combining matched-fragment count scaled by fragment correlation, mean
co-elution, log apex intensity, and an RT-error penalty
(`rust/mumdia/crates/mumdia/src/stages/features.rs:901`). It orders candidates for
`compete` (winner-take-all and margin-gated modes use it) and seeds the native
rescorer's positive-set selection; it is not the reported score.

**ProForma-lite**. The subset of the ProForma peptidoform notation MuMDIA parses
and emits: an optional `[Mod]-` N-terminal group, residues each optionally followed
by `[Mod]`, and an optional trailing `-[Mod]` C-terminal group, where a `[Mod]` is
a UniMod name or a signed mass such as `[+15.9949]`
(`rust/mumdia/crates/mumdia-core/src/mass.rs:138`). Peptidoform strings in the
library and PSM tables are ProForma-lite.

**protein group**. The set of protein accessions a peptide maps to, used as the
unit for protein-level FDR. The MVP grouping keys on the protein-accession-set
string (decoys carry a `DECOY_` prefix), and full parsimony/razor inference is a
later option; `pg_q_value` is the target-decoy q over the best PSM per group
(`rust/mumdia/crates/mumdia/src/stages/rescore.rs:306`).

**PSM (peptide-spectrum match)**. One hypothesis pairing a candidate peptidoform
with observed spectral evidence at a retention-time apex. `extract` emits one
apex-level PSM per surviving candidate; downstream stages carry PSMs through
features, competition, and rescoring. A PSM's `label` must be exactly `"target"` or
`"decoy"`, because the target-decoy null depends on exact labeling
(`rust/mumdia/crates/mumdia/src/fdr.rs:122`).

**q-value**. The minimum false discovery rate threshold at which a given PSM (or
peptide, or protein group) would be accepted, the trusted FDR estimate MuMDIA
reports. The native estimator is the conservative no-pi0 target-decoy form
`q = (n_decoys + 1) / max(1, n_targets)`, computed in score order with tied-score
blocks collapsed to one q and monotonized so q is non-increasing with score
(`rust/mumdia/crates/mumdia/src/fdr.rs:7`,
`rust/mumdia/crates/mumdia/src/fdr.rs:38`). `rescore` writes several q columns for
different grouping contexts (`rust/mumdia/crates/mumdia/src/stages/rescore.rs:461`):

| Column | Grouped over | Use when |
|---|---|---|
| `q_value` | per-PSM, pooled across all runs (alias of `global_q_value` / `experiment_psm_q`) | experiment-wide PSM FDR; cross-run precursor quant off a pooled rescore (`QuantQColumn::PsmQ`) |
| `experiment_psm_q` | per-PSM, pooled across all runs (identical to `q_value`) | explicit experiment-wide PSM FDR |
| `run_psm_q` | per-PSM, target-decoy re-run within each source run separately | per-run PSM FDR; the correct filter for cross-run quant off an experiment-wide rescore (`QuantQColumn::RunPsmQ`) |
| `precursor_q` | best PSM per (peptidoform + charge), among the rows competition left | precursor-level FDR, but a genuine precursor unit only under `compete.group_by = peptidoform_charge`; valid as a per-run quant filter only when rescore itself was single-run |
| `peptide_q_value` | best PSM per base (stripped) peptide | peptide-level FDR; single-run quant (`QuantQColumn::PeptideQ`, default). Note: under an experiment-wide rescore this is a GLOBAL per-peptide value carried on one best PSM, so it is wrong for per-run cross-run quant |
| `pg_q_value` | best PSM per protein group | protein-level FDR |

The `run_psm_q` / `experiment_psm_q` split and the `QuantQColumn` caveats are
documented at `rust/mumdia/crates/mumdia/src/stages/rescore.rs:340` and
`rust/mumdia/crates/mumdia-core/src/config.rs:772`.

The three grouped columns (`precursor_q`, `peptide_q_value`, `pg_q_value`) are
written only to each group's single winning row; every losing sibling keeps 1.0
(`rust/mumdia/crates/mumdia/src/stages/rescore.rs:721-728`). After an
experiment-wide rescore the grouping spans the whole experiment, so counting one
run on a grouped column understates it by roughly `1/n_runs`. Use `run_psm_q` as
the per-file unit there. Always name both the row unit and the q column when
reporting a count; see `docs/15_data_dictionary.md` for the per-column grouping
and `docs/17_troubleshooting.md` for the symptom.

**schema id (`classifier_feature_schema_id`)**. A blake3 hash of the ordered active
feature-column list, written to a companion `<features>.schema.json` so the
rescorer input is reproducible and can never be applied under a mismatched feature
set (`rust/mumdia/crates/mumdia/src/stages/features.rs:5`,
`rust/mumdia/crates/mumdia/src/stages/features.rs:233`).

**seed (search-seed)**. The native broad DIA-aware search whose purpose is
calibration, not final identification: it scores candidates by hyperscore to
provide confident PSMs for per-run mass recalibration and RT calibration
(`rust/mumdia/crates/mumdia/src/stages/search_seed.rs:1`). The final identifications
come from `extract` + `features` + `rescore` (see `final ID`).

**stripped sequence**. The bare amino-acid sequence of a peptide with all
modifications and charge removed (the base peptide). It is the grouping key for
peptide-level FDR (`peptide_q_value`) and is reported alongside the precursor in
`peptides.tsv` (`rust/mumdia/crates/mumdia/src/stages/report.rs:95`). Peptidoforms
are expanded from stripped peptides (see `peptidoform`).

**top-peaks cap (`--top-peaks-ms2`)**. The conversion-time limit on how many of
the most intense peaks of each MS2 spectrum are kept. It is destructive: the
truncation is written into `spectra_ms2.parquet`
(`rust/mumdia/crates/mumdia/src/stages/convert.rs:76-79`) and `extract` applies no
cap of its own, so the flag sets the MS2 peak budget for extraction, features, and
quantification, and peaks removed here can only be recovered by reconverting. It
defaults to `0` (uncapped) at both conversion entry points and is
acquisition-specific: a value tuned on one acquisition scheme can delete the
majority of another run's fragment evidence. Distinct from
`search_seed.top_n_peaks` (default 300), which is non-destructive and bounds only
the seed's index-probing cost. See `docs/04_convert.md` for the canonical
treatment.

**target-decoy FDR**. The community-standard false discovery rate control MuMDIA
uses: score real (target) library entries against paired decoys and estimate the
FDR at each score threshold from the decoy count. The engine reports native
target-decoy q-values at PSM, peptide, and protein-group level as its trusted FDR
estimate (`rust/mumdia/crates/mumdia/src/fdr.rs:1`). A decoy-free library makes
these q-values invalid, so `Library::load` fails loudly when it finds no decoys
(`rust/mumdia/crates/mumdia/src/index.rs:152`). See `q-value`, `decoy`,
`entrapment`.
