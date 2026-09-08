# MuMDIA full code review — 2026-09-08

Baseline: `21b8ac7`, the head of `review/e-followup` (work packages A to E). Whole
repository rather than a diff: engine, core, IO, the Python workers and library helpers,
the desktop application, the tests, the benchmark utilities and the delivery
configuration. Twelve parallel finders and one verification sweep; the maintainer
re-verified eleven findings against the source before recording them here.

Twenty-one findings. Six are confirmed correctness or security defects in code that
predates this review series, five are regressions introduced by packages A, D and E, and
ten are second-tier robustness, coverage and delivery problems. Package F below closes
the first eleven.

**Verdict at the time of review: do not tag a release from this tree.** Two of the
findings produce a plausible wrong answer rather than an error, and one is a code-execution
path through an untrusted working directory. The 2026-09-08 HYE release check passing does
not contradict that: it exercised the library path with calibration available and no
match-between-runs transfer, so it could not reach any of them.

## Findings

Priority P1 means fix before relying on the affected workflow. *Reproduced* means an
executed probe demonstrated the failure; *source-confirmed* means the trigger and the
consequence follow from the implementation without a full run.

| | Priority | Finding | Origin |
|---|---|---|---|
| F1 | P1 | prescan reads the no-calibration sentinel as "cannot be screened" and discards the entire library at exit 0 | pre-existing |
| F2 | P1 | a malformed `is_transferred` silently deletes every transfer from both TSV reports | pre-existing |
| F3 | P1 | `sidecar::resolve_script` tries the working directory first, executing a planted worker | pre-existing |
| F4 | P1 | `Loess::predict` indexes out of bounds when the query is NaN | pre-existing |
| F5 | P1 | three rescorer backends, two standardisations, and an unchecked fold-key companion | pre-existing |
| F6 | P1 | the output-over-input guard is wired into 2 of 18 stages | pre-existing |
| F7 | P2 | the LOESS boundary extrapolation slope is unbounded and may be negative | package D (#65) |
| F8 | P2 | a desktop stop can kill a recycled process id after the reap | package E (#66) |
| F9 | P2 | the conversion lock spins, mis-detects staleness three ways, and leaks partial files | package E (#66) |
| F10 | P2 | a position-free pair key deletes positional isomers that predicted correctly | package A (#62) |
| F11 | P2 | ten further robustness, coverage and delivery findings, listed at the end | mixed |

**F1. prescan discards the library on the documented no-calibration path. Reproduced.**

`rt_im_train::candidate_window` returns `(NaN, -inf, +inf)` whenever calibration is
unavailable and its doc comment calls the infinite bounds recall-safe. prescan's guard
rejected exactly that: `if !lo.is_finite() || !hi.is_finite() { return None; }`. This is the
FASTA/MS2PIP failure CLAUDE.md already records, where the seed search finds no confident
PSMs and every window row is infinite; prescan then wrote a zero-row survivors table with a
NaN target/decoy ratio and exited 0. The single-label bail could not see it, being gated on
`surv.len() > 1000`. A candidate simply absent from `run_windows` was dropped the same way,
uncounted.

**F2. A malformed transfer flag deletes every transfer from the reports. Reproduced.**

`mbr_worker.py` writes `is_transferred` through pandas, so a nullable boolean dtype, a null
or int8 0/1 reaches the report stage. `match t.bool(...) { Err(_) => vec![false; n] }` then
degraded acceptance to the q threshold alone: every transferred identification vanished
from `peptides.tsv` and `proteins.tsv`, indistinguishable from an MBR run that transferred
nothing, while the parquet still carried them. Four sites. The same absent-versus-malformed
mistake was removed from `quant` in package A and from `audit` in package D, and left in
the stage that decides what users read.

**F3. Script resolution prefers the working directory. Source-confirmed.**

`python::resolve_script_dir` was reordered in an earlier pass to config-relative,
exe-relative, working-directory-last, with a comment explaining the attack: the shipped
default is the relative `"scripts"`, which both example configurations carry, so unpacking
a dataset archive and running with an example config could execute a worker the archive
contained. That resolver only claims a directory holding `mbr_worker.py` or
`deeplc_worker.py`, so a directory containing any of the other ten workers passed through
unresolved, and `sidecar::resolve_script`, used at all eleven call sites, still tried the
working directory first.

**F4. A NaN query indexes before the start of the grid. Reproduced.**

With `x = NaN` both boundary comparisons are false and `partition_point` returns 0, so
`grid_x[j - 1]` underflows: a panic in debug, an out-of-bounds read in release. One library
row with a null `predicted_irt` is enough, because the parquet reader maps a null f32 to
NaN. The anchor loop in `rt_im_train` filters for finite values; the whole-library
application does not, and `align` and `search_seed` share the exposure.

**F5. The rescorer's three backends do not agree. Reproduced.**

In-memory TSV standardised with median/IQR while in-memory parquet and the streaming
memmap used mean/std, so the same pool scored differently depending on `rescore.handoff`
and on which side of `MUMDIA_NN_STREAM_GB` the matrix fell, with nothing logged. The
docs/28 comparison cited as identical identifications was measured on a 4.03 GB matrix, so
both arms streamed and the divergence was masked. Separately, the `MUMDIA_NN_FOLD_KEYS`
companion was sliced without a length check: numpy returns a short array rather than
raising, the tail rows land in no fold, are never scored, keep the zero initialiser and
emerge from the final rank-normalisation as a plausible tied mid-rank score, which
satisfies the caller's completeness contract so `rescore.strict` does not catch it.

**F6. The output-over-input guard is almost unwired. Reproduced.**

`refuse_output_over_input` had three references in the workspace: its definition and two
call sites. `mumdia compete --features features.parquet --out features.parquet` opens the
table footer-only, streams the surviving rows, and publishes by rename after the read has
finished, so nothing errors and the widest artifact of the run is replaced by the competed
subset at exit 0. Recovering it means re-running extract and features, the tallest stage.
The same shape applies to `rescore`, `features`, `quant`, `extract`, `search_seed`,
`rt-im-train` and `audit`.

**F7 to F10** are regressions from packages A, D and E and are described in the fix table
below.

## Package F

One stacked pull request on `review/e-followup`, with a test per finding.

| | Change | Test |
|---|---|---|
| F1 | an unbounded window means "screen over the whole gradient", the recall-safe reading of the sentinel; candidates with no window row are treated the same; both are counted and warned about; screening every candidate away is now an error naming the likely causes | `prescan` counts and bail |
| F2 | `transfer_columns` reads present columns in their declared type and errors on a type mismatch; only an absent column falls back | `a_malformed_transfer_column_is_an_error_not_a_silent_loss_of_every_transfer`, `a_table_that_never_saw_mbr_reports_no_transfers_without_complaint` |
| F3 | absolute directories as given, then the executable's directory, then `<exe>/scripts`, and the working directory last | `the_shipped_directory_beside_the_binary_wins_over_the_working_directory` and two more |
| F4 | `Loess::predict` returns NaN for a non-finite query; `rt_im_train` treats a non-finite library iRT as "no calibrated RT" and counts it | `a_nan_query_returns_nan_instead_of_indexing_out_of_bounds` |
| F5 | one mean/std standardisation in all three backends, which leaves the shipped parquet default and every published benchmark unchanged; the fold-key companion is length-checked; the backend-size estimate counts feature columns by name | `a_short_fold_key_file_is_refused_rather_than_leaving_rows_unfolded` |
| F6 | the guard is wired into `search_seed`, `rt-im-train`, `extract`, `features`, `compete`, `rescore`, `quant` and `audit`, for every output each writes | `writing_the_output_over_the_input_is_refused` |
| F7 | the extrapolation slope is the secant of the fitted curve over its end decile, clamped non-negative and to at most four times the global least-squares slope | `boundary_extrapolation_stays_monotone_and_bounded_under_noise`, on noisy anchors rather than a noiseless quadratic |
| F8 | the waiter retires the process id the instant `wait` returns, before it reads the output directory | `a_stop_after_the_reap_has_no_pid_to_kill` |
| F9 | bounded take-over attempts with a pause and a deadline; a token written and read back, so only the real holder owns or removes a lock; a future modification time counts as fresh; the partial-file probe matches this destination only; abandoned partials of this destination are swept under the lock | five tests in `raw::tests` |
| F10 | the direct misses and the rows dropped for sharing a pair key are counted separately, and exceeding 2% of the library is an error naming the sidecar | `a_large_unpredicted_fraction_is_a_failure_not_a_warning`, `a_positional_isomer_still_shares_its_pair_key_by_design` |

### Two changes deliberately not made

**F10 does not make the pair key position-aware.** `M[Oxidation]PEPTIDEMK` and
`MPEPTIDEM[Oxidation]K` share a base peptide, a charge and a modification multiset, so a
miss on either drops both. Adding position would be worse rather than better: a reverse
decoy carries its modifications at mirrored positions, so a positional key would stop
matching a target to its decoy, and a target could be dropped while its decoy stayed. That
is an FDR defect in exchange for a sensitivity one. The collateral is counted and bounded
instead, and a test pins the trade so a future change has to argue with it.

**F9 does not add a recipe-keyed conversion cache.** Unique temporary names, an owned lock
and the staleness rules cover the reproduced failure. Two different conversion recipes
aimed at one destination name remain a documented limitation of writing beside the input.

## F11: the second tier, not in package F

Recorded for a later pass, in rough order of value:

- `matchers/binning.rs`: `LogBins::new` has no bound on the derived bin count while
  `validate()` only requires a positive tolerance, so `frag_tol_ppm: 2e-5` asks for 547 GB
  and dies on an allocation failure rather than on a message naming the setting.
- `digest.rs`: `collision_safe_decoy` XORs `fnv1a(pep)` and the identical term cancels
  inside `make_decoy`, so the scramble seed has no peptide dependence and every peptide of
  one length gets the same positional permutation. The shipped `reverse` default only
  correlates retried decoys; `strategy = scramble` gets a structured null.
- `ci/smoke.sh`: the NaN-retention-time regression compares two reports through process
  substitution with no existence check, so both files missing is a passing `diff`.
- `sbom.cdx.json`: the committed SBOM, which ships in every release archive, references two
  components it does not define, so it fails CycloneDX validation and `--check` regenerates
  the same document.
- `tests/python/test_predictor_workers.py`: the DeepLC import-order test asserts on two
  Windows-only strings and no CI job is both Windows and DeepLC-installed, so it cannot
  fail.
- `features/similarity.rs`: `rank_overlap_top3` and `frac_top3_predicted_observed` divide by
  a hardcoded 3.0 rather than the number of positive predicted fragments, so they encode how
  many fragments were predicted; `mass_uncertainty.rs` computes the same quantity correctly.
- `config.rs`: the `max_feature_matrix_gib` doc comment, copied into the CI-checked
  configuration reference, still describes the f64 layout the ceiling no longer uses.
- `run.rs`: the audit is hardcoded at q 0.01 against the report's `quant.q_threshold`.
- `build.rs` cannot distinguish a clean tree from a failed `git status`, so a dirty tree can
  be stamped clean.
- `check_doc_refs.py` skips `../..`-prefixed links, `make_fixture_mzml.py` truncates rather
  than spans the m/z range when planting peptides, `cargo install tauri-cli --version "^2"`
  is a range under a comment calling it pinned, and one workflow pins
  `softprops/action-gh-release` at two different revisions.

## Verified and sound

Recorded because a review that only lists defects invites the same checks again: all
seventeen documented defaults match the code; no `HashMap` iteration feeds a float
reduction or an output ordering; every configuration struct denies unknown fields and all
five shipped configurations parse; the four DeepLC 4.1.1 enforcement points agree with the
Rust constant and both Python literals match it; neither sidecar leaks a target/decoy label
into the feature matrix; every worker's argument surface matches the arguments its Rust
caller builds; the experiment report headers match the writer in order; and
`split_by_source` runs before per-run quantification.
