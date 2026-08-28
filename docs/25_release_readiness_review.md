# 25. Release readiness review

Pre-release audit of the tree at `d1f3f3e` (PR #47, all eight CI checks green).
Written 2026-08-28. Six parallel audits covering correctness, robustness,
interface stability, security and supply chain, test adequacy and documentation
accuracy, plus direct probing of the built binary.

Findings marked **verified** were reproduced by running the tool, by compiling a
kernel standalone, or by reading the cited line. Where a mechanism is certain but
its frequency on real data is not, that is stated. Severity: **BLOCKER** means do
not tag; **MAJOR** means fix before tagging or record as a known limitation;
**MINOR** and **NIT** can follow the release.

## Status, 2026-08-28 (later the same day)

Everything below was found at `d1f3f3e`. The findings have since been worked
through; this section records what changed, so the report can be read as a record
of the audit rather than a to-do list. Section numbers refer to the sections below.

**All four blockers are closed.** One finiteness pass at library load, with the
NaN-permeable guards fixed behind it (section 1); interpreter and script-directory
resolution moved into `load_config`, so every entry point behaves alike, and an
unused `"auto"` role is cleared rather than left as a literal (2, 5.1);
`cargo update -p mzdata -p crossbeam-epoch` clearing all three advisories, with
`cargo audit` now a required CI job (7.1, 7.2); and the two disclosure lines
rewritten (`4f999a0`).

**Measured after the fact, and one default reverted.** The AIF identification arm
and a 2x2 gate-by-rescorer A/B (`bench/README.md`) showed that
`extract.gate_min_score = 0.6` -- promoted here from a documentation claim about
`native_tda`'s optimum -- costs 4.4-4.7% of peptides at an unchanged decoy
fraction, because the sweep behind that claim predates the current defaults. It is
back to 0.2. An entrapment arm then measured the section-3 fixes as calibrated
(empirical FDP 0.0098 at a 1% threshold).

**Also closed.** The apex fallback, by promoting `apex_evidence_rank` (2). All
three entrapment defects, so the FDR-validity instrument can be trusted again (3).
All seven quantification findings, including the pooled-table refusal and the
inert fixed window (4). Every interface decision, as hard renames with no aliases
(5.2-5.5). Every robustness finding except the two named below (6). The mass and
constants reference test, verified to catch a substituted proton mass (8.1). The
apex-source assertion and whole-tree determinism comparison, which found all 18
artifacts byte-identical (8.3, 8.4). The real sidecar-import CI job (8.5). The
`THIRD_PARTY_LICENSES.md` bundle, `forbid(unsafe_code)`, all 23 actions SHA-pinned,
`build-essential` purged from the runtime image, the `.dockerignore` allowlist, and
the pushed image now being the tested image by construction (7). All twelve
documentation claims (9).

**Decided, 2026-08-28.** Two items that needed a call from the maintainer:

- **`main` is protected.** Eight always-running CI checks required (fmt+clippy,
  cargo audit, build+test on all three platforms, both smoke jobs, python
  sidecars+configs), branches must be up to date, force pushes and deletions
  blocked. Deliberately NOT enforced on admins and no required approvals, so a
  single maintainer is never locked out of their own repository; the two
  conditional jobs (`cross-platform byte equality`, which depends on the smoke
  jobs, and the two `real sidecar imports` jobs) are not required, because a
  skipped required check blocks a pull request indefinitely.
- **Licensing stays Apache-2.0**, and the notice bundle now reproduces actual
  notices rather than metadata: per-crate copyright assertions for 126 of 173
  crates and every distinct `NOTICE` file verbatim, extracted from the crates' own
  files in the cargo registry cache. `cargo-about` and formal legal sign-off were
  declined as a decision, not missed. The extraction is mechanical and says so:
  whether it satisfies a given distribution's obligations is a question for
  whoever signs the release off.

  The 47 crates without a recovered line ship no copyright assertion in a licence
  file at all; their repositories are linked in the inventory. An earlier run
  claimed 166 of 173, which was wrong: the pattern was matching the body of the
  Apache-2.0 text itself ("the copyright notice that is included in or attached to
  the work") as though it were an assertion.

The remaining governance items from the same review -- default Actions token
permissions still `write`, and Actions still permitted to approve pull requests --
were not part of that decision and are untouched.

**Deliberately still open**, and why:

| finding | why not now |
|---|---|
| signal handling (6.2) | needs a `ctrlc` dependency. Atomic writes make cancellation leave at most an inert `.tmp-<pid>` file rather than rubble under the canonical name, which was the substantive harm; a handler only tidies up. |
| incremental manifest (6.2) | a crash still discards the provenance of completed stages. Real, but it is a design change to the manifest's write points, not a guard. |
| `--work-dir` on standalone stages (6.4) | the fixed handoff FILENAMES were the corruption risk and are now PID-qualified. A configurable directory is convenience. |
| transparent rescore batching (6.7) | sub-batching changes which PSMs share a pooled `q_value`, so it is an operator decision. The size is now logged and boundable, which turns an OS kill into an error at startup. |
| `convert`, `align`, `run`, `run_experiment`, `sidecar` unit tests (8.2) | the coverage inversion is real and unaddressed. The end-to-end test is the only cover for these. |
| `bench/README.md` and four `docs/19` `file:line` citations (9) | needs re-deriving each number against its arm, which is measurement work rather than editing. |
| the 9.5 GB of stale `sidecar_work/` in the repository root (6.4) | untracked scratch on the developer's disk; deleting it is the owner's call, not this branch's. |
| FDR calibration at scale, entrapment validity as a result, interference, quantification bias | needs real data and entrapment arms. Note the ordering: section 3 had to land first, or those measurements meant nothing. |

Two things worth flagging that this work introduced rather than fixed. Three
defaults changed (`apex_evidence_rank`, `gate_min_score`, `emit_pin`) and the
NnTorch CV-fold key changed, so **`nn_torch` counts and every entrapment number
need re-measuring**. And the CLI and two config keys were renamed with no
compatibility aliases, so an old invocation or config fails loudly; local configs
need two edits, given in the rename commit.

## Verdict at the time of the audit

**Do not tag `v0.1.0` yet.** The engine's core is in better shape than its
perimeter: zero `unsafe` in any workspace crate, byte-reproducible output across
four machines, two operating systems and every thread count tested, no network
stack in the dependency graph, and concurrency that is architecturally clean
(no `Mutex`, no `static mut`, no parallel float reduction anywhere). What is not
ready is everything around that core.

| # | Blocker | Status |
|---|---|---|
| 1 | No finiteness validation at either external-input boundary. A Parquet NULL becomes `NaN`, and the guards that should reject it are NaN-permeable, so a NaN silently means "unbounded" or "matches everything" rather than "reject". | open |
| 2 | `"auto"` interpreter discovery is never applied on standalone stages: `rescore`, `predict-frag` and `mbr` try to execute a program literally named `auto`. All three shipped example configs use `"auto"`. | open |
| 3 | Two high-severity advisories (CVSS 7.5) in `quick-xml` 0.30.0, reached through `mzdata`, which is the mzML parser. `SECURITY.md` puts untrusted input files in scope, and there is no advisory scan in CI. | open |
| 4 | Two tracked lines described the contents of an unpublishable document and an unpublished collaboration's topic, and shipped in every release archive. | **fixed in `4f999a0`**, including the PR description on the remote |

Blocker 2 is a defect introduced by the release work itself, when interpreter
resolution was wired into the two orchestrators and not into `main.rs`.

Beyond the blockers, one finding deserves separate billing because of what it
undermines: **the entrapment path, which `docs/20` designates as the instrument
for validating any new sensitivity default, is not currently trustworthy**
(section 3). Every gated change in `CLAUDE.md` is gated on entrapment evidence.

## 1. Finiteness: one root cause, many consequences. BLOCKER.

MuMDIA enforces a finiteness contract at **every internal and sidecar boundary**
and at **neither external-input boundary**.

| Boundary | Guard |
|---|---|
| predict-frag sidecar output | `predict_frag.rs:139-154` bails, naming the peptidoform |
| rescore feature matrix | `rescore.rs:141-155` bails, naming the feature and row |
| rescore sidecar scores | `rescore.rs:1086-1088` bails, naming the row |
| feature computation | `features.rs:153` coerces non-finite to 0.0 |
| **user-supplied library Parquet** | **none** |
| **spectra Parquet** | **none** |

`table.rs:399-402` and `:419-422` map a Parquet NULL to `f64::NAN` / `f32::NAN`
with no error. `Library::load_with` then validates structure only: `candidate_id`
contiguity, ascending `precursor_mz`, both labels present and well-formed. No
value is checked for finiteness. `scripts/_lib_io.py`, which exists as the
correctness chokepoint for helper-written libraries, normalises encoding and
validates no values at all.

The consequences are not "a NaN propagates to a NaN result". They are that the
guards written to reject bad input are false for NaN, so a NaN is **accepted**:

- **`index.rs:221`**: the ascending-m/z hard error is `if prec_mz[c] < prec_mz[c-1]`,
  false for NaN. So a NaN passes the very check whose comment says it exists to
  stop the index returning wrong candidate windows. `candidate_range`'s
  `partition_point` then runs on a slice that is not partitioned by its
  predicate. Worked example: `prec_mz = [400, 500, NaN, 700, 800]` with window
  `[650, 750]` gives `lo = hi = 2`, so `extract.rs:296` skips the window and the
  precursor at 700, genuinely inside it, is never extracted in any scan of that
  window for the whole run. Fix: `if !(prec_mz[c] >= prec_mz[c-1])`.
- **`extract.rs:306, 710, 850, 904, 975, 1086, 1158`** (8 sites): the RT-window
  guard is `if rt < rt_lo[c] || rt > rt_hi[c] { return; }`. Both comparisons are
  false for NaN, so a NaN window accepts **every** scan. The chain is complete and
  each link was read: a NaN `predicted_irt` is correctly excluded from the RT
  training anchors (`rt_im_train.rs:151-155`), so `cal.json` looks healthy, but
  `predict(NaN)` is NaN, `candidate_window` writes `(NaN, NaN, NaN)` into
  `run_windows`, and those candidates are then searched across the whole gradient
  with no RT prior and no warning, entering the training and FDR population.
  `prescan.rs:337-339` guards this exact thing; extract has the same data and no
  such check.
- **`constants.rs:93-96`**: `within_ppm` returns `true` when one operand is NaN,
  because `f64::min`/`max` return the non-NaN operand. This is the canonical
  predicate the whole fragment-index verification path uses. One NULL fragment
  `mz` saturates to bin 0 and every low-m/z peak "verifies" against it,
  fabricating matched-fragment counts and dot-product contributions in both the
  seed hyperscore and extract's claimant list.
- **`features.rs:1517`**: `partial_cmp(...).unwrap()` on `predicted_intensity`,
  inside a rayon closure. The same file does the same sort NaN-safely at `:509`
  and `:1435`. One NULL cell panics the features stage *after* extract, the most
  expensive stage, with a bare `unwrap` message that gives no hint that a library
  value was at fault.
- **`fdr.rs:27-43, 88-105`**: one NaN score makes `target_decoy_q` **infinite-loop**.
  Verified by compiling the kernel standalone: the tie-block walk is
  `while end < n && scores[order[end]].0 == s`, and `NaN == NaN` is false, so
  `end` never advances. `rescore.rs` validates finiteness before calling it;
  `rescoring.rs:139` and `search_seed.rs:124` do not.
- **`fragindex.rs:71-75`**: one non-finite fragment m/z collapses the *whole*
  index range to `[1.0, 2.0]` rather than dropping the offending value, clamping
  every real fragment into one bin and turning the search into a linear scan over
  the entire posting list per peak. No panic, no error, no wrong answer: an
  unbounded hang.

**Why this is a blocker rather than a robustness nit.** Imported-library mode is a
first-class documented mode, is the recommended profile in the README, and skips
`predict-frag`, the one stage with a finite guard. A user bringing a Spectronaut,
MSFragger-DIA or Skyline library converted through pandas gets NaN from any
left-join, missing column or empty cell. The failure is then silent and
directional: candidates vanish from windows, or match everything.

**The fix is small and central**: one finiteness pass in `Library::load_with` in
the style of `rescore.rs:141-149`, a value check in
`_lib_io.write_engine_parquet`, and the `!(a >= b)` inversion at `index.rs:221`.
That one pass closes this section and most of the reachable panic census.

Related, same file, not NaN: `i32`/`i64`/`u32` on a NULL push the raw buffer value
with no error (`table.rs:437, 459, 481`), and `bool()` never checks nulls at all
(`:495-499`).

## 2. Apex selection can silently choose the earliest scan in the window. BLOCKER-class mechanism, unmeasured frequency.

`extract.rs:1874-1908`. `best_sig` starts at `f32::NEG_INFINITY` and the update is
strict (`score > best_sig`). Under the shipped default (`apex_evidence_rank = false`,
`config.rs:785`) the score is `sig_sum * prior`, where `sig_sum` is the *observed*
intensity of only the top-`k_sig` (default 3) **predicted** fragments. If none of
those three is observed at any qualifying scan, `sig_sum == 0.0` everywhere, the
first qualifying scan is accepted at score `0.0`, and no later `0.0` replaces it.
`groups` is RT-ascending, so the winner is the **lowest-RT qualifying scan**. The
RT prior cannot rescue it: the combination is multiplicative, so `sig_sum == 0`
annihilates the prior in exactly the case the prior exists for.

The qualifying set is wide by default: at `apex_count_window = 1`, every scan with
a matched-fragment count within `apex_count_tol = 1` of the maximum qualifies,
scattered across the whole window.

Reachable configuration: a candidate whose three highest-predicted ions fall below
the MS2 low-m/z cutoff, with scattered interference matching two fragments near
`rt_lo`. `presence_min_fragments` is satisfied by the true peak elsewhere, the row
is accepted, and `apex_rt` lands at `rt_lo`, off by up to `w_rt` (150-211 s on the
AIF benchmark). For a candidate absent from `run_windows` the window is unbounded
and the error is the whole gradient. It propagates into `prelim_score`, which
decides the pre-FDR competition winner, into `rt_error_abs` and
`log_apex_intensity`, and into quant's apex hint, hence the reported abundance.

**The mechanism is certain; the frequency is not.** Settle it by counting
candidates with `best_sig <= 0.0` on the AIF run. `apex_evidence_rank = true`
fixes it (`score = (n_frag + tie) * prior`, always positive) and is currently
default-off and benchmark-gated. If the count is non-trivial, that gate should be
reconsidered as a bug fix rather than a sensitivity change.

Two adjacent defects in the same routine:

- **`extract.rs:1874-1875`**: when *no* group qualifies, `apex_rt` becomes the RT
  window's left edge and `apex_intensity` stays `0.0`. Unreachable at
  `apex_count_window = 1`, but reachable as soon as the window exceeds 1, because
  `smoothed` becomes a rolling **sum** whose maximum can be attained at a scan
  whose own count is 0. `--profile dia` and every `config.*.json` in this tree set
  `apex_count_window: 5`.
- **`extract.rs:1848`**: `apex_count_tol` is subtracted from the rolling *sum*,
  not from a per-scan count, so its effective slack shrinks by roughly
  `1/apex_count_window` and two knobs documented as independent are coupled.

## 3. The entrapment path is not currently trustworthy. MAJOR, and it undermines every gated decision.

`docs/20` makes entrapment the instrument for promoting any sensitivity default,
and `CLAUDE.md` gates seven changes on it. Three independent defects sit on that
path.

**3.1 In entrapment mode the native rescorer trains in-silico decoys as candidate
positives.** `rescore.rs:282, 293` calls
`native_scores(&p, &feats, &is_entrapment, &base, &prelim)`, passing
`is_entrapment` in the `is_decoy` slot. Inside `percolator_lite`
(`rescoring.rs:144-153`) a row with `is_decoy == false` takes the
`q[k] <= train_fdr` branch, so a *decoy* with a low internal q is selected as a
**positive** training example, and `target_decoy_q` inside the loop counts decoys
as targets. This is the path taken whenever `rescore.python` is unset, and on a
GBM failure with `strict = false`. Decoy scores are systematically inflated, which
then feeds 3.2.

**3.2 An in-silico decoy can win a base-peptide group and delete the real target
from the analysis.** `grouped_q` (`rescore.rs:689-745`) groups on
`base_peptide_id`, which a target and its decoy **share** (verified on both library
paths: `make_shift_decoys.py:34` and `make_reverse_decoys.py:148` both
`dprec = tprec.copy()`; `peptidoforms.rs:216-220` on the native path). Under
`qmode == Entrapment`, `incoming_null` is `is_entrapment`, which is `false` for
both the target and its decoy, so the group winner is simply the higher score. If
the decoy wins, its tuple counts toward **neither** `ne` nor `nr`, the group
vanishes from the analysis entirely, and the real target gets
`peptide_q_value = 1.0`. Both `target_peptides_at_1pct` and the leak metric
`entrapment_peptides_at_1pct` are computed from `peptide_q`, so the primary
FDR-validity tool is measured on a population that decoys have partially deleted,
and **the leak count is under-reported**. Fix: exclude `is_decoy` rows from the
group competition when `qmode == Entrapment`.

**3.3 `rescore.entrapment_ratio` is unvalidated, and `0` makes every PSM pass.**
`Config::validate` checks `gate_min_score`, `fixed_window_s`, `baseline_quantile`
and `train_fdr`, but not this. Measured by compiling the `entrapment_q` kernel
standalone:

```text
ratio =  1.0 -> q = [1.0, 1.0, 1.0, 1.0, 1.0]
ratio =  0.0 -> q = [0.333333, ...]                     # = 1/n_real for every row
ratio = -1.0 -> q = [-0.5, -0.5, -0.5, -0.5, -0.333333] # negative q
```

At `ratio = 0` the estimate collapses to `1/n_real`, so with more than 100 real
targets every row passes at 1% with no null at all. A negative ratio yields
negative q, and both `count_targets_at_q` and `passes_quant_filter` accept it. A
user who computes the ratio the wrong way round gets a silently near-zero FDR.

**Consequence for the project, not just the code.** Any entrapment measurement
taken through the native rescorer before these are fixed should be re-run. Where
`nn_torch` was used, 3.1 does not apply, but 3.2 and 3.3 still do.

## 4. Quantification

**4.1 A pooled scored table plus one run's chromatograms silently emits duplicate
identical quantities. MAJOR.** `quant` keys `apex_by_cid`, `q_by_cid`, `areas` and
`frag_areas` on `candidate_id` alone (`quant.rs:455, 575, 763, 845`) and never
checks `source`. `candidate_id` is the library index and repeats across runs.
`mumdia quant` accepts any `psms_scored`. Feed
`mumdia rescore --competed out/*/psms_competed.parquet` output, which is exactly
what the recorded Astral recipe step 2 produces, straight to `mumdia quant` with
run 1's chromatograms and you get `n_runs` rows per precursor, all carrying run 1's
quantity and run 1's apex, so **every cross-run fold change comes out at 1.00**. No
error, no warning. `run_experiment.rs:495` does split by `source` first and the docs
say to, but nothing enforces it. A one-line check on `source.iter().max() > 0`
closes it.

**4.2 The fixed-window integration is silently inert when `bound_peak = false`.
MAJOR.** `integration_apex` is only populated under `p.cfg.bound_peak`
(`quant.rs:557`); otherwise it is NaN, so the `fixed` predicate at `:637` is false
and quant integrates the **whole trace**. `Config::validate` warns about
`fixed_window_s` with `fixed_scan_halfwidth`, and about `baseline_subtract` with no
fixed window, but not about this. So
`{"quant": {"bound_peak": false, "fixed_window_s": 5.0}}`, a natural way to ask for
a pure fixed window with no descent walk, silently returns whole-chromatogram
areas.

**4.3 MBR transfers are invisible to the default quant filter and to `report`.
MAJOR.** `mbr_worker.py:286` lowers `q_value`, `run_psm_q` and
`experiment_psm_q`. `quant.q_filter` defaults to `PeptideQ`, and
`report.rs:55, 101, 137` filters `peptides.tsv` on `peptide_q_value` and
`proteins.tsv` on `pg_q_value`. MBR touches none of those.
`run_experiment.rs:516` forces `q_filter = PsmQ`, so the orchestrated path works,
but `mumdia mbr` followed by a manual `quant` or `report` with a default config
reports MBR as having done nothing. This is the same class of bug the worker's own
comment records having fixed for `run_psm_q` two days ago; it survives for the
**default** column. `global_q_value` also diverges from `q_value` after MBR despite
being documented as a byte-identical alias.

**4.4 `mbr_worker.py:266, 289` writes Parquet with a bare `pq.write_table`.
MAJOR.** Not through `_lib_io.write_engine_parquet`. Under pandas 3 the string
columns come out `large_string`, which the engine rejects with
`column 'peptidoform' is not utf8`. This is the exact failure `_lib_io` was created
to prevent and that `c7f910c` fixed in the four library helpers; the MBR worker was
missed.

**4.5 `FragmentSelection::Predicted` ranks a NaN `predicted_intensity` first.
MINOR.** `total_cmp` orders NaN above every real value, so in descending order NaN
sorts to the front. The code guards the absent-column case explicitly, with a
comment naming this hazard, but a column that is *present* with NULLs becomes NaN
and those fragments are preferentially selected into the top-N.

**4.6 `fixed_window_s` and `fixed_scan_halfwidth` are not interchangeable. MINOR.**
The scan form always integrates `2h+1` samples; the seconds form integrates
`floor(2H/c)` or `floor(2H/c)+1` depending on where the apex falls on the scan
grid, so the integrated span varies by one full cycle between otherwise identical
precursors: 6 versus 7 samples at `H = 5 s, c = 1.64 s` (17% of the span), 3 versus
4 at `c = 3.4 s` (33%). The recorded note treating them as equivalent across
instruments is wrong. Inherent to integrating without edge interpolation, not a
coding error, but it should be documented.

**4.7 The fixed window has no distance guard on its anchor sample. MINOR.**
`fixed_window_indices` always returns at least the sample *nearest* the apex,
however far away, and `trapezoid_fixed_opts` can then be called on a 1-sample
slice, which returns the raw **height**, not an area. `peak_window` explicitly
guards this analogous collapse with a comment about corrupting LFQ quantities; the
fixed path does not. Not reachable at the shipped `emit_window_grid = true`, since
the grid densely spans the window; reachable with the sparse observed-scan list.

## 5. Interface stability: what a tag freezes

**5.1 `"auto"` on standalone stages. BLOCKER, verified by running it.**

```text
Error: rescore: NnTorch sidecar failed (spawning sidecar failed:
       auto scripts/nn_rescore_worker.py: program not found)
```

Only `stages/run.rs:103` and `stages/run_experiment.rs:282` call
`python::resolve`; `main.rs` never does. Companion defect, also verified: the same
config is accepted by `run` and **rejected** by `run-experiment`, which checks
`Path::exists` on any non-`None` interpreter including the literal `"auto"` for a
role it does not need:

```text
Error: mbr.python points at an interpreter that does not exist: auto
```

`sidecar_script_dir` has the same split: the orchestrators resolve it against the
config file's directory and the executable's directory, standalone stages use a
narrower fallback, so the resolution order `configs/README.md` documents holds for
two subcommands only.

**5.2 A fourth silently inert config field, undocumented. MAJOR.** An exhaustive
check of all 166 fields found exactly four with no use outside `config.rs`. Three
are the documented, warned MBR fields. The fourth, **`rescore.percolator_bin`, is
accepted, does nothing, and warns nothing**, and no document lists it. Under
`deny_unknown_fields` a user reasonably assumes an accepted key is honoured.

**5.3 Naming inconsistencies. MAJOR, and free to fix now.**
`--lib-precursors` on five subcommands versus `--lib-precursors` on the
orchestrators; `--out` on eleven versus `--out-dir` on four plus ten bespoke
`--out-*`; extract writes `--out-chromatograms` while features and quant read
`--chromatograms`; `--psms-scored` versus `--scored`; `--psms` meaning two
different tables; `--seed`, `--seed-psms` and `--seeds` for one artifact.
Separately, standalone `report` takes no `--config`, so a config setting
`quant.q_threshold = 0.05` yields 0.05 from `run` and 0.01 from `report` on the
same scored table, silently. `quant`'s `run_lfq_combine` also writes two
**undeclared** siblings, `{out}.precursor.parquet` and `{out}.peptide.parquet`,
invisible in `--help`.

**5.4 Two frozen misnomers. MAJOR.** `extract.gate_min_score` is not a correlation
under the default gate, and `compete.group_by = precursor` keys on the base
peptide. Both are documented as misnomers, which does not make them cheap to rename
after a tag. Related default oddity: `gate_min_score` defaults to 0.2, documented as
the optimum for `nn_torch`, while the default rescorer is `native_tda`, whose
documented optimum is 0.6. The shipped default gate is tuned for a non-default
rescorer.

**5.5 No output-path guard anywhere. MAJOR.** There is no path-equality check in
the workspace, so
`mumdia prescan --lib-precursors lib/x.parquet --out lib/x.parquet` replaces a
full precursor library with a two-column survivors table and exits 0, because the
library was already resident. `align.rs:57/131` has the same shape, and in FASTA
mode `run --out-dir lib` truncates `lib/fragment_library_precursors.parquet` before
the digest has produced a row. `CLAUDE.md`'s claim that DeepLC fine-tuning "writes
a new precursor table rather than modifying the input" is verified for the
orchestrators, but the protection is naming, not checking.

## 6. Robustness and operations

**6.1 A truncated mzML produces a plausible wrong answer. MAJOR, verified.**

| input | exit | convert reported | pipeline reported |
|---|---|---|---|
| intact fixture | 0 | `ms1=60 ms2=480` | 151 precursor rows |
| **truncated to 40%** | **0** | `ms1=25 ms2=199` | **105 precursor rows** |
| 4 KB random bytes | 0 | `ms1=0 ms2=0` | 0 rows |

The mechanism: `convert.rs:118` iterates `Spectrum`, not `Result<Spectrum>`, so a
parse error mid-file ends iteration silently, and nothing compares the count
against the mzML's own `spectrumList count`. The truncated run writes every
artifact as if complete; its only hints are two `WARN` lines that read like
ordinary tuning notes. An interrupted transfer of a multi-gigabyte file is among
the most common real failures in this field, and the user would attribute the
missing third of their identifications to their sample. `convert.rs`,
`search_seed.rs`, `extract.rs` and `report.rs` contain **zero `warn!` calls**
between them, so the void run is entirely silent: 0-row tables, a header-only
`peptides.tsv`, exit 0.

**6.2 No atomic writes anywhere, and the manifest is written once at the end.
MAJOR, verified.** `grep` for `fs::rename|tempfile|sync_all|\.tmp` over `crates/`
returns **zero** hits. Every artifact goes straight to its final path via
`File::create`, which truncates on open, at `table.rs:204, 234, 267` and
`json.rs:12`. There is no signal handler of any kind in the workspace. So:

- a rerun destroys the previous good artifact *before* producing a replacement;
- Ctrl-C or a crash at stage 6 of 9 leaves **no manifest at all**, discarding the
  input hashes, `config_hash`, `git_sha` and five completed stages' provenance;
- nothing pre-clears the output directory, so a **stale manifest survives a failed
  rerun** and is internally consistent, every path it lists existing, while the
  files are from a different run. The `content_hash` that would expose this is
  computed at `lib.rs:49` and **never read anywhere**.

Mitigating: the Parquet footer is written last and every reader requires it, so a
half-written artifact hard-fails to open rather than reading as a valid short
table. It is rubble, not silent corruption. Also sound: `Manifest::record` only
inserts on `Ok`, so it can never claim an incomplete stage.

**6.3 `run_experiment` silently consumes the previous experiment's MBR table.
MAJOR.** `run_experiment.rs:480` selects the downstream input with
`Path::new(&scored_mbr).exists()`. The premise is real, since `mbr_worker.py:188`
returns without writing when there are no transfer candidates, but `exists()`
cannot distinguish "this run wrote it" from "a previous run left it". Rerun the
same `--out-dir` with different mzMLs or a tighter `mbr.q_transfer`, find no
transfers this time, and the split and every per-run `quant` consume the **old**
scored table. Both key on `candidate_id` and `source`, stable across reruns of the
same library, so the join succeeds and yields plausible quantities from the wrong
data. The log says "MBR transfers applied". This is the only silent stale-artifact
consumption found; elsewhere the downstream path comes from config, not `exists()`.

**6.4 Standalone scratch is a CWD-relative literal with fixed filenames, and it is
already leaking gigabytes. MAJOR, verified on disk.** `main.rs:884` hardcodes
`work_dir: "sidecar_work"` with no CLI override, and three of four handoffs use
fixed names (`entrapment_{in,out}`, `ms2pip_{in,out}`, `deeplc_{in,out}`). Only the
PIN/NN path is PID-qualified, and its comment records the exact clobbering bug that
motivated the change. Because the readback key is a **row index into each process's
own request**, two concurrent runs over tables of the same row count, which is the
most likely reason to run two at once, **silently swap score and intensity
vectors** rather than erroring. `align_sidecar_scores` catches a coverage mismatch,
not an equal-length swap. This is exercised, not theoretical: the repository root
holds **9.5 GB** of stale `sidecar_work/`, including a 4.75 GB `rescore.pin` from
Jul 27 under the pre-PID naming scheme, sitting next to PID-qualified files from
Aug 24. Nothing prunes it, on success or error. `run` and `run-experiment` are
safe, both using `{out_dir}/sidecar_work`.

**6.5 `--run-names` is neither length- nor uniqueness-checked. MAJOR.**
`run_experiment.rs:293-296` silently substitutes `r0..rN-1` on any count mismatch
with no warning, and accepts duplicates outright. Duplicate names make two runs
share an output directory, so at `parallel_runs > 1` two `process_run` calls
concurrently write the same `spectra_ms2.parquet`, `seed_psms.parquet`,
`psms_extracted.parquet` and `chromatograms.parquet`, and the output is an
interleaving of two runs with no error. Reachable by naming runs after basenames
when an experiment spans two directories holding same-named files.

**6.6 One malformed MS2 scan commits `16 B x n_candidates` per rayon worker.
MAJOR.** `convert.rs:139-144` maps **both** a zero-width reported isolation window
and a missing precursor to the full-range window `(0.0, 1.0e6)`, which spans the
whole library. `search_seed.rs:390-399` then sizes the per-worker scratch by
`.max()` over group widths. The sizing comment gives the number: "877 MB per worker
on the profiled 54.8M-candidate library", so a default all-core pool on 32 cores is
roughly 28 GB of commit charge for scratch that is almost entirely untouched. The
same comment notes that an underestimate is safe because the arrays grow on demand,
which makes max-based sizing strictly worse than a small guess. One malformed scan
in an otherwise 50-window run is enough. The only mitigation today is `--threads`,
which the user must know to reach for.

**6.7 The pooled feature matrix is unbounded, and the documented workaround is
unreachable. MAJOR.** `rescore.rs:79` is `Vec<Vec<f64>>`, not f32: `CLAUDE.md`
documents `n_psms x n_features x 4` bytes, but that is the *Python* worker's
matrix, and the code's own comment says `feats` is "~27 GB on an experiment-wide
pool", twice the documented width plus a heap allocation and 24-byte spine entry
per PSM. `run_experiment.rs:446` passes **every** run's competed table to one
`rescore::run`; there is no batch-size config field and no CLI flag, so
`CLAUDE.md`'s and `docs/17`'s advice to sub-batch to fit RAM requires abandoning
the orchestrator. `native_tda` additionally runs all folds under `into_par_iter`,
each materialising an owned standardised copy of its training slice, so the default
`folds = 3` costs about 3x the matrix on top of `feats`. Nothing anywhere estimates
or checks available memory or disk: `grep` for
`available_space|statvfs|sysinfo|total_memory|free_space` over `crates/` returns
**zero** hits.

**6.8 All log output goes to stdout, with ANSI escapes even when piped. MAJOR,
verified.** `mumdia doctor 2>/dev/null` still prints its log lines, and `> file`
captures `^[[2m2026-08-28T07:17:23Z^[[0m ^[[32m INFO^[[0m ...`. A run's diagnostics
and its result summary cannot be separated by redirection, log files contain escape
codes, and a parser cannot use a `key=value` regex because each field is
individually wrapped. Roughly a one-line fix
(`with_writer(std::io::stderr)`, `with_ansi(stdout_is_a_tty)`), and it is the
difference between MuMDIA being scriptable and not.

**6.9 `emit_pin` defaults to `true`, writing a large file nothing reads. MINOR.**
`features.rs:1065-1066` states outright that no MuMDIA stage consumes it, since
rescore builds its own PIN, and the default's own comment says "~5.4 GB text
write". Combined with the default TSV rescore handoff, a 40-run experiment writes
hundreds of GB nothing reads back. Related: `create_dir_all(...).ok()` at six sites
swallows directory-creation failures, so an unwritable or full `--out-dir` surfaces
only as a `File::create` error after the artifact is fully materialised in RAM.

**6.10 `convert.rs:176, 177, 210, 211` use `Col::ListF32`, which panics inside
arrow above 2^31-1 values. MINOR.** `extract.rs:2537-2538` already migrated the
chromatogram columns to `LargeListF32` for exactly this reason. Verified in the
pinned dependency: `arrow-array-59.0.0/src/builder/generic_list_builder.rs:210` is
`OffsetSize::from_usize(...).unwrap()`. A 50-window Orbitrap run has about 50x
headroom; a long Astral or timsTOF run at ~500k MS2 spectra x ~2000 peaks is within
2x. One-word fix with precedent in the same repository.

**6.11 Determinism: one live ordering defect and one latent. MINOR.**
`align.rs:80` iterates a `HashMap<u32, f64>` straight into the vectors that feed
`Loess::fit`. Observed RTs tie exactly for peptides in the same MS2 scan, and tied
x values keep HashMap order, so the k-nearest window can admit a different pair at
its boundary. This is the one clear violation of the `CLAUDE.md` ordering
convention, and `rt_im_train.rs:55-64` already solves it correctly with an
order-invariance test. Blast radius today is nil, since `alignment.parquet` is
written and never read, but it becomes load-bearing the moment align is wired into
MBR. Latent: `quant.rs:699` zips `integrated` against `cand_rows.values()` without
comparing the candidate id it carries, and the justifying comment is wrong
(`BTreeMap::par_iter()` is not an `IndexedParallelIterator`; order survives
incidentally). A `debug_assert_eq!` closes it for free.

**6.12 Error messages on the paths a new user hits first. MINOR.** `main.rs:634`
reads the config with a bare `?`, so a typo'd `--config` gives
`Error: The system cannot find the file specified. (os error 2)` with **no
filename**, in a tree with about 20 root `config.*.json` in flight.
`table.rs:385` does not name the file, because `Table` does not retain its path,
and dumps 390 column names into the message. `rescore.rs:806, 1043` are far thinner
than `sidecar.rs:251-269`, which gives the interpreter, the full command, a
reproduce instruction and a `doctor` pointer, and the rescore path is the one that
fails after hours.

**6.13 Panic census.** 62 hard-panic sites in production code; about 28 reachable
from real input, and about 26 of those are one pattern, a non-finite float reaching
`partial_cmp(...).unwrap()`. Worst by file: `features.rs` 8 of 9 reachable,
`extract.rs` 5 of 6 (three inside a rayon closure), `search_seed.rs` 4 of 4.
`mass.rs`, `digest.rs`, `rt_im_train.rs` and `rescoring.rs` are provably safe.
Note that the toolchain is pinned to 1.96.1, and since 1.81 `slice::sort_by` may
panic with "user-provided comparison function does not correctly implement a total
order"; `unwrap_or(Equal)` on NaN produces exactly such a comparator, so the
NaN-safe sites are hardened against the `unwrap` but not necessarily against the
sort. Detection is best-effort, so this is a hazard rather than a certainty. Fixing
section 1 removes most of this census.

## 7. Security and supply chain

**7.1 Two high-severity advisories on the untrusted-input path. BLOCKER,
verified against a database fetched 2026-08-28 (1226 advisories, 178 crates).**

| advisory | crate | CVSS | nature |
|---|---|---|---|
| RUSTSEC-2026-0194 | quick-xml 0.30.0 | 7.5 | quadratic run time on a start tag with duplicate attribute names |
| RUSTSEC-2026-0195 | quick-xml 0.30.0 | 7.5 | unbounded namespace-declaration allocation, memory exhaustion |
| RUSTSEC-2026-0204 | crossbeam-epoch 0.9.18 | — | invalid pointer dereference in a `fmt::Pointer` impl (MuMDIA never formats one) |
| RUSTSEC-2024-0436 | paste 1.0.15 | warning | unmaintained; final release, nothing to bump to |

Both `quick-xml` advisories are triggered by the XML being parsed, which for MuMDIA
is the mzML, and they are algorithmic-complexity attacks from small crafted files
rather than the oversized inputs `SECURITY.md` discounts. Remediation, verified by
dry-run, semver-compatible, no manifest edit:

```text
cd rust/mumdia && cargo update -p mzdata -p crossbeam-epoch
#   mzdata 0.65.2 -> 0.65.5, which requires quick-xml ^0.41 (both advisories fixed)
#   crossbeam-epoch 0.9.18 -> 0.9.20; also drops base16ct and md5
```

`mzdata 0.66.5` is also available and would need a `Cargo.toml` bump; the lockfile
route above is the smaller change. Either way, run the suite: MuMDIA-facing API
breakage was not checked.

**7.2 There is no dependency-advisory scan anywhere in the project. MAJOR.** No
`cargo audit` step, no `deny.toml`, no `audit.toml`, no mention in
`CONTRIBUTING.md`. `SECURITY.md` declares dependency vulnerabilities in scope and
invites pin-bump pull requests, so the policy promises attention to a class the
project cannot detect. 7.1 exists today only because this audit had `cargo-audit`
and network access. Add `cargo audit --deny warnings` as a required job plus a
scheduled run, so advisories published after a merge still surface. `cargo-deny` is
absent too, so there is no licence-policy or duplicate-version check; `getrandom`
and `cpufeatures` each resolve twice.

**7.3 Sidecar scripts resolve against the current working directory first, so an
attacker-planted `./scripts/` executes. MAJOR.** `python.rs:376-379` accepts the
configured directory as-is when it contains `mbr_worker.py` or `deeplc_worker.py`,
and the default is the relative `"scripts"`, which both shipped sidecar example
configs carry literally. Unpack a dataset archive, `cd` into it, run
`mumdia run --config configs/examples/diann-library.json`, and if that archive
contains `scripts/deeplc_worker.py` the whole directory wins resolution and every
worker in it runs as the user. This needs no hostile *configuration*, only an
untrusted input directory, so the `SECURITY.md` config-as-code carve-out does not
cover it. The Docker configs are correct by contrast, using absolute
`/opt/mumdia/scripts`. Fix: resolve config-relative and exe-relative first, or
refuse a relative `sidecar_script_dir` that does not resolve under the config
file's directory.

**7.4 No third-party licence notices accompany the redistributed binary. MAJOR.**
The tracked tree contains `LICENSE` and nothing else: no `NOTICE`, no
`THIRD_PARTY_LICENSES`. The release archive ships a statically linked binary
containing 175 external crates, including `arrow` and `parquet` 59.0.0, which are
Apache-2.0 with upstream `NOTICE` files, plus a large MIT-licensed set. Apache-2.0
section 4(d) requires propagating the `NOTICE` contents of works redistributed, and
MIT requires the copyright notice in all copies. The same applies to the container
image. Generate a bundle with `cargo about` or `cargo-bundle-licenses` and add it
to the archive and to `/opt/mumdia/`.

**7.5 `softprops/action-gh-release@v2` is an unpinned third-party action holding
`contents: write`. MAJOR.** All ten actions in the tree are pinned by mutable major
tag; the other nine are first-party `actions/*` and `docker/*`. This one is a
single-maintainer third-party action that receives the release token and the built
archives, so if the `v2` tag moves it runs with repository write access on every
tag push. Pin to a commit SHA. Dependabot already covers `github-actions`, so the
SHAs will keep moving.

**7.6 Paired array lengths from input files are never validated. MAJOR,
verified.** `convert.rs:59-60` decodes the m/z and intensity arrays independently
and tolerates failure with `unwrap_or_default()`, then `:20` takes the loop bound
from one array and `:29-31` indexes the other, so a Profile-continuity spectrum
with at least 3 m/z values and a shorter or undecodable intensity array panics on
the first iteration. `mzdata 0.65.2` never checks `defaultArrayLength` on read, so
nothing upstream enforces the pairing. The same missing invariant recurs at
`spectra.rs:128-148`, `extract.rs:495-508`, five sites in `quant.rs`, and
`features.rs:454-463`. The sibling function at `spectra.rs:68` shows the intended
guard: `let n = mf.len().min(iff.len());`. Every access is checked Rust, so the
outcome is a panic and never an out-of-bounds read; the fix is one `.min()` per
site.

**7.7 `Dockerfile:40-41` leaves a full C toolchain in the published runtime image.
MAJOR.** `build-essential` is installed into the final stage so pip can compile
sdists, and never removed, so `ghcr.io/compomics/mumdia` ships gcc, make and the C
headers in an image whose only job is to run one static binary and two prebuilt
conda envs. Build the envs in a separate stage and `COPY --from=`, or purge in the
same layer.

**7.8 Developer paths and a private compute-server name ship in the release
archive. MINOR.** `docs/13:588`, `docs/14:152`, `docs/17:55` and
`docs/22:59-60, 68-69` carry `C:/Users/robbi/...`, `H:/OneDrive - UGent/...` and
`~/astral`, `~/hye` paths on a host named "doxy". `ci/gen_cli_reference.py:22, 70`
and `ci/smoke.sh:30` hardcode `C:/Users/robbi/mumdia_build/release/mumdia.exe` as a
binary-search candidate; those two are public in the repository but not in the
archive. `rust/mumdia/.cargo/config.toml.example:11` correctly uses
`C:/Users/you/...`, which shows the anonymisation pass was intended and simply
incomplete.

**7.9 `docs/22_release_plan.md` is the wrong document to ship. MINOR.** Beyond 7.8
it states which gates currently fail, that no release has ever been tagged,
unpublished benchmark numbers, and what is blocked. None of that is wrong to know,
but shipping it in the archive of the release it plans is an odd artifact. Exclude
it from the archive or replace it with a trimmed roadmap.

**7.10 Smaller items.** `ci.yml` declares no `permissions:` block, so all four jobs
inherit the repository default when none needs a token beyond checkout.
`docker.yml` builds twice, so the smoke-tested image is not provably the pushed
image, and neither `FROM` is digest-pinned. `release.yml:71` interpolates
`${{ github.ref_name }}` into a `run:` block, which a crafted tag could exploit
(pushing a tag needs write access, so this is defence in depth). `.dockerignore` is
a deny list and does not cover the local-only `scripts/run_*.sh` that `.gitignore`
excludes, so a locally built image would bake them in; CI builds from a clean
checkout, so this is latent. `ms2pip==4.0.0.dev9` is a development pre-release
pinned into the published image. `mokapot` is pinned in the Docker env and floating
in the local env, so the two can run different rescorers. `main` has no branch
protection, which is what makes the shared build-output cache a real residual risk.
On Windows, bare `python3`/`python` candidates resolve against the **calling
executable's own directory** before PATH (verified against the pinned toolchain's
std source), and the release archive extracts `mumdia.exe` and `scripts/` into one
directory, so resolve the discovered candidate to an absolute path before spawning.
`manifest.json` records the full command line, hence the user's absolute input
paths and, on Windows, their username, in a tree that proteomics practice
encourages uploading to ProteomeXchange.

**7.11 Positives, verified.** Zero `unsafe` anywhere in the workspace, including
`build.rs`, not even in a comment; zero `get_unchecked`. This should be enforced,
not merely true: add `[workspace.lints.rust] unsafe_code = "forbid"`. No secret,
credential, email address, IP address or remote-access invocation anywhere in the
tracked tree, and none in the full history either (613 files ever added, checked
for secret-shaped names), so nothing needs history rewriting. No vendored or copied
third-party code, no foreign copyright header, and all three crates set
`publish = false`. 176 dependencies with no TLS stack, no async runtime and no
network client, and `mzdata`'s optional `reqwest`/`tokio` never resolve because
`default-features = false`. No shell anywhere: every subprocess crosses as an argv
vector, so shell metacharacters are inert. No dangerous deserialization in any
sidecar (`torch.load`, `pickle`, `yaml.load`, `eval`, `exec`, `shell=True` all
absent) and no network call in any tracked Python. No file-derived value ever
becomes a path component. `--locked` on every dependency-resolving cargo
invocation. Fork PRs cannot reach secrets or GHCR: `pull_request_target` appears
nowhere, and the GHCR login is additionally gated on a tag ref. No allocation is
sized from a file-declared count. The container runs unprivileged and CI fails the
build if the runtime uid is 0. The clean-room claim is supported by what is
actually tracked: no proprietary constant, map or coefficient vector is present
anywhere, and the mass constants' CODATA/AME provenance is documented.

## 8. Testing and CI

**8.1 The mass model is structurally untested. MAJOR, verified.** The only external
anchor is one test pinning PEPTIDE's **neutral** monoisotopic mass at `1e-3` Da.
Neutral, so `PROTON` never enters it; and `1e-3` Da cannot distinguish the proton
mass (1.007276466812) from the hydrogen-atom value (1.007825035), a 0.55 mDa
difference that is a classic error the code comments explicitly call out. Beyond
that: no test references `residue_mass` or pins any individual residue mass; no
test asserts a fragment m/z at any charge; `WATER`, `AMMONIA` and
`ISOTOPE_SPACING` are unpinned; and the two other mass tests are self-consistency
checks that pass unchanged if every residue mass is wrong by the same amount.

This class is **invisible to the end-to-end test by construction**, because
`ci/make_fixture_mzml.py` plants peaks read out of the engine's own library, so the
fixture agrees with the mass model however wrong the model is. Not hypothetical
either: `ISOTOPE_SPACING` once shipped 485 ppm wrong and was caught by a human
reading code. **This is the single most valuable test not yet written**, and it is
small: pin the 20 residue masses, `PROTON`, `WATER` and `ISOTOPE_SPACING` against
independently published values to sub-mDa, pin b and y fragment m/z for a reference
peptidoform at charge 1 and 2 to sub-ppm, and tighten the existing anchor.

**8.2 Coverage is inverted relative to risk. MAJOR.** `solve.rs` has a 324-case
bit-exact sweep against a reference implementation, while `convert.rs` (290 lines,
the front door for every analysis), `align.rs`, `run.rs`, `run_experiment.rs` and
`sidecar.rs` have **zero** tests. `extract.rs` has 6 tests for 2,710 lines, none
calling `run`. Eleven of sixteen feature families are untested, including the file
where a degenerate-ratio defect reaching 1e15 actually lived. `align.rs:9-10`
claims the module is "unit-tested on crafted two-run input"; it contains no tests.

Two specific unguarded silent-failure paths: `features.rs` writes an all-zero
column for any active feature with no producer
(`fmap.get(key).cloned().unwrap_or_else(|| vec![0.0; n])`), so a dead feature
family costs sensitivity and nothing fails; and the only cross-family length check
is a `debug_assert_eq!`, compiled out of the release build and never firing under
`cargo test` either, because the default feature set is Minimal while the smoke run
uses Extended.

**8.3 What the end-to-end test cannot catch.** It is a real improvement and the
only coverage of `convert`, the library build, `run`, the manifest, RT calibration
on real anchors and the report writers. It is also narrower than "112 assertions"
suggests: about 19 are science-facing, the rest per-artifact bookkeeping generated
in loops. Structurally unable to detect: any systematic mass error (8.1); FDR
behaviour at scale, since 3,820 candidates put the `+1` pseudocount in charge; any
wrong feature value, since nothing reads `features.parquet`; interference and
chimeric spectra, since the noise is synthetic; and the entire imported-library
production workflow, since the fixture runs `--fasta` only. Recovery is asserted at
a 60% floor, so a change losing 39% of true positives is green. Also unasserted:
`quant` records `apex_rt_column_present` and warns when false, because without it
every quantity is integrated around a re-detected apex that the warning itself says
reproduces the identification apex only about half the time. The smoke test never
reads `peptide_quant.parquet.report.json`, so a schema change dropping `apex_rt`
would make every quantity wrong while all 112 assertions stayed green. Two lines.

**8.4 The determinism claim is real but narrower than it reads.** Verified stronger
than documented in one respect: `peptides.tsv` is byte-identical at `--threads` 1,
4, 16 and default, across Windows and Linux, on four machines. Narrower in three:
the compared artifacts are the two **presentation-rounded** TSVs, so a
nondeterministic quantity differing in the fifth significant figure is invisible,
and `features.parquet`, `chromatograms.parquet` and `psms_scored.parquet` are never
compared; four separate "bit-identical to serial" claims in `extract.rs` and
`search_seed.rs` rest on argument, not test; and the fix is already on disk unused,
since both smoke runs write a `manifest.json` with a per-artifact `content_hash`,
so comparing the two manifests upgrades a two-column check into whole-tree byte
identity for about ten lines of Python.

**8.5 CI executes less of the Python suite than the count suggests.** 54 of 71
items run; 17 skip. All of `test_nn_rescore_worker.py`, `test_mokapot_worker.py`
and `test_entrapment_worker.py` skip, so **the entire external-classifier contract
that `rescore.strict` exists to protect executes nowhere in CI** — and that is
precisely where the worst defect in this project's history lived, when mokapot
returned a targets-only table and the engine reported about 331,000 "identified"
peptides against a correct figure near 51,000. The real DeepLC import, which
catches the `WinError 1114` failure that has already shipped once, runs only in the
Docker job, which fires on tags and manual dispatch. It should run on pull requests
touching `scripts/`, `env/` or `Dockerfile`.

**8.6 Flakiness. MINOR.** Four Rust tests use fixed, non-unique temp directories,
against the convention the docs state and four other test modules follow. Two
concurrent `cargo test` runs on one machine will race.

## 9. Documentation accuracy

The docs were heavily rewritten in the last two days, and an audit checked their
claims against the code. `docs/23_cli_reference.md` came back with **zero**
disagreements across all 21 subcommands, every flag and all 14 defaults, and the
README's TSV column tables, q-unit table and archive contents are exact. The
defects:

| claim | reality | severity |
|---|---|---|
| README and `docs/19`: "the DeepLC fine-tune is unseeded" | false since yesterday: `--seed` exists, seeds numpy and torch, wired from `rng_seed` | MAJOR |
| `CONTRIBUTING.md`: "no checked-in fixture and no end-to-end smoke test" | false: both exist and run in CI | MAJOR |
| `CONTRIBUTING.md` gate list | omits three CI-enforced gates (pytest, both reference `--check`s) | MAJOR |
| `schema.rs`: artifact versions exist "so a stage can validate its inputs" | no code reads a `schema_version`; all 18 uses are writes. `CONTRIBUTING.md` states a compatibility policy nothing enforces | MAJOR |
| README: `run.pin` is "as fed to the rescorer" | it is the pre-competition features export; rescore builds its own PIN | MAJOR |
| `CHANGELOG.md` known limitations: "interpreters by absolute path, not portable" | removed by the `"auto"` entry 128 lines above it | MAJOR |
| `docs/24` Gated column | 2 false positives and 4 false negatives; one marks a shipped default as "diagnostic", which the doc tells readers means do not enable | MAJOR |
| README: the generated references "cannot drift" | only generator-output equality is checked; hardcoded prose and the derived Gated column are outside it | MAJOR |
| `docs/19` `file:line` citations | four are wrong, in a doc set whose selling point is code-grounded citation. `ci/check_doc_refs.py` validates filenames, not lines | MAJOR |
| `docs/12`: the quantification integration section | describes only `trapezoid`/`trapezoid_window` and `summarize_fragment_areas`; no mention of `fixed_window_s`, `fixed_scan_halfwidth`, `baseline_subtract`, `fragment_selection` or `select_fragment_areas`, i.e. the entire recently added path including the one the Astral submission used | MAJOR |
| `docs/09` and `CLAUDE.md`: the "62.3% `NO_PEAK_GROUP`" decomposition | `audit.rs:50-61` reads an audit table `extract.rs` never writes (`emit_candidate_audit` is unwired), so the map is always empty and `_ => NoPeakGroup` catches presence failures, matched-fraction failures **and** every gate rejection. The documented reading, "never assembled `presence_min_fragments` distinct fragments", is not supported by the code | MAJOR |
| `CLAUDE.md`: rescore matrix is `n_psms x n_features x 4` bytes | that is the Python worker; the Rust `feats` is f64 (see 6.7) | MAJOR |
| `bench/README.md` | states no q-value column for any recorded result, breaking its own stated rule; "Reproducing" step 1 is a comment pointing at unvendored code; its scripts read filenames the engine never writes | MAJOR |
| four different baseline epsilons for the same quant improvement across four documents | none names its arm | MAJOR |
| `docs/11` and `CLAUDE.md`: pooling is "scale-invariant", sub-batching "statistically free" | direction right, magnitude overstated. Measured on the real kernel: replicating a 5-row population once moved q from `[0.5, 0.5, 0.667, 0.667, 1.0]` to `[0.25, 0.25, 0.5, 0.5, 0.833]`. The floor is exactly `1/T` and scales with pool size, so the claim holds for `run_psm_q` (exactly per-source, verified) but not for the pooled `q_value` that `run-experiment` gates quant on | MINOR |
| `docs/11`: NnTorch uses "the same CV-fold scheme" as `percolator_lite` | `nn_rescore_worker.py:355` keys the fold on the peptidoform string, which retains the `DECOY_` marker, so a target and its paired decoy land in **different** folds, where `percolator_lite` keys on `base_peptide_id` and pairs them. Direction is conservative, so this is a sensitivity and reproducibility defect plus a docs mismatch, not an FDR break. `nn_torch` is the validated production rescorer | MINOR |

**One more code-level finding surfaced by the doc audit:** `extract.frag_tol_ppm`
is **dead in every orchestrated run**. `extract.rs:1398-1400` reads it from the
masscal JSON with `.unwrap_or(cfg.frag_tol_ppm)`, but `search_seed.rs:290` *always*
writes that key, including in the calibration-failure branch, and both
orchestrators always pass `mass_cal: Some(...)`. So a config with
`search_seed.fragment_tol_ppm = 20` and `extract.frag_tol_ppm = 40` silently
extracts at 20 ppm. Separately, `search_seed.rs:177` collects calibrants inside a
literal `ppm_bounds(fmz, 50.0)` independent of the configured tolerance, so on a
wide-tolerance TOF config the p95 that sets the learned tolerance is truncated. The
first half is certain; the magnitude of the second is unmeasured.

## 10. Recommended order of work

**Before tagging `v0.1.0`.** All small, and roughly a day together.

1. **One finiteness pass at library load** (section 1), plus the `!(a >= b)`
   inversion at `index.rs:221`, the `!is_finite` guard on the RT window in extract,
   the NaN case in `within_ppm`, and `unwrap_or(Equal)` at `features.rs:1517`. This
   single item closes the largest blocker and most of the reachable panic census.
2. **Resolve interpreters and the script directory in `main.rs`** for every stage
   that launches a sidecar, and make `run-experiment` treat `"auto"` on an unused
   role the way `run` does. Add a test that a standalone stage honours `"auto"`.
3. **`cargo update -p mzdata -p crossbeam-epoch`**, run the suite, and add
   `cargo audit` to CI as a required job.
4. **Fix the entrapment path** (section 3): pass `is_decoy` rather than
   `is_entrapment` to the native rescorer, exclude decoys from the group
   competition under `QMode::Entrapment`, and validate `entrapment_ratio > 0`. Then
   re-run any entrapment measurement taken through the native rescorer.
5. **The `source` check in quant** (4.1) and **`_lib_io` in `mbr_worker.py`**
   (4.4). Both one-liners guarding silent wrong numbers.
6. **Add the mass and constants reference test** (8.1).
7. **Withdraw the input-validation claim in `schema.rs`** and align
   `CONTRIBUTING.md`; fix the documentation defects in section 9. They are cheap
   and they are the project's contract.
8. **Refuse a run that finds zero MS2 spectra**; send logs to stderr and disable
   ANSI when not a terminal (6.8).
9. **Assert `apex_rt_column_present`** in the smoke test and **compare the two
   runs' `manifest.json`** files (8.3, 8.4).
10. Remove the personal paths from `ci/`, and delete the 9.5 GB stale
    `sidecar_work/` in the repository root.

**Shortly after, and worth it for the CLI alone:** signal handling and
temp-then-rename with an incrementally written manifest (6.2); replace the
`exists()` check in `run_experiment` (6.3); PID-qualify or `--work-dir` the
standalone scratch and prune it (6.4); validate `--run-names` (6.5); size the seed
scratch per group (6.6); a rescore batch-size knob reachable from `run-experiment`
(6.7); `--log-format json` and `doctor --json`; a `NOTICE`/`THIRD_PARTY_LICENSES`
bundle (7.4); SHA-pin the release action (7.5); the paired-length `.min()` guards
(7.6); drop `build-essential` from the runtime image (7.7); resolve
`sidecar_script_dir` config-relative first (7.3); and the naming and misnomer
decisions of 5.3 and 5.4, which get more expensive after a tag.

**Deliberately not release blockers.** FDR calibration at scale, entrapment
validity as a *scientific* result, interference handling and quantification bias
all need real data and entrapment arms, not tests. The honest position is a tagged,
reproducible benchmark record, which `bench/` now provides the scoring half of.
Note the ordering, though: item 4 above must land before any of those measurements
means anything.

## 11. Open questions the audit could not settle

Recorded so they are not rediscovered.

- **How often the apex falls to the earliest scan** (section 2). Mechanism certain,
  frequency unmeasured. Count `best_sig <= 0.0` on the AIF run.
- **The magnitude of the NnTorch fold-pairing defect** (section 9). Direction
  reasoned conservative, not measured. A/B the 1% count with `base_peptide_id`
  passed as an explicit fold column.
- **Whether the seed-scratch figure is resident or committed-but-untouched** under
  mimalloc on Windows (6.6). Commit charge still fails an allocation.
- **The magnitude of the learned-tolerance truncation** (section 9). Compare
  `masscal.json`'s `frag_tol_ppm` against the empirical p95 of `mass_err_ppm` on a
  `fragment_tol_ppm = 100` TOF run.
- **Whether a NaN RT can reach `spectra.rs:95/158` from a real mzML.** Requires
  `mzdata`'s `start_time()` to return NaN for a spectrum lacking a scan start time;
  mzdata was not read. The panic is certain if it happens.
- **Five `expect("preflight guarantees ...")` sites** in `run.rs` and
  `run_experiment.rs`. One was traced and holds; the other four were not. Cheapest
  remaining audit item.
- **Whether `mzdata` 0.66 is API-compatible** with MuMDIA's call sites, if the
  `Cargo.toml` route is preferred over the lockfile route.
- **Whether `max`-over-unequal-group-size biases best-per-group competition.**
  Decoy charges are rejected at extraction more often, so group sizes can differ
  between a target and its decoy, which is a textbook anti-conservative mechanism.
  Set aside because selection happens on `prelim_score` while FDR is computed on
  the rescored score, making the induced bias second order, and because it is
  inherent to picked TDC generally. Recorded so it is known to have been
  considered.
- **Repository-level settings** are not in the tree: the default `GITHUB_TOKEN`
  permission, branch protection, GHCR package visibility, and whether Actions can
  approve PRs. Check via `gh api` and the Settings pages.
- **Python advisories.** No package in `env/*.yml` or `requirements.txt` was
  checked against PyPI advisories. Run `pip-audit` inside each built env, and
  `trivy image` / `docker sbom` against the published image.

## 12. Summary judgement

The core is sound and unusually disciplined for its age. Zero `unsafe`. Ordered
reductions that hold up under byte-identity checks across platforms and thread
counts. Versioned artifacts with content hashes. A genuine end-to-end test. A
sidecar contract enforced with per-row error messages that most projects never
write. No secrets, ever, in the whole history. Concurrency with no shared mutable
state at all.

What is not ready is the perimeter. The engine trusts user-supplied Parquet
completely while validating every internal boundary meticulously, and the guards
that should catch bad input are written in a form that a NaN passes. A headline
feature of this very release is broken on the workflow the README documents. The
mzML parser carries two high-severity advisories and nothing in CI would ever have
told us. The instrument the tuning policy designates for validating every gated
change has three defects on its own path. Several documentation claims are wrong in
the direction of overstatement, and the test suite's strongest guarantees sit on
the least risky code.

None of the blockers is large, and the first item on the fix list closes the
biggest one. The gap between "CI is green" and "ready to tag" is about a day of
work, and the review found it precisely because those are not the same question.
