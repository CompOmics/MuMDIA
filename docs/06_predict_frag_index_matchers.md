# predict-frag, the library, the inverted fragment index, matchers

> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

This subsystem produces the run-independent spectral library and the data
structures that make fragment matching cheap. It has three parts:

1. **predict-frag** (Stage C, `stages/predict_frag.rs`): turn concrete
   peptidoforms into a library of precursor and b/y fragment m/z with predicted
   fragment intensities and predicted iRT, keep the top-N fragments per
   candidate, and assign each candidate a `candidate_id` equal to its rank in
   precursor-m/z order. The output is two Parquet artifacts that are constant for
   the whole run (they do not depend on the mzML being searched).

2. **the library loader** (`index.rs`, `Library::load`): read those two Parquet
   artifacts back into a Structure-of-Arrays in-memory model, group fragments by
   candidate, build the bucketed peak-major inverted fragment index, and enforce
   the preconditions the matchers rely on.

3. **the matchers** (`matchers/`): given an observed peak m/z, a ppm tolerance,
   and an isolation-window candidate range, return the predicted fragments that
   fall within tolerance. Two backends exist: the default log-bin CSR `fragindex`
   (`matchers/fragindex.rs`, spec in `fragindex_spec.md`) and the fallback
   bucketed `Library::page_search`. A naive band-join (`matchers/naive.rs`) is the
   correctness oracle, not a production path.

The predictor traits (`predict.rs`) and the sidecar file contract (`sidecar.rs`)
sit under predict-frag: they are the boundary between the native (zero external
dependency) intensity/RT models and the optional Python predictors MS2PIP and
DeepLC.

## Files

| path | role |
|---|---|
| `rust/mumdia/crates/mumdia/src/stages/predict_frag.rs` | Stage C: build the library (parse, fragments, intensity, iRT, top-N, sort, write Parquet) |
| `rust/mumdia/crates/mumdia/src/predict.rs` | `RtPredictor`/`FragmentPredictor` traits + `NativeRt`/`NativeFrag` fallbacks |
| `rust/mumdia/crates/mumdia/src/sidecar.rs` | Python sidecar clients (MS2PIP, DeepLC, DeepLC fine-tune, MBR) over the file contract |
| `rust/mumdia/crates/mumdia/src/index.rs` | `Library` (SoA model + bucketed inverted index), `load()` preconditions, `page_search`, `candidate_range`, `deconvolve` |
| `rust/mumdia/crates/mumdia/src/matchers/mod.rs` | matcher module tree; `MatcherKind` selects the backend |
| `rust/mumdia/crates/mumdia/src/matchers/binning.rs` | `LogBins`: log-space bin geometry (fragindex_spec Section 2.2) |
| `rust/mumdia/crates/mumdia/src/matchers/fragindex.rs` | `FragIndex` CSR index, `SeedScratch` epoch-stamped accumulator, `probe_peak`, equivalence-gate scorer |
| `rust/mumdia/crates/mumdia/src/matchers/naive.rs` | band-join reference for the equivalence gate |
| `rust/mumdia/crates/mumdia-core/src/constants.rs` | `within_ppm` (min-relative predicate), `ppm_bounds` (query-relative), `PROTON` |
| `fragindex_spec.md` | language-agnostic algorithm spec the fragindex matcher implements |

## Inputs and outputs

### Consumed

**peptidoforms** (Stage A2 output), read at `predict_frag.rs:53-58`:

| column | type | note |
|---|---|---|
| `id` | u32 | peptidoform id |
| `base_peptide_id` | u32 | stripped-peptide id |
| `peptidoform` | str | ProForma-lite string with UniMod names |
| `charge` | i32 | precursor charge |
| `label` | str | `"target"` or `"decoy"` |
| `protein` | str | protein accession |

### Produced (two Parquet artifacts + a `report.json` each)

**fragment_library_precursors** (schema `("fragment_library_precursors", 1)`,
`schema.rs:13`), written at `predict_frag.rs:186-200`:

| column | type |
|---|---|
| `candidate_id` | u32 (dense 0..N in precursor-m/z order) |
| `peptidoform_id` | u32 |
| `base_peptide_id` | u32 |
| `peptidoform` | str |
| `charge` | i32 |
| `precursor_mz` | f64 |
| `predicted_irt` | f32 |
| `label` | str |
| `protein` | str |
| `n_fragments` | i32 (kept fragment count after top-N) |

**fragment_library_fragments** (schema `("fragment_library_fragments", 1)`,
`schema.rs:14`), written at `predict_frag.rs:201-212`:

| column | type |
|---|---|
| `candidate_id` | u32 (foreign key into precursors) |
| `mz` | f64 (fragment m/z at its own `frag_charge`) |
| `predicted_intensity` | f32 |
| `name` | str (e.g. `b2`, `y3`) |
| `ion_type` | str (`b` or `y`) |
| `ordinal` | i32 (residue-count ordinal of the fragment) |
| `frag_charge` | i32 (1, or 2 when enabled) |

Each artifact also gets an `ArtifactReport` (`predict_frag.rs:219-238`) recording
row count, blake3 content hash, params (`top_n`, `ms2pip_model`, `rt_predictor`,
`fragment_predictor`), and `model_identity` (the concatenated RT + fragment model
ids, `predict_frag.rs:114`).

The library-input path (`--lib-precursors`/`--lib-fragments`) skips Stage C and
feeds externally built Parquet with these exact schemas directly into
`Library::load`.

## How it works

### predict-frag (`stages/predict_frag.rs:50`)

**Phase A: parse and enumerate fragments** (`predict_frag.rs:65-110`). Each
peptidoform row is parsed with `parse_peptidoform` and fragmented independently,
so rows are mapped in parallel with rayon (`into_par_iter`, line 70). The closure
returns `RowOut::Raw`, `RowOut::ParseErr`, or `RowOut::Empty`. Fragment charges
are chosen per row: charge 1 always, charge 2 added when the precursor charge is
at least `charge2_from_precursor_charge` (`predict_frag.rs:78-81`). `collect`
preserves row order, and the sequential fold at `predict_frag.rs:103-109`
reproduces the exact serial `raws` order and `n_parse_err` count. The parsed form
is stored on the `Raw` struct so intensity/iRT assignment reuses it instead of
re-parsing (`predict_frag.rs:45-47`).

**iRT assignment** (`assign_rt`, `predict_frag.rs:245`). Native path calls
`NativeRt::predict_irt` per candidate. DeepLC path deduplicates by peptidoform
string (RT is charge-independent, `predict_frag.rs:262-271`), runs the sidecar
once over the unique set, then maps results back. Peptidoforms DeepLC returns no
prediction for are anchored at `irt = 0.0` and counted; if any are missing a
`tracing::warn!` fires (`predict_frag.rs:284-289`). This is the DeepLC-miss iRT
warning: it makes the silent "unmatched peptidoform gets iRT 0.0" failure visible,
because an iRT-0 anchor collapses the RT window onto the gradient origin and
misplaces the candidate at extraction.

**intensity assignment** (`assign_intensities`, `predict_frag.rs:296`). Native
path calls `NativeFrag::predict_intensities`. MS2PIP path runs the sidecar over
all candidates; an empty whole-map result is a hard error (`bail`,
`predict_frag.rs:316-318`). Per candidate, MS2PIP supplies charge-1 b/y
intensities keyed by `(ion_byte, ordinal)`; charge-2 fragments (which MS2PIP does
not emit) fall back to the native model (`predict_frag.rs:329-337`). MS2PIP
values (TIC-fraction scale, roughly 0.02-0.3) and the native charge-2 fallback
(max-normalized, roughly 0.19-0.5) live on different scales, so each charge group
is max-normalized to its own peak before they compete for top-N slots
(`predict_frag.rs:344-359`); otherwise the larger-scale group would always win the
truncation. A candidate MS2PIP returns nothing for falls back wholesale to native.

**top-N and candidate_id** (`predict_frag.rs:119-137`). For each candidate,
fragments are ranked by predicted intensity descending, truncated to
`top_n_fragments`, then re-sorted ascending to restore stored order; an in-place
forward-swap gather compacts the kept fragments without reallocating. Candidates
left with zero fragments are dropped (`retain`, line 134). Then `raws` is sorted
by `precursor_mz` with a stable `sort_by` (`predict_frag.rs:137`), and
`candidate_id` is assigned by `enumerate` over that order
(`predict_frag.rs:163-164`). This is the single most load-bearing invariant of
the whole subsystem: `candidate_id` is the dense precursor-m/z rank, which is what
lets the index recover the isolation-window candidate slice by binary search and
lets `candidate_id` directly index the dense accumulator.

### Library load and the bucketed inverted index (`index.rs:55`)

`Library::load` reads both Parquet artifacts and rebuilds the per-candidate
fragment arrays grouped and contiguous (`index.rs:100-128`), so `cand_frags(cid)`
is a slice (`index.rs:208-213`). It then builds the bucketed inverted index
(`index.rs:158-187`):

1. Emit one `(frag_mz as f32, candidate_id, frag_int)` entry per fragment.
2. Globally sort entries by fragment m/z with `par_sort_by` (parallel stable
   sort, identical result to the serial stable sort, `index.rs:169`).
3. Chunk the sorted entries into fixed buckets of `bucket_size`; record each
   bucket's first (== minimum) m/z in `bucket_min`; within each bucket sort by
   `candidate_id` (`index.rs:172-178`).
4. Split into the three parallel arrays `idx_mz`/`idx_cid`/`idx_int`.

`page_search` (`index.rs:242`) probes this index for an observed neutral m/z `q`:
`ppm_bounds(q, tol_ppm)` gives the query window `[lo, hi]` (cast to f32); a
`partition_point` over `bucket_min` selects the buckets overlapping the window
(`index.rs:258-260`); within each bucket, because `idx_cid` is ascending, two
`partition_point` calls narrow to the `[cand_lo, cand_hi)` slice
(`index.rs:266-268`); a linear tail applies the exact f32 m/z bound
(`index.rs:269-275`). `candidate_range` (`index.rs:233`) turns an isolation window
into `[lo, hi)` over `prec_mz` by two `partition_point`s.

### The fragindex CSR matcher (`matchers/fragindex.rs`, fragindex_spec Section 2-3)

`FragIndex::build` (`fragindex.rs:46`) is a two-pass counting sort into a CSR
layout keyed by log-space bin. `LogBins::new` (`binning.rs:27`) precomputes the
geometry: `delta = tol_ppm * 1e-6`, bin width `w = ln(1 + delta)`, `inv_w`,
`ln_min`, and `n_bins = floor(span * inv_w) + 2` (the `+2` pads the top so
`bin+1` never overflows). `LogBins::bin` (`binning.rs:43`) maps m/z to
`floor((ln(mz) - ln_min) * inv_w)`, clamped to `[0, n_bins-1]`. Pass 1 counts
per-bin occupancy with the `+1` counting-sort offset (`fragindex.rs:83-86`); a
prefix sum turns counts into CSR start offsets (`fragindex.rs:88-90`); pass 2
scatters postings in candidate-id order (`fragindex.rs:98-110`) so `post_cand` is
ascending within every bin. Postings are Structure-of-Arrays
(`post_cand`/`post_mz`/`post_int`/`post_frag`) so the verify hot loop streams only
`post_mz`.

`probe_peak` (`fragindex.rs:152`) probes bins `bin(peak)-1 ..= bin(peak)+1`
(clamped, `fragindex.rs:162-165`), narrows each bin to `[cand_lo, cand_hi)` by
binary search over the ascending `post_cand` (`fragindex.rs:172-174`), and
verifies each posting with the exact `within_ppm` predicate in f64
(`fragindex.rs:176-179`). `SeedScratch` (`fragindex.rs:191`) is the epoch-stamped
dense accumulator: `stamp[cc]` records the last epoch a candidate was touched;
`epoch` is incremented before each scan (`fragindex.rs:221`) and starts at 0 so 0
is never a live epoch; on first touch the score is zeroed and the candidate pushed
to `touched` (`fragindex.rs:227-232`); after a scan only touched candidates are
read. The seed accumulates a fused `(count, obs_sum)` semiring where `obs_sum`
sums the observed peak intensity per matched posting (predicted intensity
deliberately discarded, `fragindex.rs:234`).

**The +/-1 probe exactness.** The correctness of probing only three bins rests on:
two m/z values within tolerance differ by at most one bin. Proof sketch: within
tolerance means `|ln(a) - ln(b)| <= w`, one bin width, so the two points span at
most two adjacent bins (`binning.rs` test `within_tol_pairs_are_at_most_one_bin_apart`,
`fragindex_spec.md` Section 2.2). One subtlety makes this exact in the
implementation: posting m/z is stored f32, so build bins each posting by the
**same f32-rounded value** the verify uses (`fragindex.rs:85` and
`fragindex.rs:102`), not the raw f64. Binning by raw f64 while verifying the f32
value could place a posting two bins from the peak and silently drop a
boundary-straddling within-tolerance pair. The test
`probe_finds_within_tol_across_bin_boundaries` (`fragindex.rs:426`) sweeps the m/z
range to guard this.

**The bucketed fallback vs fragindex predicate.** The two backends do not use the
same tolerance predicate. `page_search` uses `ppm_bounds` (a window symmetric
around the query `q`, `constants.rs:78`). `fragindex` uses `within_ppm` (a
min-relative predicate, `hi - lo <= tol_ppm*1e-6*lo`, `constants.rs:92`). These
differ at the tolerance edge, so a handful of edge pairs are accepted by one and
not the other. On AIF full-range-window data the predicate difference shifts
identifications enough to matter, which is why the bucketed path is retained for
A/B (`config.rs:37-39`). The fragindex `probe_peak` and its `naive` band-join both
verify under `within_ppm` on f32-rounded m/z (`naive.rs:30`), so the equivalence
gate is not contaminated by storage precision.

### Wiring into stages

Both stages build the fragindex backend only when `MatcherKind::Fragindex` is
selected and otherwise fall through to the bucketed `Library` path.
search-seed: `FragIndex::build` at `search_seed.rs:52-53`; fragindex path
`seed_fragindex_windows` parallelizes across isolation-window groups with a
per-thread `SeedScratch` and a deterministic total-order merge
(`search_seed.rs:297-382`); bucketed path uses `page_search`
(`search_seed.rs:73`). extract selects via the `probe_matched` helper
(`extract.rs:42-59`): fragindex carries the true generating fragment ordinal in
`post_frag`, whereas the bucketed path recovers the ordinal by nearest stored m/z
(`Library::local_frag_index`, `index.rs:217`).

## Key types and functions

| name | file:line | what it does |
|---|---|---|
| `predict_frag::run` | `predict_frag.rs:50` | Stage C entry: parse, fragment, assign intensity/iRT, top-N, sort, write |
| `assign_rt` | `predict_frag.rs:245` | native or DeepLC iRT; emits the DeepLC-miss warning |
| `assign_intensities` | `predict_frag.rs:296` | native or MS2PIP intensity with per-charge-group normalization + native charge-2 fallback |
| `RtPredictor` / `FragmentPredictor` | `predict.rs:13` / `predict.rs:19` | predictor traits (predict + `identity`) |
| `NativeRt` | `predict.rs:25` | additive retention-coefficient model, `identity` `native-rt-v1` |
| `NativeFrag` | `predict.rs:73` | heuristic b/y intensity model, max-normalized, `identity` `native-frag-v1` |
| `resolve_script` | `sidecar.rs:18` | locate a worker script (CWD, exe dir, exe dir/scripts) |
| `run_ms2pip` | `sidecar.rs:37` | MS2PIP client; returns `cid -> (ion_byte, ordinal) -> intensity` |
| `run_deeplc` | `sidecar.rs:77` | DeepLC client; returns `id -> predicted_rt` |
| `Library` | `index.rs:38` | SoA candidate + fragment model plus the bucketed inverted index |
| `Library::load` | `index.rs:55` | read Parquet, group fragments, build index, enforce preconditions |
| `Library::page_search` | `index.rs:242` | bucketed probe: bucket select -> candidate slice -> f32 ppm verify |
| `Library::candidate_range` | `index.rs:233` | isolation window -> `[lo, hi)` over `prec_mz` |
| `Library::local_frag_index` | `index.rs:217` | nearest-stored-m/z fragment ordinal (bucketed path only) |
| `deconvolve` | `index.rs:283` | z-charged peak m/z -> neutral m/z, in f64 |
| `LogBins` / `LogBins::bin` | `binning.rs:11` / `binning.rs:43` | log-space bin geometry and mapping |
| `FragIndex::build` | `fragindex.rs:46` | two-pass counting-sort CSR build at a fixed tolerance |
| `FragIndex::probe_peak` | `fragindex.rs:152` | +/-1 bin probe + `within_ppm` verify, candidate-window narrowed |
| `SeedScratch` | `fragindex.rs:191` | epoch-stamped dense `(count, obs_sum)` accumulator |
| `score_scan_count_dot` (fragindex / naive) | `fragindex.rs:260` / `naive.rs:16` | equivalence-gate scorers under an identical predicate |
| `within_ppm` / `ppm_bounds` | `constants.rs:92` / `constants.rs:78` | min-relative vs query-relative tolerance predicates |

## Configuration

`PredictFragConfig` (`config.rs:292-322`, `#[serde(default, deny_unknown_fields)]`,
so the struct was pruned to exactly these fields and unknown keys are rejected):

| field | default | effect |
|---|---|---|
| `predictor` | `Native` (`FragPredictorKind`) | native heuristic vs `Ms2pip` sidecar for fragment intensities |
| `rt_predictor` | `Native` (`RtPredictorKind`) | native additive model vs `Deeplc` sidecar for iRT |
| `charge2_from_precursor_charge` | `2` | precursor charge at/above which charge-2 fragments are added (was 3; lowered to keep the ~16% of charge-2 precursors' doubly-charged transitions) |
| `top_n_fragments` | `6` | fragments kept per candidate after intensity ranking (top-6 is standard DIA) |
| `ms2pip_model` | `"HCD"` | MS2PIP model name passed as argv |
| `ms2pip_python` | `None` | interpreter for the MS2PIP sidecar; required when `predictor=ms2pip`, else the stage errors |
| `deeplc_python` | `None` | interpreter for the DeepLC sidecar; required when `rt_predictor=deeplc`, else the stage errors |
| `sidecar_script_dir` | `"scripts"` | directory searched by `resolve_script` for the worker scripts |

Matcher selection (both stages default to `Fragindex`):

| field | default | effect |
|---|---|---|
| `MatcherKind` | `Fragindex` (`config.rs:46-49`) | `Fragindex` (log-bin CSR) or `Bucketed` (`Library::page_search`) |
| `search_seed.matcher` | `Fragindex` (`config.rs:340`) | backend for the seed search |
| `search_seed.fragment_tol_ppm` | `20.0` (`config.rs:352`) | tolerance the seed index is built at |
| `extract.matcher` | `Fragindex` (`config.rs:499,574`) | backend for extraction |
| `extract.bucket_size` | `8192` (`config.rs:485,570`) | fixed bucket size of the bucketed `Library` index (fragindex ignores it) |

## Invariants, determinism, gotchas

- **candidate_id contiguity.** `candidate_id` must be the dense range `0..N` in
  precursor-m/z-ascending row order. predict-frag guarantees it
  (`predict_frag.rs:137,163`); `Library::load` re-checks it and bails with a clear
  message (`index.rs:78-87`); `FragIndex::build` asserts it (`fragindex.rs:50-56`).
  An external library fed in unsorted or unindexed (for example
  `import_diann_lib.py` output not passed through `make_reverse_decoys.py`) fails
  here rather than silently misgrouping fragments.
- **precursors ascending by m/z.** `Library::load` verifies `prec_mz` is
  non-decreasing and bails otherwise (`index.rs:135-146`); the `candidate_range`
  binary search assumes it.
- **fragment foreign key.** A fragment row referencing `candidate_id >= N` is a
  hard error (`index.rs:92-96`).
- **decoy presence is a warning, not an error.** A library with zero decoys makes
  target-decoy q-values invalid, but bailing would break intentional decoy-free
  diagnostics, so `load` warns loudly instead (`index.rs:150-156`).
- **total_frags must fit u32.** `FragIndex::build` asserts this
  (`fragindex.rs:58`) because posting indices are u32.
- **f32 posting m/z, f64 math.** Index/posting m/z is stored f32; the ppm verify
  widens to f64 (`fragindex.rs:176-177`, `index.rs` module docs). f32 ULP is
  ~0.12 ppm, 200-400x below the 20-50 ppm regime, so storage precision is not a
  gate disagreement source; build and verify use the same f32-rounded value.
- **per-posting accumulation.** A candidate with two fragments within tolerance of
  one peak counts twice; a peak within tolerance of two candidate fragments counts
  twice (fragindex_spec Section 1.4). Do not deduplicate. Tests
  `two_frags_one_peak_counts_both` (`fragindex.rs:368`) and the count assertion in
  `equivalence_gate_vs_naive` (`fragindex.rs:363`) guard this.
- **epoch, not value, for first touch.** First touch is `stamp[cc] != epoch`, never
  `acc[cc] == 0`; a legitimate zero score would otherwise be misclassified
  (`fragindex.rs:227`, spec Section 3.5).
- **determinism.** predict-frag maps rows in parallel but `collect` + a sequential
  fold reproduce the serial order; the precursor-m/z sort is stable and top-N is a
  deterministic ranking; `Library::load` uses a parallel stable sort equal to the
  serial one; `FragIndex::build` is fully serial (candidate-order scatter, no
  hashing); `seed_fragindex_windows` merges partial results in a total order
  independent of thread/group order (`search_seed.rs:364-380`). `SeedScratch`
  sums `obs_sum` in the caller's fixed peak order and `touched()` is in
  first-touch order, so callers sort before any float reduction
  (`fragindex.rs:239-243`).
- **DeepLC nondeterminism and the iRT-0 anchor.** The DeepLC sidecar and fine-tune
  are not seeded, so iRT values vary run to run. Any peptidoform DeepLC does not
  return lands at iRT 0.0; the warning at `predict_frag.rs:284-289` reports the
  count so a large miss is visible rather than silent.
- **predicate mismatch between backends.** `page_search` (query-relative
  `ppm_bounds`) and `fragindex` (`within_ppm`) accept slightly different edge
  pairs; switching `MatcherKind` can move a small number of identifications. This
  is expected, not a bug (`config.rs:37-39`).
- **`naive.rs` is not production.** It is O(C x frags x peaks), used only as the
  equivalence oracle.

## How to extend / modify

- **Add a fragment or RT predictor.** Implement `FragmentPredictor` /
  `RtPredictor` (`predict.rs:13,19`) for a native model, or add a variant to
  `FragPredictorKind` / `RtPredictorKind` (`config.rs:114,99`) plus a branch in
  `assign_intensities` / `assign_rt`. Match the existing `identity()` convention;
  the id flows into `model_identity` and the artifact report. Keep the native
  fallback path intact so the engine still runs with no Python.
- **Add a sidecar.** Follow the positional-CLI file contract in `sidecar.rs`:
  write an input Parquet, invoke `run_worker(python, script, &[argv...], utf8)`,
  read an output Parquet keyed by id. Use `resolve_script` so a deployed binary
  finds the worker regardless of CWD. Set `utf8 = true` for Keras/torch workers
  (they crash on the Windows cp1252 console). Never hardcode the interpreter path;
  it comes from config (`ms2pip_python`, `deeplc_python`, or the rescore/finetune
  equivalents).
- **Change the matcher.** New backends go under `matchers/` behind a `MatcherKind`
  variant. Any new backend must pass the equivalence gate against `naive.rs` at
  `K = C` under the same `within_ppm` predicate (test pattern in
  `fragindex.rs:331`) before its speed is trusted. Before attempting a fragindex
  optimization, read `fragindex_spec.md` Section 5: cache-blocking, accumulator
  prefetch, radix partitioning, bin-major inversion, and distinct-m/z dedup were
  all measured null or negative in the realistic DIA regime. The real levers are
  top-N reduction (already applied, `top_n_fragments`) and per-window parallelism
  (already applied in `seed_fragindex_windows`).
- **Change fragment charges or top-N.** `charge2_from_precursor_charge` and
  `top_n_fragments` are the two knobs; both are sensitivity/speed tradeoffs.
  Lowering top-N removes collisions roughly proportionally (spec Section 5.5) and
  shrinks the library; it does not change matcher correctness.
- **Feeding an external library.** It must satisfy the `load()` preconditions
  (dense `candidate_id` in precursor-m/z order, ascending `prec_mz`, decoys
  present). The decoy-builder scripts (`make_reverse_decoys.py`) sort and reindex
  to satisfy them; do not bypass that step.
