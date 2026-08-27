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

2. **the library loader** (`index.rs`, `Library::load` / `load_with`): read those
   two Parquet artifacts back into a Structure-of-Arrays in-memory model, group
   fragments by candidate, enforce the preconditions the matchers rely on, and
   optionally build the bucketed peak-major inverted fragment index (skipped for
   the default fragindex backend, which never reads it).

3. **the matchers** (`matchers/`): given an observed peak m/z, a ppm tolerance,
   and an isolation-window candidate range, return the predicted fragments that
   fall within tolerance. Two backends exist: the default log-bin CSR `fragindex`
   (`matchers/fragindex.rs`, specified in this document) and the fallback
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
| `rust/mumdia/crates/mumdia/src/index.rs` | `Library` (SoA model + optional bucketed inverted index), `load`/`load_with` preconditions, `page_search`, `candidate_range`, `local_frag_index`, `deconvolve` |
| `rust/mumdia/crates/mumdia/src/matchers/mod.rs` | matcher module tree; `MatcherKind` selects the backend |
| `rust/mumdia/crates/mumdia/src/matchers/binning.rs` | `LogBins`: log-space bin geometry (this document) |
| `rust/mumdia/crates/mumdia/src/matchers/fragindex.rs` | `FragIndex` CSR index, `WindowNarrow` per-window narrowing cache, `SeedScratch` epoch-stamped accumulator, `probe_peak`/`probe_peak_win`, equivalence-gate scorer |
| `rust/mumdia/crates/mumdia/src/matchers/naive.rs` | band-join reference for the equivalence gate |
| `rust/mumdia/crates/mumdia-core/src/constants.rs` | `within_ppm` (min-relative predicate), `ppm_bounds` (query-relative), `PROTON` |

## Inputs and outputs

### Consumed

**peptidoforms** (Stage A2 output), read at `predict_frag.rs:52-58`:

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
`schema.rs:13`), written at `predict_frag.rs:233-247`:

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
`schema.rs:14`), written at `predict_frag.rs:254-266`:

| column | type |
|---|---|
| `candidate_id` | u32 (foreign key into precursors) |
| `mz` | f64 (fragment m/z at its own `frag_charge`) |
| `predicted_intensity` | f32 |
| `name` | str (e.g. `b2`, `y3`) |
| `ion_type` | str (`b` or `y`) |
| `ordinal` | i32 (residue-count ordinal of the fragment) |
| `frag_charge` | i32 (1, or 2 when enabled) |
| `cardinality` | i32 (distinct precursors sharing this fragment's 0.01 Da m/z bin) |

`cardinality` is computed once over the whole library by `fragment_cardinality`
(`predict_frag.rs:457`), matching the binning the imported-library path uses. It
is diagnostic: `Library::load` does not read it, and no stage consumes it yet.

Each artifact also gets an `ArtifactReport` (`predict_frag.rs:277-295`) recording
row count, blake3 content hash, params (`top_n`, `ms2pip_model`, `rt_predictor`,
`fragment_predictor`), a shared `stats` map (`candidates`, `fragments`,
`parse_errors`, `predict_frag.rs:269-272`), and `model_identity` (the concatenated
RT + fragment model ids, `predict_frag.rs:133`). Both reports share the same
`stats`/`model_identity`; only `rows` differs per artifact (`predict_frag.rs:282-286`).
The schema ids are the constants `artifact::FRAGMENT_LIBRARY_PRECURSORS` /
`artifact::FRAGMENT_LIBRARY_FRAGMENTS` (`schema.rs:13-14`). `run` itself returns
`(n_prec, n_frag)` (`predict_frag.rs:50,304`).

The library-input path (`--lib-precursors`/`--lib-fragments`) skips Stage C and
feeds externally built Parquet with these exact schemas directly into the library
loader. Library load never regenerates `predicted_irt`: the imported
value is read at `index.rs:86`, stored on the `Candidate` at `index.rs:200`
without validation beyond the column type, and consumed as-is by RT calibration.
Whatever the external builder put in that column is what the RT window is centered
on (see the modform-iRT gotcha below). The only thing that replaces it is the
optional DeepLC fine-tune, which is a separate orchestrator pre-step writing a new
precursor table; with fine-tuning off the imported column is used verbatim. Note
the asymmetry: Stage C guards its own predictions, bailing on a non-finite
predicted iRT or fragment intensity (`predict_frag.rs:140-153`), while the
imported path has no equivalent check.

`predict_frag::run` is driven by `PredictFragParams` (`predict_frag.rs:24-31`):
`peptidoforms` (input path), `out_precursors` / `out_fragments` (the two output
paths), `cfg` (`&PredictFragConfig`), `work_dir` (scratch dir for sidecar
Parquet), and `config_hash` (carried for provenance).

## How it works

### predict-frag (`stages/predict_frag.rs:50`)

**Phase A: parse and enumerate fragments** (`predict_frag.rs:60-129`). Each
peptidoform row is parsed with `parse_peptidoform` and fragmented independently,
so rows are mapped in parallel with rayon (`into_par_iter`, line 72). The closure
returns `RowOut::Raw`, `RowOut::ParseErr`, or `RowOut::Empty`. Fragment charges
are chosen per row by one of two rules. The default rule is charge 1 always, plus
charge 2 when the precursor charge is at least `charge2_from_precursor_charge`
(`predict_frag.rs:91-97`). When `charge_by_basic_residues` is set it supersedes
that rule (`predict_frag.rs:79-90`): every charge up to
`min(precursor_charge, 1 + basic_residue_count)` is requested, and each fragment
is then kept only at charges its own basic sites (Arg/His/Lys inside that
fragment) plus its N-terminal amine can hold. `collect` preserves row order, and
the sequential fold at `predict_frag.rs:116-124` reproduces the exact serial
`raws` order and `n_parse_err` count. The parsed form is stored on the `Raw`
struct so intensity/iRT assignment reuses it instead of re-parsing
(`predict_frag.rs:45-47`).

**iRT assignment** (`assign_rt`, `predict_frag.rs:308`). Native path calls
`NativeRt::predict_irt` per candidate. DeepLC path deduplicates by peptidoform
string (RT is charge-independent, `predict_frag.rs:324-334`), runs the sidecar
once over the unique set, then maps results back. Peptidoforms DeepLC returns no
prediction for are anchored at `irt = 0.0` and counted; if any are missing a
`tracing::warn!` fires (`predict_frag.rs:347-352`). This is the DeepLC-miss iRT
warning: it makes the silent "unmatched peptidoform gets iRT 0.0" failure visible,
because an iRT-0 anchor collapses the RT window onto the gradient origin and
misplaces the candidate at extraction. The DeepLC branch requires `deeplc_python`
and errors otherwise (`predict_frag.rs:318-321`); its returned model id is the
hardcoded string `"deeplc-4.0-mt"` (`predict_frag.rs:353`), not a trait
`identity()` (the sidecar path has no `RtPredictor` impl to query).

**intensity assignment** (`assign_intensities`, `predict_frag.rs:359`). Native
path calls `NativeFrag::predict_intensities`. MS2PIP path runs the sidecar over
all candidates; an empty whole-map result is a hard error (`bail`,
`predict_frag.rs:389-391`). Per candidate, MS2PIP supplies charge-1 b/y
intensities keyed by `(ion_byte, ordinal)`; charge-2 fragments (which MS2PIP does
not emit) fall back to the native model (`predict_frag.rs:405-418`). MS2PIP
values (TIC-fraction scale, roughly 0.02-0.3) and the native charge-2 fallback
(max-normalized, roughly 0.19-0.5) live on different scales, so each charge group
is max-normalized to its own peak before they compete for top-N slots
(`predict_frag.rs:419-438`); otherwise the larger-scale group would always win the
truncation. There are two distinct MS2PIP-miss fallbacks: a candidate MS2PIP
returns nothing for (absent from the map, or an empty per-candidate map) falls
back wholesale to native (`predict_frag.rs:441-443`), whereas a single charge-1
fragment whose `(ion_byte, ordinal)` key MS2PIP omits gets intensity `0.0`, not
the native value (`unwrap_or(&0.0)`, `predict_frag.rs:413`); a fragment at
`0.0` can then be dropped by top-N. MS2PIP requires `ms2pip_python` and errors
otherwise (`predict_frag.rs:371-374`); its model id is `format!("ms2pip-{model}")`
(`predict_frag.rs:446`).

**finite guard** (`predict_frag.rs:140-153`). Between assignment and top-N, every
candidate's iRT and every fragment intensity is checked for finiteness and a
non-finite value is a hard error. A NaN from a misbehaving sidecar would otherwise
misorder the top-N sort and corrupt every downstream spectral-similarity feature.

**top-N and candidate_id** (`predict_frag.rs:155-173`). For each candidate,
fragments are ranked by predicted intensity descending, truncated to
`top_n_fragments`, then re-sorted ascending to restore stored order; an in-place
forward-swap gather compacts the kept fragments without reallocating. Candidates
are independent here, so the pass runs under `par_iter_mut` (line 158). Candidates
left with zero fragments are dropped (`retain`, line 173). Then `raws` is sorted
by `precursor_mz` with a parallel STABLE `par_sort_by` (`predict_frag.rs:178`,
same result as the serial stable `sort_by`), and `candidate_id` is assigned by
`enumerate` over that order (`predict_frag.rs:210-211`). This is the single most
load-bearing invariant of
the whole subsystem: `candidate_id` is the dense precursor-m/z rank, which is what
lets the index recover the isolation-window candidate slice by binary search and
lets `candidate_id` directly index the dense accumulator.

### The native predictor models (`predict.rs`)

Both native fallbacks are deterministic and Python-free, so the engine runs with
zero external dependencies. `NativeRt` (`predict.rs:55-70`) sums a self-derived
per-residue hydrophobicity coefficient (`rt_coeff`, `predict.rs:27-53`, clean-room,
not a borrowed vector), adds `0.01 * mod_mass` per modified residue, and adds a
`sqrt(length)` term so very long peptides do not elute infinitely late; `identity`
is `native-rt-v1`. `NativeFrag` (`predict.rs:75-105`) weights y ions at 1.0 and b
ions at 0.75, scales by a mid-sequence positional factor `1 - 0.5*|ordinal-L/2|/(L/2)`
(mid-sequence fragments are more intense), halves charge-2 fragments, then
max-normalizes the whole vector to its peak; `identity` is `native-frag-v1`. The
predictor traits `RtPredictor` / `FragmentPredictor` (`predict.rs:13-22`) each
expose `predict*` plus `identity`; only the native structs implement them (the
sidecar paths bypass the traits and emit their own id strings, above).

### The sidecar file contract (`sidecar.rs`)

Sidecars follow a positional-CLI Parquet contract (no JSON request file): write an
input Parquet, invoke `python script arg...`, read an output Parquet keyed by id.

- `run_ms2pip` (`sidecar.rs:42-78`): input columns `id` (u32), `peptidoform`
  (str), `charge` (i32); output columns `id`, `ion_type` (str), `ordinal` (i32),
  `intensity` (f32). Returns `candidate_id -> (ion_byte, ordinal) -> intensity`;
  `ion_byte` is the first byte of `ion_type` (`b'?'` if empty, `sidecar.rs:72`).
  Argv is `[input, output, model]` (`sidecar.rs:63`).
- `run_deeplc` (`sidecar.rs:81-105`): input columns `id`, `peptidoform`; output
  columns `id`, `predicted_rt` (f32). Returns `id -> predicted_rt`. Argv is
  `[input, output]`.
- `run_deeplc_finetune` (`sidecar.rs:111-155`): fine-tunes the DeepLC RT model on
  confident seed PSMs and writes a new output precursor table with updated
  `predicted_irt`; it does not modify the input library.
  Positional contract `deeplc_finetune.py <lib_in> <seed> <lib_out> --epochs E
  --patience P --q-train Q --batch B`. Invoked by `run` between search-seed and RT
  calibration, not by predict-frag. Its output is a library artifact, not a
  per-run one, so it can be produced once and reused. Measured on one run, a
  library fine-tuned once plus the per-run LOESS in rt-im-train gave a reported
  median |RT residual| of 6.06 s against 6.14 s for a per-file fine-tune of the
  same run, while the per-file fine-tune cost 2,166 s of a 5,127 s run (the
  fine-tune itself plus whole-library iRT prediction over roughly 5M
  peptidoforms). Both residual figures are the in-sample `cal.json` fit
  diagnostic, so read them as "equal or marginally better", not as an RT error
  estimate. See `docs/08_rt_im_train.md`.
- `run_mbr` (`sidecar.rs:162-213`): match-between-runs transfer (Stage D3, needs
  >= 2 runs). Positional contract `mbr_worker.py <scored_combined> <psms_csv>
  <out_transferred> [flags]`, where `psms_csv` is the per-run psms paths joined by
  `,` in `source` order; the launcher always passes `--q-anchor`,
  `--min-anchor-runs`, `--q-transfer`, and `--seed`, adds `--out-scored` when an
  output-scored path is supplied, and adds `--frag-csv` / `--consensus-corr-min`
  only when fragments are supplied and the threshold is > 0.
- `run_worker` (`sidecar.rs:217-233`) is the shared launcher; it bails if the
  process exits non-zero. `utf8 = true` sets `PYTHONUTF8=1` and
  `PYTHONIOENCODING=utf-8` (DeepLC/Keras/torch crash on the Windows cp1252 console);
  it is on for the DeepLC and fine-tune calls, off for MS2PIP and MBR.

### Library load and the bucketed inverted index (`index.rs:66`)

`Library::load(precursors, fragments, bucket_size)` (`index.rs:66`) is a thin
wrapper over `Library::load_with(..., build_bucketed)` (`index.rs:73`). Callers
that use the default fragindex backend pass `build_bucketed = false`, which skips
the bucketed arrays entirely: they are a full extra copy of every library fragment
plus a global sort, all of it dead when `page_search` is never called.
`page_search` early-returns on the resulting empty index, so skipping is safe.

`load_with` reads the precursor artifact in full and the fragment artifact
column-projected to `candidate_id`/`mz`/`predicted_intensity`/`name`
(`index.rs:102-105`; `ion_type`, `ordinal`, `frag_charge` and `cardinality` are
never read). It validates the label column (only `"target"`/`"decoy"` allowed, via
`fdr::validate_labels`, `index.rs:89`), then regroups fragments per candidate with
a counting sort into flat contiguous arrays (`index.rs:126-207`), so
`cand_frags(cid)` returns three parallel slices for one candidate
(`index.rs:312-321`): m/z, predicted intensity, and INTERNED fragment-name ids
(`u16` into `frag_name_dict`, resolved with `frag_name_str`, `index.rs:303`).
Names are interned rather than stored as `String` because they come from a tiny
repeating vocabulary and a per-fragment `String` costs about 24 bytes of struct
before any text. `n_candidates()` is the candidate count (`index.rs:297`).
Each row becomes a `Candidate` (`index.rs:23-35`) with `candidate_id`,
`peptidoform_id`, `base_peptide_id`, `peptidoform`, `charge`, `precursor_mz`,
`predicted_irt`, `protein`, `frag_start`/`n_frag` (the slice bounds into the flat
fragment arrays), and `is_decoy`, which is derived from the `label` string
(`label == "decoy"`, `index.rs:201`); the string label is not otherwise retained.
When `build_bucketed` is true, `load_with` then builds the bucketed inverted index
(`index.rs:245-280`):

1. Emit one `(frag_mz as f32, candidate_id, frag_int)` entry per fragment.
2. Globally sort entries by fragment m/z with `par_sort_by` (parallel stable
   sort, identical result to the serial stable sort, `index.rs:263`).
3. Chunk the sorted entries into fixed buckets of `bucket_size` (floored at 1 via
   `bucket_size.max(1)`, `index.rs:250`); record each bucket's first (== minimum)
   m/z in `bucket_min`; within each bucket sort by `candidate_id`
   (`index.rs:266-270`).
4. Split into the three parallel arrays `idx_mz`/`idx_cid`/`idx_int`.

`page_search` (`index.rs:350`) probes this index for an observed neutral m/z `q`.
It early-returns on a degenerate window or empty index (`cand_hi <= cand_lo ||
idx_mz.is_empty()`, `index.rs:358-360`). Otherwise `ppm_bounds(q, tol_ppm)` gives
the query window `[lo, hi]` (cast to f32); the first bucket is
`partition_point(|m| m <= lo32).saturating_sub(1)` and the last is
`partition_point(|m| m <= hi32)` over `bucket_min` (`index.rs:361-371`); within
each bucket, because `idx_cid` is ascending, two `partition_point` calls narrow to
the `[cand_lo, cand_hi)` slice (`index.rs:378-379`); a linear tail applies the
exact f32 m/z bound (`index.rs:380-386`). `candidate_range` (`index.rs:341`) turns
an isolation window into `[lo, hi)` over `prec_mz` by two `partition_point`s (`m <
win_lo` for `lo`, `m <= win_hi` for `hi`, so `win_hi` is inclusive and `win_lo`
exclusive).

### The fragindex CSR matcher (`matchers/fragindex.rs`)

`FragIndex::build` (`fragindex.rs:46`) is a two-pass counting sort into a CSR
layout keyed by log-space bin. It first derives the m/z range by scanning
`lib.frag_mz` for the min and max (`fragindex.rs:61-70`); if the library is empty
(no finite bound) it falls back to `[1.0, 2.0]` (`fragindex.rs:71-74`), and it
clamps the arguments to `LogBins::new` so `mz_min >= 1.0` and `mz_max > mz_min`
(`fragindex.rs:75`). `LogBins::new` (`binning.rs:27`) asserts `mz_min > 0 &&
mz_max >= mz_min` and precomputes the geometry: `delta = tol_ppm * 1e-6`, bin
width `w = ln(1 + delta)`, `inv_w`, `ln_min`, and `n_bins = floor(span * inv_w)
+ 2` (the `+2` pads the top so `bin+1` never overflows). `LogBins::bin`
(`binning.rs:49`) maps m/z to `floor((ln(mz) - ln_min) * inv_w)`, clamped to
`[0, n_bins-1]` (m/z <= 0 or <= `mz_min` maps to bin 0). Pass 1 counts
per-bin occupancy with the `+1` counting-sort offset (`fragindex.rs:83-86`); a
prefix sum turns counts into CSR start offsets (`fragindex.rs:88-90`); pass 2
scatters postings in candidate-id order (`fragindex.rs:98-110`) so `post_cand` is
ascending within every bin. Postings are Structure-of-Arrays
(`post_cand`/`post_mz`/`post_int`/`post_frag`) so the verify hot loop streams only
`post_mz`.

`FragIndex` keeps its own copy of `prec_mz` and exposes `n_cand()`
(`fragindex.rs:125`), `tol_ppm()` (`fragindex.rs:129`), and a `candidate_range`
(`fragindex.rs:137-141`) with the same `[lo, hi)` semantics as
`Library::candidate_range`, so a caller holding only the index can still narrow to
the isolation window. `probe_peak` (`fragindex.rs:152`) early-returns on a
degenerate window (`cand_hi <= cand_lo`, `fragindex.rs:159`), then probes bins
`bin(peak)-1 ..= bin(peak)+1` (clamped, `fragindex.rs:162-165`), narrows each bin
to `[cand_lo, cand_hi)` by binary search over the ascending `post_cand`
(`narrow_bin`, `fragindex.rs:205-215`), and verifies each posting with the exact
`within_ppm` predicate in f64 (`emit_range`, `fragindex.rs:220-233`). Its callback
receives `(cid, post_mz_f64, post_int, post_frag)`, where `post_frag` is the
candidate-local fragment ordinal that extract carries through directly (the
bucketed path has to recover it via `local_frag_index`).

`probe_peak_win` (`fragindex.rs:176`) is a drop-in for `probe_peak` that takes a
`WindowNarrow` (`fragindex.rs:263`), a lazily filled per-isolation-window cache of
each bin's `[cand_lo, cand_hi)` posting sub-range built by `window_narrow`
(`fragindex.rs:236`). `cand_lo`/`cand_hi` are fixed for a whole isolation window
and every scan of that window revisits the same bins, so the two binary searches
per bin happen once per `(window, bin)` instead of once per peak. Semantics and
callback order are identical, which the test
`probe_peak_win_matches_probe_peak_callback_for_callback` (`fragindex.rs:442`)
asserts posting-for-posting.

`SeedScratch` (`fragindex.rs:275`) is the epoch-stamped dense accumulator:
`stamp[cc]` records the last epoch a candidate was touched; `epoch` is incremented
before each scan (`fragindex.rs:324`) and starts at 0 so 0 is never a live epoch;
on first touch the score is zeroed and the candidate pushed to `touched`
(`fragindex.rs:334-339`); after a scan only touched candidates are read. Its
arrays are indexed WINDOW-relative against a `base` of `cand_lo`, so they are
sized by the widest candidate window rather than by the library (16 bytes per
candidate per rayon worker would otherwise be allocated and almost entirely never
touched); `ensure` grows them on demand, so an undersized `new(cap)` is safe. The
seed accumulates a fused `(count, obs_sum)` semiring where `obs_sum` sums the
observed peak intensity per matched posting (predicted intensity deliberately
discarded, `fragindex.rs:341`), and exposes the touched set plus per-candidate
`count(cid)` / `obs_sum(cid)` getters (`fragindex.rs:348-363`).

The free function `score_scan_count_dot` (`fragindex.rs:370`) is a separate,
non-`SeedScratch` scorer used only by the equivalence gate: it accumulates
`(count, dot)` per candidate over one scan, where `dot` is the sum over matched
postings of `predicted_intensity * peak_intensity` (both widened to f64, distinct
from `SeedScratch`'s observed-only `obs_sum`), collects into a `HashMap`, and
returns the result sorted by candidate id (`fragindex.rs:385-387`). The `naive`
band-join scorer (`naive.rs:16`) computes the identical `(count, dot)` under the
same f32-rounded `within_ppm` predicate (`naive.rs:30-33`), which is what lets the
gate assert exact Count equality and near-exact Dot equality.

**The +/-1 probe exactness.** The correctness of probing only three bins rests on:
two m/z values within tolerance differ by at most one bin. Proof sketch: within
tolerance means `|ln(a) - ln(b)| <= w`, one bin width, so the two points span at
most two adjacent bins (`binning.rs` test
`within_tol_pairs_are_at_most_one_bin_apart`). One subtlety makes this exact in the
implementation: posting m/z is stored f32, so build bins each posting by the
**same f32-rounded value** the verify uses (`fragindex.rs:85` and
`fragindex.rs:102`), not the raw f64. Binning by raw f64 while verifying the f32
value could place a posting two bins from the peak and silently drop a
boundary-straddling within-tolerance pair. The test
`probe_finds_within_tol_across_bin_boundaries` (`fragindex.rs:597`) sweeps the m/z
range to guard this.

**The bucketed fallback vs fragindex predicate.** The two backends do not use the
same tolerance predicate. `page_search` uses `ppm_bounds` (a window symmetric
around the query `q`, `constants.rs:78`). `fragindex` uses `within_ppm` (a
min-relative predicate, `hi - lo <= tol_ppm*1e-6*lo`, `constants.rs:92`). These
differ at the tolerance edge, so a handful of edge pairs are accepted by one and
not the other. On AIF full-range-window data the predicate difference shifts
identifications enough to matter, which is why the bucketed path is retained for
A/B (`config.rs:30-35`). The fragindex `probe_peak` and its `naive` band-join both
verify under `within_ppm` on f32-rounded m/z (`naive.rs:30`), so the equivalence
gate is not contaminated by storage precision.

### Wiring into stages

Both stages build the fragindex backend only when `MatcherKind::Fragindex` is
selected and otherwise fall through to the bucketed `Library` path.
search-seed: `FragIndex::build` at `search_seed.rs:63-64`; fragindex path
`seed_fragindex_windows` parallelizes across isolation-window groups with a
per-worker `SeedScratch` sized to the widest candidate window and a deterministic
total-order merge (`search_seed.rs:367-477`); bucketed path stays serial and uses
`page_search` (`search_seed.rs:84`). extract selects inside `Prober::probe`
(`extract.rs:44-81`): fragindex carries the true generating fragment ordinal in
`post_frag` and uses the cached `probe_peak_win` when the caller supplies a
`WindowNarrow`, whereas the bucketed path recovers the ordinal by nearest stored
m/z (`Library::local_frag_index`, `index.rs:325`).

## Key types and functions

| name | file:line | what it does |
|---|---|---|
| `PredictFragParams` | `predict_frag.rs:24` | Stage C entry args: in/out paths, `cfg`, `work_dir`, `config_hash` |
| `predict_frag::run` | `predict_frag.rs:50` | Stage C entry: parse, fragment, assign intensity/iRT, top-N, sort, write; returns `(n_prec, n_frag)` |
| `Raw` | `predict_frag.rs:34` | one candidate pre-assignment; caches the `ParsedPeptidoform` so RT/intensity reuse the parse |
| `assign_rt` | `predict_frag.rs:308` | native or DeepLC iRT; emits the DeepLC-miss warning; DeepLC id `"deeplc-4.0-mt"` |
| `assign_intensities` | `predict_frag.rs:359` | native or MS2PIP intensity with per-charge-group normalization + native charge-2 fallback; MS2PIP id `"ms2pip-{model}"` |
| `fragment_cardinality` | `predict_frag.rs:457` | distinct precursors per 0.01 Da fragment-m/z bin, per fragment row; diagnostic column, no consumer yet |
| `RtPredictor` / `FragmentPredictor` | `predict.rs:13` / `predict.rs:19` | predictor traits (predict + `identity`); implemented only by the native structs |
| `NativeRt` | `predict.rs:25` | additive retention-coefficient model + `sqrt(len)` + `0.01*mod` term, `identity` `native-rt-v1` |
| `NativeFrag` | `predict.rs:73` | heuristic b/y intensity model (y=1.0, b=0.75, mid-seq positional, charge-2 x0.5), max-normalized, `identity` `native-frag-v1` |
| `resolve_script` | `sidecar.rs:20` | locate a worker script (CWD, exe dir/dir, exe dir/scripts, else CWD-relative) |
| `run_ms2pip` | `sidecar.rs:42` | MS2PIP client; in `id`/`peptidoform`/`charge`, out `id`/`ion_type`/`ordinal`/`intensity`; returns `cid -> (ion_byte, ordinal) -> intensity` |
| `run_deeplc` | `sidecar.rs:81` | DeepLC client; in `id`/`peptidoform`, out `id`/`predicted_rt`; returns `id -> predicted_rt` |
| `run_deeplc_finetune` | `sidecar.rs:111` | DeepLC multitask fine-tune; `deeplc_finetune.py <lib_in> <seed> <lib_out>` + epoch/patience/q-train/batch flags (called by `run`, not predict-frag) |
| `run_mbr` | `sidecar.rs:162` | MBR transfer (Stage D3); `mbr_worker.py <scored> <psms_csv> <out>` + flags |
| `run_worker` | `sidecar.rs:217` | shared launcher; bails on non-zero exit; `utf8` sets `PYTHONUTF8`/`PYTHONIOENCODING` |
| `Candidate` | `index.rs:23` | one library row in SoA; `is_decoy` derived from the `label` string |
| `Library` | `index.rs:37` | SoA candidate + fragment model plus the optional bucketed inverted index |
| `Library::load` / `load_with` | `index.rs:66` / `index.rs:73` | read Parquet, validate labels, group fragments, enforce preconditions; `load_with` can skip the bucketed index and is what both stages call |
| `Library::n_candidates` / `cand_frags` / `frag_name_str` | `index.rs:297` / `index.rs:312` / `index.rs:303` | candidate count; per-candidate (m/z, intensity, interned name id) slices; name-id resolver |
| `Library::page_search` | `index.rs:350` | bucketed probe: bucket select -> candidate slice -> f32 ppm verify |
| `Library::candidate_range` | `index.rs:341` | isolation window -> `[lo, hi)` over `prec_mz` |
| `Library::local_frag_index` | `index.rs:325` | nearest-stored-m/z fragment ordinal (bucketed path only) |
| `deconvolve` | `index.rs:394` | z-charged peak m/z -> neutral m/z, in f64 |
| `LogBins` / `LogBins::bin` | `binning.rs:11` / `binning.rs:49` | log-space bin geometry and mapping |
| `FragIndex::build` | `fragindex.rs:46` | two-pass counting-sort CSR build at a fixed tolerance; derives m/z range from the library |
| `FragIndex::probe_peak` | `fragindex.rs:152` | +/-1 bin probe + `within_ppm` verify, candidate-window narrowed; callback `(cid, mz, int, frag)` |
| `FragIndex::probe_peak_win` / `window_narrow` / `WindowNarrow` | `fragindex.rs:176` / `:236` / `:263` | same probe with the per-window bin-narrowing cache amortized |
| `FragIndex::candidate_range` / `n_cand` / `tol_ppm` | `fragindex.rs:137` / `:125` / `:129` | index-side isolation-window narrowing + accessors |
| `SeedScratch` | `fragindex.rs:275` | epoch-stamped, window-relative dense `(count, obs_sum)` accumulator; `touched`/`count`/`obs_sum` getters |
| `score_scan_count_dot` (fragindex / naive) | `fragindex.rs:370` / `naive.rs:16` | equivalence-gate scorers, `dot = predicted*observed`, under an identical predicate |
| `within_ppm` / `ppm_bounds` | `constants.rs:92` / `constants.rs:78` | min-relative vs query-relative tolerance predicates |

## Configuration

`PredictFragConfig` (`config.rs:359-384`, `#[serde(default, deny_unknown_fields)]`,
so the struct holds exactly these fields and unknown keys are rejected):

| field | default | effect |
|---|---|---|
| `predictor` | `Native` (`FragPredictorKind`) | native heuristic vs `Ms2pip` sidecar for fragment intensities |
| `rt_predictor` | `Native` (`RtPredictorKind`) | native additive model vs `Deeplc` sidecar for iRT |
| `charge2_from_precursor_charge` | `2` | precursor charge at/above which charge-2 fragments are added (was 3; lowered to keep the ~16% of charge-2 precursors' doubly-charged transitions) |
| `charge_by_basic_residues` | `false` | composition cap: keep a fragment at charge z only if `z <= 1 + (#R+#H+#K in that fragment)` and `z <= precursor charge`. Supersedes `charge2_from_precursor_charge`. Benchmark-gated, it changes the scored transition set. Pairs with `peptidoforms.charge_by_basic_residues` (`config.rs:318`) |
| `top_n_fragments` | `6` | fragments kept per candidate after intensity ranking (top-6 is standard DIA) |
| `ms2pip_model` | `"HCD"` | MS2PIP model name passed as argv |
| `ms2pip_python` | `None` | interpreter for the MS2PIP sidecar; required when `predictor=ms2pip`, else the stage errors |
| `deeplc_python` | `None` | interpreter for the DeepLC sidecar; required when `rt_predictor=deeplc`, else the stage errors |
| `sidecar_script_dir` | `"scripts"` | directory searched by `resolve_script` for the worker scripts |

Matcher selection (both stages default to `Fragindex`):

| field | default | effect |
|---|---|---|
| `MatcherKind` | `Fragindex` (`config.rs:36-42`) | `Fragindex` (log-bin CSR) or `Bucketed` (`Library::page_search`) |
| `search_seed.matcher` | `Fragindex` (`config.rs:417,440`) | backend for the seed search |
| `search_seed.fragment_tol_ppm` | `20.0` (`config.rs:405,436`) | tolerance the seed index is built at |
| `extract.matcher` | `Fragindex` (`config.rs:617,715`) | backend for extraction |
| `extract.bucket_size` | `8192` (`config.rs:579,706`) | fixed bucket size of the bucketed `Library` index (fragindex ignores it, and the bucketed arrays are then not built at all) |

## Invariants, determinism, gotchas

- **candidate_id contiguity.** `candidate_id` must be the dense range `0..N` in
  precursor-m/z-ascending row order. predict-frag guarantees it
  (`predict_frag.rs:178,210`); `Library::load_with` re-checks it and bails with a
  clear message (`index.rs:112-125`); `FragIndex::build` asserts it
  (`fragindex.rs:50-56`).
  An external library fed in unsorted or unindexed (for example
  `import_diann_lib.py` output not passed through `make_reverse_decoys.py`) fails
  here rather than silently misgrouping fragments.
- **precursors ascending by m/z.** `Library::load_with` verifies `prec_mz` is
  non-decreasing and bails otherwise (`index.rs:215-231`); the `candidate_range`
  binary search assumes it.
- **fragment foreign key.** A fragment row referencing `candidate_id >= N` is a
  hard error (`index.rs:133-139`).
- **both label classes must be present.** A library with zero targets or zero
  decoys makes target-decoy q-values meaningless, so load bails at library
  load with a clear message rather than completing a long search on an invalid
  null (`index.rs:232-243`). The label column is also validated: any value other
  than `"target"`/`"decoy"` is rejected up front by `fdr::validate_labels`
  (`index.rs:89`, `fdr.rs:127`).
- **total_frags must fit u32.** `FragIndex::build` asserts this
  (`fragindex.rs:58`) because posting indices are u32.
- **f32 posting m/z, f64 math.** Index/posting m/z is stored f32; the ppm verify
  widens to f64 (`fragindex.rs:228-229`, `index.rs` module docs). f32 ULP is
  ~0.12 ppm, 200-400x below the 20-50 ppm regime, so storage precision is not a
  gate disagreement source; build and verify use the same f32-rounded value.
- **per-posting accumulation.** A candidate with two fragments within tolerance of
  one peak counts twice; a peak within tolerance of two candidate fragments counts
  twice (see the duplicate-fragment note below). Do not deduplicate. Tests
  `two_frags_one_peak_counts_both` (`fragindex.rs:539`) and the count assertion in
  `equivalence_gate_vs_naive` (`fragindex.rs:524-527`) guard this.
- **epoch, not value, for first touch.** First touch is `stamp[cc] != epoch`, never
  `acc[cc] == 0`; a legitimate zero score would otherwise be misclassified
  (`fragindex.rs:334`, spec Section 3.5).
- **determinism.** predict-frag maps rows in parallel but `collect` + a sequential
  fold reproduce the serial order; the precursor-m/z sort is a parallel STABLE sort
  equal to the serial one and top-N is a deterministic per-candidate ranking;
  `Library::load_with` uses a parallel stable sort equal to the
  serial one; `FragIndex::build` is fully serial (candidate-order scatter, no
  hashing); `seed_fragindex_windows` merges partial results in a total order
  independent of thread/group order (`search_seed.rs:459-475`). `SeedScratch`
  sums `obs_sum` in the caller's fixed peak order and `touched()` is in
  first-touch order, so callers sort before any float reduction
  (`fragindex.rs:346-350`).
- **DeepLC nondeterminism and the iRT-0 anchor.** The DeepLC sidecar and fine-tune
  are not seeded, so iRT values vary run to run. Any peptidoform DeepLC does not
  return lands at iRT 0.0; the warning at `predict_frag.rs:347-352` reports the
  count so a large miss is visible rather than silent.
- **An imported library can carry one iRT per stripped peptide.** The importer
  copies `predicted_irt` verbatim from the source library's RT column
  (`import_diann_lib.py:148`) and library load accepts it unchecked
  (`index.rs:86,200`), so a library that was expanded with modification variants
  can hand every modform of a peptide the unmodified form's retention time.
  Measured on one modification-expanded imported library, 79.7% of
  stripped-peptide groups had an identical raw `predicted_irt` across all of
  their modforms, and Spearman correlation against a proper per-modform
  prediction was 0.9876 for unmodified peptides but only 0.4980 for modified
  ones. Unlike the Stage C iRT-0 case above there is no warning and no load-time
  error; the only symptom is modified candidates extracted at the wrong RT. Check
  the variance of `predicted_irt` within each `base_peptide_id` before a PTM
  search, and re-predict the expanded peptidoform set through the DeepLC path if
  that variance is zero.
- **predicate mismatch between backends.** `page_search` (query-relative
  `ppm_bounds`) and `fragindex` (`within_ppm`) accept slightly different edge
  pairs; switching `MatcherKind` can move a small number of identifications. This
  is expected, not a bug (`config.rs:30-35`).
- **`naive.rs` is not production.** It is O(C x frags x peaks), used only as the
  equivalence oracle.

## How to extend / modify

- **Add a fragment or RT predictor.** Implement `FragmentPredictor` /
  `RtPredictor` (`predict.rs:13,19`) for a native model, or add a variant to
  `FragPredictorKind` / `RtPredictorKind` (`config.rs:90,79`) plus a branch in
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
  `fragindex.rs:499-536`) before its speed is trusted. Before attempting a fragindex
  optimization, note that cache-blocking, accumulator prefetch, radix
  partitioning, bin-major inversion, and distinct-m/z dedup were all measured null
  or negative in the realistic DIA regime. The real levers are top-N reduction
  (already applied, `top_n_fragments`) and per-window parallelism (already applied
  in `seed_fragindex_windows`).
- **Change fragment charges or top-N.** `charge2_from_precursor_charge`,
  `charge_by_basic_residues` and `top_n_fragments` are the knobs; all are
  sensitivity/speed tradeoffs. Lowering top-N removes collisions roughly
  proportionally (spec Section 5.5) and shrinks the library; it does not change
  matcher correctness. `charge_by_basic_residues` changes which transitions are
  scored at all, so it stays benchmark-gated.
- **Feeding an external library.** It must satisfy the loader's preconditions
  (dense `candidate_id` in precursor-m/z order, ascending `prec_mz`, decoys
  present). The decoy-builder scripts (`make_reverse_decoys.py`) sort and reindex
  to satisfy them; do not bypass that step. `predicted_irt` is not a precondition
  but is equally load-bearing, because nothing downstream regenerates it unless
  the optional DeepLC fine-tune is enabled: it must be finite, on one consistent
  scale for targets and decoys, and genuinely modification-aware if the library
  contains modforms.
