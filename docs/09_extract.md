# extract (Stage D): the core targeted extraction
> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

`extract` is the central stage of the pipeline. It takes the run-independent
spectral library (precursor and b/y fragment m/z, predicted intensities, iRT),
the per-candidate RT windows learned in `rt-im-train`, and the converted MS2
(optionally MS1) scans, and produces one apex-level PSM per surviving candidate
plus per-fragment chromatograms. Exact intensity-based scores are not computed
here; they are computed downstream in `features` from the chromatograms this
stage emits. `extract` is therefore an evidence-gathering and coarse-acceptance
stage, not a scoring stage.

The design is **data-driven and peak-major** (see the module doc comment,
`extract.rs:1`). Observed peaks probe an inverted fragment index, and a candidate
hypothesis is materialized only where fragment evidence actually collides with an
observed peak. Work scales with peak-candidate collisions, not with library size.
In wide-window DIA roughly 98% of fragment m/z collide within tolerance
(`extract.rs:743`), so one observed peak typically matches many co-isolated
candidates; the peak-claim strategies decide how that shared intensity is
apportioned.

RT is applied as a per-candidate window post-filter (the documented Stage D part
2 fallback), and the MVP is 3D so the ion-mobility (IM) dimension is absent
(`extract.rs:8`, `apex_im` is always written `None` at `extract.rs:1435`).

## Files

| Path | Role |
|---|---|
| `rust/mumdia/crates/mumdia/src/stages/extract.rs` | The stage: accumulation, cascade, apex, chromatogram/MS1 emission, schema writing |
| `rust/mumdia/crates/mumdia/src/peaks.rs` | Pure top-K chromatographic peak enumerator (`enumerate_peaks`) for the `retain_top_peaks` path |
| `rust/mumdia/crates/mumdia/src/index.rs` | `Library` (SoA library + bucketed `page_search`, `candidate_range`, `cand_frags`) |
| `rust/mumdia/crates/mumdia/src/matchers/fragindex.rs` | `FragIndex` backend (`build`, `probe_peak`, `candidate_range`), the default matcher |
| `rust/mumdia/crates/mumdia/src/spectra.rs` | `load_ms2` / `load_ms1` / `Ms1Scan` scan loaders |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | `ExtractConfig`, `GateMode`, `PeakClaim`, `MatcherKind` |
| `rust/mumdia/crates/mumdia-core/src/constants.rs` | `ISOTOPE_SPACING`, `ppm_bounds` |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` | `artifact::PSMS_EXTRACTED`, `artifact::CHROMATOGRAMS` schema ids |

Entry point: `stages::extract::run(ExtractParams)` (`extract.rs:604`), wired from
the CLI in `main.rs:495` (`Cmd::Extract`) and from the orchestrator in
`stages/run.rs:303`.

## Inputs and outputs

### Inputs (`ExtractParams`, `extract.rs:61`)

- `ms2` (Parquet): converted MS2 scans, loaded via `load_ms2` (`extract.rs:643`).
  Each `Ms2Scan` carries `rt_seconds`, an isolation `window` (`lower_mz`,
  `upper_mz`), and centroided `peaks` (`mz`, `intensity`).
- `library_precursors` + `library_fragments` (Parquet): loaded once into
  `Library::load` (`extract.rs:606`) at `cfg.bucket_size`.
- `run_windows` (Parquet): per-candidate RT windows, columns `candidate_id`,
  `rt_pred_cal`, `rt_lo`, `rt_hi` (read at `extract.rs:625`, scattered into the
  dense `rt_lo`/`rt_hi`/`rt_cal` arrays indexed by `candidate_id`,
  `extract.rs:634`). Candidates with no window row keep `[-inf, +inf]` and
  `rt_cal = 0.0` (`extract.rs:631`); the `0.0` disables the Gaussian RT prior for
  those candidates (the prior requires `rt_cal > 0`, `extract.rs:1049`).
- `ms1` (optional Parquet): MS1 scans via `load_ms1` (`extract.rs:645`). When
  absent, all MS1 columns are null.
- `mass_cal` (optional JSON, the seed's `<seed>.masscal.json`): reads
  `frag_ppm_offset` and `frag_tol_ppm` (`extract.rs:678`). Missing file =
  offset 0 and `cfg.frag_tol_ppm`.
- `restrict_candidates` (optional prior `psms.parquet`): a `candidate_id`
  allowlist (`extract.rs:611`) for the "gate first, then compete" workflow.

### Output: `psms_extracted` (`out_psms`, schema `psms_extracted` v1)

Column order and types from `extract.rs:1474`:

| Column | Type | Meaning |
|---|---|---|
| `candidate_id` | U32 | Library candidate id (sorted ascending in the file) |
| `apex_rt` | F64 | Selected apex RT (seconds) |
| `apex_im` | OptF64 | Always null (3D MVP) |
| `apex_intensity` | F32 | Full summed intensity of the apex scan group |
| `n_matched_fragments` | I32 | Distinct matched predicted fragments |
| `n_predicted_fragments` | I32 | Predicted fragment count for the candidate |
| `coelution_run` | I32 | Longest consecutive-scan co-elution run |
| `rt_pred_cal` | F64 | Calibrated predicted RT for this candidate |
| `precursor_mz` | F64 | Candidate precursor m/z |
| `charge` | I32 | Precursor charge |
| `label` | Str | `"target"` or `"decoy"` |
| `base_peptide_id` | U32 | Stripped-peptide id (for competition grouping) |
| `peptidoform` | Str | ProForma-lite string |
| `protein` | Str | Protein accession |
| `predicted_irt` | F32 | Library iRT |
| `contested_frac` | F64 | Intensity fraction lost to a better co-eluter (0 when two-pass off) |
| `ms1_isom1` | OptF64 | MS1 intensity at (mono - spacing/z) |
| `ms1_mono` | OptF64 | MS1 monoisotopic intensity |
| `ms1_iso1` | OptF64 | MS1 +1 isotope intensity |
| `ms1_iso2` | OptF64 | MS1 +2 isotope intensity |

Conditional columns (default-off, added only when the knob is set so the
production schema stays byte-identical):
- `emit_contested_features` -> `contested_count_frac` F64, `apportioned_frac` F64 (`extract.rs:1499`).
- `emit_gate_diagnostics` -> `gate_apex`, `gate_peak_spectral`, `gate_coelution`, `gate_spectral_entropy` (all F32, `extract.rs:1505`).

The three soft-competition columns are all derived from the per-candidate
`Contested` accumulator (`extract.rs:104`), whose fields are `won`/`lost`
(summed observed intensity of shared peaks this candidate won or lost as
most-eluting claimant), `n_won`/`n_lost` (the corresponding peak-instance
counts), and `apportioned` (the co-elution-weighted proportional share the
candidate would keep under `CoelutionProportional`). The columns are (all 0 when
the two-pass path did not run):
- `contested_frac` = `lost / (won + lost)` (`extract.rs:1238`): fraction of
  contested *intensity* lost to a better co-eluter.
- `contested_count_frac` = `n_lost / (n_won + n_lost)` (`extract.rs:1246`):
  fraction of contested fragment-*peaks* lost.
- `apportioned_frac` = `apportioned / (won + lost)` (`extract.rs:1254`): fraction
  of contested intensity the candidate retains under proportional apportionment
  (1 = keeps all; ~0 = a peak-borrower stripped by its co-eluting competitors).

### Output: `chromatograms` (`out_chrom`, schema `chromatograms` v1)

Column order from `extract.rs:1512`:

| Column | Type | Meaning |
|---|---|---|
| `candidate_id` | U32 | Candidate id |
| `frag_name` | Str | Fragment name (`b3`, `y7`, or `ms1_mono`/`ms1_iso1`/`ms1_iso2`) |
| `frag_mz` | F64 | Theoretical fragment m/z (or precursor isotope m/z for MS1 rows) |
| `frag_obs_mz` | F64 | Intensity-weighted observed m/z (falls back to theoretical) |
| `predicted_intensity` | F32 | Library predicted intensity (0.0 for MS1 rows) |
| `rt` | LargeListF32 | Per-scan RT axis of the trace |
| `intensity` | LargeListF32 | Per-scan intensity, 0-filled on the window grid |

One row is emitted for **every** predicted transition (`extract.rs:1295`), so
`features` sees the full predicted set and can penalize a missing strong ion. A
never-observed fragment carries an **empty** trace (not a grid-length zero
vector) to avoid bloating the list column (`extract.rs:1320`). `rt`/`intensity`
are `LargeList` (64-bit offsets) because the total list-value count can exceed
the ~2.1B 32-bit `ListArray` offset ceiling when gates are opened wide
(`extract.rs:1520`).

### Output: top-K peaks sidecar (`<out_psms>.peaks.parquet`)

Written only when `retain_top_peaks > 1` (`extract.rs:1530`), one row per
(candidate, peak). Columns: `candidate_id` U32, `peak_rank` I32, `apex_rt` F64,
`start_rt` F64, `end_rt` F64, `evidence_count` F64, `area` F64. **These peaks are
not scored** and do not affect FDR; they are candidate peaks for an offline
peak-selection model (see below).

Each artifact (`psms_extracted` and `chromatograms`, but **not** the top-K peaks
sidecar) also gets an `<artifact>.report.json` (`extract.rs:1551`) recording the
row count, blake3 content hash, stage name, schema name+version, and
`elapsed_ms`. The `params` object (`extract.rs:1562`) carries `frag_tol_ppm`
(nominal), `effective_frag_tol_ppm` (post mass-cal), `frag_ppm_offset`,
`presence_min_fragments`, `presence_min_coelution`, `min_frag_corr`, `gate_mode`,
`gate_coelution_min`, and `scan_window`. The `stats` map (`extract.rs:1548`)
carries `accepted` (the accepted-candidate count) and `scan_window`.
`model_identity` is `None` (no model in this stage).

## How it works

The control flow of `run` (`extract.rs:604`) is: load library and windows, load
scans, build the matcher, accumulate peak-candidate hits (peak-major), then per
candidate run the acceptance cascade and apex selection, emit chromatograms and
MS1 XICs, and write the tables.

### 1. Matcher and mass recalibration

The fragment tolerance and a systematic ppm offset come from `mass_cal`
(`extract.rs:678`). The offset is applied as a divisor `offset_factor = 1 +
offset*1e-6` (`extract.rs:698`); every observed peak m/z is corrected by
`q_mz = peak.mz / offset_factor` before probing (`extract.rs:740`).

Two matcher backends dispatch through `probe_matched` (`extract.rs:43`):
- `MatcherKind::Fragindex` (default): builds a `FragIndex` once at the learned
  tolerance (`extract.rs:704`). `FragIndex::probe_peak` (`fragindex.rs:152`)
  probes bins `bin-1 ..= bin+1`, verifies each posting with the exact f64 ppm
  predicate, and carries the **true generating fragment ordinal** in `post_frag`.
- Bucketed fallback: `Library::page_search` (`index.rs:247`) resolves the
  fragment ordinal by nearest stored m/z via `Library::local_frag_index`
  (`index.rs:222`). This is a semantic difference for fragments at
  sub-f32-identical m/z (`extract.rs:40`).

`Library::candidate_range` / `FragIndex::candidate_range` (`index.rs:238`,
`fragindex.rs:137`) give the half-open candidate-id range `[lo, hi)` whose
precursor m/z falls in an isolation window, exploiting that the library is sorted
by precursor m/z. This is what makes a per-window probe cheap.

### 2. Peak-major accumulation

The accumulator `acc: HashMap<u32, Vec<Hit>>` (`extract.rs:707`) maps each
candidate to the observed hits it collected. A `Hit` (`extract.rs:86`) is
`{rt, frag, inten, obs_mz}`. Entries are created lazily on the first collision.

There are three accumulation paths:

- **Parallel per-window, single-pass** (`extract_accumulate_windows`,
  `extract.rs:264`): used when `fidx` is present and there is no `restrict` list
  and the path is not two-pass. Scans are grouped by isolation window (each scan
  belongs to exactly one window), and the ~150 windows are processed in parallel
  with rayon. It is bit-identical to the serial loop because the per-candidate
  cascade rt-sorts hits before summing, and same-rt hits for a candidate all come
  from one window (`extract.rs:260`).
- **Serial single-pass** (`extract.rs:731`): the fallback when there is a
  `restrict` allowlist or no fragindex. It honors every non-co-elution
  `peak_claim` strategy.
- **Two-pass co-elution** (`extract_twopass_windows`, `extract.rs:385`): used
  when `peak_claim` is one of the `Coelution*` variants **or**
  `emit_contested_features` is set (`extract.rs:715`). Like the single-pass
  parallel path it partitions scans by isolation window and fans out over the
  ~150 windows with rayon (`extract.rs:413`), so it stays parallel even with a
  `restrict` allowlist. Pass 1 builds each candidate's per-scan elution profile
  (summed matched intensity, `extract.rs:459`). Pass 2 arbitrates each shared
  peak to the claimant most eluting at that scan (highest profile height at `rt`,
  `extract.rs:503`; ties break by higher predicted intensity then lower
  `candidate_id`, `extract.rs:508`), tracks won/lost/apportioned contested
  intensity in `Contested` (`extract.rs:104`), and reassigns intensity per the
  co-elution claim variant. The `reassign` flag (`extract.rs:817`) is true **only**
  for the three `Coelution*` variants; when the two-pass path is triggered by
  `emit_contested_features` alone (a non-co-elution `peak_claim`), pass 2 still
  computes the contested statistics but the returned accumulation is the base
  full-intensity `acc1`, not the reassigned `acc2` (`extract.rs:578`). So
  `emit_contested_features` adds the soft-competition features **without** altering
  any extracted intensity. `restrict` is honored inside both passes via the push
  closure (`extract.rs:440`, `extract.rs:484`).

Per-peak claim strategies (`PeakClaim`, applied in the single-pass loop at
`extract.rs:764` and mirrored in the parallel path):
- `None`: every matching candidate gets the full peak intensity (legacy default).
- `WinnerPredictedIntensity`: only the claimant with the highest predicted
  intensity keeps the peak; ties break by lowest `candidate_id` for determinism
  (`extract.rs:770`).
- `Proportional`: split the peak by predicted-intensity share (`extract.rs:782`).
- `CoelutionWinner` / `CoelutionProportional` / `CoelutionWinnerMargin`: two-pass
  variants keyed on elution-profile height, arbitrated in `extract.rs:540`.
  `CoelutionWinnerMargin` only strips a peak from the runner-up when the top
  eluter dominates by `peak_claim_margin` (`extract.rs:521`); otherwise the peak
  stays shared, avoiding stripping real peptides at ambiguous peaks.

### 3. Per-candidate cascade (parallel over candidates)

`acc` is drained into `(cid, hits)` in sorted `candidate_id` order
(`extract.rs:880`, `cand_hits` at `extract.rs:892`) for determinism, then each
candidate is processed in parallel via `into_par_iter().map(...)` returning
`Option<CandOut>` (`extract.rs:939`). Each candidate's work depends only on its
own hits plus read-only library/window/MS1 data, so `collect()` in the sorted
order reproduces the serial push order byte-for-byte (`extract.rs:883`).

The cheap-to-expensive acceptance cascade, in order:

1. **Distinct-fragment presence (tier b)**: distinct matched fragments must be at
   least `presence_min_matched` (`extract.rs:946`), else the candidate is dropped
   before any grouping work. `presence_min_matched`, `presence_min_fragments`, and
   `presence_min_coelution` are each floored at 1 via `.max(1)` (`extract.rs:946`,
   `extract.rs:1120`, `extract.rs:1109`), so a configured 0 still requires at
   least one fragment.
2. **Scan grouping**: hits are rt-sorted and grouped into scan groups
   `Vec<(rt, BTreeMap<frag, intensity>)>`, deduping the same fragment within one
   scan by max (`extract.rs:951`). The `BTreeMap` fixes per-scan fragment order so
   the f32 apex sum is deterministic.
3. **Acquisition-scan grid projection** (`extract.rs:978`): when
   `emit_window_grid` is on, the sparse groups are projected onto the full set of
   covering-window scans inside the RT window, so missing acquisition scans count
   as 0 and break a co-elution run rather than being invisible.
4. **Apex selection** (see below).
5. **Co-elution run**: the longest run of consecutive scan groups with at least
   `presence_min_coelution` fragments present (`extract.rs:1106`).
6. **Acceptance (tier c)** (`extract.rs:1120`): reject unless distinct fragments
   >= `presence_min_fragments`, `best_run >= scan_window` (the
   `fixed_scan_window` floor), `best_run >= min_coelution_run`, and
   `matched_fraction >= min_matched_fraction`. `matched_fraction` is
   `distinct / n_predicted` (`extract.rs:1119`); it is the primary symmetric
   discriminator (real peptides match a large fraction, chimeric and decoy matches
   a small fraction alike, keeping the target-decoy null valid).
7. **MS1 isotope evidence** (`extract.rs:1133`) computed *before* the Pearson gate
   so it can rescue a candidate. `ms1_support` requires a present mono and a +1/mono
   ratio in `[0.1, 1.5]` (`extract.rs:1150`).
8. **Pearson gate (tier d, optional)** (`extract.rs:1194`): only when
   `min_frag_corr > 0.0`. It thresholds the score of the **active** `GateMode`
   and only the active score is computed (the closures at `extract.rs:1179` are
   lazy). If `ms1_rescue` is set, a gate failure is overridden when the candidate
   has MS1 support and enough matched fragments (`extract.rs:1210`).

### 4. Apex selection (`extract.rs:1011`)

The apex is chosen among scan groups whose smoothed distinct-fragment count
qualifies, then by a per-scan score:

- **Rolling-window count** (`apex_count_window`): a centered rolling **sum** of
  the per-scan distinct-fragment count (`extract.rs:1030`). It is deliberately a
  sum, not a mean: edge truncation makes interior positions accumulate more,
  center-weighting the apex toward the RT-window centre (a mild RT prior). Window
  1 reproduces exact per-scan counts. Only scans with smoothed count `>= maxc -
  apex_count_tol` qualify (`extract.rs:1042`).
- **Signature ions** (`apex_top_fragments`, default 3 via `k_sig`,
  `extract.rs:1054`): the top-K predicted fragments, so a bright interferent on a
  non-signature ion cannot define the apex.
- **RT prior** (`apex_rt_prior_s`): when > 0 and `rt_cal > 0`, each qualifying
  scan's score is multiplied by `exp(-0.5*((rt - rt_cal)/sigma)^2)`
  (`extract.rs:1079`).
- **Scoring mode** (`apex_evidence_rank`, `extract.rs:1084`): when true, the score
  is `n_frag + sig_sum/(sig_sum+1)` times the prior, so the number of distinct
  co-eluting predicted fragments dominates and observed signature intensity only
  breaks sub-integer ties. This is interference-resistant in wide-window DIA
  because intensity is chimeric. When false (default), the legacy
  `sig_sum * prior` is used, bit-identical to the pre-feature behaviour.

`apex_intensity` reported is the full summed intensity of the winning scan group
(`extract.rs:1101`), not the signature-only score.

### 5. Gate scoring internals

Spectral-agreement scores share a helper `peak_window` (`extract.rs:158`) that
finds the contiguous elution-peak scan range `[lo, hi]` around the signature-ion
apex (scans above 10% of the reference apex height), returning `None` when there
are fewer than 3 scans or no reference signal. This restriction matters: over the
full wide extraction window the traces are mostly zeros and any correlation is
noise, so the co-elution and spectral gates are only meaningful across the
elution peak itself (`extract.rs:155`). `GateMode` scores:
- `ApexPearson`: Pearson of observed-vs-predicted intensities at the single apex
  scan (`extract.rs:1179`); one chimeric scan can dominate it.
- `PeakSpectral`: `peak_spectral_score` (`extract.rs:199`) correlates the
  peak-summed observed spectrum (each fragment integrated over the peak scans)
  with predicted intensities, averaging out a single interfered scan.
- `SpectralEntropy`: Li spectral-entropy similarity of the sqrt-transformed apex
  spectrum via the shared `features::entropy` kernel (`extract.rs:1185`).
- `Coelution`: `coelution_gate_score` (`extract.rs:225`), the
  predicted-intensity-weighted mean Pearson of each matched fragment's XIC to the
  signature-ion reference profile, restricted to the elution peak. Orthogonal to
  intensity agreement (temporal, not shape).
- `Combined`: requires `peak_spectral >= min_frag_corr` **and** `coelution >=
  gate_coelution_min` (`extract.rs:1205`).

All four diagnostic scores are computed for every accepted candidate only when
`emit_gate_diagnostics` is set (`extract.rs:1224`); otherwise they are zeroed and
not written, so the default chain pays no extra cost.

### 6. Chromatograms and MS1 XICs

Per-fragment observed m/z is an intensity-weighted mean per fragment
(`extract.rs:1264`) for mass-accuracy features. Every predicted transition emits a
chromatogram row (`extract.rs:1295`); observed fragments carry the grid-sampled
(or rt-sorted) trace, absent ones an empty trace. When MS1 is present and grid
mode is on, three MS1 isotope XICs (`ms1_mono`, `ms1_iso1`, `ms1_iso2`) are
sampled on the same scan grid via nearest-MS1-scan lookup (`extract.rs:1337`),
using `ISOTOPE_SPACING / charge` (`constants.rs:22`, value `1.003354835`) for the
isotope offsets and `sum_near` (`extract.rs:130`) to integrate within
`prec_tol_ppm`. The apex MS1 isotope intensities (`ms1_isom1`/mono/`iso1`/`iso2`)
are the per-PSM columns, taken from the nearest MS1 scan to the apex RT
(`extract.rs:1133`).

### 7. Top-K peak enumeration (`retain_top_peaks`)

When `retain_top_peaks > 1` and the candidate has groups (`extract.rs:1360`), the
per-scan distinct-fragment count profile (`count_prof`, `extract.rs:1363`) is
passed to `crate::peaks::enumerate_peaks` (`peaks.rs:52`) with `k =
retain_top_peaks`, `bound_fraction = 1/3` and `min_prominence_frac = 0.1`
(`extract.rs:1364`). `enumerate_peaks` is a pure, side-effect-free function
(`peaks.rs:1`):
1. An empty profile, `k == 0`, or an all-non-positive profile returns an empty
   vector (`peaks.rs:58`, `peaks.rs:63`).
2. **Local maxima** (`peaks.rs:71`): index `i` qualifies when its value is `>
   0`, is `>= prom_floor` (the prominence floor `min_prominence_frac *
   global_max`, `peaks.rs:66`), is **strictly** greater than its left neighbour
   (or is the left edge), and is `>=` its right neighbour (or is the right edge).
   The strict-left / non-strict-right rule makes a flat plateau register once, at
   its left edge; edges count as maxima against their single neighbour.
3. **Boundary walk** (`peaks.rs:88`): from each apex, walk left and right,
   stopping when the profile drops below `bound_fraction * apex` **or** turns back
   upward (a valley); `area` is the integrated profile over `[start_idx,
   end_idx]`.
4. **Dedup + rank** (`peaks.rs:125`): peaks are sorted strongest-first by `area`,
   then by `apex_intensity`, then by earliest `apex_idx`; a maximum whose apex
   falls inside an already-kept peak's `[start_idx, end_idx]` envelope is dropped
   (`peaks.rs:140`); the first `k` survivors are kept and assigned `rank`
   0..k-1 (`peaks.rs:148`).

`enumerate_peaks(.., k=1, ..)` returns the single global-argmax peak, so callers
can adopt it incrementally. The retained peaks rank by **co-eluting fragment
breadth (count), not intensity**, per finding A6 in
docs/18_findings_and_decisions.md (intensity is chimeric in DIA). Back in `extract`, each returned `PeakGroup` is mapped to a sidecar row
(`extract.rs:1365`): `apex_rt`/`start_rt`/`end_rt` are the RTs of the
apex/start/end scan groups, `evidence_count` is the distinct-fragment count at
the apex group (`groups[pk.apex_idx].1.len()`), and `area` is `pk.area`. The main
PSM still reports the single selected apex, so FDR is unaffected; the sidecar
peaks are unscored candidate peaks for an offline peak-selection model
(`extract.rs:1354`).

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `run` | `extract.rs:604` | Stage entry point; orchestrates load, accumulate, cascade, write |
| `ExtractParams` | `extract.rs:61` | Input path bundle + config + config hash |
| `Hit` | `extract.rs:86` | One observed hit: `rt`, `frag`, `inten`, `obs_mz` |
| `Contested` | `extract.rs:104` | Per-candidate two-pass contested-peak stats: `won`/`lost` intensity, `n_won`/`n_lost` peak counts, `apportioned` share |
| `CandOut` | `extract.rs:897` | Per-candidate parallel-map result (PSM row + chrom rows + peaks) |
| `probe_matched` | `extract.rs:43` | Dispatch a peak probe to fragindex or bucketed backend |
| `extract_accumulate_windows` | `extract.rs:264` | Parallel single-pass per-window accumulation |
| `extract_twopass_windows` | `extract.rs:385` | Parallel two-pass co-elution arbitration + contested stats |
| `peak_window` | `extract.rs:158` | Elution-peak scan range around the signature apex (10% height) |
| `peak_spectral_score` | `extract.rs:199` | Peak-integrated observed-vs-predicted Pearson |
| `coelution_gate_score` | `extract.rs:225` | Weighted mean XIC-to-reference co-elution Pearson |
| `nearest_index` | `extract.rs:113` | Binary search for nearest RT in a sorted array |
| `sum_near` | `extract.rs:130` | Sum intensities within a ppm window (m/z-sorted) |
| `enumerate_peaks` | `peaks.rs:52` | Pure top-K peak-group enumerator |
| `PeakGroup` | `peaks.rs:22` | One enumerated peak (apex/start/end idx, apex intensity, area, rank) |
| `Library::candidate_range` | `index.rs:238` | Candidate-id range for an isolation window |
| `Library::cand_frags` | `index.rs:209` | Fragment m/z, predicted intensity, name slices |
| `FragIndex::probe_peak` | `fragindex.rs:152` | Verified postings for one observed peak in a candidate range |

## Configuration

All fields live in `ExtractConfig` (`config.rs:391`); defaults in
`config.rs:508`. The config was recently pruned of dead fields in the
"clean-slate declutter" commit: `extract.scan_window_mode` (and the
`ScanWindowMode` enum), `extract.scan_scale`, `extract.k_select`,
`extract.max_fragment_charge`, and `extract.tolerance_regime` were removed. Do
not reintroduce them.

| Field | Default | Effect |
|---|---|---|
| `fixed_scan_window` | 3 | Minimum co-elution run length (`scan_window` floor); `.max(1)` at `extract.rs:845` |
| `frag_tol_ppm` | 20.0 | Fragment match tolerance (overridden by `mass_cal`) |
| `prec_tol_ppm` | 20.0 | MS1 isotope integration tolerance |
| `presence_min_matched` | 3 | Tier-b: minimum distinct matched fragments (`extract.rs:946`) |
| `presence_min_fragments` | 3 | Acceptance: minimum distinct fragments (`extract.rs:1120`) |
| `presence_min_coelution` | 2 | Min simultaneously-present fragments to extend a run (`extract.rs:1109`) |
| `min_frag_corr` | **0.2** | Pearson/gate threshold; 0 disables the gate. Relaxed from a historical 0.5 to recover low-abundance candidates (`config.rs:522`) |
| `min_matched_fraction` | 0.0 | Acceptance: min matched/predicted fraction (default off) |
| `apex_top_fragments` | 0 | Signature-ion count for apex; 0 -> default 3 (`extract.rs:1054`). Config marks it superseded by `apex_count_tol`, kept for compat (`config.rs:524`) |
| `apex_rt_prior_s` | 0.0 | Gaussian RT-prior sigma on apex tiebreak; 0 = off |
| `apex_count_tol` | 1 | Count slack for qualifying apex scans |
| `apex_count_window` | 1 | Rolling-sum width for the count profile; 1 = no smoothing. Window 5 cut AIF apex misassignment (median \|dRT\| 131s -> 9s) |
| `apex_evidence_rank` | false | Breadth-of-evidence apex vs legacy signature-intensity apex |
| `emit_window_grid` | true | Zero-filled full-window-grid chromatograms |
| `bucket_size` | 8192 | m/z bucket size (power of two) |
| `peak_claim` | `None` | Shared-peak apportionment strategy (`PeakClaim`) |
| `peak_claim_margin` | 2.0 | Dominance factor for `CoelutionWinnerMargin` |
| `emit_contested_features` | false | Adds `contested_count_frac`/`apportioned_frac`; forces the two-pass path |
| `matcher` | `Fragindex` | Fragment-matcher backend |
| `min_coelution_run` | 0 | Extra co-elution-run floor (0 = off; `scan_window` still applies) |
| `ms1_rescue` | false | Rescue Pearson-gate failures with MS1 isotope support |
| `retain_top_peaks` | 1 | K>1 writes the `.peaks.parquet` sidecar (unscored) |
| `emit_candidate_audit` | false | Candidate-audit sidecar (diagnostic) |
| `emit_gate_diagnostics` | false | Adds the four `gate_*` diagnostic columns |
| `gate_mode` | `ApexPearson` | Which score `min_frag_corr` thresholds (`GateMode`) |
| `gate_coelution_min` | 0.5 | Second threshold for `GateMode::Combined` |

Note: `emit_candidate_audit` is a declared knob but the candidate-audit write is
not present in this `extract.rs`; the audit ladder is produced by the separate
`mumdia audit` command. Treat the in-stage audit as unwired here.

### Default-off sensitivity knobs (index)

Every knob below is default-off, so the production PSM schema and per-candidate
compute stay byte-identical unless the knob is set. None has an end-to-end
identification-gain measurement yet; each must pass the entrapment-holdout gate
on >=2 datasets before being enabled by default
(`sensitivity_plan/NEXT_STEPS.md`). Two of the knobs live in other stages'
configs (`search_seed`, `rt_im_train`) but are listed here for one index.

| knob | extra columns / artifact | validation status |
|---|---|---|
| `extract.retain_top_peaks` (`config.rs:478`, default 1) | `<out_psms>.peaks.parquet` sidecar, unscored, written only when K>1 (`extract.rs:1530`); no PSM columns | ID loop not closed (sidecar peaks are unscored); gate pending |
| `extract.emit_candidate_audit` (`config.rs:483`) | none in `extract.rs` (unused there); in `run` it gates the `audit` stage -> `candidate_audit.parquet` (`run.rs:405`) | diagnostic; no ID effect |
| `extract.emit_gate_diagnostics` (`config.rs:498`) | 4 F32 cols `gate_apex`, `gate_peak_spectral`, `gate_coelution`, `gate_spectral_entropy` (`extract.rs:1505`) | diagnostic; no ID effect |
| `extract.apex_evidence_rank` (`config.rs:492`) | none; changes the apex-selection score (`extract.rs:1084`) | diagnostic support (finding A6, docs/18); no end-to-end gain measured |
| `search_seed.two_pass_mass_cal` (`config.rs:306`) | none; refits the `<seed>.masscal.json` offset + tolerance | not measured; gate pending |
| `rt_im_train.adaptive_rt_window` (`config.rs:362`) | none; per-region RT half-window widths in `run_windows` | not measured; gate pending |
| `extract.emit_contested_features` (`config.rs:454`) | 2 F64 cols `contested_count_frac`, `apportioned_frac` (`extract.rs:1499`); forces the two-pass path (`extract.rs:715`) | not measured; gate pending |
| `compete.mode` (`CompetitionMode`, `config.rs:641`, default `winner_take_all`) | retains more rows in `competed`; optional `<out>.compete_audit.parquet` via `emit_competition_audit` (`config.rs:651`) | not measured; gate pending |

## Invariants, determinism, gotchas

- **Determinism**: output is emitted in ascending `candidate_id` order
  (`extract.rs:880`); the parallel per-candidate map preserves that order via
  `collect()`. Per-scan fragment maps are `BTreeMap` so f32 apex sums have a fixed
  addition order (`extract.rs:952`). The parallel window accumulation is documented
  as bit-identical to the serial loop (`extract.rs:260`, `extract.rs:725`). A
  HashMap f32 sum shifting the apex once broke reproducibility; keep ordered maps
  and sorted iteration wherever floats are summed.
- **Default-off contract**: `retain_top_peaks=1`, `apex_evidence_rank=false`,
  `emit_contested_features=false`, `emit_gate_diagnostics=false`, `peak_claim=None`
  make the schema and per-candidate compute byte-identical to the production chain.
  Every sensitivity knob added here must keep that property.
- **The gate optimum depends on the rescorer**, not the gate in isolation. The
  full-feature search found `spectral_entropy_similarity_sqrt` the single best
  target/decoy discriminator (AUC 0.826), yet gating on it *regressed*
  end-to-end identifications versus the apex gate, because gating on the
  rescorer's own best feature enriches hard decoys. "Best discriminator" is the
  wrong criterion for a gate; the lever is the rescorer. This is why every gate
  change must pass an entrapment-holdout gate before being enabled by default.
- **Mass recal is a divisor, not a subtraction**: observed m/z is corrected by
  `q_mz = peak.mz / (1 + offset*1e-6)` (`extract.rs:740`); a missing/absent
  `masscal.json` yields offset 0 and the config tolerance.
- **Empty vs zero traces**: a never-observed predicted fragment carries an empty
  trace, not a grid-length zero vector (`extract.rs:1320`). Downstream code must
  treat an empty trace as `obs_apex = 0`.
- **`apex_im` is always null** (3D MVP); do not assume an IM value.
- **`restrict_candidates` only forces the serial path in the non-two-pass case.**
  In the non-two-pass branch the parallel window accumulation is used only when a
  fragindex is present **and** there is no `restrict` list (`extract.rs:724`); a
  `restrict` list therefore routes to the slower serial single-pass loop, which
  honors the allowlist and every non-co-elution `peak_claim` strategy. In the
  two-pass branch (`Coelution*` or `emit_contested_features`) extraction stays on
  the parallel `extract_twopass_windows` path regardless of `restrict`, which
  applies the allowlist inside each pass's push closure (`extract.rs:440`,
  `extract.rs:484`, wired at `extract.rs:832`).
- **`peaks.parquet` is not scored** and never enters FDR; it is a research sidecar.
- Chromatogram list columns are `LargeListF32` on purpose (`extract.rs:1520`); do
  not downgrade to 32-bit `ListF32` or wide-open gates overflow the offset buffer.

## Tests

Only the pure gate-scoring helpers and the peak enumerator are unit-tested; there
is no stage-level test of `run` itself (consistent with the "no stage tests for
extract" gap in CLAUDE.md). The tests encode the behavioral invariants that gate
tuning must preserve.

- `coelution_tests` (`extract.rs:1590`): co-eluting fragments score `> 0.95`
  (`extract.rs:1611`); a strong non-co-eluting interferent drops the co-elution
  score `< 0.8` (`extract.rs:1625`); fewer than 3 scan groups returns `1.0`
  (do-not-reject) rather than a low score (`extract.rs:1634`); `peak_spectral`
  scores `> 0.99` when the peak-integrated pattern matches predicted
  (`extract.rs:1648`) and still recovers a fragment that is momentarily unsampled
  at the apex scan by integrating over the peak (`extract.rs:1665`, the DIA
  scan-gap case the single-scan apex Pearson fails).
- `peaks::tests` (`peaks.rs:154`): empty/all-zero profiles yield no peaks; a
  clean triangular peak resolves apex and 1/3-height boundaries; `k=1` keeps only
  the strongest-area peak; a dominant interference peak with `k=1` discards a
  weaker true peak but `k>=2` retains it (the core sensitivity behavior,
  `peaks.rs:190`); two maxima rank by area; a left-edge-truncated peak apexes at
  index 0; the prominence filter suppresses a noise bump; a shoulder inside a
  stronger envelope collapses into it; and repeated calls are bit-identical with
  ties broken by earliest apex (`peaks.rs:259`).

## How to extend / modify

- **A new gate metric**: add a variant to `GateMode` (`config.rs:551`), add a lazy
  score closure next to `apex_pearson`/`peak_spec`/`coel` (`extract.rs:1179`), and
  a match arm in the acceptance gate (`extract.rs:1200`). If it is worth
  diagnosing, also wire it into the `emit_gate_diagnostics` tuple
  (`extract.rs:1224`) and the conditional column block (`extract.rs:1505`). Default
  the gate off and validate against entrapment before enabling.
- **A new peak-claim strategy**: add a `PeakClaim` variant (`config.rs:132`); if it
  needs elution profiles, extend the two-pass trigger (`extract.rs:715`) and the
  reassignment match in `extract_twopass_windows` (`extract.rs:540`); otherwise add
  it to the single-pass match (`extract.rs:764`) and the parallel accumulation
  (`extract.rs:316`). Keep tie-breaks deterministic (lowest `candidate_id`).
- **Scoring the top-K peaks**: the sidecar peaks are currently unscored. To close
  the loop, per-peak feature computation must be added (each retained peak needs
  the full per-peak feature vector `features` computes for the selected apex),
  then an out-of-fold peak-selection model chooses. This is the open
  `retain_top_peaks` work item in the sensitivity plan.
- **New per-PSM columns**: append to the `CandOut` struct (`extract.rs:897`), set
  it in the `Some(CandOut { .. })` block (`extract.rs:1381`), push it in the serial
  append loop (`extract.rs:1433`), and add the `Col` in the `psms_cols` vector
  (`extract.rs:1474`). Gate any non-production column behind an `emit_*` flag to
  preserve the byte-identical default schema, and bump the schema version in
  `schema.rs` if the default schema changes.
- **IM / 4D**: `apex_im` and the IM data-model hooks exist but are unfilled; a
  diaPASEF extension adds an IM window post-filter alongside the RT window and IM
  apex/feature families. It cannot be validated without diaPASEF data.
