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
(`extract.rs:678`), so one observed peak typically matches many co-isolated
candidates; the peak-claim strategies decide how that shared intensity is
apportioned.

RT is applied as a per-candidate window post-filter (the documented Stage D part
2 fallback), and the MVP is 3D so the ion-mobility (IM) dimension is absent
(`extract.rs:6`, `apex_im` is always written `None` at `extract.rs:1319`).

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

Entry point: `stages::extract::run(ExtractParams)` (`extract.rs:550`), wired from
the CLI in `main.rs:492` (`Cmd::Extract`) and from the orchestrator in
`stages/run.rs:224`.

## Inputs and outputs

### Inputs (`ExtractParams`, `extract.rs:61`)

- `ms2` (Parquet): converted MS2 scans, loaded via `load_ms2` (`extract.rs:586`).
  Each `Ms2Scan` carries `rt_seconds`, an isolation `window` (`lower_mz`,
  `upper_mz`), and centroided `peaks` (`mz`, `intensity`).
- `library_precursors` + `library_fragments` (Parquet): loaded once into
  `Library::load` (`extract.rs:552`) at `cfg.bucket_size`.
- `run_windows` (Parquet): per-candidate RT windows, columns `candidate_id`,
  `rt_pred_cal`, `rt_lo`, `rt_hi` (read at `extract.rs:568`, scattered into the
  dense `rt_lo`/`rt_hi`/`rt_cal` arrays indexed by `candidate_id`,
  `extract.rs:574`). Candidates with no window row keep `[-inf, +inf]`.
- `ms1` (optional Parquet): MS1 scans via `load_ms1` (`extract.rs:587`). When
  absent, all MS1 columns are null.
- `mass_cal` (optional JSON, the seed's `<seed>.masscal.json`): reads
  `frag_ppm_offset` and `frag_tol_ppm` (`extract.rs:621`). Missing file =
  offset 0 and `cfg.frag_tol_ppm`.
- `restrict_candidates` (optional prior `psms.parquet`): a `candidate_id`
  allowlist (`extract.rs:557`) for the "gate first, then compete" workflow.

### Output: `psms_extracted` (`out_psms`, schema `psms_extracted` v1)

Column order and types from `extract.rs:1358`:

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
- `emit_contested_features` -> `contested_count_frac` F64, `apportioned_frac` F64 (`extract.rs:1382`).
- `emit_gate_diagnostics` -> `gate_apex`, `gate_peak_spectral`, `gate_coelution`, `gate_spectral_entropy` (all F32, `extract.rs:1388`).

### Output: `chromatograms` (`out_chrom`, schema `chromatograms` v1)

Column order from `extract.rs:1396`:

| Column | Type | Meaning |
|---|---|---|
| `candidate_id` | U32 | Candidate id |
| `frag_name` | Str | Fragment name (`b3`, `y7`, or `ms1_mono`/`ms1_iso1`/`ms1_iso2`) |
| `frag_mz` | F64 | Theoretical fragment m/z (or precursor isotope m/z for MS1 rows) |
| `frag_obs_mz` | F64 | Intensity-weighted observed m/z (falls back to theoretical) |
| `predicted_intensity` | F32 | Library predicted intensity (0.0 for MS1 rows) |
| `rt` | LargeListF32 | Per-scan RT axis of the trace |
| `intensity` | LargeListF32 | Per-scan intensity, 0-filled on the window grid |

One row is emitted for **every** predicted transition (`extract.rs:1191`), so
`features` sees the full predicted set and can penalize a missing strong ion. A
never-observed fragment carries an **empty** trace (not a grid-length zero
vector) to avoid bloating the list column (`extract.rs:1213`). `rt`/`intensity`
are `LargeList` (64-bit offsets) because the total list-value count can exceed
the ~2.1B 32-bit `ListArray` offset ceiling when gates are opened wide
(`extract.rs:1404`).

### Output: top-K peaks sidecar (`<out_psms>.peaks.parquet`)

Written only when `retain_top_peaks > 1` (`extract.rs:1414`), one row per
(candidate, peak). Columns: `candidate_id` U32, `peak_rank` I32, `apex_rt` F64,
`start_rt` F64, `end_rt` F64, `evidence_count` F64, `area` F64. **These peaks are
not scored** and do not affect FDR; they are candidate peaks for an offline
peak-selection model (see below).

Each artifact also gets an `<artifact>.report.json` (`extract.rs:1439`) recording
row count, content hash, the effective/nominal fragment tolerance, and the gate
parameters.

## How it works

The control flow of `run` (`extract.rs:550`) is: load library and windows, load
scans, build the matcher, accumulate peak-candidate hits (peak-major), then per
candidate run the acceptance cascade and apex selection, emit chromatograms and
MS1 XICs, and write the tables.

### 1. Matcher and mass recalibration

The fragment tolerance and a systematic ppm offset come from `mass_cal`
(`extract.rs:621`). The offset is applied as a divisor `offset_factor = 1 +
offset*1e-6` (`extract.rs:634`); every observed peak m/z is corrected by
`q_mz = peak.mz / offset_factor` before probing (`extract.rs:676`).

Two matcher backends dispatch through `probe_matched` (`extract.rs:43`):
- `MatcherKind::Fragindex` (default): builds a `FragIndex` once at the learned
  tolerance (`extract.rs:639`). `FragIndex::probe_peak` (`fragindex.rs:152`)
  probes bins `bin-1 ..= bin+1`, verifies each posting with the exact f64 ppm
  predicate, and carries the **true generating fragment ordinal** in `post_frag`.
- Bucketed fallback: `Library::page_search` (`index.rs:242`) resolves the
  fragment ordinal by nearest stored m/z via `Library::local_frag_index`
  (`index.rs:217`). This is a semantic difference for fragments at
  sub-f32-identical m/z (`extract.rs:38`).

`Library::candidate_range` / `FragIndex::candidate_range` (`index.rs:233`,
`fragindex.rs:137`) give the half-open candidate-id range `[lo, hi)` whose
precursor m/z falls in an isolation window, exploiting that the library is sorted
by precursor m/z. This is what makes a per-window probe cheap.

### 2. Peak-major accumulation

The accumulator `acc: HashMap<u32, Vec<Hit>>` (`extract.rs:643`) maps each
candidate to the observed hits it collected. A `Hit` (`extract.rs:86`) is
`{rt, frag, inten, obs_mz}`. Entries are created lazily on the first collision.

There are three accumulation paths:

- **Parallel per-window, single-pass** (`extract_accumulate_windows`,
  `extract.rs:257`): used when `fidx` is present and there is no `restrict` list
  and the path is not two-pass. Scans are grouped by isolation window (each scan
  belongs to exactly one window), and the ~150 windows are processed in parallel
  with rayon. It is bit-identical to the serial loop because the per-candidate
  cascade rt-sorts hits before summing, and same-rt hits for a candidate all come
  from one window (`extract.rs:249`).
- **Serial single-pass** (`extract.rs:668`): the fallback when there is a
  `restrict` allowlist or no fragindex. It honors every non-co-elution
  `peak_claim` strategy.
- **Two-pass co-elution** (`extract_twopass_windows`, `extract.rs:360`): used
  when `peak_claim` is one of the `Coelution*` variants **or**
  `emit_contested_features` is set (`extract.rs:651`). Pass 1 builds each
  candidate's per-scan elution profile (summed matched intensity). Pass 2
  arbitrates each shared peak to the claimant most eluting at that scan (highest
  profile height at `rt`, `extract.rs:463`), tracks won/lost/apportioned
  contested intensity in `Contested` (`extract.rs:101`), and reassigns intensity
  per the co-elution claim variant.

Per-peak claim strategies (`PeakClaim`, applied in the single-pass loop at
`extract.rs:700` and mirrored in the parallel path):
- `None`: every matching candidate gets the full peak intensity (legacy default).
- `WinnerPredictedIntensity`: only the claimant with the highest predicted
  intensity keeps the peak; ties break by lowest `candidate_id` for determinism
  (`extract.rs:706`).
- `Proportional`: split the peak by predicted-intensity share (`extract.rs:713`).
- `CoelutionWinner` / `CoelutionProportional` / `CoelutionWinnerMargin`: two-pass
  variants keyed on elution-profile height, arbitrated in `extract.rs:501`.
  `CoelutionWinnerMargin` only strips a peak from the runner-up when the top
  eluter dominates by `peak_claim_margin` (`extract.rs:483`); otherwise the peak
  stays shared, avoiding stripping real peptides at ambiguous peaks.

### 3. Per-candidate cascade (parallel over candidates)

`acc` is drained into `(cid, hits)` in sorted `candidate_id` order
(`extract.rs:800`, `cand_hits` at `extract.rs:812`) for determinism, then each
candidate is processed in parallel via `into_par_iter().map(...)` returning
`Option<CandOut>` (`extract.rs:857`). Each candidate's work depends only on its
own hits plus read-only library/window/MS1 data, so `collect()` in the sorted
order reproduces the serial push order byte-for-byte (`extract.rs:805`).

The cheap-to-expensive acceptance cascade, in order:

1. **Distinct-fragment presence (tier b)**: distinct matched fragments must be at
   least `presence_min_matched` (`extract.rs:864`), else the candidate is dropped
   before any grouping work.
2. **Scan grouping**: hits are rt-sorted and grouped into scan groups
   `Vec<(rt, BTreeMap<frag, intensity>)>`, deduping the same fragment within one
   scan by max (`extract.rs:869`). The `BTreeMap` fixes per-scan fragment order so
   the f32 apex sum is deterministic.
3. **Acquisition-scan grid projection** (`extract.rs:896`): when
   `emit_window_grid` is on, the sparse groups are projected onto the full set of
   covering-window scans inside the RT window, so missing acquisition scans count
   as 0 and break a co-elution run rather than being invisible.
4. **Apex selection** (see below).
5. **Co-elution run**: the longest run of consecutive scan groups with at least
   `presence_min_coelution` fragments present (`extract.rs:1013`).
6. **Acceptance (tier c)** (`extract.rs:1028`): reject unless distinct fragments
   >= `presence_min_fragments`, `best_run >= scan_window` (the
   `fixed_scan_window` floor), `best_run >= min_coelution_run`, and
   `matched_fraction >= min_matched_fraction`. `matched_fraction` is
   `distinct / n_predicted` (`extract.rs:1027`); it is the primary symmetric
   discriminator (real peptides match a large fraction, chimeric and decoy matches
   a small fraction alike, keeping the target-decoy null valid).
7. **MS1 isotope evidence** (`extract.rs:1041`) computed *before* the Pearson gate
   so it can rescue a candidate. `ms1_support` requires a present mono and a +1/mono
   ratio in `[0.1, 1.5]` (`extract.rs:1058`).
8. **Pearson gate (tier d, optional)** (`extract.rs:1102`): only when
   `min_frag_corr > 0.0`. It thresholds the score of the **active** `GateMode`
   and only the active score is computed (the closures at `extract.rs:1087` are
   lazy). If `ms1_rescue` is set, a gate failure is overridden when the candidate
   has MS1 support and enough matched fragments (`extract.rs:1118`).

### 4. Apex selection (`extract.rs:942`)

The apex is chosen among scan groups whose smoothed distinct-fragment count
qualifies, then by a per-scan score:

- **Rolling-window count** (`apex_count_window`): a centered rolling **sum** of
  the per-scan distinct-fragment count (`extract.rs:945`). It is deliberately a
  sum, not a mean: edge truncation makes interior positions accumulate more,
  center-weighting the apex toward the RT-window centre (a mild RT prior). Window
  1 reproduces exact per-scan counts. Only scans with smoothed count `>= maxc -
  apex_count_tol` qualify (`extract.rs:957`).
- **Signature ions** (`apex_top_fragments`, default 3 via `k_sig`,
  `extract.rs:969`): the top-K predicted fragments, so a bright interferent on a
  non-signature ion cannot define the apex.
- **RT prior** (`apex_rt_prior_s`): when > 0 and `rt_cal > 0`, each qualifying
  scan's score is multiplied by `exp(-0.5*((rt - rt_cal)/sigma)^2)`
  (`extract.rs:987`).
- **Scoring mode** (`apex_evidence_rank`, `extract.rs:992`): when true, the score
  is `n_frag + sig_sum/(sig_sum+1)` times the prior, so the number of distinct
  co-eluting predicted fragments dominates and observed signature intensity only
  breaks sub-integer ties. This is interference-resistant in wide-window DIA
  because intensity is chimeric. When false (default), the legacy
  `sig_sum * prior` is used, bit-identical to the pre-feature behaviour.

`apex_intensity` reported is the full summed intensity of the winning scan group
(`extract.rs:1009`), not the signature-only score.

### 5. Gate scoring internals

Spectral-agreement scores share a helper `peak_window` (`extract.rs:156`) that
finds the contiguous elution-peak scan range `[lo, hi]` around the signature-ion
apex (scans above 10% of the reference apex height), returning `None` when there
are fewer than 3 scans or no reference signal. This restriction matters: over the
full wide extraction window the traces are mostly zeros and any correlation is
noise, so the co-elution and spectral gates are only meaningful across the
elution peak itself (`extract.rs:151`). `GateMode` scores:
- `ApexPearson`: Pearson of observed-vs-predicted intensities at the single apex
  scan (`extract.rs:1087`); one chimeric scan can dominate it.
- `PeakSpectral`: `peak_spectral_score` (`extract.rs:192`) correlates the
  peak-summed observed spectrum (each fragment integrated over the peak scans)
  with predicted intensities, averaging out a single interfered scan.
- `SpectralEntropy`: Li spectral-entropy similarity of the sqrt-transformed apex
  spectrum via the shared `features::entropy` kernel (`extract.rs:1093`).
- `Coelution`: `coelution_gate_score` (`extract.rs:218`), the
  predicted-intensity-weighted mean Pearson of each matched fragment's XIC to the
  signature-ion reference profile, restricted to the elution peak. Orthogonal to
  intensity agreement (temporal, not shape).
- `Combined`: requires `peak_spectral >= min_frag_corr` **and** `coelution >=
  gate_coelution_min` (`extract.rs:1113`).

All four diagnostic scores are computed for every accepted candidate only when
`emit_gate_diagnostics` is set (`extract.rs:1132`); otherwise they are zeroed and
not written, so the default chain pays no extra cost.

### 6. Chromatograms and MS1 XICs

Per-fragment observed m/z is an intensity-weighted mean per fragment
(`extract.rs:1159`) for mass-accuracy features. Every predicted transition emits a
chromatogram row (`extract.rs:1191`); observed fragments carry the grid-sampled
(or rt-sorted) trace, absent ones an empty trace. When MS1 is present and grid
mode is on, three MS1 isotope XICs (`ms1_mono`, `ms1_iso1`, `ms1_iso2`) are
sampled on the same scan grid via nearest-MS1-scan lookup (`extract.rs:1222`),
using `ISOTOPE_SPACING / charge` (`constants.rs:22`, value `1.003354835`) for the
isotope offsets and `sum_near` (`extract.rs:128`) to integrate within
`prec_tol_ppm`. The apex MS1 isotope intensities (`ms1_isom1`/mono/`iso1`/`iso2`)
are the per-PSM columns, taken from the nearest MS1 scan to the apex RT
(`extract.rs:1041`).

### 7. Top-K peak enumeration (`retain_top_peaks`)

When `retain_top_peaks > 1` (`extract.rs:1245`), the per-scan distinct-fragment
count profile is passed to `crate::peaks::enumerate_peaks` (`peaks.rs:52`) with
`bound_fraction = 1/3` and `min_prominence_frac = 0.1`. `enumerate_peaks` finds
local maxima above a prominence floor, walks fractional-height boundaries
(stopping below `bound_fraction * apex` or at a valley, `peaks.rs:88`),
deduplicates maxima falling inside a stronger peak's envelope (`peaks.rs:136`),
and returns up to K groups ranked strongest-first by integrated `area` with ties
broken by earliest `apex_idx`. `enumerate_peaks(.., k=1, ..)` returns the single
global-argmax peak, so callers can adopt it incrementally. The retained peaks
rank by **co-eluting fragment breadth (count), not intensity**, per the finding
that intensity is chimeric in DIA. The main PSM still reports the single selected
apex, so FDR is unaffected; the sidecar peaks are unscored candidate peaks for an
offline peak-selection model (`extract.rs:1239`).

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `run` | `extract.rs:550` | Stage entry point; orchestrates load, accumulate, cascade, write |
| `ExtractParams` | `extract.rs:61` | Input path bundle + config + config hash |
| `Hit` | `extract.rs:86` | One observed hit: `rt`, `frag`, `inten`, `obs_mz` |
| `Contested` | `extract.rs:101` | Per-candidate won/lost/apportioned contested intensity (two-pass) |
| `CandOut` | `extract.rs:815` | Per-candidate parallel-map result (PSM row + chrom rows + peaks) |
| `probe_matched` | `extract.rs:43` | Dispatch a peak probe to fragindex or bucketed backend |
| `extract_accumulate_windows` | `extract.rs:257` | Parallel single-pass per-window accumulation |
| `extract_twopass_windows` | `extract.rs:360` | Parallel two-pass co-elution arbitration + contested stats |
| `peak_window` | `extract.rs:156` | Elution-peak scan range around the signature apex (10% height) |
| `peak_spectral_score` | `extract.rs:192` | Peak-integrated observed-vs-predicted Pearson |
| `coelution_gate_score` | `extract.rs:218` | Weighted mean XIC-to-reference co-elution Pearson |
| `nearest_index` | `extract.rs:111` | Binary search for nearest RT in a sorted array |
| `sum_near` | `extract.rs:128` | Sum intensities within a ppm window (m/z-sorted) |
| `enumerate_peaks` | `peaks.rs:52` | Pure top-K peak-group enumerator |
| `PeakGroup` | `peaks.rs:22` | One enumerated peak (apex/start/end idx, apex intensity, area, rank) |
| `Library::candidate_range` | `index.rs:233` | Candidate-id range for an isolation window |
| `Library::cand_frags` | `index.rs:208` | Fragment m/z, predicted intensity, name slices |
| `FragIndex::probe_peak` | `fragindex.rs:152` | Verified postings for one observed peak in a candidate range |

## Configuration

All fields live in `ExtractConfig` (`config.rs:431`); defaults in
`config.rs:548`. The config was recently pruned of dead fields in the
"clean-slate declutter" commit: `extract.scan_window_mode` (and the
`ScanWindowMode` enum), `extract.scan_scale`, `extract.k_select`,
`extract.max_fragment_charge`, and `extract.tolerance_regime` were removed. Do
not reintroduce them.

| Field | Default | Effect |
|---|---|---|
| `fixed_scan_window` | 3 | Minimum co-elution run length (`scan_window` floor); `.max(1)` at `extract.rs:763` |
| `frag_tol_ppm` | 20.0 | Fragment match tolerance (overridden by `mass_cal`) |
| `prec_tol_ppm` | 20.0 | MS1 isotope integration tolerance |
| `presence_min_matched` | 3 | Tier-b: minimum distinct matched fragments (`extract.rs:864`) |
| `presence_min_fragments` | 3 | Acceptance: minimum distinct fragments (`extract.rs:1028`) |
| `presence_min_coelution` | 2 | Min simultaneously-present fragments to extend a run (`extract.rs:1017`) |
| `min_frag_corr` | **0.2** | Pearson/gate threshold; 0 disables the gate. Relaxed from a historical 0.5 to recover low-abundance candidates (`config.rs:559`) |
| `min_matched_fraction` | 0.0 | Acceptance: min matched/predicted fraction (default off) |
| `apex_top_fragments` | 0 | Signature-ion count for apex; 0 -> default 3 (`extract.rs:969`) |
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

## Invariants, determinism, gotchas

- **Determinism**: output is emitted in ascending `candidate_id` order
  (`extract.rs:800`); the parallel per-candidate map preserves that order via
  `collect()`. Per-scan fragment maps are `BTreeMap` so f32 apex sums have a fixed
  addition order (`extract.rs:872`). The parallel window accumulation is documented
  as bit-identical to the serial loop (`extract.rs:249`, `extract.rs:659`). A
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
  `q_mz = peak.mz / (1 + offset*1e-6)` (`extract.rs:676`); a missing/absent
  `masscal.json` yields offset 0 and the config tolerance.
- **Empty vs zero traces**: a never-observed predicted fragment carries an empty
  trace, not a grid-length zero vector (`extract.rs:1213`). Downstream code must
  treat an empty trace as `obs_apex = 0`.
- **`apex_im` is always null** (3D MVP); do not assume an IM value.
- **`restrict_candidates` routes to the serial path** (`extract.rs:660`), which is
  slower but honors the allowlist filter and every `peak_claim` strategy.
- **`peaks.parquet` is not scored** and never enters FDR; it is a research sidecar.
- Chromatogram list columns are `LargeListF32` on purpose (`extract.rs:1404`); do
  not downgrade to 32-bit `ListF32` or wide-open gates overflow the offset buffer.

## How to extend / modify

- **A new gate metric**: add a variant to `GateMode` (`config.rs:591`), add a lazy
  score closure next to `apex_pearson`/`peak_spec`/`coel` (`extract.rs:1087`), and
  a match arm in the acceptance gate (`extract.rs:1108`). If it is worth
  diagnosing, also wire it into the `emit_gate_diagnostics` tuple
  (`extract.rs:1132`) and the conditional column block (`extract.rs:1388`). Default
  the gate off and validate against entrapment before enabling.
- **A new peak-claim strategy**: add a `PeakClaim` variant (`config.rs:164`); if it
  needs elution profiles, extend the two-pass trigger (`extract.rs:651`) and the
  reassignment match in `extract_twopass_windows` (`extract.rs:501`); otherwise add
  it to the single-pass match (`extract.rs:700`) and the parallel accumulation
  (`extract.rs:306`). Keep tie-breaks deterministic (lowest `candidate_id`).
- **Scoring the top-K peaks**: the sidecar peaks are currently unscored. To close
  the loop, per-peak feature computation must be added (each retained peak needs
  the full per-peak feature vector `features` computes for the selected apex),
  then an out-of-fold peak-selection model chooses. This is the open
  `retain_top_peaks` work item in the sensitivity plan.
- **New per-PSM columns**: append to the `CandOut` struct (`extract.rs:815`), set
  it in the `Some(CandOut { .. })` block (`extract.rs:1265`), push it in the serial
  append loop (`extract.rs:1305`), and add the `Col` in the `psms_cols` vector
  (`extract.rs:1358`). Gate any non-production column behind an `emit_*` flag to
  preserve the byte-identical default schema, and bump the schema version in
  `schema.rs` if the default schema changes.
- **IM / 4D**: `apex_im` and the IM data-model hooks exist but are unfilled; a
  diaPASEF extension adds an IM window post-filter alongside the RT window and IM
  apex/feature families. It cannot be validated without diaPASEF data.
