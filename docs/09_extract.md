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
(`extract.rs:1479`), so one observed peak typically matches many co-isolated
candidates; the peak-claim strategies decide how that shared intensity is
apportioned.

RT is applied as a per-candidate window post-filter (the documented Stage D part
2 fallback), and the MVP is 3D so the ion-mobility (IM) dimension is absent
(`extract.rs:8`, `apex_im` is always written `None` at `extract.rs:2484`).

## Files

| Path | Role |
|---|---|
| `rust/mumdia/crates/mumdia/src/stages/extract.rs` | The stage: accumulation, cascade, apex, chromatogram/MS1 emission, schema writing |
| `rust/mumdia/crates/mumdia/src/peaks.rs` | Pure top-K chromatographic peak enumerator (`enumerate_peaks`) for the `retain_top_peaks` path |
| `rust/mumdia/crates/mumdia/src/index.rs` | `Library` (SoA library + bucketed `page_search`, `candidate_range`, `cand_frags`) |
| `rust/mumdia/crates/mumdia/src/matchers/fragindex.rs` | `FragIndex` backend (`build`, `probe_peak`, `candidate_range`), the default matcher |
| `rust/mumdia/crates/mumdia/src/spectra.rs` | `load_ms2` / `load_ms1` / `Ms1Scan` scan loaders |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | `ExtractConfig`, `GateMode`, `PeakClaim`, `ClaimCues`, `MatcherKind` |
| `rust/mumdia/crates/mumdia-core/src/constants.rs` | `ISOTOPE_SPACING`, `ppm_bounds` |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` | `artifact::PSMS_EXTRACTED`, `artifact::CHROMATOGRAMS` schema ids |

Entry point: `stages::extract::run(ExtractParams)` (`extract.rs:1300`), wired from
the CLI in `main.rs:535` (`Cmd::Extract`) and from the orchestrator in
`stages/run.rs:326`.

## Inputs and outputs

### Inputs (`ExtractParams`, `extract.rs:83`)

- `ms2` (Parquet): converted MS2 scans, loaded via `load_ms2` (`extract.rs:1348`).
  Each `Ms2Scan` carries `rt_seconds`, an isolation `window` (`lower_mz`,
  `upper_mz`), and centroided `peaks` (`mz`, `intensity`). This stage applies no
  peak cap of its own: the MS2 peak budget is fixed at conversion time by
  `--top-peaks-ms2` (`convert.rs:76-79`) and everything below operates on
  whatever survived it. `search_seed.top_n_peaks` (`config.rs:410-415`) is a
  separate, non-destructive seed-probing limit and does not change the evidence
  available here.
- `library_precursors` + `library_fragments` (Parquet): loaded once into
  `Library::load_with` (`extract.rs:1306`) at `cfg.bucket_size`. `load_with`
  skips building the bucketed `page_search` index when the fragindex backend is
  selected (`index.rs:73`).
- `run_windows` (Parquet): per-candidate RT windows, columns `candidate_id`,
  `rt_pred_cal`, `rt_lo`, `rt_hi` (read at `extract.rs:1330`, scattered into the
  dense `rt_lo`/`rt_hi`/`rt_cal` arrays indexed by `candidate_id`,
  `extract.rs:1342`). Candidates with no window row keep `[-inf, +inf]` and
  `rt_cal = 0.0` (`extract.rs:1336`); the `0.0` disables the Gaussian RT prior for
  those candidates (the prior requires `rt_cal > 0`, `extract.rs:1855`).
- `ms1` (optional Parquet): MS1 scans via `load_ms1` (`extract.rs:1350`). When
  absent, all MS1 columns are null.
- `mass_cal` (optional JSON, the seed's `<seed>.masscal.json`): reads
  `frag_ppm_offset` and `frag_tol_ppm` (`extract.rs:1394`), and optionally an
  m/z-dependent offset grid. Missing file = offset 0 and `cfg.frag_tol_ppm`.
- `restrict_candidates` (optional prior `psms.parquet`): a `candidate_id`
  allowlist (`extract.rs:1316`) for the "gate first, then compete" workflow.

### Output: `psms_extracted` (`out_psms`, schema `psms_extracted` v1)

Column order and types from `extract.rs:2480`:

| Column | Type | Meaning |
|---|---|---|
| `candidate_id` | U32 | Library candidate id (sorted ascending in the file) |
| `peak_rank` | I32 | Chromatographic peak rank; 0 = selected apex. Ranks >= 1 exist only when `promote_top_peaks > 1` (`extract.rs:2205`) |
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
- `emit_contested_features` -> `contested_count_frac` F64, `apportioned_frac` F64 (`extract.rs:2505`).
- `emit_gate_diagnostics` -> `gate_apex`, `gate_peak_spectral`, `gate_coelution`, `gate_spectral_entropy` (all F32, `extract.rs:2511`).
- `emit_demix_features` -> `deconv_explained_frac`, `deconv_active`,
  `deconv_share`, `deconv_max_collinearity`, `shadow_kept_frac` (all F32,
  `extract.rs:2517`), from the spectrum-centric NNLS demix at the apex scan.

The three soft-competition columns are all derived from the per-candidate
`Contested` accumulator (`extract.rs:130`), whose fields are `won`/`lost`
(summed observed intensity of shared peaks this candidate won or lost as
most-eluting claimant), `n_won`/`n_lost` (the corresponding peak-instance
counts), and `apportioned` (the co-elution-weighted proportional share the
candidate would keep under `CoelutionProportional`). The columns are (all 0 when
the two-pass path did not run):
- `contested_frac` = `lost / (won + lost)` (`extract.rs:2048`): fraction of
  contested *intensity* lost to a better co-eluter.
- `contested_count_frac` = `n_lost / (n_won + n_lost)` (`extract.rs:2056`):
  fraction of contested fragment-*peaks* lost.
- `apportioned_frac` = `apportioned / (won + lost)` (`extract.rs:2064`): fraction
  of contested intensity the candidate retains under proportional apportionment
  (1 = keeps all; ~0 = a peak-borrower stripped by its co-eluting competitors).

### Output: `chromatograms` (`out_chrom`, schema `chromatograms` v1)

Column order from `extract.rs:2528`:

| Column | Type | Meaning |
|---|---|---|
| `candidate_id` | U32 | Candidate id |
| `frag_name` | Str | Fragment name (`b3`, `y7`, or `ms1_mono`/`ms1_iso1`/`ms1_iso2`) |
| `frag_mz` | F64 | Theoretical fragment m/z (or precursor isotope m/z for MS1 rows) |
| `frag_obs_mz` | F64 | Intensity-weighted observed m/z (falls back to theoretical) |
| `predicted_intensity` | F32 | Library predicted intensity (0.0 for MS1 rows) |
| `rt` | LargeListF32 | Per-scan RT axis of the trace |
| `intensity` | LargeListF32 | Per-scan intensity, 0-filled on the window grid |

One row is emitted for **every** predicted transition (`extract.rs:2095`), so
`features` sees the full predicted set and can penalize a missing strong ion. A
never-observed fragment carries an **empty** trace (not a grid-length zero
vector) to avoid bloating the list column (`extract.rs:2098`). `rt`/`intensity`
are `LargeList` (64-bit offsets) because the total list-value count can exceed
the ~2.1B 32-bit `ListArray` offset ceiling when gates are opened wide
(`extract.rs:2534`). `frag_name` is written as a string but the library stores
fragment names as interned u16 dictionary ids, resolved back through
`Library::frag_name_str` (`extract.rs:2137`).

### Output: top-K peaks sidecar (`<out_psms>.peaks.parquet`)

Written only when `retain_top_peaks > 1` produced rows (`extract.rs:2544`), one
row per (candidate, peak). Columns: `candidate_id` U32, `peak_rank` I32,
`apex_rt` F64, `start_rt` F64, `end_rt` F64, `evidence_count` F64, `area` F64.
**These peaks are not scored** and do not affect FDR; they are candidate peaks
for an offline peak-selection model (see below). This sidecar is distinct from
`promote_top_peaks`, which instead emits extra `psms_extracted` rows carrying
`peak_rank >= 1`.

Each artifact (`psms_extracted` and `chromatograms`, but **not** the top-K peaks
sidecar) also gets an `<artifact>.report.json` (`extract.rs:2569`) recording the
row count, blake3 content hash, stage name, schema name+version, and
`elapsed_ms`. The `params` object (`extract.rs:2577`) carries `frag_tol_ppm`
(nominal), `effective_frag_tol_ppm` (post mass-cal), `frag_ppm_offset`,
`presence_min_fragments`, `presence_min_coelution`, `gate_min_score`, `gate_mode`,
`gate_coelution_min`, and `scan_window`. The `stats` map (`extract.rs:2562`)
carries `accepted` (the accepted-candidate count) and `scan_window`.
`model_identity` is `None` (no model in this stage).

## How it works

The control flow of `run` (`extract.rs:1300`) is: load library and windows, load
scans, build the matcher, accumulate peak-candidate hits (peak-major), then per
candidate run the acceptance cascade and apex selection, emit chromatograms and
MS1 XICs, and write the tables.

### 1. Matcher and mass recalibration

The fragment tolerance and a systematic ppm offset come from `mass_cal`
(`extract.rs:1394`). The offset is applied as a **divisor**, not a subtraction:
`MassOffset::factor_at(mz)` returns `1 + ppm(mz)*1e-6` (`extract.rs:640`) and
every observed peak m/z is corrected by `q_mz = peak.mz / factor_at(peak.mz)`
before probing (`extract.rs:705`). The offset may be a single scalar ppm or, when
the seed fitted one, an m/z-dependent grid that is linearly interpolated and
clamped at the ends (`extract.rs:632`), so the correction is not necessarily
constant across the m/z range.

Two matcher backends dispatch through `Prober::probe` (`extract.rs:57`):
- `MatcherKind::Fragindex` (default): builds a `FragIndex` once at the learned
  tolerance (`extract.rs:1431`). `FragIndex::probe_peak` (`fragindex.rs:152`)
  probes bins `bin-1 ..= bin+1`, verifies each posting with the exact f64 ppm
  predicate, and carries the **true generating fragment ordinal** in `post_frag`.
- Bucketed fallback (`MatcherKind::Bucketed`): `Library::page_search`
  (`index.rs:350`) resolves the fragment ordinal by nearest stored m/z via
  `Library::local_frag_index` (`index.rs:325`). This is a semantic difference for
  fragments at sub-f32-identical m/z (`extract.rs:48`). The bucketed index is
  only built when this backend is selected, so choosing it is the only reason to
  pay for it (`extract.rs:1302`).

`Library::candidate_range` / `FragIndex::candidate_range` (`index.rs:341`,
`fragindex.rs:137`) give the half-open candidate-id range `[lo, hi)` whose
precursor m/z falls in an isolation window, exploiting that the library is sorted
by precursor m/z. This is what makes a per-window probe cheap.

### 2. Peak-major accumulation

The accumulator `acc: HashMap<u32, Vec<Hit>>` (`extract.rs:1435`) maps each
candidate to the observed hits it collected. A `Hit` (`extract.rs:108`) is
`{rt, frag, inten, obs_mz}`. Entries are created lazily on the first collision.

There are three accumulation paths:

- **Parallel per-window, single-pass** (`extract_accumulate_windows`,
  `extract.rs:662`): used when `fidx` is present and there is no `restrict` list
  and the path is not two-pass. Scans are grouped by isolation window (each scan
  belongs to exactly one window), and the ~150 windows are processed in parallel
  with rayon. It is bit-identical to the serial loop because the per-candidate
  cascade rt-sorts hits before summing, and same-rt hits for a candidate all come
  from one window (`extract.rs:1456`).
- **Serial single-pass** (`extract.rs:1462`): the fallback when there is a
  `restrict` allowlist or no fragindex. It honors every non-co-elution
  `peak_claim` strategy.
- **Two-pass co-elution** (`extract_twopass_windows`, `extract.rs:787`): used
  when `peak_claim` is one of the `Coelution*` variants **or**
  `emit_contested_features` is set (`extract.rs:1443`). Like the single-pass
  parallel path it partitions scans by isolation window and fans out over the
  ~150 windows with rayon (`extract.rs:782`), so it stays parallel even with a
  `restrict` allowlist. Pass 1 builds each candidate's per-scan elution profile
  (summed matched intensity, `extract.rs:872`). Pass 2 arbitrates each shared
  peak to the claimant with the highest competition weight at that scan
  (`extract.rs:1200`), tracks won/lost/apportioned contested intensity in
  `Contested` (`extract.rs:130`), and reassigns intensity per the co-elution
  claim variant (`extract.rs:1236`). The `reassign` flag (`extract.rs:1557`) is
  true for `CoelutionWinner`, `CoelutionProportional`, `CoelutionWinnerMargin`,
  `CoelutionDemix`, and `CoelutionShadow`, and for `CoelutionMultiCue` **only**
  when `claim_cues.reassign` is also set: `CoelutionMultiCue` ships
  non-destructive, feeding the contested/apportioned features only. When the
  two-pass path is triggered by `emit_contested_features` alone (a non-co-elution
  `peak_claim`), pass 2 still computes the contested statistics but the returned
  accumulation is the base full-intensity `acc1`, not the reassigned `acc2`
  (`extract.rs:1274`). So `emit_contested_features` adds the soft-competition
  features **without** altering any extracted intensity. `restrict` is honored
  inside both passes via the push closure (`extract.rs:853`, `extract.rs:907`).

Per-peak claim strategies (`PeakClaim`, `config.rs:132`; applied in the serial
single-pass loop at `extract.rs:1500` and mirrored in the parallel path at
`extract.rs:718`):
- `None`: every matching candidate gets the full peak intensity (legacy default).
- `WinnerPredictedIntensity`: only the claimant with the highest predicted
  intensity keeps the peak; ties break by lowest `candidate_id` for determinism
  (`extract.rs:1501`).
- `Proportional`: split the peak by predicted-intensity share (`extract.rs:1518`).
- `CoelutionWinner` / `CoelutionProportional` / `CoelutionWinnerMargin`: two-pass
  variants keyed on elution-profile height, arbitrated in `extract.rs:1236`.
  `CoelutionWinnerMargin` only strips a peak from the runner-up when the top
  eluter dominates by `peak_claim_margin` (`extract.rs:1218`); otherwise the peak
  stays shared, avoiding stripping real peptides at ambiguous peaks.
- `CoelutionMultiCue`: the per-claimant weight is the elution-profile height
  multiplied by the composable cues enabled in `ClaimCues` (`config.rs:192`):
  sub-tolerance m/z proximity, RT prior, MS1 precursor support, and so on
  (`claim_cue_multiplier`, `extract.rs:163`). Every cue defaults to 1.0, so with
  no cue enabled the arbitration reduces bit-identically to `CoelutionWinner`
  (`extract.rs:1184`).
- `CoelutionDemix`: at each scan (subject to `demix_scan_stride`) the co-isolated
  candidate-by-fragment design matrix is solved by ridge NNLS
  (`demix_solve_scan`, `extract.rs:279`) and each shared peak is split in
  proportion to the joint deconvolution rather than stripped winner-take-all.
  Always destructive.
- `CoelutionShadow`: solver-free shadow subtraction; each claimant's peak
  intensity is reduced by the portion other co-eluters explain.

These last three, and the `emit_demix_features` columns they share machinery
with, are recent and unvalidated. Treat them as research paths, not tuning knobs.

### 3. Per-candidate cascade (parallel over candidates)

`acc` is drained into `(cid, hits)` in sorted `candidate_id` order (`cand_hits`
at `extract.rs:1651`) for determinism, then each candidate is processed in
parallel via `into_par_iter().map(...)` returning `Vec<CandOut>`
(`extract.rs:1721`). Each candidate's work depends only on its own hits plus
read-only library/window/MS1 data, so `collect()` in the sorted order reproduces
the serial push order byte-for-byte (`extract.rs:1645`). The map returns a `Vec`
rather than an `Option` because `promote_top_peaks > 1` can emit more than one row
per candidate.

The cheap-to-expensive acceptance cascade, in order:

1. **Distinct-fragment presence (tier b)**: distinct matched fragments must be at
   least `presence_min_matched` (`extract.rs:1727`), else the candidate is dropped
   before any grouping work. `presence_min_matched`, `presence_min_fragments`, and
   `presence_min_coelution` are each floored at 1 via `.max(1)` (`extract.rs:1727`,
   `extract.rs:1926`, `extract.rs:1915`), so a configured 0 still requires at
   least one fragment.
2. **Scan grouping**: hits are rt-sorted and grouped into scan groups
   `Vec<(rt, BTreeMap<frag, intensity>)>`, deduping the same fragment within one
   scan by max (`extract.rs:1735`). The `BTreeMap` fixes per-scan fragment order so
   the f32 apex sum is deterministic.
3. **Acquisition-scan grid projection** (`extract.rs:1782`): when
   `emit_window_grid` is on, the sparse groups are projected onto the full set of
   covering-window scans inside the RT window, so missing acquisition scans count
   as 0 and break a co-elution run rather than being invisible.
4. **Apex selection** (see below).
5. **Co-elution run**: the longest run of consecutive scan groups with at least
   `presence_min_coelution` fragments present (`extract.rs:1911`).
6. **Acceptance (tier c)** (`extract.rs:1926`): reject unless distinct fragments
   >= `presence_min_fragments`, `best_run >= scan_window` (the
   `fixed_scan_window` floor), `best_run >= min_coelution_run`, and
   `matched_fraction >= min_matched_fraction`. `matched_fraction` is
   `distinct / n_predicted` (`extract.rs:1925`); it is the primary symmetric
   discriminator (real peptides match a large fraction, chimeric and decoy matches
   a small fraction alike, keeping the target-decoy null valid).
7. **MS1 isotope evidence** (`extract.rs:1948`) computed *before* the Pearson gate
   so it can rescue a candidate. `ms1_support` requires a present mono and a +1/mono
   ratio in `[0.1, 1.5]` (`extract.rs:1960`).
8. **Pearson gate (tier d, optional)** (`extract.rs:2004`): only when
   `gate_min_score > 0.0`. It thresholds the score of the **active** `GateMode`
   and only the active score is computed (the closures at `extract.rs:1989` are
   lazy). If `ms1_rescue` is set, a gate failure is overridden when the candidate
   has MS1 support and enough matched fragments (`extract.rs:2020`).

The two presence thresholds (`presence_min_matched` at tier b,
`presence_min_fragments` at tier c) are the main coupling between this stage and
`convert`. Because the `--top-peaks-ms2` truncation is baked into the spectra
artifact and extract adds no cap of its own, a cap that removes most MS2 peaks
removes the fragment evidence these thresholds count, and the candidate is
rejected before any gate or score is reached. Measured on a 50-window Orbitrap
DIA run, `--top-peaks-ms2 300` discarded 78.6% of MS2 peaks; a `mumdia audit`
ladder restricted to peptides that an external library-free search confirms are
present in the run showed 49,105 of 78,782 (62.3%) stopping at
`candidate_generated` with rejection code `NO_PEAK_GROUP` (`rejection.rs:62`).
Only 5,380 were lost to FDR and 355 to competition. Replaying those candidates
against the uncapped artifact recovered 41,948 of the 49,105 (85.4%), which is
what identifies the cap as the cause.

`NO_PEAK_GROUP` does not mean "never assembled `presence_min_fragments` distinct
fragments", although this document previously said so. `audit.rs` reads a
per-candidate audit table that `extract` does not write -- `emit_candidate_audit`
is unwired -- so the reason map is always empty and the `_ => NoPeakGroup`
catch-all absorbs presence failures, matched-fraction failures and every
extraction-gate rejection alike. Read it as "did not survive extract". The
attribution to the peak cap above stands on the replay experiment, not on the
label.

When `NO_PEAK_GROUP` dominates an audit ladder, check the conversion cap before
tuning the presence thresholds or gates here; see docs/04_convert.md for the
peak census and the cap dose-response.

### 4. Apex selection (`extract.rs:1793`)

The apex is chosen among scan groups whose smoothed distinct-fragment count
qualifies, then by a per-scan score:

- **Rolling-window count** (`apex_count_window`): a centered rolling **sum** of
  the per-scan distinct-fragment count (`extract.rs:1809`). It is deliberately a
  sum, not a mean: edge truncation makes interior positions accumulate more,
  center-weighting the apex toward the RT-window centre (a mild RT prior). Window
  1 reproduces exact per-scan counts. Only scans with smoothed count `>= maxc -
  apex_count_tol` qualify (`extract.rs:1848`). An opt-in Gaussian smoother over
  the same profile is available via `apex_gaussian_sigma_scans` (default 0.0 =
  off, `extract.rs:1813`).
- **Signature ions** (`apex_top_fragments`, default 3 via `k_sig`,
  `extract.rs:1860`): the top-K predicted fragments, so a bright interferent on a
  non-signature ion cannot define the apex.
- **RT prior** (`apex_rt_prior_s`): when > 0 and `rt_cal > 0`
  (`extract.rs:1855`), each qualifying scan's score is multiplied by
  `exp(-0.5*((rt - rt_cal)/sigma)^2)` (`extract.rs:1886`).
- **Scoring mode** (`apex_evidence_rank`, `extract.rs:1890`): when true, the score
  is `n_frag + sig_sum/(sig_sum+1)` times the prior, so the number of distinct
  co-eluting predicted fragments dominates and observed signature intensity only
  breaks sub-integer ties. This is interference-resistant in wide-window DIA
  because intensity is chimeric. When false (default), the legacy
  `sig_sum * prior` is used (`extract.rs:1902`), bit-identical to the pre-feature
  behaviour.

`apex_intensity` reported is the full summed intensity of the winning scan group
(`extract.rs:1907`), not the signature-only score.

### 5. Gate scoring internals

Spectral-agreement scores share a helper `peak_window` (`extract.rs:523`) that
finds the contiguous elution-peak scan range `[lo, hi]` around the signature-ion
apex (scans above 10% of the reference apex height), returning `None` when there
are fewer than 3 scans or no reference signal. This restriction matters: over the
full wide extraction window the traces are mostly zeros and any correlation is
noise, so the co-elution and spectral gates are only meaningful across the
elution peak itself (`extract.rs:520`). `GateMode` scores:
- `ApexPearson`: Pearson of observed-vs-predicted intensities at the single apex
  scan (`extract.rs:1989`); one chimeric scan can dominate it.
- `PeakSpectral`: `peak_spectral_score` (`extract.rs:564`) correlates the
  peak-summed observed spectrum (each fragment integrated over the peak scans)
  with predicted intensities, averaging out a single interfered scan.
- `SpectralEntropy`: Li spectral-entropy similarity of the sqrt-transformed apex
  spectrum via the shared `features::entropy` kernel (`extract.rs:1995`).
- `Coelution`: `coelution_gate_score` (`extract.rs:590`), the
  predicted-intensity-weighted mean Pearson of each matched fragment's XIC to the
  signature-ion reference profile, restricted to the elution peak. Orthogonal to
  intensity agreement (temporal, not shape).
- `Combined`: requires `peak_spectral >= gate_min_score` **and** `coelution >=
  gate_coelution_min` (`extract.rs:2016`).

All four diagnostic scores are computed for every accepted candidate only when
`emit_gate_diagnostics` is set (`extract.rs:2035`); otherwise they are zeroed and
not written, so the default chain pays no extra cost.

### 6. Chromatograms and MS1 XICs

Per-fragment observed m/z is an intensity-weighted mean per fragment
(`extract.rs:2073`) for mass-accuracy features. Every predicted transition emits a
chromatogram row (`extract.rs:2095`); observed fragments carry the grid-sampled
(or rt-sorted) trace, absent ones an empty trace. When MS1 is present and grid
mode is on, three MS1 isotope XICs (`ms1_mono`, `ms1_iso1`, `ms1_iso2`) are
sampled on the same scan grid via nearest-MS1-scan lookup (`extract.rs:2151`),
using `ISOTOPE_SPACING / charge` (`constants.rs:22`, value `1.003354835`) for the
isotope offsets and `sum_near` (`extract.rs:495`) to integrate within
`prec_tol_ppm`. The apex MS1 isotope intensities (`ms1_isom1`/mono/`iso1`/`iso2`)
are the per-PSM columns, taken from the nearest MS1 scan to the apex RT
(`extract.rs:1957`).

### 7. Top-K peak enumeration (`retain_top_peaks`)

Two independent knobs consume the enumerator, and they must not be confused:
`retain_top_peaks` writes the **unscored** `.peaks.parquet` sidecar, whereas
`promote_top_peaks` emits additional `psms_extracted` rows with `peak_rank >= 1`
that do flow downstream. Both default to 1 (off).

When `retain_top_peaks > 1` and the candidate has groups (`extract.rs:2173`), the
per-scan distinct-fragment count profile (`count_prof`, `extract.rs:2176`) is
passed to `crate::peaks::enumerate_peaks` (`peaks.rs:52`) with `k =
retain_top_peaks`, `bound_fraction = 1/3` and `min_prominence_frac = 0.1`
(`extract.rs:2177`). `enumerate_peaks` is a pure, side-effect-free function
(`peaks.rs:1`):
1. An empty profile, `k == 0`, or an all-non-positive profile returns an empty
   vector (`peaks.rs:58`, `peaks.rs:63`).
2. **Local maxima** (`peaks.rs:71`): index `i` qualifies when its value is `>
   0` and `>= prom_floor` (the prominence floor `min_prominence_frac *
   global_max`, `peaks.rs:66`, tested at `peaks.rs:74`), is **strictly** greater than its left neighbour
   (or is the left edge), and is `>=` its right neighbour (or is the right edge).
   The strict-left / non-strict-right rule makes a flat plateau register once, at
   its left edge; edges count as maxima against their single neighbour.
3. **Boundary walk** (`peaks.rs:87`): from each apex, walk left and right,
   stopping when the profile drops below `bound_fraction * apex` **or** turns back
   upward (a valley); `area` is the integrated profile over `[start_idx,
   end_idx]`.
4. **Dedup + rank** (`peaks.rs:123`): peaks are sorted strongest-first by `area`,
   then by `apex_intensity`, then by earliest `apex_idx` (`peaks.rs:125`); a
   maximum whose apex falls inside an already-kept peak's `[start_idx, end_idx]`
   envelope is dropped (`peaks.rs:140`); the first `k` survivors are kept and
   assigned `rank` 0..k-1 (`peaks.rs:148`).

`enumerate_peaks(.., k=1, ..)` returns the single global-argmax peak, so callers
can adopt it incrementally. The retained peaks rank by **co-eluting fragment
breadth (count), not intensity**, per finding A6 in
docs/18_findings_and_decisions.md (intensity is chimeric in DIA). Back in
`extract`, each returned `PeakGroup` is mapped to a sidecar row
(`extract.rs:2179`): `apex_rt`/`start_rt`/`end_rt` are the RTs of the
apex/start/end scan groups, `evidence_count` is the distinct-fragment count at
the apex group (`groups[pk.apex_idx].1.len()`), and `area` is `pk.area`. At the
default `promote_top_peaks = 1` the PSM table still reports only the single
selected apex, so FDR is unaffected and the sidecar peaks are unscored candidate
peaks for an offline peak-selection model (`extract.rs:1700`).

`promote_top_peaks > 1` (`extract.rs:2238`) is the separate, opt-in path that does
change FDR: it re-enumerates the same count profile (`extract.rs:2252`), filters
alternates by `alt_peak_min_area_frac` and `alt_peak_min_separation_s` plus the
`presence_min_matched` floor (`extract.rs:2292`), and pushes up to
`promote_top_peaks - 1` extra `CandOut` rows with `peak_rank >= 1`
(`extract.rs:2297`). Because these become real scored rows, enabling it changes
the training/FDR population and needs entrapment validation, not a count check.

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `run` | `extract.rs:1300` | Stage entry point; orchestrates load, accumulate, cascade, write |
| `ExtractParams` | `extract.rs:83` | Input path bundle + config + config hash |
| `Hit` | `extract.rs:108` | One observed hit: `rt`, `frag`, `inten`, `obs_mz` |
| `Contested` | `extract.rs:130` | Per-candidate two-pass contested-peak stats: `won`/`lost` intensity, `n_won`/`n_lost` peak counts, `apportioned` share |
| `CandOut` | `extract.rs:1656` | Per-candidate parallel-map result (PSM row + chrom rows + peaks); one candidate may yield several |
| `Prober` / `Prober::probe` | `extract.rs:38`, `extract.rs:57` | Dispatch a peak probe to fragindex or bucketed backend |
| `MassOffset::factor_at` | `extract.rs:640` | Scalar or m/z-interpolated ppm divisor for observed m/z |
| `claim_cue_multiplier` | `extract.rs:163` | Composite per-claimant cue weight for `CoelutionMultiCue` |
| `demix_apex_scan` / `demix_solve_scan` | `extract.rs:259`, `extract.rs:279` | Spectrum-centric ridge-NNLS demix at one scan |
| `demix_features_for` | `extract.rs:414` | Per-candidate demix features from a solved scan |
| `extract_accumulate_windows` | `extract.rs:662` | Parallel single-pass per-window accumulation |
| `extract_twopass_windows` | `extract.rs:787` | Parallel two-pass co-elution arbitration + contested stats |
| `peak_window` | `extract.rs:523` | Elution-peak scan range around the signature apex (10% height) |
| `peak_spectral_score` | `extract.rs:564` | Peak-integrated observed-vs-predicted Pearson |
| `coelution_gate_score` | `extract.rs:590` | Weighted mean XIC-to-reference co-elution Pearson |
| `nearest_index` | `extract.rs:139` | Binary search for nearest RT in a sorted array |
| `sum_near` | `extract.rs:495` | Sum intensities within a ppm window (m/z-sorted) |
| `enumerate_peaks` | `peaks.rs:52` | Pure top-K peak-group enumerator |
| `PeakGroup` | `peaks.rs:22` | One enumerated peak (apex/start/end idx, apex intensity, area, rank) |
| `Library::load_with` | `index.rs:73` | Load the SoA library, optionally skipping the bucketed index |
| `Library::candidate_range` | `index.rs:341` | Candidate-id range for an isolation window |
| `Library::cand_frags` | `index.rs:312` | Fragment m/z, predicted intensity, and interned name-id slices |
| `Library::page_search` | `index.rs:350` | Bucketed-backend fragment probe |
| `FragIndex::probe_peak` | `fragindex.rs:152` | Verified postings for one observed peak in a candidate range |

## Configuration

All fields live in `ExtractConfig` (`config.rs:516`); defaults in
`config.rs:683`. The config was pruned of dead fields in the "clean-slate
declutter" commit: `extract.scan_window_mode` (and the `ScanWindowMode` enum),
`extract.scan_scale`, `extract.k_select`, `extract.max_fragment_charge`, and
`extract.tolerance_regime` were removed. Do not reintroduce them.

| Field | Default | Effect |
|---|---|---|
| `fixed_scan_window` | 3 | Minimum co-elution run length (`scan_window` floor); `.max(1)` at `extract.rs:1591` |
| `frag_tol_ppm` | 20.0 | Fragment match tolerance (overridden by `mass_cal`) |
| `prec_tol_ppm` | 20.0 | MS1 isotope integration tolerance |
| `presence_min_matched` | 3 | Tier-b: minimum distinct matched fragments (`extract.rs:1727`) |
| `presence_min_fragments` | 3 | Acceptance: minimum distinct fragments (`extract.rs:1926`) |
| `presence_min_coelution` | 2 | Min simultaneously-present fragments to extend a run (`extract.rs:1915`) |
| `gate_min_score` | **0.6** | Pearson/gate threshold; 0 disables the gate. 0.6 is the documented optimum for the default `native_tda` rescorer; the 0.2 that suits `nn_torch` is set explicitly by the shipped example configs. Renamed from `min_frag_corr`, which is not a correlation under any `gate_mode` except by coincidence |
| `min_matched_fraction` | 0.0 | Acceptance: min matched/predicted fraction (default off) |
| `apex_top_fragments` | 0 | Signature-ion count for apex; 0 -> default 3 (`extract.rs:1860`). Config marks it superseded by `apex_count_tol`, kept for compat (`config.rs:545`) |
| `apex_rt_prior_s` | 0.0 | Gaussian RT-prior sigma on apex tiebreak; 0 = off |
| `apex_count_tol` | 1 | Count slack for qualifying apex scans |
| `apex_count_window` | 1 | Rolling-sum width for the count profile; 1 = no smoothing. Window 5 cut AIF apex misassignment (median \|dRT\| 131s -> 9s) |
| `apex_gaussian_sigma_scans` | 0.0 | Opt-in Gaussian apex smoother; 0 = off (`config.rs:572`) |
| `apex_evidence_rank` | **true** | Breadth-of-evidence apex. The legacy signature-intensity apex silently falls back to the lowest-RT qualifying scan when none of the top-K predicted fragments is observed |
| `emit_window_grid` | true | Zero-filled full-window-grid chromatograms |
| `bucket_size` | 8192 | m/z bucket size (power of two) |
| `peak_claim` | `None` | Shared-peak apportionment strategy (`PeakClaim`, `config.rs:132`) |
| `claim_cues` | all off | Composable cue weights for `CoelutionMultiCue` (`ClaimCues`, `config.rs:192`); `claim_cues.reassign` is what makes MultiCue destructive |
| `peak_claim_margin` | 2.0 | Dominance factor for `CoelutionWinnerMargin` |
| `emit_demix_features` | false | Adds the five `deconv_*`/`shadow_*` columns; forces the apex demix solve (`config.rs:592`) |
| `demix_lambda` | 1.0 | Ridge term of the NNLS demix (`config.rs:595`) |
| `demix_max_candidates` | 64 | Column cap of the per-scan demix design matrix (`config.rs:598`) |
| `demix_scan_stride` | 1 | Solve the demix every Nth scan (`config.rs:606`) |
| `emit_contested_features` | false | Adds `contested_count_frac`/`apportioned_frac`; forces the two-pass path |
| `matcher` | `Fragindex` | Fragment-matcher backend |
| `min_coelution_run` | 0 | Extra co-elution-run floor (0 = off; `scan_window` still applies) |
| `ms1_rescue` | false | Rescue Pearson-gate failures with MS1 isotope support |
| `retain_top_peaks` | 1 | K>1 writes the `.peaks.parquet` sidecar (unscored) |
| `promote_top_peaks` | 1 | K>1 emits extra `psms_extracted` rows with `peak_rank >= 1`; **does** change FDR (`config.rs:645`) |
| `alt_peak_min_area_frac` | 0.10 | Alternate peak must reach 10% of the rank-0 area (`config.rs:649`) |
| `alt_peak_min_separation_s` | 5.0 | Alternate apex must sit >= 5 s from the rank-0 apex (`config.rs:653`) |
| `emit_candidate_audit` | false | Candidate-audit sidecar (diagnostic) |
| `emit_gate_diagnostics` | false | Adds the four `gate_*` diagnostic columns |
| `gate_mode` | `ApexPearson` | Which score `gate_min_score` thresholds (`GateMode`, `config.rs:735`) |
| `gate_coelution_min` | 0.5 | Second threshold for `GateMode::Combined` |

Note: `emit_candidate_audit` is a declared knob but the candidate-audit write is
not present in this `extract.rs`; the audit ladder is produced by the separate
`mumdia audit` command. Treat the in-stage audit as unwired here.

### Default-off sensitivity knobs (index)

Every knob below is default-off, so the production PSM schema and per-candidate
compute stay byte-identical unless the knob is set. None has an end-to-end
identification-gain measurement yet; each must pass the entrapment-holdout gate
on >=2 datasets before being enabled by default (see the benchmark-gated list in
CLAUDE.md and docs/20_sensitivity_and_quantification_playbook.md). Two of the
knobs live in other stages' configs (`search_seed`, `rt_im_train`) but are listed
here for one index.

| knob | extra columns / artifact | validation status |
|---|---|---|
| `extract.retain_top_peaks` (`config.rs:635`, default 1) | `<out_psms>.peaks.parquet` sidecar, unscored, written only when K>1 (`extract.rs:2544`); no PSM columns | ID loop not closed (sidecar peaks are unscored); gate pending |
| `extract.promote_top_peaks` (`config.rs:645`, default 1) | extra `psms_extracted` rows with `peak_rank >= 1` (`extract.rs:2297`); no new columns | **not schema-neutral in rows**: changes the scored population and therefore FDR; gate pending |
| `extract.emit_candidate_audit` (`config.rs:658`) | none in `extract.rs` (unused there); in `run` it gates the `audit` stage -> `candidate_audit.parquet` (`run.rs:428`) | diagnostic; no ID effect |
| `extract.emit_gate_diagnostics` (`config.rs:673`) | 4 F32 cols `gate_apex`, `gate_peak_spectral`, `gate_coelution`, `gate_spectral_entropy` (`extract.rs:2511`) | diagnostic; no ID effect |
| `extract.apex_evidence_rank` (`config.rs:667`) | none; changes the apex-selection score (`extract.rs:1890`) | **now the default**, promoted on correctness grounds rather than a count (docs/25 section 2): the legacy score cannot distinguish "no signature fragment observed anywhere" from "the earliest qualifying scan is the apex" |
| `extract.apex_gaussian_sigma_scans` (`config.rs:572`) | none; smooths the apex count profile | not measured; gate pending |
| `search_seed.two_pass_mass_cal` (`config.rs:423`) | none; refits the `<seed>.masscal.json` offset + tolerance | not measured; gate pending |
| `rt_im_train.adaptive_rt_window` (`config.rs:487`) | none; per-region RT half-window widths in `run_windows` | not measured; gate pending |
| `extract.emit_contested_features` (`config.rs:611`) | 2 F64 cols `contested_count_frac`, `apportioned_frac` (`extract.rs:2505`); forces the two-pass path (`extract.rs:1443`) | not measured; gate pending |
| `extract.emit_demix_features` (`config.rs:592`) | 5 F32 cols `deconv_explained_frac`, `deconv_active`, `deconv_share`, `deconv_max_collinearity`, `shadow_kept_frac` (`extract.rs:2517`) | not measured; gate pending |
| `extract.peak_claim` = `CoelutionMultiCue` / `CoelutionDemix` / `CoelutionShadow` (`config.rs:132`) | none; MultiCue is non-destructive unless `claim_cues.reassign`, the other two always rewrite extracted intensity | not measured; gate pending |
| `compete.mode` (`CompetitionMode`, `config.rs:840`, default `winner_take_all`) | retains more rows in `competed`; optional `<out>.compete_audit.parquet` via `emit_competition_audit` (`config.rs:850`) | not measured; gate pending |

## Invariants, determinism, gotchas

- **Determinism**: output is emitted in ascending `candidate_id` order
  (`extract.rs:1651`); the parallel per-candidate map preserves that order via
  `collect()`. Per-scan fragment maps are `BTreeMap` so f32 apex sums have a fixed
  addition order (`extract.rs:1735`). The parallel window accumulation is documented
  as bit-identical to the serial loop (`extract.rs:1456`). A
  HashMap f32 sum shifting the apex once broke reproducibility; keep ordered maps
  and sorted iteration wherever floats are summed.
- **Default-off contract**: `retain_top_peaks=1`, `promote_top_peaks=1`,
  `apex_evidence_rank=false`, `emit_contested_features=false`,
  `emit_demix_features=false`, `emit_gate_diagnostics=false`, `peak_claim=None`
  make the schema and per-candidate compute byte-identical to the production chain.
  Every sensitivity knob added here must keep that property. Note that
  `promote_top_peaks` is schema-neutral but **not** row-neutral, so it is the one
  knob in this list that alters the FDR population when enabled.
- **The gate optimum depends on the rescorer**, not the gate in isolation. The
  full-feature search found `spectral_entropy_similarity_sqrt` the single best
  target/decoy discriminator (AUC 0.826), yet gating on it *regressed*
  end-to-end identifications versus the apex gate, because gating on the
  rescorer's own best feature enriches hard decoys. "Best discriminator" is the
  wrong criterion for a gate; the lever is the rescorer. This is why every gate
  change must pass an entrapment-holdout gate before being enabled by default.
- **The MS2 peak budget is set in `convert`, not here.** `convert.rs:76-79` is
  the only MS2 peak truncation in the chain; extract reads whatever
  `spectra_ms2.parquet` contains. The two peak limits in the pipeline are
  independent: on a 50-window Orbitrap DIA run, moving `--top-peaks-ms2` from
  300 to uncapped left seed output identical (80,474 seed PSMs) while moving
  extract's accepted-candidate count from 188,027 to 2,286,840 (12.2x) and
  extract wall clock from 57.9 s to 91.9 s. Uncapping is cheap at this stage
  relative to the identification difference it produces, but the accepted-row
  count it feeds into `features` and `compete` grows by an order of magnitude,
  so size the downstream stages accordingly.
- **Mass recal is a divisor, not a subtraction**: observed m/z is corrected by
  `q_mz = peak.mz / factor_at(peak.mz)` where `factor_at` returns
  `1 + ppm(mz)*1e-6` (`extract.rs:640`, applied at `extract.rs:705`); a
  missing/absent `masscal.json` yields offset 0 and the config tolerance. The ppm
  term may be m/z-dependent, so do not assume one constant offset per run.
- **Empty vs zero traces**: a never-observed predicted fragment carries an empty
  trace, not a grid-length zero vector (`extract.rs:2098`). Downstream code must
  treat an empty trace as `obs_apex = 0`.
- **`apex_im` is always null** (3D MVP); do not assume an IM value.
- **`restrict_candidates` only forces the serial path in the non-two-pass case.**
  In the non-two-pass branch the parallel window accumulation is used only when a
  fragindex is present **and** there is no `restrict` list (`extract.rs:1455`); a
  `restrict` list therefore routes to the slower serial single-pass loop, which
  honors the allowlist and every non-co-elution `peak_claim` strategy. In the
  two-pass branch (`Coelution*` or `emit_contested_features`) extraction stays on
  the parallel `extract_twopass_windows` path regardless of `restrict`, which
  applies the allowlist inside each pass's push closure (`extract.rs:853`,
  `extract.rs:907`, wired at `extract.rs:1578`).
- **`peaks.parquet` is not scored** and never enters FDR; it is a research
  sidecar. The `peak_rank` **column** of `psms_extracted` is a different thing:
  rank-0 rows are the ordinary selected apex, and ranks >= 1 appear only under
  `promote_top_peaks > 1`, where they are scored and do enter FDR.
- Chromatogram list columns are `LargeListF32` on purpose (`extract.rs:2534`); do
  not downgrade to 32-bit `ListF32` or wide-open gates overflow the offset buffer.

## Tests

Only the pure gate-scoring helpers, the mass-offset interpolator, and the peak
enumerator are unit-tested; there is no stage-level test of `run` itself. The
tests encode the behavioral invariants that gate tuning must preserve.

- `mass_offset_tests` (`extract.rs:2604`): scalar and m/z-grid ppm interpolation
  agree at the anchors and clamp outside them.
- `coelution_tests` (`extract.rs:2630`): co-eluting fragments score `> 0.95`
  (`extract.rs:2651`); a strong non-co-eluting interferent drops the co-elution
  score (`extract.rs:2665`); fewer than 3 scan groups returns `1.0`
  (do-not-reject) rather than a low score (`extract.rs:2674`); `peak_spectral`
  scores `> 0.99` when the peak-integrated pattern matches predicted
  (`extract.rs:2688`) and still recovers a fragment that is momentarily unsampled
  at the apex scan by integrating over the peak (`extract.rs:2705`, the DIA
  scan-gap case the single-scan apex Pearson fails).
- `peaks::tests` (`peaks.rs:154`): empty/all-zero profiles yield no peaks; a
  clean triangular peak resolves apex and 1/3-height boundaries; `k=1` keeps only
  the strongest-area peak; a dominant interference peak with `k=1` discards a
  weaker true peak but `k>=2` retains it (the core sensitivity behavior,
  `peaks.rs:190`); two maxima rank by area; a left-edge-truncated peak apexes at
  index 0; the prominence filter suppresses a noise bump; a shoulder inside a
  stronger envelope collapses into it; and repeated calls are bit-identical with
  ties broken by earliest apex (`peaks.rs:260`).

## How to extend / modify

- **A new gate metric**: add a variant to `GateMode` (`config.rs:735`), add a lazy
  score closure next to `apex_pearson`/`peak_spec`/`coel` (`extract.rs:1989`), and
  a match arm in the acceptance gate (`extract.rs:2010`). If it is worth
  diagnosing, also wire it into the `emit_gate_diagnostics` tuple
  (`extract.rs:2035`) and the conditional column block (`extract.rs:2511`). Default
  the gate off and validate against entrapment before enabling.
- **A new peak-claim strategy**: add a `PeakClaim` variant (`config.rs:132`); if it
  needs elution profiles, extend the two-pass trigger (`extract.rs:1443`), the
  `reassign` predicate (`extract.rs:1557`), and the reassignment match in
  `extract_twopass_windows` (`extract.rs:1236`); otherwise add it to the serial
  single-pass match (`extract.rs:1500`) and the parallel accumulation
  (`extract.rs:718`). Keep tie-breaks deterministic (lowest `candidate_id`). Prefer
  a cue in `ClaimCues` (`config.rs:192`) over a new variant when the change is
  really a new per-claimant weight: `claim_cue_multiplier` (`extract.rs:163`)
  already composes them and defaults each to 1.0.
- **Scoring the top-K peaks**: the `retain_top_peaks` sidecar peaks are unscored.
  `promote_top_peaks` is the partial closure of that loop: it emits alternate peaks
  as real `psms_extracted` rows carrying `peak_rank`, so they pick up the full
  feature vector downstream. What is still missing is per-candidate q-collapse over
  `(candidate_id, peak_rank)` and entrapment validation, so it stays default-off.
- **New per-PSM columns**: append to the `CandOut` struct (`extract.rs:1656`), set
  it in the `rank0` construction (`extract.rs:2203`) and in the promoted-alternate
  construction (`extract.rs:2297`), push it in the serial append loop
  (`extract.rs:2435`), and add the `Col` in the `psms_cols` vector
  (`extract.rs:2480`). Gate any non-production column behind an `emit_*` flag to
  preserve the byte-identical default schema, and bump the schema version in
  `schema.rs` if the default schema changes.
- **IM / 4D**: `apex_im` and the IM data-model hooks exist but are unfilled; a
  diaPASEF extension adds an IM window post-filter alongside the RT window and IM
  apex/feature families. It cannot be validated without diaPASEF data.
