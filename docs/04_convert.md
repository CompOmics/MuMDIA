# convert (Stage 0): mzML to normalized spectra
> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

`convert` is Stage 0 of the pipeline. It is the single point in the engine that
touches the vendor mass-spectrometry format. It reads one mzML run through the
`mzdata` crate and writes a normalized, self-describing spectra artifact set (four
Parquet files) that every downstream stage consumes instead of the raw file.
Everything after this stage (`search-seed`, `rt-im-train`, `extract`, `features`,
`quant`) reads spectra only through
`crates/mumdia/src/spectra.rs`, never through `mzdata`. Consequences: the
vendor-format dependency is isolated here, and any format quirk (profile vs
centroid, AIF/all-ion windows, missing precursor) must be resolved at this stage
because later stages assume the normalized shape.

The MVP is mzML-only and 3D. Ion-mobility columns are therefore absent from the
artifacts. convert writes no IM columns at all; the in-memory `Peak`/
`IsolationWindow` types carry `Option` IM fields (`Peak.ion_mobility` at
`crates/mumdia-core/src/types.rs:12`; `IsolationWindow.im_lower`/`im_upper` at
`:21`/`:22`), and the read side (`spectra.rs`) fills them with `None`
(`spectra.rs:74`, `:87-88`).

## Files

| Path | Role |
|---|---|
| `rust/mumdia/crates/mumdia/src/stages/convert.rs` | The whole stage: mzML reading, centroiding, peak capping, window synthesis, artifact writing. |
| `rust/mumdia/crates/mumdia/src/main.rs` (`Cmd::Convert` flags at lines 28-47, dispatch at 430-454) | CLI subcommand; builds the provenance `config_hash` and calls `convert::run`. |
| `rust/mumdia/crates/mumdia/src/stages/run.rs` (lines 201-215) | The `run` orchestrator's call into convert (top_peaks_ms1 hardcoded to 0). |
| `rust/mumdia/crates/mumdia/src/spectra.rs` | The read-back side: `load_ms1` / `load_ms2` turn the artifacts back into in-memory scans for downstream stages. |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` (lines 7-10) | Frozen `(logical name, version)` identifiers for the four output artifacts. |
| `rust/mumdia/crates/mumdia-io/src/table.rs` | `Col` / `write_table` typed Parquet writer used to emit the artifacts. |
| `rust/mumdia/crates/mumdia-io/src/report.rs` | `ArtifactReport`, the `<artifact>.report.json` sidecar written per output. |
| `rust/mumdia/crates/mumdia-core/src/types.rs` | `Peak`, `Ms2Scan`, `IsolationWindow` (used on the read-back side). |

## Inputs and outputs

Input: one mzML file path (`--mzml`). `mzdata` is pinned at version `0.65` with
`default-features = false, features = ["mzml", "miniz_oxide"]`
(`rust/mumdia/Cargo.toml:26`), so only mzML is compiled in; `miniz_oxide` gives
pure-Rust gzip for compressed mzML. No other input is read. The stage takes no
config file (see Configuration).

Outputs: four Parquet artifacts written to `--out-dir`, each with a sibling
`<file>.report.json`. All are written with SNAPPY compression via
`write_table` (`table.rs:166`; the compression is set at `table.rs:205`). List
columns (`mz`, `intensity`) are Arrow
`List<Float32>` (the `Col::ListF32` variant); both the outer list column and its
inner `item` element are marked nullable in the Arrow schema (`table.rs:84`
builds the nullable inner `item` field, `table.rs:98` the nullable list column),
but convert writes neither as null. An empty scan is a non-null empty list
(`ListBuilder::append(true)` at `table.rs:131`).

### `spectra_ms1.parquet` (`SPECTRA_MS1`, schema v1; written at `convert.rs:171`)

| Column | Type | Meaning |
|---|---|---|
| `scan_index` | `u32` | Global monotonic index over all spectra in the run. |
| `rt_seconds` | `f64` | Retention time in seconds (mzdata minutes x 60). |
| `mz` | `List<f32>` | Centroided, m/z-ascending peak m/z (widened to f64 on read). |
| `intensity` | `List<f32>` | Peak intensities, aligned to `mz`. |

### `spectra_ms2.parquet` (`SPECTRA_MS2`, schema v1; written at `convert.rs:198`)

| Column | Type | Meaning |
|---|---|---|
| `scan_index` | `u32` | Global monotonic index (shares the same counter as MS1). |
| `id` | `Utf8` | Native mzML spectrum id string (`spec.id()`), kept for traceability/USI. |
| `rt_seconds` | `f64` | Retention time in seconds. |
| `window_id` | `u32` | Index into `isolation_windows.parquet`, dedup by (lower, upper). |
| `window_target` | `f64` | Isolation window center m/z (0.0 for AIF/all-ion). |
| `window_lower` | `f64` | Isolation window lower bound m/z. |
| `window_upper` | `f64` | Isolation window upper bound m/z (1.0e6 for AIF/all-ion). |
| `precursor_mz` | `f64?` (nullable) | Selected precursor m/z, or null when absent. |
| `precursor_charge` | `i32?` (nullable) | Precursor charge, or null when absent. |
| `mz` | `List<f32>` | Centroided, m/z-ascending fragment m/z. |
| `intensity` | `List<f32>` | Fragment intensities, aligned to `mz`. |

### `isolation_windows.parquet` (`ISOLATION_WINDOWS`, schema v1; written at `convert.rs:215`)

| Column | Type | Meaning |
|---|---|---|
| `window_id` | `u32` | First-seen id (0-based) of a distinct (lower, upper) window. |
| `target` | `f64` | Window center m/z. |
| `lower` | `f64` | Window lower bound m/z. |
| `upper` | `f64` | Window upper bound m/z. |

### `ms2_to_ms1.parquet` (`MS2_TO_MS1`, schema v1; written at `convert.rs:228`)

| Column | Type | Meaning |
|---|---|---|
| `ms2_scan_index` | `u32` | MS2 scan_index. |
| `ms1_scan_index` | `i32` | scan_index of the most recent preceding MS1, or `-1` if none seen yet. |

The `-1` sentinel (not null) is the "no preceding MS1" marker; it occurs for MS2
scans acquired before the first MS1 in the run (`convert.rs:160`).

## How it works

Control flow is `run` (`convert.rs:103-267`), a single linear pass over the mzML
reader plus four table writes.

1. **Open the run.** `mzdata::MZReader::open_path(p.mzml)` (`convert.rs:107`)
   returns an iterator over spectra in acquisition order. The out directory is
   created first (`convert.rs:105`).

2. **Iterate spectra.** The loop zips the reader with an infinite `(0_u32..)`
   range and `.enumerate()` (`convert.rs:120`), so `scan_index` (the range value)
   and `count` (the enumerate index) advance together for every spectrum the
   reader yields, before the MS-level dispatch. If `max_spectra > 0` and `count`
   reached it, break (`convert.rs:121-123`). Because `scan_index` is drawn from the
   range regardless of MS level, it is a run-global monotonic counter. Retention
   time is `spec.start_time() * 60.0` because mzdata returns start time in minutes
   and the artifact stores seconds (`convert.rs:124`).

3. **Dispatch on MS level** (`convert.rs:125`):
   - **MS1** (`convert.rs:126-133`): centroid+cap peaks with `top_peaks_ms1`,
     push into the MS1 accumulators, and record `last_ms1_index = scan_index` so
     subsequent MS2 scans can point back to it.
   - **MS2** (`convert.rs:134-161`): centroid+cap peaks with `top_peaks_ms2`, then
     resolve the isolation window and precursor (details below), and append the
     `(ms2_scan_index, ms1_scan_index)` mapping row.
   - **Any other level** (MS3, etc.): ignored by the `_ => {}` arm
     (`convert.rs:162`), but it still consumed a `scan_index`, so the per-level
     tables have globally unique but non-contiguous indices.

4. **Centroiding** happens inside `peaks_of` (`convert.rs:56`, generic over
   `SpectrumLike`). It first pulls the raw arrays via `spec.raw_arrays()` ->
   `mzs()` (f64) and `intensities()` (f32) (`convert.rs:57-64`). Each access is
   `.map(|c| c.to_vec()).unwrap_or_default()`, so a spectrum whose `raw_arrays()`
   is `None`, or that is missing either the m/z or intensity array, degrades to
   empty vectors and therefore an empty peak list rather than an error. If
   `spec.signal_continuity() == SignalContinuity::Profile`
   (`convert.rs:65`) it calls `centroid` (`convert.rs:19`); already-centroided
   spectra pass through unchanged. `centroid` does simple local-maxima detection
   with 3-point parabolic m/z refinement:
   - If fewer than 3 samples, return the input as-is (`convert.rs:21-23`).
   - Compute a relative noise floor `floor = max_intensity * 1e-4`
     (`convert.rs:24-25`), i.e. 0.01% of the base peak. This threshold is
     hardcoded, not a config field.
   - For each interior sample `i` in `1..n-1` with neighbors `y0, y1, y2`
     (`convert.rs:28-34`): keep it only if `y1 > floor` and it is a local maximum
     under the asymmetric test `y1 >= y0 && y1 > y2` (left inclusive, right
     strict, so a flat-topped pair keeps the left sample once). Otherwise skip.
   - Parabolic apex refinement on m/z (`convert.rs:35-43`): with
     `denom = y0 - 2*y1 + y2`, the sub-sample offset is
     `delta = 0.5 * (y0 - y2) / denom` when `|denom| > 1e-12`, else 0. The local
     m/z spacing is `spacing = (mz[i+1] - mz[i-1]) * 0.5`, and the refined center
     is `cm = mz[i] + delta * spacing`. Only the m/z is refined; the emitted
     intensity is the raw apex sample `y1`, not a parabola-interpolated height
     (`convert.rs:44-45`).
   - If no local maximum survived, `centroid` returns the original profile arrays
     as a fallback (`convert.rs:47-51`). This is a safety net; a pathological
     profile scan can therefore leak raw profile samples downstream.

5. **Filter, cap, sort** (still in `peaks_of`, `convert.rs:70-83`): the m/z and
   intensity vectors are joined with `mz.into_iter().zip(inten)`
   (`convert.rs:71-73`), which stops at the shorter of the two, so a length
   mismatch silently drops the tail of the longer array rather than erroring.
   Drop peaks with intensity `<= 0`. If `top_n > 0` and there are more than
   `top_n` peaks,
   sort descending by intensity and truncate to `top_n` (`convert.rs:76-79`).
   The truncation is baked into `spectra_ms2.parquet` and no later stage can undo
   it: `extract` applies no peak cap of its own and consumes whatever convert
   wrote. Sizing this cap is therefore a per-acquisition decision, not a portable
   default (see "Choosing `--top-peaks-ms2`").
   Then always sort ascending by m/z (`convert.rs:80`). Finally, cast m/z to
   `f32` for output (`*m as f32`, `convert.rs:81`) while intensity stays `f32`.
   Storing observed m/z as f32 halves peak storage; the read side widens back to
   f64 (`spectra.rs:72`, `spectra.rs:137`). At the ppm tolerances used in DIA
   matching, f32 m/z (~7 significant digits) is adequate for observed peaks;
   library/theoretical m/z stay f64.

6. **Isolation window and precursor resolution** (`convert.rs:136-148`): read
   `spec.precursor()` and clone its `isolation_window`. If a real window is
   present (not both bounds zero), use `(target, lower_bound, upper_bound)`
   (`convert.rs:139-141`); mzdata exposes these three window fields as `f32`, and
   each is widened with `as f64` before storage, so the stored window columns are
   f64 even though the source precision is f32. Otherwise, the AIF / all-ion path
   synthesizes a
   full-range window `(target=0.0, lower=0.0, upper=1.0e6)` (`convert.rs:142-143`).
   This `_` arm fires both when the quadrupole reported a zero-width window
   (AIF/all-ion acquisition) and when there is no precursor at all, so any MS2
   with no usable window is treated as covering the entire m/z range. The
   downstream `IsolationWindow::covers` (inclusive on both bounds) then returns
   true for every fragment (`types.rs:27`). The precursor m/z and charge come from
   the first precursor ion
   or are `None` (`convert.rs:145-148`). Window synthesis and precursor extraction
   are independent code paths: a scan that reports a zero-width window but still
   carries a precursor ion gets the synthesized full-range window
   (`window_target = 0.0`) together with a non-null `precursor_mz`/
   `precursor_charge`, so `window_target = 0.0` does not imply a null precursor.

7. **Distinct isolation windows** (`convert.rs:181-196`): a `HashMap` keyed by the
   raw bit patterns of `(window_lower, window_upper)` via `f64::to_bits`
   (`convert.rs:188`) assigns a first-seen `window_id`. The key uses bits, not
   float equality, so identical window bounds always collapse to the same id and
   the id order is the acquisition order of first appearance. Each MS2 row records
   its `window_id` in `win_id_col`.

8. **Write the four tables** (`convert.rs:171-234`) with `write_table`, then
   `write_reports` (`convert.rs:269-290`) emits one `ArtifactReport` per file:
   `logical_name` and `schema_name` both set to the artifact's `schema.0` name
   (`convert.rs:276-277`), `schema_version` = `schema.1`, `stage = "convert"`, row
   count, a blake3 content hash of the written file (`convert.rs:281`), the same
   resolved params for all four (`mzml`, `max_spectra`, `top_peaks_ms2`,
   `top_peaks_ms1`, `config_hash`), and `elapsed_ms` (shared across the four
   reports, measured once for the whole stage). convert leaves the report's `stats`
   empty (`Default::default()`, an empty `BTreeMap`) and `model_identity` `None`
   (`convert.rs:283-284`), since it applies no model and computes no summary
   distributions. The report is written next to the artifact as
   `<artifact>.report.json` (`report.rs:28-31`). The function returns
   `ConvertOutputs` with the four paths for chaining (`convert.rs:261-266`).

Note the artifacts are written in acquisition order. RT-sorting is deferred to the
read side: `spectra::load_ms2` / `load_ms1` sort by `rt_seconds` after loading
(`spectra.rs:95`, `spectra.rs:158`).

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `centroid` | `convert.rs:19` | Local-maxima centroiding with parabolic m/z refinement and a relative noise floor. |
| `peaks_of` | `convert.rs:56` | Profile-detect + centroid, drop non-positive intensity, top-N cap, m/z sort, cast m/z to f32. |
| `ConvertParams` | `convert.rs:86` | Inputs: `mzml`, `out_dir`, `max_spectra`, `top_peaks_ms2`, `top_peaks_ms1`, `config_hash`. |
| `ConvertOutputs` | `convert.rs:96` | Returned paths: `ms1`, `ms2`, `isolation_windows`, `ms2_to_ms1`. |
| `run` | `convert.rs:103` | The stage entry point; single pass over mzML, then four table writes. |
| `write_reports` | `convert.rs:269` | Writes the per-artifact `report.json` sidecars. |
| `artifact::SPECTRA_MS1/_MS2/ISOLATION_WINDOWS/MS2_TO_MS1` | `schema.rs:7-10` | Frozen `(name, version)` schema identifiers, all v1. |
| `Col` / `write_table` | `table.rs:23` / `table.rs:166` | Typed columns and the SNAPPY Parquet writer; rejects duplicate names (`table.rs:172-180`) and unequal column lengths (`table.rs:181-191`). |
| `ArtifactReport` | `report.rs:11` | The report struct written next to each artifact. |
| `load_ms2` / `load_ms1` | `spectra.rs:20` / `spectra.rs:100` | Read-back into `Ms2Scan` / `Ms1Scan`, RT-sorted, m/z widened to f64; per-scan peak count is `mf.len().min(iff.len())` (`spectra.rs:68`), tolerant of an m/z vs intensity length mismatch. |
| `Ms1Scan` / `Ms2Scan` | `spectra.rs:12` / `types.rs:55` | Read-back structs. `Ms1Scan` (scan_index, rt_seconds, mz, intensity) is defined in `spectra.rs`, not `types.rs`; `Ms2Scan` (adds `id`, `window`, `peaks`) is in `types.rs`. |

## Configuration

convert reads no `Config` fields, and its subcommand has no `--config` flag
(`main.rs:28-47`). The stage function signature does not take a `Config`; the
`config_hash` it receives is only recorded for provenance. The CLI wrapper still
loads the default config via `load_config(&None)` (`main.rs:437`) purely to seed
that hash: `config_hash = blake3(cfg.canonical_json() + separators + caps)`
(`main.rs:442-445`), so the default config's canonical JSON is embedded in the
hash even though no config field alters the output.

| CLI flag | Default | Effect |
|---|---|---|
| `--mzml` | (required) | Input mzML path. |
| `--out-dir` | (required) | Output directory for the four artifacts. |
| `--max-spectra` | `0` (all) | Read at most N spectra, for fast iteration. Counts all MS levels. |
| `--top-peaks-ms2` | `0` (uncapped) | Keep at most N most-intense MS2 peaks per scan. Irreversible conversion-time cap; acquisition-specific, see below. |
| `--top-peaks-ms1` | `0` (uncapped) | Keep at most N most-intense MS1 peaks per scan. Irreversible. |

Defaults are asserted by `conversion_caps_default_to_uncapped` (`main.rs:823`) and
the explicit-cap case by `explicit_conversion_cap_is_preserved` (`main.rs:863`).
The `run` orchestrator exposes `--max-spectra` and `--top-peaks-ms2` but not
`--top-peaks-ms1`; it hardcodes `top_peaks_ms1: 0` when calling convert
(`run.rs:213`). The MS2 cap is documented as "irreversible" because it discards
peaks before they reach extraction, features, and quantification
(`main.rs:37-41`).

`--top-peaks-ms2` and `search_seed.top_n_peaks` (`config.rs:410-415`, default
300) are different quantities. The seed limit bounds only how many peaks per
scan the seed probes against the fragment index (`select_peaks`,
`search_seed.rs:342`); it is non-destructive and never touches the spectra
artifact. It is not fully independent of the conversion cap, because the seed
selects its peaks from whatever convert already wrote, so a conversion cap below
`top_n_peaks` also shrinks the seed's input. Above it the two do not interact:
on a 50-window Orbitrap DIA run, changing `--top-peaks-ms2` from 300 to uncapped
left the seed output identical (80,474 seed PSMs, 14,877 confident) because both
arms funnel to the same 300 most intense peaks per scan, while the end-to-end
result changed substantially.

Provenance handling of the caps: because the caps change the spectra output but
are not part of the `Config`, `main.rs:442-445` folds them into the blake3
`config_hash` with a unit-separator (`\u{1f}`) so two different caps do not
collapse to an identical hash. `run` does the same for its own convert call
(`run.rs:201-207`). The same values are recorded in the convert report
`params` (`convert.rs:245-251`).

The centroid noise floor (`1e-4` relative, `convert.rs:25`) and the full-range AIF
window value (`1.0e6`, `convert.rs:143`) are hardcoded constants, not config
fields. `Config` contains no convert-stage field at all and no centroiding
strategy enum. Any change to centroiding or window synthesis is a code change
here, not a config toggle.

### Choosing `--top-peaks-ms2`

The default (`0`, uncapped) is the only portable setting. Any non-zero value
depends on how many peaks the acquisition actually produces per MS2 spectrum, so
it must be measured per acquisition scheme rather than carried over from another
run.

Uncapped peak census on one 50-window Orbitrap DIA run (32,950 MS2 spectra, 660
MS1, 43.4M MS2 peaks in total):

| statistic | peaks per MS2 spectrum |
|---|---|
| p25 | 572 |
| p50 | 1320 |
| p95 | 2756 |
| max | 3596 |

At `--top-peaks-ms2 300` on that run, 9.3M of the 43.4M peaks survive: 78.6% of
all MS2 peaks are discarded and 85.5% of spectra are truncated, since even the
p25 spectrum exceeds the cap. On the chimeric AIF benchmark run where 300 was
originally chosen, only 47.8% of spectra reach the cap. The same value therefore
behaves very differently on the two acquisitions.

End-to-end effect on the 50-window run, with only this flag changed, so both
arms are counted on the same row and q-value unit:

| setting | peptides.tsv rows at `peptide_q_value` <= 0.01 | protein groups | empirical decoy fraction |
|---|---|---|---|
| `--top-peaks-ms2 300` | 25,425 | 4,554 | 0.99% |
| uncapped | 63,237 | 7,336 | 0.99% |

The empirical decoy fraction is the same in both arms, so the difference is
sensitivity and not a loosened threshold. For scale, the two counts are 32.3%
and 80.3% of what a library-free DIA-NN 2.2.0 search reports on the same file.
The mechanism is downstream: with most peaks gone, candidates cannot assemble
enough distinct matched fragments to satisfy `extract.presence_min_fragments`
(`config.rs:523`, default 3 at `config.rs:690`), so they fail peak-group
formation and are recorded with rejection code `NO_PEAK_GROUP`
(`rejection.rs:62`). This was confirmed with `mumdia audit` on the capped arm,
restricted to the peptides that same DIA-NN search reports as present (78,782
distinct `Stripped.Sequence` at DIA-NN `Q.Value` <= 0.01): 49,105 of 78,782 (62.3%) stopped at `candidate_generated` with `NO_PEAK_GROUP`, against only
5,380 lost to FDR and 355 to competition, and a counterfactual replay on the
uncapped artifact recovered 41,948 (85.4%) of them. The loss is therefore
extraction-side, not a scoring or competition effect. See docs/09_extract.md for
that side of the interaction.

Dose-response on the same run, expressed as the fraction of the peptides lost at
cap 300 that a larger cap recovers:

| `--top-peaks-ms2` | lost peptides recovered |
|---|---|
| 300 | 0% |
| 400 | 17.8% |
| 600 | 44.3% |
| 900 | 69.0% |
| 1400 | 83.1% |
| 2000 | 86.8% |
| uncapped | 87.2% |

The curve saturates well below the uncapped peak volume. A cap of 1400 buys 83
of the 87 achievable points at about 77% of the peak volume, which is the
cheaper setting when storage or memory is the binding constraint. Uncapping is
not free at extraction time: on that run accepted candidates went from 188,027
to 2,286,840 (12.2x) and extract wall clock from 57.9 s to 91.9 s.

## Invariants, determinism, gotchas

- **Every externally supplied float is checked here, retention time included.**
  `peaks_of` drops peaks whose m/z or intensity is not finite (and any intensity
  <= 0), and the spectrum loop drops spectra whose scan start time is not finite.
  Both report a count, and the retention-time drop names the first offending scan.

  Retention time was the gap, and it was reproduced rather than reasoned about:
  editing one `scan start time` value in the fixture mzML to `NaN` passed convert
  with no warning, was written into `spectra_ms2.parquet` as `nan`, and then
  aborted `mumdia run` inside extract with `called `Option::unwrap()` on a `None`
  value`, naming neither the file, nor the scan, nor the value. A spectrum with no
  retention time cannot be placed in a chromatogram, so dropping it loses nothing;
  leaving it in propagated NaN into every retention-time window. `ci/smoke.sh`
  step 4b is the regression test, and it also asserts that dropping the scan does
  not change which peptides are identified.

  The backstop for a file whose retention times are ALL unusable is the existing
  "yielded no MS2 spectra" bail, which turns an empty result into an error rather
  than a silent zero-identification run.
- **Determinism.** The output is a deterministic function of the input file and
  the CLI caps. `scan_index` is assigned in reader order; `window_id` is assigned
  in first-appearance order via a bit-keyed map, not float equality. The
  intensity-descending truncation sort (`convert.rs:77`) uses Rust's stable
  `sort_by`, so ties preserve the incoming (m/z-ascending) order; the final
  m/z-ascending sort (`convert.rs:80`) fixes output order regardless.
- **`scan_index` is run-global, not per-level.** MS3 and other unhandled levels
  still advance the range-derived counter (`convert.rs:120`), so the MS1 and MS2 tables
  have unique but non-contiguous indices. Do not assume contiguity; use
  `ms2_to_ms1.parquet` to relate MS2 to its parent MS1.
- **AIF and no-precursor collapse to the same full-range window.** The `_` arm at
  `convert.rs:143` handles both a zero-width reported window (AIF/all-ion) and a
  missing precursor. If a future non-AIF format reports a genuinely absent window,
  it will be silently treated as full-range. `window_target = 0.0` is the marker
  for a synthesized window.
- **`--top-peaks-ms2` is destructive and acquisition-specific.** The truncation
  at `convert.rs:76-79` is written into `spectra_ms2.parquet`, and `extract`
  applies no peak cap of its own, so this flag sets the MS2 peak budget for the
  whole chain. A value tuned on one acquisition scheme can discard the majority
  of another run's peaks (78.6% of MS2 peaks on a 50-window Orbitrap DIA run at
  cap 300) and cost more than half the identifications at an unchanged decoy
  fraction. Measure the per-spectrum peak census before setting it; see
  "Choosing `--top-peaks-ms2`".
- **Observed m/z is f32 on disk.** `convert.rs:81` casts to f32; `spectra.rs`
  widens back to f64. This is intentional (storage) and fine at DIA ppm
  tolerances, but do not round-trip observed m/z through convert expecting f64
  precision.
- **Intensity is the raw apex sample.** Parabolic refinement adjusts m/z only; the
  reported intensity is `y1` (`convert.rs:45`), not an interpolated peak height.
- **Centroid fallback can leak profile samples.** If no local maximum clears the
  floor, `centroid` returns the original profile arrays (`convert.rs:47-51`).
  Rare, but a downstream stage could then see profile-shaped data for that scan.
- **List columns are nullable in the Arrow schema but never null in practice.**
  An empty scan is written as a non-null empty list; the read side treats null and
  empty identically (`spectra.rs:55`, `table.rs:557-559` and `table.rs:565-567`).
- **`partial_cmp().unwrap()` on sorts** (`convert.rs:77`, `:80`,
  `spectra.rs` sorts) would panic on NaN. Convert filters intensity `<= 0` before
  sorting and does not sort on m/z NaN in practice, so this is safe for real mzML
  but is a latent trap if malformed data ever reaches it.
- **Out-dir creation errors are swallowed.** `std::fs::create_dir_all(p.out_dir)
  .ok()` (`convert.rs:105`) discards a creation failure; `write_table` retries the
  parent-directory creation and also discards the result (`table.rs:199-201`), so a
  genuinely unwritable out-dir does not fail at either point and surfaces only as
  the `File::create` error (`table.rs:203`).
- **Observability.** The stage is otherwise side-effect-free apart from its file
  writes; it emits two `tracing::info!` records, one on open
  (`convert.rs:106`) and one on completion carrying the MS1/MS2/window counts and
  `elapsed_ms` (`convert.rs:254-260`).
- **`elapsed_ms` is shared, not per-artifact.** A single `Instant` started at
  `convert.rs:104` times the whole stage; the same value is written into all four
  `report.json` sidecars, so per-artifact timing cannot be read from them.
- **Test coverage.** Only the two CLI-parsing tests above exercise this area.
  `convert.rs` has no `#[cfg(test)]` module at all, so the centroiding math,
  window synthesis, and artifact writing have no stage-level unit test. MS1
  extraction and mass-calibration paths that depend on convert output are
  exercised only in full runs.

## How to extend / modify

- **Add a vendor format** (Thermo `.raw`, Bruker `.d`/TDF): this stage is the only
  place to touch. Either extend `mzdata` features or add a reader that yields the
  same per-spectrum interface, and keep the four output schemas byte-compatible so
  no downstream stage changes. Convert must stay the sole vendor-format touch
  point.
- **Ion mobility / 4D (diaPASEF).** The artifact schemas here are 3D. Adding IM
  means new nullable columns on `spectra_ms2` (and the isolation-window IM bounds
  already modeled as `Option` in `types.rs:21`), plus populating `Peak.ion_mobility`
  on the read side. Bump the affected schema versions in `schema.rs` when columns
  change, since the version guards downstream model/schema matching.
- **Change centroiding.** Edit `centroid` (`convert.rs:19`). If the choice should
  be user-selectable, add a config field and strategy enum in `mumdia-core`
  (per the project convention that every algorithmic choice is a typed config
  field) rather than a second hardcoded branch, and thread it through
  `ConvertParams`. Remember the noise floor and parabolic step are currently
  hardcoded.
- **Change the AIF window sentinel.** The full-range bound `1.0e6`
  (`convert.rs:143`) and `window_target = 0.0` marker are relied on by extraction's
  `IsolationWindow::covers`. Changing either requires auditing the extract stage.
- **Add an artifact column.** Add the column to the relevant `write_table` call,
  add a matching getter/reader in `spectra.rs`, and bump the schema version in
  `schema.rs`. `write_table` rejects duplicate names and mismatched lengths, so
  every new column vector must match the row count.
- **Preserve provenance semantics.** Any new conversion-time parameter that
  changes the output but is not part of `Config` must be folded into the
  `config_hash` key in `main.rs` (as the caps are), or two different settings will
  produce artifacts with an identical hash.
