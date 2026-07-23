# MuMDIA developer documentation

This directory is the deep code and workflow reference for the MuMDIA engine. It
describes every subsystem as it exists in the current tree: the crates and their
responsibilities, each pipeline stage and the artifacts it reads and writes, the
config and data model that tie them together, the Python sidecars, and the build,
test, and deployment machinery. It is written for a developer or agent taking
over the codebase who needs to understand not just what each part does but why it
is built that way. Read `CLAUDE.md` at the repository root first for quick
orientation (layout, build commands, implementation-status table). For the
validated findings and the interstage, determinism, and sidecar contracts, read
`docs/18_findings_and_decisions.md`, which is self-contained and depends on no
gitignored file. `plan.md` (gitignored, local-only) holds the deeper algorithmic
specification but is not required to use these docs. These docs sit between
`CLAUDE.md` and the code: more detail than `CLAUDE.md`, grounded in the actual
source rather than the spec.

## Recommended reading order

1. Start with **01 (overview and dataflow)** for the pipeline as a graph and the
   two library sources.
2. Read **02 (config and data model)** and **03 (IO layer)** next, since every
   stage depends on the shared config, mass model, and Parquet contract.
3. Then walk the stages in pipeline order: **04 -> 05 -> 06 -> 07 -> 08 -> 09 ->
   10 -> 11**, which follows a run from mzML input to FDR-controlled output.
4. Read **12 (quant and cross-run stages)** for the tail subcommands and
   experiment-level stages.
5. Read **13 (sidecars)** when you need the real ML predictors and rescorers.
6. Read **14 (build, test, deploy, gotchas)** before touching the build, the test
   suite, or the release machinery.
7. Keep the reference docs to hand: **15 (Parquet data dictionary)** for the
   column schema of every artifact, **16 (glossary)** for domain and codebase
   terms, and **17 (troubleshooting)** for symptom -> cause -> fix lookups.
8. Read **18 (findings, decisions, and contracts)** for the self-contained
   restatement of the validated findings, the interstage/determinism/sidecar
   contracts, the current best workflow, and the ranked roadmap. It depends on no
   gitignored spec file and stands alone.
9. Read **19 (getting started)** for the reproducible local setup and two
   copy-pasteable end-to-end runs (native and best-sensitivity library).

## Document index

| Doc | Description |
|---|---|
| [01_overview_and_dataflow.md](01_overview_and_dataflow.md) | The pipeline as a stage-and-artifact graph, the two library sources, the `run` orchestrator, and `manifest.json`. |
| [02_config_and_data_model.md](02_config_and_data_model.md) | The `mumdia-core` crate: typed config with per-stage overrides and strategy enums, mass model, constants, schema, and run manifest. |
| [03_io_layer.md](03_io_layer.md) | The `mumdia-io` crate: `Col`/`Table` over Arrow+Parquet, SNAPPY read/write, blake3 hashing, per-artifact `report.json`, and `inspect`. |
| [04_convert.md](04_convert.md) | Stage 0: mzML read through `mzdata`, profile centroiding, AIF full-range window fallback, and the normalized spectra artifacts. |
| [05_digest_peptidoforms.md](05_digest_peptidoforms.md) | Stage A and A2: fully-tryptic in-silico digest with reverse/scramble decoys, then expansion into concrete peptidoforms with mods and charges. |
| [06_predict_frag_index_matchers.md](06_predict_frag_index_matchers.md) | Stage C: the run-independent library (b/y m/z, intensities, iRT), the peak-major inverted fragment index, and the fragment matchers. |
| [07_search_seed.md](07_search_seed.md) | Stage S: the native Sage-lite broad DIA search that produces calibration anchors and per-run mass recalibration, not final IDs. |
| [08_rt_im_train.md](08_rt_im_train.md) | Stage B: per-run LOESS/linear RT calibration, residual-percentile RT windows, and the optional DeepLC multitask fine-tune. |
| [09_extract.md](09_extract.md) | Stage D: the core peak-major targeted extraction cascade, apex selection, chromatograms, MS1 isotope XICs, and the sensitivity knobs. |
| [10_features.md](10_features.md) | Stage E: the minimal/rich/extended feature battery (~381 features), `prelim_score`, PIN emission, and the hashed feature schema. |
| [11_compete_rescore_fdr.md](11_compete_rescore_fdr.md) | Stage F: within-group competition, semi-supervised rescoring (native / mokapot / PyTorch-NN / percolator / entrapment), and target-decoy q-values at PSM/run/precursor/peptide/protein level. |
| [12_quant_lfq_align_mbr_report_audit.md](12_quant_lfq_align_mbr_report_audit.md) | The tail subcommands: `quant`, `quant-lfq` (MaxLFQ/directLFQ), `align`, `mbr`, `report`, and `audit`. |
| [13_sidecars.md](13_sidecars.md) | The 10 Python sidecar workers (MS2PIP/DeepLC/mokapot/entrapment/diagnostics), the positional-CLI file contract, and the conda envs. |
| [14_build_test_deploy_gotchas.md](14_build_test_deploy_gotchas.md) | The Rust workspace build, test coverage and gaps, the determinism contract, the clean-room boundary, and CI/Docker/release. |
| [15_data_dictionary.md](15_data_dictionary.md) | Consolidated Parquet data dictionary: every column of every artifact, its Arrow type and nullability, sourced from the `Col`/`write_table` construction with `file:line` citations. |
| [16_glossary.md](16_glossary.md) | Alphabetical glossary of domain and codebase terms as MuMDIA uses them, each entry self-contained and cited to `file:line` where it asserts code behavior. |
| [17_troubleshooting.md](17_troubleshooting.md) | Symptom -> cause -> fix lookup table for the quiet failure modes (silent fallbacks, nondeterminism, void results), cited to `file:line`. |
| [18_findings_and_decisions.md](18_findings_and_decisions.md) | Self-contained findings and contracts: validated results, interstage/determinism/sidecar contracts, current best workflow, and ranked roadmap, with no dependency on `plan.md`. |
| [19_getting_started.md](19_getting_started.md) | Reproducible getting-started: local sidecar environments, the pre-built E. coli test library, and two copy-pasteable end-to-end runs plus a smoke check. |
