# MuMDIA developer documentation

This directory is the deep code and workflow reference for the MuMDIA engine. It
describes every subsystem as it exists in the current tree: the crates and their
responsibilities, each pipeline stage and the artifacts it reads and writes, the
config and data model that tie them together, the Python sidecars, and the build,
test, and deployment machinery. It is written for a developer or agent taking
over the codebase who needs to understand not just what each part does but why it
is built that way. Read `CLAUDE.md` at the repository root first for quick
orientation (layout, build commands, the rules that hold across the tree).
`CLAUDE.md` states each rule and then routes here for the measurement behind it;
these documents, not `CLAUDE.md`, carry the numbers. For the validated findings
and the interstage, determinism, and sidecar contracts, read
`docs/18_findings_and_decisions.md`, which is self-contained and depends on no
gitignored file. The deeper algorithmic specification lives in local-only design
notes that are not required to use these docs. These docs sit between
`CLAUDE.md` and the code: more detail than `CLAUDE.md`, grounded in the actual
source rather than the spec.

## Recommended reading order

1. Start with **01 (overview and dataflow)** for the pipeline as a graph and the
   two library sources.
2. Read **17 (troubleshooting)** early, not only when something breaks. Its
   symptom -> cause -> fix table covers the failure modes that produce a
   plausible-looking but badly degraded result: a conversion peak cap that
   deletes most fragment evidence, competition that removes modified forms
   before FDR, per-run counts taken on an experiment-wide q column, and the
   sidecar import-order and Parquet-encoding contracts. Each of these has cost
   real result quality, and none of them announces itself as an error.
3. Read **02 (config and data model)** and **03 (IO layer)** next, since every
   stage depends on the shared config, mass model, and Parquet contract.
4. Then walk the stages in pipeline order: **04 -> 05 -> 06 -> 07 -> 08 -> 09 ->
   10 -> 11**, which follows a run from mzML input to FDR-controlled output.
5. Read **12 (quant and cross-run stages)** for the tail subcommands and
   experiment-level stages.
6. Read **13 (sidecars)** when you need the real ML predictors and rescorers.
7. Read **14 (build, test, deploy, gotchas)** before touching the build, the test
   suite, or the release machinery.
8. Keep the reference docs to hand: **15 (Parquet data dictionary)** for the
   column schema of every artifact and the exact grouping of every q column, and
   **16 (glossary)** for domain and codebase terms.
9. Read **18 (findings, decisions, and contracts)** for the self-contained
   restatement of the validated findings, the interstage/determinism/sidecar
   contracts, the current best workflow, and the ranked roadmap. It depends on no
   gitignored spec file and stands alone.
10. Read **19 (getting started)** for the reproducible local setup and two
    copy-pasteable end-to-end runs (native and best-sensitivity library).
11. Use **20 (sensitivity and quantification playbook)** when choosing or
    promoting settings: it separates the validated AIF reference from
    acquisition-specific tuning, FDR/entrapment gates, and quant accuracy.

## Where a cross-cutting question is answered

Several topics necessarily appear in more than one document. Each has one
canonical treatment that carries the measurements; every other mention is scoped
to its own stage and points back to that treatment. When adding material, extend
the canonical document and cite it from elsewhere rather than restating a
measurement locally, so the same finding cannot drift into two versions.

| Question | Canonical | Also relevant |
|---|---|---|
| What should `--top-peaks-ms2` be, and what does a cap cost? | **04** ("Choosing `--top-peaks-ms2`"): peak census, end-to-end effect, mechanism, dose-response | **20** for the pre-flight saturation check and the promotion policy, **09** for the extraction-side rejection, **17** for the symptom, **18 A3** for the decision record |
| Which q column do I count, and over what group? | **15** (per-column grouping and the winner-only assignment) | **16** for one-line definitions, **11** for how each is computed, **17** for the per-run/experiment-wide trap, **12** for the quant filter |
| Why is a modified form or a charge sibling missing? | **11** (competition key, `group_by` semantics, the `precursor` misnomer) | **17** for the symptom, **02** for the enum, **18** and **20** for the PTM policy |
| Does DeepLC fine-tuning have to run per file? | **08** (fine-tune scope, in-sample vs out-of-sample RT residuals, modform iRT) | **20** for the operational policy, **17** for the symptom, **13** for the worker contract |
| Why did a sidecar fail, or a rescore take far too long? | **13** (worker contracts, import order, backend selection, env vars) | **17** for symptom lookup, **14** for the build-side codec/string constraints, **11** for rescore scale |
| What must an externally written Parquet or library satisfy? | **03** (codec, string type, and the library preconditions) | **17** and **14** for the same contract stated from the failure side |

## Document index

| Doc | Description |
|---|---|
| [01_overview_and_dataflow.md](01_overview_and_dataflow.md) | The pipeline as a stage-and-artifact graph, the two library sources, the `run` orchestrator, and `manifest.json`. |
| [02_config_and_data_model.md](02_config_and_data_model.md) | The `mumdia-core` crate: typed config with per-stage overrides and strategy enums, mass model, constants, schema, and run manifest. |
| [03_io_layer.md](03_io_layer.md) | The `mumdia-io` crate: `Col`/`Table` over Arrow+Parquet, SNAPPY read/write, blake3 hashing, per-artifact `report.json`, and `inspect`. |
| [04_convert.md](04_convert.md) | Stage 0: mzML read through `mzdata`, profile centroiding, AIF full-range window fallback, the normalized spectra artifacts, and the canonical treatment of the destructive `--top-peaks-ms2` cap. |
| [05_digest_peptidoforms.md](05_digest_peptidoforms.md) | Stage A and A2: fully-tryptic in-silico digest with reverse/scramble decoys, then expansion into concrete peptidoforms with mods and charges. |
| [06_predict_frag_index_matchers.md](06_predict_frag_index_matchers.md) | Stage C: the run-independent library (b/y m/z, intensities, iRT), the peak-major inverted fragment index, and the fragment matchers. |
| [21_prescan.md](21_prescan.md) | Optional per-run sequence-tag prescan for modification searches: prunes modform hypotheses with no anchored tag support, why it is label-blind by construction, and why the decoy screen must be symmetric. |
| [07_search_seed.md](07_search_seed.md) | Stage S: the native Sage-lite broad DIA search that produces calibration anchors and per-run mass recalibration, not final IDs. |
| [08_rt_im_train.md](08_rt_im_train.md) | Stage B: per-run LOESS/linear RT calibration, residual-percentile RT windows, and the optional DeepLC multitask fine-tune. |
| [09_extract.md](09_extract.md) | Stage D: the core peak-major targeted extraction cascade, apex selection, chromatograms, MS1 isotope XICs, and the sensitivity knobs. |
| [10_features.md](10_features.md) | Stage E: the minimal/rich/extended feature battery (~381 features), `prelim_score`, PIN emission, and the hashed feature schema. |
| [11_compete_rescore_fdr.md](11_compete_rescore_fdr.md) | Stage F: within-group competition, semi-supervised rescoring (native / mokapot / PyTorch-NN / percolator / entrapment), and target-decoy q-values at PSM/run/precursor/peptide/protein level. |
| [12_quant_lfq_align_mbr_report_audit.md](12_quant_lfq_align_mbr_report_audit.md) | The tail subcommands: `quant`, `quant-lfq` (MaxLFQ/directLFQ), `align`, `mbr`, `report`, and `audit`. |
| [13_sidecars.md](13_sidecars.md) | The 11 Python scripts (7 engine-invoked sidecar workers: MS2PIP/DeepLC/mokapot/entrapment/NN/MBR; plus 4 imported-library helpers incl. `augment_library.py`), the positional-CLI file contract, and the conda envs. |
| [14_build_test_deploy_gotchas.md](14_build_test_deploy_gotchas.md) | The Rust workspace build, test coverage and gaps, the determinism contract, the clean-room boundary, and CI/Docker/release. |
| [15_data_dictionary.md](15_data_dictionary.md) | Consolidated Parquet data dictionary: every column of every artifact, its Arrow type and nullability, sourced from the `Col`/`write_table` construction with `file:line` citations. |
| [16_glossary.md](16_glossary.md) | Alphabetical glossary of domain and codebase terms as MuMDIA uses them, each entry self-contained and cited to `file:line` where it asserts code behavior. |
| [17_troubleshooting.md](17_troubleshooting.md) | Symptom -> cause -> fix lookup for the quiet failure modes (silent fallbacks, nondeterminism, void results, peak truncation, deleted modforms, misread q units, sidecar import order and Parquet encoding), cited to `file:line`. Read it early. |
| [18_findings_and_decisions.md](18_findings_and_decisions.md) | Self-contained findings and contracts: validated results, interstage/determinism/sidecar contracts, current best workflow, and ranked roadmap, with no dependency on untracked design notes. |
| [19_getting_started.md](19_getting_started.md) | Reproducible getting-started: the fixture run that needs nothing but the repository, the shipped sidecar environment specs and `"auto"` interpreter resolution, the pre-built E. coli test library, and two copy-pasteable end-to-end runs plus the regression guard. |
| [20_sensitivity_and_quantification_playbook.md](20_sensitivity_and_quantification_playbook.md) | Operational playbook separating validated AIF sensitivity, acquisition-specific choices, FDR/entrapment promotion gates, quantification accuracy, and benchmark-gated research. |
| [22_release_plan.md](22_release_plan.md) | Release plan (2026-08-27): tree state versus the stated gates, why the engine is hard to run today, the two-release scope (`v0.1.0` installable engine, `v0.2.0` second-pass workflow), work packages with acceptance criteria, sequencing, and the definition of done. |
| [23_cli_reference.md](23_cli_reference.md) | GENERATED by `ci/gen_cli_reference.py` from the binary's own `--help`: every subcommand, its arguments, which ones accept `--config`, and the four `global = true` flags documented once. Regenerate rather than edit. |
| [24_config_reference.md](24_config_reference.md) | GENERATED by `ci/gen_config_reference.py` from `config.rs`: every config section with each field's type, default, doc comment, and benchmark-gated marker, the enum value sets, the named `--profile` overrides, and the only complete table of the environment variables the engine and the sidecars read. Regenerate rather than edit. |
| [25_release_readiness_review.md](25_release_readiness_review.md) | Pre-release audit (2026-08-28) of the tree at `d1f3f3e`: the open blockers, the NaN-permeable external-input boundary, the apex fallback, the three defects on the entrapment path, quantification and interface findings, security and supply-chain advisories, what the end-to-end test structurally cannot catch, documentation claims that overstate, a prioritised order of work, and the questions the audit could not settle. |
| [26_gui_plan.md](26_gui_plan.md) | Design for a simple graphical interface: why an in-binary local web UI (`mumdia serve`) rather than a desktop toolkit or an Electron app, the four-phase path from machine-readable progress to a browser front end, and what each phase is worth on its own. |
