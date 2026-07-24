# MuMDIA Rust workspace

This directory is the Cargo workspace for MuMDIA, a clean-room DIA proteomics
search engine. The current implementation extends beyond the original MVP: it
includes imported-library search, Extended features, optional DeepLC/MS2PIP and
rescoring sidecars, quantification/LFQ, alignment, audit, and a partially wired
MBR path.

The authoritative user overview is [`../../README.md`](../../README.md). The
code-grounded developer guide starts at
[`../../docs/README.md`](../../docs/README.md); use
[`../../docs/19_getting_started.md`](../../docs/19_getting_started.md) for
commands and
[`../../docs/20_sensitivity_and_quantification_playbook.md`](../../docs/20_sensitivity_and_quantification_playbook.md)
for scientific tuning and validation policy.

## Build and test

```text
cargo test --workspace
cargo build --release --locked
```

On the repository's Windows development machine, `.cargo/config.toml` redirects
the target directory away from OneDrive. Copy `.cargo/config.toml.example` and
choose another local path if needed.

## Execution model

`mumdia run` resolves a FASTA-built or imported spectral library and converted
mzML spectra, then executes seed search, optional RT fine-tuning, RT calibration,
extraction, features, competition, rescoring, quantification, and report
generation. It records `manifest.json` but does not read it for caching or
resume. Use a fresh output directory for each orchestrated run.

Every major stage is also a standalone subcommand over path-addressable Parquet
artifacts. Run `mumdia --help`, `mumdia <stage> --help`, or
`mumdia inspect <artifact.parquet>` for the live CLI and schema.

Converted MS2 spectra are uncapped by default (`--top-peaks-ms2 0`).
`search_seed.top_n_peaks = 300` is a separate calibration-search cap. The
validated Orbitrap AIF sensitivity recipe explicitly converts the top 300 peaks,
but that choice must be retuned for other acquisition geometries.

External rescorers should run with `rescore.strict = true`; confirm the backend
actually used in `psms_scored.parquet.report.json`. Target-decoy FDR requires a
library containing both valid target and decoy candidates.
