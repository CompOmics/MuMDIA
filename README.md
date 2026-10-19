# MuMDIA

[![CI](https://img.shields.io/github/actions/workflow/status/CompOmics/MuMDIA/ci.yml?branch=main&label=CI)](https://github.com/CompOmics/MuMDIA/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Container](https://img.shields.io/badge/ghcr.io-compomics%2Fmumdia-1f6feb?logo=docker&logoColor=white)](https://github.com/CompOmics/MuMDIA/pkgs/container/mumdia)

MuMDIA is a search engine for data-independent acquisition (DIA) proteomics, written in
Rust. It reads Orbitrap and Astral `mzML` and Bruker diaPASEF `.d` files (natively, with
ion mobility), searches them against an in-silico FASTA digest or an imported predicted
spectral library, and reports peptides and protein groups under target-decoy FDR control
with label-free quantities. Optional Python sidecars add machine-learning components:
DeepLC retention time, MS2PIP, AlphaPeptDeep and IM2Deep predictions, and a PyTorch or
mokapot rescorer. A desktop application wraps the engine for users who prefer not to use a
terminal.

The engine is validated on public DIA data with entrapment experiments. The command line,
configuration and artifact formats can still change between minor versions.

## What it does

One binary with one subcommand per stage. Each stage also runs standalone on the outputs
of the previous one.

```text
FASTA -> digest -> peptidoforms -> predict-frag --+
imported spectral library ------------------------+-> search-seed -> rt-im-train
mzML / diaPASEF .d / vendor file -> convert ------+   (multi-head DeepLC calibration)
                                                                  |
          extract -> (retrace, diaPASEF) -> features -> compete -> rescore -> quant -> report
```

`mumdia run` runs the whole chain. Given several files it searches them as one experiment:
a pooled rescore, a second pass that matches between runs, and cross-run MaxLFQ. Every
artifact is Parquet with a `report.json` that records its schema version, row count,
content hash, resolved parameters and the model that ran, and every run writes a
`manifest.json`.

Development is measured on three separate objectives: identification sensitivity, FDR
validity under entrapment, and quantification accuracy on known-ratio data. A higher
identification count is not evidence for the other two
([`docs/20`](docs/20_sensitivity_and_quantification_playbook.md)).

## Install

- **Desktop application.** Each [release](https://github.com/CompOmics/MuMDIA/releases)
  has a Windows installer (`.msi`) and a Linux `.AppImage`. It sets up the Python
  environments, builds libraries and runs searches ([`desktop/README.md`](desktop/README.md)).
- **Docker**, the reference environment with the sidecars installed:

  ```bash
  docker run --rm -v "$PWD:/data" --user "$(id -u):$(id -g)" ghcr.io/compomics/mumdia \
      run --fasta /data/proteome.fasta --mzml /data/sample.mzML \
          --out-dir /data/results --config /opt/mumdia/config.dia.json
  ```

  `--user` is required when writing into a bind mount (on Windows PowerShell, drop it and
  use `-v "${PWD}:/data"`). The image carries two configurations:
  `/opt/mumdia/config.dia.json` (FASTA digest, MS2PIP, DeepLC, mokapot) and
  `/opt/mumdia/config.diann-lib.json` (imported library, multi-head DeepLC calibration,
  `nn_torch`).
- **Release archive**, one per platform (`x86_64-unknown-linux-musl`,
  `x86_64-pc-windows-msvc`, `aarch64-apple-darwin`), with the binary, the sidecar
  `scripts/`, the conda specifications in `env/`, example configurations and the docs.
- **From source:**

  ```bash
  git clone https://github.com/CompOmics/MuMDIA.git
  cd MuMDIA/rust/mumdia && cargo build --release --locked
  ```

  The toolchain is pinned in `rust-toolchain.toml`; a C compiler is needed. On a synced
  drive (OneDrive, Dropbox), move the Cargo target directory off it first
  (`rust/mumdia/.cargo/config.toml.example`).

Outside Docker, the sidecars need a conda environment, then a check:

```bash
conda env create -f env/mumdia-deeplc.yml     # DeepLC, MS2PIP and the nn_torch rescorer
conda env create -f env/mumdia-peptdeep.yml   # AlphaPeptDeep, for the diaPASEF recipe
mumdia doctor --config configs/examples/diann-library.json
```

`doctor` resolves every interpreter as a run would, reports the versions it found and exits
non-zero if the configuration cannot run. DeepLC 4.5.0 or newer is required.

## Quickstart

```bash
# no Python at all: proves the installation and the input files
mumdia run --fasta proteome.fasta --mzml sample.mzML --out-dir results \
           --config configs/examples/native.json

# imported spectral library with the sidecars (the most sensitive workflow)
mumdia run --lib-precursors lib_precursors.parquet --lib-fragments lib_fragments.parquet \
           --mzml sample.mzML --out-dir results --config configs/examples/diann-library.json

# several runs as one experiment; repeat --mzml
mumdia run --lib-precursors lib_precursors.parquet --lib-fragments lib_fragments.parquet \
           --mzml a.mzML --mzml b.mzML --mzml c.mzML --out-dir exp \
           --config configs/examples/diann-library.json

# Bruker diaPASEF, read natively with ion mobility
mumdia run --fasta proteome.fasta --mzml sample.d --out-dir results \
           --config configs/examples/diapasef.json
```

Use a fresh output directory: `run` always recomputes its outputs. Building an imported
library (predicted library, import, paired decoys with predicted spectra) is described in
[`docs/13_sidecars.md`](docs/13_sidecars.md#dia-nn-recipe-offline-license-clean), and
[`configs/README.md`](configs/README.md) explains each example configuration.
[`docs/19_getting_started.md`](docs/19_getting_started.md) has complete worked runs.

## Output

| file | contents |
|---|---|
| `peptides.tsv`, `proteins.tsv` | identified precursors and protein groups, for reading |
| `psms_scored.parquet` | every scored PSM with all q columns; the unit for analysis |
| `peptide_quant.parquet`, `protein_group_quant.parquet` | quantities for analysis |
| `lfq_maxlfq.parquet` | cross-run protein matrix (experiments) |
| `manifest.json`, `*.report.json` | provenance, hashes, parameters, model identities |

Use the Parquet tables for analysis; the TSVs are rounded for reading. Each q column
controls its own row unit (`run_psm_q` per run, `precursor_q` per precursor,
`peptide_q_value` per peptide, `pg_q_value` per protein group), so name the column with
any count. `peptides.tsv` lists precursors but is filtered on the peptide-level q. In an
experiment the TSVs are experiment-wide, with one quantity column per run. Details in
[`docs/11_compete_rescore_fdr.md`](docs/11_compete_rescore_fdr.md) and
[`docs/15_data_dictionary.md`](docs/15_data_dictionary.md).

## Limits

- Input is mzML or a Bruker diaPASEF `.d`. Thermo `.raw` is converted with
  ThermoRawFileParser; other vendor formats go through ProteoWizard `msconvert` and are
  not verified end to end ([`docs/04_convert.md`](docs/04_convert.md)).
- Wildcard and terminal variable modifications are not supported.
- CI builds and tests the engine on three platforms and runs an end-to-end smoke test, but
  does not run the Python sidecars on data. The PyTorch rescorer and the DeepLC fine-tune
  are not bit-reproducible, so counts vary slightly between runs.
- A peak cap at conversion (`--top-peaks-ms2`) deletes fragment evidence. The default is
  uncapped; set a cap only from a sweep on your own acquisition.

## Documentation

- [`docs/README.md`](docs/README.md): the developer guide, one document per subsystem.
- [`docs/19_getting_started.md`](docs/19_getting_started.md) and
  [`docs/17_troubleshooting.md`](docs/17_troubleshooting.md): worked runs and common
  failure modes.
- [`docs/35_diapasef.md`](docs/35_diapasef.md): diaPASEF support and its measurements.
- [`docs/23_cli_reference.md`](docs/23_cli_reference.md) and
  [`docs/24_config_reference.md`](docs/24_config_reference.md): every subcommand and
  configuration field, generated from the code.
- [`bench/README.md`](bench/README.md): benchmarks and how to reproduce them.
- [`CHANGELOG.md`](CHANGELOG.md), [`CONTRIBUTING.md`](CONTRIBUTING.md),
  [`SECURITY.md`](SECURITY.md).

## Citation

There is no `CITATION.cff` or DOI yet. Cite the repository URL with the release tag you
used.

## License and attribution

Apache-2.0, see [LICENSE](LICENSE). The notices of the statically linked Rust crates are in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), and `sbom.cdx.json` lists them in
CycloneDX form.

Developed at [CompOmics](https://www.compomics.com), VIB-UGent Center for Medical
Biotechnology, Ghent University. This is a ground-up reimplementation of the earlier
MuMDIA, which is archived on the `legacy-python` branch (tag `legacy-python-v1`). MuMDIA
keeps a clean-room boundary against closed implementations. It does not ship or invoke
DIA-NN; users build imported libraries under their own DIA-NN licence.
