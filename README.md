# MuMDIA

A data-independent-acquisition (DIA) peptide search engine, reimplemented from
scratch in Rust. MuMDIA searches DIA mass spectrometry data (mzML) against a
protein sequence database or a spectral library and reports identified peptides
and proteins with target-decoy FDR control.

> Status: early. This is a clean-room Rust rewrite of MuMDIA. The core pipeline
> runs end to end and is validated on E. coli DIA data (see below), but the
> packaging, defaults, and documentation are still being hardened for general
> use. Interfaces may change.

## What it does

- Reads DIA `mzML`, builds a fragment library from a FASTA digest or an imported
  predicted spectral library, and searches with a peak-major inverted fragment
  index.
- Per-run retention-time calibration and windowing, apex detection, chromatogram
  extraction, a rich per-PSM feature battery, target-decoy competition, and
  semi-supervised rescoring.
- Native, dependency-free defaults, with optional Python sidecars for stronger
  models (DeepLC retention time, MS2PIP fragment intensities, mokapot rescoring,
  and an entrapment-based rescorer).

Each stage is an independent subcommand that reads path-addressable inputs and
writes Parquet plus a per-artifact `report.json`, so the pipeline is inspectable
and resumable at any step.

## Pipeline

```
convert -> digest -> peptidoforms -> predict-frag -> search-seed ->
rt-im-train -> extract -> features -> compete -> rescore
```

`mumdia run` orchestrates the whole chain on one file and writes a
`manifest.json`; `mumdia inspect <artifact>` prints schema, head, and row count
for any Parquet output.

## Run with Docker (bundles all sidecars)

The published image contains the engine plus the Python sidecars (mokapot,
MS2PIP, DeepLC), so the full high-sensitivity recipe runs with nothing to install
but Docker:

```
docker pull ghcr.io/compomics/mumdia:latest

docker run --rm -v "$PWD:/data" ghcr.io/compomics/mumdia \
    run --fasta   /data/proteome.fasta \
        --mzml    /data/sample.mzML \
        --out-dir /data/results \
        --config  /opt/mumdia/config.dia.json
```

Mount your working directory at `/data`; the outputs (including `peptides.tsv`
and `proteins.tsv`) appear under `results/`. The baked
`/opt/mumdia/config.dia.json` selects the Extended feature set and the DIA apex
settings, and wires DeepLC, MS2PIP, and mokapot (logistic regression) to the
in-image conda environments. To run the native, dependency-free models instead,
drop `--config` and add `--profile dia`.

## Build

Requires Rust >= 1.85 (the dependencies use edition 2024; `rustup update` if
older). All dependencies are pure Rust, so no C toolchain is needed.

```
cd rust/mumdia
cargo build --release
cargo test          # unit + integration tests
```

The binary is written to `target/release/mumdia` (or `mumdia.exe` on Windows).
If you build on a cloud-synced drive (OneDrive, Dropbox), redirect the build
directory to a local path to avoid sync corruption; see
`rust/mumdia/.cargo/config.toml.example`.

## Quickstart

One command from a FASTA and a DIA mzML, using the validated DIA preset:

```
mumdia run \
  --fasta  proteome.fasta \
  --mzml   sample.mzML \
  --out    results \
  --profile dia
```

`--profile dia` applies the tuned settings (extended features, rolling-window
apex, retention-time prior). `run` writes human-readable `results/peptides.tsv`
and `results/proteins.tsv` (identified peptides and protein groups with q-values
and quantities), alongside the Parquet artifacts and a `manifest.json`; use
`mumdia inspect <artifact>` to view any Parquet. Any stage, and the report
(`mumdia report --scored … --out-dir …`), can also be run standalone on prior
outputs.

## Optional Python sidecars

The native predictors and rescorer run with zero external dependencies. For
higher sensitivity, MuMDIA can call Python sidecars over a simple file contract
(input Parquet in, output Parquet out). The mokapot rescorer, for example, needs
only a small environment (`mokapot`, `scikit-learn`, `numpy`, `pyarrow`,
`pandas`); DeepLC and MS2PIP need their own environments. Sidecar selection and
the Python interpreter path are set in the configuration. The Docker image above
bundles all three so no manual environment setup is needed; the environment
specifications are under `env/` (`mumdia-rescore.yml`, `docker-rescore.yml`,
`docker-deeplc.yml`).

## FDR

MuMDIA controls the false discovery rate with target-decoy competition (reverse
or fragment-shift decoys), the standard approach in the field. As with other DIA
engines, decoy-based FDR can be optimistic on highly chimeric data; a foreign
proteome spike-in (entrapment) can be used to measure a decoy-independent true
FDR when a rigorous estimate is required.

## Benchmark

On the ProteomeXchange E. coli AIF file `LFQ_Orbitrap_AIF_Ecoli_01`, with a
DIA-NN-predicted library and per-run fine-tuned retention time, MuMDIA reports on
the order of 10,000 peptides at 1% FDR (mokapot), and about 9,000 E. coli
peptides at a genuine 1% FDR measured by human entrapment, at roughly 98%
sequence concordance with DIA-NN.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Attribution

Developed at [CompOmics](https://www.compomics.com), VIB-UGent Center for Medical
Biotechnology, Ghent University. This project is a ground-up reimplementation of
the earlier MuMDIA; the previous version is archived on the `legacy-python`
branch.
