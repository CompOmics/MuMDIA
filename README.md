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
and `proteins.tsv`) appear under `results/`. On Windows PowerShell, use
`-v "${PWD}:/data"` (not `$PWD`, which PowerShell parses as a drive reference). The baked
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

## Using a DIA-NN spectral library (highest sensitivity)

By default MuMDIA builds its library from a FASTA digest and predicts fragment
intensities with the native model or MS2PIP. For the highest sensitivity, and to
reproduce the benchmark numbers, you can supply a spectral library predicted by
DIA-NN and have MuMDIA consume it directly. In this **library-input mode**, `run`
skips the digest, MS2PIP, and DeepLC steps and uses DIA-NN's fragment intensities
and retention times.

MuMDIA does not include or download DIA-NN. You run DIA-NN yourself, under your
own license: the DIA-NN "Academia" build is free for non-profit academic research
(https://github.com/vdemichev/DiaNN). MuMDIA only reads the library file you
produce, and never redistributes DIA-NN.

The Python steps below need `pandas` and `pyarrow` (the mokapot sidecar env has
both; in Docker use `/opt/conda/envs/rescore/bin/python`).

1. **Predict a spectral library from your FASTA with DIA-NN** (library-free /
   in-silico), matching your search parameters, and output a fragment-level
   parquet library. For example:

   ```
   diann --fasta proteome.fasta --fasta-search --gen-spec-lib --predictor \
         --cut "K*,R*" --missed-cleavages 1 \
         --min-pep-len 7 --max-pep-len 30 --min-pr-charge 2 --max-pr-charge 4 \
         --unimod4 --var-mods 1 --var-mod "UniMod:35,15.994915,M" \
         --out-lib lib --threads 8
   ```

   MuMDIA maps only Carbamidomethyl (fixed) and Oxidation (variable);
   precursors with other modifications are dropped on import.

2. **Import the DIA-NN library into MuMDIA's schema** (targets only):

   ```
   python scripts/import_diann_lib.py \
       lib.parquet lib_precursors_targets.parquet lib_fragments_targets.parquet
   ```

3. **Add the decoy population.** This also sorts by precursor m/z and re-indexes
   `candidate_id`, which MuMDIA's fragment index requires:

   ```
   python scripts/make_reverse_decoys.py \
       lib_precursors_targets.parquet lib_fragments_targets.parquet \
       lib_precursors.parquet lib_fragments.parquet
   ```

4. **Run MuMDIA in library-input mode** (no `--fasta`):

   ```
   mumdia run \
     --lib-precursors lib_precursors.parquet \
     --lib-fragments  lib_fragments.parquet \
     --mzml    sample.mzML \
     --out-dir results \
     --profile dia
   ```

   Everything downstream (search-seed, RT calibration, extraction, features,
   competition, rescoring, quant, report) is unchanged. Because the DIA-NN
   library supplies both fragment intensities and retention times, no fragment
   or RT prediction sidecar is required in this mode.

**Optional: fine-tune retention time (matches the benchmark).** You can adapt
DeepLC's multitask retention-time model to this run and rewrite the library's
retention times before extraction (the benchmark's per-run RT lever). Enable it
in the config and point at a DeepLC 4.0 multitask environment:

```json
{
  "rt_im_train": { "finetune_deeplc": true, "rt_window_multiplier": 1.5 },
  "predict_frag": { "deeplc_python": "/path/to/deeplc/python", "sidecar_script_dir": "scripts" }
}
```

Pass it with `--config`. `run` then fine-tunes on the confident seed PSMs and
re-predicts iRT for the whole library between search-seed and RT calibration. A
ready-made config for the Docker image is `docker/config.diann-lib.json` (it
targets the image's bundled `deeplc` environment). The fine-tune uses no fixed
random seed, so identification counts vary slightly between runs.

In Docker, mount the library files and point `--lib-precursors` /
`--lib-fragments` at the mounted paths; steps 2-3 (and the fine-tune) run in the
image's bundled environments.

## FDR

MuMDIA controls the false discovery rate with target-decoy competition (reverse
or fragment-shift decoys), the standard, community-accepted approach, and reports
target-decoy q-values at PSM, peptide, and protein-group level. An optional
entrapment (foreign-proteome spike-in) rescorer is available as a
decoy-independent cross-check for experiments that want one, but it is not
required.

## Benchmark

On the ProteomeXchange E. coli AIF file `LFQ_Orbitrap_AIF_Ecoli_01`, with a
DIA-NN-predicted library and per-run fine-tuned retention time, MuMDIA reports on
the order of 9,000 to 10,000 peptides at 1% FDR (mokapot), at roughly 97 to 98%
sequence concordance with DIA-NN.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Attribution

Developed at [CompOmics](https://www.compomics.com), VIB-UGent Center for Medical
Biotechnology, Ghent University. This project is a ground-up reimplementation of
the earlier MuMDIA; the previous version is archived on the `legacy-python`
branch.
