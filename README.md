# MuMDIA

[![CI](https://img.shields.io/github/actions/workflow/status/CompOmics/MuMDIA/ci.yml?branch=main&label=CI)](https://github.com/CompOmics/MuMDIA/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Container](https://img.shields.io/badge/ghcr.io-compomics%2Fmumdia-1f6feb?logo=docker&logoColor=white)](https://github.com/CompOmics/MuMDIA/pkgs/container/mumdia)

MuMDIA is a data-independent-acquisition (DIA) peptide search engine written in
Rust, with optional Python sidecars for the machine-learning components (DeepLC
retention time, MS2PIP fragment intensities, mokapot and a PyTorch rescorer). It
reads DIA `mzML`, searches against either an in-silico FASTA digest or an
imported predicted spectral library, and reports peptides and protein groups
under target-decoy FDR control together with label-free quantities. This is early
software. The engine runs end to end and is validated on public DIA data, but the
Python sidecars are not exercised by continuous integration, and the
command-line, configuration, and artifact interfaces may change between minor
versions.

## What it does

One binary, one subcommand per stage, each runnable standalone on
path-addressable inputs:

```text
FASTA -> digest -> peptidoforms -> predict-frag --+
                                                   +-> search-seed
imported spectral library ------------------------+       |
mzML -> convert ----------------------------------+       v
                                                  optional RT fine-tune
                                                           |
                                                           v
                                      rt-im-train -> extract -> features
                                      -> compete -> rescore -> quant -> report
```

`mumdia run` orchestrates the whole chain on one file and writes a
`manifest.json`. Every primary artifact is Parquet with an adjacent
`report.json` recording schema version, row count, blake3 content hash, resolved
parameters, and the model that actually ran. `run` has no cache and no resume: it
always recomputes and overwrites its named outputs, so use a fresh output
directory.

MuMDIA is tuned against three objectives that are measured separately:

1. **identification sensitivity**: more true discoveries at a stated q threshold;
2. **FDR validity**: the q threshold stays calibrated under entrapment and
   exchangeable paired decoys;
3. **quantification accuracy**: low bias, low CV, and low missingness on
   known-ratio data.

A higher identification count is not evidence that the other two improved. More
identifications at 1% q says nothing about whether 1% is still 1%, and nothing
about quantification. A claim about one objective needs a measurement of that
objective. See
[`docs/20_sensitivity_and_quantification_playbook.md`](docs/20_sensitivity_and_quantification_playbook.md).

## Install

Three ways, in increasing order of what you have to set up yourself.

### Docker (the reference environment)

The published image contains the engine plus the Python sidecars (mokapot,
MS2PIP, DeepLC, PyTorch), so nothing but Docker is needed:

```bash
docker pull ghcr.io/compomics/mumdia:latest

docker run --rm -v "$PWD:/data" --user "$(id -u):$(id -g)" \
    ghcr.io/compomics/mumdia \
    run --fasta   /data/proteome.fasta \
        --mzml    /data/sample.mzML \
        --out-dir /data/results \
        --config  /opt/mumdia/config.dia.json
```

`--user "$(id -u):$(id -g)"` is required, not optional, whenever the engine
writes into a bind mount. The image drops to an unprivileged user whose uid does
not match yours, and without `--user` even creating the output directory fails
with `mkdir: cannot create directory '/data': Permission denied` (measured
2026-08-27; the Docker workflow asserts both the failure and the fix). With it,
the results are owned by you. `id` is a POSIX command, so that form applies on
Linux and macOS; on Windows PowerShell, drop the `--user` argument and use
`-v "${PWD}:/data"`, because `$PWD` is parsed there as a drive reference.

Two configurations are baked into the image:

| path | what it selects |
|---|---|
| `/opt/mumdia/config.dia.json` | FASTA digest, MS2PIP fragment intensities, DeepLC retention time, strict mokapot rescoring |
| `/opt/mumdia/config.diann-lib.json` | imported spectral library, per-run DeepLC fine-tune, strict `nn_torch` rescoring |

Both set `rescore.strict = true`, so a requested sidecar fails loudly instead of
silently becoming a native run. Check either one before a long run:

```bash
docker run --rm ghcr.io/compomics/mumdia doctor --config /opt/mumdia/config.dia.json
```

The published image is `linux/amd64`, so it runs under emulation on Apple
Silicon.

### Release archive

Each tagged release attaches one archive per platform
(`x86_64-unknown-linux-musl`, `x86_64-pc-windows-msvc`, `aarch64-apple-darwin`)
plus a `.sha256` checksum file per archive. Windows ships a `.zip`, the others a
`.tar.gz`. There is no Intel Mac archive: it required a GitHub runner label that
is no longer served, and cross-compiling it would mean publishing the one binary
nobody had executed. Build from source or use the container image on an Intel
Mac. An archive contains:

- the `mumdia` binary, smoke-tested on its own architecture before upload;
- `scripts/`, the Python sidecar workers the engine launches by path;
- `env/`, the conda environment specifications for those workers;
- `configs/`, the example configurations;
- `docs/`, `README.md`, `CHANGELOG.md`, `LICENSE`,
  `THIRD_PARTY_LICENSES.md` (the notices of the 173 crates linked into the
  binary) and `sbom.cdx.json` (the same inventory as CycloneDX 1.5, for
  vulnerability scanners and software inventories);
- `ci/smoke.sh` with its two helpers and `test_data/fixture.fasta`, so the
  installation check in `docs/19_getting_started.md` can be run from the archive.
  Every archive is unpacked and put through that check on its own platform before
  it is published.

The binary alone cannot run the Python sidecars. Without a Python environment it
is limited to the native predictors and the native `native_tda` rescorer, which
is a supported but much less sensitive path (see the benchmark below). `scripts/`
and `env/` ship in the archive precisely so that the sidecar path is reachable
from a downloaded release.

### From source

```bash
git clone https://github.com/CompOmics/MuMDIA.git
cd MuMDIA/rust/mumdia
cargo build --release --locked
```

The build needs no cmake and no system C libraries, but it does need a C compiler:
`libmimalloc-sys` compiles the vendored mimalloc allocator and `blake3` compiles its
SIMD paths, both via the `cc` build dependency. Every supported platform's default
toolchain provides one. The Rust toolchain is pinned by
`rust/mumdia/rust-toolchain.toml`, so rustup installs the right version on the
first `cargo` invocation. The binary lands in `target/release/mumdia`
(`mumdia.exe` on Windows). If you build on a cloud-synced drive (OneDrive,
Dropbox), redirect the Cargo target directory off it first: copy
`rust/mumdia/.cargo/config.toml.example` to `rust/mumdia/.cargo/config.toml` and
set a local path. Syncing incremental build artifacts causes file-locking
corruption.

### Sidecar environments

Outside Docker, create the environment for the sidecars you intend to use:

```bash
# mokapot rescoring only (no torch, no DeepLC, no MS2PIP)
conda env create -f env/mumdia-rescore.yml

# DeepLC retention-time prediction and per-run fine-tuning, plus CPU PyTorch
conda env create -f env/mumdia-deeplc.yml
```

`env/mumdia-deeplc.yml` pins `deeplc==4.1.1` and `torch==2.12.1+cpu`. DeepLC
4.1.1 is a floor rather than merely the current release: the 4.0.0a2 multitask
preview overfits per-run fine-tuning badly enough to invert retention-time model
rankings, so an older version changes results and not only speed.

Then, before any long run:

```bash
mumdia doctor --config configs/examples/diann-library.json
```

`doctor` resolves each sidecar interpreter exactly as a run does, reports how it
was found and which versions it has, checks that the worker scripts are where the
engine will look for them, warns below the DeepLC floor, and exits non-zero if the
configuration cannot run.

## Quickstart

### Native, no Python at all

```bash
mumdia run \
  --fasta   proteome.fasta \
  --mzml    sample.mzML \
  --out-dir results \
  --config  configs/examples/native.json
```

`configs/examples/native.json` selects the Extended feature set, the DIA apex
settings, and the loose extraction gate, and leaves every predictor and the
rescorer native. It needs no interpreter, no environment, and no network. Run
this path first: it proves the installation and the input files before any
sidecar is involved.

Outputs land in `results/`. `peptides.tsv` and `proteins.tsv` are the
human-readable reports; `mumdia inspect results/psms_scored.parquet` prints the
schema, a head sample, and the row count of any Parquet artifact.

### With the sidecars, on an imported spectral library

```bash
mumdia doctor --config configs/examples/diann-library.json

mumdia run \
  --lib-precursors lib_precursors.parquet \
  --lib-fragments  lib_fragments.parquet \
  --mzml           sample.mzML \
  --out-dir        results \
  --config         configs/examples/diann-library.json
```

This is the highest-sensitivity workflow measured so far: imported library,
per-run DeepLC fine-tuning, Extended features, the loose `apex_pearson`
extraction gate, and `nn_torch` rescoring. Building the two library tables is
described under [Two library sources](#two-library-sources).

`configs/examples/fasta-sidecars.json` is the middle option: digest a FASTA,
predict fragment intensities with MS2PIP and retention time with DeepLC, rescore
with mokapot.

### Other useful invocations

```bash
# any stage runs standalone on prior outputs
mumdia report --psms-scored results/psms_scored.parquet --out-dir results \
              --peptide-quant results/peptide_quant.parquet \
              --protein-quant results/protein_group_quant.parquet --q 0.01

# pooled multi-run search plus one combined rescore; repeat --mzml per run
mumdia run-experiment --mzml a.mzML --mzml b.mzML --mzml c.mzML \
                      --out-dir exp --config configs/examples/diann-library.json

# combine per-run quant tables into a protein-by-run matrix
mumdia quant-lfq --inputs exp/r0/peptide_quant.parquet exp/r1/peptide_quant.parquet \
                 --method maxlfq --out exp/protein_matrix.parquet
```

`run-experiment` writes one subdirectory per run (`r0`, `r1`, ... unless
`--run-names` is given) and never calls the report stage, so it produces no
`peptides.tsv` or `proteins.tsv` anywhere in its output tree. See
[Status of experimental features](#status-of-experimental-features).

## Configuration

A configuration is a JSON file passed with `--config`. It is strictly typed and
rejects unknown keys, so a misspelled field fails at load rather than being
silently ignored. Every field not mentioned keeps its default.

- [`configs/README.md`](configs/README.md) describes the three example
  configurations and when to use each.
- [`docs/02_config_and_data_model.md`](docs/02_config_and_data_model.md) is the
  field-level reference.

**Interpreter resolution.** A Python field (`rescore.python`,
`predict_frag.deeplc_python`, `predict_frag.ms2pip_python`, `mbr.python`) may be
an absolute path, or `"auto"`, or absent. Under `"auto"` the engine searches, in
order, the role's own variable (`MUMDIA_PYTHON_RESCORE`, `MUMDIA_PYTHON_DEEPLC`,
`MUMDIA_PYTHON_MS2PIP`, `MUMDIA_PYTHON_MBR`), then `MUMDIA_PYTHON`, then an
activated environment via `CONDA_PREFIX` or `VIRTUAL_ENV`, then `python3` and
`python` on `PATH`. A candidate is accepted only if it can import the modules
that role's workers import, so `auto` cannot quietly select a Python without
torch and defer the failure to hour three of a run. An explicit path is honoured
as given and never second-guessed.

**Global flags**, accepted on either side of the subcommand: `--threads N` bounds
the engine's thread pool and is forwarded to the sidecars as `MUMDIA_NN_THREADS`
and `OMP_NUM_THREADS` when those are unset; `--log-level LEVEL` takes any
`RUST_LOG` filter; `-v` and `-vv` raise verbosity to debug and trace; `-q`
restricts output to warnings and errors.

`--config` is not accepted by `convert`, `quant-lfq`, `inspect`, `audit`, and
`report`; those subcommands take their parameters as flags.

## Two library sources

### Imported spectral library (highest measured sensitivity)

In library-input mode `run` skips digest, peptidoform expansion, and initial
fragment prediction, and uses the imported library's fragment intensities. For
retention time the default (`rt_im_train.library_irt = auto`) re-predicts the
imported iRT once with the DeepLC base model when a DeepLC interpreter is
configured, then calibrates it per run; without an interpreter the imported values
are kept and a warning says so. DeepLC 4.1.1 or newer is required (`mumdia doctor`
and the workers refuse older versions). Measured at 1%: AIF 10,416 peptides against
10,015 from the imported iRT and 10,181 from a per-run fine-tune; HYE B01 (NN seeds
1-3) 58,842 against 56,556, and 60,278 with a library fine-tuned once. The optional
fine-tune (`finetune_deeplc`, on the original imported table) remains the
recommended extra step on a large reference.

MuMDIA neither ships nor invokes DIA-NN, and never redistributes it. You run
DIA-NN yourself under your own licence; the DIA-NN "Academia" build is free for
non-profit academic research (<https://github.com/vdemichev/DiaNN>). MuMDIA only
reads the library file you produce.

The Python steps below need `pandas` and `pyarrow` (the mokapot sidecar
environment has both; in Docker use `/opt/conda/envs/rescore/bin/python`).

1. **Predict a spectral library from your FASTA with DIA-NN** (library-free,
   in-silico), matching your search parameters, and output a fragment-level
   Parquet library. For example:

   ```bash
   diann --fasta proteome.fasta --fasta-search --gen-spec-lib --predictor \
         --cut "K*,R*" --missed-cleavages 1 \
         --min-pep-len 7 --max-pep-len 30 --min-pr-charge 2 --max-pr-charge 4 \
         --unimod4 --var-mods 1 --var-mod "UniMod:35,15.994915,M" \
         --out-lib lib --threads 8
   ```

   MuMDIA maps only Carbamidomethyl (fixed) and Oxidation (variable);
   precursors carrying other modifications are dropped on import.

2. **Import the library into MuMDIA's schema** (targets only):

   ```bash
   python scripts/import_diann_lib.py \
       lib.parquet lib_precursors_targets.parquet lib_fragments_targets.parquet
   ```

3. **Add the decoy population.** This also sorts by precursor m/z and re-indexes
   `candidate_id`, both of which the fragment index requires:

   ```bash
   python scripts/make_reverse_decoys.py \
       lib_precursors_targets.parquet lib_fragments_targets.parquet \
       lib_precursors.parquet lib_fragments.parquet
   ```

4. **Run MuMDIA in library-input mode** (no `--fasta`), as in the quickstart
   above. Everything downstream is unchanged. No fragment prediction sidecar is
   required in this mode; a DeepLC interpreter is needed for the default
   retention-time re-prediction, and without one the imported iRT is used.

**Prefer the augmented tables when you can build them.** An imported DIA-NN
library is missing the N-terminal methionine-excised peptides that DIA-NN's own
digest produces: on the AIF benchmark it lacked 209 of DIA-NN's own 1% peptides,
all of that form. `scripts/augment_library.py` adds the tryptic FASTA peptides an
imported library lacks, reusing the engine's own digest and fragment prediction so
the added entries carry byte-identical peptidoform strings, then hands off to
`make_shift_decoys.py` or `make_reverse_decoys.py` for the paired decoys. The
augmented tables added 18,903 tryptic base peptides and recovered about 80 of the
209 at an unchanged 0.98% empirical decoy fraction, with identification parity
elsewhere. The remaining peptides enter the search space but stay below
threshold, which is an abundance limit and not a library hole.

### Native FASTA digest

The default, and the option that needs no third-party licence. `digest` performs
a fully-tryptic in-silico digest with paired, collision-checked reverse or
scramble decoys, and emits N-terminal methionine-excised forms by default
(`digest.n_term_met_excision = true`). `peptidoforms` expands modifications and
charges. `predict-frag` builds the fragment library with either the native model
or MS2PIP, and retention times with the native model or DeepLC. This path is
substantially less sensitive than the imported library; see the benchmark.

### The Python sidecars

`scripts/` holds eleven Python programs. Seven are engine-invoked workers, called
by the relevant stage over a positional file contract (input Parquet in, output
Parquet out): MS2PIP (`ms2pip_worker.py`), DeepLC (`deeplc_worker.py`), the
DeepLC fine-tune (`deeplc_finetune.py`), mokapot (`mokapot_worker.py`), the
PyTorch rescorer (`nn_rescore_worker.py`), the entrapment rescorer
(`entrapment_worker.py`), and match-between-runs (`mbr_worker.py`). The other
four are run by hand for the imported-library recipe above:
`import_diann_lib.py`, `make_reverse_decoys.py`, `make_shift_decoys.py`, and
`augment_library.py`.

Neither the DeepLC fine-tune nor the PyTorch rescorer is bit-reproducible. The
rescorer seeds NumPy and PyTorch but its training kernels are not guaranteed
deterministic (`MUMDIA_NN_SEEDS > 1` ensembles seeds as mitigation); the DeepLC
fine-tune is seeded from `rng_seed` (`deeplc_finetune.py --seed`, which seeds both
numpy and torch), but torch's training kernels are not guaranteed bit-for-bit
reproducible. Identification counts therefore vary slightly between runs
of the same configuration, which matters when comparing small gains.

## Output

A `run` writes into `--out-dir`:

| file | use it for |
|---|---|
| `peptides.tsv` | reading identified precursors; one row per `(peptidoform, charge)` |
| `proteins.tsv` | reading identified protein groups |
| `peptide_quant.parquet` | peptide and precursor quantities for analysis |
| `protein_group_quant.parquet` | protein-group quantities for analysis |
| `fragment_quant.parquet` | per-fragment areas, for ion-level LFQ |
| `psms_scored.parquet` | the scored PSM table with every q column; the analysis unit |
| `psms_competed.parquet`, `features.parquet` | the feature matrix, before and after competition |
| `run.pin` | Percolator-style text export of `features.parquet`, for external tools. NOT what the rescorer reads: `rescore` builds its own PIN from the competed table. Off by default (`features.emit_pin`) |
| `psms_extracted.parquet`, `chromatograms.parquet` | extraction results and the extracted traces |
| `run_windows.parquet`, `cal.json` | per-run RT calibration and windows |
| `seed_psms.parquet` | the broad calibration search, not final identifications |
| `spectra/` | the normalized `spectra_ms1`, `spectra_ms2`, `isolation_windows`, `ms2_to_ms1` tables |
| `peptides.parquet`, `peptidoforms.parquet`, `fragment_library_*.parquet` | FASTA mode only: the digest and the library built from it |
| `fragment_library_precursors_ft.parquet` | written when DeepLC fine-tuning runs |
| `manifest.json` | provenance: engine version, the git commit and date it was built from, the command line, hashes of the run inputs, resolved config and its hash, model identities, per-artifact hashes |
| `<artifact>.report.json` | per-artifact schema version, row count, hash, resolved parameters, model identity, elapsed time |
| `candidate_audit.parquet` | written only under `extract.emit_candidate_audit = true`: the per-candidate identification-loss ladder |

Use the Parquet tables for analysis. TSV values are formatted for reading:
`q_value` is printed to six decimals, `score` to four, and `quantity` to one.

The `cal.json` retention-time residuals are in-sample: the calibration is fitted
and the window derived from the same anchor points. Treat them as fit
diagnostics, never as error estimates or as a way to rank two RT models.

### `peptides.tsv`

One row per confident precursor, deduplicated on `(peptidoform, charge)` keeping
the best q. Targets only, filtered on `peptide_q_value <= --q` (default 0.01).

| column | source | notes |
|---|---|---|
| `precursor` | `peptidoform` | ProForma peptidoform, modifications included |
| `stripped_sequence` | `strip(peptidoform)` | residues only; drops a `DECOY_` prefix and every bracketed or parenthesised modification block |
| `charge` | `charge` | precursor charge |
| `protein` | `protein` | the PSM's protein string, not the inferred group |
| `q_value` | `peptide_q_value` | **base-peptide** q, six decimals |
| `score` | `score` | rescorer score, four decimals |
| `quantity` | `peptide_quant.parquet` joined on `(peptidoform, charge)` | empty when the precursor was not quantifiable or no `--peptide-quant` was passed |

The row unit and the filter column deliberately disagree, and this is the single
most common misreading of MuMDIA output: rows are precursors
`(peptidoform, charge)` while the threshold is applied to `peptide_q_value`,
which is a base-peptide q. A row count is therefore **not** a
precursor-q-controlled count. The stage logs both numbers, `precursors` and
`stripped_sequences`; quote whichever you mean and name it.

### `proteins.tsv`

One row per confident protein group, deduplicated keeping the best q. Targets
with a non-empty group, filtered on `pg_q_value <= --q`.

| column | source | notes |
|---|---|---|
| `protein_group` | `protein_group` | accession set as grouped by rescore |
| `q_value` | `pg_q_value` | protein-group q, six decimals |
| `quantity` | `protein_group_quant.parquet` joined on `protein_group` | empty when not quantifiable or no `--protein-quant` was passed |

Identification and quantifiability are distinct. An accepted identification whose
signal cannot support a quantity keeps its row and gets a null quantity plus a
status, not an abundance of zero.

## FDR and q-value units

MuMDIA estimates false discovery rates by paired target-decoy competition and
writes several q columns that control different row units. They are not
interchangeable, and a count is uninterpretable without naming which one selected
it.

| column | row unit it controls |
|---|---|
| `q_value`, `experiment_psm_q` | pooled PSM, across every table given to `rescore` |
| `run_psm_q` | PSM within one run; the correct per-file unit under a pooled rescore |
| `precursor_q` | peptidoform plus charge, but only under `compete.group_by = peptidoform_charge` |
| `peptide_q_value` | base (stripped) peptide; what `peptides.tsv` is filtered on |
| `pg_q_value` | protein-accession-set group; what `proteins.tsv` is filtered on |

Two traps:

- **`precursor_q` is a precursor unit only under
  `compete.group_by = peptidoform_charge`.** The default competition key
  `base_peptide` (renamed from `precursor`, which it was not) groups on the
  stripped sequence, so every charge
  and every modification variant of one peptide collapses to a single winner
  before FDR. Under that key `precursor_q` therefore counts base peptides
  (measured 1.000 precursors per peptide, against 1.174 with
  `peptidoform_charge`).
- **The grouped columns are written only to each group's single winning row.**
  `peptide_q_value`, `precursor_q`, and `pg_q_value` are 1.0 on every loser.
  Under an experiment-wide rescore the grouping is experiment-wide, so a per-run
  count taken from them is diluted by roughly one over the number of runs and is
  meaningless. Use `run_psm_q` for a per-file count there.

Pooling more runs does not tighten q. The estimator is
`q = (decoys + 1) / max(1, targets)`, which is scale-invariant under replicating
the population; the only pool-size term is the `+1` pseudocount, which makes a
larger pool marginally looser. Do not attribute per-run count changes to pool
size.

These estimates depend on a valid, exchangeable decoy population, and the engine
rejects malformed or decoy-free libraries. Entrapment with a foreign-proteome
spike-in is available as an empirical cross-check and should be part of
validation before a new sensitivity setting is promoted. Details in
[`docs/11_compete_rescore_fdr.md`](docs/11_compete_rescore_fdr.md) and
[`docs/15_data_dictionary.md`](docs/15_data_dictionary.md).

## Benchmark

### Identification

Single-run validation targets on the ProteomeXchange E. coli AIF file
`LFQ_Orbitrap_AIF_Ecoli_01.mzML`, with an imported DIA-NN E. coli library whose
iRT has been fine-tuned, the Extended feature set, and converted MS2 spectra
capped at 300 peaks. **Row unit: `(peptidoform, charge)` rows in `peptides.tsv`.
Selection column: `peptide_q_value <= 0.01`.** The measured decoy fraction of the
accepted set was 0.98 to 0.99% at the 1% threshold, which is a target-decoy
sanity check and not an independent empirical null.

| configuration | rescorer | rows |
|---|---|---|
| native FASTA digest, zero dependencies | `native_tda` | about 1,213 |
| imported library, per-run DeepLC fine-tune | `native_tda` | 10,847 |
| imported library, per-run DeepLC fine-tune | `nn_torch` | 10,914 |

Both re-measured 2026-08-28 on the augmented library under the current defaults, at
an extraction gate of 0.2 and an empirical decoy fraction of 0.0098. The earlier
figures for this row -- about 9,300 to 9,500 for `native_tda` and about 10,300 for
`nn_torch` -- were taken on the raw imported library before the augmented tables,
`apex_evidence_rank` and the paired CV folds.

The rescorer was previously the dominant lever, at +8.5% for `nn_torch` over
`native_tda`; that gap is now +0.6% (10,914 against 10,847), so it no longer is.
On the
same feature set. The gate optimum inverts by rescorer, so tune the extraction
gate and the classifier together. Source:
[`docs/18_findings_and_decisions.md`](docs/18_findings_and_decisions.md) finding
A1. These are historical regression targets, not CI assertions; no commit is
recorded for the gate sweep itself.

Against DIA-NN 2.2.0 run library-free with `--reanalyse` on the same file, which
reported 11,817 stripped peptides at 1%, this workflow reached 90.4 to 91.6%
depending on the DeepLC version and the retention-time window sizing. **Row unit
here: stripped peptides at 1%.** Measured 2026-08-24, commit `b39769e`.

The 300-peak conversion cap belongs to that chimeric AIF acquisition and is not a
default. On a 50-window Orbitrap DIA run the same cap discarded 78.6% of all MS2
peaks and cost 60% of the peptides (25,425 capped against 63,237 uncapped) at an
unchanged 0.99% empirical decoy fraction, so the loss is sensitivity and not a
loosened threshold. Both `convert` and `run` default `--top-peaks-ms2` to `0`
(uncapped); leave it there unless a sweep on your own acquisition says otherwise.
[`docs/04_convert.md`](docs/04_convert.md), section "Choosing
`--top-peaks-ms2`", is the full treatment.

### Quantification

Measured on the ProteoBench Astral HYE set at commit `83ba81d`. Setting
`quant.fixed_scan_halfwidth` and `quant.fixed_window_s` together, which integrate
a fixed window centred on the identification apex instead of the descent-walk
bounds, moved median absolute epsilon from 0.273 to 0.195 and CV from 0.175 to
0.107. Both options default to off, so an existing configuration produces
bit-identical results. Promotion to a default is gated on entrapment, which has
not been run.

The full recorded ProteoBench comparisons live in
[`bench/README.md`](bench/README.md), together with the scoring scripts. **Row
unit: ions at `min_obs = 3`, that is quantified in at least three of the six
runs.** DIA-NN 2.2.0 was run library-free with `--reanalyse` on the same files.

| set | tool | features | median abs eps (global) | CV median |
|---|---|---|---|---|
| Astral `LFQ_Astral_DIA_15min_50ng` | MuMDIA | 100,528 | 0.176 | 0.105 |
| Astral `LFQ_Astral_DIA_15min_50ng` | DIA-NN 2.2.0 | 115,045 | 0.203 | 0.141 |
| AIF HYE `LFQ_Orbitrap_AIF_Condition_{A,B}` | MuMDIA | 70,689 | 0.154 | 0.314 |
| AIF HYE `LFQ_Orbitrap_AIF_Condition_{A,B}` | DIA-NN 2.2.0 | 89,800 | 0.234 | 0.182 |

Read those with one caveat that matters: the MuMDIA figures were produced by the
second-pass workflow described in
[`docs/22_release_plan.md`](docs/22_release_plan.md) WP7, which is prototype shell
code on the benchmark machine and is **not in the engine**. This release does not
reproduce them on its own. Accuracy is competitive or better, completeness trails
DIA-NN by 13% on Astral and 21% on AIF, and per-ion precision is comparable on
Astral but about 1.7 times worse on AIF, where the all-ion isolation window leaves
interference the current fragment selection does not remove.

Per-stage wall clock and artifact sizes for a reference run are in
[`bench/README.md`](bench/README.md): one 1.94 GB AIF file takes about 85 minutes
and writes 13.1 GB of artifacts, of which rescoring is 80% of the time. Peak
memory is still not measured, and neither is the six-file experiment profile
([`docs/22_release_plan.md`](docs/22_release_plan.md), WP5).

## Status of experimental features

These exist in the code and are measured, but are not enabled by default. They
are listed so the documentation cannot imply a capability that is not there.

| feature | status |
|---|---|
| model-visible top-K peaks | `extract.retain_top_peaks > 1` writes diagnostic peak alternatives only. They do not become feature or rescore rows. The selected apex was historically strongest only about 48 to 52% of the time while the correct peak was in the top five about 86 to 88%, so promoting alternatives is plausible future work, not a present capability |
| adaptive RT windows | not enabled by default |
| held-out RT window sizing | `rt_im_train.window_holdout_frac`, default off. Gained 1.1% of peptides with DeepLC 4.1.0 at an unchanged 0.98% decoy fraction, but lost 1.5% with the overfitting 4.0.0a2 model, so it interacts with retention-time model quality |
| `compete.group_by = peptidoform_charge` | accepted and correct, but not the shipped default. It is **required**, not optional, for a PTM or modification search: under the default key the modified form is deleted whenever an unmodified or alkylated sibling scores higher, which is usually. Measured on a modification-rich library, the default key deleted 880,464 of 1,890,239 extracted candidates (46.6%) while `peptidoform_charge` removed none |
| MBR tiers | `mbr.strategy` distinguishes only none from not-none. `mbr.rt_window_s`, `mbr.decoy_transfer`, and `mbr.requant_all` are accepted by the config but not wired; setting them changes nothing, and the engine warns that it did nothing |
| fixed-window and library-ranked quantification | `quant.fragment_selection`, `fixed_scan_halfwidth`, `fixed_window_s`, and `baseline_subtract` all default to off pending entrapment validation |
| acquisition-specific peak caps | the shipped default is uncapped at both conversion entry points. A cap must come from a sweep on the acquisition it will be used on |
| percolator rescoring | declared in the config enum and rejected by validation |

Hard limits of this release:

- **mzML input only.** No vendor formats.
- **No ion mobility.** The pipeline is 3D; the ion-mobility columns exist in the
  artifacts and are always null.
- **No wildcard or terminal variable modifications.** Both are rejected at
  peptidoform expansion.
- **Several files are one experiment by default.** `run` with several `--mzml`, like
  `run-experiment`, rescores them together; only one `--mzml` is a single-file search.
  Use `run-experiment` directly for a pooled multi-run
  search.
- **`run-experiment` does not write the TSV reports.** It never calls the report
  stage, so there is no `peptides.tsv` or `proteins.tsv` anywhere in its output
  tree. Take per-run counts from the split scored tables on `run_psm_q`, or
  invoke `mumdia report` yourself. It also overrides the configured
  `quant.q_filter` to gate per-run quantification on the pooled `q_value`, and
  warns when it does.
- **The Python sidecars are not covered by continuous integration.** CI builds
  and tests the engine on three platforms and runs an end-to-end smoke test on a
  generated fixture in native mode, but no job runs DeepLC, MS2PIP, mokapot, the
  PyTorch rescorer, or the MBR worker on data. A green test run is not sidecar
  validation.

## Hardware sizing

Only measured quantities are stated here; no RAM figure is published yet.

- The pooled rescore feature matrix is `n_psms x n_features x 4` bytes. Size
  rescore batches from that expression rather than from a guessed total. Batching
  is statistically free: `rescore --competed` accepts many tables, stamps each
  PSM with the index of the table it came from, and computes a per-source
  `run_psm_q` alongside the pooled `q_value`, so sub-batching never costs per-run
  FDR. Batch only to fit RAM.
- Pooled rescore scales linearly, measured 0.834 ms/PSM on the streaming backend.
- `MUMDIA_NN_STREAM_GB` (default 4) selects the PyTorch rescorer's backend. A
  feature matrix marginally over the threshold silently falls to the much slower
  disk-backed streaming memmap: a 4.31 GB matrix against the 4.00 GB default took
  the slow path. Compute the matrix size up front and raise
  `MUMDIA_NN_STREAM_GB` if the RAM is available.
- `--threads N` bounds the engine's thread pool and is forwarded to the sidecars.
  More is not always better there: the PyTorch rescore worker measured faster on
  8 threads than on 32.

## Documentation

- [`docs/README.md`](docs/README.md): the developer guide. A per-subsystem
  reference grounded in the current code, with `file:line` citations. Start
  there. [`docs/19_getting_started.md`](docs/19_getting_started.md) has
  copy-pasteable end-to-end runs, and
  [`docs/17_troubleshooting.md`](docs/17_troubleshooting.md) is a
  symptom-to-cause table for the quiet failure modes, worth reading before
  something breaks rather than after.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): the build, the checks a change must pass,
  and the project rules that are easy to break without noticing.
- [`SECURITY.md`](SECURITY.md): threat model and how to report a vulnerability
  privately.
- [`bench/README.md`](bench/README.md): the quantitative benchmarks, the recorded
  ProteoBench results with their row and q units, and how to reproduce them.
- [`CHANGELOG.md`](CHANGELOG.md): what changed, in Keep a Changelog format.

### Generated references

Two documents are generated from the code, and CI regenerates them and fails on a
difference. That pins everything the generator derives -- every flag, field, type,
default and enum value -- but not the hand-written prose around it, and not a
column the generator computes from a heuristic (`docs/24`'s benchmark-gated
marker reads doc comments, so it is as right as they are):

- [`docs/23_cli_reference.md`](docs/23_cli_reference.md): every subcommand's help,
  the four global flags, and which subcommands accept `--config`.
- [`docs/24_config_reference.md`](docs/24_config_reference.md): all 166
  configuration fields with their types, defaults and gating status, the 21
  enumerations, and every environment variable the engine and the sidecars read
  or set.

## Citation

A `CITATION.cff` and a Zenodo DOI are pending for the first tagged release. Until
they exist, cite the repository URL together with the commit or tag you used. No
author list is stated here because it is not derivable from the repository and
must not be guessed.

## License

Apache-2.0. See [LICENSE](LICENSE).

The engine is statically linked, so a distributed binary contains 173 Rust
crates. Their notices are reproduced in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), generated from `Cargo.lock`
and from the crates' own licence files; `sbom.cdx.json` is the machine-readable
form. Both ship in every release archive and in the container image, and both are
checked for staleness in CI. The extraction is mechanical: whether it satisfies a
given distribution's obligations is a question for whoever signs off the release.

## Attribution

Developed at [CompOmics](https://www.compomics.com), VIB-UGent Center for Medical
Biotechnology, Ghent University. This project is a ground-up reimplementation of
the earlier MuMDIA; the previous version is archived on the `legacy-python`
branch and tagged `legacy-python-v1`.

MuMDIA maintains a clean-room boundary against closed implementations. No
coefficient vector, intensity model, decoy mutation map, or constant table is
copied from DIA-NN or any other proprietary engine, and physical constants are
public-domain values with their provenance stated in the source. MuMDIA consumes
but does not ship or invoke a DIA-NN binary; users create imported libraries
under their own DIA-NN licence.
