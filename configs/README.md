# Example configurations

Starting points, in increasing order of what they need installed. Copy one,
edit it, and pass it with `--config`. Every field not mentioned keeps its default;
`mumdia` rejects unknown keys, so a typo fails at load rather than being ignored.

| file | needs | what it selects |
|---|---|---|
| `examples/native.json` | nothing beyond the binary | Extended features, the DIA apex settings, and the native predictors and `native_tda` rescorer. Runs on a machine with no Python at all. |
| `examples/fasta-sidecars.json` | MS2PIP, DeepLC, nn_torch | Digest a FASTA, predict fragment intensities with MS2PIP and retention time with DeepLC, rescore with `nn_torch`. One environment (`env/mumdia-deeplc.yml`) serves all three. |
| `examples/diann-library.json` | DeepLC, PyTorch | Consume an imported DIA-NN library, re-predict its retention times once with the DeepLC base model (`rt_im_train.library_irt = auto`; DeepLC >= 4.1.1) and calibrate them per run, rescore with the PyTorch NN. This is the highest-sensitivity workflow measured so far (`docs/20_sensitivity_and_quantification_playbook.md`). |
| `examples/fasta-peptdeep.json` | AlphaPeptDeep, DeepLC, PyTorch | The FASTA workflow with AlphaPeptDeep fragment intensities (`env/mumdia-peptdeep.yml`) in place of MS2PIP. |
| `examples/diapasef.json` | AlphaPeptDeep, DeepLC, IM2Deep, PyTorch | diaPASEF (timsTOF `.d`, read natively): the FASTA workflow with AlphaPeptDeep at the timsTOF settings, IM2Deep 1/K0, the 1/K0 gate and features, retrace with the re-pick, the second pass and the cross-run quant selectors. Every TIMS key it sets is off by default. Keep its `rt_window_multiplier` of 1.0: the 1.5 of the Orbitrap examples costs peptides on diaPASEF (`docs/35_diapasef.md` section 9). |

Check any of them before a long run:

```
mumdia doctor --config configs/examples/diann-library.json
```

`doctor` resolves the interpreters exactly as a run does, reports the versions it
finds, and fails with the reason if the configuration cannot run.

## `"auto"` and the sidecar interpreters

MuMDIA launches its ML predictors and rescorers as Python workers, so it needs to
know which interpreter to use for each. The examples say `"auto"`, which means:
find one that can import what that worker imports. The search order is

1. the role's own environment variable: `MUMDIA_PYTHON_RESCORE`,
   `MUMDIA_PYTHON_DEEPLC`, `MUMDIA_PYTHON_MS2PIP`, `MUMDIA_PYTHON_PEPTDEEP`,
   `MUMDIA_PYTHON_IM2DEEP`, `MUMDIA_PYTHON_MBR`;
2. `MUMDIA_PYTHON`, for all roles at once;
3. an activated environment, via `CONDA_PREFIX` or `VIRTUAL_ENV`;
4. `python3`, then `python`, on `PATH`.

A candidate is accepted only if it can import that role's modules, so `auto`
cannot quietly select a Python without torch and defer the failure to hour three
of a run. Naming an absolute path instead still works and is never
second-guessed; that is what the Docker configs do, since the interpreters in the
image are at fixed locations.

The environments themselves are specified under `env/`:
`env/mumdia-rescore.yml` for the mokapot path, `env/mumdia-deeplc.yml` for the
DeepLC and IM2Deep workers and `env/mumdia-peptdeep.yml` for AlphaPeptDeep. DeepLC must be 4.5.0 or newer: the 4.0.0a2 preview overfits
per-run fine-tuning badly enough to invert retention-time model rankings, the
multi-head calibration needs 4.4.0 and the default projection cache 4.5.0, so an
older version changes results and not only speed. `doctor` fails on one.

## Where the worker scripts have to be

`predict_frag.sidecar_script_dir` names the directory holding the Python workers.

An ABSOLUTE value is taken as given. A relative value is resolved against, in order:

1. the directory the config file itself lives in,
2. `scripts/` next to the `mumdia` executable, which is the layout of the release
   archive,
3. the current working directory, with a warning.

So `"scripts"` works both from a git checkout and from an unpacked release, and a
config that sits next to its own `scripts/` directory keeps working when invoked
from elsewhere.

The working directory is deliberately LAST. It used to be first, and the shipped
default is the relative `"scripts"`, so unpacking an untrusted dataset archive that
happened to contain a `scripts/` directory with a worker file in it and running
`mumdia` from inside it would have executed those workers. That needs no hostile
configuration -- which `SECURITY.md` treats as trusted, like a shell script -- only
an untrusted input directory. Prefer an absolute path, or keep the scripts beside
the config.

## The configurations shipped in the container image

`docker/config.dia.json` and `docker/config.diann-lib.json` are the same two
workflows wired to the interpreters inside the image
(`/opt/conda/envs/{rescore,deeplc}/bin/python`) and to `/opt/mumdia/scripts`.
They are baked into the image, not meant to be edited in place.
