"""Helpers shared by the DeepLC workers, the IM2Deep worker and the shard recipe.

`deeplc_worker.py` and `deeplc_finetune.py` used to carry these as character-for-character
copies, because neither can import the other: `deeplc_finetune.py`'s module body pins the
OpenMP pools and imports psm_utils. `mh_shard_predict.py` copied the sequence rules for the
same reason (its `uniq` runs without DeepLC), and `im2deep_worker.py` the version parse.

Standard library only at import (pyarrow is imported inside the two Arrow helpers, the
DeepLC internals inside `load_base_model`), so importing it has no load-order constraint of
its own; the workers import it after their torch-backed package all the same, keeping that
package's import first (see `deeplc_worker.py`). Python puts a script's own directory on
`sys.path`, so `from _deeplc_common import ...` resolves wherever the worker is invoked
from, as `_torch_threads` does.
"""

import contextlib
import io
import os
import sys

# The engine's floor (`mumdia_core::constants::MIN_DEEPLC_VERSION`), repeated here because
# `mumdia doctor` cannot see a version that changes under its feet. The base-model
# calibration is only sound from 4.1.1 on (4.0.0a2 memorised anchors: in-sample 15.9 s
# against held-out 195 s residuals), the multi-head calibration needs 4.4.0, and the
# projection cache, on by default, the factored prediction matrix of 4.5.0.
MIN_DEEPLC = (4, 5, 0)

# A bracketed modification, and the dash of a terminal group with it: `[Acetyl]-PEPK` strips
# to `PEPK`. Without the dash the stripped base was `-PEPK`, which is not a standard
# sequence, so every N-terminally modified row was left out of the prediction and kept a
# retention time on another scale.
MOD_RE = r"-?\[[^\]]*\]-?"
# The decoy marker, stripped before prediction so a decoy is predicted on its underlying
# sequence (`deeplc_finetune.base_pf`). Anchored at the start: `replace_substring` would
# strip every occurrence.
DECOY_PREFIX_RE = r"^DECOY_"
# A standard sequence once modifications are stripped. `*` rather than `+` because
# `is_std("")` is true: a bare "DECOY_" row is predicted.
STD_FULL_RE = r"^[ACDEFGHIKLMNPQRSTVWY]*$"


def version_tuple(raw):
    """The leading digits of the first three dot-separated parts, missing parts as 0."""
    parts = []
    for piece in str(raw).split(".")[:3]:
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (3 - len(parts)))


def require_version(module, minimum):
    """Exit when the imported `module` is older than `minimum`, the engine's floor for it."""
    raw = getattr(module, "__version__", None)
    if raw is None:
        try:
            import importlib.metadata as _m

            raw = _m.version(module.__name__)
        except Exception:  # pragma: no cover
            raw = ""
    if version_tuple(raw) < tuple(minimum):
        name = module.__name__
        sys.exit("%s %s is older than the required %d.%d.%d (pip install '%s>=%d.%d.%d')"
                 % (name, raw or "unknown", *minimum, name, *minimum))


def load_base_model():
    """The DeepLC base model, loaded once, or None to let every call load its own.

    `deeplc.predict(batch)` loads the checkpoint from disk on every call, and
    `predict_and_calibrate` twice (once to size its head source, once to predict), so a
    whole-library prediction in 100,000-peptidoform chunks read the model 50 to 100 times.
    `load_model` hands a module instance back unchanged, and prediction runs in eval mode
    under `no_grad`, so passing the one instance gives the same numbers (measured
    bit-identical on 3,002 peptidoforms, DeepLC 4.5.0). The helpers are private DeepLC API
    (present in 4.4.0 and 4.5.0) and the engine sets no DeepLC ceiling, so a release that
    moves, renames or re-signatures them returns None here, with a warning, and every call
    loads its own model as before: slower, same numbers.
    """
    try:
        from deeplc import _model_ops
        from deeplc.core import DEFAULT_MODEL

        return _model_ops.load_model(DEFAULT_MODEL)
    except (ImportError, AttributeError, TypeError) as exc:
        print("WARNING: could not load the DeepLC base model once (%s: %s); every "
              "prediction call loads its own" % (type(exc).__name__, exc), flush=True)
        return None


class _DropBlankProgress(io.TextIOBase):
    """`sys.stdout` with DeepLC's empty progress writes removed.

    DeepLC's prediction loop writes a carriage-return progress indicator that renders as
    nothing when stdout is not a terminal, so every update arrives as a bare `\r\n`.
    Measured on a real library build: 5,697 blank lines from a 2.9M-peptide prediction,
    99% of the entire log. The engine INHERITS a worker's stdout rather than capturing it
    (`sidecar.rs::run_worker`, deliberately, so long-running progress reaches the user
    live), so those lines land in the terminal, in any redirected log file, and in the
    desktop application's run log, where they push the real messages out of view.

    Only segments that are empty once carriage returns and whitespace are stripped get
    dropped. Anything DeepLC actually says still comes through, in order, and stderr --
    where a traceback goes -- is not touched at all. Set `MUMDIA_DEEPLC_RAW_OUTPUT=1` to
    disable the filter when debugging the worker itself.

    `\r` counts as a terminator as well as `\n`: a progress writer that never emits a
    newline would otherwise accumulate in the buffer for the whole run.
    """

    def __init__(self, inner):
        self._inner = inner
        self._buf = ""

    def write(self, s):
        self._buf += s
        while True:
            i = min(
                (p for p in (self._buf.find("\n"), self._buf.find("\r")) if p >= 0),
                default=-1,
            )
            if i < 0:
                break
            line, self._buf = self._buf[:i], self._buf[i + 1 :]
            if line.strip():
                self._inner.write(line + "\n")
                self._inner.flush()
        return len(s)

    def flush(self):
        self._inner.flush()

    def close(self):
        # Whatever is left had no terminator; emit it if it says anything.
        if self._buf.strip():
            self._inner.write(self._buf)
        self._buf = ""
        self._inner.flush()


@contextlib.contextmanager
def quiet_deeplc_progress():
    """Install the filter for the duration of a block, unless disabled by env."""
    if os.environ.get("MUMDIA_DEEPLC_RAW_OUTPUT", "").strip() not in ("", "0"):
        yield
        return
    original = sys.stdout
    proxy = _DropBlankProgress(original)
    sys.stdout = proxy
    try:
        yield
    finally:
        proxy.close()
        sys.stdout = original


def library_bases(peptidoforms):
    """The DECOY_-stripped sequence of every library row, as `deeplc_finetune.base_pf`
    gives it: a leading marker only."""
    import pyarrow.compute as pc

    return pc.replace_substring_regex(peptidoforms, pattern=DECOY_PREFIX_RE, replacement="")


def unique_standard_bases(bases):
    """Unique stripped sequences in first-occurrence order, standard residues only.

    What the per-row loop `if b not in seen and is_std(pf)` produced, computed in Arrow so
    a library of 1e7 to 1e8 rows does not become that many Python strings and a set. The
    per-chunk uniques are widened to large_string before they are joined: at 1e8 rows they
    can pass the 2 GiB offset limit of `string`.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    parts = [pc.unique(chunk).cast(pa.large_string()) for chunk in bases.chunks]
    if not parts:
        return pa.array([], pa.large_string())
    uniq = parts[0] if len(parts) == 1 else pc.unique(pa.concat_arrays(parts))
    stripped = pc.replace_substring_regex(uniq, pattern=MOD_RE, replacement="")
    return uniq.filter(pc.match_substring_regex(stripped, pattern=STD_FULL_RE))
