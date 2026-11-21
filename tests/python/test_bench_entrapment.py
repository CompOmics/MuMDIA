"""bench/make_entrapment_fasta.py::fdp_estimates, the entrapment FDP estimates every bench
script prints (entrapment_fdp.py, fs_lib.evaluate, ps_agg.py, ps_rtfree_agg.py).

Pinned on the docs/08 section 4d counts: both arms ran rescore.classifier = entrapment, so the
real-only value is the engine's 1% cut restated; the combined estimate and the lower bound are
what the accepted target list as a whole carries.
"""
from __future__ import annotations

import importlib.util

import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location("make_entrapment_fasta",
                                              ROOT / "bench" / "make_entrapment_fasta.py")
mef = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mef)

RATIO = 0.560632  # real / entrapment target candidates of the AIF entrapment library, 1 / 1.7837


@pytest.mark.parametrize("n_real, n_entrap, real_only, combined, lower", [
    (7879, 138, 0.995, 2.686, 1.721),  # baseline
    (8216, 144, 0.995, 2.688, 1.722),  # multi-head calibration
])
def test_docs08_counts(n_real, n_entrap, real_only, combined, lower):
    est = mef.fdp_estimates(n_real, n_entrap, RATIO)
    assert round(100 * est["real_only"], 3) == real_only
    assert round(100 * est["combined"], 3) == combined
    assert round(100 * est["lower_bound"], 3) == lower


@pytest.mark.parametrize("ratio", [None, 0, -1.0])
def test_ratio_is_required(ratio):
    with pytest.raises(ValueError):
        mef.fdp_estimates(100, 1, ratio)
