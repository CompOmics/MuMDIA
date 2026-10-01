"""The column sets `ci/check_smoke.py` checks the feature storage with are the Rust ones.

The smoke check keeps its own copy of two constants of
`rust/mumdia/crates/mumdia/src/stages/features.rs`: `F64_FEATURE_COLUMNS`, the feature
columns stored as float64, and `NON_FEATURE_COLUMNS`, the bookkeeping columns that are
not features at all. A copy that drifted would either fail the smoke on a correct table
(a bookkeeping column read as a feature of the wrong width) or stop checking a column
the engine changed. The copies are deliberate, as the frozen schema versions are, so a
change to either Rust list must be made in both places; this test makes that visible.
"""

from __future__ import annotations

import importlib.util
import re

import pytest

from conftest import ROOT

FEATURES_RS = ROOT / "rust" / "mumdia" / "crates" / "mumdia" / "src" / "stages" / "features.rs"
CHECK_SMOKE = ROOT / "ci" / "check_smoke.py"


def rust_str_list(source: str, name: str) -> set[str]:
    """The string literals of `pub const <name>: &[&str] = &[ ... ];` in `source`."""
    m = re.search(rf"pub const {name}: &\[&str\] = &\[(.*?)\];", source, re.S)
    assert m, f"{name} not found in {FEATURES_RS}"
    items = re.findall(r'"([^"]*)"', m.group(1))
    assert items, f"{name} is empty in {FEATURES_RS}"
    assert len(items) == len(set(items)), f"{name} lists a column twice"
    return set(items)


@pytest.fixture(scope="module")
def check_smoke():
    pytest.importorskip("pyarrow")
    spec = importlib.util.spec_from_file_location("check_smoke", CHECK_SMOKE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def features_rs() -> str:
    return FEATURES_RS.read_text(encoding="utf-8")


def test_f64_feature_columns_match_the_rust_constant(check_smoke, features_rs):
    assert check_smoke.F64_FEATURE_COLUMNS == rust_str_list(features_rs, "F64_FEATURE_COLUMNS")


def test_non_feature_columns_match_the_rust_constant(check_smoke, features_rs):
    assert check_smoke.NON_FEATURE_COLUMNS == rust_str_list(features_rs, "NON_FEATURE_COLUMNS")


def test_the_f64_bookkeeping_columns_are_bookkeeping_columns(check_smoke):
    assert check_smoke.F64_BOOKKEEPING_COLUMNS <= check_smoke.NON_FEATURE_COLUMNS
    assert not check_smoke.F64_FEATURE_COLUMNS & check_smoke.NON_FEATURE_COLUMNS
