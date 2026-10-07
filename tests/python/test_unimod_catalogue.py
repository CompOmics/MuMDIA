"""The Python copy of the modification catalogue must equal the engine's JSON."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import _unimod  # noqa: E402


def test_the_python_catalogue_equals_the_engine_catalogue():
    cat = json.loads(
        (ROOT / "rust/mumdia/crates/mumdia-core/src/modifications.json").read_text(encoding="utf-8")
    )
    want = tuple((m["name"], m["unimod"], m["mass"]) for m in cat["modifications"])
    assert _unimod.MODIFICATIONS == want, (
        "scripts/_unimod.py is out of date with modifications.json; regenerate it"
    )


def test_accessions_and_names_are_unique():
    names = [m[0] for m in _unimod.MODIFICATIONS]
    ids = [m[1] for m in _unimod.MODIFICATIONS]
    assert len(set(names)) == len(names)
    assert len(set(ids)) == len(ids)
