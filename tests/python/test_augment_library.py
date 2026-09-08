"""Contract tests for the id rules of `scripts/augment_library.py` (no engine needed).

The helper's pipeline runs the engine's digest and predict-frag, which these tests do
not; `assign_base_ids` is the pure rule that decides whether an added form joins an
existing peptide or founds a new one (docs/29 #12).
"""

from __future__ import annotations

import importlib.util
import sys

from conftest import SCRIPTS


def _load():
    # The helper imports the shared writer `_lib_io` from its own directory.
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "mumdia_augment_library", SCRIPTS / "augment_library.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_added_form_of_an_existing_peptide_keeps_that_peptides_base_id():
    """`--match-level peptidoform_charge` adds missing charge states and modforms of
    sequences the imported library already has. Offsetting every added base id split
    such a peptide across two competition groups and two CV folds; the added forms
    must reuse the imported id, and only new sequences get fresh ids, one per sequence.
    """
    m = _load()
    imported = {"PEPTIDEK": 5, "SAMPLER": 9}
    added = ["PEPTIDEK", "NEWSEQK", "SAMPLER", "NEWSEQK", "OTHERK", "PEPTIDEK"]
    ids = m.assign_base_ids(imported, added, next_id=12)
    assert ids == [5, 12, 9, 12, 13, 5]
    # Nothing added: nothing allocated.
    assert m.assign_base_ids(imported, [], next_id=12) == []
