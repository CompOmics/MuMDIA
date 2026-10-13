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


def test_added_entries_get_a_null_library_im_when_the_import_has_one(tmp_path):
    """The merge takes the imported table's columns from both halves. An imported
    library with `predicted_im` and added entries without it must still concatenate,
    and the added rows carry no 1/K0 rather than a made-up one."""
    import pandas as pd

    m = _load()
    imported = pd.DataFrame({"candidate_id": [0, 1], "predicted_im": [0.9, None]})
    added = pd.DataFrame({"candidate_id": [7]})
    out = m.align_library_im(imported, added)
    assert out["predicted_im"].isna().all() and str(out["predicted_im"].dtype) == "float64"
    merged = pd.concat([imported[list(imported.columns)], out[list(imported.columns)]])
    assert merged["predicted_im"].notna().sum() == 1
    # Without 1/K0 in the import nothing is added, so the merged table stays version 1.
    plain = pd.DataFrame({"candidate_id": [0]})
    assert "predicted_im" not in m.align_library_im(plain, added).columns


def test_the_internal_predict_frag_call_never_predicts_im(tmp_path):
    import json

    m = _load()
    assert m.predict_frag_config(None, str(tmp_path)) is None
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"predict_frag": {"top_n_fragments": 12}}))
    assert m.predict_frag_config(str(plain), str(tmp_path)) == str(plain)
    im = tmp_path / "im.json"
    im.write_text(json.dumps({"predict_frag": {"im_predictor": "im2deep",
                                               "top_n_fragments": 12}}))
    out = m.predict_frag_config(str(im), str(tmp_path / ""))
    cfg = json.loads(open(out, encoding="utf-8").read())
    assert cfg["predict_frag"]["im_predictor"] == "none"
    assert cfg["predict_frag"]["top_n_fragments"] == 12
    assert cfg["predict_frag"]["sidecar_script_dir"] == str(SCRIPTS)
