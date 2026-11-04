"""`scripts/prune_outputs.py`: the retention tiers on a synthetic output directory.

Standard library only: the script reads JSON and walks the directory, so these tests need
neither pyarrow nor the engine.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prune_outputs", ROOT / "scripts" / "prune_outputs.py")
prune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prune)

RUN_FILES = [
    "cal.json",
    "scored.parquet",
    "peptide_quant.parquet",
    "peptide_quant.parquet.report.json",
    "protein_group_quant.parquet",
    "fragment_quant.parquet",
    "run_windows.parquet",
    "run_windows.parquet.report.json",
    "chromatograms.parquet",
    "features.parquet",
    "features.parquet.schema.json",
    "psms_extracted.parquet",
    "psms_extracted.parquet.peaks.parquet",
    "seed_psms.parquet",
    "seed_psms.parquet.masscal.json",
    "fragment_library_precursors_multihead.parquet",
    "fragment_library_precursors_multihead.parquet.summary.json",
    "rt_bootstrap/run_windows.parquet",
    "rt_bootstrap/cal.json",
    "spectra/spectra_ms2.parquet",
    "spectra/spectra_ms2.parquet.report.json",
]
SECOND_FILES = ["chromatograms.parquet", "features.parquet", "psms_extracted.parquet", "cal.json"]


def _write(p: Path, n: int = 10) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * n)


def make_experiment(tmp: Path, *, second_pass: bool, vendor: bool = False) -> Path:
    out = tmp / "out"
    runs = ["r0", "r1"]
    for r in runs:
        for f in RUN_FILES:
            _write(out / r / f)
        os.link(out / r / "features.parquet", out / r / "psms_competed.parquet")
        if second_pass:
            for f in SECOND_FILES:
                _write(out / "second_pass" / r / f, 3)
    if second_pass:
        _write(out / "second_pass" / "library" / "lib_precursors.parquet", 5)
        _write(out / "second_pass" / "library" / "empirical_library.json", 5)
    _write(out / "first_pass" / "scored_combined.parquet")
    _write(out / "first_pass" / "scored_combined.parquet.report.json")
    (out / "sidecar_work").mkdir()
    _write(out / "sidecar_work" / "handoff.npy")
    for f in ["scored_combined.parquet", "lfq_maxlfq.parquet", "peptides.tsv", "proteins.tsv"]:
        _write(out / f)
    _write(out / "my_notes.txt")
    inputs = {}
    cli = ["mumdia", "run", "--out-dir", "out"]
    for i, r in enumerate(runs):
        stem = f"in/run{i}"
        if vendor:
            _write(tmp / f"{stem}.raw")
            cli += ["--mzml", f"{stem}.raw"]
        else:
            cli += ["--mzml", f"{stem}.mzML"]
        _write(tmp / f"{stem}.mzML", 100)
        inputs[f"mzml[{i}]"] = {"path": f"{stem}.mzML", "content_hash": f"h{i}"}
    manifest = {
        "artifacts": {
            "scored_first_pass": {"path": "out/first_pass/scored_combined.parquet", "content_hash": "fp"},
        },
        "cli_args": cli,
        "inputs": inputs,
        "experiment": {
            "runs": runs,
            "second_pass": {"runs": ["out/second_pass/r0", "out/second_pass/r1"]} if second_pass else None,
        },
    }
    (out / "experiment_manifest.json").write_text(json.dumps(manifest))
    return out


def removed(p: dict) -> set:
    return {e["path"] for e in p["removed"]}


def test_refuses_without_manifest(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="did not finish"):
        prune.plan(tmp_path / "out", "results", with_mzml=False)


def test_all_removes_nothing(tmp_path):
    out = make_experiment(tmp_path, second_pass=True)
    assert prune.plan(out, "all", with_mzml=True)["removed"] == []


def test_second_pass_viewer_and_results(tmp_path):
    out = make_experiment(tmp_path, second_pass=True)
    v = prune.plan(out, "viewer", with_mzml=False)
    rv = removed(v)
    # First-pass evidence, the library copy, rt_bootstrap, first_pass/ and scratch go.
    for r in ("r0", "r1"):
        for f in ("run_windows.parquet", "chromatograms.parquet", "features.parquet",
                  "psms_competed.parquet", "psms_extracted.parquet",
                  "psms_extracted.parquet.peaks.parquet", "seed_psms.parquet",
                  "fragment_library_precursors_multihead.parquet",
                  "rt_bootstrap/run_windows.parquet"):
            assert f"{r}/{f}" in rv, f
        # Kept: spectra, results, every json sidecar.
        for f in ("spectra/spectra_ms2.parquet", "scored.parquet", "fragment_quant.parquet",
                  "cal.json", "seed_psms.parquet.masscal.json", "rt_bootstrap/cal.json",
                  "run_windows.parquet.report.json"):
            assert f"{r}/{f}" not in rv, f
        assert f"second_pass/{r}/chromatograms.parquet" not in rv
    assert "first_pass/scored_combined.parquet" in rv
    assert "first_pass/scored_combined.parquet.report.json" not in rv
    assert "sidecar_work/handoff.npy" in rv
    assert "my_notes.txt" not in rv and "my_notes.txt" in v["unclassified_kept"]
    hashes = {e["path"]: e["content_hash"] for e in v["removed"]}
    assert hashes["first_pass/scored_combined.parquet"] == "fp"

    res = prune.plan(out, "results", with_mzml=False)
    rr = removed(res)
    assert rv <= rr
    for r in ("r0", "r1"):
        assert f"{r}/spectra/spectra_ms2.parquet" in rr
        assert f"second_pass/{r}/chromatograms.parquet" in rr
        assert f"second_pass/{r}/cal.json" not in rr
    assert "second_pass/library/lib_precursors.parquet" not in rr
    assert res["tier_bytes"]["results"] < res["tier_bytes"]["viewer"] < res["tier_bytes"]["all"]


def test_without_second_pass_viewer_keeps_first_pass_evidence(tmp_path):
    out = make_experiment(tmp_path, second_pass=False)
    rv = removed(prune.plan(out, "viewer", with_mzml=False))
    for f in ("run_windows.parquet", "chromatograms.parquet", "features.parquet",
              "psms_competed.parquet", "psms_extracted.parquet", "seed_psms.parquet"):
        assert f"r0/{f}" not in rv, f
    assert "r0/fragment_library_precursors_multihead.parquet" in rv
    rr = removed(prune.plan(out, "results", with_mzml=False))
    assert "r0/chromatograms.parquet" in rr and "r0/scored.parquet" not in rr


def test_hard_links_count_once(tmp_path):
    out = make_experiment(tmp_path, second_pass=True)
    p = prune.plan(out, "viewer", with_mzml=False)
    by_path = {e["path"]: e["bytes"] for e in p["removed"]}
    # features.parquet and psms_competed.parquet are one inode: removed twice, freed once.
    assert p["bytes_removed"] - p["bytes_freed"] == 2 * by_path["r0/features.parquet"]


def test_apply_writes_record_and_keeps_kept(tmp_path):
    out = make_experiment(tmp_path, second_pass=True)
    p = prune.plan(out, "results", with_mzml=False)
    prune.apply(out, p)
    assert not (out / "r0" / "chromatograms.parquet").exists()
    assert not (out / "r0" / "spectra" / "spectra_ms2.parquet").exists()
    assert (out / "r0" / "spectra" / "spectra_ms2.parquet.report.json").exists()
    assert not (out / "first_pass" / "scored_combined.parquet").exists()
    assert (out / "r0" / "scored.parquet").exists()
    assert (out / "my_notes.txt").exists()
    rec = json.loads((out / "pruned.json").read_text())
    assert rec["prunes"][0]["tier"] == "results"
    assert len(rec["prunes"][0]["manifest"]["sha256"]) == 64
    # A second prune appends, and has nothing left to remove.
    p2 = prune.plan(out, "results", with_mzml=False)
    assert p2["removed"] == []
    prune.apply(out, p2)
    assert len(json.loads((out / "pruned.json").read_text())["prunes"]) == 2


def test_converted_mzml_only_beside_vendor_inputs(tmp_path):
    user = make_experiment(tmp_path / "a", second_pass=True, vendor=False)
    assert prune.plan(user, "results", with_mzml=True)["converted_mzml"] == []
    conv = make_experiment(tmp_path / "b", second_pass=True, vendor=True)
    p = prune.plan(conv, "results", with_mzml=True)
    assert sorted(Path(e["path"]).name for e in p["converted_mzml"]) == ["run0.mzML", "run1.mzML"]
    assert prune.plan(conv, "results", with_mzml=False)["converted_mzml"] == []
    prune.apply(conv, p)
    assert not (tmp_path / "b" / "in" / "run0.mzML").exists()
    assert (tmp_path / "b" / "in" / "run0.raw").exists()


def test_single_run(tmp_path):
    out = tmp_path / "single"
    for f in ("psms_scored.parquet", "chromatograms.parquet", "features.parquet",
              "spectra/spectra_ms2.parquet", "peptide_quant.parquet",
              "fragment_library_precursors.parquet", "peptides.parquet"):
        _write(out / f)
    (out / "manifest.json").write_text(json.dumps({"artifacts": {}, "cli_args": []}))
    rv = removed(prune.plan(out, "viewer", with_mzml=False))
    assert rv == {"peptides.parquet"}
    rr = removed(prune.plan(out, "results", with_mzml=False))
    assert rr == {"peptides.parquet", "chromatograms.parquet", "features.parquet",
                  "spectra/spectra_ms2.parquet", "fragment_library_precursors.parquet"}
