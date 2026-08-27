#!/usr/bin/env python
"""Assert that an end-to-end fixture run produced what it should.

Paired with `ci/make_fixture_mzml.py`, which plants a known set of target
precursors, and with `ci/smoke.sh`, which runs the pipeline over them. This script
is the part that fails CI.

The assertions are deliberately of two kinds. Structural facts that are exactly
determined (spectrum counts, artifact presence, hash shape, schema versions) are
asserted exactly. Scientific outcomes (how many planted peptides come back) are
asserted as bands, because pinning them exactly would turn every legitimate
improvement in sensitivity into a red build. A band still catches the failures
that matter: a stage silently producing nothing, decoys leaking above threshold,
or recovery collapsing.

Usage:
    python ci/check_smoke.py --out-dir out --planted work/planted.json
    python ci/check_smoke.py --out-dir out --planted work/planted.json \\
        --compare-peptides out2/peptides.tsv    # determinism check
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

# Artifact schema versions frozen in mumdia-core/src/schema.rs. A bump here
# without a deliberate decision means an on-disk format changed silently, which is
# the thing artifact versioning exists to prevent.
EXPECTED_SCHEMA_VERSIONS = {
    "spectra_ms1": 1,
    "spectra_ms2": 1,
    "isolation_windows": 1,
    "ms2_to_ms1": 1,
    "peptides": 1,
    "peptidoforms": 1,
    "fragment_library_precursors": 1,
    "fragment_library_fragments": 1,
    "seed_psms": 1,
    "run_windows": 1,
    "psms_extracted": 2,
    "chromatograms": 1,
    "features": 1,
    "psms_competed": 3,
    "psms_scored": 4,
    "peptide_quant": 2,
    "protein_group_quant": 2,
}

BLAKE3_HEX = re.compile(r"^[0-9a-f]{64}$")


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passed = 0

    def ok(self, condition: bool, message: str, detail: str = "") -> bool:
        suffix = f": {detail}" if detail else ""
        if condition:
            self.passed += 1
            print(f"  [ ok ] {message}{suffix}")
            return True
        self.failures.append(f"{message}{suffix}")
        print(f"  [FAIL] {message}{suffix}")
        return False

    def band(self, value: float, lo: float, hi: float, message: str) -> bool:
        return self.ok(lo <= value <= hi, f"{message} = {value} (expected {lo} to {hi})")


def strip_mods(peptidoform: str) -> str:
    """Residue letters only, matching `report.rs::strip`."""
    out, depth = [], 0
    for c in peptidoform.removeprefix("DECOY_"):
        if c in "[(":
            depth += 1
        elif c in "])":
            depth = max(0, depth - 1)
        elif depth == 0 and c.isalpha():
            out.append(c)
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="run output directory")
    ap.add_argument("--planted", required=True, help="manifest written by make_fixture_mzml.py")
    ap.add_argument("--compare-peptides", default=None, help="peptides.tsv from a second run")
    ap.add_argument("--min-recovery", type=float, default=0.60,
                    help="minimum fraction of planted stripped peptides in peptides.tsv")
    ap.add_argument("--max-decoy-fraction", type=float, default=0.02,
                    help="maximum decoy fraction among accepted PSMs at 1 percent")
    a = ap.parse_args()

    out = Path(a.out_dir)
    planted = json.loads(Path(a.planted).read_text(encoding="utf-8"))
    c = Checks()

    # ---------------------------------------------------------------- artifacts
    print("artifacts and manifest")
    manifest_path = out / "manifest.json"
    if not c.ok(manifest_path.is_file(), "manifest.json exists"):
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    c.ok(bool(manifest.get("mumdia_version")), "manifest records mumdia_version",
         str(manifest.get("mumdia_version")))
    c.ok(BLAKE3_HEX.match(manifest.get("config_hash", "") or "") is not None,
         "manifest config_hash is a blake3 hex digest")
    c.ok(bool(manifest.get("config_json")), "manifest embeds the resolved config")

    # Source identity. A benchmark record is supposed to name the commit it came
    # from; before this the manifest could only say "0.1.0", which every build says.
    sha = manifest.get("git_sha", "")
    c.ok(bool(sha), "manifest records git_sha", sha)
    if sha == "unknown":
        print("  [note] git_sha is 'unknown': built without git available")
    else:
        c.ok(
            re.match(r"^[0-9a-f]{7,40}(-dirty)?$", sha) is not None,
            "git_sha looks like a short commit, optionally -dirty",
            sha,
        )
    c.ok(bool(manifest.get("commit_date")), "manifest records commit_date",
         str(manifest.get("commit_date")))
    args = manifest.get("cli_args", [])
    c.ok(len(args) > 1 and "run" in args,
         "manifest records the command line", f"{len(args)} arguments")

    # Hashed inputs: recording only a path does not tie a result to the bytes it
    # came from, because a path gets reused.
    inputs = manifest.get("inputs", {})
    for role in ("mzml", "fasta"):
        rec = inputs.get(role)
        if c.ok(rec is not None, f"manifest hashed the {role} input"):
            c.ok(BLAKE3_HEX.match(rec.get("content_hash", "") or "") is not None,
                 f"{role} input hash is a blake3 hex digest")
            c.ok(int(rec.get("bytes", 0)) > 0, f"{role} input records its size",
                 str(rec.get("bytes")))

    models = manifest.get("model_identities", {})
    for key in ("rt_predictor", "fragment_predictor", "rescorer", "feature_schema_id"):
        c.ok(key in models and bool(models[key]), f"manifest model_identities has {key}",
             str(models.get(key)))
    # The rescorer identity must be the one the scored artifact reports, because
    # `psms_scored.parquet.report.json` is the documented source of truth for what
    # actually ran; the configured enum is not.
    scored_report = json.loads(
        (out / "psms_scored.parquet.report.json").read_text(encoding="utf-8")
    )
    c.ok(models.get("rescorer") == scored_report.get("model_identity"),
         "manifest rescorer matches the scored artifact's model_identity",
         f"{models.get('rescorer')!r} vs {scored_report.get('model_identity')!r}")
    # Strict mode exists so an explicitly requested classifier cannot silently
    # become native_tda. The fixture asks for native_tda, so the assertion is that
    # the classifier that ran is the one requested, whatever it is.
    ran = scored_report.get("stats", {}).get("classifier")
    requested = scored_report.get("params", {}).get("classifier_requested", "")
    c.ok(ran == "native_tda", "the classifier that ran is the configured one", str(ran))
    c.ok(requested.lower().replace("_", "") == "nativetda",
         "the scored report records which classifier was requested", str(requested))

    artifacts = manifest.get("artifacts", {})
    seen_schemas = {}
    for name, rec in sorted(artifacts.items()):
        path = out / Path(rec["path"]).name if not Path(rec["path"]).is_absolute() else Path(rec["path"])
        c.ok(BLAKE3_HEX.match(rec.get("content_hash", "") or "") is not None,
             f"{name}: content_hash is a blake3 hex digest")
        c.ok(int(rec.get("rows", 0)) > 0, f"{name}: rows > 0", str(rec.get("rows")))
        c.ok(bool(rec.get("producing_stage")), f"{name}: names its producing stage")
        seen_schemas[rec.get("schema_name")] = rec.get("schema_version")
    for schema, version in sorted(EXPECTED_SCHEMA_VERSIONS.items()):
        if schema in seen_schemas:
            c.ok(seen_schemas[schema] == version,
                 f"schema {schema} is v{version}", f"found v{seen_schemas[schema]}")

    # ------------------------------------------------------------------- convert
    print("convert")
    opts = planted["options"]
    ms1 = pq.read_metadata(out / "spectra" / "spectra_ms1.parquet").num_rows
    ms2 = pq.read_metadata(out / "spectra" / "spectra_ms2.parquet").num_rows
    windows = pq.read_metadata(out / "spectra" / "isolation_windows.parquet").num_rows
    c.ok(ms1 == opts["cycles"], f"MS1 spectra = {opts['cycles']}", str(ms1))
    c.ok(ms2 == opts["cycles"] * opts["windows"],
         f"MS2 spectra = {opts['cycles'] * opts['windows']}", str(ms2))
    c.ok(windows == opts["windows"], f"isolation windows = {opts['windows']}", str(windows))
    # Retention time must arrive in seconds. `convert` multiplies mzdata's minutes
    # by 60; if that ever changes, every RT tolerance in the engine is wrong by 60x.
    rt = pq.read_table(out / "spectra" / "spectra_ms2.parquet", columns=["rt_seconds"]).column(0).to_pylist()
    c.ok(max(rt) <= planted["gradient_seconds"] * 1.05,
         "MS2 retention times are in seconds and within the gradient",
         f"max {max(rt):.1f} s vs gradient {planted['gradient_seconds']:.0f} s")

    # -------------------------------------------------------- RT calibration ran
    print("retention-time calibration")
    cal = json.loads((out / "cal.json").read_text(encoding="utf-8"))
    c.ok(cal.get("calibration_status") == "loess",
         "RT calibration fitted a LOESS rather than falling back to unbounded windows",
         str(cal.get("calibration_status")))
    c.ok(cal.get("method") == "loess", "cal.json records the method", str(cal.get("method")))
    c.ok(int(cal.get("n_train") or 0) >= 20,
         "RT calibration had enough anchors to fit", f"n_train = {cal.get('n_train')}")
    w_rt = cal.get("w_rt")
    c.ok(isinstance(w_rt, (int, float)) and w_rt > 0,
         "cal.json has a finite positive w_rt", str(w_rt))
    # The fixture plants retention time as an affine function of the library iRT
    # with jitter bounded at 1.5 s, so a correct calibration lands within a few
    # seconds. These residuals are in-sample and optimistic by construction (see
    # docs/08_rt_im_train.md); the assertion is a floor on sanity, not an estimate
    # of accuracy.
    resid = cal.get("rt_residual_abs_median_s")
    c.ok(isinstance(resid, (int, float)) and resid <= 5.0,
         "in-sample RT residual median <= 5 s on planted affine retention times",
         f"{resid} s")

    # ------------------------------------------------------------ FDR and recovery
    print("identifications")
    scored = pq.read_table(
        out / "psms_scored.parquet",
        columns=["peptidoform", "label", "peptide_q_value", "q_value"],
    ).to_pydict()
    at1 = [i for i, q in enumerate(scored["peptide_q_value"]) if q is not None and q <= 0.01]
    n_target = sum(1 for i in at1 if scored["label"][i] == "target")
    n_decoy = len(at1) - n_target
    c.ok(n_target > 0, "some targets are accepted at 1 percent peptide q", str(n_target))
    frac = n_decoy / max(1, len(at1))
    c.ok(frac <= a.max_decoy_fraction,
         f"decoy fraction at 1 percent <= {a.max_decoy_fraction}",
         f"{frac:.4f} ({n_decoy} of {len(at1)})")

    # No planted peptide may come back as a decoy identification: the decoys are
    # never planted, so a decoy at threshold means the labels or the null broke.
    planted_stripped = {strip_mods(p["peptidoform"]) for p in planted["planted"]}
    decoy_hits = {
        strip_mods(scored["peptidoform"][i]) for i in at1 if scored["label"][i] == "decoy"
    } & planted_stripped
    c.ok(not decoy_hits, "no planted sequence is accepted as a decoy", str(sorted(decoy_hits)[:5]))

    with (out / "peptides.tsv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    c.ok(len(rows) > 0, "peptides.tsv has rows", str(len(rows)))
    reported = {r["stripped_sequence"] for r in rows}
    recovered = planted_stripped & reported
    recovery = len(recovered) / max(1, len(planted_stripped))
    c.ok(recovery >= a.min_recovery,
         f"planted peptide recovery >= {a.min_recovery:.0%}",
         f"{recovery:.1%} ({len(recovered)} of {len(planted_stripped)})")
    c.ok(not any(r["precursor"].startswith("DECOY_") for r in rows),
         "peptides.tsv contains no decoy rows")

    # ---------------------------------------------------------------------- quant
    print("quantification")
    pep_q = pq.read_table(
        out / "peptide_quant.parquet", columns=["quantity", "quant_status", "n_fragments_used"]
    ).to_pydict()
    quantified = [
        (q, n) for q, s, n in zip(pep_q["quantity"], pep_q["quant_status"], pep_q["n_fragments_used"])
        if s == "quantified"
    ]
    c.ok(len(quantified) > 0, "some peptides are quantified", str(len(quantified)))
    c.ok(all(q is not None and q > 0 for q, _ in quantified),
         "every quantified peptide has a positive finite quantity")
    c.ok(all(n >= 1 for _, n in quantified),
         "every quantified peptide used at least one fragment")
    # An unquantifiable identification must keep a null quantity, never a zero:
    # zero is a measurement, absence is not.
    zeros = [
        q for q, s in zip(pep_q["quantity"], pep_q["quant_status"]) if s != "quantified" and q == 0
    ]
    c.ok(not zeros, "unquantifiable rows are null, not zero", f"{len(zeros)} zero-valued rows")

    # ---------------------------------------------------------------- determinism
    if a.compare_peptides:
        print("determinism")
        first = (out / "peptides.tsv").read_bytes()
        second = Path(a.compare_peptides).read_bytes()
        c.ok(first == second,
             "a second identical run produces a byte-identical peptides.tsv",
             f"{len(first)} vs {len(second)} bytes")

    print()
    if c.failures:
        print(f"smoke check FAILED: {len(c.failures)} of {c.passed + len(c.failures)} assertions")
        for f in c.failures:
            print(f"  - {f}")
        return 1
    print(f"smoke check OK: {c.passed} assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
