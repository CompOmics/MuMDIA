"""Prune a finished MuMDIA output directory to a retention tier.

  python prune_outputs.py <out_dir> --tier results|viewer|all [--apply] [--converted-mzml]
                          [--run-cwd DIR] [--json]

Without ``--apply`` nothing is deleted: the script lists what the tier would remove and prints
the size of every tier. With ``--apply`` it deletes those files and appends a record to
``<out_dir>/pruned.json`` (each removed path, its size and the content hash the manifest
recorded for it, plus the SHA-256 of the manifest itself), so the provenance of the directory
stays complete.

The three tiers (docs/01_overview_and_dataflow.md, "Output retention"):

* ``all``: everything; what the engine writes. Pruning to it removes nothing.
* ``viewer``: ``results`` plus the per-run evidence tables a viewer or a re-run of ``quant``,
  ``localise --experiment-dir`` and ``mumdia mobilogram`` read, and the converted spectra
  (``<run>/spectra/``). When the experiment's manifest has a non-null
  ``experiment.second_pass``, the evidence tables are the second pass's
  (``second_pass/<run>/``) and the first pass's per-run tables are removed; without a second
  pass (a single run, or an experiment where it did not apply) the first-pass tables ARE the
  evidence and are kept.
* ``results``: the manifest, the configuration, the scored tables, the quantities, the LFQ
  matrices, the TSV reports, the second pass's empirical library and every ``.json`` sidecar
  (``cal.json``, ``*.masscal.json`` and the ``.report.json`` / ``.schema.json`` /
  ``.summary.json`` of every table, removed ones included). ``mumdia report --experiment-dir``
  runs on it.

The rules are the engine's own (``output.retain``, ``mumdia::retain``): a file is removed only
when it belongs to a class the tier drops. A file the rules do not know is kept and listed, so
a user's own files in the directory are never deleted. The script refuses a directory without
a manifest (``experiment_manifest.json`` or ``manifest.json``): the engine writes it last, so
its absence means the run did not finish.

``--converted-mzml`` also deletes the mzML the engine converted from a vendor input
(``raw::ensure_mzml`` writes it beside the input, outside the output directory). It is
removed only when the manifest records it as the input of a run whose command line named a
vendor file with the same stem, so an mzML the user supplied is never touched. Relative input
paths in the manifest are relative to the directory the run was started in; it is inferred
from ``--out-dir`` on the recorded command line, or given with ``--run-cwd``.

Hard links (``psms_competed.parquet`` is a hard link to ``features.parquet``) are counted once:
``bytes_freed`` adds a file's size only when every link to it inside the directory is removed.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath

TIERS = ("results", "viewer", "all")

# The class of a file decides which tiers keep it.
RESULTS = "results"  # every tier
VIEWER = "viewer"  # viewer and all
INTERMEDIATE = "intermediate"  # all only
SCRATCH = "scratch"  # all only (sidecar handoff files left by a killed worker)
UNKNOWN = "unknown"  # never removed

KEEPS = {
    "all": {RESULTS, VIEWER, INTERMEDIATE, SCRATCH, UNKNOWN},
    "viewer": {RESULTS, VIEWER, UNKNOWN},
    "results": {RESULTS, UNKNOWN},
}

MANIFESTS = ("experiment_manifest.json", "manifest.json")

# Files at the experiment root (or a single run's root) that are results.
ROOT_RESULTS = {
    "scored_combined.parquet",
    "scored_mbr.parquet",
    "mbr_transferred.parquet",
    "lfq_maxlfq.parquet",
    "lfq_maxlfq.parquet.peptide.parquet",
    "lfq_maxlfq.parquet.precursor.parquet",
    "quant_normalisation.parquet",
    "localisation_experiment.parquet",
}
# A FASTA-built library written into the output directory: the searched library, which a
# viewer shows predicted fragments from. Under a second pass the empirical library replaces it.
ROOT_LIBRARY = {
    "fragment_library_precursors.parquet",
    "fragment_library_fragments.parquet",
}
# Built on the way to a library or a calibration; read by nothing after the run.
ROOT_INTERMEDIATE = {
    "peptides.parquet",
    "peptidoforms.parquet",
    "fragment_library_precursors_deeplc.parquet",
}
# A run's results.
RUN_RESULTS = {
    "scored.parquet",
    "psms_scored.parquet",
    "peptide_quant.parquet",
    "protein_group_quant.parquet",
    "fragment_quant.parquet",
    "quant_fragments.parquet",
    "localisation.parquet",
}
# A run's evidence tables: what a viewer, `quant`, `localise` and `mobilogram` read. Under a
# second pass the second pass writes them again and these first-pass copies are not read.
RUN_EVIDENCE = {
    "run_windows.parquet",
    "chromatograms.parquet",
    "features.parquet",
    "psms_competed.parquet",
    "psms_extracted.parquet",
    "psms_extracted.parquet.peaks.parquet",
    "seed_psms.parquet",
}
# A run's working copies: libraries with recalibrated retention times, prefiltered
# libraries, the candidate audit and a PIN export.
RUN_INTERMEDIATE = {
    "fragment_library_precursors_multihead.parquet",
    "fragment_library_precursors_ft.parquet",
    "fragment_library_precursors_deeplc.parquet",
    "library_tag_prefiltered_precursors.parquet",
    "library_tag_prefiltered_fragments.parquet",
    "peptidoforms_tag_prefiltered.parquet",
    "candidate_audit.parquet",
    "run.pin",
}
TMP = re.compile(r"\.tmp-\d+-\d+")
VENDOR_SUFFIXES = (".raw", ".d", ".wiff", ".wiff2")


def classify(rel: str, layout: dict) -> str:
    """The class of the file at ``rel`` (POSIX, relative to the output directory).

    ``layout`` has ``experiment`` (bool), ``runs`` (the run directory names) and
    ``second_pass`` (bool: the manifest's ``experiment.second_pass`` is not null).
    """
    p = PurePosixPath(rel)
    parts = p.parts
    name = p.name
    if TMP.search(name):
        return UNKNOWN
    if "sidecar_work" in parts[:-1]:
        return SCRATCH
    if name.endswith(".json") or name.endswith(".tsv"):
        return RESULTS
    second = layout["second_pass"]
    if not layout["experiment"]:
        return _run_class(parts, second=False, root=True)
    top = parts[0]
    if len(parts) == 1:
        return _root_class(name, second)
    if top == "first_pass":
        # The first-pass rescore: superseded by the second pass's, or copied up to the
        # experiment root when the second pass did not apply.
        return INTERMEDIATE
    if top == "second_pass":
        if len(parts) >= 2 and parts[1] == "library":
            return RESULTS
        if len(parts) >= 3 and parts[1] in layout["runs"]:
            return VIEWER if second else UNKNOWN
        return UNKNOWN
    if top in layout["runs"]:
        return _run_class(parts[1:], second=second, root=False)
    return UNKNOWN


def _root_class(name: str, second: bool) -> str:
    if name in ROOT_RESULTS:
        return RESULTS
    if name in ROOT_LIBRARY:
        return INTERMEDIATE if second else VIEWER
    if name in ROOT_INTERMEDIATE:
        return INTERMEDIATE
    return UNKNOWN


def _run_class(parts: tuple, *, second: bool, root: bool) -> str:
    name = parts[-1]
    if len(parts) == 1:
        if root:
            c = _root_class(name, second)
            if c != UNKNOWN:
                return c
        if name in RUN_RESULTS:
            return RESULTS
        if name in RUN_EVIDENCE:
            return INTERMEDIATE if second else VIEWER
        if name in RUN_INTERMEDIATE:
            return INTERMEDIATE
        return UNKNOWN
    sub = parts[0]
    if sub == "spectra":
        return VIEWER
    if sub == "rt_bootstrap":
        return INTERMEDIATE
    if sub == "groups":
        # Window-group bands (docs/33): each band's evidence tables and the overlap losers.
        return INTERMEDIATE if second else VIEWER
    return UNKNOWN


def load_manifest(out: Path) -> tuple[Path, dict]:
    for n in MANIFESTS:
        m = out / n
        if m.is_file():
            with open(m, encoding="utf-8") as fh:
                return m, json.load(fh)
    raise SystemExit(
        f"{out}: no {' or '.join(MANIFESTS)}; the run did not finish (the engine writes the "
        "manifest last), so nothing is pruned"
    )


def layout_of(manifest_name: str, m: dict) -> dict:
    e = m.get("experiment")
    if manifest_name == "experiment_manifest.json" and isinstance(e, dict):
        runs = [str(r) for r in e.get("runs") or []]
        if not runs:
            raise SystemExit("experiment_manifest.json: experiment.runs is empty")
        return {
            "experiment": True,
            "runs": set(runs),
            "second_pass": e.get("second_pass") is not None,
        }
    return {"experiment": False, "runs": set(), "second_pass": False}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cli_value(args: list, flag: str) -> list:
    out = []
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            out.append(args[i + 1])
        elif a.startswith(flag + "="):
            out.append(a.split("=", 1)[1])
    return out


def run_cwd(out: Path, m: dict) -> Path | None:
    """The directory the run was started in, from ``--out-dir`` on its command line."""
    given = _cli_value(m.get("cli_args") or [], "--out-dir")
    if not given:
        return None
    g = Path(given[-1])
    if g.is_absolute():
        return None  # absolute paths need no base
    a = out.resolve()
    gp = [x for x in g.parts if x not in (".", "")]
    if len(gp) <= len(a.parts) and list(a.parts[len(a.parts) - len(gp):]) == gp:
        return Path(*a.parts[: len(a.parts) - len(gp)])
    return None


def artifact_hashes(out: Path, m: dict, base: Path | None) -> dict:
    """Relative path (POSIX) -> the manifest's content hash, for every recorded artifact."""
    res = {}
    o = out.resolve()
    for rec in (m.get("artifacts") or {}).values():
        if not isinstance(rec, dict) or "path" not in rec:
            continue
        p = Path(rec["path"])
        if not p.is_absolute() and base is not None:
            p = base / p
        try:
            rel = p.resolve().relative_to(o).as_posix()
        except ValueError:
            continue
        res[rel] = rec.get("content_hash")
    return res


def converted_mzml(m: dict, base: Path | None) -> list:
    """The mzML files the engine converted from a vendor input, as recorded inputs."""
    cli = _cli_value(m.get("cli_args") or [], "--mzml")
    vendor_stems = {}
    for c in cli:
        cp = Path(c)
        if cp.suffix.lower() in VENDOR_SUFFIXES:
            vendor_stems[(str(cp.parent), cp.stem)] = c
    found = []
    for key, rec in sorted((m.get("inputs") or {}).items()):
        if not key.startswith("mzml[") or not isinstance(rec, dict):
            continue
        p = Path(rec.get("path", ""))
        if p.suffix.lower() != ".mzml":
            continue
        if (str(p.parent), p.stem) not in vendor_stems:
            continue
        if not p.is_absolute():
            if base is None:
                continue
            p = base / p
        found.append({"input": key, "path": p, "content_hash": rec.get("content_hash")})
    return found


def walk(out: Path) -> list:
    files = []
    for dirpath, dirnames, filenames in os.walk(out):
        dirnames.sort()
        # A `.d` directory inside the output (not expected) is data, never walked into.
        dirnames[:] = [d for d in dirnames if not d.lower().endswith(".d")]
        for f in sorted(filenames):
            full = Path(dirpath) / f
            if full.is_symlink():
                continue
            files.append(full)
    return files


def plan(out: Path, tier: str, *, with_mzml: bool, cwd: Path | None = None) -> dict:
    if tier not in TIERS:
        raise SystemExit(f"--tier must be one of {', '.join(TIERS)}")
    mpath, m = load_manifest(out)
    layout = layout_of(mpath.name, m)
    base = cwd if cwd is not None else run_cwd(out, m)
    hashes = artifact_hashes(out, m, base)
    entries = []
    for f in walk(out):
        rel = f.relative_to(out).as_posix()
        if rel == "pruned.json":
            continue
        st = f.stat()
        entries.append(
            {
                "path": rel,
                "class": classify(rel, layout),
                "bytes": st.st_size,
                "inode": (st.st_dev, st.st_ino) if st.st_ino else None,
                "links": st.st_nlink,
            }
        )
    links_in_dir = {}
    for e in entries:
        if e["inode"] is not None:
            links_in_dir[e["inode"]] = links_in_dir.get(e["inode"], 0) + 1

    def unique_bytes(sel) -> int:
        seen, total = set(), 0
        for e in sel:
            if e["inode"] is not None:
                if e["inode"] in seen:
                    continue
                seen.add(e["inode"])
            total += e["bytes"]
        return total

    sizes = {t: unique_bytes(e for e in entries if e["class"] in KEEPS[t]) for t in TIERS}
    removed = [e for e in entries if e["class"] not in KEEPS[tier]]
    removed_inodes = {}
    for e in removed:
        if e["inode"] is not None:
            removed_inodes[e["inode"]] = removed_inodes.get(e["inode"], 0) + 1
    freed, seen = 0, set()
    for e in removed:
        ino = e["inode"]
        if ino is None:
            freed += e["bytes"]
        elif ino not in seen and removed_inodes[ino] == links_in_dir[ino] == e["links"]:
            seen.add(ino)
            freed += e["bytes"]
    mz = []
    if with_mzml and tier != "all":
        for c in converted_mzml(m, base):
            p = c["path"]
            if p.is_file():
                mz.append(
                    {
                        "path": str(p),
                        "input": c["input"],
                        "bytes": p.stat().st_size,
                        "content_hash": c["content_hash"],
                    }
                )
    return {
        "out_dir": str(out),
        "tier": tier,
        "manifest": {"path": mpath.name, "sha256": _sha256(mpath)},
        "experiment": layout["experiment"],
        "second_pass": layout["second_pass"],
        "tier_bytes": sizes,
        "removed": [
            {
                "path": e["path"],
                "class": e["class"],
                "bytes": e["bytes"],
                "content_hash": hashes.get(e["path"]),
            }
            for e in removed
        ],
        "bytes_removed": sum(e["bytes"] for e in removed),
        "bytes_freed": freed,
        "unclassified_kept": [e["path"] for e in entries if e["class"] == UNKNOWN],
        "converted_mzml": mz,
    }


def apply(out: Path, p: dict) -> None:
    for e in p["removed"]:
        (out / e["path"]).unlink()
    for e in p["converted_mzml"]:
        Path(e["path"]).unlink()
    # Directories the prune emptied (never the output directory itself, nor a directory
    # that was empty before).
    emptied = set()
    for e in p["removed"]:
        d = (out / e["path"]).parent
        while d != out and out in d.parents:
            emptied.add(d)
            d = d.parent
    for d in sorted(emptied, key=lambda x: -len(x.parts)):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    rec_path = out / "pruned.json"
    history = []
    if rec_path.is_file():
        with open(rec_path, encoding="utf-8") as fh:
            history = json.load(fh).get("prunes", [])
    entry = {k: v for k, v in p.items() if k not in ("out_dir", "unclassified_kept")}
    entry["tool"] = "scripts/prune_outputs.py"
    entry["date"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    history.append(entry)
    tmp = rec_path.with_name(rec_path.name + f".tmp-{os.getpid()}-0")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"prunes": history}, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, rec_path)


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--tier", required=True, choices=TIERS)
    ap.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    ap.add_argument(
        "--converted-mzml",
        action="store_true",
        help="also delete the mzML converted from each vendor input (outside out_dir)",
    )
    ap.add_argument("--run-cwd", help="the directory the run was started in")
    ap.add_argument("--json", action="store_true", help="print the plan as JSON")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    if not out.is_dir():
        raise SystemExit(f"{out}: not a directory")
    p = plan(
        out,
        a.tier,
        with_mzml=a.converted_mzml,
        cwd=Path(a.run_cwd).resolve() if a.run_cwd else None,
    )
    if a.converted_mzml and not p["converted_mzml"] and a.tier != "all":
        print("note: no converted mzML found (none recorded, or the run directory is unknown: "
              "pass --run-cwd)", file=sys.stderr)
    if a.json:
        json.dump(p, sys.stdout, indent=1)
        print()
    else:
        kind = "experiment" if p["experiment"] else "single run"
        sp = "second pass applied" if p["second_pass"] else "no second pass"
        print(f"{out}: {kind}, {sp}")
        for t in TIERS:
            print(f"  tier {t:8s} keeps {_gb(p['tier_bytes'][t])}")
        print(
            f"tier {a.tier}: removes {len(p['removed'])} files, {_gb(p['bytes_freed'])} freed"
            + (
                f", plus {len(p['converted_mzml'])} converted mzML "
                f"({_gb(sum(e['bytes'] for e in p['converted_mzml']))})"
                if p["converted_mzml"]
                else ""
            )
        )
        for e in p["removed"]:
            print(f"  - {e['path']}  ({e['class']}, {e['bytes']} B)")
        for e in p["converted_mzml"]:
            print(f"  - {e['path']}  (converted mzML, {e['bytes']} B)")
        if p["unclassified_kept"]:
            print(f"kept, not known to the rules ({len(p['unclassified_kept'])}):")
            for r in p["unclassified_kept"]:
                print(f"  ? {r}")
    if a.apply:
        apply(out, p)
        print(f"pruned; recorded in {out / 'pruned.json'}")
    elif p["removed"] or p["converted_mzml"]:
        print("dry run: nothing deleted; pass --apply to delete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
