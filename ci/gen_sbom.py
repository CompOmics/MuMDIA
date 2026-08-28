#!/usr/bin/env python3
"""Generate sbom.cdx.json, a CycloneDX 1.5 software bill of materials.

Why this exists, given that THIRD_PARTY_LICENSES.md already lists every crate. That
file is a human-readable notice document, written to discharge licence obligations. An
SBOM answers a different question, asked by a different reader: given a published
binary, which exact package versions are inside it, and how are they connected. That
question is asked by vulnerability scanners, by institutional software inventories,
and by anyone who has to decide whether a new advisory affects a copy of MuMDIA they
already deployed. Prose cannot be queried; a CycloneDX document can, by any of the
standard tools.

What it contains:

- one component per resolved crate, with a `purl` (`pkg:cargo/name@version`), the
  SPDX licence expression, and the upstream repository;
- the dependency GRAPH from `cargo metadata`'s resolve section, not just a flat list,
  so a scanner can tell a direct dependency from something pulled in four levels down;
- the workspace's own version as the metadata component.

Deliberately absent: a timestamp and a serial number. Both are optional in CycloneDX,
and both would make the output differ on every regeneration, which would defeat the
`--check` gate in CI and make the file churn in every diff. The provenance that
matters is the commit the file is generated from.

Only the Rust side. The Python sidecars are separate resolved environments rather than
statically linked dependencies, and their exact package sets are recorded per build by
the `sidecar-imports` CI job (`resolved-env-*` artifacts). Merging a conda resolution
into this document would require resolving it, which needs a Linux runner and network.

Usage:
    python ci/gen_sbom.py            # write the file
    python ci/gen_sbom.py --check    # fail if it is stale
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "rust" / "mumdia" / "Cargo.toml"
OUT = ROOT / "sbom.cdx.json"

# Crates in this workspace: our own code, so they are components of the application
# rather than third-party dependencies of it.
OURS = {"mumdia", "mumdia-core", "mumdia-io"}


def cargo_metadata() -> dict:
    """Full metadata, WITH dependencies, so `resolve` carries the graph.

    `--locked` matters: it fails rather than quietly updating Cargo.lock, so the SBOM
    describes the versions a `--locked` release build actually compiles.
    """
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked",
         "--manifest-path", str(MANIFEST)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def bom_ref(name: str, version: str) -> str:
    return f"pkg:cargo/{name}@{version}"


def component(p: dict) -> dict:
    """One CycloneDX component from one cargo package."""
    c: dict = {
        "type": "library",
        "bom-ref": bom_ref(p["name"], p["version"]),
        "name": p["name"],
        "version": p["version"],
        "purl": bom_ref(p["name"], p["version"]),
    }
    if p.get("description"):
        # Single line: cargo descriptions occasionally carry newlines, and a
        # multi-line value in a JSON string is legal but makes the diff noisy.
        c["description"] = " ".join(p["description"].split())
    lic = p.get("license")
    if lic:
        # `licenses[].expression` is the correct field for an SPDX EXPRESSION such as
        # `MIT OR Apache-2.0`. Putting a disjunction in `licenses[].license.id`, which
        # is what a naive converter does, produces a document that claims a licence
        # identifier that does not exist.
        c["licenses"] = [{"expression": lic}]
    elif p.get("license_file"):
        c["licenses"] = [{"license": {"name": f"see {p['license_file']}"}}]
    refs = []
    if p.get("repository"):
        refs.append({"type": "vcs", "url": p["repository"]})
    if p.get("homepage") and p.get("homepage") != p.get("repository"):
        refs.append({"type": "website", "url": p["homepage"]})
    if refs:
        c["externalReferences"] = refs
    return c


def render(meta: dict) -> str:
    by_id = {p["id"]: p for p in meta["packages"]}
    workspace_version = next(
        (p["version"] for p in meta["packages"] if p["name"] == "mumdia"), "0.0.0"
    )

    components = sorted(
        (component(p) for p in meta["packages"] if p["name"] not in OURS),
        key=lambda c: (c["name"].lower(), c["version"]),
    )

    # The graph. `resolve.nodes` is keyed by package id; translate to purls, drop our
    # own crates as targets only where they are not real nodes (they are, so they stay
    # and correctly show what the application depends on directly).
    deps = []
    for node in meta.get("resolve", {}).get("nodes", []):
        me = by_id.get(node["id"])
        if not me:
            continue
        depends = sorted(
            bom_ref(by_id[d]["name"], by_id[d]["version"])
            for d in node.get("dependencies", [])
            if d in by_id
        )
        deps.append({
            "ref": bom_ref(me["name"], me["version"]),
            "dependsOn": depends,
        })
    deps.sort(key=lambda d: d["ref"])

    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": bom_ref("mumdia", workspace_version),
                "name": "mumdia",
                "version": workspace_version,
                "description": "Clean-room Rust DIA proteomics search engine",
                "licenses": [{"expression": "Apache-2.0"}],
                "purl": bom_ref("mumdia", workspace_version),
            },
            "tools": [{"name": "ci/gen_sbom.py", "vendor": "CompOmics"}],
        },
        "components": components,
        "dependencies": deps,
    }
    return json.dumps(bom, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="fail if sbom.cdx.json is stale")
    args = ap.parse_args()

    text = render(cargo_metadata())
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current.replace("\r\n", "\n") != text:
            print(f"{OUT} is stale. Regenerate with:\n"
                  f"    python ci/gen_sbom.py", file=sys.stderr)
            return 1
        n = len(json.loads(text)["components"])
        print(f"{OUT} is up to date ({n} components).")
        return 0

    OUT.write_text(text, encoding="utf-8", newline="\n")
    n = len(json.loads(text)["components"])
    print(f"wrote {OUT} ({n} components, {len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
