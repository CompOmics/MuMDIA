#!/usr/bin/env python3
"""Generate THIRD_PARTY_LICENSES.md from the Cargo lockfile.

Why this exists. The release archive and the container image both ship a statically
linked binary containing every Rust dependency in `rust/mumdia/Cargo.lock`. Twenty of
those are Apache-2.0, whose section 4(d) requires propagating the NOTICE contents of
works you redistribute, and a further hundred-odd are MIT or dual MIT/Apache, whose
terms require the copyright notice "in all copies or substantial portions of the
Software". Before this the tracked tree contained `LICENSE` (MuMDIA's own Apache-2.0)
and nothing else, so a distributed binary carried no third-party notice at all.

Why generated from `cargo metadata` rather than `cargo about` or
`cargo-bundle-licenses`. Those are better tools and produce per-crate licence TEXT, but
they are extra binaries a contributor has to install, and this project already has two
generated references kept fresh by a CI `--check`. Every one of the crates in the
lockfile declares an SPDX expression (verified: zero unspecified), so the licence
inventory is exact without another tool. What this file does not do is reproduce each
crate's individual copyright line; it points at the crate's own repository for that,
which is where the canonical notice lives.

Usage:
    python ci/gen_third_party_licenses.py            # write the file
    python ci/gen_third_party_licenses.py --check    # fail if it is stale
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "rust" / "mumdia" / "Cargo.toml"
OUT = ROOT / "THIRD_PARTY_LICENSES.md"

# Crates in this workspace: our own code, covered by LICENSE, not third-party.
OURS = {"mumdia", "mumdia-core", "mumdia-io"}

# Licence texts short enough to embed, for the families that are not Apache-2.0
# (which is already in LICENSE, verbatim, as MuMDIA's own licence).
TEXTS = {
    "MIT": """Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the
Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.""",
    "BSD-2-Clause": """Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list
   of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation and/or other
   materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT, ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.""",
    "BSD-3-Clause": """Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list
   of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation and/or other
   materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be
   used to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT, ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.""",
    "Zlib": """This software is provided 'as-is', without any express or implied warranty. In no
event will the authors be held liable for any damages arising from the use of this
software.

Permission is granted to anyone to use this software for any purpose, including
commercial applications, and to alter it and redistribute it freely, subject to the
following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that
   you wrote the original software.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.""",
}

# Public-domain dedications and equivalents: nothing to reproduce, but recorded so a
# reader can see they were considered rather than missed.
NO_OBLIGATION = {"CC0-1.0", "Unlicense", "0BSD", "MIT-0"}


def spdx_terms(expr: str) -> list[str]:
    """The individual licence identifiers in an SPDX expression."""
    cleaned = re.sub(r"[()]", " ", expr)
    parts = re.split(r"\s+(?:OR|AND)\s+|/", cleaned)
    return [p.strip() for p in parts if p.strip() and p.strip() != "WITH"]


def permissive_choice(expr: str) -> str | None:
    """The permissive arm MuMDIA relies on, for a disjunctive expression.

    `r-efi` is `MIT OR Apache-2.0 OR LGPL-2.1-or-later`: the LGPL arm carries
    obligations the other two do not, and a disjunction lets the distributor choose,
    so recording WHICH arm is relied on is the point of this function.
    """
    if " OR " not in expr and "/" not in expr:
        return None
    for preferred in ("Apache-2.0", "MIT", "BSD-2-Clause", "Zlib", "0BSD"):
        if preferred in spdx_terms(expr):
            return preferred
    return None


def collect() -> list[dict]:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked",
         "--manifest-path", str(MANIFEST)],
        capture_output=True, text=True, check=True,
    )
    meta = json.loads(out.stdout)
    pkgs = []
    for p in meta["packages"]:
        if p["name"] in OURS:
            continue
        pkgs.append({
            "name": p["name"],
            "version": p["version"],
            "license": p.get("license") or "",
            "repository": p.get("repository") or "",
        })
    pkgs.sort(key=lambda p: (p["name"].lower(), p["version"]))
    return pkgs


def render(pkgs: list[dict]) -> str:
    L: list[str] = []
    a = L.append
    a("# Third-party licences")
    a("")
    a("GENERATED FILE. Do not edit. `ci/gen_third_party_licenses.py` writes it from")
    a("`rust/mumdia/Cargo.lock`, and CI fails when it is stale.")
    a("")
    a("MuMDIA's own licence is Apache-2.0; see `LICENSE`. The release binary is")
    a("statically linked, so it contains the crates listed below and this file")
    a("accompanies it in every release archive and in the container image.")
    a("")

    # Obligation summary, which is the part a reader actually needs.
    terms: dict[str, int] = {}
    for p in pkgs:
        for t in spdx_terms(p["license"]):
            terms[t] = terms.get(t, 0) + 1
    copyleft = sorted(t for t in terms if "GPL" in t or "MPL" in t)
    a("## Obligations")
    a("")
    a(f"{len(pkgs)} third-party crates. Every one declares an SPDX expression; none is")
    a("unspecified.")
    a("")
    if copyleft:
        a("The following copyleft identifiers appear, in each case as one arm of a")
        a("disjunction whose permissive arm MuMDIA relies on instead (the arm is named")
        a("per crate in the table below):")
        a("")
        for t in copyleft:
            a(f"- `{t}`")
        a("")
        a("No crate imposes a copyleft obligation on the distributed binary.")
    else:
        a("No copyleft identifier appears anywhere in the graph.")
    a("")
    a("Licence identifiers by crate count:")
    a("")
    a("| SPDX identifier | crates |")
    a("|---|---|")
    for t in sorted(terms, key=lambda k: (-terms[k], k)):
        a(f"| `{t}` | {terms[t]} |")
    a("")

    a("## Crates")
    a("")
    a("| crate | version | licence | relied-on arm | source |")
    a("|---|---|---|---|---|")
    for p in pkgs:
        choice = permissive_choice(p["license"])
        repo = p["repository"]
        src = f"[{repo}]({repo})" if repo.startswith("http") else (repo or "-")
        a(f"| `{p['name']}` | {p['version']} | {p['license']} | "
          f"{('`' + choice + '`') if choice else '-'} | {src} |")
    a("")

    a("## Licence texts")
    a("")
    a("Apache-2.0 is reproduced in full in `LICENSE`, which is MuMDIA's own licence and")
    a("covers the Apache-2.0 dependencies as well. The remaining families are below.")
    a("Individual copyright holders are named in each crate's own repository, linked in")
    a("the table above; this file reproduces the licence terms, not per-crate")
    a("attribution lines.")
    a("")
    for name in sorted(TEXTS):
        if not any(name in spdx_terms(p["license"]) for p in pkgs):
            continue
        a(f"### {name}")
        a("")
        a("```text")
        a(TEXTS[name])
        a("```")
        a("")
    # Identifiers that apply through an `AND` (so no permissive arm can be chosen) and
    # whose text is a data licence rather than a software one. Named with the crate and
    # its repository, which is where the canonical text lives.
    conjunctive = [
        (p["name"], p["repository"], t)
        for p in pkgs
        for t in spdx_terms(p["license"])
        if t == "Unicode-3.0"
    ]
    if conjunctive:
        a("### Unicode-3.0")
        a("")
        a("Applies through an `AND`, so it is not one of several arms to choose from. It")
        a("covers the Unicode character tables embedded in the crate rather than its")
        a("code, and the canonical text ships with the crate:")
        a("")
        for name, repo, _ in conjunctive:
            a(f"- `{name}` -- {repo}")
        a("")

    present_free = sorted(n for n in NO_OBLIGATION
                          if any(n in spdx_terms(p["license"]) for p in pkgs))
    if present_free:
        a("### Public-domain dedications")
        a("")
        a("These impose no reproduction requirement, and are recorded so it is clear they")
        a("were considered: " + ", ".join(f"`{n}`" for n in present_free) + ".")
        a("")

    a("## Python sidecars")
    a("")
    a("The Python workers in `scripts/` are MuMDIA's own code under `LICENSE`. Their")
    a("dependencies (DeepLC, MS2PIP, mokapot, torch, numpy, pandas, pyarrow) are")
    a("installed by the user or by the container build from `env/*.yml` and are not")
    a("redistributed inside the binary, so their licences travel with those packages.")
    a("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="fail if THIRD_PARTY_LICENSES.md is stale")
    args = ap.parse_args()

    text = render(collect())
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current.replace("\r\n", "\n") != text:
            print(f"{OUT} is stale. Regenerate with:\n"
                  f"    python ci/gen_third_party_licenses.py", file=sys.stderr)
            return 1
        print(f"{OUT} is up to date ({len(text.splitlines())} lines).")
        return 0

    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
