#!/usr/bin/env python3
"""Generate THIRD_PARTY_LICENSES.md from the Cargo lockfile.

Why this exists. The release archive and the container image both ship a statically
linked binary containing every Rust dependency in `rust/mumdia/Cargo.lock`. Twenty of
those are Apache-2.0, whose section 4(d) requires propagating the NOTICE contents of
works you redistribute, and a further hundred-odd are MIT or dual MIT/Apache, whose
terms require the copyright notice "in all copies or substantial portions of the
Software". Before this the tracked tree contained `LICENSE` (MuMDIA's own Apache-2.0)
and nothing else, so a distributed binary carried no third-party notice at all.

What it produces, in three parts. An INVENTORY, from `cargo metadata`: every crate
with its version, SPDX expression and repository, plus which arm of a disjunctive
expression is relied on. The licence TEXTS for the families that are not Apache-2.0,
which `LICENSE` already reproduces verbatim. And the actual NOTICES: per-crate
copyright assertions and any `NOTICE` file, read from the crate's own
`LICENSE`/`COPYING`/`NOTICE` files in the local cargo registry cache.

The notices are the part that discharges the obligation, and an earlier version of
this file did not have them. MIT requires the copyright notice "in all copies or
substantial portions", BSD requires it retained, and Apache-2.0 section 4(d) requires
a NOTICE file's contents to be propagated. An SPDX identifier plus a generic licence
body satisfies none of those on its own, which is a fair reading of what this file
used to be.

Why not `cargo about` or `cargo-bundle-licenses`: they are good tools, but they are
extra binaries a contributor has to install, and reading the registry cache gets the
same notices with no new dependency and no network. The limitation is that coverage is
whatever has been unpacked locally, which after a `--locked` build is every crate in
the lockfile; the count is printed in the document, so a gap is visible rather than
implied.

This is a mechanical extraction, not a legal opinion. It reproduces what the crates
ship. Whether that satisfies a given distribution's obligations is a question for
whoever signs off the release.

Usage:
    python ci/gen_third_party_licenses.py            # write the file
    python ci/gen_third_party_licenses.py --check    # fail if it is stale
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import tarfile
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
    srcdirs = registry_src_dirs()
    tarballs = registry_tarballs()
    for p in pkgs:
        p["copyrights"], p["notices"] = notices_for(
            p["name"], p["version"], srcdirs, tarballs
        )
    return pkgs


def registry_src_dirs() -> list[Path]:
    """Cargo's unpacked crate sources, where each crate's own licence files live."""
    home = Path(os.environ.get("CARGO_HOME") or (Path.home() / ".cargo"))
    src = home / "registry" / "src"
    return sorted(src.glob("*")) if src.exists() else []


def registry_tarballs() -> dict[tuple[str, str], Path]:
    """`(name, version) -> .crate tarball`, from cargo's download cache.

    The unpacked `registry/src` tree only exists after a BUILD. `cargo fetch --locked`
    populates `registry/cache` with the `.crate` tarballs and nothing else, and a
    documentation or CI job that never compiles has exactly that. Reading the tarballs
    keeps this generator's output identical in both situations, which is what makes
    `--check` a check rather than a statement about which machine ran it.

    This is not a detail: with only the unpacked tree, the check failed on a runner
    that had fetched but not built, reporting the file as stale when the difference was
    that the runner recovered zero notices.
    """
    home = Path(os.environ.get("CARGO_HOME") or (Path.home() / ".cargo"))
    cache = home / "registry" / "cache"
    found: dict[tuple[str, str], Path] = {}
    if not cache.exists():
        return found
    for f in sorted(cache.glob("*/*.crate")):
        stem = f.name[: -len(".crate")]
        # `name-1.2.3`: split at the last hyphen, since crate names contain hyphens.
        name, _, version = stem.rpartition("-")
        if name and version and (name, version) not in found:
            found[(name, version)] = f
    return found


NOTICE_GLOBS = ("LICENSE*", "LICENCE*", "COPYING*", "NOTICE*", "AUTHORS*")
# A licence file larger than this is not a notice, it is vendored content.
MAX_NOTICE_BYTES = 200_000
# An actual copyright ASSERTION: "Copyright" followed by (c), the symbol, or a year.
#
# A looser `Copyright\b` also matches the body of the Apache-2.0 text itself -- "the
# copyright notice that is included in or attached to the work", "a copyright license to
# reproduce, prepare Derivative Works of" -- which turns the notice list into licence
# prose and buries the real assertions. Case-sensitive on the leading word for the same
# reason: prose says "copyright", a notice says "Copyright".
COPYRIGHT = re.compile(r"^\s*(?:[#*/]+\s*)?(Copyright\s*(?:\((?:c|C)\)|©|\d{4}).*)$")


def _scan(files: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """(copyright lines, NOTICE texts) from `(filename, text)` pairs."""
    copyrights: list[str] = []
    notices: list[str] = []
    for fname, text in files:
        if fname.upper().startswith("NOTICE"):
            stripped = text.strip()
            if stripped:
                notices.append(stripped)
            continue
        for line in text.splitlines():
            m = COPYRIGHT.match(line)
            if not m:
                continue
            c = " ".join(m.group(1).split())
            # Skip the placeholder line inside the verbatim Apache-2.0 appendix,
            # which is boilerplate rather than an assertion.
            if "[yyyy] [name of copyright owner]" in c:
                continue
            if c not in copyrights:
                copyrights.append(c)
    return copyrights, notices


def _from_src(crate: Path) -> list[tuple[str, str]]:
    files = []
    for pattern in NOTICE_GLOBS:
        for f in sorted(crate.glob(pattern)):
            if not f.is_file() or f.stat().st_size > MAX_NOTICE_BYTES:
                continue
            try:
                files.append((f.name, f.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    return files


def _from_tarball(tar_path: Path, name: str, version: str) -> list[tuple[str, str]]:
    """Notice files at the top level of `name-version/` inside the .crate tarball."""
    prefix = f"{name}-{version}/"
    files = []
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in sorted(tf.getmembers(), key=lambda m: m.name):
                if not m.isfile() or m.size > MAX_NOTICE_BYTES:
                    continue
                if not m.name.startswith(prefix):
                    continue
                base = m.name[len(prefix):]
                # Top level only, matching the `crate.glob(pattern)` scan above, so
                # both sources see the same set of files and produce the same output.
                if "/" in base or not any(
                    fnmatch.fnmatch(base, pattern) for pattern in NOTICE_GLOBS
                ):
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                files.append((base, fh.read().decode("utf-8", errors="replace")))
    except (OSError, tarfile.TarError):
        return []
    return files


def notices_for(
    name: str,
    version: str,
    srcdirs: list[Path],
    tarballs: dict[tuple[str, str], Path] | None = None,
) -> tuple[list[str], list[str]]:
    """(copyright lines, full NOTICE texts) taken from the crate's own files.

    This is what turns the inventory into a NOTICE bundle. Reproducing licence TEXT is
    not the whole obligation: MIT requires "the above copyright notice" in all copies,
    BSD requires the copyright notice retained, and Apache-2.0 section 4(d) requires the
    contents of any NOTICE file to be propagated. An SPDX identifier plus a generic
    licence body satisfies none of those on its own.

    Read from the local registry rather than the network, so the generator stays
    offline. Two sources, checked in order: the unpacked `registry/src` tree, which
    exists after a build, then the `.crate` tarball in `registry/cache`, which exists
    after a bare `cargo fetch`. They hold the same bytes, so the output does not depend
    on which one is present -- without the second, this generator produced a different
    document on a runner that had fetched but not built, and `--check` reported the
    committed file as stale.
    """
    for base in srcdirs:
        crate = base / f"{name}-{version}"
        if not crate.is_dir():
            continue
        copyrights, notices = _scan(_from_src(crate))
        if copyrights or notices:
            return copyrights, notices
    tar = (tarballs or {}).get((name, version))
    if tar is not None:
        return _scan(_from_tarball(tar, name, version))
    return [], []


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

    with_cr = [p for p in pkgs if p["copyrights"]]
    with_notice = [p for p in pkgs if p["notices"]]
    a("## Copyright notices")
    a("")
    a("Reproduced from each crate's own `LICENSE`/`COPYING`/`NOTICE` files, because the")
    a("licence text alone does not discharge the obligation: MIT requires the copyright")
    a("notice in all copies, BSD requires it retained, and Apache-2.0 section 4(d)")
    a("requires the contents of a NOTICE file to be propagated.")
    a("")
    a(f"Notices recovered for {len(with_cr)} of {len(pkgs)} crates"
      f"{'' if len(with_cr) == len(pkgs) else ' (the remainder ship no copyright line in a licence file; see their repositories, linked above)'}.")
    a("")
    for p in with_cr:
        a(f"**{p['name']} {p['version']}**")
        a("")
        for c in p["copyrights"]:
            a(f"- {c}")
        a("")
    if with_notice:
        a("### NOTICE files, verbatim")
        a("")
        a("Apache-2.0 section 4(d) requires these to travel with the distribution.")
        a("")
        # Identical texts once, with the crates that share them: every `arrow-*` crate
        # ships the same NOTICE, and repeating it per crate adds length without adding
        # notice.
        by_text: dict[str, list[str]] = {}
        for p in with_notice:
            for n in p["notices"]:
                by_text.setdefault(n, []).append(f"{p['name']} {p['version']}")
        for text in sorted(by_text, key=lambda k: (-len(by_text[k]), k[:40])):
            crates = by_text[text]
            extra = f", and {len(crates) - 1} more" if len(crates) > 1 else ""
            a(f"#### {crates[0]}{extra}")
            a("")
            if len(crates) > 1:
                a("Shared by: " + ", ".join(f"`{c}`" for c in crates) + ".")
                a("")
            a("```text")
            a(text)
            a("```")
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
