#!/usr/bin/env python
"""Fail when a tracked file cites a Markdown document that is not tracked.

MuMDIA's source carries its provenance in comments: module docs cite the document
that specifies the behavior. Several of those documents are local-only design
notes that `.gitignore` keeps out of the repository, so a public clone used to
contain about 141 references to files it does not ship, and the reader had no way
to reach the explanation. Worse, one of them (`plan.md`) quotes proprietary
constants from a closed-source engine, so tracking it to fix the references is
not an option.

This check keeps the two rules consistent: a citation must point at something a
reader actually receives, and untracked design notes must stay untracked.

Usage:
    python ci/check_doc_refs.py            # check, exit 1 on any dangling ref
    python ci/check_doc_refs.py --list     # also print every resolved reference

Resolution is by basename against the set of tracked `*.md` files, so
`docs/03_io_layer.md`, `03_io_layer.md` and `../docs/03_io_layer.md` all resolve.
That is deliberately loose: the point is to catch references to documents that do
not exist at all, not to police relative paths.

Tracked, not merely present: a document you have written but not yet staged does
not ship either, so the check reports it and says to `git add` it. In CI, where
everything is committed, the two sets coincide.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import defaultdict

# Files that legitimately name Markdown documents which do not exist yet: the
# release plan lists deliverables that release work will create. Keep this list
# short, and delete an entry as soon as its file lands.
ALLOWED_MISSING = {
    "CODE_OF_CONDUCT.md",
    "23_cli_reference.md",
    "24_config_reference.md",
}

# Documents that are untracked BY POLICY, and the few files allowed to name them.
# Those files explain the policy, so they have to say which document it applies to;
# everywhere else, naming one is the defect this check exists to catch. Each of
# these files must also state that the document is not distributed, otherwise a
# reader is still sent looking for a file they do not have.
UNTRACKED_BY_POLICY = {"plan.md", "PLAN.md"}
POLICY_FILES = {
    "ci/check_doc_refs.py",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "docs/14_build_test_deploy_gotchas.md",
    "docs/22_release_plan.md",
}

# Extensions worth scanning. Everything else in the repo is data or build input.
SCANNED_SUFFIXES = (".rs", ".py", ".md", ".toml", ".yml", ".yaml", ".json", ".sh")

# A Markdown filename, optionally with a path in front of it. `[\w./-]` keeps
# section anchors and trailing punctuation out of the match.
REF = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.md\b")


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def hint_for(base: str) -> str:
    """Tell the two local cases apart, and never advise tracking an ignored file.

    A document that exists but is not staged yet has a one-command fix. A document
    that `.gitignore` excludes deliberately does not: `plan.md` quotes third-party
    source and must stay out of the repository, so the fix there is to cite a
    tracked `docs/` page instead. Note the filesystem may be case-insensitive, so
    `PLAN.md` finds `plan.md`; asking git settles it either way.
    """
    for candidate in (base, f"docs/{base}", f"ci/{base}"):
        if not os.path.exists(candidate):
            continue
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", candidate], capture_output=True
        ).returncode == 0
        if ignored:
            return (
                f"  [{candidate} exists but .gitignore excludes it on purpose:"
                " cite a tracked docs/ page instead]"
            )
        return f"  [{candidate} exists but is not tracked: git add {candidate}]"
    return ""


def main() -> int:
    show_all = "--list" in sys.argv[1:]
    tracked = tracked_files()
    known_md = {path.rsplit("/", 1)[-1] for path in tracked if path.endswith(".md")}

    dangling: dict[str, set[str]] = defaultdict(set)
    resolved = 0
    for path in tracked:
        if not path.endswith(SCANNED_SUFFIXES):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError as exc:  # unreadable file is a CI problem in itself
            print(f"error: cannot read {path}: {exc}", file=sys.stderr)
            return 2
        for match in REF.finditer(text):
            base = match.group(0).rsplit("/", 1)[-1]
            if base in UNTRACKED_BY_POLICY and path in POLICY_FILES:
                continue
            if base in known_md or base in ALLOWED_MISSING:
                resolved += 1
                if show_all:
                    print(f"ok   {path}: {match.group(0)}")
                continue
            line = text.count("\n", 0, match.start()) + 1
            dangling[base].add(f"{path}:{line}")

    if not dangling:
        print(
            f"doc references OK: {resolved} references in tracked files, "
            f"all resolvable ({len(known_md)} tracked Markdown documents)."
        )
        return 0

    total = sum(len(v) for v in dangling.values())
    print(
        f"error: {total} reference(s) to {len(dangling)} Markdown document(s) that are "
        f"not tracked:\n",
        file=sys.stderr,
    )
    for base in sorted(dangling, key=lambda b: (-len(dangling[b]), b)):
        sites = sorted(dangling[base])
        print(f"  {base}  ({len(sites)} reference(s)){hint_for(base)}", file=sys.stderr)
        for site in sites[:12]:
            print(f"      {site}", file=sys.stderr)
        if len(sites) > 12:
            print(f"      ... and {len(sites) - 12} more", file=sys.stderr)
    print(
        "\nEither cite a tracked document under docs/ instead, or add the filename to "
        "ALLOWED_MISSING in this script if release work is about to create it. Do not "
        "track a local design note just to satisfy this check: some of them quote "
        "third-party source and must stay out of the repository. A document that is "
        "untracked by policy may be named only in the files listed in POLICY_FILES, "
        "which exist to explain the policy.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
