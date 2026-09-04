#!/usr/bin/env python3
"""Reject workflow YAML that GitHub would refuse, before pushing it.

The specific failure this exists for: deleting a matrix entry left an orphaned
`exe: ""` under the previous entry, so that mapping had the key twice. PyYAML's
`safe_load` accepts a duplicate key silently, keeping the last value, so every local
check passed. GitHub's parser rejects it outright:

    HTTP 422: failed to parse workflow: (Line: 143, Col: 13): 'exe' is already defined

which surfaces only when a workflow is dispatched or a tag is pushed. For
`release.yml` that means the error appears at the moment a release is attempted.

Two checks, both about failures that are silent locally:

1. duplicate mapping keys, at any depth, rejected;
2. the structural keys a workflow needs (`on`, `jobs`, and a `runs-on` or a `uses`
   per job) present.

This is not a substitute for `actionlint`, which checks expressions, action inputs
and shell syntax. It is the subset that needs no external binary, and it covers the
class of error that reached the remote.

Usage:
    python ci/check_workflows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


class StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate mapping keys."""


def _no_duplicates(loader: StrictLoader, node, deep: bool = False) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            mark = key_node.start_mark
            raise yaml.constructor.ConstructorError(
                None, None,
                f"duplicate key {key!r} at line {mark.line + 1}, column {mark.column + 1}; "
                f"GitHub rejects this with \"'{key}' is already defined\"",
                node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
)


def main() -> int:
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    if not files:
        print(f"no workflows found under {WORKFLOWS}", file=sys.stderr)
        return 1

    failed = False
    for f in files:
        try:
            doc = yaml.load(f.read_text(encoding="utf-8"), Loader=StrictLoader)
        except yaml.YAMLError as e:
            print(f"{f.relative_to(ROOT)}: {e}", file=sys.stderr)
            failed = True
            continue

        problems = []
        if not isinstance(doc, dict):
            problems.append("top level is not a mapping")
        else:
            # `on:` is parsed by PyYAML as the boolean True, since YAML 1.1 treats
            # `on` as a boolean. Accept either spelling rather than pretending the
            # file is wrong.
            if "on" not in doc and True not in doc:
                problems.append("no `on:` trigger")
            jobs = doc.get("jobs")
            if not isinstance(jobs, dict) or not jobs:
                problems.append("no `jobs:`")
            else:
                for name, job in jobs.items():
                    if not isinstance(job, dict):
                        problems.append(f"job {name} is not a mapping")
                    elif "runs-on" not in job and "uses" not in job:
                        problems.append(f"job {name} has neither `runs-on` nor `uses`")
        for p in problems:
            print(f"{f.relative_to(ROOT)}: {p}", file=sys.stderr)
            failed = True
        if not problems:
            n = len(doc.get("jobs", {}))
            print(f"ok {f.relative_to(ROOT)} ({n} job{'s' if n != 1 else ''})")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
