#!/usr/bin/env python3
"""Resolve the desktop application's Python environments, without installing them.

Nothing else in CI touches these two files. The conda specifications beside them are
resolved by the sidecar-import jobs, but the desktop installs neither of those: it
installs `env/console-*.txt` with `uv`, and uv resolves differently from pip. That gap
shipped a v0.3.0 installer whose first step could not resolve at all, because uv gives
an `--extra-index-url` priority over PyPI and the PyTorch index carries an old
`setuptools`, so `setuptools>=83` was unsatisfiable.

Two checks, both cheap:

1. the desktop still passes `--index-strategy unsafe-best-match`, without which the
   primary environment is unresolvable;
2. both requirement files resolve for the interpreter the application asks uv for.

Resolution only. No package is downloaded and no environment is built, so this is
seconds rather than the several hundred megabytes an install would move.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ROOT / "desktop" / "src-tauri" / "src" / "components.rs"
# (requirements file, the Python version `Env::python_version` asks uv for)
ENVS = [
    ("env/console-requirements.txt", "3.11"),
    ("env/console-ms2pip-requirements.txt", "3.11"),
]
STRATEGY = "unsafe-best-match"


def check_flag() -> list[str]:
    """The installer must still ask uv to look past the first index that has a name."""
    text = COMPONENTS.read_text(encoding="utf-8")
    if '"--index-strategy".into()' in text and f'"{STRATEGY}".into()' in text:
        return []
    return [
        f"{COMPONENTS.relative_to(ROOT)} no longer passes --index-strategy {STRATEGY} to "
        "uv. env/console-requirements.txt adds the PyTorch index, which uv prefers over "
        "PyPI, and that index carries an old setuptools; without the flag the "
        "setuptools>=83 pin makes the environment unresolvable and the desktop install "
        "fails on its first step."
    ]


def resolve(uv: str, rel: str, python: str) -> list[str]:
    req = ROOT / rel
    if not req.is_file():
        return [f"{rel} is missing"]
    proc = subprocess.run(
        # No `-o`: uv writes the resolution to stdout, and `-o -` would create a file
        # literally named "-" in the working directory.
        [uv, "pip", "compile", "--quiet", "--python-version", python,
         "--index-strategy", STRATEGY, str(req)],
        capture_output=True, text=True, cwd=ROOT,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-14:]
        return [f"{rel} does not resolve for Python {python}:\n    " + "\n    ".join(tail)]
    n = sum(1 for line in proc.stdout.splitlines()
            if line and not line[0].isspace() and not line.startswith(("#", "-")))
    if n == 0:
        return [f"{rel} resolved to no packages, which cannot be right"]
    print(f"  ok {rel}: resolves for Python {python} ({n} packages)")
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-if-no-uv", action="store_true",
                    help="succeed instead of failing when uv is absent (local convenience; "
                         "CI installs uv, so it never passes this)")
    a = ap.parse_args()

    problems = check_flag()
    uv = shutil.which("uv")
    if not uv:
        if a.skip_if_no_uv:
            print("uv not found; skipping the resolution check (--skip-if-no-uv)")
        else:
            problems.append(
                "uv was not found on PATH. It is what the desktop application installs "
                "these environments with, so this check cannot mean anything without it: "
                "`pip install uv`, or pass --skip-if-no-uv to check only the flag."
            )
    else:
        for rel, python in ENVS:
            problems += resolve(uv, rel, python)

    if problems:
        print("\nconsole environment check FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"console environments ok: the installer flag is present and "
          f"{len(ENVS)} requirement files resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
