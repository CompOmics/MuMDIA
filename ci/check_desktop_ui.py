#!/usr/bin/env python3
"""Check that the desktop frontend and its Rust backend agree.

The interface has no build step and no framework, which keeps Node out of the
release pipeline but also means nothing catches a typo in an element id or a command
name. Those are exactly the mistakes that survive review and fail on a user's
machine, so they are checked here instead:

- every `$("id")` the frontend looks up exists in `index.html`;
- every `invoke("name")` it calls is registered in `generate_handler!`;
- every registered command is called by something, so a command that lost its caller
  is noticed rather than left as dead weight.

Usage:
    python ci/check_desktop_ui.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "desktop" / "ui" / "index.html"
JS = ROOT / "desktop" / "ui" / "app.js"
MAIN = ROOT / "desktop" / "src-tauri" / "src" / "main.rs"


def main() -> int:
    for f in (HTML, JS, MAIN):
        if not f.is_file():
            print(f"missing {f.relative_to(ROOT).as_posix()}", file=sys.stderr)
            return 1

    html = HTML.read_text(encoding="utf-8")
    js = JS.read_text(encoding="utf-8")
    rust = MAIN.read_text(encoding="utf-8")

    problems: list[str] = []

    ids = set(re.findall(r'id="([^"]+)"', html))
    used = set(re.findall(r'\$\("([^"]+)"\)', js))
    if missing := sorted(used - ids):
        problems.append(f"element ids used by app.js but absent from index.html: {missing}")

    called = set(re.findall(r'invoke\("([^"]+)"', js))
    handler = re.search(r"generate_handler!\[(.*?)\]", rust, re.S)
    if handler is None:
        problems.append("no generate_handler! block found in main.rs")
        registered: set[str] = set()
    else:
        registered = {x.strip() for x in handler.group(1).split(",") if x.strip()}

    if unknown := sorted(called - registered):
        problems.append(f"commands called by app.js but not registered in Rust: {unknown}")
    if unused := sorted(registered - called):
        problems.append(f"commands registered in Rust but never called: {unused}")

    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        return 1

    print(
        f"desktop frontend ok: {len(ids)} element ids, "
        f"{len(called)} commands, both directions agree."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
