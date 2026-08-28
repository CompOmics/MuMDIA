#!/usr/bin/env python
"""Generate `docs/23_cli_reference.md` from the `mumdia` binary's own `--help`.

A hand-written CLI reference drifts the moment a `clap` derive changes, and
nothing in CI notices. This script instead asks the built binary what its
interface is: it runs the top-level `--help`, parses the `Commands:` block to
enumerate subcommands, runs `<subcommand> --help` for each one, and writes the
result as one Markdown document. The source of truth is therefore
`rust/mumdia/crates/mumdia/src/main.rs`, not this file and not the document.

Usage:
    python ci/gen_cli_reference.py                  # write docs/23_cli_reference.md
    python ci/gen_cli_reference.py --out PATH       # write somewhere else
    python ci/gen_cli_reference.py --bin PATH       # use a specific binary
    python ci/gen_cli_reference.py --check          # exit 1 if the file is stale

Binary resolution order, first hit wins:
    1. `--bin PATH`
    2. `$MUMDIA_BIN`
    3. `rust/mumdia/target/release/mumdia`
    4. `rust/mumdia/target/release/mumdia.exe`
    5. `$MUMDIA_BIN`, if set  (a target directory redirected off the repository,
       redirected off OneDrive; see CLAUDE.md "Build and validation")
    6. `mumdia` on `PATH`

Determinism. The output must not depend on the machine that produced it, so:
the binary's own basename is rewritten to `mumdia` (a Windows build reports
`mumdia.exe` in its usage lines), every line is right-stripped (clap pads its
paragraph separators with the help indent), no timestamp or version string is
embedded, and no path is emitted. `clap` is built with only the `derive`
feature, so it does not wrap help text to the terminal width and the text is
identical piped or not.

Global flags. The four `global = true` flags are accepted on either side of the
subcommand, so clap repeats them in every subcommand help text, as it repeats
`-h, --help`. Reproducing all of that would add roughly 300 lines of noise, so
this script takes the intersection of the option entries present in EVERY
subcommand, splits `-h, --help` out of it (that one is clap's own, not a
`global = true` declaration), documents the rest once, and removes both from the
per-subcommand blocks. The set is derived, not hardcoded: a flag that stops
being global stops being filtered.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "docs" / "23_cli_reference.md"
GENERATOR = "ci/gen_cli_reference.py"
SOURCE_OF_TRUTH = "rust/mumdia/crates/mumdia/src/main.rs"

# The canonical program name written into the document.
PROG = "mumdia"

# `mumdia help --help` reprints the top-level help verbatim, so a section for it
# would duplicate the whole document head. It still appears in the table.
SKIP_SECTIONS = ("help",)

BIN_CANDIDATES = (
    "rust/mumdia/target/release/mumdia",
    "rust/mumdia/target/release/mumdia.exe",
    # A redirected target directory, named by the environment rather than baked in:
    # this used to be one contributor's absolute Windows path, in a public repo.
    os.environ.get("MUMDIA_BIN", ""),
)

# An option entry in clap's long help starts at 2 to 6 spaces followed by a dash;
# its help text is indented 10 spaces.
ENTRY_START = re.compile(r"^ {2,6}-\S")
FLAG_TOKEN = re.compile(r"(?<![\w-])--?[A-Za-z][A-Za-z0-9-]*")

# Sentence boundary for the one-line purpose column. Guarded against the usual
# abbreviations so "e.g. Foo" is not treated as a sentence end.
SENTENCE_END = re.compile(r"(?<=[A-Za-z0-9\)\]\.])\.\s+(?=[A-Z])")
ABBREVIATIONS = ("e.g", "i.e", "vs", "etc", "cf", "Fig", "no", "approx")


# ---------------------------------------------------------------------------
# Running the binary
# ---------------------------------------------------------------------------


def find_binary(explicit: str | None) -> Path:
    """Resolve the binary to interrogate, or exit with the search order shown."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            sys.exit(f"error: --bin {explicit} is not a file")
        return p
    from_env = os.environ.get("MUMDIA_BIN")
    if from_env:
        p = Path(from_env)
        if not p.is_file():
            sys.exit(f"error: $MUMDIA_BIN={from_env} is not a file")
        return p
    for rel in BIN_CANDIDATES:
        p = Path(rel)
        if not p.is_absolute():
            p = REPO_ROOT / rel
        if p.is_file():
            return p
    on_path = shutil.which(PROG)
    if on_path:
        return Path(on_path)
    sys.exit(
        "error: no mumdia binary found. Pass --bin PATH, set $MUMDIA_BIN, build\n"
        "       `cargo build --release` so one of these exists, or put mumdia on PATH:\n"
        + "".join(f"         {c}\n" for c in BIN_CANDIDATES)
    )


def run_help(binary: Path, args: list[str]) -> str:
    """Capture `--help` output, normalized so it cannot carry machine detail."""
    env = dict(os.environ)
    # Neither affects a derive-only clap build, but removing them makes the
    # capture identical under a terminal, a pipe, and CI.
    env.pop("COLUMNS", None)
    env.pop("RUST_LOG", None)
    proc = subprocess.run(
        [str(binary), *args, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    text = proc.stdout or proc.stderr
    if proc.returncode != 0 or not text.strip():
        sys.exit(
            f"error: `{PROG} {' '.join(args)} --help` failed "
            f"(exit {proc.returncode}):\n{proc.stderr.strip()}"
        )
    basename = binary.name
    if basename != PROG:
        text = text.replace(basename, PROG)
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing clap's long help
# ---------------------------------------------------------------------------


def first_sentence(text: str) -> str:
    """First sentence of a command description, for the summary table."""
    text = " ".join(text.split())
    for match in SENTENCE_END.finditer(text):
        head = text[: match.start()]
        tail_word = re.split(r"[\s(]", head)[-1]
        if tail_word.rstrip(".") in ABBREVIATIONS:
            continue
        return head + "."
    return text


def parse_commands(top_help: str) -> list[tuple[str, str]]:
    """Extract (name, description) from the top-level `Commands:` block."""
    lines = top_help.split("\n")
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith("Commands:"))
    except StopIteration:
        sys.exit("error: top-level --help has no `Commands:` block")
    out: list[tuple[str, str]] = []
    for ln in lines[start + 1 :]:
        if not ln.strip():
            break
        m = re.match(r"^ {2}(\S+)\s\s+(.*)$", ln)
        if m:
            out.append((m.group(1), m.group(2).strip()))
        elif re.match(r"^ {2}(\S+)\s*$", ln):
            out.append((ln.strip(), ""))
    if not out:
        sys.exit("error: could not parse any subcommand out of the `Commands:` block")
    return out


def split_options(help_text: str) -> tuple[list[str], list[str], list[list[str]]]:
    """Split help into (head lines, options preamble, option entries).

    The head is everything up to and including the `Options:` header. Each entry
    is the flag-spec line plus its indented help lines, trailing blanks removed.
    """
    lines = help_text.split("\n")
    try:
        opt_at = next(i for i, ln in enumerate(lines) if ln.startswith("Options:"))
    except StopIteration:
        return lines, [], []
    head = lines[: opt_at + 1]
    preamble: list[str] = []
    entries: list[list[str]] = []
    cur: list[str] | None = None
    for ln in lines[opt_at + 1 :]:
        if ENTRY_START.match(ln):
            if cur is not None:
                entries.append(cur)
            cur = [ln]
        elif cur is not None:
            cur.append(ln)
        elif ln.strip():
            preamble.append(ln)
    if cur is not None:
        entries.append(cur)
    for entry in entries:
        while entry and not entry[-1]:
            entry.pop()
    return head, preamble, entries


def entry_key(entry: list[str]) -> tuple[str, ...]:
    """Identity of an option entry: its flag names, order-independent."""
    spec = entry[0]
    return tuple(sorted(set(FLAG_TOKEN.findall(spec))))


def entry_spec(entry: list[str]) -> str:
    return entry[0].strip()


def entry_body(entry: list[str]) -> str:
    """The entry's help text as one paragraph, for the globals table."""
    body = [ln.strip() for ln in entry[1:]]
    return " ".join(w for w in body if w)


def reassemble(head: list[str], preamble: list[str], entries: list[list[str]]) -> str:
    """Rebuild a help text after dropping some option entries."""
    if not entries:
        # Drop a now-empty `Options:` header and the blank line before it.
        while head and (head[-1].startswith("Options:") or not head[-1]):
            head.pop()
        out = head
    else:
        out = list(head)
        out.extend(preamble)
        for i, entry in enumerate(entries):
            out.extend(entry)
            if i != len(entries) - 1:
                out.append("")
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------


def md_cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def fence(text: str) -> str:
    return "```text\n" + text + "\n```"


def build_document(binary: Path) -> str:
    top = run_help(binary, [])
    commands = parse_commands(top)
    names = [name for name, _ in commands]

    helps: dict[str, str] = {}
    parsed: dict[str, tuple[list[str], list[str], list[list[str]]]] = {}
    for name in names:
        if name in SKIP_SECTIONS:
            continue
        helps[name] = run_help(binary, [name])
        parsed[name] = split_options(helps[name])

    # Global flags = option entries present in every subcommand.
    per_cmd_keys = [
        {entry_key(e) for e in parsed[n][2]} for n in names if n in parsed
    ]
    top_head, top_preamble, top_entries = split_options(top)
    top_keys = {entry_key(e): e for e in top_entries}
    repeated_keys = set.intersection(*per_cmd_keys) if per_cmd_keys else set()
    repeated_keys &= set(top_keys)
    # `-h, --help` is repeated in every subcommand too, but clap adds it itself
    # rather than from a `global = true` declaration, so it is reported apart.
    help_keys = {k for k in repeated_keys if "--help" in k}
    global_keys = repeated_keys - help_keys
    global_entries = [top_keys[k] for k in sorted(global_keys)]
    help_entries = [top_keys[k] for k in sorted(help_keys)]
    help_specs = ", ".join(f"`{entry_spec(e)}`" for e in help_entries)

    # `--config` is per-subcommand, so it must never be in the global set.
    def own_flags(name: str) -> set[str]:
        return {
            flag
            for e in parsed[name][2]
            if entry_key(e) not in repeated_keys
            for flag in FLAG_TOKEN.findall(e[0])
        }

    takes_config = {n: "--config" in own_flags(n) for n in parsed}

    out: list[str] = []
    a = out.append

    a("# MuMDIA command-line reference")
    a("")
    a(f"GENERATED FILE. Do not edit. `{GENERATOR}` writes it from the")
    a(f"`{PROG}` binary's own `--help` output, so the interface described here is")
    a("the interface the binary actually has. The source of truth is the `clap`")
    a(f"derive in `{SOURCE_OF_TRUTH}`: change the flag")
    a("or its doc comment there, rebuild, and regenerate. An edit made to this")
    a("file is lost on the next run.")
    a("")
    a("```text")
    a("python ci/gen_cli_reference.py            # regenerate")
    a("python ci/gen_cli_reference.py --check    # fail if this file is stale")
    a("```")
    a("")
    a(
        "For what each stage does with these arguments, read `docs/README.md` and the"
    )
    a(
        "per-stage documents it routes to. For the configuration file passed with"
    )
    a("`--config`, read `docs/24_config_reference.md`.")
    a("")

    a("## How to read this document")
    a("")
    a("Every block below is the binary's own help text, right-stripped and with")
    a(f"the program name normalized to `{PROG}` (a Windows build reports")
    a(f"`{PROG}.exe`). Nothing is paraphrased. Two mechanical edits are applied:")
    a("")
    a(f"- the {len(global_entries)} flags clap marks `global = true`, and {help_specs}, are removed")
    a("  from the per-subcommand blocks and documented once under \"Global flags\"")
    a("  below, because clap repeats them in every subcommand;")
    a(f"- `{PROG} help` gets a table row but no section, because")
    a(f"  `{PROG} help --help` reprints the top-level help unchanged.")
    a("")

    a("## Top-level help")
    a("")
    a(fence(top))
    a("")

    a("## Global flags")
    a("")
    a(f"These {len(global_entries)} options are declared `global = true`, so they are accepted on")
    a(f"EITHER side of the subcommand: `{PROG} --threads 8 extract ...` and")
    a(f"`{PROG} extract --threads 8 ...` are equivalent and reach the same value.")
    a("They are removed from the per-subcommand blocks below to keep this document")
    a(f"readable, as is {help_specs}, which every subcommand also accepts.")
    a("")
    a("| Flag | Purpose |")
    a("|---|---|")
    for entry in global_entries + help_entries:
        a(f"| `{md_cell(entry_spec(entry))}` | {md_cell(entry_body(entry))} |")
    a("")
    a("The same text as the binary prints it:")
    a("")
    a(fence(reassemble(["Options:"], [], global_entries + help_entries)))
    a("")

    a("## Subcommands")
    a("")
    a(
        "One row per subcommand. `--config` says whether the subcommand reads a JSON"
    )
    a(
        "config file (see `docs/24_config_reference.md`); the purpose column is the"
    )
    a("first sentence of the description, with the full text in the section below.")
    a("")
    a("| Subcommand | `--config` | Purpose |")
    a("|---|---|---|")
    for name, desc in commands:
        if name in SKIP_SECTIONS:
            cell, cfg = f"`{name}`", "n/a"
        else:
            cell = f"[`{name}`](#{name})"
            cfg = "yes" if takes_config[name] else "no"
        a(f"| {cell} | {cfg} | {md_cell(first_sentence(desc))} |")
    a("")

    with_cfg = sorted(n for n, v in takes_config.items() if v)
    without_cfg = sorted(n for n, v in takes_config.items() if not v)
    a(
        f"{len(with_cfg)} of the {len(takes_config)} documented subcommands accept `--config`:"
    )
    a(" " + ", ".join(f"`{n}`" for n in with_cfg) + ".")
    a("")
    a(
        f"{len(without_cfg)} do not, so every setting they use comes from their own flags:"
    )
    a(" " + ", ".join(f"`{n}`" for n in without_cfg) + ".")
    a("")

    for name, _desc in commands:
        if name in SKIP_SECTIONS:
            continue
        head, preamble, entries = parsed[name]
        kept = [e for e in entries if entry_key(e) not in repeated_keys]
        dropped = len(entries) - len(kept)
        a(f"## {name}")
        a("")
        a(fence(reassemble(list(head), preamble, kept)))
        a("")
        if dropped:
            a(f'Plus the {dropped} repeated flags removed above: see "Global flags".')
            a("")

    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def check(out_path: Path, generated: str) -> int:
    if not out_path.is_file():
        print(
            f"error: {out_path} does not exist. Run `python {GENERATOR}`.",
            file=sys.stderr,
        )
        return 1
    committed = out_path.read_text(encoding="utf-8")
    if committed == generated:
        print(
            f"{out_path.as_posix()} is up to date "
            f"({len(generated.splitlines())} lines)."
        )
        return 0
    diff = list(
        difflib.unified_diff(
            committed.splitlines(),
            generated.splitlines(),
            fromfile=f"{out_path.as_posix()} (committed)",
            tofile="regenerated",
            lineterm="",
            n=1,
        )
    )
    added = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
    removed = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
    print(
        f"error: {out_path.as_posix()} is stale: {added} line(s) added, "
        f"{removed} removed by regeneration.",
        file=sys.stderr,
    )
    for line in diff[:60]:
        print(f"  {line}", file=sys.stderr)
    if len(diff) > 60:
        print(f"  ... and {len(diff) - 60} more diff line(s)", file=sys.stderr)
    print(
        f"\nThe CLI changed without the reference being regenerated. Run "
        f"`python {GENERATOR}` and commit the result.",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bin", help="path to the mumdia binary to interrogate")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output Markdown path")
    ap.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and exit 1 if --out differs",
    )
    args = ap.parse_args(argv)

    binary = find_binary(args.bin)
    generated = build_document(binary)
    out_path = Path(args.out)

    if args.check:
        return check(out_path, generated)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(generated)
    sections = generated.count("\n## ")
    print(
        f"wrote {out_path.as_posix()}: {len(generated.splitlines())} lines, "
        f"{sections} sections."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
