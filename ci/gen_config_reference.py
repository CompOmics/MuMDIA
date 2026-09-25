#!/usr/bin/env python
"""Generate `docs/24_config_reference.md` from the Rust config source.

Every MuMDIA setting is declared once, in
`rust/mumdia/crates/mumdia-core/src/config.rs`: the field, its type, its default
(in the matching `impl Default`), and the doc comment that says what it means and
whether it is safe to enable. A hand-written config reference would restate all
four and drift from every one of them. This script instead parses that file and
emits the reference, so the document cannot describe a field the code does not
have or a default the code does not use.

It also emits the environment-variable table, which exists nowhere else: the
variables are scattered through `docs/13_sidecars.md` and the script docstrings.
Those are collected by scanning the crates for `std::env::var`/`var_os` and the
sidecars for `os.environ`/`os.getenv`, plus the sites where either side SETS a
variable for a child process.

Usage:
    python ci/gen_config_reference.py              # write docs/24_config_reference.md
    python ci/gen_config_reference.py --out PATH   # write somewhere else
    python ci/gen_config_reference.py --check      # exit 1 if the file is stale
    python ci/gen_config_reference.py --self-test  # only the blank-line invariance check

Parsing approach. A line state machine with brace/bracket/quote tracking, not a
Rust parser: enough for this one 1,800-line file, and it fails loudly rather than
quietly when an assumption breaks. Every field in the file is `    pub name:
Type,` on one line, and every default is a `Name: expr,` entry inside
`impl Default for X { fn default() -> Self { Self { ... } } }`. A field whose
default cannot be resolved is REPORTED in the generated document, never dropped:
a silently missing row is exactly the drift this script exists to prevent.

Determinism. Sections follow the declaration order of `Config`, which is pipeline
order and therefore meaningful. Enums, environment variables, and the unresolved
lists are sorted by name. No timestamp, path, or version is embedded.

Stable anchors. No line number is emitted. A site is cited as `path::scope`: the
tracked file plus the enclosing Rust `fn` (qualified by its `impl` type, `trait`
and inline `mod`) or Python `def` (qualified by its class), nested scopes joined,
`<module>` outside any function; a config struct or enum is cited by its name.
The reference used to cite `path:line`, so every merge that moved lines in
`rescore.rs`, `config.rs`, `main.rs` or a sidecar script made the committed
document stale on every other open pull request, although no variable, field or
default had changed. `--check` also regenerates from copies of every input with
blank lines inserted and fails if the result differs, so a line dependency cannot
come back unnoticed.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_RS = REPO_ROOT / "rust" / "mumdia" / "crates" / "mumdia-core" / "src" / "config.rs"
CRATES_DIR = REPO_ROOT / "rust" / "mumdia" / "crates"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_OUT = REPO_ROOT / "docs" / "24_config_reference.md"
DEFAULT_SCHEMA_OUT = REPO_ROOT / "configs" / "config-schema.json"
GENERATOR = "ci/gen_config_reference.py"
CONFIG_RS_REL = "rust/mumdia/crates/mumdia-core/src/config.rs"

# Which tracked document explains the stage a config section drives. Keyed by the
# JSON path of the section; validated against docs/ at generation time, so a
# renamed document fails the run instead of producing a dead reference.
STAGE_DOC = {
    "": "docs/02_config_and_data_model.md",
    "prescan": "docs/21_prescan.md",
    "digest": "docs/05_digest_peptidoforms.md",
    "digest.decoy": "docs/05_digest_peptidoforms.md",
    "peptidoforms": "docs/05_digest_peptidoforms.md",
    "predict_frag": "docs/06_predict_frag_index_matchers.md",
    "search_seed": "docs/07_search_seed.md",
    "rt_im_train": "docs/08_rt_im_train.md",
    "extract": "docs/09_extract.md",
    "extract.claim_cues": "docs/09_extract.md",
    "features": "docs/10_features.md",
    "compete": "docs/11_compete_rescore_fdr.md",
    "rescore": "docs/11_compete_rescore_fdr.md",
    "quant": "docs/12_quant_lfq_align_mbr_report_audit.md",
    "mbr": "docs/12_quant_lfq_align_mbr_report_audit.md",
    "experiment": "docs/01_overview_and_dataflow.md",
}
# Structs that are not a section of their own (they appear inside a Vec).
ITEM_STRUCT_DOC = {"ResidueMod": "docs/05_digest_peptidoforms.md"}

# A doc comment carrying any of these marks the field as experimental: not part of
# the shipped, validated default chain. Word-boundary matched, case-insensitive.
GATE_MARKERS = (
    "benchmark-gated",
    "gated",
    "not yet wired",
    "diagnostic",
    "not currently",
    "do not default",
)

RUST_STRING_SUFFIXES = (".to_string()", ".to_owned()", ".into()", ".to_vec()")
INT_TYPES = ("u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64", "isize")
FLOAT_TYPES = ("f32", "f64")


# ---------------------------------------------------------------------------
# Small scanning helpers
# ---------------------------------------------------------------------------


# A Rust char literal. A bare `'` that does not match this is a lifetime tick
# (`&'static str`, `'a`), which must NOT open a literal: treating it as one
# desynchronizes every quote and brace after it.
CHAR_LIT = re.compile(r"'(?:[^'\\\n]|\\.)'")


def strip_comments(line: str) -> str:
    """Remove a trailing `//` comment, respecting string and char literals."""
    out: list[str] = []
    i = 0
    in_str = False
    while i < len(line):
        c = line[i]
        if in_str:
            if c == "\\":
                out.append(line[i : i + 2])
                i += 2
                continue
            if c == '"':
                in_str = False
            out.append(c)
            i += 1
            continue
        if c == "/" and line[i : i + 2] == "//":
            break
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "'":
            m = CHAR_LIT.match(line, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
        out.append(c)
        i += 1
    return "".join(out)


def code_chars(text: str):
    """Yield `(index, char)` for the characters outside string and char literals."""
    i = 0
    in_str = False
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if c == "'":
            m = CHAR_LIT.match(text, i)
            if m:
                i = m.end()
                continue
        yield i, c
        i += 1


def decomment(text: str) -> str:
    """Strip `//` comments line by line, keeping every line so numbers hold."""
    return "\n".join(strip_comments(ln) for ln in text.split("\n"))


def collect_block(lines: list[str], start: int) -> tuple[int, str]:
    """Return (next line index, comment-free text) for the braced item at `start`."""
    depth = 0
    started = False
    out: list[str] = []
    i = start
    while i < len(lines):
        code = strip_comments(lines[i])
        out.append(code)
        for _, c in code_chars(code):
            if c == "{":
                depth += 1
                started = True
            elif c == "}":
                depth -= 1
        if started and depth <= 0:
            return i + 1, "\n".join(out)
        i += 1
    return i, "\n".join(out)


def match_brace(text: str, open_at: int) -> int | None:
    """Index of the `}` closing the `{` at `open_at`, or None."""
    depth = 0
    for idx, c in code_chars(text):
        if idx < open_at:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return idx
    return None


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on `sep` at bracket depth zero, ignoring separators inside literals.

    Angle brackets are deliberately NOT counted: a comparison operator would
    unbalance them, and no expression this parser splits needs generic nesting.
    """
    parts: list[str] = []
    depth = 0
    last = 0
    for idx, c in code_chars(text):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == sep and depth == 0:
            parts.append(text[last:idx])
            last = idx + 1
    parts.append(text[last:])
    return [p.strip() for p in parts if p.strip()]


def doc_paragraph(doc_lines: list[str]) -> str:
    """Join `///` lines into one unwrapped paragraph fit for a table cell."""
    text = " ".join(ln.strip() for ln in doc_lines)
    text = " ".join(text.split())
    # Rust intra-doc links: [`Thing`] -> `Thing`, [Thing] -> `Thing`.
    text = re.sub(r"\[`([^`\]]+)`\]", r"`\1`", text)
    text = re.sub(r"\[([A-Z][A-Za-z0-9_:]*)\]", r"`\1`", text)
    return text


def md_cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def snake_case(variant: str) -> str:
    """serde's `rename_all = "snake_case"` on a CamelCase variant name."""
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", variant)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def eval_numeric(expr: str) -> float | int | None:
    """Evaluate a literal arithmetic expression such as `1.0 / 3.0`, else None."""
    cleaned = expr.replace("_", "")
    for suffix in INT_TYPES + FLOAT_TYPES:
        cleaned = re.sub(rf"(\d){suffix}\b", r"\1", cleaned)
    if not re.fullmatch(r"[0-9eE.+\-*/() ]+", cleaned):
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return None

    def walk(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = walk(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError("unsupported")

    try:
        return walk(tree)
    except (ValueError, ZeroDivisionError):
        return None


# ---------------------------------------------------------------------------
# Parsed model
# ---------------------------------------------------------------------------


class Field:
    def __init__(self, name: str, rtype: str, doc: str, attrs: list[str]):
        self.name = name
        self.rtype = rtype
        self.doc = doc
        self.attrs = attrs
        self.default_expr: str | None = None
        self.default_rendered: str | None = None


class Struct:
    def __init__(self, name: str, doc: str):
        self.name = name
        self.doc = doc
        self.fields: list[Field] = []
        self.has_default_impl = False


class EnumVariant:
    def __init__(self, name: str, doc: str, is_default: bool):
        self.name = name
        self.doc = doc
        self.is_default = is_default


class Enum:
    def __init__(self, name: str, doc: str, rename_all: str | None):
        self.name = name
        self.doc = doc
        self.rename_all = rename_all
        self.variants: list[EnumVariant] = []

    def serde_name(self, variant: str) -> str:
        if self.rename_all == "snake_case":
            return snake_case(variant)
        return variant

    def default_variant(self) -> EnumVariant | None:
        for v in self.variants:
            if v.is_default:
                return v
        return None


# ---------------------------------------------------------------------------
# config.rs parsing
# ---------------------------------------------------------------------------

FIELD_RE = re.compile(r"^\s*pub\s+([a-z_][a-z0-9_]*)\s*:\s*(.+?),\s*$")
STRUCT_RE = re.compile(r"^(?:pub\s+)?struct\s+([A-Za-z0-9_]+)\s*\{")
ENUM_RE = re.compile(r"^(?:pub\s+)?enum\s+([A-Za-z0-9_]+)\s*\{")
DEFAULT_IMPL_RE = re.compile(r"^impl\s+Default\s+for\s+([A-Za-z0-9_]+)\s*\{")
RENAME_ALL_RE = re.compile(r'rename_all\s*=\s*"([a-z_]+)"')
VARIANT_RE = re.compile(r"^\s{4}([A-Z][A-Za-z0-9_]*)\s*,?\s*$")


def parse_config_rs(text: str) -> tuple[dict[str, Struct], dict[str, Enum], list[str]]:
    """Parse structs, enums, and `impl Default` blocks out of config.rs."""
    lines = text.split("\n")
    # The test module at the end holds raw strings and config JSON; stop there.
    end = len(lines)
    for i, ln in enumerate(lines):
        if ln.startswith("#[cfg(test)]"):
            end = i
            break
    lines = lines[:end]

    structs: dict[str, Struct] = {}
    enums: dict[str, Enum] = {}
    warnings: list[str] = []

    doc: list[str] = []
    attrs: list[str] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if stripped.startswith("///"):
            doc.append(stripped[3:])
            i += 1
            continue
        if stripped.startswith("#["):
            attrs.append(stripped)
            i += 1
            continue
        if not stripped or stripped.startswith("//"):
            if not stripped:
                doc = []
                attrs = []
            i += 1
            continue

        m = STRUCT_RE.match(raw)
        if m:
            struct = Struct(m.group(1), doc_paragraph(doc))
            i = parse_struct_body(lines, i + 1, struct, warnings)
            structs[struct.name] = struct
            doc, attrs = [], []
            continue

        m = ENUM_RE.match(raw)
        if m:
            rename = None
            for at in attrs:
                found = RENAME_ALL_RE.search(at)
                if found:
                    rename = found.group(1)
            enum = Enum(m.group(1), doc_paragraph(doc), rename)
            i = parse_enum_body(lines, i + 1, enum)
            enums[enum.name] = enum
            doc, attrs = [], []
            continue

        m = DEFAULT_IMPL_RE.match(raw)
        if m:
            target = m.group(1)
            i, assignments = parse_default_impl(lines, i + 1)
            struct = structs.get(target)
            if struct is None:
                warnings.append(
                    f"`impl Default for {target}` has no struct parsed before it"
                )
            else:
                struct.has_default_impl = True
                known = {f.name for f in struct.fields}
                for name, expr in assignments.items():
                    if name not in known:
                        warnings.append(
                            f"`impl Default for {target}` sets unknown field `{name}`"
                        )
                for field in struct.fields:
                    if field.name in assignments:
                        field.default_expr = assignments[field.name]
            doc, attrs = [], []
            continue

        doc, attrs = [], []
        i += 1

    return structs, enums, warnings


def parse_struct_body(
    lines: list[str], start: int, struct: Struct, warnings: list[str]
) -> int:
    doc: list[str] = []
    attrs: list[str] = []
    i = start
    while i < len(lines):
        raw = lines[i]
        if raw.startswith("}"):
            return i + 1
        stripped = raw.strip()
        if stripped.startswith("///"):
            doc.append(stripped[3:])
            i += 1
            continue
        if stripped.startswith("#["):
            attrs.append(stripped)
            i += 1
            continue
        if not stripped or stripped.startswith("//"):
            i += 1
            continue
        m = FIELD_RE.match(strip_comments(raw))
        if m:
            struct.fields.append(
                Field(m.group(1), m.group(2).strip(), doc_paragraph(doc), attrs)
            )
        elif "pub " in stripped:
            warnings.append(
                f"{CONFIG_RS_REL}::{struct.name}: unparsed member: {stripped}"
            )
        doc, attrs = [], []
        i += 1
    warnings.append(f"unterminated struct `{struct.name}`")
    return i


def parse_enum_body(lines: list[str], start: int, enum: Enum) -> int:
    doc: list[str] = []
    is_default = False
    i = start
    while i < len(lines):
        raw = lines[i]
        if raw.startswith("}"):
            return i + 1
        stripped = raw.strip()
        if stripped.startswith("///"):
            doc.append(stripped[3:])
            i += 1
            continue
        if stripped.startswith("#["):
            if "default" in stripped:
                is_default = True
            i += 1
            continue
        m = VARIANT_RE.match(strip_comments(raw))
        if m:
            enum.variants.append(EnumVariant(m.group(1), doc_paragraph(doc), is_default))
            doc, is_default = [], False
            i += 1
            continue
        if stripped:
            doc, is_default = [], False
        i += 1
    return i


def parse_default_impl(lines: list[str], start: int) -> tuple[int, dict[str, str]]:
    """Collect `field: expr` pairs from the `Self { ... }` of a `Default` impl.

    `fn default() -> Self {` also reads as `Self {`, so a candidate whose preceding
    text ends in `->` is the return type, not the struct literal, and is skipped.
    """
    end, block = collect_block(lines, start - 1)
    for m in re.finditer(r"\bSelf\s*\{", block):
        if block[: m.start()].rstrip().endswith("->"):
            continue
        open_at = block.index("{", m.start())
        close_at = match_brace(block, open_at)
        if close_at is None:
            return end, {}
        return end, parse_assignments(block[open_at + 1 : close_at])
    return end, {}


def parse_assignments(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in split_top_level(body):
        if ":" not in part:
            continue
        name, _, expr = part.partition(":")
        name = name.strip()
        if re.fullmatch(r"[a-z_][a-z0-9_]*", name):
            out[name] = expr.strip()
    return out


# ---------------------------------------------------------------------------
# Default rendering
# ---------------------------------------------------------------------------


def base_type(rtype: str) -> str:
    return rtype.strip()


def type_default(rtype: str, enums: dict[str, Enum], structs: dict[str, Struct]) -> str | None:
    """The value `t()` (i.e. `T::default()`) produces, rendered for the doc."""
    t = base_type(rtype)
    if t == "bool":
        return "false"
    if t in INT_TYPES:
        return "0"
    if t in FLOAT_TYPES:
        return "0.0"
    if t == "String":
        return '""'
    if t.startswith("Option<"):
        return "null"
    if t.startswith("Vec<"):
        return "[]"
    if t in enums:
        variant = enums[t].default_variant()
        if variant is not None:
            return f"`{enums[t].serde_name(variant.name)}`"
        return None
    if t in structs:
        return f"the `{t}` section's own defaults"
    return None


def render_default(
    expr: str, rtype: str, enums: dict[str, Enum], structs: dict[str, Struct]
) -> str | None:
    """Turn a Rust default expression into the JSON-ish value a config would set."""
    e = expr.strip().rstrip(",").strip()
    for suffix in RUST_STRING_SUFFIXES:
        if e.endswith(suffix):
            e = e[: -len(suffix)].strip()
    if e in ("t()", "Default::default()") or re.fullmatch(
        r"[A-Za-z0-9_]+::default\(\)", e
    ):
        target = rtype
        m = re.fullmatch(r"([A-Za-z0-9_]+)::default\(\)", e)
        if m:
            target = m.group(1)
        return type_default(target, enums, structs)
    if e == "None":
        return "null"
    if e in ("Vec::new()", "vec![]", "Vec::default()"):
        return "[]"
    if e in ("true", "false"):
        return e
    if re.fullmatch(r'"[^"]*"', e):
        return e
    if re.fullmatch(r"'[^']*'", e):
        return '"' + e[1:-1] + '"'
    m = re.fullmatch(r"([A-Za-z0-9_]+)::([A-Za-z0-9_]+)", e)
    if m and m.group(1) in enums:
        enum = enums[m.group(1)]
        if any(v.name == m.group(2) for v in enum.variants):
            return f"`{enum.serde_name(m.group(2))}`"
        return None
    if e.startswith("vec![") and e.endswith("]"):
        inner = e[len("vec![") : -1]
        items = [render_struct_literal(x, enums, structs) for x in split_top_level(inner)]
        if all(items):
            return "[" + ", ".join(i for i in items if i) + "]"
        return None
    numeric = eval_numeric(e)
    if numeric is not None:
        literal = " ".join(e.split())
        if re.fullmatch(r"-?[0-9][0-9_]*(\.[0-9]+)?([eE][+-]?[0-9]+)?", e):
            return literal
        return f"{literal} ({numeric:.6g})"
    return None


def render_struct_literal(
    expr: str, enums: dict[str, Enum], structs: dict[str, Struct]
) -> str | None:
    """Render `Name { field: value, ... }` as `{"field": value, ...}`."""
    e = expr.strip()
    m = re.fullmatch(r"([A-Za-z0-9_]+)\s*\{(.*)\}", e, re.S)
    if not m:
        return render_default(e, "", enums, structs)
    name = m.group(1)
    struct = structs.get(name)
    parts: list[str] = []
    for name_expr in split_top_level(m.group(2)):
        if ":" not in name_expr:
            return None
        fname, _, fexpr = name_expr.partition(":")
        fname = fname.strip()
        ftype = ""
        if struct:
            for f in struct.fields:
                if f.name == fname:
                    ftype = f.rtype
        rendered = render_default(fexpr, ftype, enums, structs)
        if rendered is None:
            return None
        parts.append(f'"{fname}": {rendered.strip("`")}')
    return "{" + ", ".join(parts) + "}"


def gate_markers(doc: str) -> list[str]:
    found = []
    low = doc.lower()
    for marker in GATE_MARKERS:
        if re.search(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", low):
            found.append(marker)
    # "benchmark-gated" already implies "gated"; do not report both.
    if "benchmark-gated" in found and "gated" in found:
        found.remove("gated")
    return found


# ---------------------------------------------------------------------------
# Reachability from `Config`
# ---------------------------------------------------------------------------


def inner_types(rtype: str) -> list[str]:
    """Type names mentioned in a field type, outermost first."""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rtype)


def walk_sections(
    structs: dict[str, Struct]
) -> tuple[list[tuple[str, str, Struct]], list[tuple[list[str], Struct]]]:
    """Depth-first walk from `Config`, returning (path, kind, struct) sections.

    Declaration order is pipeline order, so it is preserved. A struct reachable
    only inside a `Vec` is returned separately as an item struct, together with
    EVERY field path whose elements it types (`ResidueMod` types both
    `fixed_mods` and `variable_mods`).
    """
    root = structs.get("Config")
    if root is None:
        sys.exit(f"error: no `pub struct Config` found in {CONFIG_RS_REL}")
    sections: list[tuple[str, str, Struct]] = [("", "section", root)]
    item_paths: dict[str, list[str]] = {}
    item_structs: dict[str, Struct] = {}

    def visit(prefix: str, struct: Struct) -> None:
        for field in struct.fields:
            for tname in inner_types(field.rtype):
                child = structs.get(tname)
                if child is None or child is struct:
                    continue
                path = f"{prefix}.{field.name}" if prefix else field.name
                if field.rtype.startswith("Vec<"):
                    item_structs[tname] = child
                    item_paths.setdefault(tname, []).append(path)
                else:
                    sections.append((path, "section", child))
                    visit(path, child)

    visit("", root)
    items = [
        (item_paths[name], item_structs[name]) for name in sorted(item_structs)
    ]
    return sections, items


def reachable_enums(
    sections: list[tuple[str, str, Struct]],
    items: list[tuple[list[str], Struct]],
    enums: dict[str, Enum],
) -> list[Enum]:
    used: set[str] = set()
    for _, _, struct in sections:
        for field in struct.fields:
            used.update(t for t in inner_types(field.rtype) if t in enums)
    for _, struct in items:
        for field in struct.fields:
            used.update(t for t in inner_types(field.rtype) if t in enums)
    return [enums[name] for name in sorted(used)]


# ---------------------------------------------------------------------------
# Named profiles (`Config::apply_profile`)
# ---------------------------------------------------------------------------


def tracked_files(root: Path, suffix: str, fallback, recursive: bool = True) -> list[Path]:
    """Files under `root` with `suffix` that git tracks; `fallback()` when git cannot say.

    `git ls-files` is asked for `root` and the answer is filtered, so the input set of
    this generator is the tracked source and nothing a developer left beside it.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", str(root)],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return fallback()
    files = []
    for rel in out.decode("utf-8").split("\0"):
        if not rel or not rel.endswith(suffix):
            continue
        path = REPO_ROOT / rel
        if not recursive and path.parent != root:
            continue
        if path.is_file():
            files.append(path)
    if not files:
        return fallback()
    return sorted(files)


def parse_profiles(text: str) -> dict[str, list[tuple[str, str]]]:
    """Extract `--profile NAME` overrides from `Config::apply_profile`."""
    m = re.search(r"pub fn apply_profile\(.*?\n    \}\n", text, re.S)
    if not m:
        return {}
    body = m.group(0)
    out: dict[str, list[tuple[str, str]]] = {}
    for arm in re.finditer(r'"([a-z0-9_-]+)"\s*=>\s*\{(.*?)\n            \}', body, re.S):
        assignments = re.findall(
            r"self\.([a-z_][a-z0-9_.]*)\s*=\s*([^;]+);", arm.group(2)
        )
        out[arm.group(1)] = [(k, " ".join(v.split())) for k, v in assignments]
    return out


# ---------------------------------------------------------------------------
# Stable source anchors
# ---------------------------------------------------------------------------
#
# A site is cited as `path::scope`, never `path:line`. A line number changes
# whenever anything above it changes, so a cited line made the committed reference
# stale on every merge that touched a large file, and two open pull requests that
# each touched `rescore.rs` conflicted in this document although neither changed an
# environment variable. The enclosing function changes only when the read moves to
# another function or the function is renamed, and then the reference should
# change.

# The scope of a read outside any function: a Rust `static` initializer, or Python
# module-level code. Python's own name for that scope.
MODULE_SCOPE = "<module>"

RUST_FN_DECL = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")
RUST_IMPL = re.compile(r"\bimpl\b")
# An inline module or a trait: both name the functions inside them.
RUST_NAMED_BLOCK = re.compile(r"\b(mod|trait)\s+([A-Za-z_][A-Za-z0-9_]*)")
# What may stand between an item keyword and the end of the previous item:
# visibility (`pub`, `pub(crate)`, `pub(in path)`) and the qualifiers of an
# `unsafe impl`, an `unsafe trait`, an `auto trait` or a `default impl`.
RUST_ITEM_QUALIFIER = re.compile(r"(?:pub(?:\s*\([^()]*\))?|unsafe|auto|default)\Z")
RUST_PATH = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)*")
RUST_RAW_STR = re.compile(r'b?r(#*)"')


def mask_rust_literals(text: str) -> str:
    """`text` with every literal and comment body replaced by spaces.

    Covers string, byte string, raw string (`r#"..."#`) and char literals, and `//`
    and (nested) `/* */` comments. Length and newlines are preserved, so an offset
    into the result is an offset into `text`, and a brace or `fn` that survives is
    code. `code_chars` is not enough here: it reads the backslash of `r"\\"` as an
    escape, and a `fn` or `{` inside a string would open a false scope.
    """
    out = list(text)
    n = len(text)

    def blank(start: int, end: int) -> None:
        for k in range(start, min(end, n)):
            if out[k] != "\n":
                out[k] = " "

    i = 0
    while i < n:
        c = text[i]
        if c == "/" and text.startswith("//", i):
            end = text.find("\n", i)
            end = n if end == -1 else end
            blank(i, end)
            i = end
            continue
        if c == "/" and text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
            continue
        if c in "rb" and not (i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = RUST_RAW_STR.match(text, i)
            if m:
                close = '"' + m.group(1)
                end = text.find(close, m.end())
                end = n if end == -1 else end + len(close)
                blank(i, end)
                i = end
                continue
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            blank(i, j + 1)
            i = j + 1
            continue
        if c == "'":
            m = CHAR_LIT.match(text, i)
            if m:
                blank(i, m.end())
                i = m.end()
                continue
        i += 1
    return "".join(out)


def rust_impl_type(header: str) -> str | None:
    """The self type of an `impl` block, from the header between `impl` and `{`.

    `impl<T: Bound> Trait for path::Type<T> where ...` names `Type`: the impl's own
    generics and any `where` clause are dropped, then the last path segment of the
    first type after the last `for`. Lifetimes are removed first, so `&'a Type`
    names `Type` rather than `a`. None when no type name is left.
    """
    h = re.sub(r"'[A-Za-z_][A-Za-z0-9_]*", " ", header.replace("->", "  "))
    h = re.split(r"\bwhere\b", h, maxsplit=1)[0].lstrip()
    if h.startswith("<"):
        depth = 0
        for k, c in enumerate(h):
            if c == "<":
                depth += 1
            elif c == ">":
                depth -= 1
                if depth == 0:
                    h = h[k + 1 :]
                    break
    h = re.split(r"\bfor\b", h)[-1]
    for m in RUST_PATH.finditer(h):
        if m.group(0) in ("dyn", "mut", "const"):
            continue
        return re.split(r"\s*::\s*", m.group(0))[-1]
    return None


def rust_item_position(code: str, start: int) -> bool:
    """Whether the keyword at `start` of masked `code` begins an item.

    It does when only whitespace, visibility and the qualifiers `unsafe`, `auto`
    and `default` stand between it and a `{`, `}`, `;`, an attribute's `]` or the
    start of the file. `-> impl Iterator<..> {` and `x: impl Trait` fail the test:
    there `impl` is part of a type.
    """
    k = start
    while True:
        while k > 0 and code[k - 1].isspace():
            k -= 1
        if k == 0 or code[k - 1] in "{};]":
            return True
        m = RUST_ITEM_QUALIFIER.search(code, max(0, k - 80), k)
        if m is None:
            return False
        if m.start() > 0 and (code[m.start() - 1].isalnum() or code[m.start() - 1] == "_"):
            return False
        k = m.start()


def rust_scopes(text: str, rel: str) -> list[tuple[int, int, str, str]]:
    """`(start, end, name, kind)` of every Rust `fn`, `impl`, `mod` and `trait` body.

    A body opens at the first `{` outside parentheses and brackets after the
    keyword; a `;` there first is a declaration without a body (a trait method, or
    `mod name;`). Brace pairs are matched once for the whole file, so the cost is
    linear in its size. An `impl`, `mod` or `trait` counts only in item position
    (`rust_item_position`): `-> impl Iterator<..> {` is a return type, and its `{`
    opens the function body, not an impl block. An impl span carries the self type
    and a `mod` or `trait` span its own name, so a method is cited as
    `Type::method`, a trait's default method as `Trait::method` and a function in
    an inline module as `module::function`, as Python cites `Class.method`.

    The braces left after masking must balance. If they do not, a literal or a
    comment was masked wrongly, and every later read in the file would be cited
    under the wrong function or as `<module>`. The blank-line self-check cannot
    see that, because a wrong scope is as stable under inserted blank lines as a
    right one, so the file is rejected here instead. `rel` names it in the error.
    """
    code = mask_rust_literals(text)
    close_of: dict[int, int] = {}
    stack: list[int] = []
    unmatched_close = 0
    for idx, c in enumerate(code):
        if c == "{":
            stack.append(idx)
        elif c == "}":
            if stack:
                close_of[stack.pop()] = idx
            else:
                unmatched_close += 1
    if stack or unmatched_close:
        sys.exit(
            f"error: {rel}: unbalanced braces after masking literals and comments "
            f"({len(stack)} unclosed '{{', {unmatched_close} unmatched '}}'), so the "
            "enclosing function of an environment read cannot be cited"
        )

    def body_open(after: int) -> int | None:
        depth = 0
        for j in range(after, len(code)):
            c = code[j]
            if c in "([":
                depth += 1
            elif c in ")]":
                depth -= 1
            elif depth == 0 and c == ";":
                return None
            elif depth == 0 and c == "{":
                return j if j in close_of else None
        return None

    spans: list[tuple[int, int, str, str]] = []
    for m in RUST_FN_DECL.finditer(code):
        j = body_open(m.end())
        if j is not None:
            spans.append((m.start(), close_of[j], m.group(1), "fn"))
    for m in RUST_IMPL.finditer(code):
        if not rust_item_position(code, m.start()):
            continue
        j = body_open(m.end())
        if j is None:
            continue
        name = rust_impl_type(code[m.end() : j])
        if name:
            spans.append((m.start(), close_of[j], name, "impl"))
    for m in RUST_NAMED_BLOCK.finditer(code):
        if not rust_item_position(code, m.start()):
            continue
        j = body_open(m.end())
        if j is not None:
            spans.append((m.start(), close_of[j], m.group(2), m.group(1)))
    return spans


def rust_scope_at(spans: list[tuple[int, int, str, str]], offset: int) -> str:
    """The enclosing `fn` of `offset`, qualified by every enclosing item.

    `Type::method`, `Trait::method`, `module::function`, `outer::inner`, or
    `<module>` outside any function. An `impl`, `mod` or `trait` alone is not a
    function scope, so a read in an associated `const` or a `static` inside a
    module is `<module>` like one in a top-level `static`.
    """
    chain = sorted(span for span in spans if span[0] <= offset <= span[1])
    if not any(kind == "fn" for _, _, _, kind in chain):
        return MODULE_SCOPE
    return "::".join(name for _, _, name, _ in chain)


def python_scopes(text: str, rel: str) -> list[tuple[int, int, str]]:
    """`(first line, last line, qualified name)` of every `def` and `class`.

    Taken from the syntax tree rather than from indentation, so a decorator, a
    continuation line or a docstring cannot misplace a boundary. The qualified name
    is Python's `__qualname__` without `<locals>`: `Class.method`, `outer.inner`.
    A script that does not parse is an error; `compileall` rejects it as well.
    """
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as exc:
        sys.exit(f"error: {rel} does not parse, so its scopes cannot be cited: {exc}")
    spans: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}.{child.name}" if prefix else child.name
                spans.append((child.lineno, child.end_lineno or child.lineno, name))
                visit(child, name)
            else:
                visit(child, prefix)

    visit(tree, "")
    return spans


def python_scope_at(spans: list[tuple[int, int, str]], line: int) -> str:
    """The innermost `def` or `class` containing the 1-based `line`."""
    best: tuple[int, int, str] | None = None
    for span in spans:
        if span[0] <= line <= span[1] and (best is None or span[0] >= best[0]):
            best = span
    return best[2] if best else MODULE_SCOPE


def site_anchor(rel: str, scope: str) -> str:
    """A cited site: the tracked file and the enclosing scope, `path::scope`."""
    return f"{rel}::{scope}"


def site_file(site: str) -> str:
    """The file part of a `path::scope` site."""
    return site.split("::", 1)[0]


# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

RUST_READ = re.compile(r"std::env::(?:var|var_os)\(")
RUST_SET = re.compile(r"std::env::set_var\(")
RUST_CHILD_ENV = re.compile(r"\.env\(")
RUST_FOR_LIST = re.compile(
    r"for\s+(?:\(\s*([a-z_]+)\s*,[^)]*\)|([a-z_]+))\s+in\s+\[", re.M
)
RUST_FN_STR = re.compile(r"fn\s+([a-z_][a-z0-9_]*)\s*\([^)]*\)\s*->\s*&'static str")
RUST_ENV_CLOSURE = re.compile(r"let\s+mut\s+([a-z_][a-z0-9_]*)\s*=\s*\|\s*([a-z_]+)\s*:")
# Mixed case and parentheses allowed. Every name reaching `add` came from
# `resolve`, which accepts only a quoted literal in environment-call position or a
# known loop/function literal, so this cannot admit unrelated strings. An
# upper-case-only filter silently DROPPED real reads rather than reporting them:
# `ProgramFiles` and `ProgramFiles(x86)`, which decide where msconvert is found on
# Windows, were resolved and then discarded, so the table under-reported what
# actually changes the engine's behaviour.
ENV_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_()]{2,}$")


def call_arguments(text: str, open_paren: int) -> list[str] | None:
    """Split the argument list of the call whose `(` is at `open_paren`."""
    depth = 0
    for idx, c in code_chars(text):
        if idx < open_paren:
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return split_top_level(text[open_paren + 1 : idx])
    return None


def find_calls(text: str, pattern: re.Pattern[str]) -> list[tuple[int, list[str]]]:
    """Every match of `pattern` (which must end at a `(`) with its arguments."""
    out: list[tuple[int, list[str]]] = []
    for m in pattern.finditer(text):
        args = call_arguments(text, m.end() - 1)
        if args is not None:
            out.append((m.start(), args))
    return out

PY_GET = re.compile(
    r"os\.(?:environ\.get|getenv)\(\s*\"([A-Z0-9_]+)\"\s*(?:,\s*(.*?)\s*)?\)"
)
PY_IN = re.compile(r"\"([A-Z0-9_]+)\"\s+in\s+os\.environ")
PY_INDEX_READ = re.compile(r"os\.environ\[\s*\"([A-Z0-9_]+)\"\s*\](?!\s*=)")
PY_SET = re.compile(r"os\.environ\[\s*\"([A-Z0-9_]+)\"\s*\]\s*=\s*(.+?)\s*(?:#.*)?$")
PY_ENV_FN = re.compile(
    r"def\s+([a-z_][a-z0-9_]*)\s*\(\s*([a-z_]+)\s*,\s*([a-z_]+)\s*\)\s*:"
)


# Variables whose unset behaviour is computed by the reading code rather than given as a
# literal fallback, so the scanner finds no default and would otherwise print "none (unset
# means off)" for a setting that is ON when unset. Each entry is the behaviour when the
# variable is unset, then what setting it does, as the reading function implements it.
# Keep an entry in step with that function; a name that no longer appears among the reads
# is an error, so a removed variable cannot leave a stale row behind.
COMPUTED_ENV_DEFAULTS: dict[str, str] = {
    "MUMDIA_PARQUET_COMPRESSION": (
        "snappy. `zstd`, or `uncompressed` / `none`, changes the codec (`table.rs` `codec`)"
    ),
    "MUMDIA_PARQUET_DECODE_THREADS": (
        "automatic column groups, up to the codec pool's threads; one reader for a "
        "coalesced scan or inside a rayon pool. `k` asks for k groups, `1` is one reader "
        "(`table.rs` `automatic_decode_groups`)"
    ),
    "MUMDIA_PARQUET_PLAN": (
        "on (capped writers plan their float encodings). `0` / `off` / `false` / `no` "
        "restores the unplanned layout (`table.rs` `plan_enabled`)"
    ),
    "MUMDIA_PARQUET_THREADS": (
        "min(`--threads`, 8), or min(cores, 8) without `--threads`. `0` or `1` is serial "
        "(`codec.rs` `codec_threads`)"
    ),
    "MUMDIA_QUANT_SELECTIVE_READ": (
        "on (quant skips, unread, the chromatogram data pages that hold no kept row). "
        "`0` reads every page of every row group it opens; the outputs are the same "
        "(`stages/quant.rs` `selective_read_enabled`)"
    ),
    "MUMDIA_SIDECAR_DIR": (
        "`<out-dir>/sidecar_work` under `run` and `run-experiment`; `sidecar_work` in the "
        "current directory, or `--work-dir`, for `mumdia rescore`. A path moves the "
        "rescore sidecar files there (`stages/rescore.rs` `sidecar_work_dir`)"
    ),
    "MUMDIA_SIDECAR_SPACE_CHECK": (
        "on (before the handoff is written, a rescore sidecar run is refused when its "
        "work directory has less room than a PIN or raw handoff cannot be smaller than, "
        "and warned about below the files' usual size). `0` / `off` / `false` / `no` "
        "skips the check (`stages/rescore.rs` `check_sidecar_space`)"
    ),
    "MUMDIA_WIDE_SCAN": (
        "`plain`: the plain reader with its parallel decode for rescore's feature "
        "stream and compete's pass-through copy. `rowgroup` decodes the feature stream "
        "one row group a batch; `coalesced` reads each row group's projected column "
        "chunks in one sequential read (`stages/mod.rs` `WideScan`)"
    ),
}


class EnvVar:
    """One variable, with every distinct default the code shows for it.

    Two workers can disagree: `MUMDIA_NN_HIDDEN` falls back to `128,64,64,32` in
    `mokapot_worker.py` and `128,64` in `nn_rescore_worker.py`. Collapsing that to
    one value would be wrong, so every distinct fallback is kept with its site.
    """

    def __init__(self, name: str):
        self.name = name
        self.defaults: dict[str, set[str]] = {}
        self.sites: set[str] = set()

    def note_default(self, value: str | None, site: str) -> None:
        if value is None:
            return
        value = " ".join(value.split()).strip().rstrip(",")
        if value:
            self.defaults.setdefault(value, set()).add(site)

    def render_default(self) -> str:
        if not self.defaults and self.name in COMPUTED_ENV_DEFAULTS:
            return f"computed: {md_cell(COMPUTED_ENV_DEFAULTS[self.name])}"
        if not self.defaults:
            return "none (unset means off)"
        if len(self.defaults) == 1:
            return f"`{md_cell(next(iter(self.defaults)))}`"
        parts = []
        for value in sorted(self.defaults):
            where = sorted({Path(site_file(s)).name for s in self.defaults[value]})
            parts.append(f"`{md_cell(value)}` in {', '.join(where)}")
        return "; ".join(parts)


def add(store: dict[str, EnvVar], name: str, default: str | None, site: str) -> None:
    if not ENV_NAME.match(name):
        return
    var = store.setdefault(name, EnvVar(name))
    var.note_default(default, site)
    var.sites.add(site)


def blank_cfg_test(text: str) -> str:
    """Blank out `#[cfg(test)]` blocks, preserving line numbers.

    The environment-variable table is built by scanning the Rust sources for reads and
    sets. Test code is not the engine, so a `std::env::set_var` inside a test must not
    appear in a user-facing table -- and one did: a regression test that points discovery
    at a deliberately bogus interpreter made the generated reference state that MuMDIA
    sets `MUMDIA_PYTHON_RESCORE=/definitely/not/an/interpreter`, cited to the test's own
    line. A generated document whose selling point is that it cannot drift from the code
    is worse than a hand-written one when it faithfully reports a fixture.

    Blanked rather than deleted because the loop scopes below are resolved by line
    range and the enclosing function by offset, so the numbering has to survive; no
    line number reaches the document. `parse_config_rs` truncates at the first
    `#[cfg(test)]` instead, which is fine for one file that keeps its tests at the end;
    this has to cope with any source file, including an inline test module followed by
    more real code.

    The attribute does not only sit on items with a body. A test-only struct field, a
    field initializer in a struct literal, an enum variant, a match arm or a statement
    ends at a `,` or `;` and has no braces of its own, while the item after it usually
    has them. Searching for the next `{` from such an attribute swallowed the rest of
    the enclosing struct literal and the next function's header, which lost a closing
    brace and made `rust_scopes` reject the file. `cfg_test_item_end` finds the real
    end instead. Only the attribute and its item are blanked, and a line that blanking
    leaves holding nothing but whitespace is emptied, as the whole-line blanking did.
    """
    code = mask_rust_literals(text)
    out = list(text)
    pos = 0
    while True:
        m = CFG_TEST_ATTR.search(text, pos)
        if m is None:
            break
        end = cfg_test_item_end(code, m.end())
        for k in range(m.start(), end):
            if out[k] != "\n":
                out[k] = " "
        pos = max(end, m.end())
    old_lines = text.split(chr(10))
    new_lines = "".join(out).split(chr(10))
    return chr(10).join(
        "" if new != old and not new.strip() else new
        for old, new in zip(old_lines, new_lines)
    )


# A `#[cfg(test)]` attribute at the start of a line, where the whole-line blanking
# looked for it.
CFG_TEST_ATTR = re.compile(r"(?m)^[ \t]*#\[cfg\(test\)\]")


def cfg_test_item_end(code: str, start: int) -> int:
    """Offset just past the item that an attribute ending at `start` applies to.

    `code` is masked (`mask_rust_literals`), so a brace, comma or semicolon that is
    left is code. Scanning forward, at parenthesis and bracket depth 0:

    - a `{` opens the item's body (a `fn`, `impl`, `mod`, a braced struct or enum
      variant, a match arm with a block); the item ends at its matching `}`;
    - a `;` ends a statement or a braceless item (`use`, `const`, a unit struct);
    - a `,` ends a field, a field initializer, an enum variant or a match arm,
      unless it separates generic parameters (inside `<...>`) or bounds after
      `where`, which belong to an item header whose body follows;
    - a `}`, `)` or `]` that closes an enclosing group ends the item just before it:
      the attribute sat on the last element of that group, written without a
      trailing comma.

    Braces inside parentheses or brackets (a closure argument, a `match` passed to a
    call) are matched and skipped, so a statement such as
    `let l = l.with(match x { .. });` ends at its `;`.
    """
    n = len(code)
    paren = angle = 0
    after_where = False

    def match_brace(open_at: int) -> int:
        depth = 0
        for k in range(open_at, n):
            if code[k] == "{":
                depth += 1
            elif code[k] == "}":
                depth -= 1
                if depth == 0:
                    return k + 1
        return n

    j = start
    while j < n:
        c = code[j]
        if c in "([":
            paren += 1
        elif c in ")]":
            if paren == 0:
                return j
            paren -= 1
        elif c == "{":
            end = match_brace(j)
            if paren == 0:
                return end
            j = end
            continue
        elif c == "}":
            return j
        elif paren == 0 and c == ";":
            return j + 1
        elif paren == 0 and c == "," and angle == 0 and not after_where:
            return j + 1
        elif c == "<" and j > 0 and (code[j - 1].isalnum() or code[j - 1] in "_:"):
            angle += 1
        elif c == ">" and angle > 0 and code[j - 1] not in "-=":
            angle -= 1
        elif (
            c == "w"
            and paren == 0
            and code.startswith("where", j)
            and not (code[j - 1].isalnum() or code[j - 1] == "_")
            and not (j + 5 < n and (code[j + 5].isalnum() or code[j + 5] == "_"))
        ):
            after_where = True
        j += 1
    return n


def scan_rust_env(
    sources: list[tuple[str, str]],
) -> tuple[dict[str, EnvVar], dict[str, EnvVar], list[str]]:
    """Collect engine-side environment reads, sets, and child-process injections.

    Three indirections in the tree have to be followed or the table would be
    wrong rather than merely incomplete:

    - `for var in ["A", "B"] { env::var_os(var) }`: the names are in the loop
      list. Resolved only for reads INSIDE that loop's own line range, since the
      same identifier is a closure parameter elsewhere in the same file.
    - `push_env(role.env_var())`: `push_env` is a closure that reads env from its
      parameter, so the read is resolved at each call site instead.
    - `role.env_var()`: a `-> &'static str` helper whose body is a match over
      variable-name literals. Its body is brace-matched and harvested.

    Anything that resolves to no literal is returned as unresolved, one list entry
    per read, and printed in the document with its count, never dropped.

    `sources` holds `(repository-relative path, text)` pairs. Every site is cited
    as `path::fn`, the enclosing function, so the result does not depend on where
    in the file a read sits.
    """
    reads: dict[str, EnvVar] = {}
    sets: dict[str, EnvVar] = {}
    unresolved: list[str] = []
    for rel, raw in sources:
        # Comments are stripped line by line, so `//` prose cannot contribute a
        # stray quote (an apostrophe in "engine's") and offsets still hold.
        text = blank_cfg_test(decomment(raw))
        lines = text.split("\n")
        scopes = rust_scopes(text, rel)

        def line_of(offset: int) -> int:
            return text.count("\n", 0, offset) + 1

        # `fn name(..) -> &'static str` bodies, harvested for variable-name literals.
        fn_literals: dict[str, list[str]] = {}
        for m in RUST_FN_STR.finditer(text):
            open_at = text.find("{", m.end())
            close_at = match_brace(text, open_at) if open_at != -1 else None
            body = text[open_at:close_at] if close_at else ""
            fn_literals[m.group(1)] = sorted(
                {s for s in re.findall(r'"([A-Z0-9_]+)"', body) if ENV_NAME.match(s)}
            )

        # Closures that read env from a parameter: resolved at their call sites.
        env_closures = {m.group(1): m.group(2) for m in RUST_ENV_CLOSURE.finditer(text)}
        closure_params = set(env_closures.values())

        # `for <var> in [ "A", "B" ] { ... }` scopes, with their line range.
        loop_scopes: list[tuple[str, list[str], int, int]] = []
        for m in RUST_FOR_LIST.finditer(text):
            var = m.group(1) or m.group(2)
            end_bracket = text.find("]", m.end())
            names = [
                s
                for s in re.findall(r'"([A-Z0-9_]+)"', text[m.end() : end_bracket])
                if ENV_NAME.match(s)
            ]
            open_at = text.find("{", end_bracket)
            close_at = match_brace(text, open_at) if open_at != -1 else None
            loop_scopes.append(
                (
                    var,
                    names,
                    line_of(m.start()),
                    line_of(close_at) if close_at else len(lines),
                )
            )

        def resolve(arg: str, at_line: int) -> list[str] | None:
            arg = arg.strip()
            # Mixed case and parentheses are allowed, not just SHOUTING_SNAKE.
            # Windows variable names are legitimately mixed case and one of the ones
            # the tree reads is literally `ProgramFiles(x86)`, so an upper-case-only
            # pattern reported real literal reads as unresolved. Anything appearing
            # as the first argument of `std::env::var`/`var_os` is an environment
            # variable name by construction, so widening here cannot admit
            # unrelated strings.
            m = re.fullmatch(r'"([A-Za-z0-9_()]+)"', arg)
            if m:
                return [m.group(1)]
            for var, names, start, end in loop_scopes:
                if var == arg and start <= at_line <= end and names:
                    return names
            m = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.([a-z_]+)\(\)", arg)
            if m and fn_literals.get(m.group(1)):
                return fn_literals[m.group(1)]
            if re.fullmatch(r"[a-z_][a-z0-9_]*", arg) and fn_literals.get(arg):
                return fn_literals[arg]
            return None

        def record(
            store: dict[str, EnvVar],
            args: list[str],
            offset: int,
            kind: str,
            value_index: int | None = None,
        ) -> None:
            at_line = line_of(offset)
            site = site_anchor(rel, rust_scope_at(scopes, offset))
            arg = args[0] if args else ""
            names = resolve(arg, at_line)
            if names is None:
                if arg.strip() in closure_params:
                    # Read through a closure parameter: covered at the call sites.
                    # Checked only AFTER resolution, because the same identifier is
                    # a loop variable elsewhere in `python.rs` and that read is real.
                    return
                unresolved.append(f"{site}: env {kind} of `{arg.strip()}`")
                return
            value = (
                args[value_index]
                if value_index is not None and len(args) > value_index
                else None
            )
            for name in names:
                add(store, name, value, site)

        for offset, args in find_calls(text, RUST_READ):
            record(reads, args, offset, "read")
        for offset, args in find_calls(text, RUST_SET):
            record(sets, args, offset, "set", value_index=1)
        for offset, args in find_calls(text, RUST_CHILD_ENV):
            record(sets, args, offset, "child-process set", value_index=1)
        for name, param in env_closures.items():
            call_re = re.compile(rf"(?<![\w.]){re.escape(name)}\(")
            for offset, args in find_calls(text, call_re):
                if args and args[0].strip() == param:
                    continue
                record(reads, args, offset, "read via closure")
    return reads, sets, unresolved


def scan_python_env(
    sources: list[tuple[str, str]],
) -> tuple[dict[str, EnvVar], dict[str, EnvVar]]:
    """Collect sidecar-side environment reads and sets from `(path, text)` pairs.

    Every site is cited as `path::qualname`, the enclosing `def` or `class`, or as
    `path::<module>` for module-level code.
    """
    reads: dict[str, EnvVar] = {}
    sets: dict[str, EnvVar] = {}
    for rel, text in sources:
        lines = text.split("\n")
        scopes = python_scopes(text, rel)
        last_line = {
            (start, name.rsplit(".", 1)[-1]): end for start, end, name in scopes
        }
        # Helper functions of the shape `def f(name, default): ... os.environ.get(name, default)`.
        # The body is searched to the helper's last line, taken from the syntax tree.
        # A fixed character window after the `def` made detection depend on how many
        # blank or docstring lines stood between the signature and the read.
        helpers: dict[str, None] = {}
        for m in PY_ENV_FN.finditer(text):
            fname, p_name, p_default = m.group(1), m.group(2), m.group(3)
            def_line = text.count("\n", 0, m.start()) + 1
            end_line = last_line.get((def_line, fname), def_line)
            block = "\n".join(lines[def_line - 1 : end_line])
            if re.search(
                rf"os\.(?:environ\.get|getenv)\(\s*{p_name}\s*,\s*{p_default}\s*\)", block
            ):
                helpers[fname] = None
        for i, line in enumerate(lines, 1):
            site = site_anchor(rel, python_scope_at(scopes, i))
            code = line.split("#", 1)[0] if not line.strip().startswith("#") else ""
            for m in PY_GET.finditer(code):
                add(reads, m.group(1), m.group(2), site)
            for m in PY_IN.finditer(code):
                add(reads, m.group(1), None, site)
            for m in PY_INDEX_READ.finditer(code):
                add(reads, m.group(1), None, site)
            for m in PY_SET.finditer(code):
                add(sets, m.group(1), m.group(2), site)
            for fname in helpers:
                for m in re.finditer(
                    rf"(?<![\w.]){re.escape(fname)}\(\s*\"([A-Z0-9_]+)\"\s*,\s*([^),]+)\)", code
                ):
                    add(reads, m.group(1), m.group(2), site)
    return reads, sets


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------


def gh_anchor(heading: str) -> str:
    """GitHub's heading anchor: lowercase, punctuation dropped, spaces hyphenated."""
    s = re.sub(r"[^a-z0-9 _-]", "", heading.strip().lower())
    return s.replace(" ", "-")


def item_heading(paths: list[str]) -> str:
    """Heading for a `Vec` element struct, naming every list it types."""
    return " / ".join(f"{p}[]" for p in paths)


def field_rows(
    struct: Struct,
    path: str,
    enums: dict[str, Enum],
    structs: dict[str, Struct],
    unresolved: list[str],
    no_default: list[str],
    undocumented: list[str],
) -> list[str]:
    rows = ["| Field | Type | Default | Gated | Description |", "|---|---|---|---|---|"]
    for field in struct.fields:
        rendered = None
        if field.default_expr is not None:
            rendered = render_default(field.default_expr, field.rtype, enums, structs)
        field.default_rendered = rendered
        key = f"{path}.{field.name}" if path else field.name
        if rendered is None:
            if not struct.has_default_impl:
                no_default.append(f"`{key}` (`{field.rtype}`)")
                shown = "none: must be set"
            else:
                reason = (
                    f"unresolved expression `{md_cell(field.default_expr)}`"
                    if field.default_expr
                    else "not assigned in the `impl Default` block"
                )
                unresolved.append(f"`{key}` (`{field.rtype}`): {reason}")
                shown = "unresolved"
        elif "`" in rendered:
            # Already carries its own code span (an enum value, or prose naming a
            # struct); wrapping it again would nest backticks.
            shown = rendered
        else:
            shown = f"`{rendered}`"
        markers = gate_markers(field.doc)
        gate = ", ".join(markers) if markers else ""
        if not field.doc:
            undocumented.append(f"`{key}`")
        rows.append(
            f"| `{field.name}` | `{md_cell(field.rtype)}` | {shown} | {gate} | "
            f"{md_cell(field.doc)} |"
        )
    return rows


class Inputs:
    """Every text the reference and the schema are generated from.

    Held as `(repository-relative path, text)` pairs rather than read inside the
    builders, so the blank-line self-check can run the same builders on modified
    copies without touching the files on disk.
    """

    def __init__(
        self,
        config_text: str,
        rust_sources: list[tuple[str, str]],
        py_sources: list[tuple[str, str]],
    ):
        self.config_text = config_text
        self.rust_sources = rust_sources
        self.py_sources = py_sources

    def transformed(self, edit) -> "Inputs":
        """A copy with `edit(text)` applied to every input text."""
        return Inputs(
            edit(self.config_text),
            [(rel, edit(text)) for rel, text in self.rust_sources],
            [(rel, edit(text)) for rel, text in self.py_sources],
        )


def load_inputs() -> Inputs:
    # Tracked sources only. A filesystem glob also picked up untracked scratch copies
    # (`*-covr2.rs`, `*-covr2.py`) that sit beside the real files on a developer's
    # machine, and their environment-variable reads then entered this document and
    # failed `--check` in a workspace that was fine as far as git was concerned
    # (docs/29 #21). Outside a git checkout, the release archive for instance, the
    # glob is the only option and the archive holds tracked files only.
    rust_files = tracked_files(
        CRATES_DIR,
        ".rs",
        lambda: sorted(p for p in CRATES_DIR.rglob("*.rs") if "target" not in p.parts),
    )
    py_files = tracked_files(
        SCRIPTS_DIR,
        ".py",
        lambda: sorted(SCRIPTS_DIR.glob("*.py")),
        recursive=False,
    )

    def read(paths: list[Path]) -> list[tuple[str, str]]:
        return [
            (
                p.relative_to(REPO_ROOT).as_posix(),
                p.read_text(encoding="utf-8", errors="replace"),
            )
            for p in paths
        ]

    return Inputs(
        CONFIG_RS.read_text(encoding="utf-8"), read(rust_files), read(py_files)
    )


def build_document(inputs: Inputs) -> tuple[str, dict[str, object]]:
    text = inputs.config_text
    structs, enums, warnings = parse_config_rs(text)
    sections, items = walk_sections(structs)
    section_enums = reachable_enums(sections, items, enums)
    profiles = parse_profiles(text)

    for path, target in STAGE_DOC.items():
        if not (REPO_ROOT / target).is_file():
            sys.exit(f"error: STAGE_DOC['{path}'] points at missing {target}")
    for name, target in ITEM_STRUCT_DOC.items():
        if not (REPO_ROOT / target).is_file():
            sys.exit(f"error: ITEM_STRUCT_DOC['{name}'] points at missing {target}")

    rust_reads, rust_sets, env_unresolved = scan_rust_env(inputs.rust_sources)
    py_reads, py_sets = scan_python_env(inputs.py_sources)

    # Field type by JSON path, so a profile override can be rendered as the value a
    # config file would carry rather than as the Rust expression.
    field_types = {
        (f"{path}.{field.name}" if path else field.name): field.rtype
        for path, _kind, struct in sections
        for field in struct.fields
    }

    unresolved: list[str] = []
    no_default: list[str] = []
    undocumented: list[str] = []
    out: list[str] = []
    a = out.append

    a("# MuMDIA configuration reference")
    a("")
    a(f"GENERATED FILE. Do not edit. `{GENERATOR}` parses")
    a(f"`{CONFIG_RS_REL}` and the environment-variable")
    a("reads in the crates and the sidecar scripts, so every field, type, default,")
    a("and description below is the one the code actually uses. Change the field or")
    a("its doc comment in the Rust source and regenerate. An edit made to this file")
    a("is lost on the next run.")
    a("")
    a("```text")
    a("python ci/gen_config_reference.py            # regenerate")
    a("python ci/gen_config_reference.py --check    # fail if this file is stale")
    a("```")
    a("")
    a("For the command-line interface that loads these files, read")
    a("`docs/23_cli_reference.md`. For how the config is loaded, validated, and")
    a("hashed into the run manifest, read `docs/02_config_and_data_model.md`.")
    a("")

    a("## How to read this document")
    a("")
    a("The configuration is one JSON object with a per-stage section. Every field")
    a("carries `#[serde(default)]` and the top-level object is")
    a("`deny_unknown_fields`, so a config may omit any field but may not contain a")
    a("key the engine does not know: a typo is a hard parse error, not a silently")
    a("ignored line. `--config` therefore always describes a complete")
    a("configuration, with the defaults in this document filling the rest.")
    a("")
    a("Columns:")
    a("")
    a("- **Default** is the value from the `impl Default` block, rendered as the")
    a("  JSON a config file would carry. A field whose default the parser could not")
    a("  resolve is marked `unresolved` and listed at the end of this document,")
    a("  never omitted.")
    a("- **Gated** is non-empty when the field's own doc comment marks it as not")
    a("  part of the shipped, validated chain. It repeats the phrase that matched:")
    a("  `benchmark-gated` and `gated` mean the change needs entrapment plus a")
    a("  second acquisition before it becomes a default (CLAUDE.md, \"Changes that")
    a("  remain benchmark-gated\"); `diagnostic` means the field only adds a sidecar")
    a("  artifact or extra columns; `not yet wired` means no code reads the field")
    a("  yet. Treat a non-empty cell as: do not enable this because it sounds")
    a("  useful.")
    a("- **Description** is the field's Rust doc comment, unwrapped to one")
    a("  paragraph. Nothing is paraphrased.")
    a("")
    a("Enum-valued fields show their default as the serde spelling; the accepted")
    a("values of every enum are in \"Enumerations\" at the end. An empty description")
    a("means the field carries no doc comment in the source, not that it is")
    a("undocumented on purpose; those fields are counted under \"Coverage\".")
    a("")

    # ---- section index -----------------------------------------------------
    a("## Sections")
    a("")
    a("| Section | Struct | Fields | Stage document |")
    a("|---|---|---|---|")
    for path, _kind, struct in sections:
        heading = "(top level)" if path == "" else path
        label = f"[{'(top level)' if path == '' else '`' + path + '`'}]"
        label += f"(#{gh_anchor(heading)})"
        doc_ref = STAGE_DOC.get(path, "")
        ref = f"[{doc_ref}]({Path(doc_ref).name})" if doc_ref else ""
        a(f"| {label} | `{struct.name}` | {len(struct.fields)} | {ref} |")
    for paths, struct in items:
        heading = item_heading(paths)
        ref_target = ITEM_STRUCT_DOC.get(struct.name, "")
        ref = f"[{ref_target}]({Path(ref_target).name})" if ref_target else ""
        a(
            f"| [`{heading}`](#{gh_anchor(heading)}) | `{struct.name}` | "
            f"{len(struct.fields)} | {ref} |"
        )
    a("")

    # ---- per-section tables ------------------------------------------------
    for path, _kind, struct in sections:
        heading = "(top level)" if path == "" else path
        a(f"## {heading}")
        a("")
        meta = [f"`{struct.name}` ({CONFIG_RS_REL})"]
        doc_ref = STAGE_DOC.get(path)
        if doc_ref:
            meta.append(f"stage document: [{doc_ref}]({Path(doc_ref).name})")
        a(". ".join(meta) + ".")
        a("")
        if struct.doc:
            a(md_cell(struct.doc))
            a("")
        if not struct.has_default_impl:
            a(f"`{struct.name}` has no `impl Default` block in the source.")
            a("")
        a(
            "\n".join(
                field_rows(
                    struct, path, enums, structs, unresolved, no_default, undocumented
                )
            )
        )
        a("")

    for paths, struct in items:
        heading = item_heading(paths)
        a(f"## {heading}")
        a("")
        meta = [f"`{struct.name}` ({CONFIG_RS_REL})"]
        ref_target = ITEM_STRUCT_DOC.get(struct.name)
        if ref_target:
            meta.append(f"stage document: [{ref_target}]({Path(ref_target).name})")
        a(". ".join(meta) + ".")
        a("")
        a(
            "Element type of "
            + " and ".join(f"`{p}`" for p in paths)
            + ". Each element is a JSON object with these keys; the list default is"
            " on the owning field."
        )
        a("")
        if struct.doc:
            a(md_cell(struct.doc))
            a("")
        if not struct.has_default_impl:
            a(
                f"`{struct.name}` has no `impl Default` block, so an element must set "
                "every key."
            )
            a("")
        a(
            "\n".join(
                field_rows(
                    struct,
                    paths[0] + "[]",
                    enums,
                    structs,
                    unresolved,
                    no_default,
                    undocumented,
                )
            )
        )
        a("")

    # ---- profiles ----------------------------------------------------------
    a("## Named profiles")
    a("")
    if profiles:
        a("`mumdia run --profile NAME` applies a named override set on top of")
        a("`--config` and the defaults, from `Config::apply_profile`. The overrides:")
        a("")
        a("| Profile | Overrides |")
        a("|---|---|")
        for name in sorted(profiles):
            parts = []
            for key, value in profiles[name]:
                rtype = field_types.get(key, "")
                shown = render_default(value, rtype, enums, structs) or value
                if "`" not in shown:
                    shown = f"`{shown}`"
                parts.append(f"`{key}` = {shown}")
            a(f"| `{name}` | {md_cell('; '.join(parts))} |")
        a("")
        a("A profile is applied after the config file is parsed, so it wins over a")
        a("value the config file set for the same field.")
    else:
        a("The parser found no `Config::apply_profile` match arms to report.")
    a("")

    # ---- enumerations ------------------------------------------------------
    a("## Enumerations")
    a("")
    a("Accepted values for every enum-typed field above, with the serde spelling a")
    a("config file must use. The default variant is marked. Sorted by type name.")
    a("")
    for enum in section_enums:
        a(f"### `{enum.name}`")
        a("")
        a(f"({CONFIG_RS_REL})")
        a("")
        if enum.doc:
            a(md_cell(enum.doc))
            a("")
        a("| Value | Default | Description |")
        a("|---|---|---|")
        for variant in enum.variants:
            a(
                f"| `{enum.serde_name(variant.name)}` | "
                f"{'yes' if variant.is_default else ''} | {md_cell(variant.doc)} |"
            )
        a("")

    unreachable = sorted(set(enums) - {e.name for e in section_enums})
    if unreachable:
        a(
            f"{len(unreachable)} enum(s) are declared in `{CONFIG_RS_REL}` but are not "
            "reachable from `Config`, so they are not config values: "
            + ", ".join(f"`{n}`" for n in unreachable)
            + ". They are CLI-only or helper types."
        )
        a("")

    # ---- environment variables --------------------------------------------
    all_read = sorted(set(rust_reads) | set(py_reads))
    a("## Environment variables")
    a("")
    a("Collected by scanning `std::env::var`/`var_os` across")
    a("`rust/mumdia/crates/**/*.rs` and `os.environ`/`os.getenv` across")
    a("`scripts/*.py`. These are not config keys: nothing validates them, a typo is")
    a("silently ignored, and none of them appears in the run manifest. Prefer a")
    a("config field or a CLI flag where one exists, and treat this table as the")
    a("record of what the code will read if the variable happens to be set.")
    a("")
    a("`Side` says which process reads the variable. **engine** is the Rust binary;")
    a("**sidecar** is a Python worker, which the engine launches as a child process")
    a("and which therefore inherits the engine's environment. A variable read on")
    a("both sides is marked **both**.")
    a("")
    a("`Default in code` is the fallback the reading code supplies when the variable")
    a("is unset. Two workers can disagree, in which case every distinct fallback is")
    a("listed with the file it is in. A default marked `computed:` has no literal")
    a("fallback in the code: the reading function works out the behaviour, and the")
    a("text describes it (`COMPUTED_ENV_DEFAULTS` in `ci/gen_config_reference.py`).")
    a("")
    a("`Read at` and `Site` name the file and the enclosing function as")
    a("`path::function`: `Type::method` for a Rust method, `Trait::method` for a")
    a("default trait method, inline modules and nested Rust functions joined with")
    a(f"`::`, `Class.method` in Python, `{MODULE_SCOPE}` outside any")
    a("function. No line number is cited, so the tables change only when a read")
    a("moves to another function.")
    a("")
    injected = sorted(set(py_reads) & set(rust_sets))
    if injected:
        a(
            f"{len(injected)} of these are also SET by the engine before the worker "
            "starts, so the worker's own fallback applies only when the engine did not "
            "set it: " + ", ".join(f"`{n}`" for n in injected) + ". See the next table."
        )
        a("")
    a("| Variable | Side | Default in code | Read at |")
    a("|---|---|---|---|")
    for name in all_read:
        in_rust = name in rust_reads
        in_py = name in py_reads
        side = "both" if in_rust and in_py else ("engine" if in_rust else "sidecar")
        merged = EnvVar(name)
        for store in (rust_reads, py_reads):
            if name in store:
                for value, sites in store[name].defaults.items():
                    merged.defaults.setdefault(value, set()).update(sites)
                merged.sites |= store[name].sites
        sites = sorted(merged.sites)
        shown_sites = ", ".join(f"`{s}`" for s in sites[:3])
        if len(sites) > 3:
            shown_sites += f", +{len(sites) - 3} more"
        a(f"| `{name}` | {side} | {merged.render_default()} | {shown_sites} |")
    a("")
    a(
        f"{len(all_read)} variables are read: "
        f"{len(rust_reads)} engine-side, {len(py_reads)} sidecar-side, "
        f"{len(set(rust_reads) & set(py_reads))} on both sides."
    )
    a("")

    all_set = sorted(set(rust_sets) | set(py_sets))
    a("### Variables the code sets")
    a("")
    a("A variable set here overrides whatever the caller exported, so exporting one")
    a("of these has no effect on the process listed. The engine's `--threads` is the")
    a("one exception noted in its own help text: it sets `MUMDIA_NN_THREADS` and")
    a("`OMP_NUM_THREADS` for the sidecars only if they are not already set.")
    a("")
    a("| Variable | Set by | Value | Site |")
    a("|---|---|---|---|")
    for name in all_set:
        in_rust = name in rust_sets
        in_py = name in py_sets
        setter = "both" if in_rust and in_py else ("engine" if in_rust else "sidecar")
        merged = EnvVar(name)
        for store in (rust_sets, py_sets):
            if name in store:
                for value, sites in store[name].defaults.items():
                    merged.defaults.setdefault(value, set()).update(sites)
                merged.sites |= store[name].sites
        sites = sorted(merged.sites)
        value_cell = merged.render_default() if merged.defaults else ""
        a(
            f"| `{name}` | {setter} | {value_cell} | "
            f"{', '.join(f'`{s}`' for s in sites[:3])} |"
        )
    a("")

    # ---- what the generator could not resolve -----------------------------
    a("## Unresolved by the generator")
    a("")
    a("Listed rather than omitted, so a parsing gap is visible in the document")
    a("instead of looking like an absent field.")
    a("")
    if unresolved:
        a(f"{len(set(unresolved))} field default(s) could not be resolved:")
        a("")
        for entry in sorted(set(unresolved)):
            a(f"- {entry}")
    else:
        a("Every field whose struct has an `impl Default` resolved from the source.")
    a("")
    if no_default:
        a(
            f"{len(set(no_default))} field(s) have no default because the owning struct "
            "has no `impl Default`. That is the source's intent, not a parsing gap: a "
            "list element must carry every key."
        )
        a("")
        for entry in sorted(set(no_default)):
            a(f"- {entry}")
        a("")
    if warnings:
        a(f"{len(warnings)} structural warning(s) while parsing the config source:")
        a("")
        for entry in sorted(set(warnings)):
            a(f"- {entry}")
        a("")
    if env_unresolved:
        # Counted, not deduplicated. An entry names the function, not the line, so
        # two reads of the same argument in one function share it; without the
        # count, a second such read would leave this list, and `--check`, unchanged.
        counts = Counter(env_unresolved)
        a(
            f"{len(env_unresolved)} environment read(s) whose name is not a literal. "
            "Reads with the same function, access and argument share one entry, "
            "which gives their number when there is more than one:"
        )
        a("")
        for entry in sorted(counts):
            n = counts[entry]
            a(f"- `{entry}`" + (f" ({n} reads)" if n > 1 else ""))
        a("")

    # ---- coverage ----------------------------------------------------------
    n_fields = sum(len(s.fields) for _, _, s in sections) + sum(
        len(s.fields) for _, s in items
    )
    n_structs = len(sections) + len(items)
    n_gated = sum(
        1
        for s in [st for _, _, st in sections] + [st for _, st in items]
        for f in s.fields
        if gate_markers(f.doc)
    )
    a("## Coverage")
    a("")
    a(
        f"{n_structs} structs and {n_fields} fields emitted from "
        f"`{CONFIG_RS_REL}`, plus {len(section_enums)} enumerations, "
        f"{len(profiles)} named profile(s), {len(all_read)} environment variables "
        f"read and {len(all_set)} set."
    )
    a("")
    a(
        f"{n_gated} field(s) carry a gating marker in their doc comment. "
        f"{len(set(undocumented))} field(s) carry no doc comment at all, so their "
        f"description is empty above. {len(set(unresolved))} default(s) could not be "
        f"resolved and {len(set(no_default))} have none by design."
    )

    while out and not out[-1]:
        out.pop()
    stats = {
        "structs": n_structs,
        "fields": n_fields,
        "enums": len(section_enums),
        "profiles": len(profiles),
        "env_read": len(all_read),
        "env_read_engine": len(rust_reads),
        "env_read_sidecar": len(py_reads),
        "env_set": len(all_set),
        "unresolved": len(set(unresolved)),
        "warnings": len(set(warnings)),
        # Entries of COMPUTED_ENV_DEFAULTS that no scanned source reads. The entry
        # point refuses to write or to pass --check while this is non-empty; the
        # builder only reports it, so it stays a pure function of `inputs` and also
        # runs on the synthetic sources the tests give it.
        "stale_computed_env": sorted(set(COMPUTED_ENV_DEFAULTS) - set(all_read)),
    }
    return "\n".join(out) + "\n", stats


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
        f"\nThe config or an environment read changed without the reference being "
        f"regenerated. Run `python {GENERATOR}` and commit the result.",
        file=sys.stderr,
    )
    return 1



# ---------------------------------------------------------------------------
# Machine-readable schema, for the desktop application's settings editor
# ---------------------------------------------------------------------------


def build_schema(config_text: str) -> dict:
    """The same parse the reference document uses, as data instead of prose.

    The desktop application renders one form control per field, and it must not
    carry its own copy of the field list, the types, the defaults or the help text:
    a second copy is a second thing to keep in step, and the one that drifts is the
    one a user reads. So the form is generated from this, and this is generated from
    `config.rs`, checked for staleness in CI exactly as the Markdown is.

    Every field carries the doc comment verbatim as `help`, and `gates` carries the
    markers already used in the reference ("benchmark-gated", "experimental", ...),
    so the interface can mark a parameter that must not be changed casually.
    """
    text = config_text
    structs, enums, _warnings = parse_config_rs(text)
    sections, items = walk_sections(structs)

    def field_entry(path: str, field: Field, struct: Struct) -> dict:
        base = base_type(field.rtype)
        # `Option<usize>` is an integer the form should offer as one, not an opaque
        # "other" it renders as free text. The desktop settings screen keys its input
        # conversion on `kind`, so an optional number typed there used to reach the
        # engine as the STRING "0" and be rejected -- which is how the off-switch for an
        # optional setting became unreachable from the interface. `Option<String>` is
        # left alone: it is already rendered as text and means the same thing either way.
        if base.startswith("Option<") and base.endswith(">"):
            inner = base[len("Option<"): -1].strip()
            if inner in ("f32", "f64") or inner in (
                "u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64"
            ):
                base = inner
        # `default_rendered` is populated by the Markdown row builder, which the
        # schema does not run. Render it here with the same function, so the value
        # the form shows and the value the table shows are produced by one code
        # path and cannot disagree.
        rendered = (
            render_default(field.default_expr, field.rtype, enums, structs)
            if field.default_expr is not None
            else None
        )
        # `render_default` produces a Markdown cell, so an enum arrives as
        # `` `base_peptide` `` and a number as text. The schema is data: strip the
        # formatting and give the value its JSON type, so a form control can be
        # populated without the interface having to unpick Markdown.
        def typed_default(text_value: str | None, kind: str):
            if text_value is None:
                return None
            v = text_value.strip().strip("`").strip()
            # A computed default is rendered for prose as `1.0 / 3.0 (0.333333)`.
            # The parenthesised value is the number a form needs.
            m = re.fullmatch(r".*\(([-+0-9.eE]+)\)", v)
            if m:
                v = m.group(1)
            if kind == "bool":
                return {"true": True, "false": False}.get(v, v)
            if kind in ("integer", "float"):
                try:
                    return int(v) if kind == "integer" else float(v)
                except ValueError:
                    return v
            return v

        kind = (
            "enum"
            if base in enums
            else "bool"
            if base == "bool"
            else "float"
            if base in ("f32", "f64")
            else "integer"
            if base in ("u8", "u16", "u32", "u64", "usize", "i8", "i16", "i32", "i64")
            else "string"
            if base in ("String", "str")
            else "other"
        )
        entry: dict = {
            "path": f"{path}.{field.name}" if path else field.name,
            "name": field.name,
            "section": path,
            "rust_type": field.rtype,
            "kind": kind,
            "optional": field.rtype.startswith("Option<"),
            "default": typed_default(rendered, kind),
            "default_text": rendered,
            "help": field.doc.strip(),
            "gates": gate_markers(field.doc),
            # The declaring struct, not a line. A line number changed with every edit
            # above the field, which made this file stale on every other open branch.
            "source_struct": struct.name,
        }
        if base in enums:
            entry["choices"] = [snake_case(v.name) for v in enums[base].variants]
        return entry

    fields: list[dict] = []
    for path, _kind, struct in sections:
        for field in struct.fields:
            if base_type(field.rtype) in structs:
                continue  # a nested section, not a leaf setting
            fields.append(field_entry(path, field, struct))

    # Vec<T> item structs are reachable settings too, but they are edited as lists
    # rather than as single controls; name them so the interface can say so instead
    # of silently omitting them.
    list_sections = []
    for paths, struct in items:
        list_sections.append(
            {
                "paths": paths,
                "item": struct.name,
                "fields": [field_entry("", f, struct) for f in struct.fields],
            }
        )

    return {
        "generated_by": GENERATOR,
        "source": CONFIG_RS_REL,
        "sections": [p for p, _k, _s in sections],
        "fields": fields,
        "list_sections": list_sections,
        "profiles": {
            name: [{"path": p, "value": v} for p, v in changes]
            for name, changes in parse_profiles(text).items()
        },
    }


def schema_text(config_text: str) -> str:
    return json.dumps(build_schema(config_text), indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Blank-line invariance self-check
# ---------------------------------------------------------------------------


def insert_blank_lines(text: str) -> str:
    """`text` with blank lines added and nothing else changed.

    Three blank lines go in front of the first line, which moves every line of the
    file, and every existing blank line is doubled, which moves each later line by
    a different amount. Only positions that are already blank receive a new one, so
    no statement, doc comment block or Python line continuation is split, and the
    Rust or Python reader sees the same program. What changes is what a merge
    elsewhere in the file changes: the line on which every item sits.
    """
    return "\n\n\n" + re.sub(r"\n(?=\n)", "\n\n", text)


def blank_line_problems(inputs: Inputs, document: str, schema: str) -> list[str]:
    """Regenerate from blank-line-shifted copies of `inputs`; name what moved.

    `document` and `schema` are the artifacts generated from `inputs` unchanged.
    An empty list means neither artifact depends on a line number. Any entry means
    a line dependency has come back, and the committed reference would again go
    stale on every merge that moved lines in a large file.
    """
    shifted = inputs.transformed(insert_blank_lines)
    problems: list[str] = []
    for label, before, after in (
        ("the reference document", document, build_document(shifted)[0]),
        ("the settings schema", schema, schema_text(shifted.config_text)),
    ):
        if before == after:
            continue
        diff = list(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"{label} (inputs as committed)",
                tofile=f"{label} (blank lines inserted)",
                lineterm="",
                n=0,
            )
        )
        problems.append(f"{label} changed when only blank lines were inserted:")
        problems.extend(f"  {line}" for line in diff[:40])
        if len(diff) > 40:
            problems.append(f"  ... and {len(diff) - 40} more diff line(s)")
    return problems


def report_blank_line_problems(problems: list[str]) -> int:
    """Print the self-check result; 1 when a line dependency was found."""
    if not problems:
        print(
            "blank-line invariance: the reference and the schema are unchanged when "
            "blank lines are inserted into every input."
        )
        return 0
    for line in problems:
        print(line if line.startswith("  ") else f"error: {line}", file=sys.stderr)
    print(
        "\nThe generator emitted something that depends on a source line number. "
        "Cite the file and the enclosing function instead (`site_anchor`).",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output Markdown path")
    ap.add_argument(
        "--schema-out",
        default=str(DEFAULT_SCHEMA_OUT),
        help="output JSON schema path, read by the desktop settings editor",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help=(
            "regenerate in memory and exit 1 if --out or --schema-out differs, or if "
            "inserting blank lines into the inputs changes either"
        ),
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="only check that inserting blank lines into the inputs changes nothing",
    )
    args = ap.parse_args(argv)

    if not CONFIG_RS.is_file():
        sys.exit(f"error: {CONFIG_RS_REL} not found")

    inputs = load_inputs()
    generated, stats = build_document(inputs)
    if stats["stale_computed_env"]:
        sys.exit(
            "error: COMPUTED_ENV_DEFAULTS names variables no code reads: "
            + ", ".join(stats["stale_computed_env"])
        )
    out_path = Path(args.out)
    schema = schema_text(inputs.config_text)
    schema_path = Path(args.schema_out)

    if args.self_test:
        return report_blank_line_problems(blank_line_problems(inputs, generated, schema))

    if args.check:
        rc = check(out_path, generated)
        # Both artifacts come from one parse of one file, so they go stale together
        # and must be checked together.
        current = (
            schema_path.read_text(encoding="utf-8") if schema_path.is_file() else ""
        )
        if current.replace("\r\n", "\n") != schema:
            print(
                f"{schema_path.as_posix()} is stale. Regenerate with:\n"
                f"    python {GENERATOR}",
                file=sys.stderr,
            )
            rc = 1
        else:
            n = len(json.loads(schema)["fields"])
            print(f"{schema_path.as_posix()} is up to date ({n} settings).")
        # Current now is not enough: a reference that cites line numbers is current
        # only until the next merge that moves a line.
        if report_blank_line_problems(blank_line_problems(inputs, generated, schema)):
            rc = 1
        return rc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(generated)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schema_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(schema)
    print(
        f"wrote {schema_path.as_posix()}: "
        f"{len(json.loads(schema)['fields'])} settings."
    )
    print(
        f"wrote {out_path.as_posix()}: {len(generated.splitlines())} lines, "
        f"{generated.count(chr(10) + '## ')} sections, "
        f"{stats['structs']} structs, {stats['fields']} fields, "
        f"{stats['enums']} enums, {stats['env_read']} env vars read "
        f"({stats['env_read_engine']} engine, {stats['env_read_sidecar']} sidecar), "
        f"{stats['env_set']} set, {stats['unresolved']} unresolved defaults, "
        f"{stats['warnings']} parse warnings."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
