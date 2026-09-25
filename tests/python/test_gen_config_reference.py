"""`ci/gen_config_reference.py` cites environment reads by function, not by line.

The generator used to cite every environment-variable read as `path:line`, so any
merge that moved lines in `rescore.rs`, `config.rs`, `main.rs` or a sidecar script
made `docs/24_config_reference.md` and `configs/config-schema.json` stale on every
other open branch, although no variable, field or default had changed. It now cites
`path::function`. These tests pin that: two versions of a source that differ only
in inserted blank lines must produce the same sites, and the committed inputs must
produce the same reference and schema with blank lines inserted into every file.

The generator needs only the standard library, so nothing here skips.
"""

from __future__ import annotations

import importlib.util
import re
import sys

import pytest
from conftest import ROOT

_SPEC = importlib.util.spec_from_file_location(
    "gen_config_reference", ROOT / "ci" / "gen_config_reference.py"
)
gen = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gen
_SPEC.loader.exec_module(gen)


RUST_SOURCE = """\
use std::process::Command;

static CAP: Lazy<Option<String>> = Lazy::new(|| std::env::var("MUMDIA_T_STATIC").ok());

pub struct Runner;

impl Runner {
    /// A method; its reads are cited with the impl type.
    pub fn run(&self) {
        // A `fn` and an unbalanced brace inside a literal open no scope.
        let _decoy = "fn fake() {";
        let _raw = r#"impl Fake { fn fake() { "#;
        let _ = std::env::var("MUMDIA_T_METHOD");

        fn inner() {
            let _ = std::env::var_os("MUMDIA_T_NESTED");
        }
    }
}

impl<'a, T: Clone> Default for Wrapper<'a, T>
where
    T: Send,
{
    fn default() -> Self {
        let _ = std::env::var("MUMDIA_T_TRAIT_IMPL");
        todo!()
    }
}

fn free() -> impl Iterator<Item = u32> {
    let _ = std::env::var("MUMDIA_T_FREE");
    std::iter::empty()
}

fn child(cmd: &mut Command) {
    cmd.env("MUMDIA_T_CHILD", "1");
}

pub(crate) mod alpha {
    pub fn run() {
        let _ = std::env::var("MUMDIA_T_MOD_A");
    }
}

mod beta {
    static IN_MOD: Lazy<Option<String>> = Lazy::new(|| std::env::var("MUMDIA_T_MOD_STATIC").ok());

    fn run() {
        let _ = std::env::var("MUMDIA_T_MOD_B");
    }

    impl super::Runner {
        fn in_mod(&self) {
            let _ = std::env::var("MUMDIA_T_MOD_IMPL");
        }
    }
}

mod declared_elsewhere;

pub trait Source {
    fn required(&self);

    fn provided(&self) {
        let _ = std::env::var("MUMDIA_T_TRAIT_DEFAULT");
    }
}

unsafe impl Backend for Shared {
    fn open() {
        let _ = std::env::var("MUMDIA_T_UNSAFE_IMPL");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fixture() {
        std::env::set_var("MUMDIA_T_TEST_ONLY", "x");
    }
}
"""

PY_SOURCE = '''\
import os

TOP = os.environ.get("MUMDIA_T_TOP", "1")


def env_int(name, default):
    """A helper of the shape the scanner follows to its call sites.

    The read sits several lines below the signature.
    """

    return int(os.environ.get(name, default))


class Model:
    def fit(self):
        x = os.getenv("MUMDIA_T_METHOD", "a")

        def inner():
            return os.environ["MUMDIA_T_INNER"]

        return inner


def main():
    n = env_int("MUMDIA_T_HELPER", 3)
    os.environ["MUMDIA_T_SET"] = "1"
'''


def snapshot(store):
    """Every variable with its sorted sites and defaults, comparable with `==`."""
    return {
        name: (
            sorted(var.sites),
            sorted((value, tuple(sorted(sites))) for value, sites in var.defaults.items()),
        )
        for name, var in sorted(store.items())
    }


def scan_rust(text):
    reads, sets, unresolved = gen.scan_rust_env([("src/x.rs", text)])
    return snapshot(reads), snapshot(sets), sorted(unresolved)


def scan_python(text):
    reads, sets = gen.scan_python_env([("scripts/x.py", text)])
    return snapshot(reads), snapshot(sets)


def line_of(text, needle):
    return text[: text.index(needle)].count("\n") + 1


def test_rust_reads_are_cited_by_enclosing_function():
    """Each engine read is cited by its function, qualified by every enclosing item.

    A wrong scope sends a reader of docs/24 to the wrong function; a `fn` or `{`
    inside a literal opening a scope would misattribute every later read in the file.
    Inline modules and traits are part of the chain, so two same-named functions in
    different modules get different citations, and a read that moves from one to
    the other changes the reference. `unsafe impl` is an impl block like any other.
    """
    reads, sets, unresolved = scan_rust(RUST_SOURCE)
    sites = {name: entry[0] for name, entry in reads.items()}
    assert sites == {
        "MUMDIA_T_FREE": ["src/x.rs::free"],
        "MUMDIA_T_METHOD": ["src/x.rs::Runner::run"],
        "MUMDIA_T_MOD_A": ["src/x.rs::alpha::run"],
        "MUMDIA_T_MOD_B": ["src/x.rs::beta::run"],
        "MUMDIA_T_MOD_IMPL": ["src/x.rs::beta::Runner::in_mod"],
        "MUMDIA_T_MOD_STATIC": ["src/x.rs::<module>"],
        "MUMDIA_T_NESTED": ["src/x.rs::Runner::run::inner"],
        "MUMDIA_T_STATIC": ["src/x.rs::<module>"],
        "MUMDIA_T_TRAIT_DEFAULT": ["src/x.rs::Source::provided"],
        "MUMDIA_T_TRAIT_IMPL": ["src/x.rs::Wrapper::default"],
        "MUMDIA_T_UNSAFE_IMPL": ["src/x.rs::Shared::open"],
    }
    # Test code is not the engine, and a child-process `.env` is a set.
    assert sets == {"MUMDIA_T_CHILD": (["src/x.rs::child"], [('"1"', ("src/x.rs::child",))])}
    assert unresolved == []


def test_unbalanced_braces_after_masking_fail_loudly():
    """A masking error stops the run instead of citing later reads wrongly.

    The trigger is real: comments are stripped line by line before masking, so a
    `//` on a continuation line of a multi-line string (a URL) cuts the closing
    quote off, and the rest of the file is read with inverted quote parity. The read
    in `b` would then be cited under `a` or as `<module>`, and the blank-line
    self-check would not notice, because a wrong scope is stable under moved lines.
    """
    source = (
        "fn a() {\n"
        '    let _s = "first line\n'
        'see https://example.org/x";\n'
        "}\n"
        "\n"
        "fn b() {\n"
        '    let _ = std::env::var("MUMDIA_T_AFTER");\n'
        "}\n"
    )
    with pytest.raises(SystemExit) as exc:
        gen.scan_rust_env([("src/x.rs", source)])
    assert "src/x.rs" in str(exc.value)
    assert "unbalanced braces" in str(exc.value)
    # A `}` without its `{` is rejected as well, not skipped.
    with pytest.raises(SystemExit) as exc:
        gen.rust_scopes("fn a() {}\n}\n", "src/y.rs")
    assert "src/y.rs" in str(exc.value)
    # The same file without the URL is balanced and cites `b`.
    fixed = source.replace("https://example.org/x", "example.org")
    reads, _sets, _unresolved = scan_rust(fixed)
    assert reads["MUMDIA_T_AFTER"][0] == ["src/x.rs::b"]


CFG_TEST_ITEMS = """\
enum Finish {
    Normal,
    #[cfg(test)]
    Fail,
}

impl Finish {
    fn apply(self) {
        let _ = std::env::var("MUMDIA_T_AFTER_VARIANT");
        match self {
            Finish::Normal => {}
            #[cfg(test)]
            Finish::Fail => {
                std::env::set_var("MUMDIA_T_TEST_ARM", "1");
            }
        }
    }
}

struct Loader<'a> {
    rows: &'a [usize],
    #[cfg(test)]
    fault: Option<u8>,
}

impl<'a> Loader<'a> {
    fn new(rows: &'a [usize]) -> Loader<'a> {
        Loader {
            rows,
            #[cfg(test)]
            fault: None,
        }
    }

    #[cfg(test)]
    fn with_fault(mut self, fault: Option<u8>) -> Loader<'a> {
        std::env::set_var("MUMDIA_T_TEST_FN", "1");
        self.fault = fault;
        self
    }

    fn run(&self) {
        #[cfg(test)]
        let _ = std::env::var("MUMDIA_T_TEST_STMT");
        #[cfg(test)]
        let _l = self.with_fault(match self.rows.len() {
            0 => None,
            _ => Some(1),
        });
        let _ = std::env::var("MUMDIA_T_AFTER_STMT");
    }
}

#[cfg(test)]
fn generic<A, B>(a: A, b: B) -> Vec<(A, B)>
where
    A: Clone,
    B: Clone,
{
    std::env::set_var("MUMDIA_T_TEST_GENERIC", "1");
    vec![(a, b)]
}

fn last() {
    let _ = std::env::var("MUMDIA_T_LAST");
}
"""


def test_cfg_test_on_braceless_items_blanks_only_that_item():
    """A test-only field, variant, arm or statement blanks itself and nothing more.

    Searching for the next `{` from `#[cfg(test)] fault: None,` inside a struct
    literal swallowed the literal's closing brace and the next function's header,
    so the braces no longer balanced and `rust_scopes` rejected features.rs. Here
    every braceless form is followed by real code whose read must be cited under
    its own function, and every test-only read must be absent.
    """
    reads, sets, unresolved = scan_rust(CFG_TEST_ITEMS)
    assert {name: entry[0] for name, entry in reads.items()} == {
        "MUMDIA_T_AFTER_STMT": ["src/x.rs::Loader::run"],
        "MUMDIA_T_AFTER_VARIANT": ["src/x.rs::Finish::apply"],
        "MUMDIA_T_LAST": ["src/x.rs::last"],
    }
    assert sets == {}
    assert unresolved == []
    # Line count is preserved, and the lines left after the items are untouched.
    blanked = gen.blank_cfg_test(CFG_TEST_ITEMS)
    assert blanked.count("\n") == CFG_TEST_ITEMS.count("\n")
    kept = [line for line in blanked.split("\n") if line.strip()]
    assert "            rows," in kept and "        }" in kept
    assert not any("fault: None" in line or "Fail," in line for line in kept)


def test_repeated_unresolved_reads_are_counted_not_merged():
    """A second non-literal read in the same function changes the document.

    The entry names the function, not the line, so both reads share it. Printed
    once without a count, the second read would leave docs/24 and `--check`
    unchanged, which is the drift the unresolved list exists to show.
    """
    config_text = gen.load_inputs().config_text

    def unresolved_section(n_reads):
        body = "".join("    let _ = std::env::var(name);\n" for _ in range(n_reads))
        source = f"fn dynamic(name: &str) {{\n{body}}}\n"
        inputs = gen.Inputs(config_text, [("src/x.rs", source)], [])
        document, _stats = gen.build_document(inputs)
        start = document.index("environment read(s) whose name is not a literal")
        return document[document.rindex("\n", 0, start) + 1 : document.index("## ", start)]

    one, two = unresolved_section(1), unresolved_section(2)
    assert one.startswith("1 environment read(s)")
    assert "- `src/x.rs::dynamic: env read of `name``\n" in one
    assert two.startswith("2 environment read(s)")
    assert "- `src/x.rs::dynamic: env read of `name`` (2 reads)\n" in two


def test_python_reads_are_cited_by_enclosing_def():
    """Each sidecar read is cited by its qualified `def`, or `<module>`.

    The helper call is followed to its call site even though the helper's read sits
    below a docstring and a blank line; a fixed character window after the signature
    made that depend on the docstring's length.
    """
    reads, sets = scan_python(PY_SOURCE)
    sites = {name: entry[0] for name, entry in reads.items()}
    assert sites == {
        "MUMDIA_T_HELPER": ["scripts/x.py::main"],
        "MUMDIA_T_INNER": ["scripts/x.py::Model.fit.inner"],
        "MUMDIA_T_METHOD": ["scripts/x.py::Model.fit"],
        "MUMDIA_T_TOP": ["scripts/x.py::<module>"],
    }
    assert reads["MUMDIA_T_HELPER"][1] == [("3", ("scripts/x.py::main",))]
    assert sets == {
        "MUMDIA_T_SET": (["scripts/x.py::main"], [('"1"', ("scripts/x.py::main",))])
    }


def test_inserted_blank_lines_do_not_change_rust_sites():
    """Moving Rust code down a file must not change a single cited site.

    If it did, docs/24 would go stale on every merge that touched lines above a read.
    """
    shifted = gen.insert_blank_lines(RUST_SOURCE)
    # The edit really moves the reads; otherwise the comparison below proves nothing.
    assert line_of(shifted, "MUMDIA_T_METHOD") != line_of(RUST_SOURCE, "MUMDIA_T_METHOD")
    assert scan_rust(shifted) == scan_rust(RUST_SOURCE)
    # A blank line inserted at one arbitrary item boundary, as a merge above would.
    one = RUST_SOURCE.replace("fn free()", "\n\n\nfn free()", 1)
    assert scan_rust(one) == scan_rust(RUST_SOURCE)


def test_inserted_blank_lines_do_not_change_python_sites():
    """Moving sidecar code down a file must not change a single cited site."""
    shifted = gen.insert_blank_lines(PY_SOURCE)
    assert line_of(shifted, "MUMDIA_T_METHOD") != line_of(PY_SOURCE, "MUMDIA_T_METHOD")
    assert scan_python(shifted) == scan_python(PY_SOURCE)
    one = PY_SOURCE.replace("class Model:", "\n\n\nclass Model:", 1)
    assert scan_python(one) == scan_python(PY_SOURCE)


def test_repository_reference_has_no_line_numbers_and_is_blank_line_invariant():
    """The committed inputs give the same reference and schema with lines moved.

    This is the property CI relies on: an open pull request stays current when
    another one that only moved lines in rescore.rs or config.rs merges first.
    """
    inputs = gen.load_inputs()
    # The shifted copy really differs in every text; if `transformed` or
    # `insert_blank_lines` stopped editing one kind of input, the comparison below
    # would pass without testing it.
    shifted = inputs.transformed(gen.insert_blank_lines)
    assert shifted.config_text != inputs.config_text
    assert inputs.rust_sources and inputs.py_sources
    for before, after in (
        (inputs.rust_sources, shifted.rust_sources),
        (inputs.py_sources, shifted.py_sources),
    ):
        assert [rel for rel, _ in after] == [rel for rel, _ in before]
        assert all(a != b for (_, b), (_, a) in zip(before, after))
    document, _stats = gen.build_document(inputs)
    schema = gen.schema_text(inputs.config_text)
    assert gen.blank_line_problems(inputs, document, schema) == []
    # No `file.rs:123` / `file.py:45` citation, and no line field in the schema.
    assert re.search(r"\.(?:rs|py):\d", document) is None
    assert '"source_line"' not in schema


def test_blank_line_check_detects_a_line_dependency(monkeypatch):
    """The self-check fails when a citation depends on where a read sits.

    The repository test above asserts that the check finds nothing. This one asserts
    that it can find something: a citation that carries the Rust offset or the
    Python line of a read must be reported for the reference document. Without it,
    a no-op in the shifting would make that test, and `--check` in CI, pass
    whatever the generator emits.
    """
    config_text = gen.load_inputs().config_text
    inputs = gen.Inputs(
        config_text, [("src/x.rs", RUST_SOURCE)], [("scripts/x.py", PY_SOURCE)]
    )

    def problems():
        document, _stats = gen.build_document(inputs)
        return gen.blank_line_problems(inputs, document, gen.schema_text(config_text))

    assert problems() == []

    rust_scope_at = gen.rust_scope_at
    python_scope_at = gen.python_scope_at
    with monkeypatch.context() as patch:
        patch.setattr(
            gen,
            "rust_scope_at",
            lambda spans, offset: f"{rust_scope_at(spans, offset)}@{offset}",
        )
        found = problems()
        assert found and "the reference document changed" in found[0]
    with monkeypatch.context() as patch:
        patch.setattr(
            gen,
            "python_scope_at",
            lambda spans, line: f"{python_scope_at(spans, line)}:{line}",
        )
        found = problems()
        assert found and "the reference document changed" in found[0]
    assert problems() == []


def test_computed_env_defaults_are_read_somewhere():
    """Every `computed:` default names a variable the repository reads.

    The builder only reports a stale entry, so it runs on synthetic sources that read
    none of them; the entry point refuses to write or pass `--check` while one is
    stale, so a removed variable cannot leave its row behind.
    """
    _document, stats = gen.build_document(gen.load_inputs())
    assert stats["stale_computed_env"] == []

    config_text = gen.load_inputs().config_text
    _document, stats = gen.build_document(gen.Inputs(config_text, [], []))
    assert stats["stale_computed_env"] == sorted(gen.COMPUTED_ENV_DEFAULTS)
