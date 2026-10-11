"""The version-tolerant check of the documents generated from the Cargo lockfile."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ci"))

from _lockfile_versions import check, mask_versions  # noqa: E402

ROW = "| `{name}` | {version} | MIT OR Apache-2.0 | `Apache-2.0` | [repo](https://x) |"


def doc(*rows):
    return "\n".join(ROW.format(name=n, version=v) for n, v in rows) + "\n"


def stale(*_args):
    return 1


def run(current, text, strict=False):
    return check("doc.md", current, text, "regen", strict, stale)


def test_identical_is_up_to_date():
    d = doc(("thiserror", "2.0.20"))
    assert run(d, d) == -1


def test_a_version_only_bump_passes_but_not_under_strict():
    old = doc(("thiserror", "2.0.20"), ("zstd-sys", "2.0.16+zstd.1.5.7"))
    new = doc(("thiserror", "2.0.21"), ("zstd-sys", "2.1.1+zstd.1.5.7"))
    assert run(old, new) == 0
    assert run(old, new, strict=True) == 1


def test_an_added_crate_fails():
    assert run(doc(("a", "1.0.0")), doc(("a", "1.0.0"), ("b", "1.0.0"))) == 1


def test_a_second_version_of_one_crate_fails():
    # #158: mzdata-bindata bumped without mzdata left both versions in the lockfile.
    old = doc(("mzdata-bindata", "0.66.6"))
    new = doc(("mzdata-bindata", "0.66.7"), ("mzdata-bindata", "0.67.2"))
    assert run(old, new) == 1


def test_a_licence_change_fails():
    old = doc(("a", "1.0.0"))
    assert run(old, old.replace("MIT OR Apache-2.0", "GPL-3.0"), strict=False) == 1


def test_licence_texts_and_purls_are_masked_correctly():
    assert mask_versions("pkg:cargo/thiserror@2.0.21") == "pkg:cargo/thiserror@<version>"
    # Two-part numbers and years are not versions.
    text = "Apache License Version 2.0, January 2004. Copyright 2014-2020."
    assert mask_versions(text) == text
    assert mask_versions("1.2.3-rc.1") == "<version>"
