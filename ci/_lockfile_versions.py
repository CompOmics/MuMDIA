"""Version-tolerant `--check` for the documents generated from `rust/mumdia/Cargo.lock`.

`THIRD_PARTY_LICENSES.md` and `sbom.cdx.json` name every crate with its exact version, so
any lockfile change made them stale, including a Dependabot patch bump that changes no
crate, no licence and no notice. Dependabot does not run the generators, so every one of
its pull requests failed CI until someone pushed a regeneration commit by hand (#154,
#156, #157, #158, #176 in October 2026).

The CI check therefore compares the two texts with every semantic version masked. It
still fails when a crate is added or removed, when a crate appears in a second version,
when a licence expression or repository changes, and when a crate's notice text changes:
everything the licence obligations rest on. A version-only difference passes with a
note. `--strict` compares exactly, and the release gate (`release.yml`, validate-tag)
runs it, so a tagged release always ships documents that name the exact versions inside
the binary. The release pull request regenerates both files, as it always has.
"""

from __future__ import annotations

import re
import sys

# A semantic version as cargo writes one: x.y.z with an optional pre-release and build
# suffix (`zstd-sys` is `2.0.16+zstd.1.5.7`). Not preceded or followed by a word
# character or a dot, so the two-part "Version 2.0" of a licence text and the years of a
# copyright line are left alone.
SEMVER = re.compile(
    r"(?<![\w.])\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?(?![\w.])"
)


def mask_versions(text: str) -> str:
    """`text` with every semantic version replaced by `<version>`."""
    return SEMVER.sub("<version>", text)


def check(out, current: str, text: str, regen: str, strict: bool, report_stale) -> int:
    """Exit code of `--check` for generated file `out`.

    `current` is the committed text (line endings normalised), `text` the regenerated
    one. Exact equality passes; under `strict` anything else fails. Otherwise a
    difference in versions alone passes with a note, and any other difference fails with
    the diff `report_stale` prints.
    """
    if current == text:
        return -1  # up to date; the caller prints its own summary
    if not strict and mask_versions(current) == mask_versions(text):
        changed = sorted(
            set(SEMVER.findall(text)) ^ set(SEMVER.findall(current)),
        )
        print(
            f"{out} differs from the lockfile in crate versions only ("
            f"{len(changed)} version strings), which the CI check tolerates; the release "
            f"gate does not. Regenerate with:\n    {regen}",
            file=sys.stderr,
        )
        return 0
    return report_stale(out, current, text, regen)
