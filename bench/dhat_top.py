"""Rank the allocation sites in a dhat-heap.json by bytes held at the global peak.

`bench/mem_profile.py` says which stage holds the memory and `mumdia::memlog` says how
much the buffers we chose to name hold. This closes the loop: dhat records every
allocation, so the program points below account for the whole peak, including whatever
the model in docs/27 section 1 never named.

Usage::

    python bench/dhat_top.py dhat-heap.json --top 25
    python bench/dhat_top.py dhat-heap.json --top 40 --frames 6 --filter mumdia

The dhat format (version 2) reports, per program point (`pps`): `gb` bytes live at the
instant of the global heap maximum, `mb` the maximum this point alone ever held, and
`tb` the bytes it allocated in total over the run. Peak attribution is `gb`: those are
the bytes actually resident when the process was at its worst. A point with a large `tb`
and a small `gb` is allocation churn, which costs time rather than resident set.
"""

from __future__ import annotations

import argparse
import json
import sys

MIB = 1024**2


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if "pps" not in d or "ftbl" not in d:
        sys.exit(f"{path} is not a dhat heap profile (no pps/ftbl)")
    return d


def frames(d: dict, pp: dict, keep: int, needle: str | None) -> list[str]:
    ftbl = d["ftbl"]
    out = []
    for fi in pp.get("fs", []):
        if fi >= len(ftbl):
            continue
        f = ftbl[fi]
        if needle and needle not in f:
            continue
        # drop the allocator shims that head every stack
        if any(s in f for s in ("alloc::alloc", "__rust_alloc", "dhat::", "RawVec")):
            continue
        out.append(f)
        if len(out) >= keep:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=25, help="program points to print")
    ap.add_argument("--frames", type=int, default=4, help="stack frames per point")
    ap.add_argument("--filter", help="only show frames containing this substring")
    ap.add_argument(
        "--sort",
        choices=("gb", "mb", "tb"),
        default="gb",
        help="gb = bytes live at the global peak (default), mb = this point's own peak, "
        "tb = total bytes ever allocated (churn)",
    )
    args = ap.parse_args()

    d = load(args.path)
    pps = d["pps"]
    total_gb = sum(p.get("gb", 0) for p in pps)
    print(f"{args.path}: {len(pps)} program points, {total_gb / MIB:,.0f} MiB live at the global peak")
    print(f"{'bytes@peak':>14}  {'share':>6}  {'own peak':>12}  {'total alloc':>14}  site")
    ranked = sorted(pps, key=lambda p: p.get(args.sort, 0), reverse=True)[: args.top]
    for pp in ranked:
        gb = pp.get("gb", 0)
        share = 100.0 * gb / total_gb if total_gb else 0.0
        fs = frames(d, pp, args.frames, args.filter)
        head = fs[0] if fs else "(no matching frame)"
        print(
            f"{gb / MIB:>11,.1f} MiB  {share:>5.1f}%  "
            f"{pp.get('mb', 0) / MIB:>9,.1f} MiB  {pp.get('tb', 0) / MIB:>11,.1f} MiB  {head}"
        )
        for f in fs[1:]:
            print(f"{'':>14}  {'':>6}  {'':>12}  {'':>14}    {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
