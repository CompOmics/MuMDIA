"""Peak-RSS profiler for MuMDIA stages.

Runs a command, samples the resident set size of the whole process tree (the Rust
engine plus any Python sidecars it spawns) at a fixed interval, and reports the peak
per stage. Stage boundaries come from stdout/stderr lines matching ``--stage-regex``
(default: ``mumdia run`` log lines); anything before the first match is stage
``startup``. Alternatively pass ``--stages FILE`` with a JSON list of
``{"name": ..., "cmd": [...]}`` objects to profile each stage as its own process.

Requires ``psutil`` (``pip install psutil``). Output is a TSV with one row per stage:
name, wall seconds, peak RSS of the tree in GB, peak RSS of child processes only in GB
(the sidecar share), and the number of samples. The sampler undercounts transients
shorter than the interval; on Linux cross-check with ``/usr/bin/time -v``.

Example::

    python bench/mem_profile.py --out mem.tsv -- mumdia run --config config.json ...
    python bench/mem_profile.py --stages stages.json --out mem.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

try:
    import psutil
except ImportError:  # pragma: no cover
    sys.exit("mem_profile.py needs psutil: pip install psutil")

GB = 1024**3
ANSI_RE = re.compile(r"\[[0-9;]*[A-Za-z]")
DEFAULT_STAGE_REGEX = r"(?i)\bstage[=: ]+\s*\"?([A-Za-z_][A-Za-z0-9_\-]*)"


def tree_rss(root: psutil.Process) -> tuple[int, int]:
    """Return (total RSS, children-only RSS) in bytes for the process tree."""
    total = 0
    children = 0
    try:
        total = root.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0, 0
    try:
        for c in root.children(recursive=True):
            try:
                r = c.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            children += r
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total + children, children


class StageBook:
    """Per-stage accumulator: peak RSS and wall time, switched by name."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.lock = threading.Lock()
        self.begin("startup")

    def begin(self, name: str) -> None:
        now = time.monotonic()
        with self.lock:
            if self.rows:
                self.rows[-1]["end"] = now
            self.rows.append(
                {"name": name, "start": now, "end": None, "peak": 0, "peak_child": 0, "n": 0}
            )

    def sample(self, total: int, child: int) -> None:
        with self.lock:
            row = self.rows[-1]
            row["peak"] = max(row["peak"], total)
            row["peak_child"] = max(row["peak_child"], child)
            row["n"] += 1

    def close(self) -> None:
        with self.lock:
            if self.rows and self.rows[-1]["end"] is None:
                self.rows[-1]["end"] = time.monotonic()


def run_profiled(cmd: list[str], interval: float, stage_re: re.Pattern | None, book: StageBook,
                 env: dict | None, stage_name: str | None) -> int:
    if stage_name is not None:
        book.begin(stage_name)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        errors="replace", bufsize=1,
    )
    ps = psutil.Process(proc.pid)
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            book.sample(*tree_rss(ps))
            stop.wait(interval)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        if stage_re is not None:
            # tracing's default formatter colours "INFO" and the stage prefix; match on the
            # plain text so a regex like r"INFO\s+(extract|features):" works.
            m = stage_re.search(ANSI_RE.sub("", line))
            if m:
                book.begin(m.group(1))
    rc = proc.wait()
    stop.set()
    t.join()
    return rc


def write_tsv(path: str, book: StageBook) -> None:
    book.close()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("stage\twall_s\tpeak_rss_gb\tpeak_child_rss_gb\tsamples\n")
        for r in book.rows:
            wall = (r["end"] or time.monotonic()) - r["start"]
            fh.write(
                f"{r['name']}\t{wall:.1f}\t{r['peak'] / GB:.3f}\t{r['peak_child'] / GB:.3f}\t{r['n']}\n"
            )
    overall = max((r["peak"] for r in book.rows), default=0) / GB
    print(f"[mem_profile] wrote {path}; overall peak tree RSS {overall:.3f} GB", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="TSV output path")
    ap.add_argument("--interval", type=float, default=0.2, help="sampling interval in seconds")
    ap.add_argument("--stage-regex", default=DEFAULT_STAGE_REGEX,
                    help="regex with one group naming the stage, applied to each output line")
    ap.add_argument("--stages", help="JSON file: list of {name, cmd} to profile one process per stage")
    ap.add_argument("--rust-log", default="info", help="value for RUST_LOG in the child (default info)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command to profile (after --)")
    args = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("RUST_LOG", args.rust_log)
    book = StageBook()
    rc = 0
    if args.stages:
        with open(args.stages, encoding="utf-8") as fh:
            stages = json.load(fh)
        for st in stages:
            rc = run_profiled(st["cmd"], args.interval, None, book, env, st["name"])
            if rc != 0:
                print(f"[mem_profile] stage {st['name']} exited {rc}; stopping", file=sys.stderr)
                break
    else:
        cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
        if not cmd:
            ap.error("no command given; pass it after --")
        rc = run_profiled(cmd, args.interval, re.compile(args.stage_regex), book, env, None)
    write_tsv(args.out, book)
    return rc


if __name__ == "__main__":
    sys.exit(main())
