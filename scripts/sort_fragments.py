"""Rewrite a fragment table ordered by `candidate_id`, streaming.

Usage: python sort_fragments.py <lib_fragments.parquet> [--buckets N] [--tmp DIR]

The engine's range load (`Library::load_range_with`) reads one candidate-id range of the
fragment table by parquet row-group statistics, which needs the table sorted by
`candidate_id` at row-group granularity. The library writers produce that order; this is
the one-time fix for a table written before they did. Two streaming passes over the file
(bucket by candidate-id range, then sort each bucket in memory and append), so the resident
set is one bucket, about 1/N of the table, and the temporary files take one extra copy of
it on disk. The rewrite is in place via a temporary file next to the output.
"""
import argparse
import sys

from _lib_io import sort_fragments_by_candidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fragments")
    ap.add_argument("--buckets", type=int, default=64, help="candidate-id ranges to partition into (default 64)")
    ap.add_argument("--tmp", default=None, help="directory for the bucket files (default: next to the table)")
    a = ap.parse_args()
    n = sort_fragments_by_candidate(a.fragments, buckets=a.buckets, tmp_dir=a.tmp)
    print(f"sorted {n} fragment rows by candidate_id -> {a.fragments}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
