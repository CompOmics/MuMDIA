# Searching very large predicted libraries

Measured 2026-09-17/18 on one Astral immunopeptidomics run (`AT10234AUH`, 71 min, 114
isolation windows of 4 m/z over 270-727) against two DIA-NN 2.3.2 predicted libraries of the
human proteome: all 9-mers (29.5M precursors, 59M with reverse decoys) and all 8-12-mers
(142.7M precursors, 203M with decoys after the isolation-range filter). Host: EPYC 7H12,
256 threads, 2 TB. Everything below is single-run; the multi-file route is `mumdia run`
with several `--mzml`, which pays the calibration once.

## Recipe

1. **Import** the DIA-NN parquet with `scripts/import_diann_lib.py` (streams the input; 9-mers
   19 min at 15 GB, 8-12-mers 1:40 h at 69 GB).
2. **Drop what the run cannot measure.** `scripts/mz_range_survivors.py <lib_precursors>
   <isolation_windows.parquet> <survivors>` lists the precursors inside the run's isolation
   range; `scripts/assemble_survivors.py` renumbers the library to them and remaps the
   fragments. 41% of the 9-mer library and 29% of the 8-12-mer library were charge-1
   precursors above the top window and could never yield a fragment. Do this before the
   decoys: every later stage scales with the row count.
3. **Decoys** with `scripts/make_reverse_decoys.py` (8-12-mers: 1:09 h, 225 GB).
4. **Seed search** as usual (9-mers 2:13 at 30 GB; 8-12-mers 10 min at 96 GB, 951 confident).
5. **Multi-head RT calibration, sharded.** `deeplc_finetune.py --multihead 80` predicts about
   6,000 sequences per second per process on CPU or GPU (the featurisation is the bound), so a
   125.9M-sequence library is 6 hours in one process. `scripts/mh_shard_predict.py uniq`
   writes the unique DECOY_-stripped sequences into N shards (the charge states of one
   peptidoform are far apart in an m/z-sorted table, so row-range shards would predict each
   sequence 2.4 times), `predict` fits the same ridge from the same seeds in every shard and
   predicts its slice, `merge` joins the predictions back. 12 shards: 36 min plus 6 min merge.
6. **rt-im-train** on the calibrated table. Residual median 81 s -> 29 s on the 9-mers, `w_rt`
   469 s -> 239 s; 8-12-mers 33 s and 286 s from 910 anchors.
7. **No tag prescan.** Uncapped `prescan.top_peaks` keeps 59.4% of the 9-mer library on wide
   windows, which is exactly the isolation-range fraction, and 59.1% on the narrow ones: the
   trimer screen removes nothing this run cannot measure anyway. Capped, it removes real
   peptides (seed retention 67% at 25 peaks, 96% at 60, 100% from 300). On the 8-12-mers the
   60-peak screen kept 41% of candidates and cost 19% of the peptides for 40% less wall.
8. **Extract on the streaming path** with `extract.windows_in_flight` 2-8, then features,
   compete and `nn_torch` rescore. Use `mumdia run` rather than stage-by-stage chains: it
   passes the MS1 scans and the seed's mass calibration to extract (see below).
9. **Second pass** from the union of first-pass identifications across the runs
   (`assemble_survivors.py` on the identified candidate ids), per-run multi-head
   calibration, pooled rescore.

## Two omissions in the hand-built chains

The arm table below was produced by stage-by-stage shell chains that called `extract` with
`--ms2` only. `mumdia run` also passes `--ms1` (the converted MS1 scans; 6,404 here) and
`--mass-cal <seed>.masscal.json` (the seed had learned an 11.6 ppm fragment tolerance; the
chains ran at the 20 ppm default). Rerun through the orchestrator, the same 9-mer search went
from 4,371 to 5,656 peptides and from 3 h 40 min to 1 h 09 min (a third fewer noise
candidates into rescore, and the streaming extract), and the 8-12-mer search from 7,961 to
10,213 peptides. Treat the arm numbers as lower bounds and use the orchestrated numbers.

## What it yields

| library | candidates into extract | PSMs into rescore | peptides at 1% | protein groups |
|---|---|---|---|---|
| 9-mers, wide windows, prescan 25 peaks | 7.3M | 2.06M | 1,872 | |
| 9-mers, multi-head windows, prescan 60 peaks | 14.8M | 3.84M | 3,533 | |
| 9-mers, multi-head windows, uncapped | 34.9M | 8.07M | 4,371 | 2,977 |
| 8-12-mers, multi-head windows, prescan 60 peaks | 83M | 19.1M | 6,434 | 3,543 |
| 8-12-mers, multi-head windows, uncapped | 203M | 39.7M | 7,961 | 4,160 |

PSM-level decoy fraction 0.91-0.98% in every arm. The 7,961 split 830 / 4,280 / 1,716 / 923 /
212 over lengths 8-12 and recover 861 of the 910 confident seed peptides; 4,042 of the 9-mers
are shared with the 9-mer-only search, which loses 329 to the larger space and gains 238.

## Orchestrated runs, seven files

First pass: one `mumdia run` per file against the full calibrated 8-12-mer library (203M
candidates), MS1 and mass calibration on, `windows_in_flight` 2-4, 96-128 threads, one host
per file. Second pass: the union of the first-pass target precursors at run-level
`q_value <= 0.01` (21,342) with their paired decoys, assembled by `assemble_survivors.py`
into a 42,684-precursor library; one `mumdia run` per file on it with per-run multi-head
calibration (`rt_im_train.multihead_calibration = 80`, about 80 s per file, RT windows
62-73 s), then one pooled `rescore` over the seven competed tables. This is the analogue of
DIA-NN's empirical-library workflow, whose second pass on these files searched a
23,186-precursor library.

| file | first pass, peptides | first pass, wall | pass 2, peptides | pass 2, precursors | DIA-NN pass 2, precursors |
|---|---|---|---|---|---|
| AT10234AUH | 10,213 | 6:36 h | 15,182 | 15,983 | 16,427 |
| AT10237AUH | 10,745 | 9:25 h (shared host) | 15,396 | 16,293 | 17,131 |
| AT10240AUH | 9,739 | 3:49 h | 14,664 | 15,379 | 16,290 |
| AT10253AUH | 9,781 | 3:55 h | 14,880 | 15,680 | 16,420 |
| AT10256AUH | 15,372 | 8:04 h | 16,888 | 18,422 | 18,420 |
| AT10265RJB | 12,984 | 12:08 h (shared host) | 16,423 | 17,738 | 18,039 |
| AT10273AUH | 10,342 | 4:10 h | 14,741 | 15,528 | 13,379 (DIA-NN's AT10673AUH) |

Pooled over the seven runs, pass 2 reports 17,829 peptides, 19,745 precursors and 7,208
protein groups at 1% (116,854 PSMs). DIA-NN's library carries 559 cysteinylated precursors
per file (`--var-mod UniMod:312`) that the imported library lacks. The second-pass FDR of
both tools is decoy-based on a library that is almost entirely real; neither has an
entrapment check here.

Stage trace of DIA-NN's 16,427 AT10234AUH precursors through the orchestrated first pass:
59.7% identified as the same precursor, 2.4% at another charge, 23.7% extracted but rescored
above q 0.01 (1,379 of them at q <= 0.05), 11.5% not extracted with the RT inside our window,
2.5% outside it, 0.1% not in the search space. The missing tail is low abundance: median
DIA-NN intensity 0.28M against 1.5M for what is identified.

## What it costs (8-12-mers, uncapped, 128 threads)

| stage | wall | peak RSS |
|---|---|---|
| extract (streaming, 8 windows in flight) | 3:52 h | 335 GB |
| features (39.7M PSMs) | 2:39 h | |
| rescore (39.7M PSMs) | 53 min | |

Extract's accumulator payload was 158 GB with 8 windows in flight on 203M candidates; halve
`windows_in_flight` to halve it. The 39.7M accepted rows' columns are the other large term.

## Pitfalls, each hit once

- **A candidate allowlist used to force the serial extract path** (fixed in #100). Before
  that, `--restrict-candidates` meant the whole run's hits stayed resident regardless of
  `windows_in_flight`: 35M allowed candidates took 2:08 h at 325 GB, 83M aborted at 471 GB
  with `memory allocation of 12288 bytes failed`. That host has no ulimit but strict
  overcommit with a 1 TB `CommitLimit`; a small allocation failing far below physical RAM is
  that limit.
- **pyarrow's 2 GiB `string` arrays** (fixed in #98). `Table.from_pandas` converts a column as
  one array, and a slice of a >2 GiB `large_string` array keeps the parent's offsets, so
  both the whole-frame write and a naive sliced write failed with `input array too large`.
  The writer now slices and re-bases. The same limit hits `pa.concat_arrays` of `string`
  arrays; concatenate as `large_string`.
- **The rescorer's init sample** (fixed in #99). On a 40M-PSM pool with a few thousand true
  PSMs, a 300k-row sample holds too few of them for any single feature to reach 1% FDR; the
  worker now escalates the sample (every fold needed it here, one to 4.8M rows).
- **Editing a chain script while bash runs it** changes what bash reads next.

## Helpers added here

| script | purpose |
|---|---|
| `scripts/mz_range_survivors.py` | candidate ids inside the run's isolation range, as a survivors table |
| `scripts/assemble_survivors.py` | a survivors table (prescan or the above) -> renumbered library with pair-linked target/decoy union on `peptidoform_id` |
| `scripts/shard_parquet.py` | split a parquet into row-group-aligned shards, or concatenate shards |
| `scripts/mh_shard_predict.py` | deduplicated, sharded multi-head calibration: `uniq`, `predict`, `merge` |
