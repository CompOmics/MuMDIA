# MuMDIA Sensitivity Work — Implementation Status

Living log for the autonomous sensitivity-improvement session. Updated throughout.

## Phase 0 — Baseline (recorded)

- **Repo:** `c:\Users\robbi\OneDrive - UGent\MuMDIA_NG` (git).
- **Branch created:** `feat/sensitivity-improvements`, forked from `fix/audit-correctness-evidence` @ `c6f268f`.
  - The base branch carries the audit-correctness fixes + the 64-bit `LargeListF32` chrom fix (commit `c6f268f`). It is **not** on `main` and is **not pushed**.
- **Build target dir:** `C:/Users/robbi/mumdia_build/{debug,release}` (redirected off OneDrive by machine-local `rust/mumdia/.cargo/config.toml`; do not "fix").
- **Build cmd:** `cd rust/mumdia && cargo build --release`. **Test cmd:** `cargo test`.
- **Baseline tests:** `cargo test` = **64 passed, 0 failed** (50 lib + 2 + 11 + 1 across binaries) — verified this session before any Phase-1 change. No pre-existing failures.
- **Baseline release build:** succeeds (`c6f268f`).
- **Empirical baseline (E. coli file, HYE library, audit-fixed binary), stripped E. coli seqs @1%:**
  - mokapot logreg: 7,907 · native_tda: 6,914 · held-out entrapment (honest, unseen null): 7,909.
  - DIA-NN reference on same file ≈ 10,072; pre-audit MuMDIA base ≈ 9,042.
  - Gate-off probe (Pearson off, presence≥3, native): honest held-out 8,078 (+2.1%) — the Pearson gate discards ~170 real IDs recoverable only by a strong rescorer.

## Assumptions (conservative; recorded per non-interactive policy)

- A1. The primary comparison unit is `normalized modified sequence + charge` (spec 01 §2). MuMDIA's `peptidoform` string + `charge` is that key; `base_peptide_id` is the stripped-sequence key.
- A2. "Candidate" in the MuMDIA code == one row keyed by `candidate_id` (a library precursor peptidoform-charge). Extraction emits **one apex-level PSM per candidate** today (spec 01 §3.1 "one apex" hypothesis holds).
- A3. All new behavior is **default-off / K=1-compatible**; production defaults are unchanged unless an ablation proves a change and tests cover it.
- A4. Large candidate-level output uses **Parquet** via the existing `mumdia-io` typed `Col`/`Table` layer (dependency stack already supports it; spec P0 prefers Parquet).
- A5. Diagnostic instrumentation must be near-zero-cost when disabled (guarded by a config flag).

## Plan (priority order, adapted to the real codebase)

| # | Work | Spec | Status | Commit |
|---|---|---|---|---|
| 0 | Baseline + branch + status log | P0 | DONE | - |
| M | Architecture map (7-agent parallel workflow) | 01 | DONE | ARCHITECTURE_MAP.md |
| 1 | Rejection-reason enum | P0.3, 01 §4 | DONE | 2f46d6d |
| 2 | Candidate audit table + `mumdia audit` (+ metrics/waterfall) | P0.3/P0.4 | DONE | eb9da89 |
| 3 | Top-K peak enumerator + config `retain_top_peaks` (K=1 compat) + tests | P1 | PARTIAL (enumerator+config+tests done; extract wiring = NEXT_STEPS #1) | 2f46d6d |
| 4 | Competition-mode enum wired in compete (none/features-only/unique-evidence/margin-gated) | P2.4 | DONE | de5ae2b |
| 5 | `ARCHITECTURE_MAP.md`, `FEATURE_REGISTRY.md`, `feature_registry.yaml` | P4.1 | DONE | docs |
| 6 | Reference-apex top-K analysis script (Python) | P0.4, 02 §5 D | IN PROGRESS (benchmark agent) | - |
| 7 | Feature-ablation runner (Python) | P4.3 | IN PROGRESS (benchmark agent) | - |
| 8 | `BENCHMARK_GUIDE.md`, `NEXT_STEPS.md` | P7 | NEXT_STEPS done; BENCHMARK_GUIDE pending scripts | - |
| 9 | Fragment claimant / conflict features | P2.1-2.3 | DEFERRED -> NEXT_STEPS #4 (nucleus exists: contested_frac) | - |
| 10 | In-extract precise reason emitter | P0.3 | DEFERRED -> NEXT_STEPS #2 (audit reads sidecar already) | - |

## Completed tasks

- Phase 0 baseline recorded; dedicated branch `feat/sensitivity-improvements`.
- 7-agent architecture-map workflow (679k tokens); ARCHITECTURE_MAP.md written.
- `RejectionReason` enum (16 codes + Reported sentinel), tested.
- `mumdia::peaks::enumerate_peaks` top-K enumerator, 9 synthetic-chromatogram tests.
- Config: `retain_top_peaks` (+validation), `emit_candidate_audit`, `CompetitionMode`
  + `CompeteConfig.mode/margin/unique_evidence_min_fragments/emit_competition_audit`.
- `mumdia audit` stage + subcommand: candidate_audit.parquet + metrics/waterfall,
  verified on real E. coli/HYE data (2 tests).
- Competition modes wired into `compete` via pure `resolve_competition()` (7 tests).
- FEATURE_REGISTRY.md (360 features, 17 families) + feature_registry.yaml.

## Partially completed

- Top-K peaks: enumerator + config + tests done; NOT wired into `extract` (the
  destructive-stage change). Exact hook site documented in NEXT_STEPS.md #1.

## Tests run

- `cargo test` baseline: 64 passed / 0 failed.
- `cargo test` after all Rust changes: 87 passed / 0 failed (68 mumdia lib incl
  peaks(9)/audit(2)/compete(7), 16 mumdia-core incl rejection(5), 2 integration, 1).
- `mumdia audit` real-data smoke on out_ecoli: waterfall reproduced (below).

## Key diagnostic result (candidate audit, E. coli file vs HYE library)

```
search_space = 8,334,126   extracted = 341,754 (trace_recall 4.1%)   reported = 8,568
waterfall: NO_PEAK_GROUP = 7,992,372   FAILED_PRECURSOR_FDR = 332,120
           FAILED_PEPTIDE_FDR = 1,066   REPORTED = 8,568
```

Interpretation: with an 8.3M-candidate combined-species library searched against a
single-species (E. coli) sample, the vast majority of candidates correctly never
form a peak. The audit now makes every loss category countable and stratifiable,
which is the P0 prerequisite for targeting the recoverable losses (peak selection
and FDR), per the spec's decision rules (05 §5).

## Known limitations / risks (running)

- Empirical FDP validation here uses the single E. coli file + HYE entrapment null (one dataset); the spec's held-out multi-dataset reproduction cannot be completed in this environment without additional data.
- Real DIA-NN reference-apex tables are not loaded into the repo; the reference-apex top-K analysis is implemented as a runnable module pending a supplied DIA-NN report.

## Recommended next steps

- See `NEXT_STEPS.md` (written at end of session).
