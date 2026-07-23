# Agent Implementation Backlog

## 1. Working conventions

The agent should:

- Implement small reviewable changes
- Add tests with every change
- Preserve backwards-compatible defaults where possible
- Record all configuration in output metadata
- Avoid irreversible candidate filtering in early stages
- Prefer diagnostic flags before production behavior changes
- Produce Parquet tables for large candidate-level outputs

## 2. Priority 0: Observability and benchmark harness

### P0.1 — Search-space manifest

Implement a machine-readable search-space manifest and validation.

Acceptance criteria:

- Effective configuration is exported.
- MuMDIA and DIA-NN settings can be compared.
- The benchmark fails on important mismatches.
- Input hashes and software versions are stored.

### P0.2 — Normalized output converter

Implement a common output schema.

Acceptance criteria:

- All top candidates are retained.
- Final reported candidates are retained.
- Modified sequences are normalized consistently.
- Precursor keys are reproducible.

### P0.3 — Candidate audit table

Create `candidate_audit.parquet`.

Required columns:

```text
run_id
precursor_id
modified_sequence
charge
target_decoy_label
entrapment_label
candidate_generated
traces_extracted
peak_generated
peak_selected
variant_selected
target_decoy_winner
passed_precursor_fdr
passed_peptide_fdr
reported
rejection_reason
```

### P0.4 — Stage-level metrics

Generate:

- Search-space recall
- Candidate recall
- Trace recall
- Top-K peak recall
- Peptide-ranking recall
- Competition losses
- FDR losses

### P0.5 — Entrapment runner

Acceptance criteria:

- Entrapment databases are generated reproducibly.
- Empirical FDP is reported at precursor, peptide, and protein levels.
- Results are stratified by modification and charge.

## 3. Priority 1: Top-K chromatographic peaks

### P1.1 — Candidate peak generator

Implement configurable local peak generation.

CLI:

```text
--retain-top-peaks K
```

Acceptance criteria:

- Supports `K = 1, 3, 5, 10`.
- Stores apex, boundaries, and initial score for every peak.
- Correct-peak top-K recall can be computed.
- Default behavior remains reproducible.

### P1.2 — Peak-level feature calculation

Calculate complete feature vectors independently for each candidate peak.

Acceptance criteria:

- No feature uses information from another candidate's final label.
- Peak features are cached.
- Computation is deterministic.

### P1.3 — Peak-selection model

Implement grouped out-of-fold peak scoring.

Acceptance criteria:

- Peak groups from one precursor remain in the same fold.
- Out-of-fold scores are stored.
- Top-1, top-3, and top-5 recall are reported.

## 4. Priority 2: Competition graph and contested evidence

### P2.1 — Fragment claimant index

For every observed fragment trace, record candidate claimants.

Acceptance criteria:

- Claimant counts are available.
- Candidate-fragment mappings are queryable.
- Memory behavior is benchmarked.

### P2.2 — Peak-group conflict graph

Create graph edges for candidates with overlapping RT and fragment evidence.

Acceptance criteria:

- Edge thresholds are configurable.
- Edge features are stored.
- Graph construction is deterministic.
- Large connected components are diagnosed.

### P2.3 — Conflict features

Add:

```text
contested_fragment_count
contested_intensity_fraction
unique_fragment_count
unique_intensity_fraction
strongest_competitor_score
score_margin
conflict_group_size
shared_trace_correlation
```

Acceptance criteria:

- Target and decoy calculations are symmetric.
- Missing values are documented.
- Runtime is reported.

### P2.4 — Competition modes

CLI:

```text
--competition-mode none
--competition-mode features-only
--competition-mode unique-evidence
--competition-mode margin-gated
```

Acceptance criteria:

- Every removed candidate records its winner and reason.
- Competition happens after initial peak scoring.
- Empirical FDP is reported for every mode.

## 5. Priority 3: Calibration and extraction

### P3.1 — Two-pass mass calibration

Acceptance criteria:

- Robust precursor and fragment calibration.
- Local uncertainty estimate.
- Before-and-after residual plots.
- Fallback for insufficient calibrants.

### P3.2 — Nonlinear RT calibration

Acceptance criteria:

- Monotonic mapping.
- Cross-validated residuals.
- Local RT uncertainty.
- Separate modified-peptide diagnostics.

### P3.3 — Adaptive extraction windows

Acceptance criteria:

- Window width is based on calibrated uncertainty.
- Minimum and maximum bounds are configurable.
- Candidate recall and interference are compared with fixed windows.

## 6. Priority 4: Feature evaluation framework

### P4.1 — Feature registry

Create `feature_registry.yaml`.

Acceptance criteria:

- Every feature has family, level, direction, missing policy, and leakage note.
- Documentation is generated automatically.
- Unknown features fail validation.

### P4.2 — Feature audit command

Suggested CLI:

```bash
mumdia-feature-audit \
  --features candidate_features.parquet \
  --metadata candidate_metadata.parquet \
  --registry feature_registry.yaml \
  --output reports/feature_audit
```

Required outputs:

- Missingness
- Distributions
- Correlation clusters
- Target/decoy/entrapment comparisons
- Run drift
- Leakage warnings

### P4.3 — Ablation runner

Suggested CLI:

```bash
mumdia-feature-ablation \
  --config feature_experiments.yaml \
  --outer-group dataset_id \
  --inner-group precursor_id \
  --metric targets_at_empirical_fdp \
  --fdp 0.01
```

Acceptance criteria:

- Supports family removal and addition.
- Supports linear and tree models.
- Stores fold assignments.
- Produces paired dataset-level results.

## 7. Priority 5: New features

### P5.1 — Uncertainty-normalized residuals

Implement RT, mass, and mobility normalized residuals.

### P5.2 — Fragment evidence distributions

Implement median, dispersion, threshold fractions, and effective fragment count.

### P5.3 — Peak-shape and apex-dispersion features

Implement consensus profile, boundary agreement, apex dispersion, symmetry, and shoulder score.

### P5.4 — MS1 and isotope features

Implement monoisotope, isotope correlation, spacing, coelution, and precursor-fragment agreement.

### P5.5 — Candidate ambiguity features

Implement alternative-target, alternative-peak, and decoy margins using an earlier-stage score.

## 8. Priority 6: Peptidoforms and localization

### P6.1 — Delay variant competition

Acceptance criteria:

- Different charge states are not prematurely collapsed.
- Modified and unmodified forms can coexist.
- Candidate counts before and after competition are reported.

### P6.2 — Localization competition

Acceptance criteria:

- Localization variants remain until site-determining evidence is calculated.
- Site-determining ion count and intensity are available.
- Localization confidence is separated from precursor confidence.

### P6.3 — Staged modification search

Acceptance criteria:

- Calibration stage and extended-modification stage are configurable.
- Combined confidence estimation is validated with entrapment.
- Modification-specific performance is reported.

## 9. Priority 7: Reporting

### P7.1 — HTML benchmark report

Include:

1. Search-space parity
2. Reported q-value versus empirical FDP
3. Identification counts at matched FDP
4. MuMDIA/DIA-NN overlap
5. Identification-loss waterfall
6. Candidate recall
7. Correct-peak recall
8. Peptide-ranking recall
9. Competition losses
10. Feature-family ablations
11. Performance by intensity
12. Performance by modification
13. Runtime and memory
14. Representative candidate chromatograms

### P7.2 — Candidate diagnostic bundle

For selected candidate IDs, export:

- Fragment chromatograms
- MS1 traces
- Peak boundaries
- Predicted and observed spectra
- Feature values
- Conflict neighbors
- Removal reason

## 10. Suggested first sprint

Implement in this exact order:

1. Search-space manifest
2. Candidate audit table
3. Rejection reason codes
4. Top-K peak retention
5. Reference-apex top-K analysis
6. Conflict graph
7. Unique and contested evidence features
8. Competition after initial rescoring
9. Entrapment validation
10. Feature-family ablation runner

## 11. Completion checklist

- [ ] Search-space parity is verified.
- [ ] Candidate-level audit output exists.
- [ ] At least 95% of missing precursors have a loss category.
- [ ] Top-K peak recall is measured.
- [ ] Competition stages are separated.
- [ ] Entrapment FDP is measured.
- [ ] Feature families have ablation results.
- [ ] Gains reproduce on held-out datasets.
- [ ] Modified peptides are evaluated separately.
- [ ] Runtime and memory are reported.
