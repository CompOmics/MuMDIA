# MuMDIA Sensitivity Evaluation and Improvement

This documentation set describes a structured program for diagnosing and improving the sensitivity of MuMDIA relative to DIA-NN.

The central principle is:

> Do not optimize the final classifier until the workflow identifies where candidate precursors are lost.

A DIA-NN-only identification can disappear because it was absent from the search space, never generated as a candidate, extracted incorrectly, assigned to the wrong chromatographic peak, outcompeted by another peptide interpretation, ranked poorly, or removed by false-discovery-rate filtering.

## Documentation map

1. [`01_workflow_and_gap_analysis.md`](01_workflow_and_gap_analysis.md)  
   Working model of the MuMDIA workflow and the major conceptual differences to investigate relative to DIA-NN.

2. [`02_sensitivity_diagnostic_plan.md`](02_sensitivity_diagnostic_plan.md)  
   End-to-end identification-loss ladder, benchmark design, oracle experiments, and reporting requirements.

3. [`03_feature_evaluation.md`](03_feature_evaluation.md)  
   Framework for auditing current features, testing new features, preventing leakage, and deciding which features to retain.

4. [`04_peak_and_peptide_competition.md`](04_peak_and_peptide_competition.md)  
   Recommended design for chromatographic peak competition, shared-fragment competition, peptidoform competition, and target-decoy competition.

5. [`05_experiment_matrix.md`](05_experiment_matrix.md)  
   Controlled experiment matrix, evaluation metrics, decision rules, and acceptance criteria.

6. [`06_agent_implementation_backlog.md`](06_agent_implementation_backlog.md)  
   Agent-ready implementation tickets ordered by priority.

## Main hypotheses

The largest potential sensitivity losses are expected to arise from one or more of the following:

1. A single chromatographic apex is selected before the final scoring model can compare alternative peak groups.
2. Candidate peptides can reuse the same observed fragment signal without sufficiently modeling contested evidence.
3. Charge states, modification states, or related precursor candidates may be competed too early.
4. Retention-time, mass-error, and spectral predictions are insufficiently calibrated to the current run.
5. The scorer lacks features describing unique evidence, contested evidence, candidate ambiguity, peak shape, or interference.
6. The decoy or competition design produces a high-scoring false-candidate tail, causing conservative q-value thresholds.
7. Feature-selection results are evaluated using target-decoy separation rather than empirical false discovery.

## Recommended first implementation sequence

1. Add full candidate observability and rejection reason codes.
2. Retain the top `K` chromatographic peak groups per precursor.
3. Build a peak-group conflict graph without removing candidates.
4. Add unique-evidence and contested-evidence features.
5. Move variant competition until after initial rescoring.
6. Introduce empirical entrapment validation.
7. Add two-pass RT and mass calibration.
8. Run grouped feature-family ablations.
9. Add conservative post-scoring interference competition.
10. Evaluate cross-run evidence only after single-run behavior is validated.

## Evidence boundary

The exact current internals of DIA-NN are not fully public and may differ by version. Treat DIA-NN-inspired competition strategies in these documents as hypotheses to benchmark, not as claims of exact implementation equivalence.

Every comparison must record:

- MuMDIA commit
- DIA-NN version
- Complete command lines
- Input hashes
- Search-space manifest
- Prediction-library version
- FDR columns and filtering rules
- Random seeds
- Runtime and memory

## Definition of success

The program is successful when:

- At least 95% of DIA-NN-only precursors receive an explicit earliest-loss category.
- Candidate recall, correct-peak recall, target-ranking recall, and FDR losses are quantified separately.
- Improvements reproduce on held-out datasets.
- Identification gains are measured at matched empirical false discovery proportion.
- Modified peptides and low-intensity precursors are evaluated independently.
- Every retained feature or competition mechanism has an ablation result.
