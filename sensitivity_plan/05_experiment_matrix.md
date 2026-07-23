# Experiment Matrix and Acceptance Criteria

## 1. Experimental principles

Each experiment must:

- Change one component at a time
- Use fixed search spaces
- Use fixed random seeds
- Preserve raw candidate outputs
- Run on development and held-out datasets
- Report empirical FDP
- Report runtime and memory
- Report subgroup behavior

## 2. Core experiment matrix

| ID | Change | Main question |
|---|---|---|
| E0 | Current baseline | What is the present sensitivity gap? |
| E1 | Exhaustive candidate generation | Is pruning losing valid candidates? |
| E2 | Top-3 peak retention | Is early peak selection limiting sensitivity? |
| E3 | Top-5 peak retention | How much additional peak recall is gained? |
| E4 | RT oracle | Is RT calibration or pruning limiting sensitivity? |
| E5 | Peak oracle | Is the correct peak generated but ranked poorly? |
| E6 | Empirical fragment oracle | Are predicted transitions limiting sensitivity? |
| E7 | Wide extraction windows | Are tolerance settings too strict? |
| E8 | Adaptive calibrated windows | Can calibration improve signal-to-background? |
| E9 | No interference competition | Baseline effect of shared evidence |
| E10 | Conflict features only | Can non-destructive competition improve ranking? |
| E11 | Unique-evidence-aware competition | Can false candidates be suppressed safely? |
| E12 | Margin-gated hard competition | Does conservative removal improve FDR? |
| E13 | Residual-evidence rescoring | Can weaker coeluting peptides be recovered? |
| E14 | Move variant competition after rescoring | Is early competition removing real candidates? |
| E15 | Add MS1/isotope features | Does precursor evidence increase discrimination? |
| E16 | Add peak-shape features | Does chromatographic modeling improve peak ranking? |
| E17 | Add uncertainty-normalized residuals | Are raw prediction errors poorly calibrated? |
| E18 | Add candidate-ambiguity features | Does local competition improve confidence? |
| E19 | Linear versus LightGBM | Are nonlinear interactions important? |
| E20 | Alternative decoy design | Is FDR estimation limiting sensitivity? |
| E21 | Staged modification search | Is broad peptidoform competition too severe? |
| E22 | Run-specific prediction calibration | Does domain adaptation improve modified peptides? |
| E23 | Cross-run re-extraction | What is the experiment-level recovery ceiling? |

## 3. Primary metrics

### Sensitivity

- Precursors at reported 1% q-value
- Precursors at 1% empirical FDP
- Peptides at 1% empirical FDP
- Modified peptides at 1% empirical FDP
- Protein groups at validated FDR

### Pipeline recall

- Search-space recall
- Candidate-generation recall
- Trace-extraction recall
- Correct peak in top 1
- Correct peak in top 3
- Correct peak in top 5
- Correct peptide rank 1
- Target-decoy win rate

### Calibration

- Reported q-value versus empirical FDP
- Posterior calibration
- Entrapment rate by score decile
- FDP by subgroup

### Cost

- Runtime
- Peak memory
- Candidate count
- Disk usage
- Feature-calculation time
- Model-training time

## 4. Subgroup reporting

Report all primary metrics by:

- Intensity decile
- Precursor charge
- Peptide length
- Modification class
- Modification count
- Number of theoretical fragments
- Number of observed fragments
- Candidate density
- Isolation-window width
- Peak width
- RT region
- Instrument
- Acquisition method

## 5. Decision rules

### Candidate recall below 95%

Prioritize:

- Search-space generation
- Isolation-window assignment
- Modification enumeration
- RT pruning
- Candidate caps
- Isotope handling

Do not prioritize final rescoring.

### Candidate recall high, peak recall low

Prioritize:

- Top-K peak generation
- RT calibration
- Peak detection
- Peak boundaries
- Fragment consensus
- Adaptive peak widths

### Correct peak available, peptide rank poor

Prioritize:

- Unique-fragment evidence
- Spectral agreement
- Fragment coelution
- Interference modeling
- Candidate ambiguity
- Prediction adaptation

### Ranking good, q-value yield poor

Prioritize:

- Decoy construction
- Target-decoy competition
- Cross-validation
- Score calibration
- Group-specific FDR
- Removal of false candidates borrowing real signal

### Single-run performance good, experiment coverage poor

Prioritize:

- Run alignment
- Empirical chromatogram libraries
- Controlled re-extraction
- Transfer-specific confidence estimation

## 6. Acceptance criteria for code changes

A change can be accepted when:

- Empirical FDP remains controlled
- Gain reproduces on at least two held-out datasets
- No major subgroup degradation occurs
- Runtime and memory are acceptable
- Output is deterministic under fixed seeds
- All new rejection decisions are logged
- Feature and model versions are recorded

Suggested quantitative thresholds:

- At least 1% overall precursor gain at 1% empirical FDP, or
- At least 5% gain in a prespecified weak subgroup, or
- At least 2% gain in correct-peak recall, or
- Meaningful q-value calibration improvement

## 7. Required statistical summaries

For each experiment report:

- Absolute identification difference
- Relative identification difference
- Bootstrap confidence interval by run or dataset
- Seed-to-seed variability
- Paired results by dataset
- Empirical FDP difference
- Runtime difference

Do not select a feature based on one favorable run.

## 8. Stop conditions

Stop expanding model complexity when:

- Candidate or peak recall, not scoring, remains the bottleneck
- Gains disappear on held-out datasets
- Empirical FDP degrades
- Runtime cost is disproportionate
- Additional features are redundant
- Feature effects are unstable across acquisition methods
