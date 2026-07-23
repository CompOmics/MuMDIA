# MuMDIA Workflow and Gap Analysis

## 1. Purpose

This document provides a working decomposition of the MuMDIA workflow so each stage can be evaluated independently.

The intended pipeline model is:

```text
FASTA and modification configuration
    ↓
peptide and peptidoform generation
    ↓
precursor charge-state generation
    ↓
fragment, retention-time, and optional mobility prediction
    ↓
candidate indexing and seed search
    ↓
run-specific calibration
    ↓
chromatogram extraction
    ↓
candidate peak-group generation
    ↓
peak-group feature calculation
    ↓
peak selection and precursor competition
    ↓
semi-supervised rescoring
    ↓
target-decoy competition and q-value estimation
    ↓
peptide and protein aggregation
```

The implementation agent must confirm which of these stages exist in the active MuMDIA branch and update this document when behavior differs.

## 2. Primary comparison unit

Use the following precursor identity as the primary comparison key:

```text
normalized modified sequence + precursor charge
```

Also retain:

- Run
- Protein assignment
- Precursor m/z
- Apex retention time
- Modification localization
- Decoy status
- Entrapment status

Do not begin with protein counts. Protein inference can conceal precursor-level losses.

## 3. Major differences to investigate relative to DIA-NN

### 3.1 Timing of chromatographic peak selection

Potential MuMDIA behavior:

```text
extract traces
    ↓
choose one apex using heuristic evidence
    ↓
calculate final feature vector
    ↓
rescore
```

Recommended behavior:

```text
extract traces
    ↓
generate top-K candidate peak groups
    ↓
calculate features for every peak group
    ↓
initial out-of-fold peak scoring
    ↓
select best peak per precursor
    ↓
final confidence scoring
```

A wrong early apex is unrecoverable if the correct peak is discarded before rescoring.

### 3.2 Treatment of shared fragment evidence

In wide-window DIA, a single observed fragment trace can match fragments from many candidate peptides.

Potential failure:

```text
one observed trace
    ↓
copied as full evidence to several candidates
    ↓
false candidates inherit real chromatographic signal
    ↓
target-decoy separation decreases
```

The preferred first step is not destructive peak ownership. Instead, calculate:

- Number of candidates claiming each fragment
- Fraction of candidate intensity that is contested
- Fraction of evidence that is unique
- Best competing candidate score
- Shared-trace correlation
- Residual spectral similarity after stronger candidates are explained

### 3.3 Candidate peak competition

Distinguish:

1. Multiple chromatographic peaks for the same precursor
2. Multiple peptides explaining the same local DIA evidence
3. Multiple peptidoforms explaining the same signal
4. Multiple charge states of the same peptide
5. Target-decoy competition
6. Duplicate reporting removal

These must be separate stages with separate logs.

### 3.4 Scoring architecture

Potential differences to investigate:

- Linear versus nonlinear feature interactions
- Separate peak-selection and confidence models
- Run normalization
- Prediction calibration
- Interference-aware scoring
- Unique-fragment evidence
- Candidate ambiguity
- Cross-charge corroboration
- Modification-aware residuals

The question is not whether MuMDIA uses fewer or more features. The question is whether its features provide independent evidence in the low-FDR operating region.

### 3.5 Calibration

Evaluate whether MuMDIA uses:

- Global or local mass calibration
- Nonlinear RT calibration
- Prediction uncertainty
- Run-specific peak-width distributions
- Ion-mobility calibration
- Modified-peptide-specific residual models

A fixed extraction window may be simultaneously too wide for clean regions and too narrow for poorly calibrated regions.

### 3.6 Decoy behavior and FDR

More permissive extraction can increase both true and false candidates. If false candidates borrow real fragment traces, the high-scoring decoy tail can force a stricter score threshold.

Therefore, sensitivity must be reported at:

```text
matched empirical false discovery proportion
```

not only at each tool's nominal 1% q-value.

## 4. Required workflow instrumentation

For every candidate, store stage flags:

```text
in_search_space
candidate_generated
traces_extracted
peak_group_generated
selected_as_peak_winner
selected_as_variant_winner
passed_initial_score
passed_target_decoy_competition
passed_precursor_fdr
passed_peptide_fdr
reported
```

Store one `rejection_reason` corresponding to the earliest failed stage.

Suggested reasons:

```text
PEPTIDE_NOT_GENERATED
MODIFICATION_NOT_ALLOWED
CHARGE_OUT_OF_RANGE
PRECURSOR_MZ_OUT_OF_RANGE
NO_VALID_FRAGMENTS
WRONG_ISOLATION_WINDOW
RT_PRUNED
CANDIDATE_CAP_REACHED
NO_FRAGMENT_TRACES
NO_PEAK_GROUP
PEAK_NOT_SELECTED
OUTCOMPETED_BY_TARGET
OUTCOMPETED_BY_DECOY
FAILED_PRECURSOR_FDR
FAILED_PEPTIDE_FDR
REMOVED_DURING_REPORTING
```

## 5. Questions the implementation must answer

1. How many DIA-NN-only precursors are absent from the MuMDIA search space?
2. How many are generated in exhaustive candidate mode?
3. For how many is the DIA-NN apex within MuMDIA's top 1, 3, 5, and 10 candidate peaks?
4. How often does a wrong MuMDIA peak win over a peak near the DIA-NN apex?
5. How often is evidence dominated by shared fragments?
6. How many valid charge states or peptidoforms are removed by early competition?
7. How many targets rank well but fail q-value filtering?
8. Does competition improve empirical FDR enough to recover additional weak targets?
9. Which losses are enriched for low-intensity or modified peptides?
10. Which differences remain after matching the search space and prediction library?
