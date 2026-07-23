# Peak and Peptide Competition Design

## 1. Purpose

Competition must prevent multiple false precursor hypotheses from borrowing the same physical DIA signal without suppressing genuinely coeluting peptides.

Do not represent all competition as a single `compete` stage.

## 2. Competition taxonomy

Implement separate stages for:

1. Chromatographic peak competition
2. Duplicate precursor competition
3. Peptide interference competition
4. Peptidoform competition
5. Modification-localization competition
6. Target-decoy competition
7. Reporting deduplication

Each stage must record:

- Input candidates
- Group identifier
- Winner
- Removed candidates
- Score used
- Removal reason

## 3. Chromatographic peak competition

### Problem

One precursor may have several plausible chromatographic peaks.

If one apex is selected before full scoring:

```text
wrong peak selected
    ↓
correct peak discarded
    ↓
final scorer cannot recover precursor
```

### Recommended design

1. Generate local maxima from consensus fragment evidence.
2. Retain top `K` peaks per precursor.
3. Calculate complete features for every peak group.
4. Score peaks out of fold.
5. Select the best peak for each modified-sequence and charge precursor.
6. Optionally retain a second peak when evidence supports repeated elution or unresolved ambiguity.

Suggested initial values:

```text
K ∈ {3, 5, 10}
```

### Evaluation

Report reference-apex recall at top 1, 3, 5, and 10.

## 4. Shared fragment evidence

### Problem

An observed MS2 trace may match fragments from many candidates:

```text
observed m/z trace
  ├── candidate A fragment
  ├── candidate B fragment
  ├── candidate C fragment
  └── decoy D fragment
```

Giving full intensity to all candidates can create false high scores. Hard assignment can cause a high-abundance peptide to steal evidence from a real low-abundance peptide.

### Initial non-destructive solution

Do not reassign raw evidence in the first implementation.

Calculate:

- Claimant count per fragment
- Contested fragment count
- Contested intensity fraction
- Unique fragment count
- Unique intensity fraction
- Shared-trace correlation
- Strongest competitor score
- Score margin
- Residual spectral similarity
- Local conflict-group size

Use these features during rescoring.

## 5. Conflict graph

Represent each candidate peak group as a node.

Node identity:

```text
run
modified sequence
charge
candidate apex
peak boundaries
```

Add an edge when candidates:

1. Occur in compatible or overlapping DIA isolation windows
2. Have overlapping peak boundaries or close apexes
3. Share fragment m/z values within tolerance
4. Use a material amount of the same chromatographic evidence

Edge features:

```text
shared_fragment_count
shared_intensity_fraction_A
shared_intensity_fraction_B
apex_rt_difference
boundary_overlap
shared_trace_correlation
unique_fragment_count_A
unique_fragment_count_B
unique_intensity_fraction_A
unique_intensity_fraction_B
score_difference
```

## 6. Competition strategies to benchmark

### Strategy A: No hard competition

- Preserve all candidates
- Add conflict features
- Let final FDR handle ambiguity

This is the baseline.

### Strategy B: Hard winner-take-all

Within sufficiently strong conflict groups:

```text
retain highest initial discriminant score
```

This is simple but may suppress real coeluting peptides.

### Strategy C: Unique-evidence-aware competition

Allow multiple candidates to survive when each has enough independent evidence.

Example rule:

```text
unique_fragment_count >= 2
and unique_intensity_fraction >= threshold
and unique_fragments_coelute
```

Otherwise, apply winner-take-all.

### Strategy D: Margin-gated competition

Remove a competitor only when:

```text
winner_score - loser_score >= margin
```

and shared evidence exceeds a threshold.

### Strategy E: Soft score penalty

Do not remove candidates. Apply a penalty derived from:

- Contested intensity
- Lack of unique evidence
- Stronger competing candidates
- Conflict-group size

### Strategy F: Residual-evidence pass

1. Score all candidates.
2. Select strongly supported candidates.
3. Model the fragment signal they explain.
4. Subtract or downweight explained signal.
5. Rescore weaker candidates using residual evidence.

This is the most complex strategy and should be implemented only after A-E are benchmarked.

## 7. Peptidoform and charge handling

### Charge states

Different charge states can both be real.

Recommended rule:

```text
do not directly compete distinct charge states
```

Instead, add cross-charge corroboration features.

### Distinct modification states

Unmodified and modified forms can co-exist.

Do not compete all forms sharing the same base sequence.

Compete only candidates that are mutually exclusive explanations of the same local signal.

### Localization variants

Candidates with the same composition but different modification sites require localization scoring.

Retain them until site-determining ion evidence is evaluated.

## 8. Target-decoy competition

Target-decoy competition is not the same as interference competition.

Recommended sequence:

```text
peak scoring
    ↓
peak selection
    ↓
local peptide-interference handling
    ↓
peptidoform/localization handling
    ↓
final rescoring
    ↓
target-decoy competition
    ↓
q-value estimation
```

Document the competition unit explicitly:

- Spectrum
- Peak group
- Precursor
- Peptide
- Peptidoform
- Protein group

## 9. Competition feature audit

For every competition feature, verify:

- Symmetric calculation for targets and decoys
- No use of final labels
- Calculation within cross-validation folds where needed
- Stability across runs
- No dependence on candidate enumeration artifacts
- No direct reuse of final model score as an input to itself

## 10. Primary experiment

Run:

| Mode | Peak candidates | Shared evidence | Variant competition |
|---|---:|---|---|
| A | 1 | full evidence | before rescoring |
| B | top 5 | full evidence | before rescoring |
| C | top 5 | conflict features | after initial rescoring |
| D | top 5 | unique-evidence-aware | after initial rescoring |
| E | top 5 | margin-gated hard competition | after initial rescoring |
| F | top 5 | residual-evidence pass | after initial rescoring |

Measure:

- Targets at matched empirical FDP
- Decoys and entrapments removed
- Correct targets removed
- Correct-peak recall
- Low-intensity identifications
- Modified-peptide identifications
- Conflict groups containing multiple independently supported targets
- Runtime and memory

## 11. Recommended initial implementation

Start with:

```text
top-K peaks
+ non-destructive conflict graph
+ unique/contested evidence features
+ competition after initial rescoring
```

Do not start with raw centroid ownership or global winner-take-all assignment.
