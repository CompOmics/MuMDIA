# digest (Stage A) and peptidoforms (Stage A2)

> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

These two stages build the run-independent, experiment-wide peptide search space
from a FASTA file, before any spectra are touched. They run once per experiment
and their outputs are reused across all runs.

- **digest** (Stage A, PLAN.md) performs a fully-tryptic in-silico digest of the
  input proteins, deduplicates the resulting peptides by stripped sequence, and
  mints a paired target-decoy null (reverse or seeded scramble). Output is one
  `peptides.parquet` table of stripped target and decoy sequences with a
  `target`/`decoy` label.
- **peptidoforms** (Stage A2) expands each stripped peptide into concrete
  peptidoforms by enumerating fixed and variable modifications and precursor
  charge states, emitting each as a ProForma-lite string with UniMod modification
  names. Output is one `peptidoforms.parquet` table.

Neither stage reads spectra or config-run parameters (tolerances, RT, etc.); they
depend only on the FASTA and the `digest` / `peptidoforms` config sections. In the
library-input path (`--lib-precursors`/`--lib-fragments`) both stages are skipped
entirely; the imported library already carries sequences, modifications, charges,
and decoys (see `run.rs` library-input branch, and CLAUDE.md "DIA-NN library
recipe").

## Files

| Path | Role |
|---|---|
| `rust/mumdia/crates/mumdia/src/stages/digest.rs` | Stage A: FASTA parse, tryptic digest, dedup, decoy generation, `peptides.parquet` writer |
| `rust/mumdia/crates/mumdia/src/stages/peptidoforms.rs` | Stage A2: fixed/variable mod + charge enumeration, ProForma-lite emission, `peptidoforms.parquet` writer |
| `rust/mumdia/crates/mumdia-core/src/config.rs` | `DigestConfig`, `DecoyConfig`, `PeptidoformsConfig`, `ResidueMod`, `Enzyme`, `DecoyStrategy`, `UnknownModPolicy`, and `Config::validate` |
| `rust/mumdia/crates/mumdia-core/src/constants.rs` | `is_standard_residue` / `residue_mass` (the 20-residue allowlist that gates the digest) |
| `rust/mumdia/crates/mumdia-core/src/mass.rs` | `unimod_mass` (the UniMod name allowlist that validates modification names) |
| `rust/mumdia/crates/mumdia-core/src/schema.rs` | `artifact::PEPTIDES` and `artifact::PEPTIDOFORMS` logical names + schema versions |
| `rust/mumdia/crates/mumdia/src/main.rs` | CLI wiring: `Cmd::Digest` (main.rs:402), `Cmd::Peptidoforms` (main.rs:413) |
| `rust/mumdia/crates/mumdia/src/stages/run.rs` | Orchestrator wiring in the FASTA-build branch (run.rs:116, run.rs:126) |

## Inputs and outputs

**digest**
- Consumes: a FASTA file path (`DigestParams.fasta`, digest.rs:140). Parsed by
  `read_fasta` (digest.rs:19) into `(accession, sequence)` pairs. FASTA parsing is
  minimal: `>` starts a record, the accession is the first whitespace-delimited
  token after `>`, and sequence lines are concatenated and trimmed. There is no
  handling of `*` stop characters or lowercase residues beyond the residue
  allowlist check applied later. An unreadable FASTA path is the one error case in
  the parser (`with_context("reading fasta {path}")`, digest.rs:20); a file with no
  `>` records is not an error, it simply yields zero proteins and an empty (but
  valid) `peptides.parquet`.
- Produces: `peptides.parquet` (logical name `peptides`, schema version 1, from
  `artifact::PEPTIDES`, schema.rs:11) plus a sibling `<out>.report.json`
  (`ArtifactReport`, digest.rs:232). Report `stats` carries `n_targets` and
  `n_decoys`; `params` records enzyme, missed cleavages, length bounds, decoy
  strategy, and `rng_seed` (digest.rs:239).

`peptides.parquet` column schema (written at digest.rs:214-226):

| Column | Type | Meaning |
|---|---|---|
| `id` | u32 | Monotonic row id, assigned interleaved (target then its decoy) |
| `peptide` | str | Stripped sequence (target sequence, or the rewritten decoy sequence) |
| `protein` | str | `;`-joined accessions for targets; `DECOY_` + the same join for decoys |
| `start` | i32 | 0-based start offset of the peptide in the protein of first occurrence |
| `end` | i32 | 0-based end offset (exclusive) of that first occurrence |
| `label` | str | Literal `"target"` or `"decoy"` |
| `target_id` | i32 | `-1` for a target; for a decoy, the `id` of its paired target |
| `decoy_strategy` | str | Lowercased strategy name (e.g. `reverse`, `scramble`) |

**peptidoforms**
- Consumes: `peptides.parquet` (`PeptidoformsParams.peptides`, peptidoforms.rs:62),
  read via `Table::read` (peptidoforms.rs:70). It reads columns `id`, `peptide`,
  `protein`, `label`, `target_id`.
- Produces: `peptidoforms.parquet` (logical name `peptidoforms`, schema version 1,
  from `artifact::PEPTIDOFORMS`, schema.rs:12) plus `<out>.report.json`. Report
  `params` record fixed/variable mods, `max_variable_mods`, and the charge range
  (peptidoforms.rs:157); `stats` is empty.

`peptidoforms.parquet` column schema (written at peptidoforms.rs:135-147):

| Column | Type | Meaning |
|---|---|---|
| `id` | u32 | Monotonic peptidoform row id (fresh counter across all output rows) |
| `peptide_id` | u32 | The digest `id` of the row this peptidoform came from |
| `base_peptide_id` | u32 | The digest `id` of the underlying **target** peptide (for decoys, the paired target id; for targets, the own id) |
| `peptide` | str | Stripped sequence, copied through unchanged |
| `peptidoform` | str | ProForma-lite string with UniMod names in brackets, e.g. `PEPC[Carbamidomethyl]M[Oxidation]K` |
| `charge` | i32 | Precursor charge (one row per charge in `[charge_min, charge_max]`) |
| `label` | str | `"target"` / `"decoy"`, copied from the digest row |
| `protein` | str | Copied from the digest row |

## How it works

### digest (digest.rs:147 `run`)

1. **Parse FASTA** with `read_fasta` (digest.rs:19).
2. **Per-protein cleavage** in `cleavage_sites` (digest.rs:43). Sites is seeded
   with `0`, then for each residue that is `K` or `R` a cut index `i+1` is pushed;
   Trypsin/P (`Enzyme::TrypsinP`, the default) always cuts, classic `Enzyme::Trypsin`
   suppresses the cut when the next residue is `P` (`before_p`, digest.rs:48). The
   sequence length is appended as a final site if not already present
   (digest.rs:58). Sites are strictly increasing, so peptides are contiguous
   half-open `[start, end)` spans.
3. **Peptide enumeration** in `digest_protein` (digest.rs:66). For every ordered
   pair of sites `(i, j)` with `i < j`, the number of missed cleavages is
   `j - i - 1`; the inner loop `break`s once that exceeds `cfg.missed_cleavages`
   (digest.rs:72), so at most `missed_cleavages + 1` fragments are joined. Length
   bounds `min_len`/`max_len` filter on residue count (digest.rs:77). A peptide
   is dropped entirely if any residue is not one of the 20 standard amino acids
   (`is_standard_residue`, digest.rs:81 -> constants.rs:54), which is how `X`, `B`,
   `Z`, `U`, `*`, and lowercase get excluded.
4. **Dedup by stripped sequence** (digest.rs:153-172). Three structures: a
   `HashMap<String, Vec<String>>` mapping peptide to accessions, a `HashMap` from
   peptide to its first `(start, end)`, and an insertion-order `Vec<String>`
   named `order`. On a repeat occurrence only the accession list is extended (and
   only if the accession is not already present, digest.rs:163); the first
   occurrence records position and pushes the peptide into `order`. Iteration for
   output is over `order`, so **insertion order is the emission order** and is
   deterministic (a protein's peptides in sequence order, proteins in FASTA
   order). The borrow-then-clone pattern (digest.rs:162) is a deliberate
   allocation optimization, not a semantic detail.
5. **Row assembly and decoy minting** (digest.rs:178-210). A single monotonic
   `next_id` counter assigns ids. For each target peptide in `order`: emit the
   target row (`label = "target"`, `target_id = -1`, digest.rs:189-191), then, if
   the strategy is neither `None` nor `DiannShift`, call `make_decoy`
   (digest.rs:197) and, on `Some`, emit the decoy row immediately after with a
   fresh id, `label = "decoy"`, `target_id = tid` (the target's id), and the
   `DECOY_`-prefixed protein string. Ids therefore interleave target/decoy in
   pairs.
6. **Write** the eight columns with `write_table` (digest.rs:214) and emit the
   `ArtifactReport` (digest.rs:232).

**Decoy generation** (`make_decoy`, digest.rs:92). Peptides shorter than 3
residues return `None` (no decoy). Both realized strategies keep the C-terminal
residue fixed and rewrite the interior `b[..n-1]`:
- `DecoyStrategy::Reverse` (digest.rs:99): reverse the first `n-1` residues, then
  re-append the original last residue. Reversing all-but-last keeps the enzyme's
  C-terminal `K`/`R` in place while moving the N-terminus. Test
  `reverse_decoy_keeps_cterm` (digest.rs:280) pins `PEPTIDER -> EDITPEPR`.
- `DecoyStrategy::Scramble` (digest.rs:105): a deterministic Fisher-Yates shuffle
  of the first `n-1` residues. The PRNG state is seeded per peptide as
  `rng_seed ^ fnv1a(pep)` (digest.rs:108) and advanced with `splitmix64`
  (digest.rs:122). The loop runs `i` from `len-2` down to `1` and swaps index `i`
  with a uniform `j` in `0..=i` (digest.rs:109-113), so every interior position
  including index 0 (the N-terminal residue) can move while the last residue is
  re-appended untouched. Test `scramble_is_deterministic` (digest.rs:287) pins
  reproducibility and the fixed C-terminus.
- `DecoyStrategy::DiannShift` and `DecoyStrategy::None` return `None`
  (digest.rs:117-118); `DiannShift` is unrealized here by design (the comment
  notes it would be a predict-frag fragment-shift, PLAN.md Section 11) and is
  additionally rejected by `Config::validate` (config.rs:1058) because it would
  yield zero decoys and an invalid FDR.

### peptidoforms (peptidoforms.rs:68 `run`)

1. **Read** the digest table (peptidoforms.rs:70-75).
2. **Validate modification names up front** (peptidoforms.rs:78-82): every fixed
   and variable mod name is looked up in `unimod_mass` (mass.rs:13); an unknown
   name is a hard `anyhow::bail!` (message `unknown modification '<name>' in config
   (fixed/variable)`). This runs once before any row is processed, so a typo fails
   fast. The `unimod_mass` allowlist (mass.rs:14-24) currently recognizes exactly
   eight names: `Carbamidomethyl`, `Oxidation`, `Acetyl`, `Phospho`, `Deamidated`,
   `Methyl`, `Dimethyl`, `Carbamyl`. Any `fixed_mods`/`variable_mods` name outside
   this set aborts the stage; extending the set is a `mass.rs` edit (see "How to
   extend").
3. **Per digest row** (peptidoforms.rs:89): compute `base_peptide_id` as
   `target_id` when it is `>= 0` (the row is a decoy) else the row's own `id`
   (peptidoforms.rs:91-95).
4. **Fixed mods** (peptidoforms.rs:98-105): for each residue index, if any
   `fixed_mods` entry's `residue` char matches, record `(index, name)`. Fixed mods
   apply to every matching residue and are always present.
5. **Variable-mod candidate positions** (peptidoforms.rs:107-114): the same
   residue-char scan collects `(index, name)` candidates for `variable_mods`.
6. **Combination enumeration** (`variable_combos`, peptidoforms.rs:33): returns the
   empty subset plus every size-`k` subset of candidate positions for
   `k = 1..=max_variable_mods.min(n)`, using an in-place combinatorial index
   advance (peptidoforms.rs:40-57). The count is `1 + sum_{k=1..min(max,n)} C(n,k)`;
   test `combos_bounded` (peptidoforms.rs:184) pins `n=3, max=1 -> 4` and
   `max=2 -> 7`.
7. **Emit** (peptidoforms.rs:116-132): for each combo, merge fixed mods and the
   combo, sort by position (peptidoforms.rs:119), build the ProForma-lite string
   with `proforma` (peptidoforms.rs:18), and emit one output row per charge in
   `[charge_min, charge_max]` (peptidoforms.rs:121). Each row gets a fresh `id`,
   the source `peptide_id`, the computed `base_peptide_id`, and the copied
   `label`/`protein`.

`proforma` (peptidoforms.rs:18) walks the stripped sequence and, at each residue
index, appends `[<name>]` for the **first** mod found at that position
(`mods.iter().find`, peptidoforms.rs:22). This is where a second modification at
the same position is silently dropped (see gotchas).

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `digest::run` | digest.rs:147 | Stage A entry point; parse, digest, dedup, mint decoys, write `peptides.parquet` |
| `read_fasta` | digest.rs:19 | Minimal FASTA parser to `(accession, sequence)` pairs |
| `cleavage_sites` | digest.rs:43 | Trypsin/P vs Trypsin cut-site indices for one sequence |
| `digest_protein` | digest.rs:66 | Enumerate length-bounded, missed-cleavage-bounded, standard-residue-only peptides |
| `make_decoy` | digest.rs:92 | Reverse or seeded-scramble decoy of the interior, C-term fixed; `None` if `len < 3` or strategy `None`/`DiannShift` |
| `splitmix64` | digest.rs:122 | Deterministic PRNG step for the scramble shuffle |
| `fnv1a` | digest.rs:130 | Per-peptide hash mixed into the scramble seed |
| `DigestParams` | digest.rs:139 | `fasta`, `out`, `cfg: &DigestConfig`, `rng_seed`, `config_hash` |
| `peptidoforms::run` | peptidoforms.rs:68 | Stage A2 entry point; validate mods, enumerate mods+charges, write `peptidoforms.parquet` |
| `proforma` | peptidoforms.rs:18 | Build ProForma-lite string from stripped peptide + `(position, name)` mods |
| `variable_combos` | peptidoforms.rs:33 | Enumerate variable-mod position subsets up to `max_var` |
| `PeptidoformsParams` | peptidoforms.rs:61 | `peptides`, `out`, `cfg: &PeptidoformsConfig`, `config_hash` |
| `unimod_mass` | mass.rs:13 | Name -> monoisotopic delta allowlist (the set of mods that validate) |
| `is_standard_residue` | constants.rs:54 | The 20-residue allowlist gating digest output |

## Configuration

`digest` reads `DigestConfig` (config.rs:219). `peptidoforms` reads
`PeptidoformsConfig` (config.rs:240). Both structs use `#[serde(default,
deny_unknown_fields)]`, so an unknown key is a load error and any omitted field
takes the default below. `rng_seed` (config.rs:1011, default `0`) is a top-level
`Config` field, passed to digest as `DigestParams.rng_seed` (main.rs:409,
run.rs:120).

| Field | Section | Default | Effect |
|---|---|---|---|
| `enzyme` | digest | `TrypsinP` | `trypsin_p` cuts after K/R including before P; `trypsin` suppresses the cut before P |
| `missed_cleavages` | digest | `2` | Max missed cleavages; joins at most `missed_cleavages + 1` tryptic fragments |
| `min_len` | digest | `5` | Minimum peptide length (residues) |
| `max_len` | digest | `50` | Maximum peptide length (residues) |
| `decoy.strategy` | digest | `reverse` | `reverse` / `scramble` realized; `diann_shift` and `none` produce no decoy (`diann_shift` rejected by `validate`) |
| `fixed_mods` | peptidoforms | `[{C: Carbamidomethyl}]` | Applied to every matching residue |
| `variable_mods` | peptidoforms | `[{M: Oxidation}]` | Enumerated as optional subsets |
| `max_variable_mods` | peptidoforms | `1` | Max simultaneous variable mods per peptidoform |
| `charge_min` | peptidoforms | `2` | Lowest precursor charge emitted |
| `charge_max` | peptidoforms | `3` | Highest precursor charge emitted |
| `unknown_modification` | peptidoforms | `error` | Declared policy `error`/`skip`; see gotchas (only `error` behavior is realized) |

`rng_seed` is `0` by default (config.rs:1028) and only affects the `scramble`
decoy strategy; `reverse` is independent of the seed.

Config was recently pruned of dead fields in this area. `ResidueMod`
(config.rs:271) is only `{ residue: char, name: String }`; there is no
position/terminal specifier field. Unlike the enclosing `DigestConfig` /
`PeptidoformsConfig`, `ResidueMod` carries `#[serde(deny_unknown_fields)]` but
**not** `#[serde(default)]` (config.rs:269-270), so both `residue` and `name` are
mandatory in every mod entry; an entry missing either key is a load error rather
than a defaulted value. The doc comment on `residue` mentions `*` for
"any / terminal handled separately in MVP", but no such handling exists in the
code (see gotchas). There is no `DecoySource` field on `DigestConfig` (the
"dead/unwired config surface" list in CLAUDE.md refers to types elsewhere, not
here). `DecoyConfig` (config.rs:208) carries only `strategy`.

## Invariants, determinism, gotchas

- **Determinism.** Emission order is FASTA order then in-protein sequence order,
  preserved by the `order` vector (digest.rs:169), not by HashMap iteration. Ids
  are a single monotonic counter. Scramble is fully deterministic for a fixed
  `rng_seed` because the PRNG is seeded per peptide from the sequence
  (digest.rs:108). No floats are summed in either stage, so the numeric-order
  determinism concerns from other stages do not apply.
- **Target/decoy pairing.** A decoy's `target_id` points at its target's `id` and
  the pair is emitted adjacently; targets carry `target_id = -1`. Downstream FDR
  keeps the label in competition keys (see CLAUDE.md FDR / compete), so this
  pairing must stay intact. `peptidoforms` propagates the link via
  `base_peptide_id` (peptidoforms.rs:91).
- **Residue allowlist.** Peptides containing any non-standard residue are dropped
  wholesale at digest time (digest.rs:81); they never reach peptidoforms. This is
  the only place ambiguity codes (`X`/`B`/`Z`/`U`) are handled.
- **No decoy-equals-target guard.** `make_decoy` does not check that the decoy
  differs from its target, and there is no cross-target decoy dedup. Short or
  palindromic sequences can yield a decoy identical to some target; this is a
  known limitation (CLAUDE.md "Correctness"). Peptides with `len < 3` silently get
  no decoy (digest.rs:95); with the default `min_len = 5` this only bites if a
  caller lowers `min_len`.
- **No terminal modifications.** `ResidueMod` matches on an exact residue char
  (peptidoforms.rs:100, peptidoforms.rs:109). There is no N-term/C-term or
  wildcard `*` matching, so protein/peptide N-terminal acetylation and similar
  terminal mods cannot be enumerated. The `mass.rs` parser does model terminal mod
  deltas, but Stage A2 never produces terminal-mod ProForma strings.
- **Second mod at the same position is dropped.** `proforma` (peptidoforms.rs:22)
  keeps only the first `(position, name)` at each index. If a residue matches both
  a fixed mod and a variable mod (or two variable mod specs), the merged `mods`
  list has two entries at that index and the second is silently lost from the
  emitted string. This is the documented "second mod at same position dropped"
  limit (CLAUDE.md, peptidoforms status "partial").
- **`unknown_modification` policy is declared but not wired.** The config field
  exists with values `error`/`skip` (config.rs:247, `UnknownModPolicy`,
  config.rs:280), but `peptidoforms::run` unconditionally `bail!`s on an unknown
  name (peptidoforms.rs:78-82) and never reads `cfg.unknown_modification`. `skip`
  behaves identically to `error` today. Treat `skip` as unimplemented.
- **`config_hash` param is unused.** Both `DigestParams.config_hash`
  (digest.rs:144) and `PeptidoformsParams.config_hash` (peptidoforms.rs:65) are
  threaded from the CLI/orchestrator but never read inside either `run`. The
  content hash written to the report comes from `blake3_file` over the output
  Parquet (digest.rs:238, peptidoforms.rs:156), not from `config_hash`.
- **`decoy_strategy` string is derived by Debug-formatting the enum**
  (`format!("{:?}", ...).to_lowercase()`, digest.rs:191). It is a display string,
  not a parseable round-trip; changing enum variant names changes the column
  value.
- **Charge range is inclusive and unvalidated.** `charge_min..=charge_max`
  (peptidoforms.rs:121) is emitted as-is; a `charge_min > charge_max` config
  yields zero peptidoform rows silently rather than an error.
- **Library-input mode bypasses both stages.** When a prebuilt library is passed,
  `run` takes the library-input branch and never calls `digest`/`peptidoforms`
  (run.rs, and the FASTA branch at run.rs:111); do not assume `peptides.parquet` /
  `peptidoforms.parquet` exist for every run.

## How to extend / modify

- **New enzyme.** Add a variant to `Enzyme` (config.rs:54) and a match arm in
  `cleavage_sites` (digest.rs:49). Keep the site list strictly increasing and
  terminated at `seq.len()`; the rest of `digest_protein` is enzyme-agnostic.
  Semi-tryptic search would require changing how spans are enumerated in
  `digest_protein` (digest.rs:66), not just the cut list.
- **New decoy scheme.** Add a `DecoyStrategy` variant (config.rs:17) and a match
  arm in `make_decoy` (digest.rs:98). Preserve the C-terminal residue and
  determinism (seed the PRNG from the sequence as scramble does, digest.rs:108).
  If the scheme can produce a decoy equal to its target, add the guard the current
  code lacks. Realizing `DiannShift` means removing the `validate` rejection
  (config.rs:1058) and implementing a fragment-m/z shift at predict-frag, not a
  sequence rewrite here.
- **New modification.** Add the UniMod name and monoisotopic delta to
  `unimod_mass` (mass.rs:13); it is the single allowlist both the Stage A2
  validation (peptidoforms.rs:79) and the mass model consult. Then reference it by
  name in `fixed_mods`/`variable_mods`.
- **Support a second mod per position or terminal mods.** Change `proforma`
  (peptidoforms.rs:18) to emit all mods at a position (e.g. concatenate multiple
  `[...]` groups) instead of `find`-ing the first, and extend `ResidueMod`
  (config.rs:271) plus the residue-matching loops (peptidoforms.rs:98-114) with a
  position/terminal specifier. The mass model in `mass.rs` already carries
  `n_term_mod` fields to build on.
- **Schema changes.** Adding or reordering `peptides.parquet` /
  `peptidoforms.parquet` columns is a schema change; bump the version in
  `artifact::PEPTIDES` / `artifact::PEPTIDOFORMS` (schema.rs:11-12) and update the
  downstream readers (`Table::read` callers in predict-frag and later stages).
- **Testing.** Existing unit tests live at the bottom of each file
  (digest.rs:256, peptidoforms.rs:173): `trypsin_p_cleaves_after_kr`
  (digest.rs:261) pins Trypsin/P cut sites, `reverse_decoy_keeps_cterm`
  (digest.rs:280) and `scramble_is_deterministic` (digest.rs:287) pin the fixed
  C-terminus plus scramble reproducibility, `proforma_places_mods`
  (peptidoforms.rs:178) pins bracket placement, and `combos_bounded`
  (peptidoforms.rs:184) pins the subset counts. They cover cleavage, decoy C-term/
  determinism, ProForma placement, and combo bounds. There is no stage-level test
  that round-trips through Parquet or exercises `run` end to end (CLAUDE.md "test
  gaps"); add one when changing the output schema.
