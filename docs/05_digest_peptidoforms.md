# digest (Stage A) and peptidoforms (Stage A2)

> Part of the MuMDIA developer documentation (see docs/README.md).

## Purpose

These two stages build the run-independent, experiment-wide peptide search space
from a FASTA file, before any spectra are touched. They run once per experiment
and their outputs are reused across all runs.

- **digest** (Stage A, PLAN.md) performs a fully-tryptic in-silico digest of the
  input proteins (also emitting the initiator-methionine-excised form of each
  protein-N-terminal peptide by default, see "How it works"), deduplicates the
  resulting peptides by stripped sequence, and
  mints a collision-checked, paired target-decoy null (reverse or seeded
  scramble). Output is one `peptides.parquet` table of stripped target and decoy
  sequences with a `target`/`decoy` label.
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
| `rust/mumdia/crates/mumdia/src/main.rs` | CLI wiring: `Cmd::Digest` (main.rs:415), `Cmd::Peptidoforms` (main.rs:426) |
| `rust/mumdia/crates/mumdia/src/stages/run.rs` | Orchestrator wiring in the FASTA-build branch (run.rs:132, run.rs:149) |

## Inputs and outputs

**digest**
- Consumes: a FASTA file path (`DigestParams.fasta`, digest.rs:177). Parsed by
  `read_fasta` (digest.rs:25) into `(accession, sequence)` pairs. FASTA parsing is
  minimal: `>` starts a record, the accession is the first whitespace-delimited
  token after `>`, and sequence lines are concatenated, trimmed, and upper-cased
  (`to_ascii_uppercase`, digest.rs:41). Lowercase residues are therefore
  normalized rather than silently discarded by the later allowlist check. There is
  still no special handling of `*` stop characters; they simply fail the
  standard-residue allowlist during digest. An unreadable FASTA path is the one
  error case in the parser (`with_context("reading fasta {path}")`, digest.rs:26);
  a file with no `>` records is not an error, it simply yields zero proteins and
  an empty (but valid) `peptides.parquet`.
- Produces: `peptides.parquet` (logical name `peptides`, schema version 1, from
  `artifact::PEPTIDES`, schema.rs:11) plus a sibling `<out>.report.json`
  (`ArtifactReport`, digest.rs:312). Report `stats` carries `n_targets`,
  `n_decoys`, `decoy_collision_retries`, and `dropped_target_decoy_pairs`
  (digest.rs:302-311); `params` records enzyme, missed cleavages, length bounds,
  decoy strategy, `rng_seed`, and `max_decoy_attempts` (digest.rs:319-326).

`peptides.parquet` column schema (written at digest.rs:288-297):

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
- Consumes: `peptides.parquet` (`PeptidoformsParams.peptides`, peptidoforms.rs:180),
  read via `Table::read` (peptidoforms.rs:189). It reads columns `id`, `peptide`,
  `protein`, `label`, `target_id`.
- Produces: `peptidoforms.parquet` (logical name `peptidoforms`, schema version 1,
  from `artifact::PEPTIDOFORMS`, schema.rs:12) plus `<out>.report.json`. Report
  `params` record fixed/variable mods, `max_variable_mods`, and the charge range
  (peptidoforms.rs:275); `stats` is empty.

`peptidoforms.parquet` column schema (written at peptidoforms.rs:255-264):

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

### digest (digest.rs:184 `run`)

1. **Parse FASTA** with `read_fasta` (digest.rs:25).
2. **Per-protein cleavage** in `cleavage_sites` (digest.rs:52). Sites is seeded
   with `0`, then for each residue that is `K` or `R` a cut index `i+1` is pushed;
   Trypsin/P (`Enzyme::TrypsinP`, the default) always cuts, classic `Enzyme::Trypsin`
   suppresses the cut when the next residue is `P` (`before_p`, digest.rs:57). The
   sequence length is appended as a final site if not already present
   (digest.rs:67). Sites are strictly increasing, so peptides are contiguous
   half-open `[start, end)` spans.
3. **Peptide enumeration** in `digest_protein` (digest.rs:75). For every ordered
   pair of sites `(i, j)` with `i < j`, the number of missed cleavages is
   `j - i - 1`; the inner loop `break`s once that exceeds `cfg.missed_cleavages`
   (digest.rs:81), so at most `missed_cleavages + 1` fragments are joined. Length
   bounds `min_len`/`max_len` filter on residue count (digest.rs:86). A peptide
   is dropped entirely if any residue is not one of the 20 standard amino acids
   (`is_standard_residue`, digest.rs:90 -> constants.rs:54), which is how `X`, `B`,
   `Z`, `U`, and `*` get excluded (lowercase is already upper-cased in
   `read_fasta`, so it is not dropped here). After a peptide is accepted,
   `digest_protein` also applies **N-terminal methionine excision**
   (digest.rs:101-110): when `cfg.n_term_met_excision` is set (the default), a
   peptide anchored at protein position `0` whose protein begins with `M`
   additionally emits the Met-removed form, with `start` shifted to `1`. That
   excised form is re-checked against `min_len`/`max_len` and the standard-residue
   allowlist (digest.rs:104-106) before it is pushed, and excision keys on protein
   position 0 and the protein's first residue, not any interior `M`. The initiator
   methionine is cleaved in vivo for most proteins, so both forms belong in the
   search space; this mirrors DIA-NN `--met-excision`, and omitting it makes the
   database structurally miss those (biologically dominant) peptides.
4. **Dedup by stripped sequence** (digest.rs:189-209). Three structures: a
   `HashMap<String, Vec<String>>` mapping peptide to accessions, a `HashMap` from
   peptide to its first `(start, end)`, and an insertion-order `Vec<String>`
   named `order`. On a repeat occurrence only the accession list is extended (and
   only if the accession is not already present, digest.rs:200); the first
   occurrence records position and pushes the peptide into `order`. Iteration for
   output is over `order`, so **insertion order is the emission order** and is
   deterministic (a protein's peptides in sequence order, proteins in FASTA
   order). The borrow-then-clone pattern (digest.rs:199) is a deliberate
   allocation optimization, not a semantic detail.
5. **Row assembly and decoy minting** (digest.rs:211-275). A single monotonic
   `next_id` counter assigns ids. The decoy is resolved **before** the target row
   is written: for each target peptide in `order`, if the strategy is `Reverse` or
   `Scramble` (digest.rs:224-227), call `collision_safe_decoy` (digest.rs:228). If
   it returns `None` (no collision-free permutation within `MAX_DECOY_ATTEMPTS`),
   the whole pair is dropped, the target row is not written, and
   `dropped_target_decoy_pairs` is incremented (digest.rs:240-243). Otherwise emit
   the target row (`label = "target"`, `target_id = -1`, digest.rs:258-259) and
   then the decoy row immediately after with a fresh id, `label = "decoy"`,
   `target_id = tid` (the target's id), and the `DECOY_`-prefixed protein string
   (digest.rs:263-274). Ids therefore interleave target/decoy in pairs. A
   `used_decoys` set (digest.rs:216) tracks emitted decoy sequences for
   cross-decoy uniqueness, and `decoy_collision_retries` accumulates the retry
   count reported in `stats`.
6. **Write** the eight columns with `write_table` (digest.rs:286) and emit the
   `ArtifactReport` (digest.rs:312).

**Decoy generation.** `collision_safe_decoy` (digest.rs:137) is the wrapper the
run loop calls; it drives the lower-level transform `make_decoy` (digest.rs:101)
and guarantees the decoy differs from its target, matches no target sequence, and
is unique among emitted decoys. The configured transform is tried first
(`attempt == 0`); on any collision it retries with an independently reseeded
`Scramble` up to `MAX_DECOY_ATTEMPTS` (64, digest.rs:22), returning the sequence
and the retry count (digest.rs:144-156). If every attempt collides, it returns
`None` and the pair is dropped. Tests `collision_safe_decoy_avoids_targets_and_other_decoys`
(digest.rs:382) and `impossible_low_complexity_decoy_drops_pair` (digest.rs:403)
pin the retry and drop behavior.

`make_decoy` (digest.rs:101) itself performs one transform. Peptides shorter than
3 residues return `None` (digest.rs:104); through `collision_safe_decoy`'s `?`
this drops the pair rather than emitting an unpaired target. Both realized
strategies keep the C-terminal residue fixed and rewrite the interior `b[..n-1]`:
- `DecoyStrategy::Reverse` (digest.rs:108): reverse the first `n-1` residues, then
  re-append the original last residue. Reversing all-but-last keeps the enzyme's
  C-terminal `K`/`R` in place while moving the N-terminus. Test
  `reverse_decoy_keeps_cterm` (digest.rs:367) pins `PEPTIDER -> EDITPEPR`.
- `DecoyStrategy::Scramble` (digest.rs:114): a deterministic Fisher-Yates shuffle
  of the first `n-1` residues. The PRNG state is seeded per peptide as
  `rng_seed ^ fnv1a(pep)` (digest.rs:117) and advanced with `splitmix64`
  (digest.rs:159). The loop runs `i` from `len-2` down to `1` and swaps index `i`
  with a uniform `j` in `0..=i` (digest.rs:118-122), so every interior position
  including index 0 (the N-terminal residue) can move while the last residue is
  re-appended untouched. Test `scramble_is_deterministic` (digest.rs:374) pins
  reproducibility and the fixed C-terminus. `collision_safe_decoy` also mixes an
  `attempt`-indexed constant into this seed (digest.rs:145) so each retry is a
  distinct but deterministic scramble.
- `DecoyStrategy::DiannShift` and `DecoyStrategy::None` return `None`
  (digest.rs:126-127); `DiannShift` is unrealized here by design (the comment
  notes it would be a predict-frag fragment-shift, PLAN.md Section 11) and is
  additionally rejected by `Config::validate` (config.rs:1021) because it would
  yield zero decoys and an invalid FDR.

### peptidoforms (peptidoforms.rs:186 `run`)

1. **Validate rules up front** (`validated_rules`, peptidoforms.rs:125, called at
   peptidoforms.rs:188 before the table is read). This performs all config
   validation once, so a bad config fails fast:
   - charge range: `charge_min >= 1` and `charge_min <= charge_max`, else a hard
     `anyhow::bail!` (peptidoforms.rs:126-138);
   - fixed and variable mod names are resolved by `known_mods` (peptidoforms.rs:85),
     which rejects a `*` residue (wildcard/terminal mods are unimplemented,
     peptidoforms.rs:92-98) and looks each name up in `unimod_mass` (mass.rs:13,
     peptidoforms.rs:99). An unknown name is handled per `unknown_modification`:
     `Error` bails with `unknown <fixed|variable> modification '<name>' in config`
     (peptidoforms.rs:101-103); `Skip` warns and drops the rule
     (peptidoforms.rs:104-112);
   - stacked fixed mods on one residue are rejected (peptidoforms.rs:145-153);
   - exact-duplicate variable rules are deduplicated with a warning
     (peptidoforms.rs:156-166);
   - a residue carrying both a fixed and a variable mod is rejected
     (fixed-variable stacking, peptidoforms.rs:167-172).

   The `unimod_mass` allowlist (mass.rs:14-24) currently recognizes exactly eight
   names: `Carbamidomethyl`, `Oxidation`, `Acetyl`, `Phospho`, `Deamidated`,
   `Methyl`, `Dimethyl`, `Carbamyl`. Extending the set is a `mass.rs` edit (see
   "How to extend").
2. **Read** the digest table (peptidoforms.rs:189-194): columns `id`, `peptide`,
   `protein`, `label`, `target_id`.
3. **Per digest row** (peptidoforms.rs:201): compute `base_peptide_id` as
   `target_id` when it is `>= 0` (the row is a decoy) else the row's own `id`
   (peptidoforms.rs:203-207).
4. **Fixed mods** (peptidoforms.rs:210-217): for each residue index, if any
   validated fixed rule's `residue` char matches, record `(index, name)`. Fixed
   mods apply to every matching residue and are always present.
5. **Variable-mod candidate sites** (peptidoforms.rs:219-228): the same
   residue-char scan collects, per modified site, a `Vec` of the alternative mod
   names configured for that residue (`var_sites: Vec<(usize, Vec<&str>)>`). A
   single residue can therefore offer more than one variable alternative.
6. **Combination enumeration** (`variable_combos`, peptidoforms.rs:58): returns the
   empty subset plus every size-`k` subset of candidate **sites** for
   `k = 1..=max_variable_mods.min(n)`, using an in-place combinatorial index
   advance (peptidoforms.rs:69-79); for each selected set, `expand_site_choices`
   (peptidoforms.rs:34) enumerates every combination of the per-site alternatives,
   picking at most one alternative per site so two mods never stack on one residue.
   When each site has a single alternative the count is `1 + sum_{k=1..min(max,n)}
   C(n,k)`; test `combos_bounded` (peptidoforms.rs:302) pins `n=3, max=1 -> 4` and
   `max=2 -> 7`, and `same_site_alternatives_never_stack_and_are_deterministic`
   (peptidoforms.rs:329) pins the multi-alternative case.
7. **Emit** (peptidoforms.rs:231-250): for each combo, merge fixed mods and the
   combo, sort by position (peptidoforms.rs:234), build the ProForma-lite string
   with `proforma` (peptidoforms.rs:19), and emit one output row per charge in
   `[charge_min, charge_max]` (peptidoforms.rs:239). A `seen_forms` set
   (peptidoforms.rs:230, 236) skips any duplicate ProForma string within a
   peptide. Each row gets a fresh `id`, the source `peptide_id`, the computed
   `base_peptide_id`, and the copied `label`/`protein`.

`proforma` (peptidoforms.rs:19) walks the stripped sequence and, at each residue
index, appends `[<name>]` for the **first** mod found at that position
(`mods.iter().find`, peptidoforms.rs:23). Because validation now forbids two mods
on one residue (see gotchas), this first-match is a defensive fallback rather than
a live silent-drop path.

## Key types and functions

| Name | file:line | What it does |
|---|---|---|
| `digest::run` | digest.rs:184 | Stage A entry point; parse, digest, dedup, mint decoys, write `peptides.parquet` |
| `read_fasta` | digest.rs:25 | Minimal FASTA parser to `(accession, sequence)` pairs; upper-cases sequence |
| `cleavage_sites` | digest.rs:52 | Trypsin/P vs Trypsin cut-site indices for one sequence |
| `digest_protein` | digest.rs:75 | Enumerate length-bounded, missed-cleavage-bounded, standard-residue-only peptides, plus the initiator-Met-excised N-terminal form when `n_term_met_excision` is set (digest.rs:101-110) |
| `collision_safe_decoy` | digest.rs:137 | Wrap `make_decoy` with target/other-decoy collision checks and up to `MAX_DECOY_ATTEMPTS` reseeded-scramble retries; `None` drops the pair |
| `make_decoy` | digest.rs:101 | One transform: reverse or seeded-scramble decoy of the interior, C-term fixed; `None` if `len < 3` or strategy `None`/`DiannShift` |
| `splitmix64` | digest.rs:159 | Deterministic PRNG step for the scramble shuffle |
| `fnv1a` | digest.rs:167 | Per-peptide hash mixed into the scramble seed |
| `DigestParams` | digest.rs:176 | `fasta`, `out`, `cfg: &DigestConfig`, `rng_seed`, `config_hash` |
| `peptidoforms::run` | peptidoforms.rs:186 | Stage A2 entry point; validate rules, enumerate mods+charges, write `peptidoforms.parquet` |
| `validated_rules` | peptidoforms.rs:125 | Validate charges and fixed/variable rules (stacking, overlap, dedup, unknown-mod policy) before expansion |
| `known_mods` | peptidoforms.rs:85 | Resolve one mod list against `unimod_mass` under `UnknownModPolicy`; reject `*` residue |
| `proforma` | peptidoforms.rs:19 | Build ProForma-lite string from stripped peptide + `(position, name)` mods |
| `variable_combos` | peptidoforms.rs:58 | Enumerate modified-site subsets up to `max_var`, expanding per-site alternatives |
| `expand_site_choices` | peptidoforms.rs:34 | Enumerate the per-site alternative assignments for one selected set of sites |
| `PeptidoformsParams` | peptidoforms.rs:179 | `peptides`, `out`, `cfg: &PeptidoformsConfig`, `config_hash` |
| `unimod_mass` | mass.rs:13 | Name -> monoisotopic delta allowlist (the set of mods that validate) |
| `is_standard_residue` | constants.rs:54 | The 20-residue allowlist gating digest output |

## Configuration

`digest` reads `DigestConfig` (config.rs:183). `peptidoforms` reads
`PeptidoformsConfig` (config.rs:204). Both structs use `#[serde(default,
deny_unknown_fields)]`, so an unknown key is a load error and any omitted field
takes the default below. `rng_seed` (config.rs:974, default `0`) is a top-level
`Config` field, passed to digest as `DigestParams.rng_seed` (main.rs:422,
run.rs:136).

| Field | Section | Default | Effect |
|---|---|---|---|
| `enzyme` | digest | `TrypsinP` | `trypsin_p` cuts after K/R including before P; `trypsin` suppresses the cut before P |
| `missed_cleavages` | digest | `2` | Max missed cleavages; joins at most `missed_cleavages + 1` tryptic fragments |
| `min_len` | digest | `5` | Minimum peptide length (residues) |
| `max_len` | digest | `50` | Maximum peptide length (residues) |
| `decoy.strategy` | digest | `reverse` | `reverse` / `scramble` realized; `diann_shift` and `none` produce no decoy (`diann_shift` rejected by `validate`) |
| `n_term_met_excision` | digest | `true` | When a protein begins with `M`, also emit the initiator-Met-removed form of its N-terminal peptides (re-checked against `min_len`/`max_len` and the residue allowlist); mirrors DIA-NN `--met-excision`. Field config.rs:194, default config.rs:204 |
| `fixed_mods` | peptidoforms | `[{C: Carbamidomethyl}]` | Applied to every matching residue |
| `variable_mods` | peptidoforms | `[{M: Oxidation}]` | Enumerated as optional subsets |
| `max_variable_mods` | peptidoforms | `1` | Max simultaneous variable mods per peptidoform |
| `charge_min` | peptidoforms | `2` | Lowest precursor charge emitted |
| `charge_max` | peptidoforms | `3` | Highest precursor charge emitted |
| `unknown_modification` | peptidoforms | `error` | `error` (default) hard-fails on an unknown mod name; `skip` warns and drops the rule. Both branches are wired (see gotchas) |

`rng_seed` is `0` by default (config.rs:991) and only affects the `scramble`
decoy strategy; `reverse` is independent of the seed.

Config was recently pruned of dead fields in this area. `ResidueMod`
(config.rs:235) is only `{ residue: char, name: String }`; there is no
position/terminal specifier field. Unlike the enclosing `DigestConfig` /
`PeptidoformsConfig`, `ResidueMod` carries `#[serde(deny_unknown_fields)]` but
**not** `#[serde(default)]` (config.rs:233-234), so both `residue` and `name` are
mandatory in every mod entry; an entry missing either key is a load error rather
than a defaulted value. The doc comment on `residue` (config.rs:236) still
mentions `*` for "any / terminal handled separately in MVP", but that handling
was never implemented; the current code instead rejects a `*` residue with a hard
error at validation (see gotchas). There is no `DecoySource` field on
`DigestConfig` (the "dead/unwired config surface" list in CLAUDE.md refers to
types elsewhere, not here). `DecoyConfig` (config.rs:172) carries only
`strategy`.

## Invariants, determinism, gotchas

- **Determinism.** Emission order is FASTA order then in-protein sequence order,
  preserved by the `order` vector (digest.rs:206), not by HashMap iteration. Ids
  are a single monotonic counter. Scramble is fully deterministic for a fixed
  `rng_seed` because the PRNG is seeded per peptide from the sequence
  (digest.rs:117); `collision_safe_decoy`'s retry reseeding is `attempt`-indexed
  (digest.rs:145), so which pairs drop is deterministic too. No floats are summed
  in either stage, so the numeric-order determinism concerns from other stages do
  not apply.
- **Target/decoy pairing.** A decoy's `target_id` points at its target's `id` and
  the pair is emitted adjacently; targets carry `target_id = -1`. Downstream FDR
  keeps the label in competition keys (see CLAUDE.md FDR / compete), so this
  pairing must stay intact. A target whose decoy cannot be built collision-free is
  dropped together with its would-be decoy (digest.rs:240-243), so the emitted
  target and decoy populations stay one-to-one. `peptidoforms` propagates the link
  via `base_peptide_id` (peptidoforms.rs:203).
- **Residue allowlist.** Peptides containing any non-standard residue are dropped
  wholesale at digest time (digest.rs:90); they never reach peptidoforms. This is
  the only place ambiguity codes (`X`/`B`/`Z`/`U`) are handled. Lowercase is not
  an ambiguity case here: `read_fasta` upper-cases sequence lines first
  (digest.rs:41), so a lowercase-but-standard residue survives.
- **N-terminal methionine excision is on by default.** For a protein-N-terminal
  peptide whose protein begins with `M`, `digest_protein` emits both the
  Met-retained and the Met-removed form (digest.rs:101-110), each subject to the
  same length and standard-residue filters. Because `DigestConfig` is
  `#[serde(default, deny_unknown_fields)]` (config.rs:182), a config predating the
  `n_term_met_excision` field still loads and silently gains the behavior; set
  `n_term_met_excision = false` to reproduce a Met-retained-only digest. Excision
  keys on protein position 0 and the first residue, so an interior `M` is never
  excised (test `met_excision_only_at_protein_n_terminus`). This is
  standard-proteomics-correct, but per repository policy any digest-output default
  change is entrapment plus second-dataset gated before it is trusted as a default.
- **Collision-checked decoys.** `collision_safe_decoy` (digest.rs:137) does
  guarantee the decoy differs from its target, equals no target sequence, and is
  unique among emitted decoys, retrying with reseeded scrambles up to
  `MAX_DECOY_ATTEMPTS` (64). When no collision-free permutation exists (a
  homopolymer or otherwise low-complexity sequence) the whole pair is dropped and
  counted in `dropped_target_decoy_pairs`, so a decoy is never silently equal to
  some target. Peptides with `len < 3` return `None` from `make_decoy`
  (digest.rs:104) and are likewise dropped as a pair; with the default
  `min_len = 5` this only bites if a caller lowers `min_len`.
- **No terminal modifications, and `*` is rejected.** `ResidueMod` matches on an
  exact residue char (peptidoforms.rs:211, peptidoforms.rs:220). There is no
  N-term/C-term matching, so protein/peptide N-terminal acetylation and similar
  terminal mods cannot be enumerated. A `*` residue is no longer silently
  unmatched: `known_mods` hard-fails on it up front (peptidoforms.rs:92-98,
  message `unsupported '*' residue ...`). The `mass.rs` parser does model terminal
  mod deltas, but Stage A2 never produces terminal-mod ProForma strings.
- **Two mods on one residue are prevented at validation.** `validated_rules`
  rejects two fixed mods on the same residue (peptidoforms.rs:145-153) and a
  fixed + variable mod on the same residue (peptidoforms.rs:167-172), and
  `variable_combos`/`expand_site_choices` pick at most one alternative per site so
  variable alternatives never stack (test
  `same_site_alternatives_never_stack_and_are_deterministic`, peptidoforms.rs:329).
  Under any config that loads, the merged `mods` list therefore never holds two
  entries at one position, so `proforma`'s first-match `find` (peptidoforms.rs:23)
  cannot silently drop a second mod. This supersedes the older "second mod at same
  position dropped" limitation.
- **`unknown_modification` policy is wired.** The config field takes `error` /
  `skip` (config.rs:212, `UnknownModPolicy`, config.rs:244) and `known_mods`
  honors it via `validated_rules` (peptidoforms.rs:140-141): `Error` bails on an
  unknown mod name (peptidoforms.rs:101-103); `Skip` logs a warning and drops the
  offending rule (peptidoforms.rs:104-112). Test
  `skip_policy_removes_unknown_modification` (peptidoforms.rs:350) pins the `skip`
  path.
- **`config_hash` param is unused.** Both `DigestParams.config_hash`
  (digest.rs:181) and `PeptidoformsParams.config_hash` (peptidoforms.rs:183) are
  threaded from the CLI/orchestrator but never read inside either `run`. The
  content hash written to the report comes from `blake3_file` over the output
  Parquet (digest.rs:318, peptidoforms.rs:274), not from `config_hash`.
- **`decoy_strategy` string is derived by Debug-formatting the enum**
  (`format!("{:?}", ...).to_lowercase()`, digest.rs:260). It is a display string,
  not a parseable round-trip; changing enum variant names changes the column
  value.
- **Charge range is validated.** `validated_rules` rejects `charge_min < 1` and an
  empty range `charge_min > charge_max` with a hard error before any row is
  processed (peptidoforms.rs:126-138); the `charge_min..=charge_max` emission
  (peptidoforms.rs:239) is therefore always non-empty. Test
  `invalid_charges_and_wildcard_modifications_are_rejected` (peptidoforms.rs:381)
  pins the rejection.
- **Library-input mode bypasses both stages.** When a prebuilt library is passed,
  `run` takes the library-input branch (run.rs:98) and never calls
  `digest`/`peptidoforms`; the FASTA-build branch is the `_ =>` arm at run.rs:127.
  Do not assume `peptides.parquet` / `peptidoforms.parquet` exist for every run.

## How to extend / modify

- **New enzyme.** Add a variant to `Enzyme` (config.rs:46) and a match arm in
  `cleavage_sites` (digest.rs:58). Keep the site list strictly increasing and
  terminated at `seq.len()`; the rest of `digest_protein` is enzyme-agnostic.
  Semi-tryptic search would require changing how spans are enumerated in
  `digest_protein` (digest.rs:75), not just the cut list.
- **New decoy scheme.** Add a `DecoyStrategy` variant (config.rs:17) and a match
  arm in `make_decoy` (digest.rs:107). Preserve the C-terminal residue and
  determinism (seed the PRNG from the sequence as scramble does, digest.rs:117).
  The target/other-decoy collision guard is applied for free by
  `collision_safe_decoy` (digest.rs:137) around any transform, but only its
  `attempt == 0` path uses the new strategy; every retry falls back to `Scramble`,
  so add the new variant to the run-loop `matches!` (digest.rs:224-227) if it
  should be attempted at all. Realizing `DiannShift` means removing the `validate`
  rejection (config.rs:1021) and implementing a fragment-m/z shift at predict-frag,
  not a sequence rewrite here.
- **New modification.** Add the UniMod name and monoisotopic delta to
  `unimod_mass` (mass.rs:13); it is the single allowlist both the Stage A2
  validation (`known_mods`, peptidoforms.rs:99) and the mass model consult. Then
  reference it by name in `fixed_mods`/`variable_mods`.
- **Support a second mod per position or terminal mods.** Change `proforma`
  (peptidoforms.rs:19) to emit all mods at a position (e.g. concatenate multiple
  `[...]` groups) instead of `find`-ing the first, relax the stacking/overlap and
  `*` checks in `validated_rules` (peptidoforms.rs:92-98, peptidoforms.rs:145-172),
  and extend `ResidueMod` (config.rs:235) plus the residue-matching loops
  (peptidoforms.rs:210-228) with a position/terminal specifier. The mass model in
  `mass.rs` already carries `n_term_mod` fields to build on.
- **Schema changes.** Adding or reordering `peptides.parquet` /
  `peptidoforms.parquet` columns is a schema change; bump the version in
  `artifact::PEPTIDES` / `artifact::PEPTIDOFORMS` (schema.rs:11-12) and update the
  downstream readers (`Table::read` callers in predict-frag and later stages).
- **Testing.** Existing unit tests live at the bottom of each file
  (digest.rs:360, peptidoforms.rs:291). digest: `trypsin_p_cleaves_after_kr`
  (digest.rs:365) pins Trypsin/P cut sites; `met_excision_emits_both_n_term_forms`
  (digest.rs:384) pins that a protein-N-terminal `M` peptide yields both the
  Met-retained and the Met-excised form with excision on and only the retained
  form with it off, and `met_excision_only_at_protein_n_terminus` (digest.rs:418)
  pins that an interior `M` is not excised; `reverse_decoy_keeps_cterm`
  (digest.rs:439) and `scramble_is_deterministic` (digest.rs:446) pin the fixed
  C-terminus plus scramble reproducibility;
  `collision_safe_decoy_avoids_targets_and_other_decoys` (digest.rs:382) pins the
  retry-on-collision behavior; and `impossible_low_complexity_decoy_drops_pair`
  (digest.rs:403) pins the pair-drop when no collision-free decoy exists.
  peptidoforms: `proforma_places_mods` (peptidoforms.rs:296) pins bracket
  placement; `combos_bounded` (peptidoforms.rs:302) and
  `default_rules_and_form_order_are_preserved` (peptidoforms.rs:315) pin the subset
  counts and legacy order; `same_site_alternatives_never_stack_and_are_deterministic`
  (peptidoforms.rs:329) pins per-site alternative expansion;
  `skip_policy_removes_unknown_modification` (peptidoforms.rs:350) pins the `skip`
  policy; `fixed_stacking_and_fixed_variable_overlap_are_rejected`
  (peptidoforms.rs:362) and `invalid_charges_and_wildcard_modifications_are_rejected`
  (peptidoforms.rs:381) pin the validation rejections. There is no stage-level test
  that round-trips through Parquet or exercises `run` end to end (CLAUDE.md "test
  gaps"); add one when changing the output schema.
