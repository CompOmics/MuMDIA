//! Fragment-index matchers (docs/06_predict_frag_index_matchers.md). `binning`
//! holds the log-space bin geometry; `fragindex` is the CSR inverted index plus
//! an epoch-stamped accumulator; `naive` is the band-join reference for the
//! equivalence gate. The concrete matcher for each stage is selected by the
//! `MatcherKind` config strategy enum
//! (in `mumdia-core::config`); `Bucketed` keeps the existing `index::Library`
//! path untouched.

pub mod binning;
pub mod fragindex;
pub mod naive;
