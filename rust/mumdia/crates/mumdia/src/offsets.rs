//! Row offsets of a CSR layout, 32-bit while the rows fit and 64-bit beyond.
//!
//! The library's per-candidate fragment offsets ([`crate::index::Library::frag_offsets`])
//! and the fragment index's per-bin posting offsets were `u32`, which capped a library at
//! 2^32 - 1 fragment rows. An imported library for an unspecific (HLA class I) digest of
//! the human proteome passes that: 4.7 billion rows on one immunopeptidomics library. A
//! library that fits keeps the 4-byte layout, and so its memory; only a larger one is held
//! as `u64`. The variant is fixed when the offsets are built, so the branch in [`get`] goes
//! the same way on every access. It is not free in the tightest loop, the windowed fragment
//! probe, which therefore matches the width once per peak instead
//! (`FragIndex::probe_peak_win_binned`). End to end on a library that fits, every stage
//! output was byte-identical to 32-bit offsets at an unchanged wall time and peak memory.
//!
//! [`get`]: Offsets::get

/// CSR row offsets: `offsets[i]..offsets[i + 1]` are the rows of entry `i`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Offsets {
    /// Every offset fits `u32`.
    Narrow(Vec<u32>),
    /// At least one offset does not.
    Wide(Vec<u64>),
}

impl Default for Offsets {
    fn default() -> Self {
        Offsets::Narrow(Vec::new())
    }
}

/// The largest total kept narrow: `u32::MAX`, or 0 inside [`force_wide`] (tests only), so
/// the 64-bit paths are exercised on fixtures far below 2^32 rows.
fn narrow_limit() -> u64 {
    #[cfg(test)]
    if FORCE_WIDE.with(|f| f.get()) {
        return 0;
    }
    u32::MAX as u64
}

#[cfg(test)]
thread_local! {
    static FORCE_WIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with every [`Offsets`] built on this thread held as `u64`, whatever its size.
#[cfg(test)]
pub fn force_wide<T>(f: impl FnOnce() -> T) -> T {
    FORCE_WIDE.with(|w| w.set(true));
    let out = f();
    FORCE_WIDE.with(|w| w.set(false));
    out
}

impl Offsets {
    /// The prefix sums of `counts` with a leading zero (`counts[0]` is ignored and must be
    /// 0, as in the loader's `counts[c + 1] += 1` layout), narrow when the total fits `u32`.
    /// On the narrow path the sums are formed in place, so a library that fits allocates
    /// nothing beyond the counts it already holds.
    pub fn from_counts(mut counts: Vec<u32>) -> Offsets {
        let total: u64 = counts.iter().map(|&c| c as u64).sum();
        if total <= narrow_limit() && narrow_limit() > 0 {
            for i in 1..counts.len() {
                counts[i] += counts[i - 1];
            }
            Offsets::Narrow(counts)
        } else {
            let mut acc = 0u64;
            Offsets::Wide(
                counts
                    .into_iter()
                    .map(|c| {
                        acc += c as u64;
                        acc
                    })
                    .collect(),
            )
        }
    }

    /// Offsets given directly, narrow when the last fits `u32`. They must not decrease.
    pub fn from_values(values: Vec<u64>) -> Offsets {
        if narrow_limit() > 0 && values.last().is_none_or(|&v| v <= narrow_limit()) {
            Offsets::Narrow(values.into_iter().map(|v| v as u32).collect())
        } else {
            Offsets::Wide(values)
        }
    }

    /// Offset `i`.
    #[inline]
    pub fn get(&self, i: usize) -> usize {
        match self {
            Offsets::Narrow(v) => v[i] as usize,
            Offsets::Wide(v) => v[i] as usize,
        }
    }

    /// The rows of entry `i`.
    #[inline]
    pub fn range(&self, i: usize) -> std::ops::Range<usize> {
        self.get(i)..self.get(i + 1)
    }

    /// Number of offsets (entries + 1 for a built CSR array).
    pub fn len(&self) -> usize {
        match self {
            Offsets::Narrow(v) => v.len(),
            Offsets::Wide(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The last offset (the row total), 0 when empty.
    pub fn last(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            self.get(self.len() - 1)
        }
    }

    /// The first index whose offset is not below `target` (`partition_point(|o| o < target)`).
    pub fn partition_point(&self, target: usize) -> usize {
        match self {
            Offsets::Narrow(v) => v.partition_point(|&o| (o as usize) < target),
            Offsets::Wide(v) => v.partition_point(|&o| (o as usize) < target),
        }
    }

    /// Every offset, in order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).map(|i| self.get(i))
    }

    /// Whether the offsets are held as `u64`.
    pub fn is_wide(&self) -> bool {
        matches!(self, Offsets::Wide(_))
    }

    /// Heap bytes, for the memory log.
    pub fn bytes(&self) -> usize {
        match self {
            Offsets::Narrow(v) => v.capacity() * 4,
            Offsets::Wide(v) => v.capacity() * 8,
        }
    }

    /// A cursor per entry, starting at each entry's first row, for a scatter that fills the
    /// rows in order ([`Cursors::next`]). Same width as the offsets.
    pub fn cursors(&self) -> Cursors {
        match self {
            Offsets::Narrow(v) => Cursors::Narrow(v.clone()),
            Offsets::Wide(v) => Cursors::Wide(v.clone()),
        }
    }
}

impl PartialEq<Vec<u32>> for Offsets {
    fn eq(&self, other: &Vec<u32>) -> bool {
        self.len() == other.len()
            && other
                .iter()
                .enumerate()
                .all(|(i, &o)| self.get(i) == o as usize)
    }
}

/// Per-entry write positions of a scatter, as wide as the [`Offsets`] they came from.
pub enum Cursors {
    Narrow(Vec<u32>),
    Wide(Vec<u64>),
}

impl Cursors {
    /// The next free row of entry `i`, advancing it.
    #[inline]
    pub fn next(&mut self, i: usize) -> usize {
        match self {
            Cursors::Narrow(v) => {
                let p = v[i];
                v[i] += 1;
                p as usize
            }
            Cursors::Wide(v) => {
                let p = v[i];
                v[i] += 1;
                p as usize
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_that_fit_stay_narrow_and_larger_ones_go_wide() {
        let o = Offsets::from_counts(vec![0, 1, 3, 0, 2]);
        assert!(!o.is_wide());
        assert_eq!(o, vec![0u32, 1, 4, 4, 6]);
        assert_eq!(o.range(1), 1..4);
        assert_eq!(o.last(), 6);
        assert_eq!(o.partition_point(4), 2);

        // Two entries of 3e9 rows: 6e9 does not fit u32.
        let w = Offsets::from_counts(vec![0, 3_000_000_000, 3_000_000_000]);
        assert!(w.is_wide());
        assert_eq!(w.get(2), 6_000_000_000);
        assert_eq!(w.range(1), 3_000_000_000..6_000_000_000);
        assert_eq!(w.partition_point(3_000_000_001), 2);
    }

    #[test]
    fn values_and_cursors_keep_the_width() {
        assert!(!Offsets::from_values(vec![0, 5, u32::MAX as u64]).is_wide());
        let w = Offsets::from_values(vec![0, 5, u32::MAX as u64 + 1]);
        assert!(w.is_wide());
        let mut c = w.cursors();
        assert_eq!(c.next(1), 5);
        assert_eq!(c.next(1), 6);
        let mut n = Offsets::from_counts(vec![0, 2, 2]).cursors();
        assert_eq!((n.next(0), n.next(0), n.next(1)), (0, 1, 2));
    }
}
