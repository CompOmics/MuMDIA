//! Cutting the accepted isomers of an isomer group and charge into chromatographic peaks
//! (docs/36_ptm_localisation.md, section 2, step 2). The isomers of one peak share one
//! localisation.

use mumdia_core::config::LocalisePeakRule;

/// What the cut needs of a row.
pub(crate) struct PeakRow {
    pub apex: f64,
    /// The row's elution bounds, when the scored table has them.
    pub bounds: Option<(f64, f64)>,
}

/// Cut `order` (row indices of one isomer group and charge, ascending in apex) into peaks.
///
/// - `ApexTolerance`: a row joins the current peak when its apex lies within `tol_s` of the
///   peak's first apex.
/// - `ElutionBounds`: a row joins the current peak when its apex lies inside the elution
///   bounds of the row before it in apex order, or that row's apex inside its bounds, and
///   `valley(previous, row)` finds no valley between the two. A pair without bounds falls
///   back to `tol_s` between the two apexes.
pub(crate) fn cut(
    rows: &[PeakRow],
    order: &[usize],
    rule: LocalisePeakRule,
    tol_s: f64,
    valley: &mut dyn FnMut(usize, usize) -> bool,
) -> Vec<Vec<usize>> {
    let mut peaks: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for &r in order {
        let joins = match (cur.first(), cur.last()) {
            (Some(&first), Some(&prev)) => match rule {
                LocalisePeakRule::ApexTolerance => rows[r].apex - rows[first].apex <= tol_s,
                LocalisePeakRule::ElutionBounds => {
                    let (a, b) = (&rows[prev], &rows[r]);
                    let overlap = match (a.bounds, b.bounds) {
                        (Some((alo, ahi)), Some((blo, bhi))) => {
                            (alo <= b.apex && b.apex <= ahi) || (blo <= a.apex && a.apex <= bhi)
                        }
                        _ => b.apex - a.apex <= tol_s,
                    };
                    overlap && !valley(prev, r)
                }
            },
            _ => true,
        };
        if !joins {
            peaks.push(std::mem::take(&mut cur));
        }
        cur.push(r);
    }
    if !cur.is_empty() {
        peaks.push(cur);
    }
    peaks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(apex: f64, lo: f64, hi: f64) -> PeakRow {
        PeakRow {
            apex,
            bounds: Some((lo, hi)),
        }
    }

    #[test]
    fn the_rules_cut_where_they_should() {
        // a broad peak split 9 s apart, then an isomer 60 s later
        let rows = vec![
            row(100.0, 90.0, 120.0),
            row(109.0, 95.0, 125.0),
            row(170.0, 160.0, 180.0),
        ];
        let order = [0, 1, 2];
        let none = &mut |_: usize, _: usize| false;
        assert_eq!(
            cut(&rows, &order, LocalisePeakRule::ApexTolerance, 5.0, none),
            vec![vec![0], vec![1], vec![2]]
        );
        assert_eq!(
            cut(&rows, &order, LocalisePeakRule::ElutionBounds, 5.0, none),
            vec![vec![0, 1], vec![2]]
        );
        // a valley between the first two separates them
        let dip = &mut |a: usize, b: usize| (a, b) == (0, 1);
        assert_eq!(
            cut(&rows, &order, LocalisePeakRule::ElutionBounds, 5.0, dip),
            vec![vec![0], vec![1], vec![2]]
        );
        // one apex inside the other's bounds is enough
        let one_way = vec![row(100.0, 99.0, 101.0), row(104.0, 95.0, 110.0)];
        assert_eq!(
            cut(
                &one_way,
                &[0, 1],
                LocalisePeakRule::ElutionBounds,
                1.0,
                none
            ),
            vec![vec![0, 1]]
        );
        // without bounds the tolerance decides
        let bare = vec![
            PeakRow {
                apex: 100.0,
                bounds: None,
            },
            PeakRow {
                apex: 104.0,
                bounds: None,
            },
        ];
        assert_eq!(
            cut(&bare, &[0, 1], LocalisePeakRule::ElutionBounds, 5.0, none),
            vec![vec![0, 1]]
        );
        assert_eq!(
            cut(&bare, &[0, 1], LocalisePeakRule::ElutionBounds, 3.0, none),
            vec![vec![0], vec![1]]
        );
    }
}
