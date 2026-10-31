//! The localisation likelihood (docs/36_ptm_localisation.md).
//!
//! Each site-determining ion is observed present or absent. Under a hypothesis that predicts
//! it, it is present with the detection probability `d` of its class; under one that does
//! not, with the false-presence probability `f` of its class. `d` is the share of the
//! peak's common ions that are present and `f` the share of its impossible variants that
//! are present, each pulled towards the run-wide share by `prior_strength` pseudo-counts.
//! The log-likelihood of a hypothesis is the sum over the site-determining ions; the
//! posterior is its softmax (divided by `temperature` first), with a flat prior over the
//! hypotheses. A site's probability is the posterior mass of the hypotheses that use it.

use super::ions::N_CLASSES;

/// Present and total counts per ion class.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Counts {
    pub present: [u64; N_CLASSES],
    pub total: [u64; N_CLASSES],
}

impl Counts {
    pub fn add(&mut self, class: usize, present: bool) {
        self.total[class] += 1;
        self.present[class] += u64::from(present);
    }

    pub fn merge(&mut self, o: &Counts) {
        for c in 0..N_CLASSES {
            self.present[c] += o.present[c];
            self.total[c] += o.total[c];
        }
    }

    /// The run-wide rate per class, with one pseudo-count each way so it is never 0 or 1.
    pub fn laplace(&self) -> [f64; N_CLASSES] {
        let mut r = [0.0; N_CLASSES];
        for (c, v) in r.iter_mut().enumerate() {
            *v = (self.present[c] as f64 + 1.0) / (self.total[c] as f64 + 2.0);
        }
        r
    }

    /// This peak's rate per class, pulled towards `global` by `strength` pseudo-counts.
    pub fn shrunk(&self, global: &[f64; N_CLASSES], strength: f64) -> [f64; N_CLASSES] {
        let mut r = [0.0; N_CLASSES];
        for (c, v) in r.iter_mut().enumerate() {
            let den = self.total[c] as f64 + strength;
            *v = if den > 0.0 {
                (self.present[c] as f64 + strength * global[c]) / den
            } else {
                global[c]
            };
        }
        r
    }
}

/// One site-determining ion: its class, whether it was seen, and which hypotheses predict it.
pub(crate) struct Observed<'a> {
    pub class: usize,
    pub present: bool,
    pub member: &'a [bool],
}

/// Log-likelihood of each of `n_hyp` hypotheses.
pub(crate) fn loglik(
    n_hyp: usize,
    sdi: &[Observed],
    d: &[f64; N_CLASSES],
    f: &[f64; N_CLASSES],
) -> Vec<f64> {
    let mut ll = vec![0.0; n_hyp];
    for o in sdi {
        let c = o.class;
        let (lt, lf) = if o.present {
            (d[c].ln(), f[c].ln())
        } else {
            ((1.0 - d[c]).ln(), (1.0 - f[c]).ln())
        };
        for (h, v) in ll.iter_mut().enumerate() {
            *v += if o.member[h] { lt } else { lf };
        }
    }
    ll
}

/// The softmax of `ll / temperature`.
pub(crate) fn posterior(ll: &[f64], temperature: f64) -> Vec<f64> {
    let m = ll.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = ll.iter().map(|&l| ((l - m) / temperature).exp()).collect();
    let z: f64 = w.iter().sum();
    w.iter().map(|&x| x / z).collect()
}

/// Per-position site probability: the posterior mass of the hypotheses using each position.
pub(crate) fn site_probabilities(n: usize, hyps: &[Vec<usize>], post: &[f64]) -> Vec<f64> {
    let mut sp = vec![0.0; n];
    for (h, sites) in hyps.iter().enumerate() {
        for &i in sites {
            sp[i] += post[h];
        }
    }
    sp
}

/// The best hypothesis: highest posterior, ties to the lexicographically first placement,
/// so the answer does not depend on the order the hypotheses were listed in.
pub(crate) fn best(hyps: &[Vec<usize>], post: &[f64]) -> usize {
    let mut b = 0;
    for h in 1..hyps.len() {
        if post[h] > post[b] || (post[h] == post[b] && hyps[h] < hyps[b]) {
            b = h;
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rates() -> ([f64; N_CLASSES], [f64; N_CLASSES]) {
        ([0.4; N_CLASSES], [0.05; N_CLASSES])
    }

    /// The posterior of a hypothesis does not depend on the order hypotheses are listed in.
    #[test]
    fn posterior_is_invariant_to_hypothesis_order() {
        let (d, f) = rates();
        let m1 = [true, false, false];
        let m2 = [false, true, false];
        let m3 = [false, true, true];
        let sdi = vec![
            Observed {
                class: 2,
                present: true,
                member: &m1,
            },
            Observed {
                class: 2,
                present: false,
                member: &m2,
            },
            Observed {
                class: 0,
                present: true,
                member: &m3,
            },
        ];
        let hyps = vec![vec![1], vec![4], vec![6]];
        let post = posterior(&loglik(3, &sdi, &d, &f), 1.0);
        // the same hypotheses listed in reverse, membership permuted to match
        let r1 = [false, false, true];
        let r2 = [false, true, false];
        let r3 = [true, true, false];
        let rsdi = vec![
            Observed {
                class: 2,
                present: true,
                member: &r1,
            },
            Observed {
                class: 2,
                present: false,
                member: &r2,
            },
            Observed {
                class: 0,
                present: true,
                member: &r3,
            },
        ];
        let rhyps = vec![vec![6], vec![4], vec![1]];
        let rpost = posterior(&loglik(3, &rsdi, &d, &f), 1.0);
        for h in 0..3 {
            assert!((post[h] - rpost[2 - h]).abs() < 1e-15);
        }
        assert_eq!(hyps[best(&hyps, &post)], rhyps[best(&rhyps, &rpost)]);
        let sp = site_probabilities(8, &hyps, &post);
        let rsp = site_probabilities(8, &rhyps, &rpost);
        for i in 0..8 {
            assert!((sp[i] - rsp[i]).abs() < 1e-15);
        }
        assert!((sp.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    /// With no site-determining evidence every hypothesis is equally likely, and a tie goes
    /// to the first placement whatever the listing order.
    #[test]
    fn no_evidence_is_a_flat_posterior() {
        let (d, f) = rates();
        let post = posterior(&loglik(4, &[], &d, &f), 1.0);
        assert!(post.iter().all(|&p| (p - 0.25).abs() < 1e-12));
        let hyps = vec![vec![5], vec![2], vec![9], vec![3]];
        assert_eq!(best(&hyps, &post), 1);
    }

    /// A present ion favours the hypotheses predicting it; temperature flattens.
    #[test]
    fn evidence_and_temperature() {
        let (d, f) = rates();
        let m = [true, false];
        let sdi = vec![Observed {
            class: 2,
            present: true,
            member: &m,
        }];
        let ll = loglik(2, &sdi, &d, &f);
        let p1 = posterior(&ll, 1.0);
        assert!((p1[0] - 0.4 / (0.4 + 0.05)).abs() < 1e-12);
        let p2 = posterior(&ll, 2.0);
        assert!(p2[0] < p1[0] && p2[0] > 0.5);
    }

    #[test]
    fn rates_are_shrunk_towards_the_run() {
        let mut c = Counts::default();
        c.add(0, true);
        c.add(0, false);
        let g = [0.1; N_CLASSES];
        let r = c.shrunk(&g, 8.0);
        assert!((r[0] - (1.0 + 0.8) / 10.0).abs() < 1e-12);
        assert!((r[1] - 0.1).abs() < 1e-12);
        assert!((c.laplace()[0] - 0.5).abs() < 1e-12);
        assert!((c.laplace()[3] - 0.5).abs() < 1e-12);
    }
}
