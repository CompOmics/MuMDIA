//! Cleavage rules of the in-silico digest (`digest.enzyme`, `digest.cleavage_rule`,
//! docs/05_digest_peptidoforms.md).
//!
//! A rule is written in the residue-pair syntax of DIA-NN's `--cut`, so the engine's digest
//! and the desktop application's DIA-NN library builder read one description: pairs
//! (residue before the cut, residue after it) separated by commas, `*` for any residue, a
//! leading `!` for a pair that is never cut. A bond is cut when some positive pair matches
//! and no `!` pair does. `K*,R*,!*P` is trypsin, `*D` Asp-N, `**` every bond.
//!
//! The named presets live in `enzymes.json`, which the desktop application includes too;
//! both sides are tested against its `test_vectors`.

use std::collections::BTreeMap;
use std::sync::OnceLock;

/// The preset table, shared with the desktop application.
pub const ENZYMES_JSON: &str = include_str!("enzymes.json");

/// One side of a pair: a residue, or any residue (`*`).
type Side = Option<u8>;

/// A parsed cleavage rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CleavageRule {
    cut: Vec<(Side, Side)>,
    never: Vec<(Side, Side)>,
}

fn side(c: u8) -> Result<Side, String> {
    match c {
        b'*' => Ok(None),
        b'A'..=b'Z' => Ok(Some(c)),
        _ => Err(format!("'{}' is not a residue letter or '*'", c as char)),
    }
}

fn matches(p: &(Side, Side), before: u8, after: u8) -> bool {
    p.0.is_none_or(|b| b == before) && p.1.is_none_or(|a| a == after)
}

impl CleavageRule {
    /// Parse `K*,R*,!*P`-style text. Whitespace around a pair is ignored; residue letters
    /// are upper-cased.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut rule = CleavageRule {
            cut: Vec::new(),
            never: Vec::new(),
        };
        for raw in text.split(',') {
            let tok = raw.trim().to_ascii_uppercase();
            if tok.is_empty() {
                continue;
            }
            let (neg, pair) = match tok.strip_prefix('!') {
                Some(rest) => (true, rest.to_string()),
                None => (false, tok.clone()),
            };
            let b = pair.as_bytes();
            if b.len() != 2 {
                return Err(format!(
                    "cleavage rule pair '{tok}' must be two characters (residue before and \
                     after the cut, '*' for any), optionally preceded by '!'"
                ));
            }
            let p = (side(b[0])?, side(b[1])?);
            if neg {
                rule.never.push(p);
            } else {
                rule.cut.push(p);
            }
        }
        if rule.cut.is_empty() {
            return Err(format!(
                "cleavage rule '{text}' has no pair that cuts (only '!' pairs or nothing)"
            ));
        }
        Ok(rule)
    }

    /// Whether the bond between `before` and `after` is cut.
    pub fn cuts(&self, before: u8, after: u8) -> bool {
        self.cut.iter().any(|p| matches(p, before, after))
            && !self.never.iter().any(|p| matches(p, before, after))
    }

    /// Every bond is cut: the digest then enumerates peptides by length.
    pub fn is_non_specific(&self) -> bool {
        self.never.is_empty() && self.cut.contains(&(None, None))
    }

    /// The cut positions of `seq` (0-based, a position `i` cuts before `seq[i]`), with the
    /// protein ends 0 and `seq.len()` included, ascending.
    pub fn sites(&self, seq: &[u8]) -> Vec<usize> {
        let mut sites = vec![0usize];
        for i in 1..seq.len() {
            if self.cuts(seq[i - 1], seq[i]) {
                sites.push(i);
            }
        }
        if *sites.last().unwrap() != seq.len() {
            sites.push(seq.len());
        }
        sites
    }
}

/// The rule text of a named preset (`enzymes.json`), `None` for `custom` or an unknown name.
pub fn preset_rule(name: &str) -> Option<&'static str> {
    static TABLE: OnceLock<BTreeMap<String, Option<String>>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let v: serde_json::Value = serde_json::from_str(ENZYMES_JSON).expect("enzymes.json parses");
        v["presets"]
            .as_object()
            .expect("enzymes.json has presets")
            .iter()
            .map(|(k, p)| (k.clone(), p["rule"].as_str().map(str::to_string)))
            .collect()
    });
    table.get(name).and_then(|r| r.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shared_test_vectors_hold() {
        let v: serde_json::Value = serde_json::from_str(ENZYMES_JSON).unwrap();
        for t in v["test_vectors"].as_array().unwrap() {
            let rule = CleavageRule::parse(t["rule"].as_str().unwrap()).unwrap();
            let want: Vec<usize> = t["sites"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(
                rule.sites(t["seq"].as_str().unwrap().as_bytes()),
                want,
                "{t}"
            );
        }
    }

    #[test]
    fn every_preset_parses_and_only_non_specific_is_non_specific() {
        let v: serde_json::Value = serde_json::from_str(ENZYMES_JSON).unwrap();
        for (name, p) in v["presets"].as_object().unwrap() {
            match p["rule"].as_str() {
                Some(r) => {
                    let rule = CleavageRule::parse(r).unwrap();
                    assert_eq!(rule.is_non_specific(), name == "non_specific", "{name}");
                    assert_eq!(preset_rule(name), Some(r));
                }
                None => assert_eq!(name, "custom"),
            }
        }
    }

    #[test]
    fn malformed_rules_are_refused() {
        for bad in ["", "K", "KRP", "!*P", "K1", "!", "k*,!*p,x"] {
            assert!(CleavageRule::parse(bad).is_err(), "{bad}");
        }
        assert_eq!(
            CleavageRule::parse(" k* , !*p ").unwrap(),
            CleavageRule::parse("K*,!*P").unwrap()
        );
    }
}
