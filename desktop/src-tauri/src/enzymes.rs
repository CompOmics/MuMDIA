//! The engine's cleavage presets, `mumdia-core/src/enzymes.json`, compiled in.
//!
//! A cleavage rule is written in the residue-pair syntax of DIA-NN `--cut` (pairs around
//! the cut, `*` for any residue, a leading `!` for a pair never cut), so the rule the Search
//! screen offers is the rule the DIA-NN library build passes on and the one the engine's own
//! digest (`digest.enzyme`, `digest.cleavage_rule`) applies. The parser mirrors the engine's
//! `mumdia_core::cleavage`, and both are tested against the file's `test_vectors`.

use serde::Serialize;

pub const ENZYMES_JSON: &str =
    include_str!("../../../rust/mumdia/crates/mumdia-core/src/enzymes.json");

/// The default preset, the engine's `digest.enzyme` default.
pub const DEFAULT_ENZYME: &str = "trypsin_p";

/// One preset as the Search screen lists it.
#[derive(Clone, Debug, Serialize)]
pub struct Preset {
    pub name: String,
    pub label: String,
    pub note: String,
    /// `None` for `custom`.
    pub rule: Option<String>,
}

/// Every preset, in the file's order.
pub fn presets() -> Vec<Preset> {
    let v: serde_json::Value = serde_json::from_str(ENZYMES_JSON).expect("enzymes.json parses");
    v["presets"]
        .as_object()
        .expect("enzymes.json has presets")
        .iter()
        .map(|(name, p)| Preset {
            name: name.clone(),
            label: p["label"].as_str().unwrap_or(name).to_string(),
            note: p["note"].as_str().unwrap_or("").to_string(),
            rule: p["rule"].as_str().map(str::to_string),
        })
        .collect()
}

/// The rule text of `enzyme` (a preset name), or `custom_rule` when the preset is `custom`,
/// checked.
pub fn resolve(enzyme: &str, custom_rule: Option<&str>) -> Result<String, String> {
    let text = if enzyme == "custom" {
        custom_rule
            .map(str::trim)
            .filter(|r| !r.is_empty())
            .ok_or("a custom cleavage needs a rule, e.g. K*,R*,!*P")?
            .to_string()
    } else {
        presets()
            .into_iter()
            .find(|p| p.name == enzyme)
            .and_then(|p| p.rule)
            .ok_or_else(|| format!("unknown cleavage preset '{enzyme}'"))?
    };
    Rule::parse(&text)?;
    Ok(text)
}

type Side = Option<u8>;

/// A parsed rule; see the module doc.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
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

impl Rule {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut rule = Rule {
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
            return Err(format!("cleavage rule '{text}' has no pair that cuts"));
        }
        Ok(rule)
    }

    pub fn cuts(&self, before: u8, after: u8) -> bool {
        self.cut.iter().any(|p| matches(p, before, after))
            && !self.never.iter().any(|p| matches(p, before, after))
    }

    /// Every bond is cut: peptides are enumerated by length.
    pub fn is_non_specific(&self) -> bool {
        self.never.is_empty() && self.cut.contains(&(None, None))
    }

    /// Whether some pair that cuts names `residue` before the cut (a rule that cuts after
    /// K, say, which `--no-cut-after-mod` for a K modification is about). A `*` before the
    /// cut does not count: that rule is about the residue after it.
    pub fn cuts_after(&self, residue: u8) -> bool {
        self.cut.iter().any(|p| p.0 == Some(residue))
    }

    /// Cut positions with the protein ends 0 and `seq.len()`, ascending.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shared_test_vectors_hold() {
        let v: serde_json::Value = serde_json::from_str(ENZYMES_JSON).unwrap();
        for t in v["test_vectors"].as_array().unwrap() {
            let rule = Rule::parse(t["rule"].as_str().unwrap()).unwrap();
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
    fn presets_resolve_and_custom_rules_are_checked() {
        assert_eq!(resolve("trypsin_p", None).unwrap(), "K*,R*");
        assert_eq!(resolve("non_specific", None).unwrap(), "**");
        assert!(Rule::parse(&resolve("non_specific", None).unwrap())
            .unwrap()
            .is_non_specific());
        assert_eq!(resolve("custom", Some(" E*,D* ")).unwrap(), "E*,D*");
        assert!(resolve("custom", None).is_err());
        assert!(resolve("custom", Some("KRP")).is_err());
        assert!(resolve("pepsin", None).is_err());
        assert!(presets().iter().any(|p| p.name == DEFAULT_ENZYME));
        assert!(Rule::parse("K*,R*").unwrap().cuts_after(b'K'));
        assert!(!Rule::parse("*D").unwrap().cuts_after(b'K'));
    }
}
