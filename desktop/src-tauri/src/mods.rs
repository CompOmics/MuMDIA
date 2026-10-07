//! The engine's modification catalogue, `mumdia-core/src/modifications.json`, compiled in.
//!
//! The same file is the engine's mass table, so a modification offered on the Search screen
//! or passed to DIA-NN is always one the engine can read back. The DIA-NN build needs the
//! UniMod accession and mass of each one for `--var-mod` / `--fixed-mod`.

use serde::Deserialize;

pub const CATALOGUE_JSON: &str =
    include_str!("../../../rust/mumdia/crates/mumdia-core/src/modifications.json");

#[derive(Clone, Debug, Deserialize)]
pub struct Modification {
    pub name: String,
    pub unimod: u32,
    pub mass: f64,
}

#[derive(Deserialize)]
struct Catalogue {
    modifications: Vec<Modification>,
}

pub fn catalogue() -> &'static [Modification] {
    static CAT: std::sync::OnceLock<Vec<Modification>> = std::sync::OnceLock::new();
    CAT.get_or_init(|| {
        serde_json::from_str::<Catalogue>(CATALOGUE_JSON)
            .expect("modifications.json is compiled in and parsed by a unit test")
            .modifications
    })
}

pub fn lookup(name: &str) -> Option<&'static Modification> {
    catalogue().iter().find(|m| m.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_catalogue_parses_and_names_the_standard_pair() {
        assert!(catalogue().len() >= 30);
        assert_eq!(lookup("Carbamidomethyl").map(|m| m.unimod), Some(4));
        assert_eq!(lookup("Oxidation").map(|m| m.unimod), Some(35));
        assert_eq!(lookup("Phospho").map(|m| m.unimod), Some(21));
        assert!(lookup("NotAModification").is_none());
    }
}
