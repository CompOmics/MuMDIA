//! The instrument that acquired a run, read from the mzML header or from the metadata of a
//! timsTOF `.d`, and the provenance `convert` records with the spectra.
//!
//! `convert` records it in the spectra artifacts' reports (`params.instrument`), and the
//! settings that are left to the engine (`extract.apex_count_window = "auto"`) are resolved
//! from that record, so a standalone `extract` on converted spectra, an ungrouped `run`, a
//! grouped run and every run of an experiment resolve them the same way.
//!
//! The same report says which reader wrote the spectra (`params.reader`, written only for
//! the native diaPASEF reader; absent means mzML) and from which source (`params.mzml`).
//! [`provenance`] reads all three. The instrument is provenance only: whether the spectra
//! carry 1/K0 is the reader's fact ([`Reader::has_im`]), never the model name's, because
//! a timsTOF mzML written by msconvert has no 1/K0.
//!
//! That fact has two sources, and both go through [`Reader::has_im`] so they cannot disagree:
//! before conversion the raw dispatch says which reader `convert` will use
//! ([`reader_for_input`], which `convert` itself calls to choose the reader it records), and
//! after it the recorded `params.reader` says which reader did ([`spectra_has_im`]). Every
//! TIMS-dependent setting (`predict_frag.im_predictor = auto` today) resolves from one of the
//! two: [`input_has_im`] for library-build keys, decided before anything is converted, and
//! [`spectra_has_im`] for a stage that reads converted spectra.
//!
//! The header is read as text, as `convert`'s declared spectrum count is: the instrument
//! description precedes `<run>` (the first few KiB of a Thermo or Bruker mzML, 559 KB into
//! a SCIEX MS Data Converter file), and mzdata does not expose the `referenceableParamGroup`
//! an instrument configuration refers to. ThermoRawFileParser writes the model there
//! (`MS:1003378` "Orbitrap Astral"), ProteoWizard `msconvert` in the same place or directly
//! in the `instrumentConfiguration`, and the SCIEX MS Data Converter directly in the
//! configuration (`MS:1003443` "ZenoTOF 8600"). Only the configurations and the groups they
//! name are read: the same list holds the groups spectra refer to ([`parse_header`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// PSI-MS accession of the Orbitrap Astral instrument model.
pub const ORBITRAP_ASTRAL: &str = "MS:1003378";
/// PSI-MS accession of the Astral analyzer ("asymmetric track lossless time-of-flight
/// analyzer"), which ThermoRawFileParser lists as the analyzer of the Astral's second
/// instrument configuration.
pub const ASTRAL_ANALYZER: &str = "MS:1003379";
/// PSI-MS accession of "instrument serial number", which shares the parameter group with
/// the model and is never the model.
const SERIAL_NUMBER: &str = "MS:1000529";
/// PSI-MS accession of "customization": free text, under which mzdata stores a Thermo
/// instrument method.
const CUSTOMIZATION: &str = "MS:1000032";

/// Spectrum-level terms, which describe a spectrum and never an instrument: "MS1 spectrum",
/// "MSn spectrum", "ms level", "centroid spectrum", "profile spectrum", "negative scan",
/// "positive scan", and the scan window limits. They live in the parameter groups a
/// spectrum refers to, and the SCIEX MS Data Converter writes 1,328 such groups ahead of
/// the instrument configuration, the first of them starting with "MS1 spectrum". Only the
/// groups an instrument configuration refers to are read, so none of these should reach the
/// model test; they are skipped all the same, for a converter that mixes the two.
const SPECTRUM_TERMS: [&str; 9] = [
    "MS:1000579",
    "MS:1000580",
    "MS:1000511",
    "MS:1000127",
    "MS:1000128",
    "MS:1000129",
    "MS:1000130",
    "MS:1000500",
    "MS:1000501",
];

/// How much of the file the header is looked for in, at most. The instrument description
/// precedes `<run>`, which the read stops at. It sits in the first few KiB of a Thermo or
/// Bruker mzML, but the SCIEX MS Data Converter writes two parameter groups per experiment
/// of the method before it (559 KB into a 664-experiment ZT Scan DIA run), so a method
/// with a few thousand experiments puts it past the 1 MiB `convert` reads for the spectrum
/// count.
const HEADER_BYTES: usize = 16 << 20;

/// The header is read in pieces of this size, and the read stops at the first `<run`.
const HEADER_STEP: usize = 1 << 20;

/// The instrument model a file declares, and whether it lists an Astral analyzer.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Instrument {
    /// PSI-MS accession of the model, e.g. `MS:1003378`. Empty when the header names none.
    pub accession: String,
    /// The model's name as written, e.g. `Orbitrap Astral`.
    pub name: String,
    /// One of the instrument configurations has the Astral analyzer (`MS:1003379`).
    pub astral_analyzer: bool,
}

impl Instrument {
    /// An Orbitrap Astral (or a later Astral model, which keeps the word in its name).
    pub fn is_astral(&self) -> bool {
        self.accession == ORBITRAP_ASTRAL
            || self.name.to_ascii_lowercase().contains("astral")
            || self.astral_analyzer
    }

    /// Short description for a log line.
    pub fn describe(&self) -> String {
        match (self.name.is_empty(), self.accession.is_empty()) {
            (false, false) => format!("{} ({})", self.name, self.accession),
            (false, true) => self.name.clone(),
            (true, false) => self.accession.clone(),
            (true, true) => "unknown".to_string(),
        }
    }
}

/// `params.reader` of spectra the native timsTOF reader wrote (`stages::convert::tdf`).
pub const READER_TIMSRUST: &str = "timsrust";

/// The instrument a timsTOF `.d` names in `analysis.tdf` (`GlobalMetadata.InstrumentName`),
/// or `None` when the table or the key is missing. Bruker files carry no PSI-MS model
/// accession, so only the name is set.
pub fn from_tdf(dir: &str) -> Option<Instrument> {
    let db = std::path::Path::new(dir).join("analysis.tdf");
    if !db.is_file() {
        return None;
    }
    let con = rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .ok()?;
    let name: String = con
        .query_row(
            "SELECT Value FROM GlobalMetadata WHERE Key = 'InstrumentName'",
            [],
            |r| r.get(0),
        )
        .ok()?;
    let name = name.trim().to_string();
    (!name.is_empty()).then_some(Instrument {
        accession: String::new(),
        name,
        astral_analyzer: false,
    })
}

/// `params.reader` of spectra read directly from a Thermo `.raw` (`convert.thermo_native`).
pub const READER_THERMO: &str = "thermorawfilereader";

/// The instrument a reader's metadata describes (a Thermo `.raw` read through mzdata): the
/// model is the first controlled parameter of an instrument configuration other than the
/// serial number, and the Astral analyzer is looked for among the components, as
/// [`parse_header`] does for an mzML. `None` when no configuration names a model.
pub fn from_configurations<'a>(
    configs: impl IntoIterator<Item = &'a mzdata::meta::InstrumentConfiguration>,
) -> Option<Instrument> {
    let mut configs: Vec<&mzdata::meta::InstrumentConfiguration> = configs.into_iter().collect();
    // A HashMap's order is not stable; the lowest configuration id decides.
    configs.sort_by_key(|c| c.id);
    let mut inst = Instrument::default();
    for c in &configs {
        for comp in &c.components {
            if comp
                .params
                .iter()
                .any(|p| p.curie_str().as_deref() == Some(ASTRAL_ANALYZER))
            {
                inst.astral_analyzer = true;
            }
        }
    }
    for c in &configs {
        for p in &c.params {
            let Some(accession) = p.curie_str() else {
                continue;
            };
            // The serial number, and the instrument method text mzdata stores as
            // `customization`, precede the model.
            if accession == SERIAL_NUMBER || accession == CUSTOMIZATION {
                continue;
            }
            inst.accession = accession;
            inst.name = p.name.clone();
            return Some(inst);
        }
    }
    inst.astral_analyzer.then_some(inst)
}

/// The instrument the mzML at `path` declares, or `None` when the header cannot be read or
/// names no instrument.
pub fn from_mzml(path: &str) -> Option<Instrument> {
    use std::io::Read as _;
    let mut f = std::fs::File::open(path).ok()?.take(HEADER_BYTES as u64);
    let mut head: Vec<u8> = Vec::with_capacity(HEADER_STEP);
    let mut buf = vec![0u8; HEADER_STEP];
    loop {
        // `read` may return less than asked before the end of the file.
        let k = match f.read(&mut buf) {
            Ok(0) => break,
            Ok(k) => k,
            Err(_) => return None,
        };
        // A `<run` split across two reads is found once the second arrives: the search
        // starts a few bytes before the new piece.
        let from = head.len().saturating_sub(4);
        head.extend_from_slice(&buf[..k]);
        if head[from..].windows(4).any(|w| w == b"<run") {
            break;
        }
    }
    parse_header(&String::from_utf8_lossy(&head))
}

/// Which reader wrote a spectra artifact (`params.reader`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reader {
    /// mzML through mzdata, including a vendor file converted to mzML first. Also every
    /// report without the field, which is every report written before it existed.
    Mzml,
    /// The native timsTOF diaPASEF reader: the spectra carry per-peak 1/K0.
    Timsrust,
}

/// What `convert` recorded about where a spectra artifact came from.
#[derive(Clone, Debug, PartialEq)]
pub struct Provenance {
    /// `params.instrument`; `None` when the source named none or predates the field.
    pub instrument: Option<Instrument>,
    /// `params.reader`; [`Reader::Mzml`] when absent.
    pub reader: Reader,
    /// `params.mzml`: the path convert read, an mzML or a `.d`.
    pub source: Option<String>,
}

impl Reader {
    /// Do spectra written by this reader carry per-peak 1/K0? The one predicate for 4D
    /// behaviour.
    pub fn has_im(self) -> bool {
        self == Reader::Timsrust
    }
}

impl Provenance {
    /// Do the spectra carry per-peak 1/K0? [`Reader::has_im`] of the recorded reader.
    pub fn has_im(&self) -> bool {
        self.reader.has_im()
    }
}

/// The reader `convert` uses for the input at `path`, from the raw dispatch: the native
/// timsTOF reader for a diaPASEF `.d` (`raw::Vendor::BrukerTdf`), mzML for everything else
/// (a vendor file is converted to mzML first). `convert` chooses its reader through this
/// function, so the reader predicted before conversion is the one it records.
pub fn reader_for_input(path: &str) -> Reader {
    if crate::raw::detect(path) == crate::raw::Vendor::BrukerTdf {
        Reader::Timsrust
    } else {
        Reader::Mzml
    }
}

/// Will the spectra converted from the input at `path` carry per-peak 1/K0? Decided before
/// conversion, for the settings that must be known first (the library build).
pub fn input_has_im(path: &str) -> bool {
    reader_for_input(path).has_im()
}

/// Do the converted spectra at `spectra_path` carry per-peak 1/K0, by the reader `convert`
/// recorded? False for a missing or unreadable report, and for spectra converted before the
/// reader was recorded (all of which are mzML).
pub fn spectra_has_im(spectra_path: &str) -> bool {
    provenance(spectra_path).is_some_and(|p| p.has_im())
}

/// The provenance `convert` recorded for the spectra artifact at `spectra_path`
/// (`<spectra>.report.json`, `params`). `None` for a missing or unreadable report.
pub fn provenance(spectra_path: &str) -> Option<Provenance> {
    let text = std::fs::read_to_string(format!("{spectra_path}.report.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let params = v.get("params")?;
    let instrument = params
        .get("instrument")
        .filter(|i| !i.is_null())
        .and_then(|i| serde_json::from_value(i.clone()).ok());
    let reader = match params.get("reader").and_then(|r| r.as_str()) {
        Some(READER_TIMSRUST) => Reader::Timsrust,
        _ => Reader::Mzml,
    };
    let source = params
        .get("mzml")
        .and_then(|m| m.as_str())
        .map(str::to_string);
    Some(Provenance {
        instrument,
        reader,
        source,
    })
}

/// The instrument `convert` recorded for the spectra artifact at `spectra_path`
/// (`<spectra>.report.json`, `params.instrument`). `None` for spectra converted before the
/// field existed, for a missing report and for a file that named no instrument.
pub fn recorded(spectra_path: &str) -> Option<Instrument> {
    provenance(spectra_path)?.instrument
}

/// Parse the instrument from the text of an mzML header.
///
/// The model is the first cvParam of an `instrumentConfiguration`, read in document order
/// from the configuration's own cvParams outside its `componentList` (the source, analyzer
/// and detector live there) and from the `referenceableParamGroup`s it refers to, other
/// than the serial number, a `customization` text and any spectrum-level term
/// ([`SPECTRUM_TERMS`]). ThermoRawFileParser puts the model in a group the configuration
/// refers to, msconvert and the SCIEX MS Data Converter in the configuration itself. When
/// no configuration names one, the first such term of the parameter group list is taken,
/// which is where tdf2mzml writes a timsTOF model without referring to it.
///
/// Until 2026-11-10 the first cvParam of the WHOLE `referenceableParamGroupList` was taken
/// before any configuration was looked at. That list also holds the groups spectra refer
/// to, and on a SCIEX MS Data Converter file it is 1,328 of them ahead of the
/// configuration, so the instrument read as "MS1 spectrum (MS:1000579)" instead of
/// "ZenoTOF 8600 (MS:1003443)".
pub fn parse_header(text: &str) -> Option<Instrument> {
    // Only the header: nothing after `<run` describes the instrument.
    let text = match text.find("<run") {
        Some(at) => &text[..at],
        None => text,
    };
    let mut inst = Instrument::default();
    let configs = element_body(text, "instrumentConfigurationList").unwrap_or("");
    inst.astral_analyzer = configs.contains(&format!("\"{ASTRAL_ANALYZER}\""));
    // The parameter groups in document order, and by id for the configurations' references.
    let group_list: Vec<(&str, &str)> = element_body(text, "referenceableParamGroupList")
        .map(|list| elements(list, "referenceableParamGroup"))
        .unwrap_or_default();
    let groups: BTreeMap<String, &str> = group_list
        .iter()
        .filter_map(|&(tag, body)| Some((attr(tag, "id")?, body)))
        .collect();
    let model = |tag: &str| -> Option<(String, String)> {
        let accession = attr(tag, "accession").unwrap_or_default();
        let skip = accession.is_empty()
            || accession == SERIAL_NUMBER
            || accession == CUSTOMIZATION
            || SPECTRUM_TERMS.contains(&accession.as_str());
        (!skip).then(|| (accession, attr(tag, "name").unwrap_or_default()))
    };
    let found = elements(configs, "instrumentConfiguration")
        .into_iter()
        .find_map(|(_, config)| {
            let own = strip_elements(config, "componentList");
            config_params(&own, &groups).into_iter().find_map(model)
        })
        // A group no configuration refers to: tdf2mzml writes the timsTOF model in a
        // `CommonInstrumentParams` group and never names it. The first instrument term of
        // the list, in document order, as before 2026-11-10, minus the spectrum terms.
        .or_else(|| {
            group_list
                .iter()
                .find_map(|&(_, body)| cv_params(body).into_iter().find_map(model))
        });
    match found {
        Some((accession, name)) => {
            inst.accession = accession;
            inst.name = name;
            Some(inst)
        }
        None => inst.astral_analyzer.then_some(inst),
    }
}

/// The inside of every `<cvParam ...>` tag of a configuration body, in document order, with
/// each `<referenceableParamGroupRef ref="..."/>` replaced by the cvParams of that group.
/// A reference to a group the header does not define contributes nothing.
fn config_params<'a>(body: &'a str, groups: &BTreeMap<String, &'a str>) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(at) = rest.find('<') {
        rest = &rest[at + 1..];
        let end = rest.find('>').unwrap_or(rest.len());
        let tag = &rest[..end];
        if let Some(inner) = tag.strip_prefix("cvParam") {
            out.push(inner);
        } else if let Some(inner) = tag.strip_prefix("referenceableParamGroupRef") {
            if let Some(group) = attr(inner, "ref").and_then(|id| groups.get(&id)) {
                out.extend(cv_params(group));
            }
        }
        rest = &rest[end..];
    }
    out
}

/// The inside of every `<cvParam ...>` tag of `text`, in document order.
fn cv_params(text: &str) -> Vec<&str> {
    text.split("<cvParam")
        .skip(1)
        .map(|p| &p[..p.find('>').unwrap_or(p.len())])
        .collect()
}

/// Every `<name ...>...</name>` element of `text`, not nested: the inside of its opening
/// tag and its body (empty for a self-closing `<name .../>`). `name` must be followed by
/// whitespace, `>` or `/`, so `instrumentConfiguration` does not match
/// `instrumentConfigurationList`, nor `referenceableParamGroup` its `Ref`.
fn elements<'a>(text: &'a str, name: &str) -> Vec<(&'a str, &'a str)> {
    let (open, close) = (format!("<{name}"), format!("</{name}>"));
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(&open) {
        let after = &rest[at + open.len()..];
        let boundary = after
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_whitespace() || c == '>' || c == '/');
        if !boundary {
            rest = after;
            continue;
        }
        let Some(tag_end) = after.find('>') else {
            break;
        };
        let tag = &after[..tag_end];
        let body_start = &after[tag_end + 1..];
        if tag.ends_with('/') {
            out.push((tag, ""));
            rest = body_start;
            continue;
        }
        let body_end = body_start.find(&close).unwrap_or(body_start.len());
        out.push((tag, &body_start[..body_end]));
        rest = &body_start[(body_end + close.len()).min(body_start.len())..];
    }
    out
}

/// The text between `<name ...>` and `</name>`, or `None`.
fn element_body<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let open = text.find(&format!("<{name}"))?;
    let rest = &text[open..];
    let start = rest.find('>')? + 1;
    let close = rest.find(&format!("</{name}>")).unwrap_or(rest.len());
    (start <= close).then(|| &rest[start..close])
}

/// `text` without any `<name ...>...</name>` element.
fn strip_elements(text: &str, name: &str) -> String {
    let (open, close) = (format!("<{name}"), format!("</{name}>"));
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(&open) {
        out.push_str(&rest[..at]);
        match rest[at..].find(&close) {
            Some(end) => rest = &rest[at + end + close.len()..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// The value of attribute `name` in the inside of a tag, either quote style.
fn attr(tag: &str, name: &str) -> Option<String> {
    let mut from = 0;
    while let Some(i) = tag[from..].find(name) {
        let at = from + i;
        let before_ok = at == 0 || tag.as_bytes()[at - 1].is_ascii_whitespace();
        let rest = tag[at + name.len()..].trim_start();
        if before_ok {
            if let Some(rest) = rest.strip_prefix('=') {
                let rest = rest.trim_start();
                let quote = rest.chars().next()?;
                if quote == '"' || quote == '\'' {
                    let body = &rest[1..];
                    let end = body.find(quote)?;
                    return Some(body[..end].to_string());
                }
            }
        }
        from = at + name.len();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ThermoRawFileParser 1.4.5 on an Orbitrap Astral (HYE REP1), abridged.
    const TRFP_ASTRAL: &str = r#"<mzML>
    <referenceableParamGroupList count="1">
      <referenceableParamGroup id="commonInstrumentParams">
        <cvParam cvRef="MS" accession="MS:1003378" value="" name="Orbitrap Astral" />
        <cvParam cvRef="MS" accession="MS:1000529" value="OA10208" name="instrument serial number" />
      </referenceableParamGroup>
    </referenceableParamGroupList>
    <softwareList count="1">
      <software id="ThermoRawFileParser" version="1.4.5">
        <cvParam cvRef="MS" accession="MS:1003145" name="ThermoRawFileParser" />
      </software>
    </softwareList>
    <instrumentConfigurationList count="2">
      <instrumentConfiguration id="IC1">
        <referenceableParamGroupRef ref="commonInstrumentParams" />
        <componentList count="3">
          <source order="1"><cvParam cvRef="MS" accession="MS:1000398" value="" name="nanoelectrospray" /></source>
          <analyzer order="2"><cvParam cvRef="MS" accession="MS:1000484" value="" name="orbitrap" /></analyzer>
          <detector order="3"><cvParam cvRef="MS" accession="MS:1000624" value="" name="inductive detector" /></detector>
        </componentList>
      </instrumentConfiguration>
      <instrumentConfiguration id="IC2">
        <referenceableParamGroupRef ref="commonInstrumentParams" />
        <componentList count="3">
          <analyzer order="2"><cvParam cvRef="MS" accession="MS:1003379" value="" name="asymmetric track lossless time-of-flight analyzer" /></analyzer>
        </componentList>
      </instrumentConfiguration>
    </instrumentConfigurationList>
    <run id="x">"#;

    #[test]
    fn an_astral_written_by_thermorawfileparser_is_recognised() {
        let i = parse_header(TRFP_ASTRAL).unwrap();
        assert_eq!(i.accession, ORBITRAP_ASTRAL);
        assert_eq!(i.name, "Orbitrap Astral");
        assert!(i.astral_analyzer);
        assert!(i.is_astral());
    }

    #[test]
    fn a_q_exactive_is_not_an_astral() {
        let t = TRFP_ASTRAL
            .replace(
                r#"accession="MS:1003378" value="" name="Orbitrap Astral""#,
                r#"accession="MS:1001911" value="" name="Q Exactive""#,
            )
            .replace(ASTRAL_ANALYZER, "MS:1000484");
        let i = parse_header(&t).unwrap();
        assert_eq!(i.accession, "MS:1001911");
        assert_eq!(i.name, "Q Exactive");
        assert!(!i.is_astral());
    }

    #[test]
    fn a_model_inside_the_instrument_configuration_is_found() {
        // msconvert writes the model directly in the configuration when there is no group;
        // Bruker files name only the vendor's generic model term.
        let t = r#"<instrumentConfigurationList count="1">
          <instrumentConfiguration id="IC1">
            <cvParam cvRef="PSI-MS" accession="MS:1000122" name="Bruker Daltonics instrument model" value=""/>
            <cvParam cvRef="PSI-MS" accession="MS:1000529" name="instrument serial number" value="1813116.00032"/>
            <componentList count="1"><analyzer order="1"><cvParam cvRef="PSI-MS" accession="MS:1000084" name="time-of-flight" value=""/></analyzer></componentList>
          </instrumentConfiguration>
        </instrumentConfigurationList><run id="x">"#;
        let i = parse_header(t).unwrap();
        assert_eq!(i.accession, "MS:1000122");
        assert!(!i.is_astral());
    }

    /// The SCIEX MS Data Converter 2.0.1.0 (SCIEX OS 5.0) header of a ZT Scan DIA run on a
    /// ZenoTOF 8600, abridged: two parameter groups per experiment of the method (1,328 in
    /// the real file, 559 KB of them), every one a spectrum's, ahead of the instrument
    /// configuration, which carries the model itself and refers to no group.
    fn sciex_header(experiments: usize) -> String {
        let mut groups = String::new();
        for e in 1..=experiments {
            let (term, level, lo) = if e == 1 {
                ("MS:1000579\" name=\"MS1 spectrum", 1, 400)
            } else {
                ("MS:1000580\" name=\"MSn spectrum", 2, 140)
            };
            groups.push_str(&format!(
                r#"
      <referenceableParamGroup id="sample1experiment{e}SpectrumParams">
        <cvParam cvRef="MS" accession="{term}" />
        <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{level}" />
        <cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" />
        <cvParam cvRef="MS" accession="MS:1000130" name="positive scan" />
      </referenceableParamGroup>
      <referenceableParamGroup id="sample1experiment{e}ScanWindowParams">
        <cvParam cvRef="MS" accession="MS:1000501" name="scan window lower limit" unitAccession="MS:1000040" unitCvRef="MS" unitName="m/z" value="{lo}" />
        <cvParam cvRef="MS" accession="MS:1000500" name="scan window upper limit" unitAccession="MS:1000040" unitCvRef="MS" unitName="m/z" value="1750" />
      </referenceableParamGroup>"#
            ));
        }
        format!(
            r#"<?xml version="1.0" encoding="iso-8859-1"?>
<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">
  <mzML version="1.1.0">
    <fileDescription>
      <fileContent>
        <cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" />
        <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" />
      </fileContent>
    </fileDescription>
    <referenceableParamGroupList count="{n}">{groups}
    </referenceableParamGroupList>
    <softwareList count="2">
      <software version="5.0.0.1539" id="SCIEXsoftware">
        <cvParam cvRef="MS" accession="MS:1003446" name="SCIEX OS" />
      </software>
      <software version="2.0.1.0" id="SCIEXMSDataConverter">
        <cvParam cvRef="MS" accession="MS:1003447" name="SCIEX MS Data Converter" />
      </software>
    </softwareList>
    <instrumentConfigurationList count="1">
      <instrumentConfiguration id="instrumentConfiguration1">
        <cvParam cvRef="MS" accession="MS:1003443" name="ZenoTOF 8600" />
        <cvParam cvRef="MS" accession="MS:1000529" name="instrument serial number" value="FR20062502PL" />
        <componentList count="3">
          <source order="1"><cvParam cvRef="MS" accession="MS:1000008" name="ionization type" /></source>
          <analyzer order="2"><cvParam cvRef="MS" accession="MS:1000084" name="time-of-flight" /></analyzer>
          <detector order="3"><cvParam cvRef="MS" accession="MS:1000117" name="analog-digital converter" /></detector>
        </componentList>
        <softwareRef ref="SCIEXsoftware" />
      </instrumentConfiguration>
    </instrumentConfigurationList>
    <run defaultInstrumentConfigurationRef="instrumentConfiguration1" id="run1">
      <spectrumList defaultDataProcessingRef="dataProcessing1" count="2150032">"#,
            n = 2 * experiments
        )
    }

    #[test]
    fn a_sciex_ms_data_converter_header_names_the_zenotof_not_a_spectrum_term() {
        // 664 experiments, as the ZT Scan DIA method (1 MS1 + 663 Q1 bins): 1,328 groups.
        let t = sciex_header(664);
        assert!(t.matches("<referenceableParamGroup ").count() == 1328);
        let i = parse_header(&t).unwrap();
        assert_eq!(i.accession, "MS:1003443");
        assert_eq!(i.name, "ZenoTOF 8600");
        assert!(!i.astral_analyzer && !i.is_astral());
    }

    #[test]
    fn a_header_past_one_mebibyte_is_still_read() {
        // A method with 3,000 experiments puts the configuration about 2.5 MB in.
        let t = sciex_header(3000);
        assert!(t.find("<instrumentConfigurationList").unwrap() > 1 << 20);
        let dir = std::env::temp_dir().join(format!("mumdia_instr_big_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("big.mzML");
        // The spectra follow the header; the read stops at `<run`.
        let mut body = t.clone();
        body.push_str(&"<spectrum/>".repeat(100_000));
        std::fs::write(&path, body).unwrap();
        let i = from_mzml(path.to_str().unwrap()).unwrap();
        assert_eq!(i.name, "ZenoTOF 8600");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn spectrum_groups_a_configuration_does_not_name_are_not_read() {
        // ThermoRawFileParser's shape, with a spectrum group FIRST in the list: the model
        // still comes from the group the configuration refers to.
        let t = TRFP_ASTRAL.replace(
            r#"<referenceableParamGroupList count="1">"#,
            r#"<referenceableParamGroupList count="2">
      <referenceableParamGroup id="spectrumGroup">
        <cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" />
      </referenceableParamGroup>"#,
        );
        let i = parse_header(&t).unwrap();
        assert_eq!(i.accession, ORBITRAP_ASTRAL);
        // A configuration that refers to a spectrum group skips its spectrum terms.
        let t = r#"<referenceableParamGroupList count="2">
          <referenceableParamGroup id="s"><cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" />
            <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="2" /></referenceableParamGroup>
          <referenceableParamGroup id="i"><cvParam cvRef="MS" accession="MS:1002533" name="TripleTOF 6600" /></referenceableParamGroup>
        </referenceableParamGroupList>
        <instrumentConfigurationList count="1"><instrumentConfiguration id="IC1">
          <referenceableParamGroupRef ref="s"/><referenceableParamGroupRef ref="i"/>
        </instrumentConfiguration></instrumentConfigurationList><run id="x">"#;
        let i = parse_header(t).unwrap();
        assert_eq!(i.accession, "MS:1002533");
        assert_eq!(i.name, "TripleTOF 6600");
    }

    #[test]
    fn a_model_in_a_group_no_configuration_names_is_still_found() {
        // tdf2mzml 0.3 on a timsTOF Pro diaPASEF run, abridged.
        let t = r#"<referenceableParamGroupList count="1">
          <referenceableParamGroup id="CommonInstrumentParams">
            <cvParam cvRef="PSI-MS" accession="MS:1000122" name="Bruker Daltonics instrument model" value=""/>
          </referenceableParamGroup>
        </referenceableParamGroupList>
        <instrumentConfigurationList count="1"><instrumentConfiguration id="IC1">
          <cvParam cvRef="PSI-MS" accession="MS:1000529" name="instrument serial number" value="1854399.193"/>
          <userParam name="dia_window" value="{'Group': 1, 'mz_center': 813.0, 'width': 26.0}"/>
          <componentList count="1"><analyzer order="2"><cvParam cvRef="PSI-MS" accession="MS:1000084" name="time-of-flight" value=""/></analyzer></componentList>
        </instrumentConfiguration></instrumentConfigurationList><run id="x">"#;
        let i = parse_header(t).unwrap();
        assert_eq!(i.accession, "MS:1000122");
    }

    #[test]
    fn msconvert_writes_the_sciex_model_in_the_configuration() {
        // ProteoWizard 3.0.23339 (ABI reader) on a TripleTOF 6600 Scanning SWATH run.
        let t = r#"<softwareList count="2"><software id="Analyst" version="unknown">
            <cvParam cvRef="MS" accession="MS:1000551" name="Analyst" value=""/></software></softwareList>
          <instrumentConfigurationList count="1">
            <instrumentConfiguration id="IC1">
              <cvParam cvRef="MS" accession="MS:1002533" name="TripleTOF 6600" value=""/>
              <cvParam cvRef="MS" accession="MS:1000529" name="instrument serial number" value="BR20921502"/>
              <componentList count="4">
                <source order="1"><cvParam cvRef="MS" accession="MS:1000073" name="electrospray ionization" value=""/></source>
                <analyzer order="2"><cvParam cvRef="MS" accession="MS:1000081" name="quadrupole" value=""/></analyzer>
              </componentList>
              <softwareRef ref="Analyst"/>
            </instrumentConfiguration>
          </instrumentConfigurationList><run id="x">"#;
        let i = parse_header(t).unwrap();
        assert_eq!(i.accession, "MS:1002533");
    }

    #[test]
    fn a_later_astral_model_is_recognised_by_name() {
        let t = TRFP_ASTRAL
            .replace(
                r#"accession="MS:1003378" value="" name="Orbitrap Astral""#,
                r#"accession="MS:1009999" value="" name="Orbitrap Astral Zoom""#,
            )
            .replace(ASTRAL_ANALYZER, "MS:1000484");
        assert!(parse_header(&t).unwrap().is_astral());
    }

    #[test]
    fn a_header_without_an_instrument_gives_none() {
        assert_eq!(
            parse_header(r#"<mzML><run id="x"><spectrumList count="0"/>"#),
            None
        );
        // The serial number alone is not a model.
        let t = r#"<referenceableParamGroupList count="1"><referenceableParamGroup id="g">
          <cvParam cvRef="MS" accession="MS:1000529" value="1" name="instrument serial number" />
          </referenceableParamGroup></referenceableParamGroupList><run id="x">"#;
        assert_eq!(parse_header(t), None);
    }

    #[test]
    fn the_recorded_instrument_is_read_back_from_the_report() {
        let dir = std::env::temp_dir().join(format!("mumdia_instr_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spectra = dir.join("spectra_ms2.parquet");
        let spectra = spectra.to_str().unwrap();
        assert_eq!(recorded(spectra), None);
        let inst = parse_header(TRFP_ASTRAL).unwrap();
        std::fs::write(
            format!("{spectra}.report.json"),
            serde_json::json!({"params": {"instrument": inst}}).to_string(),
        )
        .unwrap();
        assert_eq!(recorded(spectra), Some(inst));
        // A report written before the field existed.
        std::fs::write(
            format!("{spectra}.report.json"),
            serde_json::json!({"params": {"mzml": "x"}}).to_string(),
        )
        .unwrap();
        assert_eq!(recorded(spectra), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn provenance_reads_the_reader_and_the_source_and_defaults_to_mzml() {
        let dir = std::env::temp_dir().join(format!("mumdia_prov_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spectra = dir.join("spectra_ms2.parquet");
        let spectra = spectra.to_str().unwrap();
        assert_eq!(provenance(spectra), None);
        // An mzML report: no `reader` field.
        std::fs::write(
            format!("{spectra}.report.json"),
            serde_json::json!({"params": {"mzml": "a.mzML", "instrument": null}}).to_string(),
        )
        .unwrap();
        let p = provenance(spectra).unwrap();
        assert_eq!(p.reader, Reader::Mzml);
        assert!(!p.has_im());
        assert_eq!(p.source.as_deref(), Some("a.mzML"));
        assert_eq!(p.instrument, None);
        // A native diaPASEF report.
        let inst = Instrument {
            accession: String::new(),
            name: "timsTOF Ultra 2".into(),
            astral_analyzer: false,
        };
        std::fs::write(
            format!("{spectra}.report.json"),
            serde_json::json!({"params": {"mzml": "b.d", "reader": READER_TIMSRUST,
                "instrument": inst}})
            .to_string(),
        )
        .unwrap();
        let p = provenance(spectra).unwrap();
        assert_eq!(p.reader, Reader::Timsrust);
        assert!(p.has_im());
        assert_eq!(p.source.as_deref(), Some("b.d"));
        assert_eq!(recorded(spectra), Some(inst));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The two sources of the 4D predicate agree: the raw dispatch before conversion and the
    /// reader `convert` records after it, both through `Reader::has_im`.
    #[test]
    fn the_input_and_the_recorded_reader_answer_the_same_question() {
        let dir = std::env::temp_dir().join(format!("mumdia_has_im_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let dia = dir.join("dia.d");
        crate::raw::tdf_fixture(&dia, &[0, 9, 9], "2", true);
        let dda = dir.join("dda.d");
        crate::raw::tdf_fixture(&dda, &[0, 8], "2", true);
        let dia = dia.to_str().unwrap();
        assert_eq!(reader_for_input(dia), Reader::Timsrust);
        assert!(input_has_im(dia));
        for other in [dda.to_str().unwrap(), "run.mzML", "run.raw"] {
            assert_eq!(reader_for_input(other), Reader::Mzml, "{other}");
            assert!(!input_has_im(other), "{other}");
        }
        let spectra = dir.join("spectra_ms2.parquet");
        let spectra = spectra.to_str().unwrap();
        assert!(!spectra_has_im(spectra), "no report reads as mzML");
        for (reader, want) in [(Some(READER_TIMSRUST), true), (None, false)] {
            let mut params = serde_json::json!({"mzml": dia});
            if let Some(r) = reader {
                params["reader"] = serde_json::json!(r);
            }
            std::fs::write(
                format!("{spectra}.report.json"),
                serde_json::json!({ "params": params }).to_string(),
            )
            .unwrap();
            assert_eq!(spectra_has_im(spectra), want);
            assert_eq!(Reader::Timsrust.has_im(), input_has_im(dia));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_tdf_instrument_name_comes_from_global_metadata() {
        let dir = std::env::temp_dir().join(format!("mumdia_tdf_inst_{}.d", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_str().unwrap();
        assert_eq!(from_tdf(d), None);
        {
            let con = rusqlite::Connection::open(dir.join("analysis.tdf")).unwrap();
            con.execute_batch(
                "CREATE TABLE GlobalMetadata (Key TEXT PRIMARY KEY, Value TEXT);
                 INSERT INTO GlobalMetadata VALUES ('InstrumentName', 'timsTOF Ultra 2');
                 INSERT INTO GlobalMetadata VALUES ('InstrumentVendor', 'Bruker');",
            )
            .unwrap();
        }
        let i = from_tdf(d).unwrap();
        assert_eq!(i.name, "timsTOF Ultra 2");
        assert!(i.accession.is_empty() && !i.is_astral());
        std::fs::remove_dir_all(&dir).ok();
    }
}
