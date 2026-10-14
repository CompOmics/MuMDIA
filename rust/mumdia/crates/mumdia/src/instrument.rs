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
//! description sits in the first few KiB of every mzML, and mzdata does not expose the
//! `referenceableParamGroup` an instrument configuration refers to. ThermoRawFileParser
//! writes the model there (`MS:1003378` "Orbitrap Astral"), ProteoWizard `msconvert` in
//! the same place or directly in the `instrumentConfiguration`.

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

/// How much of the file the header is looked for in. The instrument description precedes
/// `<run>`; 1 MiB is what `convert` already reads for the spectrum count.
const HEADER_BYTES: usize = 1 << 20;

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

/// The instrument the mzML at `path` declares, or `None` when the header cannot be read or
/// names no instrument.
pub fn from_mzml(path: &str) -> Option<Instrument> {
    use std::io::Read as _;
    let mut head = vec![0u8; HEADER_BYTES];
    let mut f = std::fs::File::open(path).ok()?;
    let mut n = 0;
    // `read` may return less than asked before the end of the file.
    while n < head.len() {
        match f.read(&mut head[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(_) => return None,
        }
    }
    parse_header(&String::from_utf8_lossy(&head[..n]))
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
/// The model is the first cvParam of the `referenceableParamGroupList` or of an
/// `instrumentConfiguration` outside its `componentList` (the source, analyzer and detector
/// live there), other than the serial number.
pub fn parse_header(text: &str) -> Option<Instrument> {
    // Only the header: nothing after `<run` describes the instrument.
    let text = match text.find("<run") {
        Some(at) => &text[..at],
        None => text,
    };
    let mut inst = Instrument::default();
    let mut regions: Vec<&str> = Vec::new();
    if let Some(r) = element_body(text, "referenceableParamGroupList") {
        regions.push(r);
    }
    let configs = element_body(text, "instrumentConfigurationList");
    if let Some(c) = configs {
        inst.astral_analyzer = c.contains(&format!("\"{ASTRAL_ANALYZER}\""));
        regions.push(c);
    }
    for region in regions {
        let stripped = strip_elements(region, "componentList");
        for param in stripped.split("<cvParam").skip(1) {
            let end = param.find('>').unwrap_or(param.len());
            let tag = &param[..end];
            let accession = attr(tag, "accession").unwrap_or_default();
            if accession == SERIAL_NUMBER || accession.is_empty() {
                continue;
            }
            inst.accession = accession;
            inst.name = attr(tag, "name").unwrap_or_default();
            return Some(inst);
        }
    }
    if inst.astral_analyzer {
        return Some(inst);
    }
    None
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
