/// Fast mzML parser for extracting MS1/MS2 spectra.
/// Replaces PyOpenMS's MzMLFile().load() which is the main I/O bottleneck.
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use flate2::read::ZlibDecoder;
use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::HashMap;
use std::io::Read;

/// A parsed spectrum with m/z, intensity arrays and metadata.
#[derive(Debug, Clone)]
pub struct Spectrum {
    pub scan_id: String,
    pub ms_level: u8,
    pub retention_time: f64, // in seconds
    pub mz: Vec<f64>,
    pub intensity: Vec<f64>,
    /// Isolation window center m/z (MS2 only)
    pub isolation_window_target: Option<f64>,
    /// Isolation window lower offset from target
    pub isolation_window_lower: Option<f64>,
    /// Isolation window upper offset from target
    pub isolation_window_upper: Option<f64>,
}

/// Result of parsing an mzML file: MS1 spectra, MS2→MS1 mapping, MS2 spectra.
pub struct MzMLData {
    pub ms1_spectra: Vec<Spectrum>,
    pub ms2_spectra: Vec<Spectrum>,
    pub ms2_to_ms1_map: HashMap<String, String>, // ms2_scan_id → preceding ms1_scan_id
}

/// Parse binary data array from base64-encoded mzML content.
/// Handles both 32-bit and 64-bit floats, and optional zlib compression.
fn decode_binary_array(
    encoded: &str,
    is_64bit: bool,
    is_compressed: bool,
) -> Vec<f64> {
    let bytes = match BASE64.decode(encoded.trim()) {
        Ok(b) => b,
        Err(_) => return vec![],
    };

    let decompressed;
    let data = if is_compressed {
        let mut decoder = ZlibDecoder::new(&bytes[..]);
        decompressed = {
            let mut buf = Vec::new();
            if decoder.read_to_end(&mut buf).is_err() {
                return vec![];
            }
            buf
        };
        &decompressed[..]
    } else {
        &bytes[..]
    };

    if is_64bit {
        data.chunks(8)
            .filter_map(|chunk| chunk.try_into().ok().map(f64::from_le_bytes))
            .collect()
    } else {
        data.chunks(4)
            .filter_map(|chunk| {
                chunk
                    .try_into()
                    .ok()
                    .map(|b| f32::from_le_bytes(b) as f64)
            })
            .collect()
    }
}

/// Parse an mzML file and extract MS1/MS2 spectra with MS2→MS1 mapping.
pub fn parse_mzml(path: &str) -> Result<MzMLData, String> {
    let file_content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?;

    let mut reader = Reader::from_str(&file_content);
    reader.config_mut().trim_text(true);

    let mut ms1_spectra = Vec::new();
    let mut ms2_spectra = Vec::new();
    let mut ms2_to_ms1_map = HashMap::new();
    let mut last_ms1_id: Option<String> = None;

    // State for current spectrum being parsed
    let mut in_spectrum = false;
    let mut current_scan_id = String::new();
    let mut current_ms_level: u8 = 0;
    let mut current_rt: f64 = 0.0;
    let mut current_iso_target: Option<f64> = None;
    let mut current_iso_lower: Option<f64> = None;
    let mut current_iso_upper: Option<f64> = None;
    let mut in_isolation_window = false;

    // State for binary data arrays
    let mut in_binary_data_array = false;
    let mut is_mz_array = false;
    let mut is_intensity_array = false;
    let mut is_64bit = false;
    let mut is_compressed = false;
    let mut in_binary = false;
    let mut binary_text = String::new();

    let mut current_mz: Vec<f64> = Vec::new();
    let mut current_intensity: Vec<f64> = Vec::new();

    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                let name = e.name();
                let local = name.as_ref();

                match local {
                    b"spectrum" => {
                        in_spectrum = true;
                        current_ms_level = 0;
                        current_rt = 0.0;
                        current_iso_target = None;
                        current_iso_lower = None;
                        current_iso_upper = None;
                        current_mz.clear();
                        current_intensity.clear();

                        // Extract scan ID from "id" attribute
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"id" {
                                current_scan_id =
                                    String::from_utf8_lossy(&attr.value).to_string();
                            }
                        }
                    }
                    b"isolationWindow" if in_spectrum => {
                        in_isolation_window = true;
                    }
                    b"binaryDataArray" => {
                        in_binary_data_array = true;
                        is_mz_array = false;
                        is_intensity_array = false;
                        is_64bit = false;
                        is_compressed = false;
                    }
                    b"binary" => {
                        in_binary = true;
                        binary_text.clear();
                    }
                    b"cvParam" if in_spectrum => {
                        let mut accession = String::new();
                        let mut value = String::new();

                        for attr in e.attributes().flatten() {
                            match attr.key.as_ref() {
                                b"accession" => {
                                    accession =
                                        String::from_utf8_lossy(&attr.value).to_string();
                                }
                                b"value" => {
                                    value =
                                        String::from_utf8_lossy(&attr.value).to_string();
                                }
                                _ => {}
                            }
                        }

                        match accession.as_str() {
                            // MS level
                            "MS:1000511" => {
                                current_ms_level = value.parse().unwrap_or(0);
                            }
                            // Scan start time
                            "MS:1000016" => {
                                current_rt = value.parse().unwrap_or(0.0);
                                // Check if unit is minutes (convert to seconds)
                                for attr in e.attributes().flatten() {
                                    if attr.key.as_ref() == b"unitAccession" {
                                        let unit =
                                            String::from_utf8_lossy(&attr.value).to_string();
                                        if unit == "UO:0000031" {
                                            // minutes
                                            current_rt *= 60.0;
                                        }
                                    }
                                }
                            }
                            // 64-bit float
                            "MS:1000523" if in_binary_data_array => {
                                is_64bit = true;
                            }
                            // 32-bit float
                            "MS:1000521" if in_binary_data_array => {
                                is_64bit = false;
                            }
                            // m/z array
                            "MS:1000514" if in_binary_data_array => {
                                is_mz_array = true;
                            }
                            // intensity array
                            "MS:1000515" if in_binary_data_array => {
                                is_intensity_array = true;
                            }
                            // zlib compression
                            "MS:1000574" if in_binary_data_array => {
                                is_compressed = true;
                            }
                            // no compression
                            "MS:1000576" if in_binary_data_array => {
                                is_compressed = false;
                            }
                            // Isolation window target m/z
                            "MS:1000827" if in_isolation_window => {
                                current_iso_target = value.parse().ok();
                            }
                            // Isolation window lower offset
                            "MS:1000828" if in_isolation_window => {
                                current_iso_lower = value.parse().ok();
                            }
                            // Isolation window upper offset
                            "MS:1000829" if in_isolation_window => {
                                current_iso_upper = value.parse().ok();
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                if in_binary {
                    binary_text.push_str(&e.unescape().unwrap_or_default());
                }
            }
            Ok(Event::End(e)) => {
                let name = e.name();
                let local = name.as_ref();

                match local {
                    b"binary" => {
                        if in_binary && !binary_text.is_empty() {
                            let decoded =
                                decode_binary_array(&binary_text, is_64bit, is_compressed);
                            if is_mz_array {
                                current_mz = decoded;
                            } else if is_intensity_array {
                                current_intensity = decoded;
                            }
                        }
                        in_binary = false;
                    }
                    b"binaryDataArray" => {
                        in_binary_data_array = false;
                    }
                    b"isolationWindow" => {
                        in_isolation_window = false;
                    }
                    b"spectrum" => {
                        if in_spectrum && !current_mz.is_empty() {
                            let spec = Spectrum {
                                scan_id: current_scan_id.clone(),
                                ms_level: current_ms_level,
                                retention_time: current_rt,
                                mz: std::mem::take(&mut current_mz),
                                intensity: std::mem::take(&mut current_intensity),
                                isolation_window_target: current_iso_target,
                                isolation_window_lower: current_iso_lower,
                                isolation_window_upper: current_iso_upper,
                            };

                            match current_ms_level {
                                1 => {
                                    last_ms1_id = Some(spec.scan_id.clone());
                                    ms1_spectra.push(spec);
                                }
                                2 => {
                                    if let Some(ref ms1_id) = last_ms1_id {
                                        ms2_to_ms1_map
                                            .insert(spec.scan_id.clone(), ms1_id.clone());
                                    }
                                    ms2_spectra.push(spec);
                                }
                                _ => {} // Ignore other MS levels
                            }
                        }
                        in_spectrum = false;
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML parse error: {}", e)),
            _ => {}
        }
        buf.clear();
    }

    Ok(MzMLData {
        ms1_spectra,
        ms2_spectra,
        ms2_to_ms1_map,
    })
}
