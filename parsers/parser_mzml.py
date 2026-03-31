"""mzML file parsing: spectrum extraction, MS1/MS2 mapping, and RT-based partitioning."""

import os
import xml.etree.ElementTree as ET

import numpy as np
import pymzml
from pyopenms import MSExperiment, MzMLFile

from utilities.logger import log_info

# Optional Rust-accelerated mzML parser
try:
    import mumdia_rs

    _RUST_MZML = True
except ImportError:
    _RUST_MZML = False


def read_mzml(filename):
    """Read mzML file and return MSExperiment object"""
    exp = MSExperiment()
    MzMLFile().load(filename, exp)
    return exp


def write_mzml(filename, experiment):
    """Write MSExperiment object to an mzML file"""
    MzMLFile().store(filename, experiment)


def get_ms1_mzml(file_path):
    """
    Extract MS1 and MS2 spectra from an mzML file with MS2-to-MS1 mapping.

    Iterates spectra chronologically. Each MS2 spectrum is mapped to the most
    recent preceding MS1 spectrum, enabling precursor intensity lookup.

    Args:
        file_path: Path to the mzML file.

    Returns:
        Tuple of (ms1_spectra, ms2_to_ms1_map, ms2_spectra) where:
        - ms1_spectra: {scan_id: {mz, intensity, retention_time}}
        - ms2_to_ms1_map: {ms2_scan_id: preceding_ms1_scan_id}
        - ms2_spectra: {scan_id: {retention_time, mz, intensity}}
    """
    # Fast path: Rust mzML parser (falls back to PyOpenMS on error)
    if _RUST_MZML:
        try:
            log_info("Using Rust mzML parser (mumdia_rs)")
            return mumdia_rs.parse_mzml_file(str(file_path))
        except (OSError, RuntimeError):
            log_info("Rust mzML parser failed, falling back to PyOpenMS")

    # Fallback: PyOpenMS
    # Create MSExperiment object
    exp = MSExperiment()

    # Load the mzML file
    MzMLFile().load(file_path, exp)

    # Dictionaries to store MS1 and MS2 spectra
    ms1_spectra = {}
    ms2_spectra = {}
    ms2_to_ms1_map = {}

    # Variable to keep track of the last MS1 scan identifier
    last_ms1_id = None

    # Iterate over each spectrum in the file
    for spectrum in exp.getSpectra():
        scan_id = spectrum.getNativeID()
        mz_array, intensity_array = spectrum.get_peaks()
        # Get retention time (in seconds)
        retention_time = spectrum.getRT()

        if spectrum.getMSLevel() == 1:
            # Update MS1 spectra dictionary
            ms1_spectra[scan_id] = {
                "mz": mz_array,
                "intensity": intensity_array,
                "retention_time": retention_time,
            }

            # Update the last MS1 scan identifier
            last_ms1_id = scan_id

        elif spectrum.getMSLevel() == 2 and last_ms1_id is not None:
            # Map current MS2 scan identifier to the last MS1 scan identifier)
            ms2_to_ms1_map[scan_id] = last_ms1_id

            ms2_spectra[scan_id] = {
                "retention_time": retention_time,
                "mz": mz_array,
                "intensity": intensity_array,
            }

    return ms1_spectra, ms2_to_ms1_map, ms2_spectra


def split_mzml_by_retention_time(original_file, dir_files="", time_interval=120.0):
    """
    Split mzML file into time-windowed partitions for targeted searching.

    Iterates spectra chronologically and writes sub-experiments to separate
    mzML files when spectra exceed the current time window boundary.

    Args:
        original_file: Path to the input mzML file.
        dir_files: Base directory for output (partitions go into dir_files/temp/).
        time_interval: Duration of each partition in seconds (dynamically set to
            perc_95 from DeepLC in the main pipeline, not always 120s).

    Returns:
        Dict mapping upper RT bound (float) to partition mzML file path (str).
    """
    dict_mzml_files = {}
    exp = read_mzml(original_file)
    spectra = exp.getSpectra()

    start_time = 0
    end_time = start_time + time_interval
    part = 1
    sub_exp = MSExperiment()

    tempdir = os.path.join(dir_files, "temp")

    for spec in spectra:
        if spec.getRT() <= end_time:
            sub_exp.addSpectrum(spec)
        else:
            sub_dir = f"part_{end_time-time_interval}_{end_time}"
            log_info(f"Writing part {part} to {tempdir}/{sub_dir}...")
            if not os.path.exists(os.path.join(tempdir, sub_dir)):
                os.makedirs(os.path.join(tempdir, sub_dir))

            file_out = os.path.join(
                tempdir, sub_dir, f"part_{end_time-time_interval}_{end_time}.mzml"
            )

            dict_mzml_files[end_time] = file_out
            write_mzml(
                file_out,
                sub_exp,
            )
            part += 1
            start_time = end_time
            end_time += time_interval
            sub_exp = MSExperiment()
            sub_exp.addSpectrum(spec)

    if sub_exp.getNrSpectra() > 0:
        sub_dir = f"part_{end_time-time_interval}_{end_time}"
        if not os.path.exists(os.path.join(tempdir, sub_dir)):
            os.makedirs(os.path.join(tempdir, sub_dir))

        file_out = os.path.join(
            tempdir, sub_dir, f"part_{end_time-time_interval}_{end_time}.mzml"
        )

        dict_mzml_files[end_time] = file_out

        write_mzml(
            file_out,
            sub_exp,
        )

    return dict_mzml_files


def parse_mzml(file_path):
    """
    Parse mzML XML to extract CV parameters for each spectrum.

    Extracts accession values for isolation window (MS:1000827-1000829),
    MS level (MS:1000511), and m/z/intensity arrays (MS:1000514-1000515).

    Args:
        file_path: Path to the mzML file.

    Returns:
        Dict mapping spectrum_id to {spectrum_id, MS:1000827, MS:1000828, ...}.
        Not used in the main pipeline.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    ns = {
        "mzml": "http://psi.hupo.org/ms/mzml",
        "UO": "http://purl.obolibrary.org/obo/",
    }

    # Define the accession numbers
    accessions = [
        "MS:1000827",
        "MS:1000828",
        "MS:1000829",
        "MS:1000511",
        "MS:1000514",
        "MS:1000515",
    ]

    results = {}

    spectrum_list = root.findall(".//mzml:spectrum", namespaces=ns)

    for spectrum in spectrum_list:
        spectrum_id = spectrum.attrib.get("id")
        values_for_spectrum = {"spectrum_id": spectrum_id}

        for accession in accessions:
            xpath = f".//mzml:cvParam[@accession='{accession}']"
            element = spectrum.find(xpath, namespaces=ns)

            if element is not None:
                value = float(element.attrib["value"])
                values_for_spectrum[accession] = value

        results[spectrum_id] = values_for_spectrum

    return results


def get_spectra_mzml(file_path="./LFQ_Orbitrap_AIF_Ecoli_01.mzML"):
    """
    Read MS2 spectra using pymzml and group by isolation window.

    Groups spectra by an identifier combining isolation window lower and upper
    bounds: "{iso_mz - lower_offset}|{iso_mz + upper_offset}".

    Args:
        file_path: Path to the mzML file.

    Returns:
        Tuple of (id_to_vals, group_to_ids) where:
        - id_to_vals: {spectrum_id: [iso_mz, lower, upper, lower_delta, upper_delta, ident, mz_array, intensity_array]}
        - group_to_ids: {ident: [spectrum_id, ...]}
        Not used in the main pipeline.
    """
    run = pymzml.run.Reader(file_path)

    id_to_vals = {}
    group_to_ids = {}

    for spectrum in run:
        spectrum_id = " ".join(
            [str(k) + "=" + str(v) for k, v in spectrum.id_dict.items()]
        )

        mz_values = spectrum.mz
        intensity_values = spectrum.i
        MS_1000827 = spectrum.get("MS:1000827")
        MS_1000828 = spectrum.get("MS:1000828")
        MS_1000829 = spectrum.get("MS:1000829")
        MS_1000511 = spectrum.ms_level

        if not MS_1000827:
            continue
        ident = (
            str(
                round(
                    MS_1000827 - MS_1000828,
                    3,
                )
            )
            + "|"
            + str(
                round(
                    MS_1000827 + MS_1000829,
                    3,
                )
            )
        )

        id_to_vals[spectrum_id] = [
            MS_1000827,
            MS_1000828,
            MS_1000829,
            MS_1000827 - MS_1000828,
            MS_1000827 + MS_1000829,
            ident,
            mz_values,
            intensity_values,
        ]
        try:
            group_to_ids[ident].append(spectrum_id)
        except KeyError:
            group_to_ids[ident] = [spectrum_id]

    return id_to_vals, group_to_ids


if __name__ == "__main__":
    get_spectra_mzml()
