import os
import xml.etree.ElementTree as ET

import numpy as np
import pymzml
from pyopenms import MSExperiment, MzMLFile


def read_mzml(filename):
    """Read mzML file and return MSExperiment object"""
    exp = MSExperiment()
    MzMLFile().load(filename, exp)
    return exp


def write_mzml(filename, experiment):
    """Write MSExperiment object to an mzML file"""
    MzMLFile().store(filename, experiment)


def get_ms1_mzml(file_path):
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

        if spectrum.getMSLevel() == 1:
            # Extract m/z and intensity values
            mz_array, intensity_array = spectrum.get_peaks()

            # Get retention time (in seconds)
            retention_time = spectrum.getRT()

            # Update MS1 spectra dictionary
            ms1_spectra[scan_id] = {
                "mz": mz_array,
                "intensity": intensity_array,
                "retention_time": retention_time,
            }

            # Update the last MS1 scan identifier
            last_ms1_id = scan_id

        elif spectrum.getMSLevel() == 2 and last_ms1_id is not None:
            # Map current MS2 scan identifier to the last MS1 scan identifier
            retention_time = spectrum.getRT()
            ms2_to_ms1_map[scan_id] = last_ms1_id

            retention_time = spectrum.getRT()

            ms2_spectra[scan_id] = {
                "retention_time": retention_time,
                "mz": mz_array,
                "intensity": intensity_array,
            }

    return ms1_spectra, ms2_to_ms1_map, ms2_spectra


def get_ms1_mzml_old(file_path):
    # Create MSExperiment object
    exp = MSExperiment()

    # Load the mzML file
    MzMLFile().load(file_path, exp)

    # Dictionary to store MS1 spectra
    ms1_spectra = {}

    # Iterate over each spectrum in the file
    for spectrum in exp.getSpectra():
        # Check if the spectrum is MS1
        if spectrum.getMSLevel() == 1:
            # Extract m/z and intensity values
            mzs, intensities = spectrum.get_peaks()

            # Convert to numpy arrays
            mz_array = np.array(mzs)
            intensity_array = np.array(intensities)

            # Get the spectrum identifier
            scan_id = spectrum.getNativeID()

            # Add to dictionary with nested structure
            ms1_spectra[scan_id] = {"mz": mz_array, "intensity": intensity_array}

    return ms1_spectra


def split_mzml_by_retention_time(original_file, dir_files="", time_interval=120.0):
    """Split mzML file into smaller files based on retention time intervals"""
    dict_mzml_files = {}
    exp = read_mzml(original_file)
    spectra = exp.getSpectra()

    start_time = 0
    end_time = start_time + time_interval
    part = 1
    sub_exp = MSExperiment()

    for spec in spectra:
        if spec.getRT() <= end_time:
            sub_exp.addSpectrum(spec)
        else:
            sub_dir = f"part_{end_time-time_interval}_{end_time}"
            if not os.path.exists(os.path.join(dir_files, sub_dir)):
                os.makedirs(os.path.join(dir_files, sub_dir))

            file_out = os.path.join(
                dir_files, sub_dir, f"part_{end_time-time_interval}_{end_time}.mzml"
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
        if not os.path.exists(os.path.join(dir_files, sub_dir)):
            os.makedirs(os.path.join(dir_files, sub_dir))

        file_out = os.path.join(
            dir_files, sub_dir, f"part_{end_time-time_interval}_{end_time}.mzml"
        )

        dict_mzml_files[end_time] = file_out

        write_mzml(
            file_out,
            sub_exp,
        )

    return dict_mzml_files


def parse_mzml(file_path):
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
    run = pymzml.run.Reader(file_path)

    id_to_vals = {}
    group_to_ids = {}

    for spectrum in run:
        # print(dir(spectrum))
        # print(spectrum.id_dict)
        # print(spectrum.id_dict)
        # print(spectrum.index)
        # print(spectrum.to_string())
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


def get_spectra_mzml_old(file_path="./LFQ_Orbitrap_AIF_Ecoli_01.mzML"):
    values = parse_mzml(file_path)

    id_to_vals = {}
    group_to_ids = {}
    for spectrum_id, spectrum_values in values.items():
        try:
            # if spectrum_values.get("MS:1000511") == 1.0:
            #    continue
            ident = (
                str(
                    round(
                        spectrum_values.get("MS:1000827")
                        - spectrum_values.get("MS:1000828"),
                        3,
                    )
                )
                + "|"
                + str(
                    round(
                        spectrum_values.get("MS:1000827")
                        + spectrum_values.get("MS:1000829"),
                        3,
                    )
                )
            )
            id_to_vals[spectrum_id] = [
                spectrum_values.get("MS:1000827"),
                spectrum_values.get("MS:1000828"),
                spectrum_values.get("MS:1000829"),
                spectrum_values.get("MS:1000827") - spectrum_values.get("MS:1000828"),
                spectrum_values.get("MS:1000827") + spectrum_values.get("MS:1000829"),
                ident,
                spectrum_values.get("MS:1000514"),
                spectrum_values.get("MS:1000515"),
            ]
            try:
                group_to_ids[ident].append(spectrum_id)
            except KeyError:
                group_to_ids[ident] = [spectrum_id]
        except:
            continue

    return id_to_vals, group_to_ids


if __name__ == "__main__":
    get_spectra_mzml()
