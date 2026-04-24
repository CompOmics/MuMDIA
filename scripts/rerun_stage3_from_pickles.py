#!/usr/bin/env python3
"""Rerun Stage 3 from saved search pickles to refresh downstream artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mumdia
from data_structures import PickleConfig, SpectraData
from parsers.parser_mzml import get_ms1_mzml
from utilities import pickling as pickling_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Result directory containing df_fragment.pkl, df_psms.pkl, config.pkl, and related Stage 2 artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)

    (
        df_fragment,
        df_psms,
        df_fragment_max,
        df_fragment_max_peptide,
        config,
        dlc_transfer_learn,
        flags,
    ) = pickling_utils.read_variables_from_pickles(dir=result_dir)

    mzml_path = Path(config["sage"]["mzml_paths"][0])
    ms1_dict, ms2_to_ms1_dict, ms2_dict = get_ms1_mzml(str(mzml_path))
    spectra_data = SpectraData(
        ms1_dict=ms1_dict,
        ms2_to_ms1_dict=ms2_to_ms1_dict,
        ms2_dict=ms2_dict,
    )

    pickle_config = PickleConfig(
        read_deeplc=bool(flags.get("read_deeplc_pickle", False)),
        read_ms2pip=bool(flags.get("read_ms2pip_pickle", False)),
        read_correlation=bool(flags.get("read_correlation_pickles", False)),
        write_deeplc=bool(flags.get("write_deeplc_pickle", False)),
        write_ms2pip=bool(flags.get("write_ms2pip_pickle", False)),
        write_correlation=bool(flags.get("write_correlation_pickles", False)),
    )

    mumdia.calculate_features(
        df_psms,
        df_fragment,
        df_fragment_max,
        df_fragment_max_peptide,
        config=config,
        deeplc_model=dlc_transfer_learn,
        pickle_config=pickle_config,
        spectra_data=spectra_data,
    )


if __name__ == "__main__":
    main()
