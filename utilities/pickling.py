import pickle
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import polars as pl

from data_structures import PickleConfig
from utilities.io_utils import create_directory


def write_variables_to_pickles(
    df_fragment: pl.DataFrame,
    df_psms: pl.DataFrame,
    df_fragment_max: pl.DataFrame,
    df_fragment_max_peptide: pl.DataFrame,
    config: dict,
    dlc_transfer_learn: Any,
    pickle_config: PickleConfig,
    write_full_search_pickle: bool,
    read_full_search_pickle: bool,
    df_fragment_fname: str = "df_fragment.pkl",
    df_psms_fname: str = "df_psms.pkl",
    df_fragment_max_fname: str = "df_fragment_max.pkl",
    df_fragment_max_peptide_fname: str = "df_fragment_max_peptide.pkl",
    config_fname: str = "config.pkl",
    dlc_transfer_learn_fname: str = "dlc_transfer_learn.pkl",
    flags_fname: str = "flags.pkl",
    dir: Union[str, Path] = "./",
    write_to_tsv: bool = False,
) -> None:
    """
    Serialize DataFrames and configuration to pickle files for caching.

    This function saves all workflow state to pickle files for resuming processing
    and optionally exports DataFrames to TSV format for inspection.

    Args:
        df_fragment: Fragment matches DataFrame
        df_psms: PSM results DataFrame
        df_fragment_max: Maximum intensity fragments per PSM
        df_fragment_max_peptide: Maximum intensity fragments per peptide
        config: Configuration dictionary
        dlc_transfer_learn: Trained DeepLC model
        write_deeplc_pickle: Flag for DeepLC pickle operations
        write_ms2pip_pickle: Flag for MS2PIP pickle operations
        write_correlation_pickles: Flag for correlation pickle operations
        write_full_search_pickle: Flag for full search pickle operations
        read_deeplc_pickle: Flag for DeepLC pickle reading
        read_ms2pip_pickle: Flag for MS2PIP pickle reading
        read_correlation_pickles: Flag for correlation pickle reading
        read_full_search_pickle: Flag for full search pickle reading
        df_fragment_fname: Filename for fragment DataFrame pickle
        df_psms_fname: Filename for PSM DataFrame pickle
        df_fragment_max_fname: Filename for max fragment DataFrame pickle
        df_fragment_max_peptide_fname: Filename for max peptide DataFrame pickle
        config_fname: Filename for configuration pickle
        dlc_transfer_learn_fname: Filename for DeepLC model pickle
        flags_fname: Filename for processing flags pickle
        dir: Output directory for pickle files
        write_to_tsv: Whether to also export DataFrames as TSV files
    """
    pickle_dir = Path(dir)
    create_directory(pickle_dir)

    if write_to_tsv:
        if df_fragment is not None:
            df_fragment.write_csv(
                pickle_dir.joinpath(df_fragment_fname.replace(".pkl", ".tsv")),
                separator="\t",
            )
        if df_psms is not None:
            df_psms.write_csv(
                pickle_dir.joinpath(df_psms_fname.replace(".pkl", ".tsv")),
                separator="\t",
            )
        if df_fragment_max is not None:
            df_fragment_max.write_csv(
                pickle_dir.joinpath(df_fragment_max_fname.replace(".pkl", ".tsv")),
                separator="\t",
            )
        if df_fragment_max_peptide is not None:
            df_fragment_max_peptide.write_csv(
                pickle_dir.joinpath(
                    df_fragment_max_peptide_fname.replace(".pkl", ".tsv")
                ),
                separator="\t",
            )

    with open(pickle_dir.joinpath(df_fragment_fname), "wb") as f:
        pickle.dump(df_fragment, f)

    with open(pickle_dir.joinpath(df_psms_fname), "wb") as f:
        pickle.dump(df_psms, f)
    with open(pickle_dir.joinpath(df_fragment_max_fname), "wb") as f:
        pickle.dump(df_fragment_max, f)
    with open(pickle_dir.joinpath(df_fragment_max_peptide_fname), "wb") as f:
        pickle.dump(df_fragment_max_peptide, f)
    with open(pickle_dir.joinpath(config_fname), "wb") as f:
        pickle.dump(config, f)
    with open(pickle_dir.joinpath(dlc_transfer_learn_fname), "wb") as f:
        pickle.dump(dlc_transfer_learn, f)
    # Also save the flags
    with open(pickle_dir.joinpath(flags_fname), "wb") as f:
        pickle.dump(
            {
                "write_deeplc_pickle": pickle_config.write_deeplc,
                "write_ms2pip_pickle": pickle_config.write_ms2pip,
                "write_correlation_pickles": pickle_config.write_correlation,
                "read_deeplc_pickle": pickle_config.read_deeplc,
                "read_ms2pip_pickle": pickle_config.read_ms2pip,
                "write_full_search_pickle": write_full_search_pickle,
                "read_correlation_pickles": pickle_config.read_correlation,
                "read_full_search_pickle": read_full_search_pickle,
            },
            f,
        )


def read_variables_from_pickles(
    dir: Union[str, Path] = "./",
    df_fragment_fname: str = "df_fragment.pkl",
    df_psms_fname: str = "df_psms.pkl",
    df_fragment_max_fname: str = "df_fragment_max.pkl",
    df_fragment_max_peptide_fname: str = "df_fragment_max_peptide.pkl",
    config_fname: str = "config.pkl",
    dlc_transfer_learn_fname: str = "dlc_transfer_learn.pkl",
    flags_fname: str = "flags.pkl",
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict, Any, dict]:
    """
    Deserialize cached DataFrames and configuration from pickle files.

    This function loads all workflow state from pickle files to resume processing
    from a previously saved checkpoint.

    Args:
        dir: Directory containing pickle files
        df_fragment_fname: Filename for fragment DataFrame pickle
        df_psms_fname: Filename for PSM DataFrame pickle
        df_fragment_max_fname: Filename for max fragment DataFrame pickle
        df_fragment_max_peptide_fname: Filename for max peptide DataFrame pickle
        config_fname: Filename for configuration pickle
        dlc_transfer_learn_fname: Filename for DeepLC model pickle
        flags_fname: Filename for processing flags pickle

    Returns:
        Tuple containing:
        - df_fragment: Fragment matches DataFrame
        - df_psms: PSM results DataFrame
        - df_fragment_max: Maximum intensity fragments per PSM
        - df_fragment_max_peptide: Maximum intensity fragments per peptide
        - config: Configuration dictionary
        - dlc_transfer_learn: Trained DeepLC model
        - flags: Processing flags dictionary
    """
    pickle_dir = Path(dir)

    with open(pickle_dir.joinpath(df_fragment_fname), "rb") as f:
        df_fragment = pickle.load(f)
    with open(pickle_dir.joinpath(df_psms_fname), "rb") as f:
        df_psms = pickle.load(f)
    with open(pickle_dir.joinpath(df_fragment_max_fname), "rb") as f:
        df_fragment_max = pickle.load(f)
    with open(pickle_dir.joinpath(df_fragment_max_peptide_fname), "rb") as f:
        df_fragment_max_peptide = pickle.load(f)
    with open(pickle_dir.joinpath(config_fname), "rb") as f:
        config = pickle.load(f)
    with open(pickle_dir.joinpath(dlc_transfer_learn_fname), "rb") as f:
        dlc_transfer_learn = pickle.load(f)
    with open(pickle_dir.joinpath(flags_fname), "rb") as f:
        flags = pickle.load(f)

    return (
        df_fragment,
        df_psms,
        df_fragment_max,
        df_fragment_max_peptide,
        config,
        dlc_transfer_learn,
        flags,
    )
