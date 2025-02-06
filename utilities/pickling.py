import pickle
from pathlib import Path
from utilities.io_utils import create_directory


def write_variables_to_pickles(
    df_fragment,
    df_psms,
    df_fragment_max,
    df_fragment_max_peptide,
    config,
    dlc_transfer_learn,
    write_deeplc_pickle,
    write_ms2pip_pickle,
    write_correlation_pickles,
    write_full_search_pickles,
    read_deeplc_pickle,
    read_ms2pip_pickle,
    read_correlation_pickles,
    read_full_search_pickle,
    df_fragment_fname="df_fragment.pkl",
    df_psms_fname="df_psms.pkl",
    df_fragment_max_fname="df_fragment_max.pkl",
    df_fragment_max_peptide_fname="df_fragment_max_peptide.pkl",
    config_fname="config.pkl",
    dlc_transfer_learn_fname="dlc_transfer_learn.pkl",
    flags_fname="flags.pkl",
    dir="./",
):
    pickle_dir = Path(dir)
    create_directory(pickle_dir)

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
                "write_deeplc_pickle": write_deeplc_pickle,
                "write_ms2pip_pickle": write_ms2pip_pickle,
                "write_correlation_pickles": write_correlation_pickles,
                "read_deeplc_pickle": read_deeplc_pickle,
                "read_ms2pip_pickle": read_ms2pip_pickle,
                "write_full_search_pickles": False,
                "read_correlation_pickles": read_correlation_pickles,
                "read_full_search_pickles": False,
            },
            f,
        )


def read_variables_from_pickles(
    dir="./",
):
    pickle_dir = Path(dir)

    with open(pickle_dir.joinpath("df_fragment.pkl"), "rb") as f:
        df_fragment = pickle.load(f)
    with open(pickle_dir.joinpath("df_psms.pkl"), "rb") as f:
        df_psms = pickle.load(f)
    with open(pickle_dir.joinpath("df_fragment_max.pkl"), "rb") as f:
        df_fragment_max = pickle.load(f)
    with open(pickle_dir.joinpath("df_fragment_max_peptide.pkl"), "rb") as f:
        df_fragment_max_peptide = pickle.load(f)
    with open(pickle_dir.joinpath("config.pkl"), "rb") as f:
        config = pickle.load(f)
    with open(pickle_dir.joinpath("dlc_transfer_learn.pkl"), "rb") as f:
        dlc_transfer_learn = pickle.load(f)
    with open(pickle_dir.joinpath("flags.pkl"), "rb") as f:
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
