# Script to run the feature extraction for fragment intensity independently
import pickle
import polars as pl
from feature_generators.features_fragment_intensity import (
    get_features_fragment_intensity,
)


def main():

    with open("debug/ms2pip_predictions.pkl", "rb") as f:
        ms2pip_predictions = pickle.load(f)
    with open("debug/ms2dict.pkl", "rb") as f:
        ms2dict = pickle.load(f)

    df_fragment = pl.read_csv("debug/df_fragment_after_ms2pip.tsv", separator="\t")
    df_fragment_max_peptide = pl.read_csv(
        "debug/df_fragment_max_peptide_after_ms2pip.tsv", separator="\t"
    )

    fragment_dict, correlations_fragment_dict = get_features_fragment_intensity(
        ms2pip_predictions=ms2pip_predictions,
        df_fragment=df_fragment,
        df_fragment_max_peptide=df_fragment_max_peptide,
        read_correlation_pickles=False,  # Set to False to force computation
        write_correlation_pickles=True,  # Set to True to save results
        ms2_dict=ms2dict,
        output_dir="debug/separate_fragment_intensity_output",
    )

    # log some information about the results
    print(f"Fragment dict: {len(fragment_dict)} entries")
    print(f"Correlations fragment dict: {len(correlations_fragment_dict)} entries")


if __name__ == "__main__":
    main()
