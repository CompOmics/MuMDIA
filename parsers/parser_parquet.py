import pandas as pd
import polars as pl


def replace_mass_shift(
    peptide,
    replace_dict={
        "[+57.0215]": "[Carbamidomethyl]",
        "[+57.021465]": "[Carbamidomethyl]",
        "[+15.9949]": "[Oxidation]",
        "[-18.010565]": "[Glu->pyro-Glu]",
        "[-17.026548]": "[Gln->pyro-Glu]",
        "[+0.984016]": "[Deamidated]",
        "[+14.01565]": "[Methyl]",
        "[+27.994915]": "[Formyl]",
        "[+28.0313]": "[Dimethyl]",
        "[+79.96633]": "[Phospho]",
        "[+31.989829]": "[Dioxidation]",
        "[+31.989828]": "[Dioxidation]",
        "[+42.010565]": "[Acetyl]",
        "[+42.010567]": "[Acetyl]",
        "[+12.0000000]": "[Thiazolidine]",
        "[+12.000000]": "[Thiazolidine]",
        "[+12.00000]": "[Thiazolidine]",
        "[+12.0000]": "[Thiazolidine]",
        "[+12.000]": "[Thiazolidine]",
        "[+12.00]": "[Thiazolidine]",
        "[+12.0]": "[Thiazolidine]",
        "[-18.010565]": "[Glu->pyro-Glu]",
        "[-17.026549]": "[Gln->pyro-Glu]",
        "[-17.026549]": "[Gln->pyro-Glu]",
        "[+17.026549]": "[Ammonium]",
        "[+44.985078]": "[Nitro]",
        "[+44.985077]": "[Nitro]",
        "[+43.005814]": "[Carbamyl]",
        "[+114.042927]": "[GG]",
        "[+114.04293]": "[GG]",
        "[+114.03169]": "[Gluratylation]",
        "[+56.026215]": "[Delta:H(4)C(3)O(1)]",
        "[+71.03712]": "[Propionamide]",
    },
):
    for k, v in replace_dict.items():
        peptide = peptide.replace(k, v)
    return peptide


def parquet_reader(
    parquet_file_results="results.sage.parquet",
    parquet_file_fragments="matched_fragments.sage.parquet",
    q_value_filter=1.0,
):
    df_fragment = pd.read_parquet(parquet_file_fragments)
    df_fragment.index = df_fragment["psm_id"]

    df_psms = pd.read_parquet(parquet_file_results)
    df_psms.drop_duplicates(subset=["scannr", "peptide"], inplace=True)

    df_psms = df_psms[df_psms["spectrum_q"] < q_value_filter]
    df_fragment = df_fragment[df_fragment.index.isin(df_psms["psm_id"])]

    df_fragment = pl.DataFrame(df_fragment)
    df_psms = pl.DataFrame(df_psms)

    if len(df_fragment["psm_id"]) == 0:
        return None, None, None, None

    # df_fragment_max.index = df_fragment_max["psm_id"]
    # df_psms.index = df_psms["psm_id"]
    # df_psms = pd.concat([df_psms, df_fragment_max["fragment_intensity"]], axis=1)

    df_psms = df_psms.with_columns(
        pl.col("peptide").map_elements(replace_mass_shift).alias("peptide")
    )

    df_fragment = df_fragment.join(
        df_psms[["psm_id", "peptide", "charge", "rt"]], on="psm_id", how="left"
    )

    # Will return random order of max fragment intensity
    df_fragment_max = df_fragment.sort("fragment_intensity", descending=True).unique(
        subset="psm_id", keep="first", maintain_order=True
    )

    # might also want to do on charge
    df_fragment_max_peptide = df_fragment_max.unique(
        subset=["peptide"],
        keep="first",  # , "charge"
    )

    # df_psms = pd.concat([df_psms, df_fragment_max["fragment_intensity"]], axis=1)

    df_psms = df_psms.join(
        df_fragment_max[["psm_id", "fragment_intensity"]],
        on="psm_id",
        how="left",
    )

    return df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide
