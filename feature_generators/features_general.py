import polars as pl


def add_count_and_filter_peptides(df_psms: pl.DataFrame, min_occurrences: int = 2):
    """
    Add a count column to the PSM dataframe and filter peptides with less than min_occurrences occurrences.
    Args:
        df_psms: PSM dataframe with the following columns:
            - peptide: Peptide sequence
    """
    peptide_counts = df_psms.group_by("peptide").agg(pl.count().alias("count"))
    df_psms = df_psms.join(peptide_counts, on="peptide").filter(
        pl.col("count") >= min_occurrences
    )
    df_psms = df_psms.sample(n=50000)
    return df_psms
