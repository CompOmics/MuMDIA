import polars as pl


def add_count_and_filter_peptides(df_psms: pl.DataFrame, min_occurrences: int = 2) -> pl.DataFrame:
    """
    Add peptide occurrence counts and filter by minimum frequency.
    
    This function counts how many times each peptide appears in the dataset
    and filters out peptides that occur less than the minimum threshold.
    This helps remove unreliable identifications.
    
    Args:
        df_psms: PSM DataFrame with 'peptide' column
        min_occurrences: Minimum number of times a peptide must be observed (default: 2)
        
    Returns:
        Filtered DataFrame with 'count' column added, containing only peptides
        that meet the minimum occurrence threshold
    """
    peptide_counts = df_psms.group_by("peptide").agg(pl.count().alias("count"))
    df_psms = df_psms.join(peptide_counts, on="peptide").filter(
        pl.col("count") >= min_occurrences
    )
    return df_psms
