"""Dispatch targeted RT-partition searches to the configured backend."""

from typing import Any, Dict, Tuple

import pandas as pd
import polars as pl

from peptide_search.custom_engine import retention_window_searches_custom
from peptide_search.wrapper_sage import retention_window_searches
from utilities.logger import log_info

TargetedSearchResult = Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]


def run_targeted_search_backend(
    backend: str,
    mzml_dict: Dict[float, str],
    peptide_df: pd.DataFrame,
    config: Dict[str, Any],
    rt_split_window: float,
    backend_context: Dict[str, Any] | None = None,
) -> TargetedSearchResult:
    """Run RT-windowed stage-2 searches using the selected backend."""
    normalized_backend = str(backend or "sage").strip().lower()
    log_info(f"Stage-2 search backend: {normalized_backend}")

    if normalized_backend == "sage":
        return retention_window_searches(mzml_dict, peptide_df, config, rt_split_window)
    if normalized_backend == "custom":
        return retention_window_searches_custom(
            mzml_dict,
            peptide_df,
            config,
            rt_split_window,
            backend_context=backend_context,
        )

    raise ValueError(
        f"Unsupported stage-2 search backend '{backend}'. Expected 'sage' or 'custom'."
    )
