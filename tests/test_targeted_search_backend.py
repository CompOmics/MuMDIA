"""Tests for the stage-2 targeted-search backend dispatcher."""

from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest

from peptide_search.search_backend import run_targeted_search_backend


@pytest.mark.unit
@patch("peptide_search.search_backend.retention_window_searches_custom")
@patch("peptide_search.search_backend.retention_window_searches")
def test_dispatches_to_sage(mock_sage, mock_custom):
    empty_result = (pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), pl.DataFrame())
    mock_sage.return_value = empty_result

    result = run_targeted_search_backend(
        "sage",
        {10.0: "part.mzML"},
        pd.DataFrame({"peptide": ["PEPTIDE"]}),
        {"sage": {}},
        120.0,
        backend_context={"unused": True},
    )

    assert result == empty_result
    mock_sage.assert_called_once()
    mock_custom.assert_not_called()


@pytest.mark.unit
@patch("peptide_search.search_backend.retention_window_searches_custom")
@patch("peptide_search.search_backend.retention_window_searches")
def test_dispatches_to_custom(mock_sage, mock_custom):
    empty_result = (pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), pl.DataFrame())
    mock_custom.return_value = empty_result

    result = run_targeted_search_backend(
        "custom",
        {10.0: "part.mzML"},
        pd.DataFrame({"peptide": ["PEPTIDE"]}),
        {"sage": {}},
        120.0,
        backend_context={"ms2pip_predictions": {"PEPTIDE/2": {"b2/1": 10.0}}},
    )

    assert result == empty_result
    mock_custom.assert_called_once()
    assert mock_custom.call_args.kwargs["backend_context"] == {
        "ms2pip_predictions": {"PEPTIDE/2": {"b2/1": 10.0}}
    }
    mock_sage.assert_not_called()


@pytest.mark.unit
def test_dispatch_rejects_unknown_backend():
    with pytest.raises(ValueError):
        run_targeted_search_backend(
            "unknown",
            {10.0: "part.mzML"},
            pd.DataFrame({"peptide": ["PEPTIDE"]}),
            {"sage": {}},
            120.0,
        )
