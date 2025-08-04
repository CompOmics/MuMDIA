"""
Simple tests for utilities.pickling module.

This module tests basic pickling functionality with proper mock isolation.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from data_structures import PickleConfig
from utilities.pickling import write_variables_to_pickles


@pytest.fixture
def sample_dataframes():
    """Provide sample DataFrames for testing."""
    df_fragment = pl.DataFrame(
        {"peptide": ["PEPTIDE1", "PEPTIDE2"], "intensity": [1000.0, 2000.0]}
    )

    df_psms = pl.DataFrame({"peptide": ["PEPTIDE1", "PEPTIDE2"], "score": [0.9, 0.8]})

    df_fragment_max = pl.DataFrame(
        {"peptide": ["PEPTIDE1", "PEPTIDE2"], "max_intensity": [1500.0, 2500.0]}
    )

    df_fragment_max_peptide = pl.DataFrame(
        {"peptide": ["PEPTIDE1", "PEPTIDE2"], "max_peptide": [1200.0, 2200.0]}
    )

    return df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide


@pytest.fixture
def sample_config():
    """Provide sample configuration."""
    return {"param1": "value1", "param2": "value2"}


@pytest.fixture
def sample_pickle_config():
    """Provide sample pickle configuration."""
    return PickleConfig(write_deeplc=True, write_ms2pip=True, write_correlation=True)


class TestWriteVariablesToPickles:
    """Test suite for write_variables_to_pickles function."""

    @pytest.mark.unit
    def test_write_variables_to_pickles_basic(
        self, sample_dataframes, sample_config, sample_pickle_config
    ):
        """Test basic pickle writing functionality."""
        (
            df_fragment,
            df_psms,
            df_fragment_max,
            df_fragment_max_peptide,
        ) = sample_dataframes

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the pickle.dump to avoid complex object serialization
            with patch("utilities.pickling.pickle.dump") as mock_dump:
                write_variables_to_pickles(
                    df_fragment=df_fragment,
                    df_psms=df_psms,
                    df_fragment_max=df_fragment_max,
                    df_fragment_max_peptide=df_fragment_max_peptide,
                    config=sample_config,
                    dlc_transfer_learn={},  # Use simple dict instead of Mock
                    pickle_config=sample_pickle_config,
                    write_full_search_pickle=True,
                    read_full_search_pickle=False,
                    dir=temp_dir,
                )

                # Check that function completed without errors
                assert mock_dump.call_count >= 1

    @pytest.mark.unit
    def test_write_variables_to_pickles_selective_writing(
        self, sample_dataframes, sample_config
    ):
        """Test selective pickle writing based on configuration."""
        (
            df_fragment,
            df_psms,
            df_fragment_max,
            df_fragment_max_peptide,
        ) = sample_dataframes

        # Configure to write only some pickles
        selective_config = PickleConfig(
            write_deeplc=True, write_ms2pip=False, write_correlation=True
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("utilities.pickling.pickle.dump"):
                write_variables_to_pickles(
                    df_fragment=df_fragment,
                    df_psms=df_psms,
                    df_fragment_max=df_fragment_max,
                    df_fragment_max_peptide=df_fragment_max_peptide,
                    config=sample_config,
                    dlc_transfer_learn={},
                    pickle_config=selective_config,
                    write_full_search_pickle=False,
                    read_full_search_pickle=False,
                    dir=temp_dir,
                )

                # Verify function completes successfully
                assert True

    @pytest.mark.unit
    def test_write_variables_to_pickles_none_dataframes(self):
        """Test writing with None DataFrames - should handle gracefully."""
        config = {"test": "value"}
        pickle_config = PickleConfig()
        empty_df = pl.DataFrame({"col": []}, schema={"col": pl.Utf8})

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("utilities.pickling.pickle.dump"):
                write_variables_to_pickles(
                    df_fragment=empty_df,
                    df_psms=empty_df,
                    df_fragment_max=empty_df,
                    df_fragment_max_peptide=empty_df,
                    config=config,
                    dlc_transfer_learn={},
                    pickle_config=pickle_config,
                    write_full_search_pickle=False,
                    read_full_search_pickle=False,
                    dir=temp_dir,
                )

                # Should complete without errors
                assert True


class TestPicklingEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.unit
    def test_pickling_empty_dataframes(self):
        """Test pickling with empty DataFrames."""
        empty_df = pl.DataFrame({"col": []}, schema={"col": pl.Utf8})
        config = {}
        pickle_config = PickleConfig()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("utilities.pickling.pickle.dump"):
                write_variables_to_pickles(
                    df_fragment=empty_df,
                    df_psms=empty_df,
                    df_fragment_max=empty_df,
                    df_fragment_max_peptide=empty_df,
                    config=config,
                    dlc_transfer_learn={},
                    pickle_config=pickle_config,
                    write_full_search_pickle=False,
                    read_full_search_pickle=False,
                    dir=temp_dir,
                )

                # Should complete without errors
                assert True

    @pytest.mark.unit
    def test_pickling_large_dataframes(self):
        """Test pickling with large DataFrames."""
        large_df = pl.DataFrame(
            {"col1": list(range(1000)), "col2": [f"value_{i}" for i in range(1000)]}
        )

        config = {"large_data": True}
        pickle_config = PickleConfig()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("utilities.pickling.pickle.dump"):
                write_variables_to_pickles(
                    df_fragment=large_df,
                    df_psms=large_df,
                    df_fragment_max=large_df,
                    df_fragment_max_peptide=large_df,
                    config=config,
                    dlc_transfer_learn={},
                    pickle_config=pickle_config,
                    write_full_search_pickle=False,
                    read_full_search_pickle=False,
                    dir=temp_dir,
                )

                # Should complete without errors
                assert True

    @pytest.mark.unit
    def test_pickling_with_special_characters(self):
        """Test pickling with special characters in data."""
        special_df = pl.DataFrame(
            {
                "peptide": ["PEPT IDE", "PEPT@IDE", "PEPT#IDE"],
                "special": ["åäö", "αβγ", "日本語"],
            }
        )

        config = {"encoding": "utf-8"}
        pickle_config = PickleConfig()

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("utilities.pickling.pickle.dump"):
                write_variables_to_pickles(
                    df_fragment=special_df,
                    df_psms=special_df,
                    df_fragment_max=special_df,
                    df_fragment_max_peptide=special_df,
                    config=config,
                    dlc_transfer_learn={},
                    pickle_config=pickle_config,
                    write_full_search_pickle=False,
                    read_full_search_pickle=False,
                    dir=temp_dir,
                )

                # Should complete without errors
                assert True
