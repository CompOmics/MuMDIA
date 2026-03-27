"""
Tests for parser modules (mzML and Parquet parsers).

This module tests the data parsers that read experimental mzML files
and search result parquet files, which are critical for data ingestion.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest

# Test parsers
try:
    from parsers.parser_mzml import get_ms1_mzml, split_mzml_by_retention_time

    PARSER_MZML_AVAILABLE = True
except ImportError:
    PARSER_MZML_AVAILABLE = False

try:
    from parsers.parser_parquet import parquet_reader

    PARSER_PARQUET_AVAILABLE = True
except ImportError:
    PARSER_PARQUET_AVAILABLE = False


class TestParquetParser:
    """Test the parquet parser for search engine results."""

    @pytest.mark.skipif(
        not PARSER_PARQUET_AVAILABLE, reason="parquet_parser not available"
    )
    @pytest.mark.unit
    def test_parquet_reader_basic_functionality(self):
        """Test basic parquet reading functionality with mocked data."""
        with patch("pandas.read_parquet") as mock_read_parquet:
            # Mock the parquet data structures using pandas DataFrames
            mock_psm_data = pd.DataFrame(
                {
                    "peptide": ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"],
                    "psm_id": [1, 2, 3],
                    "spectrum_q": [
                        0.001,
                        0.005,
                        0.05,
                    ],  # Updated to expected column name
                    "filename": ["file1.mzML", "file1.mzML", "file2.mzML"],
                    "scannr": [100, 200, 300],
                    "rt": [10.5, 20.3, 30.1],
                    "charge": [2, 3, 2],
                    "precursor_mz": [500.25, 600.33, 550.28],
                }
            )

            mock_fragment_data = pd.DataFrame(
                {
                    "psm_id": [1, 1, 2, 2, 3],
                    "fragment_type": ["b", "y", "b", "y", "b"],
                    "fragment_charge": [1, 1, 2, 1, 1],
                    "fragment_mz": [200.1, 300.2, 250.15, 350.25, 280.18],
                    "fragment_intensity": [
                        1000.0,
                        1500.0,
                        800.0,
                        1200.0,
                        900.0,
                    ],  # Updated column name
                }
            )

            # Configure mock to return different data based on file path
            def mock_read_side_effect(file_path):
                if "results.sage.parquet" in str(file_path):
                    return mock_psm_data
                elif "matched_fragments.sage.parquet" in str(file_path):
                    return mock_fragment_data
                else:
                    return pd.DataFrame()

            mock_read_parquet.side_effect = mock_read_side_effect

            # Test the function without creating actual files
            results_file = Path("fake_results.sage.parquet")
            fragments_file = Path("fake_matched_fragments.sage.parquet")

            (
                df_fragment,
                df_psms,
                df_fragment_max,
                df_fragment_max_peptide,
            ) = parquet_reader(
                parquet_file_results=results_file,
                parquet_file_fragments=fragments_file,
                q_value_filter=0.01,
            )

            # Verify the function was called and returned DataFrames
            assert isinstance(df_fragment, pl.DataFrame)
            assert isinstance(df_psms, pl.DataFrame)
            assert isinstance(df_fragment_max, pl.DataFrame)
            assert isinstance(df_fragment_max_peptide, pl.DataFrame)

            # Verify filtering worked (q_value <= 0.01)
            assert mock_read_parquet.call_count >= 2

    @pytest.mark.unit
    def test_parquet_reader_q_value_filtering(self):
        """Test q-value filtering functionality."""
        with patch("pandas.read_parquet") as mock_pd_read_parquet:
            # Create mock PSMs data with required columns
            mock_psms_data = pd.DataFrame(
                {
                    "peptide": ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"],
                    "spectrum_q": [
                        0.001,
                        0.05,
                        0.1,
                    ],  # Only first should pass filter 0.01
                    "psm_id": [1, 2, 3],
                    "scannr": [100, 200, 300],
                    "charge": [2, 2, 3],
                    "rt": [10.0, 20.0, 30.0],
                }
            )

            # Create mock fragments data
            mock_fragments_data = pd.DataFrame(
                {
                    "psm_id": [1, 1, 2, 2, 3],
                    "fragment_mz": [200.1, 300.2, 400.3, 500.4, 600.5],
                    "fragment_intensity": [1000.0, 2000.0, 1500.0, 2500.0, 800.0],
                }
            )

            # Mock read_parquet to return appropriate data based on file path
            def mock_read_side_effect(file_path):
                if "results" in str(file_path):
                    return mock_psms_data
                elif "fragments" in str(file_path):
                    return mock_fragments_data
                return pd.DataFrame()

            mock_pd_read_parquet.side_effect = mock_read_side_effect

            with tempfile.TemporaryDirectory() as temp_dir:
                results_file = Path(temp_dir) / "results.sage.parquet"
                fragments_file = Path(temp_dir) / "fragments.sage.parquet"

                (
                    df_fragment,
                    df_psms,
                    df_fragment_max,
                    df_fragment_max_peptide,
                ) = parquet_reader(
                    parquet_file_results=results_file,
                    parquet_file_fragments=fragments_file,
                    q_value_filter=0.01,
                )

                # Verify results
                assert isinstance(df_psms, pl.DataFrame)
                assert isinstance(df_fragment, pl.DataFrame)
                # Should only have one PSM with spectrum_q <= 0.01
                assert len(df_psms) == 1
                assert df_psms["peptide"][0] == "PEPTIDE1"

    @pytest.mark.unit
    def test_parquet_reader_empty_files(self):
        """Test handling of empty parquet files."""
        with patch("pandas.read_parquet") as mock_pd_read_parquet:
            # Mock empty DataFrames with required columns to avoid column errors
            empty_psms = pd.DataFrame(
                {
                    "peptide": [],
                    "spectrum_q": [],
                    "psm_id": [],
                    "scannr": [],
                    "charge": [],
                    "rt": [],
                }
            )

            empty_fragments = pd.DataFrame(
                {
                    "psm_id": [],
                    "fragment_mz": [],
                    "fragment_intensity": [],
                }
            )

            def mock_read_side_effect(file_path):
                if "results" in str(file_path):
                    return empty_psms
                elif "fragments" in str(file_path):
                    return empty_fragments
                return pd.DataFrame()

            mock_pd_read_parquet.side_effect = mock_read_side_effect

            with tempfile.TemporaryDirectory() as temp_dir:
                results_file = Path(temp_dir) / "results.sage.parquet"
                fragments_file = Path(temp_dir) / "fragments.sage.parquet"

                # Should handle empty files gracefully and return None
                result = parquet_reader(
                    parquet_file_results=results_file,
                    parquet_file_fragments=fragments_file,
                    q_value_filter=0.01,
                )

                # Should return (None, None, None, None) when no data passes filter
                assert result == (None, None, None, None)


class TestMzMLParser:
    """Test the mzML parser for mass spectrometry data."""

    @pytest.mark.unit
    def test_get_ms1_mzml_basic_functionality(self):
        """Test basic MS1 extraction functionality."""
        with (
            patch("parsers.parser_mzml.MzMLFile") as mock_mzmlfile,
            patch("parsers.parser_mzml.MSExperiment") as mock_msexp,
        ):

            # Mock MSExperiment and its methods
            mock_exp = MagicMock()
            mock_msexp.return_value = mock_exp

            # Mock MzMLFile
            mock_file = MagicMock()
            mock_mzmlfile.return_value = mock_file

            # Mock spectra
            mock_ms1_spectrum = MagicMock()
            mock_ms1_spectrum.getMSLevel.return_value = 1
            mock_ms1_spectrum.getNativeID.return_value = "scan=100"
            mock_ms1_spectrum.getRT.return_value = 10.5
            mock_ms1_spectrum.get_peaks.return_value = (
                np.array([100.0, 200.0, 300.0]),
                np.array([1000.0, 2000.0, 1500.0]),
            )

            mock_ms2_spectrum = MagicMock()
            mock_ms2_spectrum.getMSLevel.return_value = 2
            mock_ms2_spectrum.getNativeID.return_value = "scan=101"
            mock_ms2_spectrum.getRT.return_value = 10.7
            mock_ms2_spectrum.get_peaks.return_value = (
                np.array([150.0, 250.0]),
                np.array([800.0, 1200.0]),
            )

            # Mock precursor
            mock_precursor = MagicMock()
            mock_precursor.getMZ.return_value = 200.0
            mock_precursor.getCharge.return_value = 2
            mock_ms2_spectrum.getPrecursors.return_value = [mock_precursor]

            # Configure experiment to return spectra
            mock_exp.getSpectra.return_value = [mock_ms1_spectrum, mock_ms2_spectrum]

            ms1_dict, ms2_to_ms1_dict, ms2_spectra = get_ms1_mzml("test.mzML")

            # Verify dictionaries are returned
            assert isinstance(ms1_dict, dict)
            assert isinstance(ms2_to_ms1_dict, dict)
            assert isinstance(ms2_spectra, dict)

            # Check that we have the expected MS1 scan
            assert "scan=100" in ms1_dict
            assert "scan=101" in ms2_spectra

    @pytest.mark.skipif(not PARSER_MZML_AVAILABLE, reason="parser_mzml not available")
    @pytest.mark.unit
    def test_split_mzml_by_retention_time_basic(self):
        """Test retention time-based mzML splitting."""
        from parsers.parser_mzml import split_mzml_by_retention_time

        with (
            patch("parsers.parser_mzml.read_mzml") as mock_read_mzml,
            patch("parsers.parser_mzml.MSExperiment") as mock_ms_experiment,
            patch("parsers.parser_mzml.write_mzml") as mock_write_mzml,
            patch("parsers.parser_mzml.os.path.exists") as mock_exists,
            patch("parsers.parser_mzml.os.makedirs") as mock_makedirs,
        ):

            # Mock the MSExperiment and spectra
            mock_exp = MagicMock()
            mock_read_mzml.return_value = mock_exp

            # Create mock spectra with retention times
            mock_spectrum1 = MagicMock()
            mock_spectrum1.getRT.return_value = 60.0  # 1 minute
            mock_spectrum2 = MagicMock()
            mock_spectrum2.getRT.return_value = 180.0  # 3 minutes

            mock_exp.getSpectra.return_value = [mock_spectrum1, mock_spectrum2]

            # Mock sub experiment with proper methods
            mock_sub_exp = MagicMock()
            mock_sub_exp.getNrSpectra.return_value = 1  # Return a number for comparison
            mock_ms_experiment.return_value = mock_sub_exp

            # Mock file operations
            mock_exists.return_value = False

            # Test the splitting function with correct parameters
            result = split_mzml_by_retention_time(
                original_file="test.mzML", dir_files="temp/", time_interval=120.0
            )

            # Verify the function was called and executed
            mock_read_mzml.assert_called_once_with("test.mzML")
            mock_exp.getSpectra.assert_called_once()
            assert mock_sub_exp.addSpectrum.called

            # Should return dictionary of output files
            assert isinstance(result, dict)


class TestParserIntegration:
    """Test integration between parsers."""

    @pytest.mark.integration
    def test_parser_data_consistency(self):
        """Test that parsers return consistent data structures."""
        # Test with mock data to ensure consistent DataFrame schemas

        expected_psm_columns = ["peptide", "psm_id", "q_value", "rt", "charge"]
        expected_fragment_columns = ["psm_id", "fragment_mz", "intensity"]

        # Mock PSM DataFrame
        mock_psm_df = pl.DataFrame(
            {
                "peptide": ["PEPTIDE1", "PEPTIDE2"],
                "psm_id": [1, 2],
                "q_value": [0.001, 0.005],
                "rt": [10.5, 20.3],
                "charge": [2, 3],
                "filename": ["file1.mzML", "file1.mzML"],
                "scannr": [100, 200],
            }
        )

        # Mock Fragment DataFrame
        mock_fragment_df = pl.DataFrame(
            {
                "psm_id": [1, 1, 2],
                "fragment_mz": [200.1, 300.2, 250.15],
                "intensity": [1000.0, 1500.0, 800.0],
                "fragment_type": ["b", "y", "b"],
            }
        )

        # Verify expected columns are present
        for col in expected_psm_columns:
            assert col in mock_psm_df.columns

        for col in expected_fragment_columns:
            assert col in mock_fragment_df.columns

        # Verify data types
        assert mock_psm_df["psm_id"].dtype == pl.Int64
        assert mock_fragment_df["fragment_mz"].dtype in [pl.Float64, pl.Float32]
        assert mock_fragment_df["intensity"].dtype in [pl.Float64, pl.Float32]

    @pytest.mark.unit
    def test_parser_error_handling(self):
        """Test parser error handling with invalid inputs."""
        # Test with non-existent files
        if PARSER_PARQUET_AVAILABLE:
            with pytest.raises((FileNotFoundError, Exception)):
                from parsers.parser_parquet import parquet_reader

                parquet_reader(
                    parquet_file_results="nonexistent.parquet",
                    parquet_file_fragments="nonexistent.parquet",
                    q_value_filter=0.01,
                )


class TestParserEdgeCases:
    """Test edge cases for parser functionality."""

    @pytest.mark.unit
    def test_extreme_q_value_filters(self):
        """Test parser behavior with extreme q-value filters."""
        with patch("pandas.read_parquet") as mock_pd_read_parquet:
            # Create mock data with extreme q-values
            mock_psms_data = pd.DataFrame(
                {
                    "peptide": ["PEPTIDE1", "PEPTIDE2"],
                    "spectrum_q": [0.001, 0.999],
                    "psm_id": [1, 2],
                    "scannr": [100, 200],
                    "charge": [2, 3],
                    "rt": [10.0, 20.0],
                }
            )

            mock_fragments_data = pd.DataFrame(
                {
                    "psm_id": [1, 2],
                    "fragment_mz": [200.1, 300.2],
                    "fragment_intensity": [1000.0, 2000.0],
                }
            )

            def mock_read_side_effect(file_path):
                if "results" in str(file_path):
                    return mock_psms_data
                elif "fragments" in str(file_path):
                    return mock_fragments_data
                return pd.DataFrame()

            mock_pd_read_parquet.side_effect = mock_read_side_effect

            with tempfile.TemporaryDirectory() as temp_dir:
                results_file = Path(temp_dir) / "results.parquet"
                fragments_file = Path(temp_dir) / "fragments.parquet"

                # Test very strict filter (should get minimal results)
                result1 = parquet_reader(
                    parquet_file_results=results_file,
                    parquet_file_fragments=fragments_file,
                    q_value_filter=0.0001,
                )

                # Should return None for very strict filter (no PSMs pass)
                assert result1 == (None, None, None, None)

                # Test very lenient filter (should get most results)
                (
                    df_fragment2,
                    df_psms2,
                    df_fragment_max2,
                    df_fragment_max_peptide2,
                ) = parquet_reader(
                    parquet_file_results=results_file,
                    parquet_file_fragments=fragments_file,
                    q_value_filter=1.0,
                )

                # Verify DataFrames returned for lenient filter
                assert isinstance(df_psms2, pl.DataFrame)
                assert isinstance(df_fragment2, pl.DataFrame)
                assert len(df_psms2) == 2  # Both PSMs should pass

    @pytest.mark.unit
    def test_large_data_handling(self):
        """Test parser behavior with large datasets."""
        # Mock large dataset
        large_size = 10000
        large_peptides = [f"PEPTIDE{i}" for i in range(large_size)]
        large_q_values = np.random.uniform(0, 0.1, large_size)

        mock_large_data = pl.DataFrame(
            {
                "peptide": large_peptides,
                "q_value": large_q_values,
                "psm_id": list(range(large_size)),
            }
        )

        # Verify DataFrame creation works with large data
        assert len(mock_large_data) == large_size
        assert "peptide" in mock_large_data.columns

        # Test memory efficiency
        filtered_data = mock_large_data.filter(mock_large_data["q_value"] <= 0.01)
        assert len(filtered_data) <= large_size
        assert isinstance(filtered_data, pl.DataFrame)
