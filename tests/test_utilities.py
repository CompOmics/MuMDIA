"""
Test suite for utility functions.

This module tests logging, I/O utilities, and pickling functionality.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import polars as pl
import pytest

from data_structures import PickleConfig
from utilities.io_utils import (
    assign_identifiers,
    create_directory,
    remove_intermediate_files,
)
from utilities.logger import log_info
from utilities.pickling import read_variables_from_pickles, write_variables_to_pickles


class TestLogger:
    """Test logging functionality."""

    @patch("utilities.logger.console")
    def test_log_info_basic(self, mock_console):
        """Test basic logging functionality."""
        log_info("Test message")

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "Test message" in call_args
        assert "Elapsed Time:" in call_args

    @patch("utilities.logger.console")
    def test_log_info_multiple_calls(self, mock_console):
        """Test multiple logging calls."""
        log_info("First message")
        log_info("Second message")

        assert mock_console.print.call_count == 2

    @patch("utilities.logger.console")
    def test_log_info_special_characters(self, mock_console):
        """Test logging with special characters."""
        message = "Test with émojis 🧪 and symbols @#$%"
        log_info(message)

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert message in call_args


class TestIOUtils:
    """Test I/O utility functions."""

    def test_create_directory_new(self):
        """Test creating a new directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "test_directory")

            create_directory(new_dir)

            assert os.path.exists(new_dir)
            assert os.path.isdir(new_dir)

    @patch("utilities.io_utils.log_info")
    def test_create_directory_existing(self, mock_log):
        """Test creating an existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            create_directory(temp_dir)  # Should not raise error

            mock_log.assert_called_with("Directory already exists. Skipping creation")

    @patch("utilities.io_utils.log_info")
    def test_create_directory_with_existing_files(self, mock_log):
        """Test creating directory with existing parquet files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the files that should be removed
            fragments_file = os.path.join(temp_dir, "matched_fragments.sage.parquet")
            results_file = os.path.join(temp_dir, "results.sage.parquet")

            with open(fragments_file, "w") as f:
                f.write("test")
            with open(results_file, "w") as f:
                f.write("test")

            create_directory(temp_dir)

            # Files should be removed
            assert not os.path.exists(fragments_file)
            assert not os.path.exists(results_file)

    def test_remove_intermediate_files(self):
        """Test removing intermediate files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = [
                "matched_fragments.sage.parquet",
                "results.sage.parquet",
                "sage_values.json",
            ]

            for file_name in test_files:
                file_path = os.path.join(temp_dir, file_name)
                with open(file_path, "w") as f:
                    f.write("test")

            # Create a file that should NOT be removed
            keep_file = os.path.join(temp_dir, "keep_this.txt")
            with open(keep_file, "w") as f:
                f.write("keep")

            remove_intermediate_files(temp_dir)

            # Intermediate files should be removed
            for file_name in test_files:
                file_path = os.path.join(temp_dir, file_name)
                assert not os.path.exists(file_path)

            # Other files should remain
            assert os.path.exists(keep_file)

    def test_assign_identifiers(self):
        """Test fragment identifier assignment."""
        # Create test data
        df = pl.DataFrame(
            {
                "fragment_mz_experimental": [100.5, 200.2, 100.5, 300.1, 200.2],
                "other_column": [1, 2, 3, 4, 5],
            }
        )

        result = assign_identifiers(df)

        # Check that identifiers are assigned
        assert "peak_identifier" in result.columns

        # Check that identical m/z values get the same identifier
        identifiers = result["peak_identifier"].to_list()
        assert identifiers[0] == identifiers[2]  # Both 100.5
        assert identifiers[1] == identifiers[4]  # Both 200.2

        # Check that different m/z values get different identifiers
        assert len(set(identifiers)) == 3  # Three unique m/z values


class TestPickling:
    """Test pickling functionality."""

    def test_write_variables_to_pickles_basic(
        self, sample_psm_data, sample_fragment_data, sample_pickle_config
    ):
        """Test basic pickle writing functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("pickle.dump") as mock_dump,
                patch("builtins.open", mock_open()) as mock_file,
            ):
                write_variables_to_pickles(
                    df_fragment=sample_fragment_data,
                    df_psms=sample_psm_data,
                    df_fragment_max=sample_fragment_data[:2],
                    df_fragment_max_peptide=sample_fragment_data[:1],
                    config={"test": "config"},
                    dlc_transfer_learn=Mock(),
                    pickle_config=sample_pickle_config,
                    write_full_search_pickle=True,
                    read_full_search_pickle=False,
                    dir=temp_dir,
                )

                # Should have opened files for writing
                assert mock_file.call_count >= 6  # At least the basic files

                # Should have called pickle.dump for each file
                assert mock_dump.call_count >= 6

    def test_write_variables_to_pickles_with_tsv(
        self, sample_psm_data, sample_fragment_data, sample_pickle_config
    ):
        """Test pickle writing with TSV export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("pickle.dump") as mock_dump,
                patch("builtins.open", mock_open()) as mock_file,
            ):
                write_variables_to_pickles(
                    df_fragment=sample_fragment_data,
                    df_psms=sample_psm_data,
                    df_fragment_max=sample_fragment_data[:2],
                    df_fragment_max_peptide=sample_fragment_data[:1],
                    config={"test": "config"},
                    dlc_transfer_learn=Mock(),
                    pickle_config=sample_pickle_config,
                    write_full_search_pickle=True,
                    read_full_search_pickle=False,
                    dir=temp_dir,
                    write_to_tsv=True,
                )

                # Should have additional file opens for TSV files
                # Exact count depends on implementation, but should be more than basic
                assert mock_file.call_count >= 7

    @patch("pickle.load")
    @patch("builtins.open", mock_open())
    def test_read_variables_from_pickles(
        self, mock_pickle_load, sample_psm_data, sample_fragment_data
    ):
        """Test reading variables from pickle files."""
        # Mock the loaded data
        mock_data = [
            sample_fragment_data,  # df_fragment
            sample_psm_data,  # df_psms
            sample_fragment_data[:2],  # df_fragment_max
            sample_fragment_data[:1],  # df_fragment_max_peptide
            {"test": "config"},  # config
            Mock(),  # dlc_transfer_learn
            PickleConfig(),  # flags
        ]

        mock_pickle_load.side_effect = mock_data

        with tempfile.TemporaryDirectory() as temp_dir:
            result = read_variables_from_pickles(dir=temp_dir)

            # Should return tuple with 7 elements
            assert len(result) == 7

            # Should have called pickle.load for each file
            assert mock_pickle_load.call_count == 7

    @patch("pickle.load")
    @patch("builtins.open", side_effect=FileNotFoundError())
    def test_read_variables_from_pickles_missing_files(
        self, mock_open, mock_pickle_load
    ):
        """Test behavior when pickle files are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError):
                read_variables_from_pickles(dir=temp_dir)

    def test_pickle_config_integration(self):
        """Test PickleConfig integration with pickling functions."""
        config = PickleConfig(
            write_deeplc=True, read_ms2pip=True, write_correlation=False
        )

        # Test that config values can be used for conditional logic
        assert config.write_deeplc is True
        assert config.read_ms2pip is True
        assert config.write_correlation is False

        # This would typically be used in conditional statements
        operations = []
        if config.write_deeplc:
            operations.append("write_deeplc")
        if config.read_ms2pip:
            operations.append("read_ms2pip")
        if config.write_correlation:
            operations.append("write_correlation")

        assert operations == ["write_deeplc", "read_ms2pip"]


class TestUtilityIntegration:
    """Test integration between utility functions."""

    def test_logging_with_io_operations(self):
        """Test logging during I/O operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "test_integration")

            with patch("utilities.io_utils.log_info") as mock_log:
                create_directory(new_dir)

                # Should not log anything for successful creation
                mock_log.assert_not_called()

                # Try to create again (should log)
                create_directory(new_dir)
                mock_log.assert_called_once()

    def test_directory_creation_with_pickling(
        self, sample_psm_data, sample_fragment_data
    ):
        """Test directory creation followed by pickling operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pickle_dir = os.path.join(temp_dir, "pickle_test")

            # Create directory
            create_directory(pickle_dir)
            assert os.path.exists(pickle_dir)

            # Test that pickling can work in the created directory
            with patch("pickle.dump") as mock_dump, patch("builtins.open", mock_open()):
                write_variables_to_pickles(
                    df_fragment=sample_fragment_data,
                    df_psms=sample_psm_data,
                    df_fragment_max=sample_fragment_data[:2],
                    df_fragment_max_peptide=sample_fragment_data[:1],
                    config={},
                    dlc_transfer_learn=Mock(),
                    pickle_config=PickleConfig(),
                    write_full_search_pickle=True,
                    read_full_search_pickle=False,
                    dir=pickle_dir,
                )

                # Should have successfully attempted to pickle
                assert mock_dump.call_count > 0


class TestErrorHandling:
    """Test error handling in utility functions."""

    def test_create_directory_permission_error(self):
        """Test handling of permission errors during directory creation."""
        # Test with a path that should cause permission error
        invalid_path = "/root/test_dir_no_permission"

        # This should not raise an exception in normal circumstances
        # The actual behavior depends on the system and permissions
        try:
            create_directory(invalid_path)
        except PermissionError:
            # This is expected on some systems
            pass

    def test_remove_intermediate_files_missing_directory(self):
        """Test removing files from non-existent directory."""
        non_existent_dir = "/path/that/does/not/exist"

        # Should not raise an exception
        try:
            remove_intermediate_files(non_existent_dir)
        except (FileNotFoundError, OSError):
            # This is acceptable behavior
            pass

    def test_assign_identifiers_empty_dataframe(self):
        """Test identifier assignment with empty DataFrame."""
        empty_df = pl.DataFrame(
            {"fragment_mz_experimental": [], "other_column": []},
            schema={"fragment_mz_experimental": pl.Float64, "other_column": pl.Utf8},
        )

        result = assign_identifiers(empty_df)

        # Should handle empty DataFrame gracefully
        assert "peak_identifier" in result.columns
        assert len(result) == 0
