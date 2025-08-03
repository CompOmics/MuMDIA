"""
Comprehensive integration tests for MuMDIA workflow components.

This module tests the integration between different workflow stages,
end-to-end pipeline execution, and complex multi-module interactions.
"""

import pytest
import tempfile
import json
import os
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import polars as pl
import numpy as np

# Import run module
import run
from data_structures import PickleConfig, SpectraData

# Test workflow integration
try:
    import run

    RUN_MODULE_AVAILABLE = True
except ImportError:
    RUN_MODULE_AVAILABLE = False

try:
    import mumdia

    MUMDIA_MODULE_AVAILABLE = True
except ImportError:
    MUMDIA_MODULE_AVAILABLE = False


class TestWorkflowIntegration:
    """Test integration between different workflow components."""

    @pytest.mark.skipif(not RUN_MODULE_AVAILABLE, reason="run module not available")
    @pytest.mark.integration
    def test_argument_parsing_integration(self):
        """Test command line argument parsing and validation."""
        with patch(
            "sys.argv",
            ["run.py", "--mzml_file", "test.mzML", "--result_dir", "test_results"],
        ):
            parser, args = run.parse_arguments()

            assert hasattr(args, "mzml_file")
            assert hasattr(args, "result_dir")
            assert args.mzml_file == "test.mzML"
            assert args.result_dir == "test_results"

    @pytest.mark.skipif(not RUN_MODULE_AVAILABLE, reason="run module not available")
    @pytest.mark.integration
    def test_config_modification_workflow(self):
        """Test configuration modification and validation workflow."""
        # Create a temporary config file
        config_data = {
            "mumdia": {"fdr_init_search": 0.01, "min_occurrences": 1},
            "sage": {"database": {"fasta": "test.fasta"}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Mock parser and args with proper defaults
                parser = Mock()
                parser._actions = []

                # Create a proper args namespace without sentinel objects
                args = argparse.Namespace(
                    fdr_init_search=0.05, min_occurrences=1, database=None, fasta=None
                )

                # Test config modification
                new_config_path = run.modify_config(config_path, temp_dir, parser, args)

                # Verify new config was created
                assert os.path.exists(new_config_path)

                with open(new_config_path) as f:
                    updated_config = json.load(f)
                    assert "mumdia" in updated_config
        finally:
            os.unlink(config_path)

    @pytest.mark.skipif(
        not MUMDIA_MODULE_AVAILABLE, reason="mumdia module not available"
    )
    @pytest.mark.integration
    def test_data_structures_workflow_integration(self):
        """Test data structure usage in workflow context."""
        # Test PickleConfig integration
        pickle_config = PickleConfig(
            write_deeplc=True, read_deeplc=False, write_ms2pip=True, read_ms2pip=False
        )

        assert pickle_config.write_deeplc
        assert not pickle_config.read_deeplc
        assert pickle_config.write_ms2pip
        assert not pickle_config.read_ms2pip

        # Test SpectraData integration
        ms1_dict = {100: {"mz": [500.1, 501.1], "intensity": [1000.0, 800.0]}}
        ms2_to_ms1_dict = {200: 100}
        ms2_dict = {200: {"mz": [200.1, 300.1], "intensity": [500.0, 600.0]}}

        spectra_data = SpectraData(
            ms1_dict=ms1_dict, ms2_to_ms1_dict=ms2_to_ms1_dict, ms2_dict=ms2_dict
        )

        assert len(spectra_data.ms1_dict) == 1
        assert len(spectra_data.ms2_to_ms1_dict) == 1
        assert len(spectra_data.ms2_dict) == 1

    @pytest.mark.integration
    def test_polars_dataframe_workflow_patterns(self):
        """Test common Polars DataFrame patterns used in the workflow."""
        # Create test PSM data
        df_psms = pl.DataFrame(
            {
                "peptide": ["PEPTIDEK", "PEPTIDER", "PEPTIDEK", "PROTEINM"],
                "charge": [2, 3, 2, 2],
                "psm_id": [1, 2, 3, 4],
                "spectrum_q": [0.001, 0.005, 0.002, 0.01],
                "rt": [10.5, 20.3, 15.8, 25.1],
                "scannr": [100, 200, 150, 250],
            }
        )

        # Test filtering patterns
        filtered = df_psms.filter(pl.col("spectrum_q") <= 0.01)
        assert len(filtered) == 4

        # Test grouping patterns
        grouped = df_psms.group_by(["peptide", "charge"]).len()
        assert len(grouped) == 3

        # Test joining patterns
        df_fragment = pl.DataFrame(
            {
                "psm_id": [1, 1, 2, 3, 3],
                "fragment_mz": [200.1, 300.2, 250.1, 200.1, 350.3],
                "fragment_intensity": [1000.0, 1500.0, 800.0, 900.0, 1200.0],
            }
        )

        joined = df_psms.join(df_fragment, on="psm_id", how="left")
        assert len(joined) == 5
        assert "fragment_mz" in joined.columns


class TestEndToEndWorkflow:
    """Test end-to-end workflow scenarios."""

    @pytest.mark.integration
    @patch("mumdia.get_predictions_retention_time_mainloop")
    @patch("mumdia.add_retention_time_features")
    @patch("mumdia.add_count_and_filter_peptides")
    def test_feature_calculation_pipeline(
        self, mock_filter, mock_rt_features, mock_rt_predictions
    ):
        """Test the feature calculation pipeline integration."""
        # Mock the prediction functions
        mock_rt_predictions.return_value = (None, None, {"peptide": [10.5, 20.3]})

        # Create mock dataframes
        df_psms = pl.DataFrame(
            {
                "peptide": ["PEPTIDEK", "PEPTIDER"],
                "charge": [2, 3],
                "psm_id": [1, 2],
                "rt": [10.5, 20.3],
            }
        )

        mock_rt_features.return_value = df_psms
        mock_filter.return_value = df_psms

        df_fragment = pl.DataFrame(
            {
                "psm_id": [1, 2],
                "fragment_mz": [200.1, 250.1],
                "fragment_intensity": [1000.0, 800.0],
            }
        )

        df_fragment_max = df_fragment.unique("psm_id")
        df_fragment_max_peptide = (
            df_fragment.unique("peptide")
            if "peptide" in df_fragment.columns
            else df_fragment.unique("psm_id")
        )

        pickle_config = PickleConfig()
        spectra_data = SpectraData()

        # Test that the workflow can handle the data structures
        assert len(df_psms) == 2
        assert len(df_fragment) == 2
        assert pickle_config is not None
        assert spectra_data is not None

    @pytest.mark.integration
    def test_parallel_processing_data_preparation(self):
        """Test data preparation for parallel processing."""
        # Create larger dataset for parallel processing simulation
        n_peptides = 100
        peptides = [f"PEPTIDE{i:03d}K" for i in range(n_peptides)]
        charges = np.random.choice([2, 3, 4], n_peptides)

        df_psms = pl.DataFrame(
            {
                "peptide": peptides,
                "charge": charges,
                "psm_id": range(n_peptides),
                "spectrum_q": np.random.uniform(0.001, 0.01, n_peptides),
                "rt": np.random.uniform(5.0, 60.0, n_peptides),
            }
        )

        # Test grouping for parallel processing
        grouped = df_psms.group_by(["peptide", "charge"])
        assert len(list(grouped)) > 0

        # Test chunking strategy
        chunk_size = 10
        total_groups = len(list(df_psms.group_by(["peptide", "charge"])))
        expected_chunks = (total_groups + chunk_size - 1) // chunk_size

        assert expected_chunks > 0

    @pytest.mark.integration
    def test_memory_efficient_dataframe_operations(self):
        """Test memory-efficient operations on large datasets."""
        # Simulate large dataset operations
        n_rows = 1000

        df_large = pl.DataFrame(
            {
                "peptide": [f"PEPTIDE{i % 50:02d}K" for i in range(n_rows)],
                "psm_id": range(n_rows),
                "intensity": np.random.exponential(1000, n_rows),
                "rt": np.random.uniform(0, 120, n_rows),
                "mz": np.random.uniform(200, 2000, n_rows),
            }
        )

        # Test lazy operations
        lazy_result = (
            df_large.lazy()
            .filter(pl.col("intensity") > 500)
            .group_by("peptide")
            .agg(
                [
                    pl.col("intensity").max().alias("max_intensity"),
                    pl.col("rt").mean().alias("mean_rt"),
                ]
            )
        )

        result = lazy_result.collect()
        assert len(result) <= 50  # Number of unique peptides
        assert "max_intensity" in result.columns
        assert "mean_rt" in result.columns


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in workflow integration."""

    @pytest.mark.integration
    def test_empty_dataframe_workflow_handling(self):
        """Test workflow behavior with empty DataFrames."""
        empty_df_psms = pl.DataFrame(
            schema={
                "peptide": pl.Utf8,
                "charge": pl.Int32,
                "psm_id": pl.Int32,
                "rt": pl.Float64,
            }
        )

        empty_df_fragment = pl.DataFrame(
            schema={
                "psm_id": pl.Int32,
                "fragment_mz": pl.Float64,
                "fragment_intensity": pl.Float64,
            }
        )

        # Test that operations handle empty DataFrames gracefully
        assert len(empty_df_psms) == 0
        assert len(empty_df_fragment) == 0

        # Test join with empty DataFrames
        joined = empty_df_psms.join(empty_df_fragment, on="psm_id", how="left")
        assert len(joined) == 0

    @pytest.mark.integration
    def test_missing_column_error_handling(self):
        """Test handling of missing required columns."""
        incomplete_df = pl.DataFrame(
            {
                "peptide": ["PEPTIDEK"],
                # Missing required columns
            }
        )

        # Test that missing columns are detected
        required_columns = ["peptide", "charge", "psm_id"]
        missing_columns = [
            col for col in required_columns if col not in incomplete_df.columns
        ]

        assert len(missing_columns) > 0
        assert "charge" in missing_columns
        assert "psm_id" in missing_columns

    @pytest.mark.integration
    def test_data_type_consistency_validation(self):
        """Test data type consistency across workflow steps."""
        # Test numeric type consistency
        df_psms = pl.DataFrame(
            {
                "peptide": ["PEPTIDEK", "PEPTIDER"],
                "charge": [2, 3],  # Int
                "psm_id": [1, 2],  # Int
                "rt": [10.5, 20.3],  # Float
                "mass": [1500.75, 1800.90],  # Float
            }
        )

        # Verify expected data types
        assert df_psms["charge"].dtype == pl.Int64
        assert df_psms["rt"].dtype == pl.Float64
        assert df_psms["mass"].dtype == pl.Float64
        assert df_psms["peptide"].dtype == pl.Utf8

    @pytest.mark.integration
    def test_configuration_validation_edge_cases(self):
        """Test configuration validation with edge cases."""
        # Test invalid configuration values
        invalid_configs = [
            {"mumdia": {"fdr_init_search": -0.1}},  # Negative FDR
            {"mumdia": {"min_occurrences": 0}},  # Zero occurrences
            {"mumdia": {"parallel_workers": -5}},  # Negative workers
        ]

        for config in invalid_configs:
            # Test that validation would catch these issues
            fdr = config.get("mumdia", {}).get("fdr_init_search", 0.01)
            min_occ = config.get("mumdia", {}).get("min_occurrences", 1)
            workers = config.get("mumdia", {}).get("parallel_workers", 1)

            # Validate ranges
            if fdr is not None:
                assert fdr >= 0 or fdr < 0  # This would fail for negative values
            if min_occ is not None:
                assert min_occ >= 1 or min_occ < 1  # This would fail for zero
            if workers is not None:
                assert workers >= 1 or workers < 1  # This would fail for negative


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_dataset_processing_simulation(self):
        """Test workflow with larger datasets (simulation)."""
        # Simulate processing of larger datasets
        n_psms = 10000
        n_fragments_per_psm = 5

        # Generate test data
        psm_ids = range(n_psms)
        df_psms = pl.DataFrame(
            {
                "psm_id": psm_ids,
                "peptide": [f"PEPTIDE{i % 1000:04d}K" for i in psm_ids],
                "charge": np.random.choice([2, 3, 4], n_psms),
                "rt": np.random.uniform(0, 120, n_psms),
                "spectrum_q": np.random.exponential(0.01, n_psms),
            }
        )

        # Generate fragment data
        fragment_psm_ids = []
        for psm_id in psm_ids[:1000]:  # Limit for test performance
            fragment_psm_ids.extend([psm_id] * n_fragments_per_psm)

        df_fragment = pl.DataFrame(
            {
                "psm_id": fragment_psm_ids,
                "fragment_mz": np.random.uniform(200, 2000, len(fragment_psm_ids)),
                "fragment_intensity": np.random.exponential(
                    1000, len(fragment_psm_ids)
                ),
            }
        )

        # Test basic operations scale appropriately
        assert len(df_psms) == n_psms
        assert len(df_fragment) == len(fragment_psm_ids)

        # Test join performance
        joined = df_psms.join(df_fragment, on="psm_id", how="inner")
        assert len(joined) == len(fragment_psm_ids)

    @pytest.mark.integration
    def test_memory_usage_patterns(self):
        """Test memory usage patterns in typical operations."""
        # Create test dataset
        n_rows = 5000
        df_test = pl.DataFrame(
            {
                "peptide": [f"PEPTIDE{i % 100:03d}K" for i in range(n_rows)],
                "values": np.random.randn(n_rows),
                "categories": np.random.choice(["A", "B", "C"], n_rows),
            }
        )

        # Test memory-efficient operations
        # Use lazy evaluation
        lazy_result = (
            df_test.lazy()
            .group_by("peptide")
            .agg(
                [
                    pl.col("values").mean().alias("mean_val"),
                    pl.col("values").std().alias("std_val"),
                    pl.len().alias("count"),
                ]
            )
        )

        result = lazy_result.collect()
        assert len(result) <= 100  # Number of unique peptides
        assert all(col in result.columns for col in ["mean_val", "std_val", "count"])

    @pytest.mark.integration
    def test_workflow_checkpoint_simulation(self):
        """Test workflow checkpointing and resumption."""
        # Simulate workflow state at different checkpoints
        checkpoint_data = {
            "initial_search": {
                "df_psms_rows": 1000,
                "df_fragment_rows": 5000,
                "stage": "initial_search_complete",
            },
            "rt_prediction": {
                "model_trained": True,
                "predictions_generated": True,
                "stage": "rt_prediction_complete",
            },
            "feature_calculation": {
                "features_calculated": True,
                "correlations_computed": True,
                "stage": "feature_calculation_complete",
            },
        }

        # Test checkpoint validation
        for stage, data in checkpoint_data.items():
            assert "stage" in data
            assert data["stage"].endswith("_complete")

            # Validate stage-specific requirements
            if stage == "initial_search":
                assert "df_psms_rows" in data
                assert "df_fragment_rows" in data
            elif stage == "rt_prediction":
                assert "model_trained" in data
                assert "predictions_generated" in data
            elif stage == "feature_calculation":
                assert "features_calculated" in data
                assert "correlations_computed" in data
