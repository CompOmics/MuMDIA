"""Tests for configuration backwards compatibility between old nested and new flat formats."""

import os

import pytest

from config import load_config_from_json


@pytest.mark.unit
class TestConfigCompatibility:
    """Verify that both old and new config formats load correctly."""

    @pytest.fixture
    def legacy_config_path(self):
        path = "configs/config.json"
        if not os.path.exists(path):
            pytest.skip("configs/config.json not found")
        return path

    @pytest.fixture
    def flat_config_path(self):
        path = "configs/config_simple.json"
        if not os.path.exists(path):
            pytest.skip("configs/config_simple.json not found")
        return path

    def _check_config(self, config):
        """Common assertions for any loaded config."""
        assert hasattr(config, "mzml_file")
        assert hasattr(config, "fasta_file")
        assert hasattr(config, "result_dir")

        initial = config.get_initial_search_config()
        full = config.get_full_search_config()

        assert "database" in initial
        assert "database" in full
        assert "enzyme" in initial["database"]
        assert "report_psms" in initial

        mumdia = config.get_mumdia_config()
        assert "read_initial_search_pickle" in mumdia
        assert "write_deeplc_pickle" in mumdia
        assert "targeted_search_engine" in mumdia
        assert "stop_after_stage2" in mumdia

    def test_legacy_nested_format(self, legacy_config_path):
        """Load the old nested (sage_basic / sage / mumdia) format."""
        config = load_config_from_json(legacy_config_path)
        self._check_config(config)

    def test_flat_format(self, flat_config_path):
        """Load the new flat format with _initial_search / _full_search overrides."""
        config = load_config_from_json(flat_config_path)
        self._check_config(config)

    def test_override_mechanism(self, flat_config_path):
        """Verify that search-stage overrides produce different configs."""
        config = load_config_from_json(flat_config_path)
        initial = config.get_initial_search_config()
        full = config.get_full_search_config()

        # At least one parameter should differ between stages
        differs = (
            initial["report_psms"] != full["report_psms"]
            or initial["deisotope"] != full["deisotope"]
            or initial["database"]["enzyme"]["cleave_at"]
            != full["database"]["enzyme"]["cleave_at"]
        )
        assert (
            differs
        ), "Initial and full search configs should have at least one difference"
