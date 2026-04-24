"""Focused tests for the experimental custom stage-2 backend helpers."""

import time

import numpy as np
import pandas as pd
import polars as pl
import pytest

from peptide_search.custom_engine import (
    CandidateEntry,
    RustPrefilterResult,
    RustXICCandidate,
    ScoredCandidate,
    _build_candidate_index,
    _build_window_candidate_indices,
    _candidate_indices_for_spectrum,
    _fast_scored_candidates_from_prefilter,
    _fast_psm_rows_from_prefilter,
    _prepare_rust_xic_candidates,
    prepare_rust_stage2_partition_payload,
    retention_window_searches_custom,
    _search_partition_rust_xic,
    _score_candidate_against_spectrum,
    _shortlist_candidate_indices_by_top_fragments,
    build_ms2pip_prediction_input,
)


def _candidate(
    peptide: str,
    charge: int,
    precursor_mz: float,
    predicted_fragment_mz: list[float],
) -> CandidateEntry:
    return CandidateEntry(
        peptide=peptide,
        proteins="P1",
        num_proteins=1,
        charge=charge,
        precursor_mz=precursor_mz,
        calcmass=1000.0,
        is_decoy=False,
        peptide_len=len(peptide),
        missed_cleavages=0,
        fragment_mz=np.asarray(predicted_fragment_mz, dtype=float),
        fragment_type=tuple("b" for _ in predicted_fragment_mz),
        fragment_ordinals=np.arange(1, len(predicted_fragment_mz) + 1, dtype=int),
        fragment_charge=np.ones(len(predicted_fragment_mz), dtype=int),
        fragment_name=tuple(
            f"b{i}/1" for i in range(1, len(predicted_fragment_mz) + 1)
        ),
        predicted_fragment_mz=np.asarray(predicted_fragment_mz, dtype=float),
        predicted_fragment_weight=np.asarray(
            [10.0] * len(predicted_fragment_mz), dtype=float
        ),
        predicted_fragment_name=tuple(
            f"b{i}/1" for i in range(1, len(predicted_fragment_mz) + 1)
        ),
    )


@pytest.mark.unit
def test_build_candidate_index_collects_predicted_fragment_bins():
    candidates = [
        _candidate("PEPTIDE", 2, 500.0, [100.02, 200.02]),
        _candidate("SEQUENCE", 2, 600.0, [100.04, 300.00]),
    ]

    index = _build_candidate_index(
        candidates,
        use_predicted_fragments=True,
        fragment_bin_size_da=0.05,
    )

    shared_bin = int(np.floor(100.02 / 0.05))
    assert shared_bin in index.top_fragment_bins
    assert set(index.top_fragment_bins[shared_bin]) == {0, 1}


@pytest.mark.unit
def test_shortlist_candidates_uses_top_fragment_matches():
    candidates = [
        _candidate("PEPTIDE", 2, 500.0, [100.02, 200.02, 300.02]),
        _candidate("OTHER", 2, 505.0, [150.0, 250.0, 350.0]),
    ]
    index = _build_candidate_index(
        candidates,
        use_predicted_fragments=True,
        fragment_bin_size_da=0.05,
    )

    shortlisted = _shortlist_candidate_indices_by_top_fragments(
        np.asarray([0, 1], dtype=int),
        candidates,
        np.asarray([100.021, 200.021, 700.0], dtype=float),
        index,
        fragment_tol_ppm=20.0,
        min_top_fragment_matches=2,
    )

    assert shortlisted.tolist() == [0]


@pytest.mark.unit
def test_shortlist_falls_back_to_original_candidates_when_no_matches():
    candidates = [_candidate("PEPTIDE", 2, 500.0, [100.02, 200.02, 300.02])]
    index = _build_candidate_index(
        candidates,
        use_predicted_fragments=True,
        fragment_bin_size_da=0.05,
    )

    shortlisted = _shortlist_candidate_indices_by_top_fragments(
        np.asarray([0], dtype=int),
        candidates,
        np.asarray([900.0], dtype=float),
        index,
        fragment_tol_ppm=20.0,
        min_top_fragment_matches=1,
    )

    assert shortlisted.tolist() == [0]


@pytest.mark.unit
def test_build_ms2pip_prediction_input_expands_charges():
    import pandas as pd

    peptide_df = pd.DataFrame(
        {
            "peptide": ["PEPTIDE", "PEPTIDE", "OTHER"],
            "predictions": [100.0, 101.0, 200.0],
        }
    )

    result = build_ms2pip_prediction_input(
        peptide_df,
        {"precursor_charge": [2, 3]},
    )

    assert result.height == 4
    assert sorted(result["charge"].to_list()) == [2, 2, 3, 3]


@pytest.mark.unit
def test_score_candidate_tracks_predicted_fragment_weight_fraction():
    candidate = _candidate("PEPTIDE", 2, 500.0, [100.0, 200.0, 300.0])
    candidate = CandidateEntry(
        **{
            **candidate.__dict__,
            "predicted_fragment_weight": np.asarray([10.0, 5.0, 1.0], dtype=float),
        }
    )

    scored = _score_candidate_against_spectrum(
        candidate,
        np.asarray([100.0, 200.0, 800.0], dtype=float),
        np.asarray([1000.0, 500.0, 10.0], dtype=float),
        {"isolation_window_target": 500.0},
        fragment_tol_ppm=20.0,
        min_matched_peaks=1,
    )

    assert scored is not None
    assert scored.matched_predicted_top_fragments == 2
    assert scored.matched_predicted_weight_fraction > 0.9


@pytest.mark.unit
def test_build_window_candidate_indices_groups_candidates_by_isolation_window():
    candidates = [
        _candidate("PEPTIDE", 2, 500.0, [100.0, 200.0]),
        _candidate("OTHER", 2, 700.0, [150.0, 250.0]),
    ]
    candidate_index = _build_candidate_index(
        candidates,
        use_predicted_fragments=True,
        fragment_bin_size_da=0.05,
    )
    ms2_spectra = {
        "scan1": {
            "isolation_window_target": 500.0,
            "isolation_window_lower": 5.0,
            "isolation_window_upper": 5.0,
        },
        "scan2": {
            "isolation_window_target": 700.0,
            "isolation_window_lower": 5.0,
            "isolation_window_upper": 5.0,
        },
    }

    window_indices = _build_window_candidate_indices(
        candidates,
        candidate_index,
        ms2_spectra,
    )

    assert len(window_indices) == 2
    assert sorted(window_indices["495.0000|505.0000"].candidate_indices.tolist()) == [0]
    assert sorted(window_indices["695.0000|705.0000"].candidate_indices.tolist()) == [1]


@pytest.mark.unit
def test_candidate_indices_for_spectrum_uses_precomputed_window_subset():
    candidates = [
        _candidate("PEPTIDE", 2, 500.0, [100.0, 200.0]),
        _candidate("OTHER", 2, 700.0, [150.0, 250.0]),
    ]
    candidate_index = _build_candidate_index(
        candidates,
        use_predicted_fragments=True,
        fragment_bin_size_da=0.05,
    )
    spectrum_entry = {
        "isolation_window_target": 500.0,
        "isolation_window_lower": 5.0,
        "isolation_window_upper": 5.0,
    }
    window_indices = _build_window_candidate_indices(
        candidates,
        candidate_index,
        {"scan1": spectrum_entry},
    )

    candidate_indices = _candidate_indices_for_spectrum(
        candidate_index.sorted_precursor_mz,
        candidate_index.sorted_candidate_indices,
        candidates,
        spectrum_entry,
        max_candidates_per_spectrum=10,
        window_candidate_indices=window_indices,
    )

    assert candidate_indices.tolist() == [0]


@pytest.mark.unit
def test_fast_scored_candidates_from_prefilter_builds_approximate_hits():
    candidates = [
        _candidate("PEPTIDE", 2, 500.0, [100.0, 200.0, 300.0]),
        _candidate("OTHER", 2, 505.0, [150.0, 250.0, 350.0]),
    ]
    rust_result = RustPrefilterResult(
        candidate_indices=np.asarray([0, 1], dtype=int),
        matched_counts=np.asarray([2, 1], dtype=int),
    )

    scored_candidates = _fast_scored_candidates_from_prefilter(
        rust_result,
        candidates,
        observed_intensity=np.asarray([1000.0, 500.0, 100.0], dtype=float),
        spectrum_entry={"isolation_window_target": 500.0},
    )

    assert len(scored_candidates) == 2
    assert scored_candidates[0].matched_peaks == 2
    assert scored_candidates[0].matched_fragments[0]["fragment_name"] == "b1/1"
    assert scored_candidates[0].score > scored_candidates[1].score


@pytest.mark.unit
def test_fast_psm_rows_from_prefilter_emits_top_ranked_rows():
    candidates = [
        _candidate("PEPTIDE", 2, 500.0, [100.0, 200.0, 300.0]),
        _candidate("OTHER", 2, 505.0, [150.0, 250.0, 350.0]),
    ]
    rust_result = RustPrefilterResult(
        candidate_indices=np.asarray([0, 1], dtype=int),
        matched_counts=np.asarray([2, 1], dtype=int),
    )

    rows, next_psm_id = _fast_psm_rows_from_prefilter(
        rust_result,
        candidates,
        observed_intensity=np.asarray([1000.0, 500.0, 100.0], dtype=float),
        spectrum_entry={"isolation_window_target": 500.0, "retention_time": 123.0},
        filename="part.mzml",
        scannr="scan=1",
        next_psm_id=10,
        report_psms=1,
    )

    assert len(rows) == 1
    assert rows[0]["psm_id"] == 10
    assert rows[0]["rank"] == 1
    assert rows[0]["peptide"] == "PEPTIDE"
    assert rows[0]["matched_peaks"] == 2
    assert next_psm_id == 11


@pytest.mark.unit
def test_prepare_rust_xic_candidates_selects_top_fragments_per_charge():
    import pandas as pd
    import pyopenms as pms

    peptide_df = pd.DataFrame(
        {
            "peptide": ["PEPTIDE", "PEPTIDE"],
            "id": ["P1", "P2"],
            "predictions_lower": [100.0, 100.0],
            "predictions_upper": [120.0, 120.0],
        }
    )
    ms2pip_predictions = {
        "PEPTIDE/2": {"b2/1": 8.0, "y3/1": 3.0, "b1/1": 1.0},
    }

    candidates = _prepare_rust_xic_candidates(
        peptide_df,
        {"precursor_charge": [2, 2], "database": {"decoy_tag": "rev_"}},
        ms2pip_predictions,
        top_n_predicted_fragments=2,
    )

    fragment_mz, _, _, _, fragment_name = __import__(
        "peptide_search.custom_engine", fromlist=["_build_fragment_cache"]
    )._build_fragment_cache("PEPTIDE", ("b", "y"), 2)
    fragment_lookup = {name: mz for name, mz in zip(fragment_name, fragment_mz)}
    expected_precursor_mz = float(pms.AASequence.fromString("PEPTIDE").getMZ(2))

    assert len(candidates) == 1
    assert isinstance(candidates[0], RustXICCandidate)
    assert candidates[0].precursor_mz == pytest.approx(expected_precursor_mz)
    assert candidates[0].predicted_fragment_names == ("b2/1", "y3/1")
    np.testing.assert_allclose(
        candidates[0].predicted_fragment_mzs,
        [fragment_lookup["b2/1"], fragment_lookup["y3/1"]],
    )
    np.testing.assert_allclose(candidates[0].predicted_fragment_weights, [8.0, 3.0])


@pytest.mark.unit
def test_search_partition_rust_xic_builds_minimal_psm_rows(monkeypatch):
    import pandas as pd

    class _StubRust:
        @staticmethod
        def search_partition_chromatograms(*args, **kwargs):
            return [
                {
                    "candidate_idx": 0.0,
                    "precursor_mz": 500.0,
                    "matched_top_fragments": 2.0,
                    "matched_b_fragments": 1.0,
                    "matched_y_fragments": 1.0,
                    "xic_coverage": 0.5,
                    "xic_n_detected_scans": 4.0,
                    "xic_apex_rt": 123.4,
                    "xic_detected_rt_start": 120.0,
                    "xic_detected_rt_end": 126.0,
                    "xic_best_coelution": 0.9,
                    "xic_apex_spectrum_corr": 0.8,
                    "xic_weighted_auc": 2.5,
                    "xic_apex_intensity": 250.0,
                }
            ]

    monkeypatch.setattr("peptide_search.custom_engine.mumdia_rs", _StubRust())

    peptide_df = pd.DataFrame(
        {
            "peptide": ["PEPTIDE"],
            "id": ["P1"],
            "predictions_lower": [100.0],
            "predictions_upper": [130.0],
        }
    )
    fragment_df, psm_df, next_psm_id = _search_partition_rust_xic(
        "part.mzML",
        peptide_df,
        {"precursor_charge": [2, 2], "database": {"decoy_tag": "rev_"}},
        {"custom_engine_fragment_top_n": 2},
        10,
        ms2pip_predictions={"PEPTIDE/2": {"b2/1": 8.0, "y3/1": 3.0}},
    )

    assert fragment_df.is_empty()
    assert psm_df.height == 1
    assert psm_df["peptide"].to_list() == ["PEPTIDE"]
    assert psm_df["matched_peaks"].to_list() == [2]
    assert psm_df["xic_coverage"].to_list() == [0.5]
    assert next_psm_id == 11


@pytest.mark.unit
def test_prepare_rust_stage2_partition_payload_flattens_candidates():
    peptide_df = pd.DataFrame(
        {
            "peptide": ["PEPTIDE"],
            "id": ["P1"],
            "predictions_lower": [100.0],
            "predictions_upper": [130.0],
        }
    )

    payload = prepare_rust_stage2_partition_payload(
        "part.mzML",
        peptide_df,
        {"precursor_charge": [2, 2], "database": {"decoy_tag": "rev_"}},
        {"custom_engine_fragment_top_n": 2},
        ms2pip_predictions={"PEPTIDE/2": {"b2/1": 8.0, "y3/1": 3.0}},
    )

    assert payload is not None
    assert payload["mzml_path"] == "part.mzML"
    assert payload["top_n"] == 2
    assert payload["peptides"] == ["PEPTIDE"]
    assert payload["charges"] == [2]
    assert payload["predicted_fragment_mz_offsets"] == [0]
    assert payload["predicted_fragment_mz_lengths"] == [2]
    assert payload["predicted_fragment_name_offsets"] == [0]
    assert payload["predicted_fragment_name_lengths"] == [2]
    assert payload["predicted_fragment_weight_offsets"] == [0]
    assert payload["predicted_fragment_weight_lengths"] == [2]
    assert payload["predicted_fragment_names"] == ["b2/1", "y3/1"]


@pytest.mark.unit
def test_retention_window_searches_custom_reindexes_parallel_partition_results(
    monkeypatch,
):
    completion_delays = {
        "part_1.mzML": 0.04,
        "part_2.mzML": 0.01,
        "part_3.mzML": 0.02,
    }

    def _stub_search_partition_rust_xic(
        mzml_path,
        peptide_df,
        sage_config,
        mumdia_config,
        psm_ident_start,
        ms2pip_predictions=None,
    ):
        time.sleep(completion_delays[mzml_path])
        peptide = peptide_df.iloc[0]["peptide"]
        charge = 2
        df_psms = pl.DataFrame(
            [
                {
                    "psm_id": 0,
                    "filename": mzml_path,
                    "scannr": f"scan-{peptide}",
                    "peptide": peptide,
                    "stripped_peptide": peptide,
                    "proteins": peptide_df.iloc[0]["id"],
                    "num_proteins": 1,
                    "rank": 1,
                    "expmass": 1000.0,
                    "calcmass": 1000.0,
                    "is_decoy": False,
                    "charge": charge,
                    "peptide_len": len(peptide),
                    "missed_cleavages": 0,
                    "fragment_ppm": 0.0,
                    "delta_next": 0.0,
                    "delta_rt_model": 0.0,
                    "matched_peaks": 2,
                    "longest_b": 1,
                    "longest_y": 1,
                    "matched_intensity_pct": 0.5,
                    "fragment_intensity": 100.0,
                    "poisson": 0.2,
                    "spectrum_q": 1.0,
                    "peptide_q": 1.0,
                    "protein_q": 1.0,
                    "rt": 100.0,
                    "precursor_ppm": 0.0,
                    "hyperscore": 10.0,
                    "delta_best": 0.0,
                }
            ]
        )
        df_fragment = pl.DataFrame(
            [
                {
                    "psm_id": 0,
                    "fragment_type": "b",
                    "fragment_ordinals": 2,
                    "fragment_charge": 1,
                    "fragment_intensity": 100.0,
                    "fragment_name": "b2/1",
                    "peptide": peptide,
                    "charge": charge,
                    "rt": 100.0,
                }
            ]
        )
        return df_fragment, df_psms, 1

    monkeypatch.setattr(
        "peptide_search.custom_engine._search_partition_rust_xic",
        _stub_search_partition_rust_xic,
    )
    monkeypatch.setattr("peptide_search.custom_engine.mumdia_rs", object())

    peptide_df = pd.DataFrame(
        {
            "peptide": ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"],
            "id": ["P1", "P2", "P3"],
            "predictions_lower": [90.0, 190.0, 290.0],
            "predictions_upper": [110.0, 210.0, 310.0],
        }
    )
    mzml_dict = {
        100.0: "part_1.mzML",
        200.0: "part_2.mzML",
        300.0: "part_3.mzML",
    }

    df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide = (
        retention_window_searches_custom(
            mzml_dict,
            peptide_df,
            {
                "sage": {},
                "mumdia": {
                    "custom_engine_partition_workers": 3,
                    "custom_engine_skip_qvalues": True,
                },
            },
            rt_split_window=20.0,
            backend_context={"ms2pip_predictions": {"PEPTIDE1/2": {}}},
        )
    )

    assert df_psms["filename"].to_list() == [
        "part_1.mzML",
        "part_2.mzML",
        "part_3.mzML",
    ]
    assert df_psms["psm_id"].to_list() == [0, 1, 2]
    assert df_fragment["psm_id"].to_list() == [0, 1, 2]
    assert df_fragment_max["psm_id"].to_list() == [0, 1, 2]
    assert df_fragment_max_peptide["psm_id"].to_list() == [0, 1, 2]
