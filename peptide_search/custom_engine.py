"""Experimental in-repo RT-window search backend used instead of Sage for stage 2.

This backend is intentionally simple: it searches each RT-split mzML partition by
matching theoretical b/y fragments from tryptic candidates against observed MS2
spectra and then estimates target-decoy q-values from the resulting scores.

It is designed as a starting point for iterative development rather than as a
feature-complete Sage replacement.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
import pathlib
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pyopenms as pms

try:
    import mumdia_rs
except ImportError:  # pragma: no cover - optional native extension
    mumdia_rs = None

from parsers.parser_mzml import get_ms1_mzml
from prediction_wrappers.wrapper_ms2pip import get_predictions_fragment_intensity
from utilities.logger import log_info

PROTON_MASS = 1.007276466812
_FRAGMENT_LABEL_PATTERN = re.compile(r"([by])(\d+)(\+{1,})$")
_FRAGMENT_NAME_PATTERN = re.compile(r"([by])(\d+)/(\d+)$")


@dataclass(frozen=True)
class CandidateEntry:
    """Precomputed candidate peptide data for the custom backend."""

    peptide: str
    proteins: str
    num_proteins: int
    charge: int
    precursor_mz: float
    calcmass: float
    is_decoy: bool
    peptide_len: int
    missed_cleavages: int
    fragment_mz: np.ndarray
    fragment_type: Tuple[str, ...]
    fragment_ordinals: np.ndarray
    fragment_charge: np.ndarray
    fragment_name: Tuple[str, ...]
    predicted_fragment_mz: np.ndarray
    predicted_fragment_weight: np.ndarray
    predicted_fragment_name: Tuple[str, ...]


@dataclass(frozen=True)
class CandidateIndex:
    """Index structures used to quickly shortlist candidates per spectrum."""

    sorted_precursor_mz: np.ndarray
    sorted_candidate_indices: np.ndarray
    top_fragment_bins: Dict[int, Tuple[int, ...]]
    fragment_bin_size_da: float
    use_predicted_fragments: bool


@dataclass(frozen=True)
class WindowCandidateIndex:
    """Precomputed candidate subset and fragment bins for one DIA window."""

    candidate_indices: np.ndarray
    top_fragment_bins: Dict[int, Tuple[int, ...]]


@dataclass
class ScoredCandidate:
    """Match result for one peptide-spectrum candidate."""

    candidate: CandidateEntry
    score: float
    matched_peaks: int
    matched_intensity_pct: float
    matched_intensity_sum: float
    fragment_ppm: float
    precursor_ppm: float
    longest_b: int
    longest_y: int
    max_fragment_intensity: float
    matched_predicted_top_fragments: int
    matched_predicted_weight_fraction: float
    matched_fragments: List[Dict[str, Any]]


@dataclass(frozen=True)
class RustPrefilterResult:
    """Shortlisted candidate ids and coarse match counts returned from Rust."""

    candidate_indices: np.ndarray
    matched_counts: np.ndarray


@dataclass(frozen=True)
class RustXICCandidate:
    """Minimal candidate metadata for the Rust chromatogram backend."""

    peptide: str
    proteins: str
    num_proteins: int
    charge: int
    precursor_mz: float
    is_decoy: bool
    peptide_len: int
    missed_cleavages: int
    rt_min: float
    rt_max: float
    predicted_fragment_mzs: np.ndarray
    predicted_fragment_names: Tuple[str, ...]
    predicted_fragment_weights: np.ndarray


@dataclass(frozen=True)
class PartitionSearchResult:
    """Search results for one mzML RT partition before global PSM reindexing."""

    partition_index: int
    mzml_path: str
    upper_mzml_partition: float
    df_fragment: pl.DataFrame
    df_psms: pl.DataFrame


_EMPTY_FRAGMENT_SCHEMA = {
    "psm_id": pl.Int64,
    "fragment_type": pl.Utf8,
    "fragment_ordinals": pl.Int64,
    "fragment_charge": pl.Int64,
    "fragment_intensity": pl.Float64,
    "fragment_name": pl.Utf8,
    "peptide": pl.Utf8,
    "charge": pl.Int64,
    "rt": pl.Float64,
}


def _get_ppm_tolerance(sage_config: Dict[str, Any]) -> float:
    ppm_window = sage_config.get("fragment_tol", {}).get("ppm", [-13.0, 13.0])
    if isinstance(ppm_window, Sequence) and len(ppm_window) == 2:
        return max(abs(float(ppm_window[0])), abs(float(ppm_window[1])))
    return 13.0


def _missed_cleavages(peptide: str) -> int:
    count = 0
    for idx in range(len(peptide) - 1):
        if peptide[idx] in {"K", "R"} and peptide[idx + 1] != "P":
            count += 1
    return count


def _longest_consecutive_run(ordinals: Iterable[int]) -> int:
    values = sorted(set(int(v) for v in ordinals))
    if not values:
        return 0
    best = 1
    current = 1
    for prev, curr in zip(values, values[1:]):
        if curr == prev + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _parse_fragment_name(fragment_name: str) -> Optional[Tuple[str, int, int]]:
    match = _FRAGMENT_NAME_PATTERN.fullmatch(fragment_name)
    if match is None:
        return None
    fragment_type, ordinal, charge = match.groups()
    return fragment_type, int(ordinal), int(charge)


def _make_fragment_generator(
    ion_kinds: Sequence[str],
    max_fragment_charge: int,
) -> pms.TheoreticalSpectrumGenerator:
    generator = pms.TheoreticalSpectrumGenerator()
    params = generator.getParameters()
    params.setValue("add_metainfo", "true")
    params.setValue("add_b_ions", "true" if "b" in ion_kinds else "false")
    params.setValue("add_y_ions", "true" if "y" in ion_kinds else "false")
    params.setValue("add_a_ions", "false")
    params.setValue("add_x_ions", "false")
    params.setValue("add_c_ions", "false")
    params.setValue("add_z_ions", "false")
    params.setValue("add_losses", "false")
    generator.setParameters(params)
    return generator


def _build_fragment_cache(
    peptide: str,
    ion_kinds: Sequence[str],
    max_fragment_charge: int,
) -> Tuple[np.ndarray, Tuple[str, ...], np.ndarray, np.ndarray, Tuple[str, ...]]:
    sequence = pms.AASequence.fromString(peptide)
    spectrum = pms.MSSpectrum()
    generator = _make_fragment_generator(ion_kinds, max_fragment_charge)
    generator.getSpectrum(spectrum, sequence, 1, max_fragment_charge)

    labels: List[str] = []
    if spectrum.getStringDataArrays():
        string_data = spectrum.getStringDataArrays()[0]
        labels = [
            label.decode() if hasattr(label, "decode") else str(label)
            for label in string_data
        ]

    fragment_mz: List[float] = []
    fragment_type: List[str] = []
    fragment_ordinals: List[int] = []
    fragment_charge: List[int] = []
    fragment_name: List[str] = []

    for index, label in enumerate(labels):
        match = _FRAGMENT_LABEL_PATTERN.fullmatch(label)
        if match is None:
            continue
        ion_type, ordinal, charge_marks = match.groups()
        if ion_type not in ion_kinds:
            continue
        fragment_mz.append(float(spectrum[index].getMZ()))
        fragment_type.append(ion_type)
        fragment_ordinals.append(int(ordinal))
        fragment_charge.append(len(charge_marks))
        fragment_name.append(f"{ion_type}{ordinal}/{len(charge_marks)}")

    return (
        np.asarray(fragment_mz, dtype=float),
        tuple(fragment_type),
        np.asarray(fragment_ordinals, dtype=int),
        np.asarray(fragment_charge, dtype=int),
        tuple(fragment_name),
    )


def _prepare_candidate_entries(
    peptide_df: pd.DataFrame,
    sage_config: Dict[str, Any],
    ms2pip_predictions: Optional[Dict[str, Dict[str, float]]] = None,
    top_n_predicted_fragments: int = 3,
) -> List[CandidateEntry]:
    if peptide_df.empty:
        return []

    precursor_charge_min, precursor_charge_max = sage_config.get(
        "precursor_charge", [1, 4]
    )
    max_fragment_charge = int(sage_config.get("max_fragment_charge", 2))
    ion_kinds = tuple(sage_config.get("database", {}).get("ion_kinds", ["b", "y"]))
    decoy_tag = str(sage_config.get("database", {}).get("decoy_tag", "rev_"))

    grouped = (
        peptide_df[["peptide", "id"]]
        .drop_duplicates()
        .groupby("peptide", sort=False)["id"]
        .agg(list)
        .reset_index()
    )

    fragment_cache: Dict[
        str, Tuple[np.ndarray, Tuple[str, ...], np.ndarray, np.ndarray, Tuple[str, ...]]
    ] = {}
    candidates: List[CandidateEntry] = []

    for row in grouped.to_dict(orient="records"):
        peptide = str(row["peptide"])
        raw_protein_ids = row.get("id", [])
        if not isinstance(raw_protein_ids, list):
            raw_protein_ids = [raw_protein_ids]
        protein_ids = sorted(
            {str(value) for value in raw_protein_ids if value is not None}
        )
        target_proteins = [
            protein for protein in protein_ids if not protein.startswith(decoy_tag)
        ]
        effective_proteins = target_proteins if target_proteins else protein_ids
        proteins_joined = ";".join(effective_proteins)
        is_decoy = not bool(target_proteins)

        if peptide not in fragment_cache:
            fragment_cache[peptide] = _build_fragment_cache(
                peptide,
                ion_kinds,
                max_fragment_charge,
            )

        (
            fragment_mz,
            fragment_type,
            fragment_ordinals,
            fragment_charge,
            fragment_name,
        ) = fragment_cache[peptide]
        if fragment_mz.size == 0:
            continue

        sequence = pms.AASequence.fromString(peptide)
        neutral_mass = float(sequence.getMonoWeight())
        missed_cleavages = _missed_cleavages(peptide)

        for charge in range(int(precursor_charge_min), int(precursor_charge_max) + 1):
            prediction_key = f"{peptide}/{charge}"
            predicted_fragment_names: List[str] = []
            predicted_fragment_mz: List[float] = []
            predicted_fragment_weight: List[float] = []
            if ms2pip_predictions is not None:
                predicted_fragment_lookup = ms2pip_predictions.get(prediction_key, {})
                predicted_fragments_ranked = sorted(
                    [
                        (
                            fragment_name_value,
                            float(predicted_fragment_lookup[fragment_name_value]),
                        )
                        for fragment_name_value in fragment_name
                        if float(
                            predicted_fragment_lookup.get(fragment_name_value, 0.0)
                        )
                        > 0.0
                    ],
                    key=lambda item: item[1],
                    reverse=True,
                )[:top_n_predicted_fragments]
                fragment_mz_lookup = {
                    fragment_name_value: float(fragment_mz_value)
                    for fragment_name_value, fragment_mz_value in zip(
                        fragment_name, fragment_mz
                    )
                }
                for (
                    fragment_name_value,
                    fragment_weight_value,
                ) in predicted_fragments_ranked:
                    predicted_fragment_names.append(fragment_name_value)
                    predicted_fragment_mz.append(
                        fragment_mz_lookup[fragment_name_value]
                    )
                    predicted_fragment_weight.append(fragment_weight_value)

            candidates.append(
                CandidateEntry(
                    peptide=peptide,
                    proteins=proteins_joined,
                    num_proteins=len(effective_proteins),
                    charge=charge,
                    precursor_mz=float(sequence.getMZ(charge)),
                    calcmass=neutral_mass,
                    is_decoy=is_decoy,
                    peptide_len=len(peptide),
                    missed_cleavages=missed_cleavages,
                    fragment_mz=fragment_mz,
                    fragment_type=fragment_type,
                    fragment_ordinals=fragment_ordinals,
                    fragment_charge=fragment_charge,
                    fragment_name=fragment_name,
                    predicted_fragment_mz=np.asarray(
                        predicted_fragment_mz, dtype=float
                    ),
                    predicted_fragment_weight=np.asarray(
                        predicted_fragment_weight, dtype=float
                    ),
                    predicted_fragment_name=tuple(predicted_fragment_names),
                )
            )

    return candidates


def build_ms2pip_prediction_input(
    peptide_df: pd.DataFrame,
    sage_config: Dict[str, Any],
) -> pl.DataFrame:
    """Create a minimal peptide/charge table for MS2PIP from Stage-2 RT candidates."""
    if peptide_df.empty:
        return pl.DataFrame(
            schema={"peptide": pl.Utf8, "charge": pl.Int64, "rt": pl.Float64}
        )

    precursor_charge_min, precursor_charge_max = sage_config.get(
        "precursor_charge", [1, 4]
    )
    unique_peptides = (
        pd.DataFrame(peptide_df.loc[:, ["peptide", "predictions"]])
        .groupby("peptide", sort=False, as_index=False)
        .first()
    )

    rows: List[Dict[str, Any]] = []
    for row in unique_peptides.to_dict(orient="records"):
        peptide = str(row["peptide"])
        rt_value = float(row.get("predictions", 0.0))
        for charge in range(int(precursor_charge_min), int(precursor_charge_max) + 1):
            rows.append({"peptide": peptide, "charge": charge, "rt": rt_value})

    return pl.DataFrame(rows)


def _prepare_rust_xic_candidates(
    peptide_df: pd.DataFrame,
    sage_config: Dict[str, Any],
    ms2pip_predictions: Optional[Dict[str, Dict[str, float]]],
    top_n_predicted_fragments: int,
) -> List[RustXICCandidate]:
    """Create minimal per-charge candidates for Rust chromatogram extraction."""
    if peptide_df.empty or ms2pip_predictions is None:
        return []

    precursor_charge_min, precursor_charge_max = sage_config.get(
        "precursor_charge", [1, 4]
    )
    max_fragment_charge = int(sage_config.get("max_fragment_charge", 2))
    ion_kinds = tuple(sage_config.get("database", {}).get("ion_kinds", ["b", "y"]))
    decoy_tag = str(sage_config.get("database", {}).get("decoy_tag", "rev_"))

    grouped = (
        peptide_df[["peptide", "id", "predictions_lower", "predictions_upper"]]
        .drop_duplicates()
        .groupby("peptide", sort=False)
        .agg(
            {
                "id": list,
                "predictions_lower": "min",
                "predictions_upper": "max",
            }
        )
        .reset_index()
    )

    candidates: List[RustXICCandidate] = []
    for row in grouped.to_dict(orient="records"):
        peptide = str(row["peptide"])
        sequence = pms.AASequence.fromString(peptide)
        (
            fragment_mz,
            _,
            _,
            _,
            fragment_name,
        ) = _build_fragment_cache(peptide, ion_kinds, max_fragment_charge)
        fragment_mz_lookup = {
            fragment_name_value: float(fragment_mz_value)
            for fragment_name_value, fragment_mz_value in zip(
                fragment_name, fragment_mz
            )
        }
        raw_protein_ids = row.get("id", [])
        if not isinstance(raw_protein_ids, list):
            raw_protein_ids = [raw_protein_ids]
        protein_ids = sorted(
            {str(value) for value in raw_protein_ids if value is not None}
        )
        target_proteins = [
            protein for protein in protein_ids if not protein.startswith(decoy_tag)
        ]
        effective_proteins = target_proteins if target_proteins else protein_ids
        proteins_joined = ";".join(effective_proteins)
        is_decoy = not bool(target_proteins)

        for charge in range(int(precursor_charge_min), int(precursor_charge_max) + 1):
            prediction_key = f"{peptide}/{charge}"
            predicted_fragment_lookup = ms2pip_predictions.get(prediction_key, {})
            predicted_fragments_ranked = sorted(
                [
                    (fragment_name, float(weight))
                    for fragment_name, weight in predicted_fragment_lookup.items()
                    if float(weight) > 0.0 and fragment_name in fragment_mz_lookup
                ],
                key=lambda item: item[1],
                reverse=True,
            )[:top_n_predicted_fragments]
            if not predicted_fragments_ranked:
                continue

            candidates.append(
                RustXICCandidate(
                    peptide=peptide,
                    proteins=proteins_joined,
                    num_proteins=len(effective_proteins),
                    charge=charge,
                    precursor_mz=float(sequence.getMZ(charge)),
                    is_decoy=is_decoy,
                    peptide_len=len(peptide),
                    missed_cleavages=_missed_cleavages(peptide),
                    rt_min=float(row.get("predictions_lower", 0.0)),
                    rt_max=float(row.get("predictions_upper", 0.0)),
                    predicted_fragment_mzs=np.asarray(
                        [
                            fragment_mz_lookup[name]
                            for name, _ in predicted_fragments_ranked
                        ],
                        dtype=float,
                    ),
                    predicted_fragment_names=tuple(
                        fragment_name for fragment_name, _ in predicted_fragments_ranked
                    ),
                    predicted_fragment_weights=np.asarray(
                        [weight for _, weight in predicted_fragments_ranked],
                        dtype=float,
                    ),
                )
            )

    return candidates


def _search_partition_rust_xic(
    mzml_path: str,
    peptide_df: pd.DataFrame,
    sage_config: Dict[str, Any],
    mumdia_config: Dict[str, Any],
    psm_ident_start: int,
    ms2pip_predictions: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, int]:
    """Thin Python wrapper around the Rust chromatogram backend."""
    if mumdia_rs is None or ms2pip_predictions is None:
        return (
            pl.DataFrame(schema=_EMPTY_FRAGMENT_SCHEMA),
            pl.DataFrame(),
            psm_ident_start,
        )

    payload = prepare_rust_stage2_partition_payload(
        mzml_path,
        peptide_df,
        sage_config,
        mumdia_config,
        ms2pip_predictions,
    )
    if payload is None:
        return (
            pl.DataFrame(schema=_EMPTY_FRAGMENT_SCHEMA),
            pl.DataFrame(),
            psm_ident_start,
        )

    results = mumdia_rs.search_partition_chromatograms(
        payload["mzml_path"],
        payload["peptides"],
        np.asarray(payload["charges"], dtype=np.uint64),
        np.asarray(payload["precursor_mzs"], dtype=np.float64),
        np.asarray(payload["rt_mins"], dtype=np.float64),
        np.asarray(payload["rt_maxs"], dtype=np.float64),
        np.asarray(payload["predicted_fragment_mzs"], dtype=np.float64),
        np.asarray(payload["predicted_fragment_mz_offsets"], dtype=np.uint64),
        np.asarray(payload["predicted_fragment_mz_lengths"], dtype=np.uint64),
        payload["predicted_fragment_names"],
        np.asarray(payload["predicted_fragment_name_offsets"], dtype=np.uint64),
        np.asarray(payload["predicted_fragment_name_lengths"], dtype=np.uint64),
        np.asarray(payload["predicted_fragment_weights"], dtype=np.float64),
        np.asarray(payload["predicted_fragment_weight_offsets"], dtype=np.uint64),
        np.asarray(payload["predicted_fragment_weight_lengths"], dtype=np.uint64),
        top_n=int(payload["top_n"]),
        ppm_tolerance=float(payload["ppm_tolerance"]),
    )

    if not results:
        return (
            pl.DataFrame(schema=_EMPTY_FRAGMENT_SCHEMA),
            pl.DataFrame(),
            psm_ident_start,
        )

    filename = pathlib.Path(mzml_path).name
    candidates = _prepare_rust_xic_candidates(
        peptide_df,
        sage_config,
        ms2pip_predictions,
        int(payload["top_n"]),
    )
    psm_rows: List[Dict[str, Any]] = []
    next_psm_id = int(psm_ident_start)
    for result in results:
        candidate_idx = int(result.get("candidate_idx", -1))
        if candidate_idx < 0 or candidate_idx >= len(candidates):
            continue
        candidate = candidates[candidate_idx]
        precursor_mz = float(result.get("precursor_mz", 0.0))
        calcmass = max(
            0.0, precursor_mz * candidate.charge - PROTON_MASS * candidate.charge
        )
        apex_rt = float(
            result.get(
                "xic_apex_rt",
                (candidate.rt_min + candidate.rt_max) / 2.0,
            )
        )
        psm_rows.append(
            {
                "psm_id": next_psm_id,
                "filename": filename,
                "scannr": f"xic_{candidate_idx}_{apex_rt:.4f}",
                "peptide": candidate.peptide,
                "stripped_peptide": candidate.peptide,
                "proteins": candidate.proteins,
                "num_proteins": candidate.num_proteins,
                "rank": 1,
                "expmass": calcmass,
                "calcmass": calcmass,
                "is_decoy": candidate.is_decoy,
                "charge": candidate.charge,
                "peptide_len": candidate.peptide_len,
                "missed_cleavages": candidate.missed_cleavages,
                "fragment_ppm": 0.0,
                "delta_next": 0.0,
                "delta_rt_model": 0.0,
                "matched_peaks": int(result.get("matched_top_fragments", 0.0)),
                "longest_b": int(result.get("matched_b_fragments", 0.0)),
                "longest_y": int(result.get("matched_y_fragments", 0.0)),
                "matched_intensity_pct": float(result.get("xic_coverage", 0.0)) * 100.0,
                "fragment_intensity": float(result.get("xic_apex_intensity", 0.0)),
                "poisson": float(result.get("xic_best_coelution", 0.0)),
                "spectrum_q": 1.0,
                "peptide_q": 1.0,
                "protein_q": 1.0,
                "rt": apex_rt,
                "precursor_ppm": 0.0,
                "hyperscore": float(
                    result.get(
                        "xic_weighted_auc",
                        result.get("xic_apex_intensity", 0.0),
                    )
                ),
                "delta_best": 0.0,
                "xic_coverage": float(result.get("xic_coverage", 0.0)),
                "xic_n_detected_scans": float(result.get("xic_n_detected_scans", 0.0)),
                "xic_apex_rt": apex_rt,
                "xic_detected_rt_start": float(
                    result.get("xic_detected_rt_start", apex_rt)
                ),
                "xic_detected_rt_end": float(
                    result.get("xic_detected_rt_end", apex_rt)
                ),
                "xic_best_coelution": float(result.get("xic_best_coelution", 0.0)),
                "xic_apex_spectrum_corr": float(
                    result.get("xic_apex_spectrum_corr", 0.0)
                ),
                "xic_weighted_auc": float(result.get("xic_weighted_auc", 0.0)),
            }
        )
        next_psm_id += 1

    if not psm_rows:
        return pl.DataFrame(schema=_EMPTY_FRAGMENT_SCHEMA), pl.DataFrame(), next_psm_id

    log_info(
        f"Custom stage-2 Rust XIC search for {filename}: {len(psm_rows)} chromatogram hits"
    )
    return (
        pl.DataFrame(schema=_EMPTY_FRAGMENT_SCHEMA),
        pl.DataFrame(psm_rows),
        next_psm_id,
    )


def prepare_rust_stage2_partition_payload(
    mzml_path: str,
    peptide_df: pd.DataFrame,
    sage_config: Dict[str, Any],
    mumdia_config: Dict[str, Any],
    ms2pip_predictions: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[Dict[str, Any]]:
    """Build the minimal flat payload required by the Rust Stage-2 backend."""
    if ms2pip_predictions is None:
        return None

    top_n_predicted_fragments = int(
        mumdia_config.get("custom_engine_fragment_top_n", 3)
    )
    candidates = _prepare_rust_xic_candidates(
        peptide_df,
        sage_config,
        ms2pip_predictions,
        top_n_predicted_fragments,
    )
    if not candidates:
        return None

    flat_fragment_names: List[str] = []
    flat_fragment_weights: List[float] = []
    flat_fragment_mzs: List[float] = []
    fragment_name_offsets: List[int] = []
    fragment_name_lengths: List[int] = []
    fragment_weight_offsets: List[int] = []
    fragment_weight_lengths: List[int] = []
    fragment_mz_offsets: List[int] = []
    fragment_mz_lengths: List[int] = []
    current_name_offset = 0
    current_weight_offset = 0
    current_mz_offset = 0

    for candidate in candidates:
        fragment_name_offsets.append(current_name_offset)
        fragment_name_lengths.append(len(candidate.predicted_fragment_names))
        flat_fragment_names.extend(candidate.predicted_fragment_names)
        current_name_offset += len(candidate.predicted_fragment_names)

        fragment_weight_offsets.append(current_weight_offset)
        fragment_weight_lengths.append(int(candidate.predicted_fragment_weights.size))
        flat_fragment_weights.extend(candidate.predicted_fragment_weights.tolist())
        current_weight_offset += int(candidate.predicted_fragment_weights.size)

        fragment_mz_offsets.append(current_mz_offset)
        fragment_mz_lengths.append(int(candidate.predicted_fragment_mzs.size))
        flat_fragment_mzs.extend(candidate.predicted_fragment_mzs.tolist())
        current_mz_offset += int(candidate.predicted_fragment_mzs.size)

    return {
        "format_version": 1,
        "mzml_path": mzml_path,
        "top_n": top_n_predicted_fragments,
        "ppm_tolerance": float(_get_ppm_tolerance(sage_config)),
        "peptides": [candidate.peptide for candidate in candidates],
        "charges": [int(candidate.charge) for candidate in candidates],
        "precursor_mzs": [float(candidate.precursor_mz) for candidate in candidates],
        "rt_mins": [float(candidate.rt_min) for candidate in candidates],
        "rt_maxs": [float(candidate.rt_max) for candidate in candidates],
        "predicted_fragment_mzs": flat_fragment_mzs,
        "predicted_fragment_mz_offsets": fragment_mz_offsets,
        "predicted_fragment_mz_lengths": fragment_mz_lengths,
        "predicted_fragment_names": flat_fragment_names,
        "predicted_fragment_name_offsets": fragment_name_offsets,
        "predicted_fragment_name_lengths": fragment_name_lengths,
        "predicted_fragment_weights": flat_fragment_weights,
        "predicted_fragment_weight_offsets": fragment_weight_offsets,
        "predicted_fragment_weight_lengths": fragment_weight_lengths,
    }


def write_rust_stage2_partition_payload(
    output_path: str | os.PathLike[str],
    mzml_path: str,
    peptide_df: pd.DataFrame,
    sage_config: Dict[str, Any],
    mumdia_config: Dict[str, Any],
    ms2pip_predictions: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[pathlib.Path]:
    """Persist one minimal Rust Stage-2 partition payload as JSON."""
    payload = prepare_rust_stage2_partition_payload(
        mzml_path,
        peptide_df,
        sage_config,
        mumdia_config,
        ms2pip_predictions,
    )
    if payload is None:
        return None

    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return output


def _maybe_get_ms2pip_predictions(
    peptide_df: pd.DataFrame,
    sage_config: Dict[str, Any],
    mumdia_config: Dict[str, Any],
) -> Optional[Dict[str, Dict[str, float]]]:
    """Return MS2PIP predictions for RT-window candidates when enabled."""
    if not bool(mumdia_config.get("custom_engine_use_predicted_fragments", True)):
        return None
    if peptide_df.empty:
        return None

    ms2pip_input = build_ms2pip_prediction_input(peptide_df, sage_config)
    if ms2pip_input.is_empty():
        return None

    log_info(
        "Generating MS2PIP predictions for custom stage-2 backend: "
        f"{ms2pip_input.height} peptide/charge candidates"
    )
    return get_predictions_fragment_intensity(ms2pip_input)


def _build_candidate_index(
    candidates: Sequence[CandidateEntry],
    use_predicted_fragments: bool,
    fragment_bin_size_da: float,
) -> CandidateIndex:
    candidate_precursor_mz = np.asarray(
        [candidate.precursor_mz for candidate in candidates], dtype=float
    )
    sorted_order = np.argsort(candidate_precursor_mz)
    sorted_precursor_mz = candidate_precursor_mz[sorted_order]
    sorted_candidate_indices = sorted_order.astype(int)

    top_fragment_bins: Dict[int, set[int]] = {}
    if use_predicted_fragments:
        for candidate_idx, candidate in enumerate(candidates):
            for fragment_mz in candidate.predicted_fragment_mz:
                bin_idx = int(np.floor(float(fragment_mz) / fragment_bin_size_da))
                top_fragment_bins.setdefault(bin_idx, set()).add(candidate_idx)

    return CandidateIndex(
        sorted_precursor_mz=sorted_precursor_mz,
        sorted_candidate_indices=sorted_candidate_indices,
        top_fragment_bins={
            bin_idx: tuple(sorted(candidate_ids))
            for bin_idx, candidate_ids in top_fragment_bins.items()
        },
        fragment_bin_size_da=float(fragment_bin_size_da),
        use_predicted_fragments=use_predicted_fragments,
    )


def _window_id_from_bounds(lower_mz: float, upper_mz: float) -> str:
    """Create a stable identifier for a DIA isolation window."""
    return f"{lower_mz:.4f}|{upper_mz:.4f}"


def _window_bounds_from_spectrum_entry(
    spectrum_entry: Dict[str, Any],
) -> Optional[Tuple[float, float, str]]:
    """Extract lower/upper precursor bounds and window id from one MS2 spectrum."""
    target = spectrum_entry.get("isolation_window_target")
    lower_offset = spectrum_entry.get("isolation_window_lower")
    upper_offset = spectrum_entry.get("isolation_window_upper")
    if target is None or lower_offset is None or upper_offset is None:
        return None

    lower_bound = float(target) - float(lower_offset)
    upper_bound = float(target) + float(upper_offset)
    return lower_bound, upper_bound, _window_id_from_bounds(lower_bound, upper_bound)


def _window_specific_fragment_bins(
    candidate_indices: np.ndarray,
    candidates: Sequence[CandidateEntry],
    fragment_bin_size_da: float,
) -> Dict[int, Tuple[int, ...]]:
    """Build top-fragment bins restricted to one window's candidate subset."""
    top_fragment_bins: Dict[int, set[int]] = {}
    for candidate_idx in candidate_indices.tolist():
        for fragment_mz in candidates[int(candidate_idx)].predicted_fragment_mz:
            bin_idx = int(np.floor(float(fragment_mz) / fragment_bin_size_da))
            top_fragment_bins.setdefault(bin_idx, set()).add(int(candidate_idx))

    return {
        bin_idx: tuple(sorted(candidate_ids))
        for bin_idx, candidate_ids in top_fragment_bins.items()
    }


def _build_window_candidate_indices(
    candidates: Sequence[CandidateEntry],
    candidate_index: CandidateIndex,
    ms2_spectra: Dict[str, Dict[str, Any]],
) -> Dict[str, WindowCandidateIndex]:
    """Precompute candidate subsets and fragment bins for each DIA isolation window."""
    window_bounds: Dict[str, Tuple[float, float]] = {}
    for spectrum_entry in ms2_spectra.values():
        bounds = _window_bounds_from_spectrum_entry(spectrum_entry)
        if bounds is None:
            continue
        lower_bound, upper_bound, window_id = bounds
        window_bounds[window_id] = (lower_bound, upper_bound)

    window_candidate_indices: Dict[str, WindowCandidateIndex] = {}
    for window_id, (lower_bound, upper_bound) in window_bounds.items():
        left = int(
            np.searchsorted(
                candidate_index.sorted_precursor_mz, lower_bound, side="left"
            )
        )
        right = int(
            np.searchsorted(
                candidate_index.sorted_precursor_mz, upper_bound, side="right"
            )
        )
        candidate_indices = np.asarray(
            candidate_index.sorted_candidate_indices[left:right], dtype=int
        )
        if candidate_indices.size == 0:
            continue

        if candidate_index.use_predicted_fragments:
            top_fragment_bins = _window_specific_fragment_bins(
                candidate_indices,
                candidates,
                candidate_index.fragment_bin_size_da,
            )
        else:
            top_fragment_bins = {}

        window_candidate_indices[window_id] = WindowCandidateIndex(
            candidate_indices=candidate_indices,
            top_fragment_bins=top_fragment_bins,
        )

    return window_candidate_indices


def _rust_prefilter_candidate_indices(
    candidates: Sequence[CandidateEntry],
    ms2_spectra: Dict[str, Dict[str, Any]],
    fragment_tol_ppm: float,
    min_top_fragment_matches: int,
    fragment_bin_size_da: float,
) -> Dict[str, RustPrefilterResult]:
    """Use the Rust matcher to prefilter candidate ids per spectrum when available."""
    if (
        mumdia_rs is None
        or not candidates
        or not ms2_spectra
        or min_top_fragment_matches <= 0
    ):
        return {}

    candidate_precursor_mz = np.asarray(
        [candidate.precursor_mz for candidate in candidates], dtype=np.float64
    )
    candidate_fragment_offsets: List[int] = []
    candidate_fragment_lengths: List[int] = []
    candidate_fragment_flat: List[np.ndarray] = []
    current_offset = 0
    for candidate in candidates:
        fragment_mz = np.asarray(candidate.predicted_fragment_mz, dtype=np.float64)
        candidate_fragment_offsets.append(current_offset)
        candidate_fragment_lengths.append(int(fragment_mz.size))
        if fragment_mz.size > 0:
            candidate_fragment_flat.append(fragment_mz)
            current_offset += int(fragment_mz.size)

    spectrum_keys: List[str] = []
    spectrum_iso_lower: List[float] = []
    spectrum_iso_upper: List[float] = []
    spectrum_peak_offsets: List[int] = []
    spectrum_peak_lengths: List[int] = []
    spectrum_peak_flat: List[np.ndarray] = []
    current_peak_offset = 0
    for scannr, spectrum_entry in ms2_spectra.items():
        bounds = _window_bounds_from_spectrum_entry(spectrum_entry)
        if bounds is None:
            continue
        lower_bound, upper_bound, _ = bounds
        observed_mz = np.asarray(spectrum_entry.get("mz", []), dtype=np.float64)
        spectrum_keys.append(scannr)
        spectrum_iso_lower.append(lower_bound)
        spectrum_iso_upper.append(upper_bound)
        spectrum_peak_offsets.append(current_peak_offset)
        spectrum_peak_lengths.append(int(observed_mz.size))
        if observed_mz.size > 0:
            spectrum_peak_flat.append(observed_mz)
            current_peak_offset += int(observed_mz.size)

    if not spectrum_keys:
        return {}

    candidate_fragment_mz_flat = (
        np.concatenate(candidate_fragment_flat)
        if candidate_fragment_flat
        else np.asarray([], dtype=np.float64)
    )
    spectrum_peak_mz_flat = (
        np.concatenate(spectrum_peak_flat)
        if spectrum_peak_flat
        else np.asarray([], dtype=np.float64)
    )

    spectrum_idx, candidate_idx, matched_counts = mumdia_rs.prefilter_window_candidates(
        candidate_precursor_mz,
        candidate_fragment_mz_flat,
        np.asarray(candidate_fragment_offsets, dtype=np.uint64),
        np.asarray(candidate_fragment_lengths, dtype=np.uint64),
        np.asarray(spectrum_iso_lower, dtype=np.float64),
        np.asarray(spectrum_iso_upper, dtype=np.float64),
        spectrum_peak_mz_flat,
        np.asarray(spectrum_peak_offsets, dtype=np.uint64),
        np.asarray(spectrum_peak_lengths, dtype=np.uint64),
        float(fragment_tol_ppm),
        int(min_top_fragment_matches),
        float(fragment_bin_size_da),
    )

    hits_by_spectrum: Dict[str, Dict[int, int]] = {}
    for spectrum_position, candidate_position, match_count in zip(
        np.asarray(spectrum_idx, dtype=np.int64).tolist(),
        np.asarray(candidate_idx, dtype=np.int64).tolist(),
        np.asarray(matched_counts, dtype=np.int64).tolist(),
    ):
        scannr = spectrum_keys[int(spectrum_position)]
        per_spectrum = hits_by_spectrum.setdefault(scannr, {})
        candidate_position_int = int(candidate_position)
        per_spectrum[candidate_position_int] = max(
            per_spectrum.get(candidate_position_int, 0), int(match_count)
        )

    shortlisted: Dict[str, RustPrefilterResult] = {}
    for scannr, candidate_match_counts in hits_by_spectrum.items():
        spectrum_entry = ms2_spectra[scannr]
        target = spectrum_entry.get("isolation_window_target")
        ordered = sorted(
            candidate_match_counts.items(),
            key=lambda item: (
                -item[1],
                (
                    abs(candidates[item[0]].precursor_mz - float(target))
                    if target is not None
                    else 0.0
                ),
                item[0],
            ),
        )
        shortlisted[scannr] = RustPrefilterResult(
            candidate_indices=np.asarray(
                [candidate_id for candidate_id, _ in ordered], dtype=int
            ),
            matched_counts=np.asarray(
                [match_count for _, match_count in ordered], dtype=int
            ),
        )

    return shortlisted


def _fast_scored_candidates_from_prefilter(
    rust_result: RustPrefilterResult,
    candidates: Sequence[CandidateEntry],
    observed_intensity: np.ndarray,
    spectrum_entry: Dict[str, Any],
) -> List[ScoredCandidate]:
    """Build coarse scored candidates from Rust prefilter output only."""
    if rust_result.candidate_indices.size == 0:
        return []

    total_intensity = float(np.sum(observed_intensity))
    isolation_target = spectrum_entry.get("isolation_window_target")
    sorted_observed_intensity = np.sort(observed_intensity)
    scored_candidates: List[ScoredCandidate] = []

    for candidate_idx, matched_count in zip(
        rust_result.candidate_indices.tolist(),
        rust_result.matched_counts.tolist(),
    ):
        candidate = candidates[int(candidate_idx)]
        predicted_names = list(candidate.predicted_fragment_name[: int(matched_count)])
        predicted_weights = np.asarray(
            candidate.predicted_fragment_weight[: int(matched_count)], dtype=float
        )
        if not predicted_names:
            continue

        matched_fragments: List[Dict[str, Any]] = []
        matched_ordinals_b: List[int] = []
        matched_ordinals_y: List[int] = []
        for fragment_name, predicted_weight in zip(
            predicted_names,
            predicted_weights.tolist(),
        ):
            parsed = _parse_fragment_name(fragment_name)
            if parsed is None:
                continue
            fragment_type, ordinal, charge = parsed
            if fragment_type == "b":
                matched_ordinals_b.append(ordinal)
            elif fragment_type == "y":
                matched_ordinals_y.append(ordinal)
            matched_fragments.append(
                {
                    "fragment_type": fragment_type,
                    "fragment_ordinals": ordinal,
                    "fragment_charge": charge,
                    "fragment_intensity": float(predicted_weight),
                    "fragment_name": fragment_name,
                    "fragment_ppm": 0.0,
                    "predicted_weight": float(predicted_weight),
                }
            )

        if not matched_fragments:
            continue

        matched_intensity_sum = float(
            sorted_observed_intensity[-len(matched_fragments) :].sum()
        )
        matched_intensity_pct = (
            (matched_intensity_sum / total_intensity) * 100.0
            if total_intensity > 0.0
            else 0.0
        )
        total_predicted_weight = float(np.sum(candidate.predicted_fragment_weight))
        matched_predicted_weight_sum = float(np.sum(predicted_weights))
        matched_predicted_weight_fraction = (
            matched_predicted_weight_sum / total_predicted_weight
            if total_predicted_weight > 0.0
            else 0.0
        )
        precursor_ppm = 0.0
        precursor_distance = 0.0
        if isolation_target is not None and candidate.precursor_mz > 0:
            precursor_ppm = (
                (float(isolation_target) - candidate.precursor_mz)
                / candidate.precursor_mz
                * 1e6
            )
            precursor_distance = abs(float(isolation_target) - candidate.precursor_mz)

        score = float(
            10.0 * len(matched_fragments)
            + 5.0 * matched_predicted_weight_fraction
            + np.log1p(matched_intensity_sum)
            - precursor_distance
        )

        scored_candidates.append(
            ScoredCandidate(
                candidate=candidate,
                score=score,
                matched_peaks=len(matched_fragments),
                matched_intensity_pct=matched_intensity_pct,
                matched_intensity_sum=matched_intensity_sum,
                fragment_ppm=0.0,
                precursor_ppm=float(precursor_ppm),
                longest_b=_longest_consecutive_run(matched_ordinals_b),
                longest_y=_longest_consecutive_run(matched_ordinals_y),
                max_fragment_intensity=max(
                    float(fragment["fragment_intensity"])
                    for fragment in matched_fragments
                ),
                matched_predicted_top_fragments=len(matched_fragments),
                matched_predicted_weight_fraction=matched_predicted_weight_fraction,
                matched_fragments=matched_fragments,
            )
        )

    return scored_candidates


def _fast_psm_rows_from_prefilter(
    rust_result: RustPrefilterResult,
    candidates: Sequence[CandidateEntry],
    observed_intensity: np.ndarray,
    spectrum_entry: Dict[str, Any],
    filename: str,
    scannr: str,
    next_psm_id: int,
    report_psms: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Create minimal PSM rows directly from Rust prefilter output."""
    if rust_result.candidate_indices.size == 0:
        return [], next_psm_id

    total_intensity = float(np.sum(observed_intensity))
    isolation_target = spectrum_entry.get("isolation_window_target")
    sorted_observed_intensity = np.sort(observed_intensity)
    rt_value = float(spectrum_entry.get("retention_time", 0.0))
    top_n = min(report_psms, int(rust_result.candidate_indices.size))
    rows: List[Dict[str, Any]] = []
    score_values: List[float] = []

    for candidate_idx, matched_count in zip(
        rust_result.candidate_indices[:top_n].tolist(),
        rust_result.matched_counts[:top_n].tolist(),
    ):
        candidate = candidates[int(candidate_idx)]
        matched_count = max(1, int(matched_count))
        matched_weights = np.asarray(
            candidate.predicted_fragment_weight[:matched_count], dtype=float
        )
        matched_intensity_sum = float(sorted_observed_intensity[-matched_count:].sum())
        matched_intensity_pct = (
            (matched_intensity_sum / total_intensity) * 100.0
            if total_intensity > 0.0
            else 0.0
        )
        total_predicted_weight = float(np.sum(candidate.predicted_fragment_weight))
        matched_predicted_weight_fraction = (
            float(np.sum(matched_weights)) / total_predicted_weight
            if total_predicted_weight > 0.0
            else 0.0
        )
        precursor_ppm = 0.0
        precursor_distance = 0.0
        if isolation_target is not None and candidate.precursor_mz > 0:
            precursor_ppm = (
                (float(isolation_target) - candidate.precursor_mz)
                / candidate.precursor_mz
                * 1e6
            )
            precursor_distance = abs(float(isolation_target) - candidate.precursor_mz)
        score = float(
            10.0 * matched_count
            + 5.0 * matched_predicted_weight_fraction
            + np.log1p(matched_intensity_sum)
            - precursor_distance
        )
        score_values.append(score)
        rows.append(
            {
                "psm_id": next_psm_id,
                "filename": filename,
                "scannr": scannr,
                "peptide": candidate.peptide,
                "stripped_peptide": candidate.peptide,
                "proteins": candidate.proteins,
                "num_proteins": candidate.num_proteins,
                "rank": 0,
                "expmass": (
                    float(isolation_target) * candidate.charge
                    - PROTON_MASS * candidate.charge
                    if isolation_target is not None
                    else candidate.calcmass
                ),
                "calcmass": candidate.calcmass,
                "is_decoy": candidate.is_decoy,
                "charge": candidate.charge,
                "peptide_len": candidate.peptide_len,
                "missed_cleavages": candidate.missed_cleavages,
                "fragment_ppm": 0.0,
                "delta_next": 0.0,
                "delta_rt_model": 0.0,
                "matched_peaks": matched_count,
                "longest_b": matched_count,
                "longest_y": 0,
                "matched_intensity_pct": matched_intensity_pct,
                "fragment_intensity": (
                    float(np.max(matched_weights)) if matched_weights.size > 0 else 0.0
                ),
                "poisson": matched_predicted_weight_fraction,
                "spectrum_q": 1.0,
                "peptide_q": 1.0,
                "protein_q": 1.0,
                "rt": rt_value,
                "precursor_ppm": float(precursor_ppm),
                "hyperscore": score,
                "delta_best": 0.0,
            }
        )
        next_psm_id += 1

    best_score = score_values[0] if score_values else 0.0
    for index, row in enumerate(rows):
        row["rank"] = index + 1
        next_score = score_values[index + 1] if index + 1 < len(score_values) else 0.0
        row["delta_next"] = float(score_values[index] - next_score)
        row["delta_best"] = float(best_score - score_values[index])

    return rows, next_psm_id


def _shortlist_candidate_indices_by_top_fragments(
    candidate_indices: np.ndarray,
    candidates: Sequence[CandidateEntry],
    observed_mz: np.ndarray,
    candidate_index: CandidateIndex,
    fragment_tol_ppm: float,
    min_top_fragment_matches: int,
    top_fragment_bins_override: Optional[Dict[int, Tuple[int, ...]]] = None,
) -> np.ndarray:
    """Shortlist precursor-window candidates using top predicted MS2PIP fragments."""
    if (
        not candidate_index.use_predicted_fragments
        or candidate_indices.size == 0
        or observed_mz.size == 0
        or min_top_fragment_matches <= 0
    ):
        return candidate_indices

    candidate_set = {int(candidate_idx) for candidate_idx in candidate_indices.tolist()}
    matched_fragment_ids: Dict[int, set[int]] = {}
    top_fragment_bins = (
        top_fragment_bins_override
        if top_fragment_bins_override is not None
        else candidate_index.top_fragment_bins
    )

    for peak_mz in observed_mz:
        base_bin = int(np.floor(float(peak_mz) / candidate_index.fragment_bin_size_da))
        for bin_idx in (base_bin - 1, base_bin, base_bin + 1):
            for candidate_idx in top_fragment_bins.get(bin_idx, ()):
                if candidate_idx not in candidate_set:
                    continue
                predicted_mz = candidates[candidate_idx].predicted_fragment_mz
                if predicted_mz.size == 0:
                    continue
                tolerance = predicted_mz * fragment_tol_ppm * 1e-6
                matched_indices = np.flatnonzero(
                    np.abs(predicted_mz - float(peak_mz)) <= tolerance
                )
                if matched_indices.size == 0:
                    continue
                matched_fragment_ids.setdefault(candidate_idx, set()).update(
                    int(matched_index) for matched_index in matched_indices.tolist()
                )

    shortlisted = np.asarray(
        [
            candidate_idx
            for candidate_idx in candidate_indices.tolist()
            if len(matched_fragment_ids.get(int(candidate_idx), set()))
            >= min_top_fragment_matches
        ],
        dtype=int,
    )
    return shortlisted if shortlisted.size > 0 else candidate_indices


def _candidate_indices_for_spectrum(
    sorted_precursor_mz: np.ndarray,
    sorted_candidate_indices: np.ndarray,
    candidates: Sequence[CandidateEntry],
    spectrum_entry: Dict[str, Any],
    max_candidates_per_spectrum: int,
    window_candidate_indices: Optional[Dict[str, WindowCandidateIndex]] = None,
) -> np.ndarray:
    bounds = _window_bounds_from_spectrum_entry(spectrum_entry)
    target = spectrum_entry.get("isolation_window_target")

    if bounds is not None and window_candidate_indices is not None:
        _, _, window_id = bounds
        window_index = window_candidate_indices.get(window_id)
        candidate_indices = (
            window_index.candidate_indices
            if window_index is not None
            else np.asarray([], dtype=int)
        )
    elif bounds is not None:
        lower_bound, upper_bound, _ = bounds
        left = int(np.searchsorted(sorted_precursor_mz, lower_bound, side="left"))
        right = int(np.searchsorted(sorted_precursor_mz, upper_bound, side="right"))
        candidate_indices = sorted_candidate_indices[left:right]
    else:
        candidate_indices = sorted_candidate_indices

    if (
        max_candidates_per_spectrum > 0
        and candidate_indices.size > max_candidates_per_spectrum
        and target is not None
    ):
        candidate_indices = np.asarray(candidate_indices, dtype=int)
        distances = np.abs(
            np.asarray([candidates[idx].precursor_mz for idx in candidate_indices])
            - float(target)
        )
        top_order = np.argsort(distances)[:max_candidates_per_spectrum]
        candidate_indices = candidate_indices[top_order]

    return np.asarray(candidate_indices, dtype=int)


def _score_candidate_against_spectrum(
    candidate: CandidateEntry,
    observed_mz: np.ndarray,
    observed_intensity: np.ndarray,
    spectrum_entry: Dict[str, Any],
    fragment_tol_ppm: float,
    min_matched_peaks: int,
) -> Optional[ScoredCandidate]:
    if observed_mz.size == 0 or observed_intensity.size == 0:
        return None

    total_intensity = float(np.sum(observed_intensity))
    if total_intensity <= 0.0:
        return None

    matched_fragments: List[Dict[str, Any]] = []
    matched_ordinals_b: List[int] = []
    matched_ordinals_y: List[int] = []
    ppm_errors: List[float] = []
    predicted_weight_lookup = {
        fragment_name_value: float(weight_value)
        for fragment_name_value, weight_value in zip(
            candidate.predicted_fragment_name,
            candidate.predicted_fragment_weight,
        )
    }
    matched_predicted_top_fragments = 0
    matched_predicted_weight_sum = 0.0

    for idx, theoretical_mz in enumerate(candidate.fragment_mz):
        tolerance = theoretical_mz * fragment_tol_ppm * 1e-6
        left = int(
            np.searchsorted(observed_mz, theoretical_mz - tolerance, side="left")
        )
        right = int(
            np.searchsorted(observed_mz, theoretical_mz + tolerance, side="right")
        )
        if right <= left:
            continue

        local_slice = observed_intensity[left:right]
        if local_slice.size == 0:
            continue
        local_best_idx = left + int(np.argmax(local_slice))
        matched_mz = float(observed_mz[local_best_idx])
        matched_intensity = float(observed_intensity[local_best_idx])
        ppm_error = ((matched_mz - theoretical_mz) / theoretical_mz) * 1e6
        ppm_errors.append(ppm_error)

        fragment_type = candidate.fragment_type[idx]
        ordinal = int(candidate.fragment_ordinals[idx])
        charge = int(candidate.fragment_charge[idx])
        fragment_name = candidate.fragment_name[idx]

        if fragment_type == "b":
            matched_ordinals_b.append(ordinal)
        elif fragment_type == "y":
            matched_ordinals_y.append(ordinal)

        predicted_weight = predicted_weight_lookup.get(fragment_name, 0.0)
        if predicted_weight > 0.0:
            matched_predicted_top_fragments += 1
            matched_predicted_weight_sum += predicted_weight

        matched_fragments.append(
            {
                "fragment_type": fragment_type,
                "fragment_ordinals": ordinal,
                "fragment_charge": charge,
                "fragment_intensity": matched_intensity,
                "fragment_name": fragment_name,
                "fragment_ppm": ppm_error,
                "predicted_weight": predicted_weight,
            }
        )

    if len(matched_fragments) < min_matched_peaks:
        return None

    matched_intensity_sum = float(
        sum(fragment["fragment_intensity"] for fragment in matched_fragments)
    )
    matched_intensity_pct = (matched_intensity_sum / total_intensity) * 100.0
    total_predicted_weight = float(np.sum(candidate.predicted_fragment_weight))
    matched_predicted_weight_fraction = (
        matched_predicted_weight_sum / total_predicted_weight
        if total_predicted_weight > 0.0
        else 0.0
    )
    score = float(
        len(matched_fragments) * np.log1p(matched_intensity_sum)
        + 3.0 * matched_predicted_top_fragments
        + 5.0 * matched_predicted_weight_fraction
    )

    isolation_target = spectrum_entry.get("isolation_window_target")
    precursor_ppm = 0.0
    if isolation_target is not None and candidate.precursor_mz > 0:
        precursor_ppm = (
            (float(isolation_target) - candidate.precursor_mz)
            / candidate.precursor_mz
            * 1e6
        )

    return ScoredCandidate(
        candidate=candidate,
        score=score,
        matched_peaks=len(matched_fragments),
        matched_intensity_pct=matched_intensity_pct,
        matched_intensity_sum=matched_intensity_sum,
        fragment_ppm=float(np.mean(ppm_errors)) if ppm_errors else 0.0,
        precursor_ppm=float(precursor_ppm),
        longest_b=_longest_consecutive_run(matched_ordinals_b),
        longest_y=_longest_consecutive_run(matched_ordinals_y),
        max_fragment_intensity=max(
            float(fragment["fragment_intensity"]) for fragment in matched_fragments
        ),
        matched_predicted_top_fragments=matched_predicted_top_fragments,
        matched_predicted_weight_fraction=matched_predicted_weight_fraction,
        matched_fragments=matched_fragments,
    )


def _search_partition(
    mzml_path: str,
    peptide_df: pd.DataFrame,
    sage_config: Dict[str, Any],
    mumdia_config: Dict[str, Any],
    psm_ident_start: int,
    ms2pip_predictions: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, int]:
    top_n_predicted_fragments = int(
        mumdia_config.get("custom_engine_fragment_top_n", 3)
    )
    min_top_fragment_matches = int(
        mumdia_config.get("custom_engine_min_top_fragment_matches", 1)
    )
    fragment_bin_size_da = float(
        mumdia_config.get("custom_engine_fragment_bin_size_da", 0.05)
    )
    if ms2pip_predictions is None:
        ms2pip_predictions = _maybe_get_ms2pip_predictions(
            peptide_df,
            sage_config,
            mumdia_config,
        )
    candidates = _prepare_candidate_entries(
        peptide_df,
        sage_config,
        ms2pip_predictions=ms2pip_predictions,
        top_n_predicted_fragments=top_n_predicted_fragments,
    )
    if not candidates:
        return pl.DataFrame(), pl.DataFrame(), psm_ident_start

    candidate_index = _build_candidate_index(
        candidates,
        use_predicted_fragments=bool(
            ms2pip_predictions
            and mumdia_config.get("custom_engine_use_predicted_fragments", True)
        ),
        fragment_bin_size_da=fragment_bin_size_da,
    )

    fragment_tol_ppm = _get_ppm_tolerance(sage_config)
    min_matched_peaks = int(sage_config.get("min_matched_peaks", 5))
    report_psms = int(sage_config.get("report_psms", 12))
    min_peaks = int(sage_config.get("min_peaks", 0))
    max_candidates_per_spectrum = int(
        sage_config.get("custom_engine_max_candidates_per_spectrum", 2000)
    )
    skip_python_scoring = bool(
        mumdia_config.get("custom_engine_skip_python_scoring", False)
    )
    skip_fragment_rows = bool(
        mumdia_config.get("custom_engine_skip_fragment_rows", False)
    )
    top1_only = bool(mumdia_config.get("custom_engine_top1_only", False))
    effective_report_psms = 1 if top1_only else report_psms

    _, _, ms2_spectra = get_ms1_mzml(mzml_path)
    if not ms2_spectra:
        return pl.DataFrame(), pl.DataFrame(), psm_ident_start

    window_candidate_indices = _build_window_candidate_indices(
        candidates,
        candidate_index,
        ms2_spectra,
    )
    if window_candidate_indices:
        log_info(
            f"Custom stage-2 DIA window index for {pathlib.Path(mzml_path).name}: "
            f"{len(window_candidate_indices)} windows"
        )

    rust_prefiltered_candidates: Dict[str, RustPrefilterResult] = {}
    if candidate_index.use_predicted_fragments and mumdia_rs is not None:
        rust_prefiltered_candidates = _rust_prefilter_candidate_indices(
            candidates,
            ms2_spectra,
            fragment_tol_ppm=fragment_tol_ppm,
            min_top_fragment_matches=min_top_fragment_matches,
            fragment_bin_size_da=fragment_bin_size_da,
        )
        if rust_prefiltered_candidates:
            log_info(
                f"Custom stage-2 Rust prefilter for {pathlib.Path(mzml_path).name}: "
                f"{len(rust_prefiltered_candidates)} spectra with shortlisted candidates"
            )
            if skip_python_scoring:
                log_info(
                    "Custom stage-2 fast mode enabled: skipping full Python scoring "
                    "and using Rust-prefiltered candidates only"
                )

    filename = pathlib.Path(mzml_path).name
    psm_rows: List[Dict[str, Any]] = []
    fragment_rows: List[Dict[str, Any]] = []
    next_psm_id = int(psm_ident_start)

    for scannr, spectrum_entry in ms2_spectra.items():
        observed_mz = np.asarray(spectrum_entry.get("mz", []), dtype=float)
        observed_intensity = np.asarray(
            spectrum_entry.get("intensity", []), dtype=float
        )
        if observed_mz.size < min_peaks or observed_intensity.size < min_peaks:
            continue

        candidate_indices = _candidate_indices_for_spectrum(
            candidate_index.sorted_precursor_mz,
            candidate_index.sorted_candidate_indices,
            candidates,
            spectrum_entry,
            max_candidates_per_spectrum,
            window_candidate_indices=window_candidate_indices,
        )
        bounds = _window_bounds_from_spectrum_entry(spectrum_entry)
        top_fragment_bins_override = None
        if bounds is not None:
            _, _, window_id = bounds
            window_index = window_candidate_indices.get(window_id)
            if window_index is not None:
                top_fragment_bins_override = window_index.top_fragment_bins
        rust_result = rust_prefiltered_candidates.get(scannr)
        if skip_python_scoring:
            if rust_result is None or rust_result.candidate_indices.size == 0:
                continue
            if skip_fragment_rows:
                fast_rows, next_psm_id = _fast_psm_rows_from_prefilter(
                    rust_result,
                    candidates,
                    observed_intensity,
                    spectrum_entry,
                    filename,
                    scannr,
                    next_psm_id,
                    effective_report_psms,
                )
                if fast_rows:
                    psm_rows.extend(fast_rows)
                continue
            scored_candidates = _fast_scored_candidates_from_prefilter(
                rust_result,
                candidates,
                observed_intensity,
                spectrum_entry,
            )
        else:
            if rust_result is not None and rust_result.candidate_indices.size > 0:
                candidate_indices = np.asarray(rust_result.candidate_indices, dtype=int)
            else:
                candidate_indices = _shortlist_candidate_indices_by_top_fragments(
                    candidate_indices,
                    candidates,
                    observed_mz,
                    candidate_index,
                    fragment_tol_ppm,
                    min_top_fragment_matches,
                    top_fragment_bins_override=top_fragment_bins_override,
                )
            if candidate_indices.size == 0:
                continue

            isolation_target = spectrum_entry.get("isolation_window_target")
            if (
                max_candidates_per_spectrum > 0
                and candidate_indices.size > max_candidates_per_spectrum
                and isolation_target is not None
            ):
                distances = np.abs(
                    np.asarray(
                        [candidates[idx].precursor_mz for idx in candidate_indices]
                    )
                    - float(isolation_target)
                )
                top_order = np.argsort(distances)[:max_candidates_per_spectrum]
                candidate_indices = np.asarray(candidate_indices, dtype=int)[top_order]

            scored_candidates = []
            for candidate_idx in candidate_indices:
                scored = _score_candidate_against_spectrum(
                    candidates[int(candidate_idx)],
                    observed_mz,
                    observed_intensity,
                    spectrum_entry,
                    fragment_tol_ppm,
                    min_matched_peaks,
                )
                if scored is not None:
                    scored_candidates.append(scored)

        if not scored_candidates:
            continue

        scored_candidates.sort(key=lambda value: value.score, reverse=True)
        top_candidates = scored_candidates[:effective_report_psms]
        best_score = float(top_candidates[0].score)
        rt_value = float(spectrum_entry.get("retention_time", 0.0))
        isolation_target = spectrum_entry.get("isolation_window_target")
        for rank, scored in enumerate(top_candidates, start=1):
            next_score = (
                float(top_candidates[rank].score) if rank < len(top_candidates) else 0.0
            )
            if isolation_target is not None:
                expmass = (
                    float(isolation_target) * scored.candidate.charge
                    - PROTON_MASS * scored.candidate.charge
                )
            else:
                expmass = scored.candidate.calcmass
            psm_id = next_psm_id
            next_psm_id += 1

            psm_rows.append(
                {
                    "psm_id": psm_id,
                    "filename": filename,
                    "scannr": scannr,
                    "peptide": scored.candidate.peptide,
                    "stripped_peptide": scored.candidate.peptide,
                    "proteins": scored.candidate.proteins,
                    "num_proteins": scored.candidate.num_proteins,
                    "rank": rank,
                    "expmass": float(expmass),
                    "calcmass": scored.candidate.calcmass,
                    "is_decoy": scored.candidate.is_decoy,
                    "charge": scored.candidate.charge,
                    "peptide_len": scored.candidate.peptide_len,
                    "missed_cleavages": scored.candidate.missed_cleavages,
                    "fragment_ppm": scored.fragment_ppm,
                    "delta_next": float(scored.score - next_score),
                    "delta_rt_model": 0.0,
                    "matched_peaks": scored.matched_peaks,
                    "longest_b": scored.longest_b,
                    "longest_y": scored.longest_y,
                    "matched_intensity_pct": scored.matched_intensity_pct,
                    "fragment_intensity": scored.max_fragment_intensity,
                    "poisson": scored.matched_predicted_weight_fraction,
                    "spectrum_q": 1.0,
                    "peptide_q": 1.0,
                    "protein_q": 1.0,
                    "rt": rt_value,
                    "precursor_ppm": scored.precursor_ppm,
                    "hyperscore": scored.score,
                    "delta_best": float(best_score - scored.score),
                }
            )

            if not skip_fragment_rows:
                for fragment in scored.matched_fragments:
                    fragment_rows.append(
                        {
                            "psm_id": psm_id,
                            "fragment_type": fragment["fragment_type"],
                            "fragment_ordinals": fragment["fragment_ordinals"],
                            "fragment_charge": fragment["fragment_charge"],
                            "fragment_intensity": fragment["fragment_intensity"],
                            "fragment_name": fragment["fragment_name"],
                            "peptide": scored.candidate.peptide,
                            "charge": scored.candidate.charge,
                            "rt": rt_value,
                        }
                    )

    if not psm_rows:
        return pl.DataFrame(schema=_EMPTY_FRAGMENT_SCHEMA), pl.DataFrame(), next_psm_id

    fragment_df = (
        pl.DataFrame(fragment_rows)
        if fragment_rows
        else pl.DataFrame(schema=_EMPTY_FRAGMENT_SCHEMA)
    )
    return fragment_df, pl.DataFrame(psm_rows), next_psm_id


def _apply_target_decoy_qvalues(df_psms: pl.DataFrame) -> pl.DataFrame:
    if df_psms.is_empty() or "hyperscore" not in df_psms.columns:
        return df_psms

    scored = (
        df_psms.sort(["hyperscore", "matched_peaks"], descending=[True, True])
        .with_row_count("_score_rank")
        .with_columns(
            [
                pl.col("is_decoy").cast(pl.Int64).cum_sum().alias("_cum_decoys"),
                pl.when(pl.col("is_decoy"))
                .then(0)
                .otherwise(1)
                .cast(pl.Int64)
                .cum_sum()
                .alias("_cum_targets"),
            ]
        )
        .with_columns(
            (
                pl.col("_cum_decoys")
                / pl.when(pl.col("_cum_targets") > 0)
                .then(pl.col("_cum_targets"))
                .otherwise(1)
            ).alias("_fdr")
        )
    )

    fdr_values = scored["_fdr"].to_numpy()
    q_values = np.minimum.accumulate(fdr_values[::-1])[::-1]
    scored = scored.with_columns(pl.Series("spectrum_q", q_values))

    peptide_q = scored.group_by(["peptide", "charge"]).agg(
        pl.col("spectrum_q").min().alias("_peptide_q")
    )
    protein_q = scored.group_by(["proteins"]).agg(
        pl.col("spectrum_q").min().alias("_protein_q")
    )

    return (
        scored.drop(["_score_rank", "_cum_decoys", "_cum_targets", "_fdr"])
        .drop(["peptide_q", "protein_q"])
        .join(peptide_q, on=["peptide", "charge"], how="left")
        .join(protein_q, on=["proteins"], how="left")
        .with_columns(
            [
                pl.coalesce([pl.col("_peptide_q"), pl.col("spectrum_q")]).alias(
                    "peptide_q"
                ),
                pl.coalesce([pl.col("_protein_q"), pl.col("spectrum_q")]).alias(
                    "protein_q"
                ),
            ]
        )
        .drop(["_peptide_q", "_protein_q"])
    )


def _resolve_partition_workers(
    mumdia_config: Dict[str, Any],
    partition_count: int,
    using_native_backend: bool,
) -> int:
    """Determine how many mzML partitions to search concurrently."""
    if partition_count <= 1:
        return 1

    configured_workers = int(mumdia_config.get("custom_engine_partition_workers", 0))
    if configured_workers > 0:
        return min(configured_workers, partition_count)

    cpu_count = os.cpu_count() or 1
    if using_native_backend:
        auto_workers = max(1, cpu_count // 4)
        auto_workers = min(auto_workers, 4)
    else:
        auto_workers = max(1, cpu_count // 2)
        auto_workers = min(auto_workers, 8)
    return min(auto_workers, partition_count)


def _reindex_partition_psm_ids(
    df_fragment: pl.DataFrame,
    df_psms: pl.DataFrame,
    psm_ident_start: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, int]:
    """Assign globally unique PSM ids to one partition result."""
    if df_psms.is_empty():
        return df_fragment, df_psms, psm_ident_start

    old_psm_ids = [int(value) for value in df_psms["psm_id"].to_list()]
    new_psm_ids = list(range(psm_ident_start, psm_ident_start + len(old_psm_ids)))
    psm_id_map = pl.DataFrame({"_old_psm_id": old_psm_ids, "psm_id": new_psm_ids})
    psm_columns = df_psms.columns

    reindexed_psms = (
        df_psms.rename({"psm_id": "_old_psm_id"})
        .join(psm_id_map, on="_old_psm_id", how="left")
        .drop("_old_psm_id")
        .select(psm_columns)
    )

    if df_fragment.is_empty():
        reindexed_fragment = df_fragment
    else:
        fragment_columns = df_fragment.columns
        reindexed_fragment = (
            df_fragment.rename({"psm_id": "_old_psm_id"})
            .join(psm_id_map, on="_old_psm_id", how="left")
            .drop("_old_psm_id")
            .select(fragment_columns)
        )

    return reindexed_fragment, reindexed_psms, psm_ident_start + len(old_psm_ids)


def _search_partition_task(
    partition_index: int,
    upper_mzml_partition: float,
    mzml_path: str,
    sub_peptide_df: pd.DataFrame,
    config: Dict[str, Any],
    mumdia_config: Dict[str, Any],
    precomputed_ms2pip_predictions: Optional[Dict[str, Dict[str, float]]],
    use_rust_xic_backend: bool,
) -> PartitionSearchResult:
    """Execute one RT-partition search independently."""
    log_info(
        f"Custom stage-2 search for {pathlib.Path(mzml_path).name}: "
        f"{len(sub_peptide_df)} peptide rows"
    )

    if use_rust_xic_backend:
        df_fragment_part, df_psms_part, _ = _search_partition_rust_xic(
            mzml_path,
            sub_peptide_df,
            config["sage"],
            mumdia_config,
            0,
            ms2pip_predictions=precomputed_ms2pip_predictions,
        )
    else:
        df_fragment_part, df_psms_part, _ = _search_partition(
            mzml_path,
            sub_peptide_df,
            config["sage"],
            mumdia_config,
            0,
            ms2pip_predictions=precomputed_ms2pip_predictions,
        )

    return PartitionSearchResult(
        partition_index=partition_index,
        mzml_path=mzml_path,
        upper_mzml_partition=upper_mzml_partition,
        df_fragment=df_fragment_part,
        df_psms=df_psms_part,
    )


def retention_window_searches_custom(
    mzml_dict: Dict[float, str],
    peptide_df: pd.DataFrame,
    config: Dict[str, Any],
    rt_split_window: float,
    backend_context: Optional[Dict[str, Any]] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Run an experimental in-repo search backend on RT-split mzML partitions."""
    df_fragment_list: List[pl.DataFrame] = []
    df_psms_list: List[pl.DataFrame] = []
    mumdia_config = config.get("mumdia", {})
    precomputed_ms2pip_predictions = None
    if backend_context is not None:
        precomputed_ms2pip_predictions = backend_context.get("ms2pip_predictions")

    partitions_to_search: List[Tuple[int, float, str, pd.DataFrame]] = []
    for partition_index, (upper_mzml_partition, mzml_path) in enumerate(
        mzml_dict.items()
    ):
        peptide_selection_mask = np.maximum(
            peptide_df["predictions_lower"], upper_mzml_partition - rt_split_window
        ) <= np.minimum(peptide_df["predictions_upper"], upper_mzml_partition)
        sub_peptide_df = peptide_df[peptide_selection_mask].copy()
        if sub_peptide_df.empty:
            continue

        partitions_to_search.append(
            (partition_index, upper_mzml_partition, mzml_path, sub_peptide_df)
        )

    if not partitions_to_search:
        empty = pl.DataFrame()
        return empty, empty, empty, empty

    use_rust_xic_backend = (
        mumdia_rs is not None and precomputed_ms2pip_predictions is not None
    )
    partition_workers = _resolve_partition_workers(
        mumdia_config,
        len(partitions_to_search),
        using_native_backend=use_rust_xic_backend,
    )
    if partition_workers > 1:
        log_info(
            f"Custom stage-2 partition parallelism enabled: {partition_workers} "
            f"workers across {len(partitions_to_search)} mzML partitions"
        )

    partition_results: List[PartitionSearchResult] = []
    if partition_workers == 1:
        for (
            partition_index,
            upper_mzml_partition,
            mzml_path,
            sub_peptide_df,
        ) in partitions_to_search:
            partition_results.append(
                _search_partition_task(
                    partition_index,
                    upper_mzml_partition,
                    mzml_path,
                    sub_peptide_df,
                    config,
                    mumdia_config,
                    precomputed_ms2pip_predictions,
                    use_rust_xic_backend,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=partition_workers) as executor:
            futures = [
                executor.submit(
                    _search_partition_task,
                    partition_index,
                    upper_mzml_partition,
                    mzml_path,
                    sub_peptide_df,
                    config,
                    mumdia_config,
                    precomputed_ms2pip_predictions,
                    use_rust_xic_backend,
                )
                for (
                    partition_index,
                    upper_mzml_partition,
                    mzml_path,
                    sub_peptide_df,
                ) in partitions_to_search
            ]
            for future in as_completed(futures):
                partition_results.append(future.result())

    psm_ident_start = 0
    for partition_result in sorted(
        partition_results, key=lambda result: result.partition_index
    ):
        df_fragment_part, df_psms_part, psm_ident_start = _reindex_partition_psm_ids(
            partition_result.df_fragment,
            partition_result.df_psms,
            psm_ident_start,
        )
        if df_psms_part.is_empty():
            continue

        if not df_fragment_part.is_empty():
            df_fragment_list.append(df_fragment_part)
        df_psms_list.append(df_psms_part)

    if not df_psms_list:
        empty = pl.DataFrame()
        return empty, empty, empty, empty

    df_fragment = (
        pl.concat(df_fragment_list, how="diagonal_relaxed")
        if df_fragment_list
        else pl.DataFrame(schema=_EMPTY_FRAGMENT_SCHEMA)
    )
    df_psms = pl.concat(df_psms_list, how="diagonal_relaxed")
    if not bool(mumdia_config.get("custom_engine_skip_qvalues", False)):
        df_psms = _apply_target_decoy_qvalues(df_psms)

    df_fragment.write_csv(
        "debug/df_fragment_after_retention_window_searches.tsv", separator="\t"
    )
    df_psms.write_csv(
        "debug/df_psms_after_retention_window_searches.tsv", separator="\t"
    )

    if df_fragment.is_empty():
        df_fragment_max = pl.DataFrame(schema=_EMPTY_FRAGMENT_SCHEMA)
        df_fragment_max_peptide = pl.DataFrame(schema=_EMPTY_FRAGMENT_SCHEMA)
    else:
        df_fragment_max = (
            df_fragment.sort("fragment_intensity", descending=True)
            .unique(subset="psm_id", keep="first", maintain_order=True)
            .sort("psm_id")
        )
        df_fragment_max_peptide = df_fragment_max.unique(
            subset=["peptide", "charge"], keep="first", maintain_order=True
        ).sort("psm_id")

    return df_fragment, df_psms, df_fragment_max, df_fragment_max_peptide
