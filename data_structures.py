"""
Shared data structures for MuMDIA.

This module contains dataclass definitions used across multiple modules
to avoid circular import issues.
"""

from dataclasses import dataclass, field
from typing import Dict
import numpy as np


@dataclass
class CorrelationResults:
    """Results from fragment correlation analysis."""
    correlations: np.ndarray
    correlations_count: np.ndarray
    sum_pred_frag_intens: np.ndarray
    correlation_matrix_psm_ids: np.ndarray
    correlation_matrix_frag_ids: np.ndarray
    correlation_matrix_psm_ids_ignore_zeros: np.ndarray
    correlation_matrix_psm_ids_ignore_zeros_counts: np.ndarray
    correlation_matrix_psm_ids_missing: np.ndarray
    correlation_matrix_psm_ids_missing_zeros_counts: np.ndarray
    correlation_matrix_frag_ids_ignore_zeros: np.ndarray
    correlation_matrix_frag_ids_ignore_zeros_counts: np.ndarray
    correlation_matrix_frag_ids_missing: np.ndarray
    correlation_matrix_frag_ids_missing_zeros_counts: np.ndarray
    most_intens_cor: float
    most_intens_cos: float
    mse_avg_pred_intens: float
    mse_avg_pred_intens_total: float


@dataclass
class PickleConfig:
    """Configuration for pickle file caching."""
    write_deeplc: bool = False
    write_ms2pip: bool = False
    write_correlation: bool = False
    read_deeplc: bool = False
    read_ms2pip: bool = False
    read_correlation: bool = False


@dataclass 
class SpectraData:
    """Container for spectral data dictionaries."""
    ms1_dict: Dict = field(default_factory=dict)
    ms2_to_ms1_dict: Dict = field(default_factory=dict)
    ms2_dict: Dict = field(default_factory=dict)
