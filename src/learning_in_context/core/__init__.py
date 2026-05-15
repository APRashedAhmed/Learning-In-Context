"""Core utilities for In-Context CPD analysis."""

from .cache import CacheManager
from .config import Config
from .constants import *
from .data_types import (
    AnalysisResult,
    AnalysisResults,
    CriticalUnitsResult,
    ModelMetadata,
    ParticipantTrial,
    PipelineResult,
    StateData
)

__all__ = [
    # Classes
    "CacheManager",
    "Config",
    "AnalysisResult",
    "AnalysisResults", 
    "CriticalUnitsResult",
    "ModelMetadata",
    "ParticipantTrial",
    "PipelineResult",
    "StateData",
    # Constants
    "NUM_COLORS",
    "COLOR_NAMES",
    "TRIAL_TYPES",
    "HAZARD_RATES",
    "CONTINGENCIES",
    "DEFAULT_ALPHA",
    "ELASTICNET_L1_RATIO",
    "ELASTICNET_ALPHAS",
    "BOOTSTRAP_ITERATIONS",
    "CV_FOLDS",
    "PERMUTATION_ITERATIONS",
    "DATASET_CONFIGS",
    "DEFAULT_DATASET",
]