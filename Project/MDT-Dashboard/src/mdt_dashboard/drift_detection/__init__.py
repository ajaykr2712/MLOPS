"""
Drift detection module for MDT Dashboard.
Provides statistical and machine learning-based drift detection algorithms.
"""

from .algorithms import (
    DriftResult,
    BaseDriftDetector,
    KolmogorovSmirnovDetector,
    PopulationStabilityIndexDetector,
    MultivariateDriftDetector,
    DriftDetectionSuite
)

__all__ = [
    "DriftResult",
    "BaseDriftDetector", 
    "KolmogorovSmirnovDetector",
    "PopulationStabilityIndexDetector",
    "MultivariateDriftDetector",
    "DriftDetectionSuite"
]

__version__ = "1.0.0"
