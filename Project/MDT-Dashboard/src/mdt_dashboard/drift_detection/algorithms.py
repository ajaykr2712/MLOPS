"""
Advanced statistical drift detection algorithms.
Implements multiple statistical tests for comprehensive drift detection.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings
import logging
from enum import Enum

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    """Drift severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DriftTestType(str, Enum):
    """Available drift detection tests."""
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    POPULATION_STABILITY_INDEX = "population_stability_index"
    JENSEN_SHANNON = "jensen_shannon"
    CHI_SQUARE = "chi_square"
    WASSERSTEIN = "wasserstein"


@dataclass(frozen=True)
class DriftResult:
    """Result of drift detection analysis."""
    
    test_name: str
    p_value: float
    statistic: float
    threshold: float
    is_drift: bool
    severity: DriftSeverity
    feature_name: Optional[str] = None
    reference_size: Optional[int] = None
    comparison_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test_name": self.test_name,
            "p_value": self.p_value,
            "statistic": self.statistic,
            "threshold": self.threshold,
            "is_drift": self.is_drift,
            "severity": self.severity.value,
            "feature_name": self.feature_name,
            "reference_size": self.reference_size,
            "comparison_size": self.comparison_size,
            "metadata": self.metadata
        }


class DriftDetectorProtocol(Protocol):
    """Protocol for drift detectors."""
    
    def detect(
        self, 
        reference: np.ndarray, 
        comparison: np.ndarray, 
        feature_name: Optional[str] = None
    ) -> DriftResult:
        """Detect drift between reference and comparison datasets."""
        ...


class BaseDriftDetector(ABC):
    """Base class for all drift detectors."""
    
    def __init__(self, threshold: float = 0.05, name: Optional[str] = None):
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def detect(
        self, 
        reference: np.ndarray, 
        comparison: np.ndarray, 
        feature_name: Optional[str] = None
    ) -> DriftResult:
        """Detect drift between reference and comparison datasets."""
        pass
    
    def _validate_inputs(
        self, 
        reference: np.ndarray, 
        comparison: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and clean input arrays."""
        # Convert to numpy arrays if needed
        ref_array = np.asarray(reference).flatten()
        comp_array = np.asarray(comparison).flatten()
        
        # Remove NaN values
        ref_clean = ref_array[~np.isnan(ref_array)]
        comp_clean = comp_array[~np.isnan(comp_array)]
        
        if len(ref_clean) == 0:
            raise ValueError("Reference data is empty after cleaning")
        if len(comp_clean) == 0:
            raise ValueError("Comparison data is empty after cleaning")
            
        return ref_clean, comp_clean
    
    def _calculate_severity(self, statistic: float, p_value: float) -> DriftSeverity:
        """Calculate drift severity based on statistical measures."""
        if p_value < 0.001 or statistic > 0.7:
            return DriftSeverity.HIGH
        elif p_value < 0.01 or statistic > 0.3:
            return DriftSeverity.MEDIUM
        elif p_value < self.threshold or statistic > 0.1:
            return DriftSeverity.LOW
        return DriftSeverity.NONE


class KolmogorovSmirnovDetector(BaseDriftDetector):
    """Kolmogorov-Smirnov two-sample test for continuous variables."""
    
    def __init__(self, threshold: float = 0.05, alternative: str = "two-sided"):
        super().__init__(threshold, "Kolmogorov-Smirnov")
        if alternative not in ["two-sided", "less", "greater"]:
            raise ValueError("Alternative must be 'two-sided', 'less', or 'greater'")
        self.alternative = alternative
    
    def detect(
        self, 
        reference: np.ndarray, 
        comparison: np.ndarray, 
        feature_name: Optional[str] = None
    ) -> DriftResult:
        """Apply KS test to detect distribution drift."""
        try:
            ref_clean, comp_clean = self._validate_inputs(reference, comparison)
            
            # Perform KS test
            statistic, p_value = stats.ks_2samp(
                ref_clean, 
                comp_clean, 
                alternative=self.alternative
            )
            
            is_drift = p_value < self.threshold
            severity = self._calculate_severity(statistic, p_value)
            
            return DriftResult(
                test_name=self.name,
                p_value=p_value,
                statistic=statistic,
                threshold=self.threshold,
                is_drift=is_drift,
                severity=severity,
                feature_name=feature_name,
                reference_size=len(reference),
                comparison_size=len(comparison),
                metadata={
                    "alternative": self.alternative,
                    "reference_mean": float(np.mean(ref_clean)),
                    "comparison_mean": float(np.mean(comp_clean)),
                    "reference_std": float(np.std(ref_clean)),
                    "comparison_std": float(np.std(comp_clean))
                }
            )
            
        except Exception as e:
            logger.error(f"KS test failed: {str(e)}")
            return DriftResult(
                test_name=self.name,
                p_value=1.0,
                statistic=0.0,
                threshold=self.threshold,
                is_drift=False,
                severity=DriftSeverity.NONE,
                feature_name=feature_name,
                reference_size=len(reference),
                comparison_size=len(comparison),
                metadata={"error": str(e)}
            )
        
        is_drift = p_value < self.threshold
        severity = self._calculate_severity(statistic, p_value)
        
        return DriftResult(
            test_name="Kolmogorov-Smirnov",
            p_value=p_value,
            statistic=statistic,
            threshold=self.threshold,
            is_drift=is_drift,
            severity=severity,
            feature_name=feature_name,
            reference_size=len(ref_clean),
            comparison_size=len(comp_clean),
            metadata={
                "reference_mean": float(np.mean(ref_clean)),
                "comparison_mean": float(np.mean(comp_clean)),
                "reference_std": float(np.std(ref_clean)),
                "comparison_std": float(np.std(comp_clean))
            }
        )


class PSIDetector(BaseDriftDetector):
    """Population Stability Index detector for categorical and binned continuous variables."""
    
    def __init__(self, threshold: float = 0.2, bins: int = 10):
        super().__init__(threshold)
        self.bins = bins
    
    def detect(self, reference: np.ndarray, comparison: np.ndarray, 
               feature_name: Optional[str] = None) -> DriftResult:
        """Calculate Population Stability Index."""
        try:
            # Handle categorical data
            if reference.dtype == 'object' or np.issubdtype(reference.dtype, np.integer):
                ref_counts = pd.Series(reference).value_counts(normalize=True, dropna=False)
                comp_counts = pd.Series(comparison).value_counts(normalize=True, dropna=False)
                
                # Align categories
                all_categories = set(ref_counts.index) | set(comp_counts.index)
                ref_dist = np.array([ref_counts.get(cat, 1e-6) for cat in all_categories])
                comp_dist = np.array([comp_counts.get(cat, 1e-6) for cat in all_categories])
            else:
                # Bin continuous data
                ref_clean = reference[~np.isnan(reference)]
                comp_clean = comparison[~np.isnan(comparison)]
                
                if len(ref_clean) == 0 or len(comp_clean) == 0:
                    raise ValueError("Empty datasets after cleaning")
                
                # Create bins based on reference data
                bin_edges = np.histogram_bin_edges(ref_clean, bins=self.bins)
                ref_counts, _ = np.histogram(ref_clean, bins=bin_edges, density=True)
                comp_counts, _ = np.histogram(comp_clean, bins=bin_edges, density=True)
                
                # Normalize to probabilities
                ref_dist = ref_counts / np.sum(ref_counts)
                comp_dist = comp_counts / np.sum(comp_counts)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-6
            ref_dist = np.maximum(ref_dist, epsilon)
            comp_dist = np.maximum(comp_dist, epsilon)
            
            # Calculate PSI
            psi = np.sum((comp_dist - ref_dist) * np.log(comp_dist / ref_dist))
            
            is_drift = psi > self.threshold
            severity = self._calculate_severity(psi, 1.0 - min(psi / self.threshold, 1.0))
            
            return DriftResult(
                test_name="Population Stability Index",
                p_value=1.0 - min(psi / self.threshold, 1.0),
                statistic=psi,
                threshold=self.threshold,
                is_drift=is_drift,
                severity=severity,
                feature_name=feature_name,
                reference_size=len(reference),
                comparison_size=len(comparison),
                metadata={
                    "bins": self.bins if reference.dtype != 'object' else len(set(reference)),
                    "reference_entropy": float(-np.sum(ref_dist * np.log(ref_dist))),
                    "comparison_entropy": float(-np.sum(comp_dist * np.log(comp_dist)))
                }
            )
            
        except Exception as e:
            logger.error(f"PSI calculation failed: {str(e)}")
            return DriftResult(
                test_name="Population Stability Index",
                p_value=1.0,
                statistic=0.0,
                threshold=self.threshold,
                is_drift=False,
                severity="none",
                feature_name=feature_name,
                reference_size=len(reference),
                comparison_size=len(comparison),
                metadata={"error": str(e)}
            )


class DriftDetectionSuite:
    """Comprehensive drift detection suite combining multiple algorithms."""
    
    def __init__(self, 
                 ks_threshold: float = 0.05,
                 psi_threshold: float = 0.2):
        
        self.detectors = {
            'ks': KolmogorovSmirnovDetector(ks_threshold),
            'psi': PSIDetector(psi_threshold)
        }
    
    def detect_drift(self, reference_data: Dict[str, np.ndarray], 
                    comparison_data: Dict[str, np.ndarray],
                    feature_types: Optional[Dict[str, str]] = None) -> Dict[str, List[DriftResult]]:
        """Detect drift across multiple features using appropriate algorithms."""
        results = {}
        
        for feature_name in reference_data.keys():
            if feature_name not in comparison_data:
                logger.warning(f"Feature {feature_name} not found in comparison data")
                continue
            
            ref_data = reference_data[feature_name]
            comp_data = comparison_data[feature_name]
            
            # Apply appropriate detectors
            feature_results = []
            
            # Apply KS and PSI tests
            feature_results.extend([
                self.detectors['ks'].detect(ref_data, comp_data, feature_name),
                self.detectors['psi'].detect(ref_data, comp_data, feature_name)
            ])
            
            results[feature_name] = feature_results
        
        return results
    """Kolmogorov-Smirnov test for continuous variables."""
    
    def detect(
        self,
        reference_data: np.ndarray,
        comparison_data: np.ndarray,
        feature_name: Optional[str] = None
    ) -> DriftResult:
        """Apply KS test for drift detection."""
        
        # Ensure 1D arrays
        ref_data = reference_data.flatten()
        comp_data = comparison_data.flatten()
        
        # Remove NaN values
        ref_data = ref_data[~np.isnan(ref_data)]
        comp_data = comp_data[~np.isnan(comp_data)]
        
        if len(ref_data) == 0 or len(comp_data) == 0:
            return DriftResult(
                test_name="kolmogorov_smirnov",
                p_value=1.0,
                statistic=0.0,
                threshold=self.threshold,
                is_drift=False,
                severity="none",
                feature_name=feature_name,
                reference_size=len(ref_data),
                comparison_size=len(comp_data),
                metadata={"error": "Empty data after NaN removal"}
            )
        
        # Perform KS test
        statistic, p_value = stats.ks_2samp(ref_data, comp_data)
        is_drift = p_value < self.threshold
        severity = self._calculate_severity(p_value, statistic)
        
        return DriftResult(
            test_name="kolmogorov_smirnov",
            p_value=p_value,
            statistic=statistic,
            threshold=self.threshold,
            is_drift=is_drift,
            severity=severity,
            feature_name=feature_name,
            reference_size=len(ref_data),
            comparison_size=len(comp_data),
            metadata={
                "critical_value": stats.ksone.ppf(1 - self.threshold, min(len(ref_data), len(comp_data))),
                "effect_size": statistic
            }
        )


class PopulationStabilityIndexDetector(BaseDriftDetector):
    """Population Stability Index (PSI) for categorical and binned continuous variables."""
    
    def __init__(self, threshold: float = 0.2, bins: int = 10):
        super().__init__(threshold)
        self.bins = bins
    
    def detect(
        self,
        reference_data: np.ndarray,
        comparison_data: np.ndarray,
        feature_name: Optional[str] = None
    ) -> DriftResult:
        """Calculate PSI between reference and comparison data."""
        
        ref_data = reference_data.flatten()
        comp_data = comparison_data.flatten()
        
        # Remove NaN values
        ref_data = ref_data[~np.isnan(ref_data)]
        comp_data = comp_data[~np.isnan(comp_data)]
        
        if len(ref_data) == 0 or len(comp_data) == 0:
            return DriftResult(
                test_name="population_stability_index",
                p_value=1.0,
                statistic=0.0,
                threshold=self.threshold,
                is_drift=False,
                severity="none",
                feature_name=feature_name,
                reference_size=len(ref_data),
                comparison_size=len(comp_data),
                metadata={"error": "Empty data after NaN removal"}
            )
        
        # Create bins based on reference data
        if len(np.unique(ref_data)) <= self.bins:
            # Categorical data
            ref_counts = pd.Series(ref_data).value_counts(normalize=True)
            comp_counts = pd.Series(comp_data).value_counts(normalize=True)
            
            # Align categories
            all_categories = set(ref_counts.index) | set(comp_counts.index)
            ref_probs = [ref_counts.get(cat, 1e-6) for cat in all_categories]
            comp_probs = [comp_counts.get(cat, 1e-6) for cat in all_categories]
        else:
            # Continuous data - create bins
            bin_edges = np.quantile(ref_data, np.linspace(0, 1, self.bins + 1))
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            ref_counts, _ = np.histogram(ref_data, bins=bin_edges)
            comp_counts, _ = np.histogram(comp_data, bins=bin_edges)
            
            ref_probs = ref_counts / len(ref_data)
            comp_probs = comp_counts / len(comp_data)
            
            # Avoid zero probabilities
            ref_probs = np.where(ref_probs == 0, 1e-6, ref_probs)
            comp_probs = np.where(comp_probs == 0, 1e-6, comp_probs)
        
        # Calculate PSI
        psi = np.sum((np.array(comp_probs) - np.array(ref_probs)) * 
                     np.log(np.array(comp_probs) / np.array(ref_probs)))
        
        is_drift = psi > self.threshold
        
        # PSI severity mapping
        if psi < 0.1:
            severity = "none"
        elif psi < 0.2:
            severity = "low"
        elif psi < 0.25:
            severity = "medium"
        else:
            severity = "high"
        
        return DriftResult(
            test_name="population_stability_index",
            p_value=1 - psi if psi <= 1 else 0.0,  # Approximate p-value
            statistic=psi,
            threshold=self.threshold,
            is_drift=is_drift,
            severity=severity,
            feature_name=feature_name,
            reference_size=len(ref_data),
            comparison_size=len(comp_data),
            metadata={
                "bins_used": self.bins,
                "interpretation": self._interpret_psi(psi)
            }
        )
    
    def _interpret_psi(self, psi: float) -> str:
        """Interpret PSI value."""
        if psi < 0.1:
            return "No significant population change"
        elif psi < 0.2:
            return "Moderate population change"
        else:
            return "Significant population change"


class ChiSquareDetector(BaseDriftDetector):
    """Chi-square test for categorical variables."""
    
    def detect(
        self,
        reference_data: np.ndarray,
        comparison_data: np.ndarray,
        feature_name: Optional[str] = None
    ) -> DriftResult:
        """Apply Chi-square test for categorical drift detection."""
        
        ref_data = reference_data.flatten()
        comp_data = comparison_data.flatten()
        
        # Get unique categories from both datasets
        all_categories = np.unique(np.concatenate([ref_data, comp_data]))
        
        # Create contingency table
        ref_counts = [(ref_data == cat).sum() for cat in all_categories]
        comp_counts = [(comp_data == cat).sum() for cat in all_categories]
        
        # Create 2x2 contingency table
        contingency_table = np.array([ref_counts, comp_counts])
        
        # Check if contingency table is valid
        if contingency_table.sum() == 0 or contingency_table.shape[1] < 2:
            return DriftResult(
                test_name="chi_square",
                p_value=1.0,
                statistic=0.0,
                threshold=self.threshold,
                is_drift=False,
                severity="none",
                feature_name=feature_name,
                reference_size=len(ref_data),
                comparison_size=len(comp_data),
                metadata={"error": "Invalid contingency table"}
            )
        
        # Perform Chi-square test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        is_drift = p_value < self.threshold
        severity = self._calculate_severity(p_value, chi2_stat)
        
        return DriftResult(
            test_name="chi_square",
            p_value=p_value,
            statistic=chi2_stat,
            threshold=self.threshold,
            is_drift=is_drift,
            severity=severity,
            feature_name=feature_name,
            reference_size=len(ref_data),
            comparison_size=len(comp_data),
            metadata={
                "degrees_of_freedom": dof,
                "expected_frequencies": expected.tolist(),
                "contingency_table": contingency_table.tolist()
            }
        )


class JensenShannonDetector(BaseDriftDetector):
    """Jensen-Shannon distance for distribution comparison."""
    
    def __init__(self, threshold: float = 0.1, bins: int = 50):
        super().__init__(threshold)
        self.bins = bins
    
    def detect(
        self,
        reference_data: np.ndarray,
        comparison_data: np.ndarray,
        feature_name: Optional[str] = None
    ) -> DriftResult:
        """Calculate Jensen-Shannon distance between distributions."""
        
        ref_data = reference_data.flatten()
        comp_data = comparison_data.flatten()
        
        # Remove NaN values
        ref_data = ref_data[~np.isnan(ref_data)]
        comp_data = comp_data[~np.isnan(comp_data)]
        
        if len(ref_data) == 0 or len(comp_data) == 0:
            return DriftResult(
                test_name="jensen_shannon",
                p_value=1.0,
                statistic=0.0,
                threshold=self.threshold,
                is_drift=False,
                severity="none",
                feature_name=feature_name,
                reference_size=len(ref_data),
                comparison_size=len(comp_data),
                metadata={"error": "Empty data after NaN removal"}
            )
        
        # Create common bins
        all_data = np.concatenate([ref_data, comp_data])
        bin_edges = np.linspace(all_data.min(), all_data.max(), self.bins + 1)
        
        # Calculate histograms
        ref_hist, _ = np.histogram(ref_data, bins=bin_edges)
        comp_hist, _ = np.histogram(comp_data, bins=bin_edges)
        
        # Normalize to probabilities
        ref_probs = ref_hist / ref_hist.sum()
        comp_probs = comp_hist / comp_hist.sum()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_probs = ref_probs + epsilon
        comp_probs = comp_probs + epsilon
        
        # Renormalize
        ref_probs = ref_probs / ref_probs.sum()
        comp_probs = comp_probs / comp_probs.sum()
        
        # Calculate Jensen-Shannon distance
        js_distance = jensenshannon(ref_probs, comp_probs)
        
        is_drift = js_distance > self.threshold
        
        # JS distance severity mapping
        if js_distance < 0.05:
            severity = "none"
        elif js_distance < 0.1:
            severity = "low"
        elif js_distance < 0.2:
            severity = "medium"
        else:
            severity = "high"
        
        return DriftResult(
            test_name="jensen_shannon",
            p_value=1 - js_distance,  # Approximate p-value
            statistic=js_distance,
            threshold=self.threshold,
            is_drift=is_drift,
            severity=severity,
            feature_name=feature_name,
            reference_size=len(ref_data),
            comparison_size=len(comp_data),
            metadata={
                "bins_used": self.bins,
                "interpretation": self._interpret_js(js_distance)
            }
        )
    
    def _interpret_js(self, js_distance: float) -> str:
        """Interpret Jensen-Shannon distance."""
        if js_distance < 0.05:
            return "Distributions are very similar"
        elif js_distance < 0.1:
            return "Moderate difference between distributions"
        else:
            return "Significant difference between distributions"


class MultivariateDriftDetector:
    """Multivariate drift detection using multiple algorithms."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.detectors = {
            "ks": KolmogorovSmirnovDetector(
                threshold=self.config.get("ks_threshold", 0.05)
            ),
            "psi": PopulationStabilityIndexDetector(
                threshold=self.config.get("psi_threshold", 0.2),
                bins=self.config.get("psi_bins", 10)
            ),
            "chi2": ChiSquareDetector(
                threshold=self.config.get("chi2_threshold", 0.05)
            ),
            "js": JensenShannonDetector(
                threshold=self.config.get("js_threshold", 0.1),
                bins=self.config.get("js_bins", 50)
            )
        }
    
    def detect_univariate(
        self,
        reference_data: Union[np.ndarray, pd.Series],
        comparison_data: Union[np.ndarray, pd.Series],
        feature_name: Optional[str] = None,
        data_type: str = "auto"
    ) -> List[DriftResult]:
        """Detect drift for a single feature using multiple tests."""
        
        if isinstance(reference_data, pd.Series):
            reference_data = reference_data.values
        if isinstance(comparison_data, pd.Series):
            comparison_data = comparison_data.values
        
        results = []
        
        # Auto-detect data type
        if data_type == "auto":
            unique_ratio = len(np.unique(reference_data)) / len(reference_data)
            data_type = "categorical" if unique_ratio < 0.1 else "continuous"
        
        # Apply appropriate tests based on data type
        if data_type == "continuous":
            # KS test for continuous data
            results.append(
                self.detectors["ks"].detect(reference_data, comparison_data, feature_name)
            )
            # PSI for binned continuous data
            results.append(
                self.detectors["psi"].detect(reference_data, comparison_data, feature_name)
            )
            # Jensen-Shannon distance
            results.append(
                self.detectors["js"].detect(reference_data, comparison_data, feature_name)
            )
        else:
            # Chi-square for categorical data
            results.append(
                self.detectors["chi2"].detect(reference_data, comparison_data, feature_name)
            )
            # PSI for categorical data
            results.append(
                self.detectors["psi"].detect(reference_data, comparison_data, feature_name)
            )
        
        return results
    
    def detect_multivariate(
        self,
        reference_df: pd.DataFrame,
        comparison_df: pd.DataFrame,
        feature_types: Optional[Dict[str, str]] = None
    ) -> Dict[str, List[DriftResult]]:
        """Detect drift for multiple features."""
        
        feature_types = feature_types or {}
        results = {}
        
        common_features = set(reference_df.columns) & set(comparison_df.columns)
        
        for feature in common_features:
            data_type = feature_types.get(feature, "auto")
            
            feature_results = self.detect_univariate(
                reference_df[feature],
                comparison_df[feature],
                feature_name=feature,
                data_type=data_type
            )
            
            results[feature] = feature_results
        
        return results
    
    def get_summary(self, results: Dict[str, List[DriftResult]]) -> Dict[str, Any]:
        """Generate summary statistics from drift detection results."""
        
        total_features = len(results)
        drift_features = set()
        severity_counts = {"none": 0, "low": 0, "medium": 0, "high": 0}
        test_results = {}
        
        for feature, feature_results in results.items():
            feature_has_drift = False
            max_severity = "none"
            
            for result in feature_results:
                test_name = result.test_name
                if test_name not in test_results:
                    test_results[test_name] = {"total": 0, "drift": 0}
                
                test_results[test_name]["total"] += 1
                
                if result.is_drift:
                    feature_has_drift = True
                    test_results[test_name]["drift"] += 1
                
                # Track maximum severity
                severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3}
                if severity_order[result.severity] > severity_order[max_severity]:
                    max_severity = result.severity
            
            if feature_has_drift:
                drift_features.add(feature)
            
            severity_counts[max_severity] += 1
        
        return {
            "total_features": total_features,
            "drift_features_count": len(drift_features),
            "drift_features": list(drift_features),
            "drift_percentage": len(drift_features) / total_features * 100 if total_features > 0 else 0,
            "severity_distribution": severity_counts,
            "test_summary": test_results,
            "overall_status": "drift_detected" if drift_features else "no_drift"
        }
