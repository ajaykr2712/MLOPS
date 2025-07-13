"""
Advanced data processing pipeline for MDT Dashboard.
Handles data ingestion, preprocessing, feature engineering, and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import logging
from pathlib import Path
import joblib
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings
import sqlite3
import psycopg2
from sqlalchemy import create_engine
import redis

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]
    duplicate_rows: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    outliers: Dict[str, int]
    data_types: Dict[str, str]
    summary_stats: Dict[str, Dict[str, float]]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "missing_values": self.missing_values,
            "missing_percentage": self.missing_percentage,
            "duplicate_rows": self.duplicate_rows,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "outliers": self.outliers,
            "data_types": self.data_types,
            "summary_stats": self.summary_stats,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass 
class ProcessingConfig:
    """Configuration for data processing pipeline."""
    
    # Missing value handling
    missing_strategy: str = "mean"  # mean, median, mode, forward_fill, knn
    missing_threshold: float = 0.5  # Drop columns with > 50% missing
    
    # Scaling
    scaling_method: str = "standard"  # standard, minmax, robust, none
    
    # Outlier detection
    outlier_method: str = "iqr"  # iqr, z_score, isolation_forest
    outlier_threshold: float = 3.0
    outlier_action: str = "clip"  # clip, remove, none
    
    # Feature engineering
    create_polynomial_features: bool = False
    polynomial_degree: int = 2
    create_interaction_features: bool = False
    
    # Validation
    train_size: float = 0.8
    validation_size: float = 0.1
    test_size: float = 0.1
    random_state: int = 42
    
    # Performance
    chunk_size: int = 10000
    use_parallel: bool = True
    n_jobs: int = -1


class BaseDataProcessor(ABC):
    """Base class for data processors."""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.is_fitted = False
        self.feature_names = None
        self.target_column = None
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, target_column: Optional[str] = None) -> 'BaseDataProcessor':
        """Fit the processor on training data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted processor."""
        pass
    
    def fit_transform(self, data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data, target_column).transform(data)


class DataLoader:
    """Universal data loader supporting multiple formats and sources."""
    
    def __init__(self, cache_enabled: bool = True, redis_client: Optional[redis.Redis] = None):
        self.cache_enabled = cache_enabled
        self.redis_client = redis_client
        
    def load_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            file_path = Path(file_path)
            cache_key = f"csv:{file_path.name}:{file_path.stat().st_mtime}"
            
            # Try cache first
            if self.cache_enabled and self.redis_client:
                cached_data = self._get_from_cache(cache_key)
                if cached_data is not None:
                    logger.info(f"Loaded {file_path.name} from cache")
                    return cached_data
            
            logger.info(f"Loading CSV file: {file_path}")
            data = pd.read_csv(file_path, **kwargs)
            
            # Cache the data
            if self.cache_enabled and self.redis_client:
                self._save_to_cache(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load CSV file {file_path}: {str(e)}")
            raise
    
    def load_database(self, connection_string: str, query: str, **kwargs) -> pd.DataFrame:
        """Load data from database."""
        try:
            logger.info("Executing database query")
            engine = create_engine(connection_string)
            
            cache_key = f"db:{hash(query)}"
            
            # Try cache first
            if self.cache_enabled and self.redis_client:
                cached_data = self._get_from_cache(cache_key)
                if cached_data is not None:
                    logger.info("Loaded query result from cache")
                    return cached_data
            
            data = pd.read_sql(query, engine, **kwargs)
            
            # Cache the data
            if self.cache_enabled and self.redis_client:
                self._save_to_cache(cache_key, data, ttl=3600)  # 1 hour TTL
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load from database: {str(e)}")
            raise
    
    def load_json(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from JSON file."""
        try:
            file_path = Path(file_path)
            logger.info(f"Loading JSON file: {file_path}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                return pd.DataFrame([data])
            else:
                raise ValueError("JSON data must be a list or dictionary")
                
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {str(e)}")
            raise
    
    def load_parquet(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from Parquet file."""
        try:
            file_path = Path(file_path)
            logger.info(f"Loading Parquet file: {file_path}")
            return pd.read_parquet(file_path)
            
        except Exception as e:
            logger.error(f"Failed to load Parquet file {file_path}: {str(e)}")
            raise
    
    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from Redis cache."""
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pd.read_json(cached_data.decode('utf-8'))
            return None
        except Exception as e:
            logger.warning(f"Cache read failed: {str(e)}")
            return None
    
    def _save_to_cache(self, key: str, data: pd.DataFrame, ttl: int = 7200):
        """Save data to Redis cache."""
        try:
            json_data = data.to_json()
            self.redis_client.setex(key, ttl, json_data)
        except Exception as e:
            logger.warning(f"Cache write failed: {str(e)}")


class DataQualityAnalyzer:
    """Comprehensive data quality analysis."""
    
    def __init__(self):
        self.report = None
    
    def analyze(self, data: pd.DataFrame) -> DataQualityReport:
        """Perform comprehensive data quality analysis."""
        logger.info("Starting data quality analysis")
        
        # Basic information
        total_rows, total_columns = data.shape
        
        # Missing values analysis
        missing_values = data.isnull().sum().to_dict()
        missing_percentage = (data.isnull().sum() / len(data) * 100).to_dict()
        
        # Duplicate rows
        duplicate_rows = data.duplicated().sum()
        
        # Column type analysis
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Data types
        data_types = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        # Outlier detection
        outliers = self._detect_outliers(data[numeric_columns])
        
        # Summary statistics
        summary_stats = {}
        for col in numeric_columns:
            summary_stats[col] = {
                'mean': float(data[col].mean()) if not data[col].empty else 0.0,
                'median': float(data[col].median()) if not data[col].empty else 0.0,
                'std': float(data[col].std()) if not data[col].empty else 0.0,
                'min': float(data[col].min()) if not data[col].empty else 0.0,
                'max': float(data[col].max()) if not data[col].empty else 0.0,
                'q25': float(data[col].quantile(0.25)) if not data[col].empty else 0.0,
                'q75': float(data[col].quantile(0.75)) if not data[col].empty else 0.0,
                'skewness': float(data[col].skew()) if not data[col].empty else 0.0,
                'kurtosis': float(data[col].kurtosis()) if not data[col].empty else 0.0
            }
        
        self.report = DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values=missing_values,
            missing_percentage=missing_percentage,
            duplicate_rows=duplicate_rows,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            outliers=outliers,
            data_types=data_types,
            summary_stats=summary_stats
        )
        
        logger.info("Data quality analysis completed")
        return self.report
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers using IQR method."""
        outliers = {}
        
        for column in data.columns:
            if data[column].dtype in [np.number]:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
                outliers[column] = int(outlier_count)
        
        return outliers


class DataPreprocessor(BaseDataProcessor):
    """Comprehensive data preprocessing pipeline."""
    
    def __init__(self, config: ProcessingConfig = None):
        super().__init__(config)
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.outlier_detectors = {}
        self.feature_selectors = {}
        
    def fit(self, data: pd.DataFrame, target_column: Optional[str] = None) -> 'DataPreprocessor':
        """Fit preprocessing pipeline on training data."""
        logger.info("Fitting data preprocessing pipeline")
        
        self.target_column = target_column
        self.feature_names = [col for col in data.columns if col != target_column]
        
        # Separate features and target
        X = data[self.feature_names] if target_column else data
        y = data[target_column] if target_column and target_column in data.columns else None
        
        # Handle missing values
        self._fit_imputers(X)
        X_imputed = self._transform_missing_values(X)
        
        # Handle outliers
        self._fit_outlier_detectors(X_imputed)
        X_outliers_handled = self._transform_outliers(X_imputed)
        
        # Encode categorical variables
        self._fit_encoders(X_outliers_handled)
        X_encoded = self._transform_categorical(X_outliers_handled)
        
        # Scale features
        self._fit_scalers(X_encoded)
        
        # Feature selection (if target is provided)
        if y is not None and target_column:
            self._fit_feature_selectors(X_encoded, y)
        
        self.is_fitted = True
        logger.info("Data preprocessing pipeline fitted successfully")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessing pipeline."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        logger.info("Transforming data using fitted preprocessing pipeline")
        
        # Select features
        X = data[self.feature_names] if self.feature_names else data
        
        # Apply transformations in order
        X_transformed = self._transform_missing_values(X)
        X_transformed = self._transform_outliers(X_transformed)
        X_transformed = self._transform_categorical(X_transformed)
        X_transformed = self._transform_scaling(X_transformed)
        
        # Apply feature selection if fitted
        if self.feature_selectors:
            X_transformed = self._transform_feature_selection(X_transformed)
        
        logger.info("Data transformation completed")
        return X_transformed
    
    def _fit_imputers(self, data: pd.DataFrame):
        """Fit imputers for missing value handling."""
        for column in data.columns:
            if data[column].isnull().sum() > 0:
                if data[column].dtype in [np.number]:
                    if self.config.missing_strategy == "mean":
                        imputer = SimpleImputer(strategy="mean")
                    elif self.config.missing_strategy == "median":
                        imputer = SimpleImputer(strategy="median")
                    elif self.config.missing_strategy == "knn":
                        imputer = KNNImputer(n_neighbors=5)
                    else:
                        imputer = SimpleImputer(strategy="mean")
                else:
                    imputer = SimpleImputer(strategy="most_frequent")
                
                imputer.fit(data[[column]])
                self.imputers[column] = imputer
    
    def _transform_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform missing values using fitted imputers."""
        data_transformed = data.copy()
        
        for column, imputer in self.imputers.items():
            if column in data_transformed.columns:
                data_transformed[column] = imputer.transform(data_transformed[[column]]).ravel()
        
        return data_transformed
    
    def _fit_outlier_detectors(self, data: pd.DataFrame):
        """Fit outlier detectors."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if self.config.outlier_method == "isolation_forest":
                detector = IsolationForest(contamination=0.1, random_state=self.config.random_state)
                detector.fit(data[[column]])
                self.outlier_detectors[column] = detector
    
    def _transform_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data."""
        data_transformed = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if self.config.outlier_method == "iqr":
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if self.config.outlier_action == "clip":
                    data_transformed[column] = data_transformed[column].clip(lower_bound, upper_bound)
                elif self.config.outlier_action == "remove":
                    outlier_mask = (data_transformed[column] >= lower_bound) & (data_transformed[column] <= upper_bound)
                    data_transformed = data_transformed[outlier_mask]
            
            elif column in self.outlier_detectors:
                detector = self.outlier_detectors[column]
                outlier_predictions = detector.predict(data_transformed[[column]])
                
                if self.config.outlier_action == "remove":
                    data_transformed = data_transformed[outlier_predictions == 1]
        
        return data_transformed
    
    def _fit_encoders(self, data: pd.DataFrame):
        """Fit encoders for categorical variables."""
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_columns:
            encoder = LabelEncoder()
            encoder.fit(data[column].astype(str))
            self.encoders[column] = encoder
    
    def _transform_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical variables using fitted encoders."""
        data_transformed = data.copy()
        
        for column, encoder in self.encoders.items():
            if column in data_transformed.columns:
                data_transformed[column] = encoder.transform(data_transformed[column].astype(str))
        
        return data_transformed
    
    def _fit_scalers(self, data: pd.DataFrame):
        """Fit scalers for numerical features."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if self.config.scaling_method == "standard":
            scaler_class = StandardScaler
        elif self.config.scaling_method == "minmax":
            scaler_class = MinMaxScaler
        elif self.config.scaling_method == "robust":
            scaler_class = RobustScaler
        else:
            return
        
        for column in numeric_columns:
            scaler = scaler_class()
            scaler.fit(data[[column]])
            self.scalers[column] = scaler
    
    def _transform_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to numerical features."""
        data_transformed = data.copy()
        
        for column, scaler in self.scalers.items():
            if column in data_transformed.columns:
                data_transformed[column] = scaler.transform(data_transformed[[column]]).ravel()
        
        return data_transformed
    
    def _fit_feature_selectors(self, X: pd.DataFrame, y: pd.Series):
        """Fit feature selectors."""
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            # Determine if regression or classification
            is_classification = y.dtype == 'object' or y.nunique() < 10
            
            if is_classification:
                selector = SelectKBest(score_func=f_classif, k='all')
            else:
                selector = SelectKBest(score_func=f_regression, k='all')
            
            selector.fit(X[numeric_columns], y)
            self.feature_selectors['statistical'] = selector
    
    def _transform_feature_selection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection."""
        # For now, return all features
        # In production, you might want to select top K features
        return data
    
    def save_pipeline(self, file_path: Union[str, Path]):
        """Save the fitted preprocessing pipeline."""
        pipeline_data = {
            'config': self.config.__dict__,
            'scalers': self.scalers,
            'imputers': self.imputers,
            'encoders': self.encoders,
            'outlier_detectors': self.outlier_detectors,
            'feature_selectors': self.feature_selectors,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'target_column': self.target_column
        }
        
        joblib.dump(pipeline_data, file_path)
        logger.info(f"Preprocessing pipeline saved to {file_path}")
    
    def load_pipeline(self, file_path: Union[str, Path]):
        """Load a fitted preprocessing pipeline."""
        pipeline_data = joblib.load(file_path)
        
        self.config = ProcessingConfig(**pipeline_data['config'])
        self.scalers = pipeline_data['scalers']
        self.imputers = pipeline_data['imputers']
        self.encoders = pipeline_data['encoders']
        self.outlier_detectors = pipeline_data['outlier_detectors']
        self.feature_selectors = pipeline_data['feature_selectors']
        self.is_fitted = pipeline_data['is_fitted']
        self.feature_names = pipeline_data['feature_names']
        self.target_column = pipeline_data['target_column']
        
        logger.info(f"Preprocessing pipeline loaded from {file_path}")


class FeatureEngineer:
    """Advanced feature engineering capabilities."""
    
    def __init__(self):
        self.feature_transformers = {}
    
    def create_temporal_features(self, data: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame:
        """Create temporal features from datetime columns."""
        data_transformed = data.copy()
        
        for col in datetime_columns:
            if col in data.columns:
                # Convert to datetime if not already
                data_transformed[col] = pd.to_datetime(data_transformed[col])
                
                # Extract temporal features
                data_transformed[f'{col}_year'] = data_transformed[col].dt.year
                data_transformed[f'{col}_month'] = data_transformed[col].dt.month
                data_transformed[f'{col}_day'] = data_transformed[col].dt.day
                data_transformed[f'{col}_dayofweek'] = data_transformed[col].dt.dayofweek
                data_transformed[f'{col}_hour'] = data_transformed[col].dt.hour
                data_transformed[f'{col}_is_weekend'] = (data_transformed[col].dt.dayofweek >= 5).astype(int)
                
        return data_transformed
    
    def create_aggregated_features(self, data: pd.DataFrame, group_by_columns: List[str], 
                                 agg_columns: List[str], agg_functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """Create aggregated features based on grouping."""
        data_transformed = data.copy()
        
        for group_col in group_by_columns:
            for agg_col in agg_columns:
                for agg_func in agg_functions:
                    try:
                        feature_name = f'{agg_col}_{agg_func}_by_{group_col}'
                        agg_values = data.groupby(group_col)[agg_col].transform(agg_func)
                        data_transformed[feature_name] = agg_values
                    except Exception as e:
                        logger.warning(f"Failed to create aggregated feature {feature_name}: {str(e)}")
        
        return data_transformed
    
    def create_ratio_features(self, data: pd.DataFrame, numerator_cols: List[str], 
                            denominator_cols: List[str]) -> pd.DataFrame:
        """Create ratio features between columns."""
        data_transformed = data.copy()
        
        for num_col in numerator_cols:
            for den_col in denominator_cols:
                if num_col != den_col and num_col in data.columns and den_col in data.columns:
                    feature_name = f'{num_col}_ratio_{den_col}'
                    # Avoid division by zero
                    data_transformed[feature_name] = data_transformed[num_col] / (data_transformed[den_col] + 1e-8)
        
        return data_transformed


@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline."""
    
    # Missing value handling
    missing_strategy: str = "median"  # mean, median, mode, knn, drop
    missing_threshold: float = 0.5  # Drop columns with missing > threshold
    
    # Scaling
    scaling_method: str = "standard"  # standard, minmax, robust, none
    
    # Outlier detection
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0
    remove_outliers: bool = False
    
    # Feature selection
    feature_selection: bool = False
    n_features: Optional[int] = None
    selection_method: str = "mutual_info"  # f_test, mutual_info, variance
    
    # Categorical encoding
    encoding_method: str = "onehot"  # onehot, label, target
    handle_unknown: str = "ignore"
    
    # Validation
    test_size: float = 0.2
    random_state: int = 42
    stratify_column: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "missing_strategy": self.missing_strategy,
            "missing_threshold": self.missing_threshold,
            "scaling_method": self.scaling_method,
            "outlier_method": self.outlier_method,
            "outlier_threshold": self.outlier_threshold,
            "remove_outliers": self.remove_outliers,
            "feature_selection": self.feature_selection,
            "n_features": self.n_features,
            "selection_method": self.selection_method,
            "encoding_method": self.encoding_method,
            "handle_unknown": self.handle_unknown,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "stratify_column": self.stratify_column
        }


class BaseDataProcessor(ABC):
    """Base class for data processors."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.is_fitted = False
        self.feature_names = None
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'BaseDataProcessor':
        """Fit the processor to training data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted processor."""
        pass
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(data, target).transform(data)


class DataQualityAssessor:
    """Comprehensive data quality assessment."""
    
    @staticmethod
    def assess(data: pd.DataFrame) -> DataQualityReport:
        """Generate comprehensive data quality report."""
        
        # Basic info
        total_rows, total_columns = data.shape
        
        # Missing values
        missing_values = data.isnull().sum().to_dict()
        missing_percentage = (data.isnull().sum() / len(data) * 100).to_dict()
        
        # Duplicate rows
        duplicate_rows = data.duplicated().sum()
        
        # Column types
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Data types
        data_types = data.dtypes.astype(str).to_dict()
        
        # Outliers (for numeric columns only)
        outliers = {}
        for col in numeric_columns:
            if data[col].notna().sum() > 0:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        
        # Summary statistics
        summary_stats = {}
        for col in numeric_columns:
            if data[col].notna().sum() > 0:
                summary_stats[col] = {
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "median": float(data[col].median()),
                    "skewness": float(data[col].skew()),
                    "kurtosis": float(data[col].kurtosis())
                }
        
        return DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values=missing_values,
            missing_percentage=missing_percentage,
            duplicate_rows=duplicate_rows,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            outliers=outliers,
            data_types=data_types,
            summary_stats=summary_stats
        )


class MissingValueHandler(BaseDataProcessor):
    """Handle missing values with various strategies."""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.imputer = None
        self.columns_to_drop = []
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'MissingValueHandler':
        """Fit missing value handler."""
        
        # Identify columns to drop based on missing threshold
        missing_ratio = data.isnull().sum() / len(data)
        self.columns_to_drop = missing_ratio[missing_ratio > self.config.missing_threshold].index.tolist()
        
        # Prepare data for imputation
        data_for_imputation = data.drop(columns=self.columns_to_drop)
        
        # Select imputation strategy
        if self.config.missing_strategy == "mean":
            self.imputer = SimpleImputer(strategy="mean")
        elif self.config.missing_strategy == "median":
            self.imputer = SimpleImputer(strategy="median")
        elif self.config.missing_strategy == "mode":
            self.imputer = SimpleImputer(strategy="most_frequent")
        elif self.config.missing_strategy == "knn":
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            # For 'drop' strategy, we'll handle it in transform
            self.imputer = None
        
        # Fit imputer if needed
        if self.imputer is not None and not data_for_imputation.empty:
            # Only fit on numeric columns for mean/median
            if self.config.missing_strategy in ["mean", "median"]:
                numeric_cols = data_for_imputation.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.imputer.fit(data_for_imputation[numeric_cols])
            else:
                self.imputer.fit(data_for_imputation)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by handling missing values."""
        
        if not self.is_fitted:
            raise ValueError("Handler must be fitted before transform")
        
        # Drop columns with high missing ratio
        data_clean = data.drop(columns=self.columns_to_drop, errors='ignore')
        
        if self.config.missing_strategy == "drop":
            # Drop rows with any missing values
            return data_clean.dropna()
        
        if self.imputer is None:
            return data_clean
        
        # Apply imputation
        if self.config.missing_strategy in ["mean", "median"]:
            numeric_cols = data_clean.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data_clean[numeric_cols] = self.imputer.transform(data_clean[numeric_cols])
        else:
            data_clean = pd.DataFrame(
                self.imputer.transform(data_clean),
                columns=data_clean.columns,
                index=data_clean.index
            )
        
        return data_clean


class OutlierDetector(BaseDataProcessor):
    """Detect and optionally remove outliers."""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.outlier_bounds = {}
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'OutlierDetector':
        """Fit outlier detector."""
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if data[col].notna().sum() > 0:
                if self.config.outlier_method == "iqr":
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    self.outlier_bounds[col] = (lower_bound, upper_bound)
                
                elif self.config.outlier_method == "zscore":
                    mean = data[col].mean()
                    std = data[col].std()
                    lower_bound = mean - self.config.outlier_threshold * std
                    upper_bound = mean + self.config.outlier_threshold * std
                    self.outlier_bounds[col] = (lower_bound, upper_bound)
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by handling outliers."""
        
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before transform")
        
        data_clean = data.copy()
        
        if self.config.remove_outliers:
            # Remove outliers
            mask = pd.Series([True] * len(data), index=data.index)
            
            for col, (lower, upper) in self.outlier_bounds.items():
                if col in data.columns:
                    col_mask = (data[col] >= lower) & (data[col] <= upper)
                    mask = mask & col_mask
            
            data_clean = data_clean[mask]
        else:
            # Clip outliers
            for col, (lower, upper) in self.outlier_bounds.items():
                if col in data.columns:
                    data_clean[col] = data_clean[col].clip(lower=lower, upper=upper)
        
        return data_clean
    
    def detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return boolean mask of outliers."""
        
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detecting outliers")
        
        outlier_mask = pd.DataFrame(False, index=data.index, columns=data.columns)
        
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in data.columns:
                outlier_mask[col] = (data[col] < lower) | (data[col] > upper)
        
        return outlier_mask


class FeatureScaler(BaseDataProcessor):
    """Scale features using various methods."""
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.scaler = None
        self.numeric_columns = None
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'FeatureScaler':
        """Fit feature scaler."""
        
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.config.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.config.scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None
        
        if self.scaler is not None and len(self.numeric_columns) > 0:
            self.scaler.fit(data[self.numeric_columns])
        
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data by scaling features."""
        
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        data_scaled = data.copy()
        
        if self.scaler is not None and len(self.numeric_columns) > 0:
            available_numeric_cols = [col for col in self.numeric_columns if col in data.columns]
            if available_numeric_cols:
                data_scaled[available_numeric_cols] = self.scaler.transform(data[available_numeric_cols])
        
        return data_scaled


class ComprehensiveDataProcessor:
    """Complete data processing pipeline."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.missing_handler = MissingValueHandler(self.config)
        self.outlier_detector = OutlierDetector(self.config)
        self.scaler = FeatureScaler(self.config)
        self.is_fitted = False
        self.feature_names = None
        self.quality_report = None
    
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> 'ComprehensiveDataProcessor':
        """Fit the complete processing pipeline."""
        
        logger.info("Starting data processing pipeline fitting")
        
        # Generate quality report
        self.quality_report = DataQualityAssessor.assess(data)
        logger.info(f"Data quality assessment completed: {self.quality_report.total_rows} rows, "
                   f"{self.quality_report.total_columns} columns")
        
        # Fit processors in sequence
        data_step1 = self.missing_handler.fit_transform(data, target)
        data_step2 = self.outlier_detector.fit_transform(data_step1, target)
        data_step3 = self.scaler.fit_transform(data_step2, target)
        
        self.feature_names = data_step3.columns.tolist()
        self.is_fitted = True
        
        logger.info("Data processing pipeline fitted successfully")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted pipeline."""
        
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        # Apply transformations in sequence
        data_transformed = self.missing_handler.transform(data)
        data_transformed = self.outlier_detector.transform(data_transformed)
        data_transformed = self.scaler.transform(data_transformed)
        
        return data_transformed
    
    def fit_transform(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(data, target).transform(data)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save the fitted processor."""
        
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted processor")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        processor_data = {
            "config": self.config.to_dict(),
            "missing_handler": self.missing_handler,
            "outlier_detector": self.outlier_detector,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "quality_report": self.quality_report.to_dict() if self.quality_report else None,
            "is_fitted": self.is_fitted
        }
        
        joblib.dump(processor_data, filepath)
        logger.info(f"Data processor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ComprehensiveDataProcessor':
        """Load a fitted processor."""
        
        processor_data = joblib.load(filepath)
        
        # Reconstruct the processor
        config = ProcessingConfig(**processor_data["config"])
        processor = cls(config)
        
        processor.missing_handler = processor_data["missing_handler"]
        processor.outlier_detector = processor_data["outlier_detector"]
        processor.scaler = processor_data["scaler"]
        processor.feature_names = processor_data["feature_names"]
        processor.is_fitted = processor_data["is_fitted"]
        
        if processor_data["quality_report"]:
            # Reconstruct quality report
            report_data = processor_data["quality_report"]
            report_data["timestamp"] = datetime.fromisoformat(report_data["timestamp"])
            processor.quality_report = DataQualityReport(**report_data)
        
        logger.info(f"Data processor loaded from {filepath}")
        return processor
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about processed features."""
        
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted to get feature info")
        
        return {
            "total_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "numeric_features": len(self.scaler.numeric_columns) if self.scaler.numeric_columns else 0,
            "dropped_features": len(self.missing_handler.columns_to_drop),
            "outlier_bounds": self.outlier_detector.outlier_bounds,
            "quality_report": self.quality_report.to_dict() if self.quality_report else None
        }


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    return X_train, X_test, y_train, y_test


def validate_data_schema(data: pd.DataFrame, expected_schema: Dict[str, str]) -> List[str]:
    """Validate data against expected schema."""
    
    errors = []
    
    # Check for missing columns
    missing_columns = set(expected_schema.keys()) - set(data.columns)
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
    
    # Check data types
    for col, expected_type in expected_schema.items():
        if col in data.columns:
            actual_type = str(data[col].dtype)
            if expected_type not in actual_type:
                errors.append(f"Column '{col}' has type '{actual_type}', expected '{expected_type}'")
    
    return errors