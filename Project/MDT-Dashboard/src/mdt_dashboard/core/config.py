"""
Core configuration management for MDT Dashboard.
Enterprise-grade configuration with environment-based settings.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseSettings, Field, validator
from functools import lru_cache
from enum import Enum


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class BaseConfig(BaseSettings):
    """Base configuration class with common functionality."""
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "forbid"  # Prevent extra fields
        
    def dict_safe(self) -> dict:
        """Return dictionary with sensitive fields masked."""
        data = self.dict()
        sensitive_fields = {'password', 'secret', 'key', 'token'}
        
        def mask_sensitive(obj: dict, path: str = "") -> dict:
            if isinstance(obj, dict):
                return {
                    k: mask_sensitive(v, f"{path}.{k}" if path else k)
                    for k, v in obj.items()
                }
            elif any(sensitive in str(path).lower() for sensitive in sensitive_fields):
                return "***MASKED***" if obj else None
            else:
                return obj
                
        return mask_sensitive(data)


class DatabaseSettings(BaseConfig):
    """Database configuration settings."""
    
    url: str = Field(
        default="postgresql://mdt_user:mdt_pass@localhost:5432/mdt_db",
        env="DATABASE_URL",
        description="Database connection URL"
    )
    echo: bool = Field(
        default=False, 
        env="DATABASE_ECHO",
        description="Enable SQL query echoing"
    )
    pool_size: int = Field(
        default=20, 
        env="DATABASE_POOL_SIZE",
        ge=1,
        le=100,
        description="Database connection pool size"
    )
    max_overflow: int = Field(
        default=30, 
        env="DATABASE_MAX_OVERFLOW",
        ge=0,
        le=100,
        description="Maximum connection pool overflow"
    )
    
    class Config(BaseConfig.Config):
        env_prefix = "DB_"


class RedisSettings(BaseConfig):
    """Redis configuration for caching and task queue."""
    
    url: str = Field(
        default="redis://localhost:6379/0", 
        env="REDIS_URL",
        description="Redis connection URL"
    )
    max_connections: int = Field(
        default=20, 
        env="REDIS_MAX_CONNECTIONS",
        ge=1,
        le=100,
        description="Maximum Redis connections"
    )
    decode_responses: bool = Field(
        default=True, 
        env="REDIS_DECODE_RESPONSES",
        description="Decode Redis responses"
    )
    
    class Config(BaseConfig.Config):
        env_prefix = "REDIS_"


class MonitoringSettings(BaseConfig):
    """Monitoring and observability settings."""
    
    prometheus_port: int = Field(
        default=8000, 
        env="PROMETHEUS_PORT",
        ge=1024,
        le=65535,
        description="Prometheus metrics port"
    )
    metrics_path: str = Field(
        default="/metrics", 
        env="METRICS_PATH",
        description="Metrics endpoint path"
    )
    enable_jaeger: bool = Field(
        default=False, 
        env="ENABLE_JAEGER",
        description="Enable Jaeger tracing"
    )
    jaeger_agent_host: str = Field(
        default="localhost", 
        env="JAEGER_AGENT_HOST",
        description="Jaeger agent hostname"
    )
    jaeger_agent_port: int = Field(
        default=6831, 
        env="JAEGER_AGENT_PORT",
        ge=1024,
        le=65535,
        description="Jaeger agent port"
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO, 
        env="LOG_LEVEL",
        description="Application log level"
    )
    
    @validator('log_level')
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is supported."""
        try:
            return LogLevel(v.upper()).value
        except ValueError:
            valid_levels = [level.value for level in LogLevel]
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")


class SecuritySettings(BaseConfig):
    """Security and authentication settings."""
    
    secret_key: str = Field(
        default="your-secret-key-here", 
        env="SECRET_KEY",
        description="Secret key for JWT tokens",
        min_length=32
    )
    algorithm: str = Field(
        default="HS256", 
        env="JWT_ALGORITHM",
        description="JWT algorithm"
    )
    access_token_expire_minutes: int = Field(
        default=30, 
        env="ACCESS_TOKEN_EXPIRE_MINUTES",
        ge=1,
        le=1440,
        description="JWT token expiration in minutes"
    )
    enable_auth: bool = Field(
        default=False, 
        env="ENABLE_AUTH",
        description="Enable authentication"
    )
    api_key: Optional[str] = Field(
        default=None, 
        env="API_KEY",
        description="API key for service authentication"
    )
    
    class Config(BaseConfig.Config):
        env_prefix = "SECURITY_"


class CloudSettings(BaseConfig):
    """Cloud storage and services settings."""
    
    # AWS Settings
    aws_access_key_id: Optional[str] = Field(
        default=None, 
        env="AWS_ACCESS_KEY_ID",
        description="AWS access key ID"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None, 
        env="AWS_SECRET_ACCESS_KEY",
        description="AWS secret access key"
    )
    aws_region: str = Field(
        default="us-east-1", 
        env="AWS_REGION",
        description="AWS region"
    )
    s3_bucket: Optional[str] = Field(
        default=None, 
        env="S3_BUCKET",
        description="S3 bucket name"
    )
    
    # GCP Settings
    gcp_project_id: Optional[str] = Field(
        default=None, 
        env="GCP_PROJECT_ID",
        description="GCP project ID"
    )
    gcp_credentials_path: Optional[str] = Field(
        default=None, 
        env="GCP_CREDENTIALS_PATH",
        description="Path to GCP credentials JSON file"
    )
    gcs_bucket: Optional[str] = Field(
        default=None, 
        env="GCS_BUCKET",
        description="GCS bucket name"
    )
    
    # Azure Settings
    azure_storage_account: Optional[str] = Field(
        default=None, 
        env="AZURE_STORAGE_ACCOUNT",
        description="Azure storage account name"
    )
    azure_storage_key: Optional[str] = Field(
        default=None, 
        env="AZURE_STORAGE_KEY",
        description="Azure storage account key"
    )
    azure_container: Optional[str] = Field(
        default=None, 
        env="AZURE_CONTAINER",
        description="Azure container name"
    )
    
    class Config(BaseConfig.Config):
        env_prefix = "CLOUD_"


class DriftDetectionSettings(BaseConfig):
    """Drift detection algorithm configuration."""
    
    # Statistical test thresholds
    ks_test_threshold: float = Field(default=0.05, env="KS_TEST_THRESHOLD")
    psi_threshold: float = Field(default=0.2, env="PSI_THRESHOLD")
    chi2_threshold: float = Field(default=0.05, env="CHI2_THRESHOLD")
    
    # Window settings
    reference_window_size: int = Field(default=1000, env="REFERENCE_WINDOW_SIZE")
    detection_window_size: int = Field(default=100, env="DETECTION_WINDOW_SIZE")
    sliding_window_step: int = Field(default=50, env="SLIDING_WINDOW_STEP")
    
    # Alert settings
    consecutive_alerts_threshold: int = Field(default=3, env="CONSECUTIVE_ALERTS_THRESHOLD")
    alert_cooldown_minutes: int = Field(default=60, env="ALERT_COOLDOWN_MINUTES")
    
    # Feature importance monitoring
    feature_importance_threshold: float = Field(default=0.1, env="FEATURE_IMPORTANCE_THRESHOLD")
    
    class Config:
        env_prefix = "DRIFT_"


class MLFlowSettings(BaseSettings):
    """MLFlow tracking configuration."""
    
    tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    experiment_name: str = Field(default="mdt-experiments", env="MLFLOW_EXPERIMENT_NAME")
    artifact_root: Optional[str] = Field(default=None, env="MLFLOW_ARTIFACT_ROOT")
    s3_endpoint_url: Optional[str] = Field(default=None, env="MLFLOW_S3_ENDPOINT_URL")
    
    class Config:
        env_prefix = "MLFLOW_"


class AlertSettings(BaseSettings):
    """Alert and notification configuration."""
    
    enable_email: bool = Field(default=True, env="ALERT_ENABLE_EMAIL")
    enable_slack: bool = Field(default=False, env="ALERT_ENABLE_SLACK")
    enable_webhook: bool = Field(default=False, env="ALERT_ENABLE_WEBHOOK")
    
    smtp_server: str = Field(default="smtp.gmail.com", env="SMTP_SERVER")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    webhook_url: Optional[str] = Field(default=None, env="WEBHOOK_URL")
    
    default_recipients: List[str] = Field(
        default=["admin@company.com"],
        env="ALERT_DEFAULT_RECIPIENTS"
    )


class BaseConfig(BaseSettings):
    """Base configuration class with common functionality."""
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "forbid"  # Prevent extra fields
        
    def dict_safe(self) -> dict:
        """Return dictionary with sensitive fields masked."""
        data = self.dict()
        sensitive_fields = {'password', 'secret', 'key', 'token'}
        
        def mask_sensitive(obj: dict, path: str = "") -> dict:
            if isinstance(obj, dict):
                return {
                    k: mask_sensitive(v, f"{path}.{k}" if path else k)
                    for k, v in obj.items()
                }
            elif any(sensitive in str(path).lower() for sensitive in sensitive_fields):
                return "***MASKED***" if obj else None
            else:
                return obj
                
        return mask_sensitive(data)


class Settings(BaseConfig):
    """Main application settings."""
    
    # Basic app settings
    app_name: str = Field(default="MDT Dashboard", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # API settings
    api_v1_str: str = Field(default="/api/v1", env="API_V1_STR")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        env="ALLOWED_HOSTS"
    )
    
    # File storage settings
    data_dir: str = Field(default="./data", env="DATA_DIR")
    models_dir: str = Field(default="./models", env="MODELS_DIR")
    logs_dir: str = Field(default="./logs", env="LOGS_DIR")
    
    # ML settings
    default_model_name: str = Field(default="best_model", env="DEFAULT_MODEL_NAME")
    max_prediction_batch_size: int = Field(default=1000, env="MAX_PREDICTION_BATCH_SIZE")
    drift_detection_threshold: float = Field(default=0.05, env="DRIFT_DETECTION_THRESHOLD")
    model_cache_ttl_minutes: int = Field(default=60, env="MODEL_CACHE_TTL_MINUTES")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    security: SecuritySettings = SecuritySettings()
    cloud: CloudSettings = CloudSettings()
    drift_detection: DriftDetectionSettings = DriftDetectionSettings()
    mlflow: MLFlowSettings = MLFlowSettings()
    alerts: AlertSettings = AlertSettings()
    
    @validator('environment')
    def validate_environment(cls, v: str) -> str:
        """Validate environment is supported."""
        try:
            return Environment(v).value
        except ValueError:
            valid_envs = [env.value for env in Environment]
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v: str | List[str]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v
    
    @validator('allowed_hosts', pre=True)
    def parse_allowed_hosts(cls, v: str | List[str]) -> List[str]:
        """Parse allowed hosts from string or list."""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
