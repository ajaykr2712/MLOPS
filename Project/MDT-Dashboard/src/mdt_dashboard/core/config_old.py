"""
Core configuration management for MDT Dashboard.
Enterprise-grade configuration with environment-based settings.
"""

from typing import List, Optional
from pydantic import BaseSettings, Field, validator
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="postgresql://mdt_user:mdt_pass@localhost:5432/mdt_db",
        env="DATABASE_URL"
    )
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    
    class Config:
        env_prefix = "DB_"


class RedisSettings(BaseSettings):
    """Redis configuration for caching and task queue."""
    
    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    decode_responses: bool = Field(default=True, env="REDIS_DECODE_RESPONSES")
    
    class Config:
        env_prefix = "REDIS_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    prometheus_port: int = Field(default=8000, env="PROMETHEUS_PORT")
    metrics_path: str = Field(default="/metrics", env="METRICS_PATH")
    enable_jaeger: bool = Field(default=False, env="ENABLE_JAEGER")
    jaeger_agent_host: str = Field(default="localhost", env="JAEGER_AGENT_HOST")
    jaeger_agent_port: int = Field(default=6831, env="JAEGER_AGENT_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class SecuritySettings(BaseSettings):
    """Security and authentication settings."""
    
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    enable_auth: bool = Field(default=False, env="ENABLE_AUTH")
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    
    class Config:
        env_prefix = "SECURITY_"


class CloudSettings(BaseSettings):
    """Cloud storage and services settings."""
    
    # AWS Settings
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    s3_bucket: Optional[str] = Field(default=None, env="S3_BUCKET")
    
    # GCP Settings
    gcp_project_id: Optional[str] = Field(default=None, env="GCP_PROJECT_ID")
    gcp_credentials_path: Optional[str] = Field(default=None, env="GCP_CREDENTIALS_PATH")
    gcs_bucket: Optional[str] = Field(default=None, env="GCS_BUCKET")
    
    # Azure Settings
    azure_storage_account: Optional[str] = Field(default=None, env="AZURE_STORAGE_ACCOUNT")
    azure_storage_key: Optional[str] = Field(default=None, env="AZURE_STORAGE_KEY")
    azure_container: Optional[str] = Field(default=None, env="AZURE_CONTAINER")
    
    class Config:
        env_prefix = "CLOUD_"


class Settings(BaseSettings):
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
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    @validator('allowed_hosts', pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()
