"""
Advanced configuration management for MLOps pipelines.

This module provides a flexible and secure configuration system supporting:
- Environment-specific configurations
- Secret management integration
- Configuration validation
- Hot reloading capabilities
- Hierarchical configuration merging

Features:
- YAML/JSON configuration files
- Environment variable substitution
- Configuration schema validation
- Encrypted secrets support
- Configuration versioning
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
from cryptography.fernet import Fernet


logger = logging.getLogger(__name__)


@dataclass
class ConfigMetadata:
    """Metadata for configuration tracking."""
    version: str
    environment: str
    last_updated: datetime
    checksum: str
    source_files: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        result = asdict(self)
        result['last_updated'] = self.last_updated.isoformat()
        return result


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


class SecretManager:
    """Manages encrypted secrets and sensitive configuration."""
    
    def __init__(self, key_file: Optional[str] = None):
        """
        Initialize SecretManager.
        
        Args:
            key_file: Path to encryption key file. If None, generates new key.
        """
        self.key_file = key_file or os.path.join(os.path.expanduser("~"), ".mdt_secret_key")
        self.cipher_suite = self._get_or_create_cipher()
    
    def _get_or_create_cipher(self) -> Fernet:
        """Get or create encryption cipher."""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            os.chmod(self.key_file, 0o600)  # Restrict permissions
        
        return Fernet(key)
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a string value."""
        encrypted = self.cipher_suite.encrypt(value.encode())
        return encrypted.decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt an encrypted string value."""
        decrypted = self.cipher_suite.decrypt(encrypted_value.encode())
        return decrypted.decode()
    
    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted (basic heuristic)."""
        try:
            self.decrypt_value(value)
            return True
        except:
            return False


class ConfigValidator:
    """Validates configuration against schema."""
    
    def __init__(self, schema_file: Optional[str] = None):
        """
        Initialize ConfigValidator.
        
        Args:
            schema_file: Path to JSON schema file for validation
        """
        self.schema = None
        if schema_file and os.path.exists(schema_file):
            with open(schema_file, 'r') as f:
                self.schema = json.load(f)
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not self.schema:
            logger.warning("No schema defined, skipping validation")
            return errors
        
        # Basic validation rules
        errors.extend(self._validate_required_fields(config))
        errors.extend(self._validate_field_types(config))
        errors.extend(self._validate_ranges(config))
        
        return errors
    
    def _validate_required_fields(self, config: Dict[str, Any]) -> List[str]:
        """Validate required fields are present."""
        errors = []
        required_fields = self.schema.get('required', [])
        
        for field in required_fields:
            if self._get_nested_value(config, field) is None:
                errors.append(f"Required field missing: {field}")
        
        return errors
    
    def _validate_field_types(self, config: Dict[str, Any]) -> List[str]:
        """Validate field types match schema."""
        errors = []
        field_types = self.schema.get('properties', {})
        
        for field, type_info in field_types.items():
            value = self._get_nested_value(config, field)
            if value is not None:
                expected_type = type_info.get('type')
                if not self._check_type(value, expected_type):
                    errors.append(f"Field {field} has incorrect type. Expected {expected_type}")
        
        return errors
    
    def _validate_ranges(self, config: Dict[str, Any]) -> List[str]:
        """Validate numeric fields are within acceptable ranges."""
        errors = []
        field_types = self.schema.get('properties', {})
        
        for field, type_info in field_types.items():
            value = self._get_nested_value(config, field)
            if value is not None and isinstance(value, (int, float)):
                minimum = type_info.get('minimum')
                maximum = type_info.get('maximum')
                
                if minimum is not None and value < minimum:
                    errors.append(f"Field {field} value {value} below minimum {minimum}")
                
                if maximum is not None and value > maximum:
                    errors.append(f"Field {field} value {value} above maximum {maximum}")
        
        return errors
    
    def _get_nested_value(self, config: Dict[str, Any], field_path: str) -> Any:
        """Get nested value from configuration using dot notation."""
        keys = field_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True


class AdvancedConfigManager:
    """
    Advanced configuration manager with multiple features.
    
    Features:
    - Environment-specific configurations
    - Secret management
    - Configuration validation
    - Hot reloading
    - Configuration versioning
    """
    
    def __init__(
        self,
        config_dir: str = "config",
        environment: str = "development",
        schema_file: Optional[str] = None,
        enable_hot_reload: bool = False
    ):
        """
        Initialize AdvancedConfigManager.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Current environment (development, staging, production)
            schema_file: Path to configuration schema file
            enable_hot_reload: Enable configuration hot reloading
        """
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.enable_hot_reload = enable_hot_reload
        
        self.secret_manager = SecretManager()
        self.validator = ConfigValidator(schema_file)
        
        self._config = {}
        self._metadata = None
        self._file_timestamps = {}
        
        # Load initial configuration
        self.reload_config()
        
        logger.info(f"ConfigManager initialized for environment: {environment}")
    
    def reload_config(self) -> None:
        """Reload configuration from files."""
        logger.info("Reloading configuration")
        
        config_files = self._discover_config_files()
        merged_config = self._load_and_merge_configs(config_files)
        
        # Environment variable substitution
        merged_config = self._substitute_env_vars(merged_config)
        
        # Decrypt secrets
        merged_config = self._decrypt_secrets(merged_config)
        
        # Validate configuration
        validation_errors = self.validator.validate(merged_config)
        if validation_errors:
            raise ConfigurationError(f"Configuration validation failed: {validation_errors}")
        
        # Calculate checksum
        checksum = self._calculate_checksum(merged_config)
        
        # Update configuration and metadata
        self._config = merged_config
        self._metadata = ConfigMetadata(
            version=self._generate_version(),
            environment=self.environment,
            last_updated=datetime.now(),
            checksum=checksum,
            source_files=[str(f) for f in config_files]
        )
        
        # Update file timestamps for hot reload
        if self.enable_hot_reload:
            self._update_file_timestamps(config_files)
        
        logger.info(f"Configuration reloaded successfully. Version: {self._metadata.version}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        # Check for hot reload
        if self.enable_hot_reload and self._config_files_changed():
            self.reload_config()
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any, encrypt: bool = False) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            encrypt: Whether to encrypt the value
        """
        if encrypt and isinstance(value, str):
            value = self.secret_manager.encrypt_value(value)
        
        keys = key.split('.')
        config_ref = self._config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        # Set the value
        config_ref[keys[-1]] = value
        
        # Update metadata
        self._metadata.last_updated = datetime.now()
        self._metadata.checksum = self._calculate_checksum(self._config)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return self._config.copy()
    
    def get_metadata(self) -> ConfigMetadata:
        """Get configuration metadata."""
        return self._metadata
    
    def save_config(self, filename: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            filename: Optional filename. If None, saves to environment-specific file.
        """
        if not filename:
            filename = f"config_{self.environment}.yaml"
        
        filepath = self.config_dir / filename
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    def _discover_config_files(self) -> List[Path]:
        """Discover configuration files in priority order."""
        config_files = []
        
        # Base configuration
        base_config = self.config_dir / "config.yaml"
        if base_config.exists():
            config_files.append(base_config)
        
        # Environment-specific configuration
        env_config = self.config_dir / f"config_{self.environment}.yaml"
        if env_config.exists():
            config_files.append(env_config)
        
        # Local configuration (highest priority)
        local_config = self.config_dir / "config_local.yaml"
        if local_config.exists():
            config_files.append(local_config)
        
        return config_files
    
    def _load_and_merge_configs(self, config_files: List[Path]) -> Dict[str, Any]:
        """Load and merge multiple configuration files."""
        merged_config = {}
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    if config_file.suffix.lower() in ['.yaml', '.yml']:
                        file_config = yaml.safe_load(f) or {}
                    elif config_file.suffix.lower() == '.json':
                        file_config = json.load(f)
                    else:
                        logger.warning(f"Unsupported config file format: {config_file}")
                        continue
                
                # Deep merge configuration
                merged_config = self._deep_merge(merged_config, file_config)
                logger.debug(f"Loaded configuration from {config_file}")
                
            except Exception as e:
                logger.error(f"Error loading config file {config_file}: {e}")
                raise ConfigurationError(f"Failed to load {config_file}: {e}")
        
        return merged_config
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values."""
        def substitute_value(value):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                parts = env_var.split(':', 1)
                var_name = parts[0]
                default_value = parts[1] if len(parts) > 1 else None
                
                return os.getenv(var_name, default_value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value
        
        return substitute_value(config)
    
    def _decrypt_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt encrypted values in configuration."""
        def decrypt_value(value):
            if isinstance(value, str) and self.secret_manager.is_encrypted(value):
                try:
                    return self.secret_manager.decrypt_value(value)
                except Exception as e:
                    logger.error(f"Failed to decrypt value: {e}")
                    return value
            elif isinstance(value, dict):
                return {k: decrypt_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [decrypt_value(item) for item in value]
            else:
                return value
        
        return decrypt_value(config)
    
    def _calculate_checksum(self, config: Dict[str, Any]) -> str:
        """Calculate MD5 checksum of configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _generate_version(self) -> str:
        """Generate version string based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _update_file_timestamps(self, config_files: List[Path]) -> None:
        """Update file modification timestamps for hot reload."""
        for file_path in config_files:
            if file_path.exists():
                self._file_timestamps[str(file_path)] = file_path.stat().st_mtime
    
    def _config_files_changed(self) -> bool:
        """Check if any configuration files have been modified."""
        for file_path, old_timestamp in self._file_timestamps.items():
            path_obj = Path(file_path)
            if path_obj.exists():
                current_timestamp = path_obj.stat().st_mtime
                if current_timestamp > old_timestamp:
                    logger.info(f"Configuration file changed: {file_path}")
                    return True
        
        return False


# Factory function for easy configuration manager creation
def create_config_manager(
    environment: Optional[str] = None,
    config_dir: str = "config",
    schema_file: Optional[str] = None,
    enable_hot_reload: bool = False
) -> AdvancedConfigManager:
    """
    Factory function to create configuration manager.
    
    Args:
        environment: Environment name (defaults to ENVIRONMENT env var or 'development')
        config_dir: Configuration directory
        schema_file: Path to configuration schema file
        enable_hot_reload: Enable hot reloading
        
    Returns:
        Configured AdvancedConfigManager instance
    """
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'development')
    
    return AdvancedConfigManager(
        config_dir=config_dir,
        environment=environment,
        schema_file=schema_file,
        enable_hot_reload=enable_hot_reload
    )
