"""
Logging utilities for MDT Dashboard.
Provides structured logging with different outputs and formats.

Refactored for improved code quality, type safety, and maintainability.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union


class LogLevel(str, Enum):
    """Logging levels."""
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output formats."""
    
    STANDARD = "standard"
    JSON = "json"
    DETAILED = "detailed"


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
        
        # Fields to exclude from extra data
        self.excluded_fields = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "lineno", "funcName", "created",
            "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "getMessage", "exc_info",
            "exc_text", "stack_info"
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add process and thread info
        if record.processName and record.processName != "MainProcess":
            log_entry["process"] = record.processName
        
        if record.threadName and record.threadName != "MainThread":
            log_entry["thread"] = record.threadName
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add stack info if present
        if record.stack_info:
            log_entry["stack_info"] = record.stack_info
        
        # Add extra fields if enabled
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in self.excluded_fields:
                    # Ensure value is JSON serializable
                    try:
                        json.dumps(value)
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)
        
        return json.dumps(log_entry, ensure_ascii=False)


class DetailedFormatter(logging.Formatter):
    """Detailed formatter with additional context."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Color codes
        self.colors = {
            'DEBUG': '\033[36m',      # Cyan
            'INFO': '\033[32m',       # Green
            'WARNING': '\033[33m',    # Yellow
            'ERROR': '\033[31m',      # Red
            'CRITICAL': '\033[35m',   # Magenta
            'RESET': '\033[0m'        # Reset
        }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors if supported."""
        # Check if colors are supported
        if not (hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()):
            return super().format(record)
        
        # Add color to level name
        level_color = self.colors.get(record.levelname, '')
        if level_color:
            record.levelname = f"{level_color}{record.levelname}{self.colors['RESET']}"
        
        return super().format(record)


class LoggingConfig:
    """Configuration for logging setup."""
    
    def __init__(
        self,
        level: Union[str, LogLevel] = LogLevel.INFO,
        format_type: LogFormat = LogFormat.STANDARD,
        log_file: Optional[Union[str, Path]] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        include_console: bool = True,
        console_format: Optional[LogFormat] = None,
        json_include_extra: bool = True,
        use_colors: bool = True
    ):
        self.level = LogLevel(level) if isinstance(level, str) else level
        self.format_type = format_type
        self.log_file = Path(log_file) if log_file else None
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.include_console = include_console
        self.console_format = console_format or format_type
        self.json_include_extra = json_include_extra
        self.use_colors = use_colors


class LoggerManager:
    """Manages logger configuration and provides utilities."""
    
    def __init__(self):
        self.configured = False
        self.config: Optional[LoggingConfig] = None
        self.handlers: Dict[str, logging.Handler] = {}
    
    def setup_logging(self, config: LoggingConfig) -> None:
        """Setup logging configuration."""
        self.config = config
        
        # Get numeric level
        numeric_level = getattr(logging, config.level.value, logging.INFO)
        
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        self.handlers.clear()
        
        # Setup console handler
        if config.include_console:
            self._setup_console_handler(config, numeric_level)
        
        # Setup file handler
        if config.log_file:
            self._setup_file_handler(config, numeric_level)
        
        # Configure third-party loggers
        self._configure_third_party_loggers()
        
        self.configured = True
        logging.info("Logging configuration completed")
    
    def _setup_console_handler(self, config: LoggingConfig, level: int) -> None:
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Choose formatter based on console format
        if config.console_format == LogFormat.JSON:
            formatter = JSONFormatter(include_extra=config.json_include_extra)
        elif config.console_format == LogFormat.DETAILED:
            formatter = DetailedFormatter()
        elif config.use_colors:
            formatter = ColoredFormatter()
        else:
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
        self.handlers['console'] = console_handler
    
    def _setup_file_handler(self, config: LoggingConfig, level: int) -> None:
        """Setup file logging handler."""
        # Create log directory if it doesn't exist
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler to manage file size
        file_handler = logging.handlers.RotatingFileHandler(
            config.log_file,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        
        # Choose formatter
        if config.format_type == LogFormat.JSON:
            formatter = JSONFormatter(include_extra=config.json_include_extra)
        elif config.format_type == LogFormat.DETAILED:
            formatter = DetailedFormatter()
        else:
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
        self.handlers['file'] = file_handler
    
    def _configure_third_party_loggers(self) -> None:
        """Configure third-party library loggers."""
        # Set appropriate levels for third-party libraries
        library_configs = {
            "uvicorn": logging.INFO,
            "fastapi": logging.INFO,
            "streamlit": logging.WARNING,
            "urllib3": logging.WARNING,
            "requests": logging.WARNING,
            "sqlalchemy": logging.WARNING,
            "alembic": logging.WARNING,
            "celery": logging.INFO,
            "redis": logging.WARNING,
            "matplotlib": logging.WARNING,
            "PIL": logging.WARNING,
        }
        
        for lib_name, level in library_configs.items():
            logging.getLogger(lib_name).setLevel(level)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        return logging.getLogger(name)
    
    def add_context_to_logger(self, logger: logging.Logger, **context) -> logging.LoggerAdapter:
        """Add context to a logger using LoggerAdapter."""
        return logging.LoggerAdapter(logger, context)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            "configured": self.configured,
            "handlers": list(self.handlers.keys()),
            "level": self.config.level.value if self.config else "Not configured"
        }
        
        if self.config and self.config.log_file:
            try:
                file_size = self.config.log_file.stat().st_size
                stats["log_file"] = {
                    "path": str(self.config.log_file),
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "max_size_mb": round(self.config.max_file_size / (1024 * 1024), 2)
                }
            except Exception as e:
                stats["log_file"] = {"error": str(e)}
        
        return stats
    
    def set_level(self, level: Union[str, LogLevel]) -> None:
        """Change logging level dynamically."""
        log_level = LogLevel(level) if isinstance(level, str) else level
        numeric_level = getattr(logging, log_level.value, logging.INFO)
        
        # Update root logger
        logging.getLogger().setLevel(numeric_level)
        
        # Update all handlers
        for handler in self.handlers.values():
            handler.setLevel(numeric_level)
        
        if self.config:
            self.config.level = log_level
        
        logging.info(f"Logging level changed to {log_level.value}")
    
    def add_handler(self, name: str, handler: logging.Handler) -> None:
        """Add a custom handler."""
        logging.getLogger().addHandler(handler)
        self.handlers[name] = handler
        logging.info(f"Added custom handler: {name}")
    
    def remove_handler(self, name: str) -> None:
        """Remove a handler."""
        if name in self.handlers:
            handler = self.handlers[name]
            logging.getLogger().removeHandler(handler)
            del self.handlers[name]
            logging.info(f"Removed handler: {name}")


# Global logger manager instance
_logger_manager = LoggerManager()


def setup_logging(
    level: Union[str, LogLevel] = LogLevel.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_type: LogFormat = LogFormat.STANDARD,
    include_console: bool = True,
    **kwargs
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Path to log file
        format_type: Log format type
        include_console: Whether to include console output
        **kwargs: Additional configuration options
    """
    config = LoggingConfig(
        level=level,
        log_file=log_file,
        format_type=format_type,
        include_console=include_console,
        **kwargs
    )
    _logger_manager.setup_logging(config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return _logger_manager.get_logger(name)


def get_context_logger(name: str, **context) -> logging.LoggerAdapter:
    """Get a logger with context."""
    logger = get_logger(name)
    return _logger_manager.add_context_to_logger(logger, **context)


def set_log_level(level: Union[str, LogLevel]) -> None:
    """Change logging level dynamically."""
    _logger_manager.set_level(level)


def get_log_stats() -> Dict[str, Any]:
    """Get logging statistics."""
    return _logger_manager.get_log_stats()


def configure_from_settings(settings) -> None:
    """Configure logging from settings object."""
    config = LoggingConfig(
        level=getattr(settings, 'log_level', LogLevel.INFO),
        format_type=getattr(settings, 'log_format', LogFormat.STANDARD),
        log_file=getattr(settings, 'log_file', None),
        include_console=getattr(settings, 'log_console', True),
        use_colors=getattr(settings, 'log_colors', True)
    )
    _logger_manager.setup_logging(config)


# Context manager for temporary log level changes
class temporary_log_level:
    """Context manager for temporary log level changes."""
    
    def __init__(self, level: Union[str, LogLevel]):
        self.new_level = LogLevel(level) if isinstance(level, str) else level
        self.old_level = None
    
    def __enter__(self):
        # Store current level
        root_logger = logging.getLogger()
        self.old_level = root_logger.level
        
        # Set new level
        set_log_level(self.new_level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old level
        if self.old_level is not None:
            numeric_level = self.old_level
            root_logger = logging.getLogger()
            root_logger.setLevel(numeric_level)
            
            # Update all handlers
            for handler in _logger_manager.handlers.values():
                handler.setLevel(numeric_level)
