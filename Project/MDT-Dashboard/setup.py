#!/usr/bin/env python3
"""
Setup and deployment script for MDT Dashboard.
Automates installation, configuration, and deployment tasks.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MDTSetup:
    """Setup and deployment manager for MDT Dashboard."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent
        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
    
    def create_directories(self):
        """Create necessary directories."""
        logger.info("Creating directories...")
        
        directories = [
            self.config_dir,
            self.data_dir,
            self.models_dir,
            self.logs_dir,
            self.data_dir / "reference",
            self.data_dir / "incoming",
            self.data_dir / "processed"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def install_dependencies(self, dev: bool = False):
        """Install Python dependencies."""
        logger.info("Installing dependencies...")
        
        try:
            # Check if poetry is available
            subprocess.run(["poetry", "--version"], check=True, capture_output=True)
            
            # Install with poetry
            cmd = ["poetry", "install"]
            if not dev:
                cmd.append("--no-dev")
            
            subprocess.run(cmd, check=True, cwd=self.project_root)
            logger.info("Dependencies installed with Poetry")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Poetry not found, falling back to pip")
            
            # Install with pip
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], check=True)
                logger.info("Dependencies installed with pip")
            else:
                logger.error("No requirements.txt found")
                raise FileNotFoundError("requirements.txt not found")
    
    def create_env_file(self, overwrite: bool = False):
        """Create .env file with default configuration."""
        env_file = self.project_root / ".env"
        
        if env_file.exists() and not overwrite:
            logger.info(".env file already exists, skipping...")
            return
        
        logger.info("Creating .env file...")
        
        env_content = """# MDT Dashboard Configuration

# Application Settings
APP_NAME=MDT Dashboard
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true

# API Settings
HOST=0.0.0.0
PORT=8000
API_V1_STR=/api/v1

# Database Settings
DATABASE_URL=postgresql://mdt_user:mdt_pass@localhost:5432/mdt_db
DATABASE_ECHO=false
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Settings
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=20

# Security Settings
SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS Settings
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://localhost:8501
ALLOWED_HOSTS=localhost,127.0.0.1

# Monitoring Settings
LOG_LEVEL=INFO
PROMETHEUS_PORT=8000
METRICS_PATH=/metrics

# Drift Detection Settings
DRIFT_DETECTION_THRESHOLD=0.05
KS_TEST_THRESHOLD=0.05
PSI_THRESHOLD=0.2

# MLflow Settings
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=mdt-experiments

# Alert Settings
ALERT_ENABLE_EMAIL=true
ALERT_ENABLE_SLACK=false
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
ALERT_DEFAULT_RECIPIENTS=admin@company.com

# File Paths
DATA_DIR=./data
MODELS_DIR=./models
LOGS_DIR=./logs
"""
        
        with open(env_file, "w") as f:
            f.write(env_content)
        
        logger.info(f"Created .env file: {env_file}")
    
    def setup_database(self):
        """Setup database (PostgreSQL)."""
        logger.info("Setting up database...")
        
        try:
            # Try to connect and create database
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
            
            # Connection parameters
            conn_params = {
                "host": "localhost",
                "port": 5432,
                "user": "postgres",
                "password": "postgres"
            }
            
            # Connect to PostgreSQL
            conn = psycopg2.connect(**conn_params)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Create database and user
            cursor.execute("CREATE DATABASE mdt_db;")
            cursor.execute("CREATE USER mdt_user WITH PASSWORD 'mdt_pass';")
            cursor.execute("GRANT ALL PRIVILEGES ON DATABASE mdt_db TO mdt_user;")
            
            cursor.close()
            conn.close()
            
            logger.info("Database setup completed")
            
        except Exception as e:
            logger.warning(f"Database setup failed: {e}")
            logger.info("Please setup PostgreSQL manually or use SQLite for development")
    
    def setup_redis(self):
        """Setup Redis (for development)."""
        logger.info("Checking Redis connection...")
        
        try:
            import redis
            
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            logger.info("Redis connection successful")
            
        except Exception as e:
            logger.warning(f"Redis setup failed: {e}")
            logger.info("Please install and start Redis manually")
    
    def initialize_database_tables(self):
        """Initialize database tables."""
        logger.info("Initializing database tables...")
        
        try:
            # Import and create tables
            from src.mdt_dashboard.core.database import db_manager
            
            db_manager.create_tables()
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def create_sample_data(self):
        """Create sample data for demonstration."""
        logger.info("Creating sample data...")
        
        # Create sample model metadata
        sample_model = {
            "name": "sample_model",
            "version": "1.0.0",
            "algorithm": "RandomForest",
            "framework": "scikit-learn",
            "created_at": "2024-01-01T00:00:00Z",
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.83,
                "recall": 0.87,
                "f1_score": 0.85
            }
        }
        
        sample_file = self.models_dir / "sample_model_metadata.json"
        with open(sample_file, "w") as f:
            json.dump(sample_model, f, indent=2)
        
        logger.info(f"Created sample model metadata: {sample_file}")
        
        # Create sample reference data
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        sample_data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.normal(2, 1.5, 1000),
            "feature3": np.random.uniform(0, 10, 1000),
            "feature4": np.random.exponential(2, 1000)
        })
        
        reference_file = self.data_dir / "reference" / "sample_reference_data.csv"
        sample_data.to_csv(reference_file, index=False)
        
        logger.info(f"Created sample reference data: {reference_file}")
    
    def run_tests(self):
        """Run test suite."""
        logger.info("Running tests...")
        
        try:
            # Run pytest
            subprocess.run([
                sys.executable, "-m", "pytest", 
                str(self.project_root / "tests"),
                "-v", "--tb=short"
            ], check=True, cwd=self.project_root)
            
            logger.info("All tests passed!")
            
        except subprocess.CalledProcessError:
            logger.error("Some tests failed")
            raise
        except FileNotFoundError:
            logger.warning("pytest not found, skipping tests")
    
    def start_services(self, service: str = "all"):
        """Start application services."""
        logger.info(f"Starting {service} services...")
        
        if service in ["all", "api"]:
            logger.info("Starting API server...")
            # This would start the FastAPI server
            # For now, just print the command
            print("To start API server manually:")
            print("cd", self.project_root)
            print("python -m src.mdt_dashboard.api.main_complete")
        
        if service in ["all", "dashboard"]:
            logger.info("Starting dashboard...")
            print("To start dashboard manually:")
            print("cd", self.project_root)
            print("streamlit run src/mdt_dashboard/dashboard/complete_dashboard.py")
        
        if service in ["all", "worker"]:
            logger.info("Starting worker...")
            print("To start worker manually:")
            print("cd", self.project_root)
            print("celery -A src.mdt_dashboard.worker worker --loglevel=info")
    
    def deploy_docker(self):
        """Deploy with Docker."""
        logger.info("Deploying with Docker...")
        
        try:
            # Build Docker image
            subprocess.run([
                "docker", "build", "-t", "mdt-dashboard", "."
            ], check=True, cwd=self.project_root)
            
            # Start with docker-compose
            subprocess.run([
                "docker-compose", "up", "-d"
            ], check=True, cwd=self.project_root)
            
            logger.info("Docker deployment completed")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker deployment failed: {e}")
            raise
        except FileNotFoundError:
            logger.error("Docker not found")
            raise
    
    def full_setup(self, dev: bool = False, skip_db: bool = False):
        """Run full setup process."""
        logger.info("Starting full setup...")
        
        try:
            # Create directories
            self.create_directories()
            
            # Install dependencies
            self.install_dependencies(dev=dev)
            
            # Create configuration
            self.create_env_file()
            
            # Setup external services
            if not skip_db:
                self.setup_database()
                self.setup_redis()
                self.initialize_database_tables()
            
            # Create sample data
            self.create_sample_data()
            
            # Run tests if in dev mode
            if dev:
                try:
                    self.run_tests()
                except Exception as e:
                    logger.warning(f"Tests failed: {e}")
            
            logger.info("Setup completed successfully!")
            logger.info("Next steps:")
            logger.info("1. Update .env file with your configuration")
            logger.info("2. Start services with: python setup.py start")
            logger.info("3. Access API docs at: http://localhost:8000/docs")
            logger.info("4. Access dashboard at: http://localhost:8501")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="MDT Dashboard Setup Tool")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Full setup")
    setup_parser.add_argument("--dev", action="store_true", help="Development setup")
    setup_parser.add_argument("--skip-db", action="store_true", help="Skip database setup")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install dependencies")
    install_parser.add_argument("--dev", action="store_true", help="Include dev dependencies")
    
    # Database command
    subparsers.add_parser("init-db", help="Initialize database tables")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start services")
    start_parser.add_argument("service", nargs="?", default="all", 
                             choices=["all", "api", "dashboard", "worker"])
    
    # Test command
    subparsers.add_parser("test", help="Run tests")
    
    # Docker command
    subparsers.add_parser("docker", help="Deploy with Docker")
    
    # Sample data command
    subparsers.add_parser("sample-data", help="Create sample data")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize setup manager
    setup = MDTSetup()
    
    try:
        if args.command == "setup":
            setup.full_setup(dev=args.dev, skip_db=args.skip_db)
        
        elif args.command == "install":
            setup.install_dependencies(dev=args.dev)
        
        elif args.command == "init-db":
            setup.initialize_database_tables()
        
        elif args.command == "start":
            setup.start_services(args.service)
        
        elif args.command == "test":
            setup.run_tests()
        
        elif args.command == "docker":
            setup.deploy_docker()
        
        elif args.command == "sample-data":
            setup.create_sample_data()
        
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
