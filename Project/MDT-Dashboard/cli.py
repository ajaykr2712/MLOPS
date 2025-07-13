#!/usr/bin/env python
"""
MDT Dashboard CLI - Command Line Interface for running the MDT platform.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import signal
import os
from concurrent.futures import ThreadPoolExecutor

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.mdt_dashboard.core.config import get_settings


def run_api_server():
    """Run the FastAPI server."""
    settings = get_settings()
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.mdt_dashboard.api.main_enhanced:app",
        "--host", settings.host,
        "--port", str(settings.port),
        "--reload" if settings.debug else "--no-reload",
        "--log-level", settings.monitoring.log_level.lower()
    ]
    
    print(f"Starting API server on {settings.host}:{settings.port}")
    return subprocess.Popen(cmd)


def run_dashboard():
    """Run the Streamlit dashboard."""
    settings = get_settings()
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/mdt_dashboard/dashboard/main.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false"
    ]
    
    print("Starting Streamlit dashboard on http://localhost:8501")
    return subprocess.Popen(cmd)


def run_worker():
    """Run the Celery worker."""
    cmd = [
        sys.executable, "-m", "celery",
        "worker",
        "-A", "src.mdt_dashboard.worker.celery_app",
        "--loglevel=info",
        "--concurrency=4"
    ]
    
    print("Starting Celery worker")
    return subprocess.Popen(cmd)


def run_beat():
    """Run the Celery beat scheduler."""
    cmd = [
        sys.executable, "-m", "celery",
        "beat",
        "-A", "src.mdt_dashboard.worker.celery_app",
        "--loglevel=info"
    ]
    
    print("Starting Celery beat scheduler")
    return subprocess.Popen(cmd)


def setup_database():
    """Setup the database tables."""
    try:
        from src.mdt_dashboard.core.database import db_manager
        
        print("Setting up database...")
        db_manager.create_tables()
        print("Database setup completed successfully")
        
    except Exception as e:
        print(f"Database setup failed: {e}")
        sys.exit(1)


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "fastapi", "uvicorn", "streamlit", "celery", "redis",
        "sqlalchemy", "pandas", "numpy", "scikit-learn", "plotly"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install them using: pip install " + " ".join(missing))
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="MDT Dashboard - Model Drift Detection & Telemetry Platform"
    )
    
    parser.add_argument(
        "command",
        choices=["run", "api", "dashboard", "worker", "beat", "setup", "check"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode"
    )
    
    args = parser.parse_args()
    
    if args.command == "check":
        check_dependencies()
        print("All dependencies are installed ✓")
        return
    
    if args.command == "setup":
        check_dependencies()
        setup_database()
        return
    
    # Set environment
    if args.dev:
        os.environ["ENVIRONMENT"] = "development"
        os.environ["DEBUG"] = "true"
    
    processes = []
    
    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        print("\nShutting down MDT Dashboard...")
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.command == "run":
            # Run all services
            print("Starting MDT Dashboard platform...")
            
            # Setup database
            setup_database()
            
            # Start all services
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Start API server
                api_future = executor.submit(run_api_server)
                time.sleep(2)  # Wait for API to start
                
                # Start dashboard
                dashboard_future = executor.submit(run_dashboard)
                time.sleep(2)  # Wait for dashboard to start
                
                # Start worker services (optional)
                try:
                    worker_future = executor.submit(run_worker)
                    beat_future = executor.submit(run_beat)
                except Exception as e:
                    print(f"Warning: Could not start worker services: {e}")
                    print("Redis may not be available. Running without background tasks.")
                
                print("\n" + "="*60)
                print("🚀 MDT Dashboard is running!")
                print("="*60)
                print("📊 Dashboard: http://localhost:8501")
                print("🔧 API Docs:  http://localhost:8000/docs")
                print("📈 Metrics:   http://localhost:8000/metrics")
                print("="*60)
                print("Press Ctrl+C to stop all services")
                print("="*60)
                
                # Wait for processes
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
        
        elif args.command == "api":
            setup_database()
            process = run_api_server()
            processes.append(process)
            print("API server started. Press Ctrl+C to stop.")
            process.wait()
        
        elif args.command == "dashboard":
            process = run_dashboard()
            processes.append(process)
            print("Dashboard started. Press Ctrl+C to stop.")
            process.wait()
        
        elif args.command == "worker":
            process = run_worker()
            processes.append(process)
            print("Worker started. Press Ctrl+C to stop.")
            process.wait()
        
        elif args.command == "beat":
            process = run_beat()
            processes.append(process)
            print("Beat scheduler started. Press Ctrl+C to stop.")
            process.wait()
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
