"""
API module for MDT Dashboard.
FastAPI-based REST API for model monitoring and management.
"""

from .main_enhanced import app, create_app

__all__ = ["app", "create_app"]
