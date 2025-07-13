"""
Database configuration and session management.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import Generator
import logging

from .config import get_settings

logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()

class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database engine and session factory."""
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                self.settings.database.url,
                pool_size=self.settings.database.pool_size,
                max_overflow=self.settings.database.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=self.settings.database.echo
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def get_session(self):
        """Get database session."""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator:
        """Database session context manager."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all tables."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise

# Global database manager instance
db_manager = DatabaseManager()

# Dependency for FastAPI
def get_db():
    """FastAPI dependency for database session."""
    with db_manager.session_scope() as session:
        yield session
