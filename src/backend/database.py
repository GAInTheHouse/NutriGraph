"""
SQLAlchemy database setup for NutriGraph.

Creates a synchronous SQLite engine and provides a FastAPI-compatible
get_db() dependency that yields a Session and guarantees it is closed
after every request, even on exceptions.
"""
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Place the DB file at the project root so it persists across restarts
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SQLALCHEMY_DATABASE_URL = f"sqlite:///{_PROJECT_ROOT}/data/nutrigraph.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    # Required for SQLite when multiple threads share the same connection
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that opens a database session for a single request
    and closes it cleanly once the response has been sent.

    Usage:
        db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
