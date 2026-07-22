"""
db/session.py
─────────────
Database session factory using SQLAlchemy async-compatible setup.

Reads DATABASE_URL from environment variables (.env file).
Creates all tables on first import if they don't exist.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base

load_dotenv()

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://audit_user:audit_pass@localhost:5432/contract_audit",
)

engine = create_engine(DATABASE_URL, echo=False, future=True)

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def init_db() -> None:
    """Create all tables if they don't exist yet."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
