"""
db/models.py
────────────
SQLAlchemy ORM models for the contract audit system.

Tables:
  - documents  : Metadata for each uploaded document
  - reports    : Full audit report (JSON) linked to a document
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Document(Base):
    """Represents an uploaded contract document."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(512), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    page_count = Column(Integer, nullable=True)
    raw_text_length = Column(Integer, nullable=True)

    # One document → one report
    report = relationship("Report", back_populates="document", uselist=False)

    def __repr__(self) -> str:
        return f"<Document {self.filename} ({self.id})>"


class Report(Base):
    """Stores the full audit report for a document."""

    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    overall_risk_score = Column(Float, nullable=False)
    total_clauses = Column(Integer, nullable=False)
    risky_clause_count = Column(Integer, nullable=False)
    report_json = Column(JSON, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="report")

    def __repr__(self) -> str:
        return f"<Report doc={self.document_id} score={self.overall_risk_score}>"
