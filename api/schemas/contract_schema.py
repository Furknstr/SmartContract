"""
api/schemas/contract_schema.py
──────────────────────────────
Pydantic data schemas:
  - ClauseRisk     → Analysis result for a single contract clause
  - ContractReport → Final structure of the complete report (LangGraph output)
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ClauseRisk(BaseModel):
    """Risk assessment for a single contract clause."""

    clause_id: str = Field(..., description="Unique identifier assigned to the clause, e.g. 'clause_003'")
    clause_text: str = Field(..., description="Original text of the clause")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="Risk severity level: low | medium | high")
    explanation: str = Field(..., description="Explanation of why the risk was detected")
    recommendation: str = Field(..., description="Corrective recommendation presented to the user")
    matched_rule: str | None = Field(
        default=None,
        description="Rule ID that triggered this risk finding (from guardrails/rules.yaml)",
    )


class ContractReport(BaseModel):
    """Final report for the complete contract analysis."""

    document_name: str = Field(..., description="Name of the uploaded document")
    total_clauses: int = Field(..., description="Total number of clauses detected")
    risky_clauses: list[ClauseRisk] = Field(
        default_factory=list,
        description="List of clauses identified as containing risk",
    )
    overall_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised score between 0.0 (no risk) and 1.0 (very high risk)",
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of when the report was generated",
    )
