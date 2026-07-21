"""
api/agents/report_generator.py
──────────────────────────────
[STUB] ReportGenerator agent.

Real implementation:
  - analyzed_risks list is converted into a validated Pydantic ContractReport.
  - The report is persisted as JSON in the SQL database.
  - Output: validated ContractReport object.

Current: Validates analyzed_risks through the Pydantic schema and writes
         the serialised dict to final_report in AgentState.
"""

from __future__ import annotations

from datetime import datetime
from loguru import logger

from api.schemas.contract_schema import ClauseRisk, ContractReport


def report_generator_node(state: dict) -> dict:
    """
    Produces the final Pydantic report from validated analysis results.

    Args:
        state: Current AgentState. Reads:
               "document_name", "clauses", "analyzed_risks"

    Returns:
        Updated state slice: {"final_report": dict}
    """
    logger.info("[ReportGenerator] Agent running — generating final report.")

    analyzed_risks: list[dict] = state.get("analyzed_risks", [])
    total_clauses: int = len(state.get("clauses", []))

    # ── Pydantic Validation ───────────────────────────────────────────────────
    risky_clauses: list[ClauseRisk] = [
        ClauseRisk(**risk)
        for risk in analyzed_risks
        if risk.get("risk_level") in ("medium", "high")
    ]

    high_count = sum(1 for r in risky_clauses if r.risk_level == "high")
    medium_count = sum(1 for r in risky_clauses if r.risk_level == "medium")

    # Simple risk score: (high * 1.0 + medium * 0.5) / total clauses
    overall_risk_score: float = 0.0
    if total_clauses > 0:
        raw_score = (high_count * 1.0 + medium_count * 0.5) / total_clauses
        overall_risk_score = round(min(raw_score, 1.0), 4)

    report = ContractReport(
        document_name=state.get("document_name", "unknown"),
        total_clauses=total_clauses,
        risky_clauses=risky_clauses,
        overall_risk_score=overall_risk_score,
        generated_at=datetime.utcnow(),
    )
    # ─────────────────────────────────────────────────────────────────────────

    logger.info(
        "[ReportGenerator] Report generated | Total clauses: %d | Risky: %d | Score: %.4f",
        total_clauses,
        len(risky_clauses),
        overall_risk_score,
    )

    return {"final_report": report.model_dump(mode="json")}
