"""
api/agents/risk_analyzer.py
───────────────────────────
[STUB] RiskAnalyzer agent.

Real implementation:
  - Each clause is sent to the LLM via Ollama (Qwen2.5-7B).
  - The RAG layer kicks in: semantic similarity search against standard
    clauses stored in ChromaDB.
  - Output: analysis list containing risk level, explanation, and recommendation.

Current: Assigns risk levels based on static keyword matching.
         On retry runs, logs the feedback received from the Judge.
"""

from __future__ import annotations

from loguru import logger

# Keyword → (risk_level, explanation, recommendation) mapping (stub logic)
_RISK_KEYWORDS: dict[str, tuple[str, str, str]] = {
    "penalty": (
        "high",
        "The contract contains no provision regarding late-payment penalties.",
        "A standard late-payment penalty clause should be added (e.g. 0.1% per day).",
    ),
    "liability": (
        "high",
        "No upper cap has been imposed on the total liability amount.",
        "Liability should be capped at the total contract value.",
    ),
    "confidentiality": (
        "medium",
        "Confidentiality obligations are indefinite and unreasonably broad in scope.",
        "The confidentiality period should be limited to a fixed term (e.g. 3 years).",
    ),
    "termination": (
        "high",
        "The termination notice period is below the legal minimum (15 days).",
        "The termination notice period must be extended to at least 15 days.",
    ),
}


def risk_analyzer_node(state: dict) -> dict:
    """
    Analyses each clause and determines its risk level.

    Args:
        state: Current AgentState. Reads "clauses" and "judge_feedback" fields.

    Returns:
        Updated state slice: {
            "analyzed_risks": list[dict],
            "retry_count": int,
            "validation_passed": bool,
        }
    """
    retry_count: int = state.get("retry_count", 0)

    if retry_count == 0:
        logger.info("[RiskAnalyzer] Agent running — starting initial analysis.")
    else:
        feedback = state.get("judge_feedback", "")
        logger.info(
            "[RiskAnalyzer] Re-analysis (attempt %d) — Judge feedback: %s",
            retry_count,
            feedback,
        )

    clauses: list[dict] = state.get("clauses", [])
    analyzed_risks: list[dict] = []

    for clause in clauses:
        clause_text_lower = clause["text"].lower()
        risk_level = "low"
        explanation = "Standard clause — no significant risk detected."
        recommendation = "No changes recommended."
        matched_rule: str | None = None

        # Keyword scan
        for keyword, (level, expl, rec) in _RISK_KEYWORDS.items():
            if keyword in clause_text_lower:
                risk_level = level
                explanation = expl
                recommendation = rec
                matched_rule = f"keyword_match::{keyword}"
                break

        analyzed_risks.append(
            {
                "clause_id": clause["clause_id"],
                "clause_text": clause["text"],
                "risk_level": risk_level,
                "explanation": explanation,
                "recommendation": recommendation,
                "matched_rule": matched_rule,
            }
        )

    logger.info(
        "[RiskAnalyzer] %d clause(s) analysed. High-risk findings: %d.",
        len(analyzed_risks),
        sum(1 for r in analyzed_risks if r["risk_level"] == "high"),
    )

    return {
        "analyzed_risks": analyzed_risks,
        "retry_count": retry_count + 1,
        "validation_passed": False,   # Judge has not yet approved — default False
    }
