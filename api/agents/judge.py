"""
api/agents/judge.py
───────────────────
[STUB] Judge / Validator (Semantic Guardrail) agent.

Real implementation:
  - Rules are loaded from guardrails/rules.yaml.
  - RiskAnalyzer output is validated deterministically against those rules
    in plain Python (no LLM involved → fast, no hallucination risk).
  - Rule violation → loop: feedback is sent back to RiskAnalyzer.
  - All rules pass → proceed to ReportGenerator.

Current: Flags a violation if any "high"-severity finding has no matched_rule
         (i.e. LLM spotted a risk but no guardrail rule was triggered).
         Since the stub always produces matched rules, validation always passes.
"""

from __future__ import annotations

from loguru import logger


def judge_node(state: dict) -> dict:
    """
    Validates RiskAnalyzer output against guardrail rules.

    Args:
        state: Current AgentState. Reads the "analyzed_risks" field.

    Returns:
        Updated state slice: {
            "validation_passed": bool,
            "judge_feedback": str,
        }
    """
    logger.info("[Judge] Agent running — starting rule validation.")

    analyzed_risks: list[dict] = state.get("analyzed_risks", [])

    # ── STUB RULE CHECK ───────────────────────────────────────────────────────
    # Rule: if a high-risk clause has no matched_rule, count it as a violation
    # (e.g. the LLM detected risk but no guardrail rule was triggered → re-analyse)
    violations: list[str] = []
    for risk in analyzed_risks:
        if risk["risk_level"] == "high" and risk.get("matched_rule") is None:
            violations.append(
                f"{risk['clause_id']}: High risk detected but no guardrail rule was matched."
            )
    # ─────────────────────────────────────────────────────────────────────────

    if violations:
        feedback = "Rule violations found: " + " | ".join(violations)
        logger.warning("[Judge] Validation FAILED. Feedback: %s", feedback)
        return {
            "validation_passed": False,
            "judge_feedback": feedback,
        }

    logger.info("[Judge] All guardrail rules passed. Validation SUCCESSFUL.")
    return {
        "validation_passed": True,
        "judge_feedback": "",
    }
