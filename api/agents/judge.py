"""
api/agents/judge.py
───────────────────
Judge / Validator (Semantic Guardrail) agent.

Loads deterministic rules from guardrails/rules.yaml and validates
the RiskAnalyzer's output against them. No LLM is involved — this
layer is fast, deterministic, and hallucination-proof.

Validation checks:
  1. Clause type matches a rule → verify the LLM's risk level is at least
     as severe as the rule's severity.
  2. A rule's clause_type is not present in the analyzed clauses at all →
     flag as "missing clause" violation.
  3. High-risk findings without a clear explanation → flag for re-analysis.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger

# ─────────────────────────────────────────────
# Load rules from YAML
# ─────────────────────────────────────────────

_RULES_PATH = Path(__file__).resolve().parent.parent.parent / "guardrails" / "rules.yaml"
_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


def _load_rules() -> list[dict]:
    """
    Loads guardrail rules from rules.yaml.

    Returns:
        List of rule dicts, each with: id, clause_type, check, severity, message.
        Returns empty list if file not found.
    """
    if not _RULES_PATH.exists():
        logger.warning(
            "[Judge] Rules file not found at {}. Running without guardrails.",
            _RULES_PATH,
        )
        return []

    with open(_RULES_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules: list[dict] = data.get("rules", [])
    logger.info("[Judge] Loaded {} guardrail rules from {}", len(rules), _RULES_PATH.name)
    return rules


def judge_node(state: dict) -> dict:
    """
    Validates RiskAnalyzer output against guardrail rules.

    Checks:
      1. Severity agreement: If a rule says clause_type X must be at least
         "high" severity, but the LLM rated it "low", flag a violation.
      2. Missing clause detection: If a rule requires a clause_type to be
         present (check contains "clause_present"), verify at least one
         analyzed clause matches that type.
      3. Unmatched high risks: If a clause is flagged "high" but has no
         recognizable clause_type, flag for review.

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
    rules = _load_rules()

    if not rules:
        logger.info("[Judge] No rules loaded. Passing validation by default.")
        return {
            "validation_passed": True,
            "judge_feedback": "",
        }

    # Violations that warrant re-analysis (severity mismatches)
    reanalysis_violations: list[str] = []

    # Informational findings (missing clauses — added directly to results)
    missing_clause_findings: list[dict] = []

    # Build a lookup of clause_types found in the analysis
    # The matched_rule field contains "llm_analysis::termination" or "keyword_match::liability"
    found_clause_types: set[str] = set()
    for risk in analyzed_risks:
        rule_ref = risk.get("matched_rule", "")
        if "::" in rule_ref:
            clause_type = rule_ref.split("::", 1)[1]
            found_clause_types.add(clause_type.lower())

    for rule in rules:
        rule_id: str = rule.get("id", "unknown")
        rule_clause_type: str = rule.get("clause_type", "").lower()
        rule_severity: str = rule.get("severity", "low").lower()
        rule_message: str = rule.get("message", "")
        rule_check: str = rule.get("check", "")

        # Check 1: Missing clause detection
        # These are INFORMATIONAL — the RiskAnalyzer can't create clauses
        # that don't exist. Add them as findings instead of looping.
        if "clause_present" in rule_check:
            if rule_clause_type not in found_clause_types:
                logger.info(
                    "[Judge] Missing clause detected: '{}' — adding as informational finding.",
                    rule_clause_type,
                )
                missing_clause_findings.append(
                    {
                        "clause_id": f"missing_{rule_clause_type}",
                        "clause_text": f"[MISSING] No {rule_clause_type} clause was found in this contract.",
                        "risk_level": rule_severity,
                        "explanation": rule_message,
                        "recommendation": f"A {rule_clause_type} clause should be added to the contract.",
                        "matched_rule": f"guardrail::{rule_id}",
                    }
                )
            continue

        # Check 2: Severity agreement
        # Find all analyzed risks that match this rule's clause_type
        matching_risks = [
            r for r in analyzed_risks if rule_clause_type in (r.get("matched_rule", "").split("::")[-1]).lower()
        ]

        for risk in matching_risks:
            llm_severity = risk.get("risk_level", "low").lower()
            llm_score = _SEVERITY_ORDER.get(llm_severity, 0)
            rule_score = _SEVERITY_ORDER.get(rule_severity, 0)

            # If the LLM rated it LESS severe than the rule requires, flag it
            if llm_score < rule_score:
                reanalysis_violations.append(
                    f"[{rule_id}] {risk['clause_id']}: LLM rated '{llm_severity}' "
                    f"but rule requires at least '{rule_severity}'. {rule_message}"
                )

    # Check 3: High-risk findings with no matched rule
    for risk in analyzed_risks:
        if risk["risk_level"] == "high" and risk.get("matched_rule") is None:
            reanalysis_violations.append(f"{risk['clause_id']}: High risk detected but no guardrail rule was matched.")

    # Merge missing clause findings into the analyzed_risks
    updated_risks = analyzed_risks + missing_clause_findings

    if reanalysis_violations:
        feedback = " | ".join(reanalysis_violations)
        logger.warning(
            "[Judge] Validation FAILED with {} severity violation(s). Triggering re-analysis. Feedback: {}",
            len(reanalysis_violations),
            feedback,
        )
        return {
            "analyzed_risks": updated_risks,
            "validation_passed": False,
            "judge_feedback": feedback,
        }

    if missing_clause_findings:
        logger.info(
            "[Judge] {} missing clause(s) added as findings. No re-analysis needed.",
            len(missing_clause_findings),
        )

    logger.info("[Judge] All guardrail rules passed. Validation SUCCESSFUL.")
    return {
        "analyzed_risks": updated_risks,
        "validation_passed": True,
        "judge_feedback": "",
    }
