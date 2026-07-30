"""
tests/conftest.py
─────────────────
Shared pytest fixtures for the Smart Contract Audit test suite.

These fixtures provide pre-built state objects that mirror the AgentState
used by the LangGraph pipeline, so tests can run without Ollama or ChromaDB.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_raw_text() -> str:
    """Raw contract text with identifiable clause patterns."""
    return (
        "CLAUSE 1 — DURATION: This agreement is valid for 3 months; "
        "termination notice must be given 5 days in advance.\n"
        "CLAUSE 2 — PENALTY: The contract contains no provision "
        "regarding late-payment penalties.\n"
        "CLAUSE 3 — LIABILITY: No upper cap has been imposed on "
        "the total liability of either party.\n"
        "CLAUSE 4 — CONFIDENTIALITY: Confidentiality obligations "
        "are indefinite and unreasonably broad in scope.\n"
    )


@pytest.fixture
def sample_clauses() -> list[dict]:
    """Pre-extracted clause list (as produced by ClauseExtractor)."""
    return [
        {
            "clause_id": "clause_001",
            "text": (
                "CLAUSE 1 — DURATION: This agreement is valid for 3 months; "
                "termination notice must be given 5 days in advance."
            ),
        },
        {
            "clause_id": "clause_002",
            "text": ("CLAUSE 2 — PENALTY: The contract contains no provision regarding late-payment penalties."),
        },
        {
            "clause_id": "clause_003",
            "text": ("CLAUSE 3 — LIABILITY: No upper cap has been imposed on the total liability of either party."),
        },
    ]


@pytest.fixture
def sample_analyzed_risks() -> list[dict]:
    """Analyzed risk findings (as produced by RiskAnalyzer)."""
    return [
        {
            "clause_id": "clause_001",
            "clause_text": "termination notice must be given 5 days in advance.",
            "risk_level": "high",
            "explanation": "Termination notice period is below the legal minimum.",
            "recommendation": "Extend notice period to at least 15 days.",
            "matched_rule": "keyword_match::termination",
        },
        {
            "clause_id": "clause_002",
            "clause_text": "The contract contains no provision regarding late-payment penalties.",
            "risk_level": "high",
            "explanation": "No penalty clause found.",
            "recommendation": "Add a late-payment penalty clause.",
            "matched_rule": "keyword_match::penalty",
        },
        {
            "clause_id": "clause_003",
            "clause_text": "No upper cap on liability.",
            "risk_level": "low",
            "explanation": "Standard clause.",
            "recommendation": "No changes recommended.",
            "matched_rule": "keyword_match::liability",
        },
    ]


@pytest.fixture
def sample_initial_state(sample_raw_text: str) -> dict:
    """A complete initial AgentState for pipeline testing."""
    return {
        "document_name": "test_contract.pdf",
        "file_bytes": None,
        "file_path": None,
        "page_count": 0,
        "raw_text": sample_raw_text,
        "clauses": [],
        "analyzed_risks": [],
        "validation_passed": False,
        "retry_count": 0,
        "judge_feedback": "",
        "final_report": None,
    }
