"""
tests/test_agents.py
────────────────────
Unit tests for the deterministic parts of the Smart Contract Audit pipeline.

These tests do NOT require Ollama, ChromaDB, or PostgreSQL.
They verify the pure-logic components: clause extraction, judge validation,
report generation, and document reader fallback.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.agents.clause_extractor import _segment_into_clauses, clause_extractor_node
from api.agents.document_reader import document_reader_node
from api.agents.judge import _load_rules, judge_node
from api.agents.report_generator import report_generator_node
from api.schemas.contract_schema import ClauseRisk, ContractReport

# ═══════════════════════════════════════════════════════════════════
# ClauseExtractor Tests
# ═══════════════════════════════════════════════════════════════════


class TestClauseExtractor:
    """Tests for the ClauseExtractor agent's regex-based splitting."""

    def test_extracts_clauses_from_raw_text(self, sample_raw_text: str):
        """Should produce at least one clause from sample text."""
        state = {"raw_text": sample_raw_text}
        result = clause_extractor_node(state)

        assert "clauses" in result
        assert len(result["clauses"]) >= 1

    def test_clause_ids_are_sequential(self, sample_raw_text: str):
        """Each clause should have a sequential ID like clause_001, clause_002."""
        state = {"raw_text": sample_raw_text}
        result = clause_extractor_node(state)

        for i, clause in enumerate(result["clauses"]):
            expected_id = f"clause_{str(i + 1).zfill(3)}"
            assert clause["clause_id"] == expected_id

    def test_empty_text_returns_no_clauses(self):
        """Empty input should return an empty clause list."""
        state = {"raw_text": ""}
        result = clause_extractor_node(state)

        assert result["clauses"] == []

    def test_numbered_section_detection(self):
        """Numbered sections (e.g., '1. ', '2.3 ') should start new clauses."""
        text = (
            "1. This is the first section of the contract agreement document.\n"
            "It continues here with additional details and obligations.\n"
            "2. This is the second section with entirely different terms and conditions.\n"
            "More text in the second clause with specific requirements.\n"
        )
        segments = _segment_into_clauses(text)
        assert len(segments) == 2

    def test_short_fragments_are_filtered(self):
        """Fragments under 40 characters should be discarded."""
        text = "Short.\n\n1. This is a sufficiently long clause that passes the minimum length filter."
        segments = _segment_into_clauses(text)
        # "Short." should be filtered, only the long clause remains
        assert all(len(s) >= 40 for s in segments)


# ═══════════════════════════════════════════════════════════════════
# DocumentReader Tests
# ═══════════════════════════════════════════════════════════════════


class TestDocumentReader:
    """Tests for the DocumentReader agent's fallback behavior."""

    def test_dummy_fallback_when_no_file(self):
        """Should return dummy text when no file_bytes or file_path is given."""
        state = {
            "document_name": "test.pdf",
            "file_bytes": None,
            "file_path": None,
            "raw_text": "",
            "page_count": 0,
        }
        result = document_reader_node(state)

        assert result["raw_text"]  # Non-empty
        assert result["page_count"] == 1
        assert "CLAUSE 1" in result["raw_text"]

    def test_preserves_existing_raw_text(self):
        """If raw_text is already set, should skip file reading."""
        existing = "This is pre-injected text for the evaluation harness."
        state = {
            "document_name": "test.pdf",
            "file_bytes": None,
            "file_path": None,
            "raw_text": existing,
            "page_count": 5,
        }
        result = document_reader_node(state)

        assert result["raw_text"] == existing
        assert result["page_count"] == 5


# ═══════════════════════════════════════════════════════════════════
# Judge Tests
# ═══════════════════════════════════════════════════════════════════


class TestJudge:
    """Tests for the Judge agent's rule-based validation."""

    def test_rules_load_from_yaml(self):
        """Should successfully load rules from guardrails/rules.yaml."""
        rules = _load_rules()
        assert isinstance(rules, list)
        assert len(rules) >= 1

        # Each rule should have required fields
        for rule in rules:
            assert "id" in rule
            assert "clause_type" in rule
            assert "severity" in rule

    def test_validation_passes_with_correct_severity(self):
        """If LLM severity matches the rule, validation should pass."""
        state = {
            "analyzed_risks": [
                {
                    "clause_id": "clause_001",
                    "clause_text": "Termination notice is 5 days.",
                    "risk_level": "high",
                    "explanation": "Below minimum.",
                    "recommendation": "Extend to 15 days.",
                    "matched_rule": "keyword_match::termination",
                },
            ],
        }
        result = judge_node(state)

        # Should pass because termination is rated "high" which meets the rule's "high" severity
        assert result["validation_passed"] is True

    def test_missing_clause_detection(self):
        """Should add missing clause findings when clause_type is absent."""
        # Provide analyzed_risks with only "termination" — no "penalty" or "governing_law"
        state = {
            "analyzed_risks": [
                {
                    "clause_id": "clause_001",
                    "clause_text": "Termination notice is 30 days.",
                    "risk_level": "high",
                    "explanation": "Standard.",
                    "recommendation": "None.",
                    "matched_rule": "llm_analysis::termination",
                },
            ],
        }
        result = judge_node(state)

        # The judge should have added missing clause findings
        updated_risks = result.get("analyzed_risks", [])
        missing_ids = [r["clause_id"] for r in updated_risks if r["clause_id"].startswith("missing_")]
        assert len(missing_ids) >= 1  # At least penalty or governing_law should be missing

    def test_severity_mismatch_triggers_reanalysis(self):
        """If LLM rates 'low' but rule says 'high', should trigger re-analysis."""
        state = {
            "analyzed_risks": [
                {
                    "clause_id": "clause_001",
                    "clause_text": "Termination notice is 5 days.",
                    "risk_level": "low",  # Too low — rule requires "high"
                    "explanation": "Seems fine.",
                    "recommendation": "None.",
                    "matched_rule": "keyword_match::termination",
                },
            ],
        }
        result = judge_node(state)

        assert result["validation_passed"] is False
        assert result["judge_feedback"]  # Should contain violation details


# ═══════════════════════════════════════════════════════════════════
# ReportGenerator Tests
# ═══════════════════════════════════════════════════════════════════


class TestReportGenerator:
    """Tests for the ReportGenerator agent's Pydantic validation."""

    def test_generates_valid_report(self, sample_analyzed_risks: list[dict], sample_clauses: list[dict]):
        """Should produce a valid ContractReport dict."""
        state = {
            "document_name": "test_contract.pdf",
            "clauses": sample_clauses,
            "analyzed_risks": sample_analyzed_risks,
        }
        result = report_generator_node(state)

        assert "final_report" in result
        report = result["final_report"]

        # Validate through Pydantic
        validated = ContractReport(**report)
        assert validated.document_name == "test_contract.pdf"
        assert validated.total_clauses == len(sample_clauses)

    def test_risk_score_calculation(self, sample_clauses: list[dict]):
        """Risk score should be bounded [0, 1] and calculated correctly."""
        risks = [
            {
                "clause_id": "clause_001",
                "clause_text": "Test clause",
                "risk_level": "high",
                "explanation": "High risk.",
                "recommendation": "Fix it.",
                "matched_rule": "test",
            },
            {
                "clause_id": "clause_002",
                "clause_text": "Test clause 2",
                "risk_level": "medium",
                "explanation": "Medium risk.",
                "recommendation": "Consider fixing.",
                "matched_rule": "test",
            },
            {
                "clause_id": "clause_003",
                "clause_text": "Test clause 3",
                "risk_level": "low",
                "explanation": "Safe.",
                "recommendation": "None.",
                "matched_rule": "test",
            },
        ]
        state = {
            "document_name": "test.pdf",
            "clauses": sample_clauses,
            "analyzed_risks": risks,
        }
        result = report_generator_node(state)
        report = ContractReport(**result["final_report"])

        assert 0.0 <= report.overall_risk_score <= 1.0
        # 1 high (1.0) + 1 medium (0.5) = 1.5 / 3 clauses = 0.5
        assert report.overall_risk_score == 0.5

    def test_empty_risks_produce_zero_score(self):
        """No risky clauses should result in a zero risk score."""
        state = {
            "document_name": "clean_contract.pdf",
            "clauses": [{"clause_id": "clause_001", "text": "Safe clause"}],
            "analyzed_risks": [],
        }
        result = report_generator_node(state)
        report = ContractReport(**result["final_report"])

        assert report.overall_risk_score == 0.0
        assert report.risky_clauses == []


# ═══════════════════════════════════════════════════════════════════
# Pydantic Schema Tests
# ═══════════════════════════════════════════════════════════════════


class TestSchemas:
    """Tests for the Pydantic data schemas."""

    def test_clause_risk_validation(self):
        """ClauseRisk should accept valid risk levels and reject invalid ones."""
        valid = ClauseRisk(
            clause_id="clause_001",
            clause_text="Test clause",
            risk_level="high",
            explanation="Risky",
            recommendation="Fix it",
        )
        assert valid.risk_level == "high"

        with pytest.raises(ValidationError):
            ClauseRisk(
                clause_id="clause_001",
                clause_text="Test",
                risk_level="critical",  # type: ignore
                explanation="Test",
                recommendation="Test",
            )

    def test_contract_report_risk_score_bounds(self):
        """Overall risk score must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            ContractReport(
                document_name="test.pdf",
                total_clauses=1,
                risky_clauses=[],
                overall_risk_score=1.5,  # Out of bounds
            )
