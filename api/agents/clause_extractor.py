"""
api/agents/clause_extractor.py
──────────────────────────────
[STUB] ClauseExtractor agent.

Real implementation:
  - Raw text is segmented into clauses/paragraphs using regex + LLM-assisted splitting.
  - A unique clause_id is assigned to each clause.
  - Output: [{"clause_id": "clause_001", "text": "..."}, ...]

Current: Splits raw_text on newlines and builds static clause objects.
"""

from __future__ import annotations

from loguru import logger


def clause_extractor_node(state: dict) -> dict:
    """
    Splits raw text into individual contract clauses.

    Args:
        state: Current AgentState. Reads the "raw_text" field.

    Returns:
        Updated state slice: {"clauses": list[dict]}
    """
    logger.info("[ClauseExtractor] Agent running — extracting clauses.")

    raw_text: str = state.get("raw_text", "")

    # ── DUMMY OUTPUT ──────────────────────────────────────────────────────────
    # Split raw text into non-empty lines and assign sequential clause IDs
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    clauses = [
        {"clause_id": f"clause_{str(i + 1).zfill(3)}", "text": line}
        for i, line in enumerate(lines)
    ]
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("[ClauseExtractor] %d clause(s) detected.", len(clauses))

    return {"clauses": clauses}
