"""
api/agents/clause_extractor.py
──────────────────────────────
ClauseExtractor agent — splits raw contract text into logical clauses.

Strategy:
  1. Detect section headings using regex patterns:
     - Numbered sections: "1.", "2.3", "Section 5", "ARTICLE IV"
     - Lettered sub-sections: "(a)", "(b)", "(i)", "(ii)"
     - All-caps headings: "TERMINATION", "GOVERNING LAW"
  2. Merge short orphan lines into their parent clause
  3. Assign sequential clause IDs to each logical block
  4. Filter out empty or trivially short fragments
"""

from __future__ import annotations

import re

from loguru import logger

# ─────────────────────────────────────────────
# Regex patterns for clause boundary detection
# ─────────────────────────────────────────────

# Matches patterns like: "1.", "2.3", "10.1.2", "Section 1", "SECTION 5",
# "Article II", "ARTICLE 3", "(a)", "(b)", "(i)", "(ii)", "(1)", "(2)"
_HEADING_PATTERNS: list[re.Pattern] = [
    # "Section 1" / "SECTION 1.2" / "Article IV" / "ARTICLE 3"
    re.compile(
        r"^\s*(Section|SECTION|Article|ARTICLE)\s+[\dIVXivx]+",
        re.IGNORECASE,
    ),
    # Numbered sections at the start of a line: "1.", "2.3", "10.1.2"
    re.compile(r"^\s*\d+(\.\d+)*\.\s"),
    # Lettered sub-sections: "(a)", "(b)", "(i)", "(ii)", "(1)", "(2)"
    re.compile(r"^\s*\([a-z]\)\s", re.IGNORECASE),
    re.compile(r"^\s*\([ivx]+\)\s", re.IGNORECASE),
    re.compile(r"^\s*\(\d+\)\s"),
    # All-caps heading lines (at least 3 words, all uppercase)
    re.compile(r"^[A-Z][A-Z\s,&-]{10,}$"),
]

# Minimum character length for a clause to be considered valid
_MIN_CLAUSE_LENGTH: int = 40


def _is_heading(line: str) -> bool:
    """Returns True if the line matches any known heading pattern."""
    stripped = line.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in _HEADING_PATTERNS)


def _segment_into_clauses(raw_text: str) -> list[str]:
    """
    Segments raw text into logical clause blocks.

    Lines that match heading patterns start a new clause block.
    Consecutive non-heading lines are appended to the current block.
    """
    lines = raw_text.split("\n")
    blocks: list[list[str]] = []
    current_block: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip completely empty lines (but use them as soft boundaries)
        if not stripped:
            # If we have accumulated content and hit a blank line,
            # check if the next heading starts a new clause
            continue

        if _is_heading(stripped) and current_block:
            # Start a new block — save the current one
            blocks.append(current_block)
            current_block = [stripped]
        elif _is_heading(stripped) and not current_block:
            # First heading in the document
            current_block = [stripped]
        else:
            # Non-heading line — append to current block
            current_block.append(stripped)

    # Don't forget the last block
    if current_block:
        blocks.append(current_block)

    # Join lines within each block into a single string
    clauses = [" ".join(block) for block in blocks]

    # Filter out trivially short fragments
    clauses = [c for c in clauses if len(c) >= _MIN_CLAUSE_LENGTH]

    return clauses


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

    if not raw_text.strip():
        logger.warning("[ClauseExtractor] No raw text found in state.")
        return {"clauses": []}

    clause_texts = _segment_into_clauses(raw_text)

    clauses = [{"clause_id": f"clause_{str(i + 1).zfill(3)}", "text": text} for i, text in enumerate(clause_texts)]

    logger.info(
        "[ClauseExtractor] {} logical clause(s) extracted from {} characters of raw text.",
        len(clauses),
        len(raw_text),
    )

    return {"clauses": clauses}
