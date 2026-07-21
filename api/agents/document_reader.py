"""
api/agents/document_reader.py
─────────────────────────────
[STUB] DocumentReader agent.

Real implementation:
  - PDF files: text extracted using PyMuPDF (fitz).
  - Scanned / visually signed documents: Tesseract OCR is triggered.
  - Output: raw text + page / location metadata.

Current: Returns static dummy text and updates State.
"""

from __future__ import annotations

from loguru import logger


def document_reader_node(state: dict) -> dict:
    """
    Reads the raw document and converts it to plain text.

    Args:
        state: Current AgentState.

    Returns:
        Updated state slice: {"raw_text": str}
    """
    logger.info("[DocumentReader] Agent running — reading document: %s", state.get("document_name", "N/A"))

    # ── DUMMY OUTPUT ──────────────────────────────────────────────────────────
    dummy_text = (
        "CLAUSE 1 — DURATION: This agreement is valid for 3 months; termination notice must be given 5 days in advance.\n"
        "CLAUSE 2 — PENALTY: The contract contains no provision regarding late-payment penalties.\n"
        "CLAUSE 3 — LIABILITY: No upper cap has been imposed on the total liability of either party.\n"
        "CLAUSE 4 — CONFIDENTIALITY: Confidentiality obligations are indefinite and unreasonably broad in scope.\n"
    )
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("[DocumentReader] Raw text extracted successfully (%d characters)", len(dummy_text))

    return {"raw_text": dummy_text}
