"""
api/agents/document_reader.py
─────────────────────────────
DocumentReader agent — extracts plain text from uploaded documents.

Supports:
  - PDF files: text extracted using PyMuPDF (fitz)
  - Fallback: if a page yields no text (scanned/image), logs a warning
    (Tesseract OCR integration is planned for a future phase)

The node accepts either:
  - A file path in state["file_path"] (for local testing)
  - Raw bytes in state["file_bytes"] (for FastAPI UploadFile integration)

If neither is provided, falls back to dummy text for backward compatibility.
"""

from __future__ import annotations

import io

import fitz  # PyMuPDF
from loguru import logger


# ── DUMMY FALLBACK (kept for backward compatibility with test_pipeline.py) ──
_DUMMY_TEXT = (
    "CLAUSE 1 — DURATION: This agreement is valid for 3 months; "
    "termination notice must be given 5 days in advance.\n"
    "CLAUSE 2 — PENALTY: The contract contains no provision "
    "regarding late-payment penalties.\n"
    "CLAUSE 3 — LIABILITY: No upper cap has been imposed on "
    "the total liability of either party.\n"
    "CLAUSE 4 — CONFIDENTIALITY: Confidentiality obligations "
    "are indefinite and unreasonably broad in scope.\n"
)


def _extract_text_from_pdf(source: str | bytes) -> tuple[str, int]:
    """
    Extracts text from a PDF file path or raw bytes.

    Args:
        source: Either a file path (str) or raw PDF bytes.

    Returns:
        Tuple of (extracted_text, page_count).
    """
    if isinstance(source, bytes):
        doc = fitz.open(stream=source, filetype="pdf")
    else:
        doc = fitz.open(source)

    pages_text: list[str] = []
    empty_pages: list[int] = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages_text.append(text)
        else:
            empty_pages.append(page_num)
            logger.warning(
                "[DocumentReader] Page {} yielded no text (possibly scanned/image). "
                "OCR integration is planned for a future phase.",
                page_num,
            )

    page_count = len(doc)
    doc.close()

    full_text = "\n\n".join(pages_text)
    logger.info(
        "[DocumentReader] Extracted {} characters from {} pages ({} empty)",
        len(full_text),
        page_count,
        len(empty_pages),
    )
    return full_text, page_count


def document_reader_node(state: dict) -> dict:
    """
    Reads the raw document and converts it to plain text.

    Checks state for "file_bytes" or "file_path". If neither exists,
    falls back to dummy text for backward compatibility with stubs.

    Args:
        state: Current AgentState.

    Returns:
        Updated state slice: {"raw_text": str, "page_count": int}
    """
    doc_name = state.get("document_name", "N/A")
    logger.info("[DocumentReader] Agent running — reading document: {}", doc_name)

    file_bytes: bytes | None = state.get("file_bytes")
    file_path: str | None = state.get("file_path")

    if file_bytes:
        logger.info("[DocumentReader] Processing uploaded file bytes ({} bytes)", len(file_bytes))
        raw_text, page_count = _extract_text_from_pdf(file_bytes)
    elif file_path:
        logger.info("[DocumentReader] Processing file from path: {}", file_path)
        raw_text, page_count = _extract_text_from_pdf(file_path)
    else:
        logger.warning("[DocumentReader] No file provided — using dummy text for testing.")
        raw_text = _DUMMY_TEXT
        page_count = 1

    return {"raw_text": raw_text, "page_count": page_count}
