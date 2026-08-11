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
    if isinstance(source, bytes):  # noqa: SIM108
        doc = fitz.open(stream=source, filetype="pdf")
    else:
        doc = fitz.open(source)

    pages_text: list[str] = []
    empty_pages: list[int] = []

    for page_num, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        text = raw_text.strip() if isinstance(raw_text, str) else ""
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

    Priority order:
      1. If raw_text is already set in state (e.g., injected by evaluation
         harness or pre-processing step) — use it directly, skip file reading.
      2. If file_bytes is set — extract text from uploaded PDF bytes.
      3. If file_path is set — extract text from a local PDF path.
      4. Fallback: dummy text for backward compatibility with test_pipeline.py.

    Args:
        state: Current AgentState.

    Returns:
        Updated state slice: {"raw_text": str, "page_count": int}
    """
    doc_name = state.get("document_name", "N/A")
    logger.info("[DocumentReader] Agent running — reading document: {}", doc_name)

    # ── Fast-path: raw_text already provided ────────────────────────────────
    existing_raw_text: str = state.get("raw_text", "")
    if existing_raw_text.strip():
        logger.info(
            "[DocumentReader] raw_text already set ({} chars) — skipping file read.",
            len(existing_raw_text),
        )
        page_count: int = state.get("page_count", 1) or 1
        return {"raw_text": existing_raw_text, "page_count": page_count}
    # ────────────────────────────────────────────────────────────────────────

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
