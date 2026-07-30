"""
api/main.py
───────────
FastAPI application entry point.

Endpoints:
  GET  /                      → Health check
  POST /upload-and-analyze    → Upload a PDF and run the full LangGraph pipeline
  POST /analyze               → Run pipeline with dummy text (for testing)
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel

from api.agents.graph import compiled_graph
from api.schemas.contract_schema import ContractReport
from langsmith_config import configure_langsmith

# ─────────────────────────────────────────────
# Loguru configuration
# ─────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> — {message}",
    level="INFO",
    colorize=True,
)


class _InterceptHandler(logging.Handler):
    """Routes stdlib logging records (uvicorn, fastapi) into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno  # type: ignore[assignment]

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


for _name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
    _log = logging.getLogger(_name)
    _log.handlers = [_InterceptHandler()]
    _log.propagate = False


# ─────────────────────────────────────────────
# Application startup / shutdown
# ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle events."""
    configure_langsmith()
    logger.info("Smart Contract Audit API started.")
    yield
    logger.info("Smart Contract Audit API shutting down.")


app = FastAPI(
    title="Smart Contract Audit API",
    description=(
        "Contract analysis system powered by LangGraph agents, "
        "Pydantic-validated output, and a semantic guardrail layer."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze (dummy mode)."""

    document_name: str = "sample_contract.pdf"


class AnalyzeResponse(BaseModel):
    """Response body for analysis endpoints."""

    status: str
    report: ContractReport


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────


def _run_pipeline(initial_state: dict) -> AnalyzeResponse:
    """Invokes the LangGraph pipeline and returns the typed response."""
    try:
        final_state: dict = compiled_graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("LangGraph pipeline error: {}", exc)
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}") from exc

    final_report_data = final_state.get("final_report")
    if not final_report_data:
        raise HTTPException(status_code=500, detail="Report could not be generated.")

    report = ContractReport(**final_report_data)

    logger.info(
        "Analysis complete — Risky clauses: {} | Overall risk score: {:.4f}",
        len(report.risky_clauses),
        report.overall_risk_score,
    )
    return AnalyzeResponse(status="success", report=report)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────


@app.get("/", summary="Health check")
async def health_check() -> dict:
    """Returns a simple status confirming the API is running."""
    return {"status": "ok", "message": "Smart Contract Audit API is running."}


@app.post(
    "/upload-and-analyze",
    response_model=AnalyzeResponse,
    summary="Upload PDF and analyse",
    description=(
        "Accepts a PDF file upload, extracts text using PyMuPDF, "
        "and runs the full LangGraph analysis pipeline. "
        "Returns the Pydantic-validated ContractReport."
    ),
)
async def upload_and_analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    """
    Uploads a PDF contract and runs the full analysis pipeline.

    - **file**: A PDF file to analyse.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Received: " + file.filename,
        )

    file_bytes = await file.read()
    logger.info(
        "POST /upload-and-analyze — file: {} ({} bytes)",
        file.filename,
        len(file_bytes),
    )

    initial_state: dict = {
        "document_name": file.filename,
        "file_bytes": file_bytes,
        "file_path": None,
        "page_count": 0,
        "raw_text": "",
        "clauses": [],
        "analyzed_risks": [],
        "validation_passed": False,
        "retry_count": 0,
        "judge_feedback": "",
        "final_report": None,
    }

    return _run_pipeline(initial_state)


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyse with dummy text",
    description=(
        "Runs the LangGraph pipeline using built-in dummy text. Use this endpoint for testing without uploading a file."
    ),
)
async def analyze_document(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Runs the LangGraph pipeline with dummy text (for testing).

    - **document_name**: Label for the document (dummy mode).
    """
    logger.info("POST /analyze received — document: {}", request.document_name)

    initial_state: dict = {
        "document_name": request.document_name,
        "file_bytes": None,
        "file_path": None,
        "page_count": 0,
        "raw_text": "",
        "clauses": [],
        "analyzed_risks": [],
        "validation_passed": False,
        "retry_count": 0,
        "judge_feedback": "",
        "final_report": None,
    }

    return _run_pipeline(initial_state)
