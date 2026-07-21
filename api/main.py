"""
api/main.py
───────────
FastAPI application entry point.

Endpoints:
  GET  /         → Health check
  POST /analyze  → Triggers the LangGraph pipeline and returns the final report
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from api.agents.graph import compiled_graph
from api.schemas.contract_schema import ContractReport

# ─────────────────────────────────────────────
# Loguru configuration
# ─────────────────────────────────────────────

# Remove the default loguru sink and replace with a formatted one
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


# Intercept uvicorn / fastapi stdlib loggers
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
    """Request body for POST /analyze."""

    document_name: str = "sample_contract.pdf"
    # Real implementation: file upload via FastAPI's UploadFile.
    # For now, only the document name is sent; DocumentReader generates dummy text.


class AnalyzeResponse(BaseModel):
    """Response body for POST /analyze."""

    status: str
    report: ContractReport


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────


@app.get("/", summary="Health check")
async def health_check() -> dict:
    """Returns a simple status confirming the API is running."""
    return {"status": "ok", "message": "Smart Contract Audit API is running."}


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Contract analysis",
    description=(
        "Triggers the LangGraph pipeline for the given document name. "
        "Pipeline: DocumentReader → ClauseExtractor → RiskAnalyzer → "
        "Judge (retry loop) → ReportGenerator. "
        "Result is returned as a Pydantic-validated ContractReport."
    ),
)
async def analyze_document(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Runs the LangGraph pipeline and returns the analysis report.

    - **document_name**: Name of the document to analyse (stub mode: used as label only).
    """
    logger.info("POST /analyze received — document: {}", request.document_name)

    # Initial state
    initial_state: dict = {
        "document_name": request.document_name,
        "raw_text": "",
        "clauses": [],
        "analyzed_risks": [],
        "validation_passed": False,
        "retry_count": 0,
        "judge_feedback": "",
        "final_report": None,
    }

    try:
        # Run the LangGraph pipeline (synchronous invoke; can be moved to BackgroundTasks later)
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
