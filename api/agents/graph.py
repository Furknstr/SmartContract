"""
api/agents/graph.py
───────────────────
LangGraph shared state (AgentState) and graph definition.

Pipeline flow:
  document_reader → clause_extractor → risk_analyzer → judge
                                                          │
                                       if invalid ◄───────┘  (retry loop)
                                                          │
                                       if valid           ▼
                                        report_generator → END
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from api.agents.clause_extractor import clause_extractor_node
from api.agents.document_reader import document_reader_node
from api.agents.judge import judge_node
from api.agents.report_generator import report_generator_node
from api.agents.risk_analyzer import risk_analyzer_node

# ─────────────────────────────────────────────
# 1. SHARED MEMORY: AgentState
# ─────────────────────────────────────────────


class AgentState(TypedDict):
    """
    Shared state object passed between LangGraph nodes.

    Every field can be read and updated by the next agent in the pipeline.
    """

    # Input
    document_name: str
    file_bytes: bytes | None  # Raw PDF bytes from UploadFile (set by FastAPI endpoint)
    file_path: str | None  # Local file path alternative (for testing)
    raw_text: str  # Output of DocumentReader

    # Intermediate layers
    clauses: list[dict]  # ClauseExtractor output: [{"clause_id": ..., "text": ...}, ...]
    analyzed_risks: list[dict]  # RiskAnalyzer output: [{"clause_id": ..., "risk_level": ..., ...}, ...]
    page_count: int  # Number of pages extracted by DocumentReader

    # Control flow
    validation_passed: bool  # Judge decision: True → generate report, False → re-analyse
    retry_count: int  # Counter to prevent infinite loops
    judge_feedback: str  # Feedback from Judge sent back to RiskAnalyzer (on loop)

    # Output
    final_report: dict | None  # ReportGenerator output (Pydantic dict)


# ─────────────────────────────────────────────
# 2. ROUTING FUNCTION (Conditional Edge)
# ─────────────────────────────────────────────

MAX_RETRIES = 3


def route_after_judge(state: AgentState) -> str:
    """
    Determines the next node based on the Judge's decision.

    Returns:
        "report_generator" → validation passed, generate report
        "risk_analyzer"    → rule violation found, re-analyse
        "report_generator" → maximum retries exceeded, force report
    """
    if state["validation_passed"]:
        logger.info("[Router] Validation passed → ReportGenerator")
        return "report_generator"

    if state["retry_count"] >= MAX_RETRIES:
        logger.warning(
            "[Router] Maximum retries (%d) exceeded → ReportGenerator (forced)",
            MAX_RETRIES,
        )
        return "report_generator"

    logger.info(
        "[Router] Rule violation detected (attempt %d/%d) → RiskAnalyzer",
        state["retry_count"],
        MAX_RETRIES,
    )
    return "risk_analyzer"


# ─────────────────────────────────────────────
# 3. GRAPH DEFINITION
# ─────────────────────────────────────────────


def build_graph() -> CompiledStateGraph:
    """Builds and compiles the LangGraph pipeline."""

    graph = StateGraph(AgentState)  # type: ignore[type-var]

    # Register nodes
    graph.add_node("document_reader", document_reader_node)  # type: ignore[type-var]
    graph.add_node("clause_extractor", clause_extractor_node)  # type: ignore[type-var]
    graph.add_node("risk_analyzer", risk_analyzer_node)  # type: ignore[type-var]
    graph.add_node("judge", judge_node)  # type: ignore[type-var]
    graph.add_node("report_generator", report_generator_node)  # type: ignore[type-var]

    # Define edges — linear flow
    graph.set_entry_point("document_reader")
    graph.add_edge("document_reader", "clause_extractor")
    graph.add_edge("clause_extractor", "risk_analyzer")
    graph.add_edge("risk_analyzer", "judge")

    # Conditional edge: judge → loop or report
    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "risk_analyzer": "risk_analyzer",
            "report_generator": "report_generator",
        },
    )

    graph.add_edge("report_generator", END)

    return graph.compile()


# Single compiled graph instance shared across the application
compiled_graph = build_graph()
