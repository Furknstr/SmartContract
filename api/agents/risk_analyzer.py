"""
api/agents/risk_analyzer.py
───────────────────────────
RiskAnalyzer agent — analyses each clause using Ollama LLM + ChromaDB RAG.

Pipeline per clause:
  1. Query ChromaDB for the 3 most similar standard clauses (RAG retrieval)
  2. Build a structured prompt with the clause + standard examples
  3. Send to Ollama (qwen2.5:7b) for risk assessment
  4. Parse the LLM's JSON response into a risk finding
  5. Fall back to keyword-based analysis if LLM/RAG is unavailable
"""

from __future__ import annotations

import json
import os
import re

import httpx
from dotenv import load_dotenv
from loguru import logger

from rag.vectorstore import get_chroma_client, get_or_create_collection

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
RAG_TOP_K: int = 3  # Number of similar standard clauses to retrieve
OLLAMA_TIMEOUT: float = 120.0  # Seconds — local LLM can be slow

# ─────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a contract risk analyst. You will be given a clause from a contract and similar standard clauses from a reference database.

Your job:
1. Determine the clause type (e.g. termination, liability, confidentiality, governing_law, penalty, indemnification, ip_ownership, non_compete, renewal, other).
2. Compare the clause against the standard examples.
3. Assess the risk level: "low" (standard/safe), "medium" (unusual but not dangerous), or "high" (risky, missing protections, or non-standard).
4. Explain WHY it is risky or safe.
5. Provide a specific recommendation if the risk is medium or high.

You MUST respond with ONLY valid JSON in this exact format (no markdown, no explanation outside the JSON):
{
  "clause_type": "termination",
  "risk_level": "high",
  "explanation": "The termination clause allows only 5 days notice, which is below the standard 30-day minimum found in reference contracts.",
  "recommendation": "Extend the termination notice period to at least 30 days to align with standard practice."
}

If the clause is safe/standard, set risk_level to "low" and recommendation to "No changes recommended."
"""

_USER_PROMPT_TEMPLATE = """## Clause to Analyze
{clause_text}

## Similar Standard Clauses from Reference Database
{standard_clauses}

Analyze the clause above and respond with JSON only."""


# ─────────────────────────────────────────────
# Keyword fallback (used when Ollama is unavailable)
# ─────────────────────────────────────────────

_RISK_KEYWORDS: dict[str, tuple[str, str, str]] = {
    "penalty": (
        "high",
        "The contract contains no provision regarding late-payment penalties.",
        "A standard late-payment penalty clause should be added (e.g. 0.1% per day).",
    ),
    "liability": (
        "high",
        "No upper cap has been imposed on the total liability amount.",
        "Liability should be capped at the total contract value.",
    ),
    "confidentiality": (
        "medium",
        "Confidentiality obligations are indefinite and unreasonably broad in scope.",
        "The confidentiality period should be limited to a fixed term (e.g. 3 years).",
    ),
    "termination": (
        "high",
        "The termination notice period is below the legal minimum (15 days).",
        "The termination notice period must be extended to at least 15 days.",
    ),
}


def _query_chromadb(clause_text: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """
    Queries ChromaDB for the most similar standard clauses.

    Returns a list of dicts with 'text' and 'metadata' keys.
    Returns an empty list if ChromaDB is unreachable.
    """
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)

        if collection.count() == 0:
            logger.warning("[RiskAnalyzer] ChromaDB collection is empty. Skipping RAG.")
            return []

        results = collection.query(
            query_texts=[clause_text],
            n_results=top_k,
        )

        standard_clauses = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)

            for doc, meta in zip(docs, metas):
                standard_clauses.append({
                    "text": doc,
                    "metadata": meta,
                })

        return standard_clauses

    except Exception as e:
        logger.warning("[RiskAnalyzer] ChromaDB query failed: {}. Proceeding without RAG.", e)
        return []


def _call_ollama(clause_text: str, standard_clauses: list[dict]) -> dict | None:
    """
    Calls Ollama with the clause + standard clauses and parses the JSON response.

    Returns a dict with keys: clause_type, risk_level, explanation, recommendation.
    Returns None if Ollama is unreachable or response is unparseable.
    """
    # Format standard clauses for the prompt
    if standard_clauses:
        formatted_standards = "\n\n".join(
            f"[Standard {i+1} — {sc['metadata'].get('clause_type', 'Unknown')}]\n{sc['text']}"
            for i, sc in enumerate(standard_clauses)
        )
    else:
        formatted_standards = "(No standard clauses found in reference database)"

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        clause_text=clause_text,
        standard_clauses=formatted_standards,
    )

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": user_prompt,
                "system": _SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()

        result = response.json()
        raw_text = result.get("response", "").strip()

        # Try to extract JSON from the response (LLM may wrap it in markdown)
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if not json_match:
            logger.warning("[RiskAnalyzer] LLM response contained no JSON: {}", raw_text[:200])
            return None

        parsed = json.loads(json_match.group())

        # Validate required keys
        required_keys = {"risk_level", "explanation", "recommendation"}
        if not required_keys.issubset(parsed.keys()):
            logger.warning("[RiskAnalyzer] LLM JSON missing keys: {}", parsed)
            return None

        # Normalize risk_level
        if parsed["risk_level"] not in ("low", "medium", "high"):
            parsed["risk_level"] = "low"

        return parsed

    except httpx.ConnectError:
        logger.warning(
            "[RiskAnalyzer] Cannot connect to Ollama at {}. Is it running?",
            OLLAMA_BASE_URL,
        )
        return None
    except httpx.TimeoutException:
        logger.warning("[RiskAnalyzer] Ollama request timed out after {}s.", OLLAMA_TIMEOUT)
        return None
    except Exception as e:
        logger.warning("[RiskAnalyzer] Ollama call failed: {}", e)
        return None


def _keyword_fallback(clause_text: str) -> dict:
    """Keyword-based fallback when Ollama/RAG is unavailable."""
    clause_lower = clause_text.lower()
    for keyword, (level, expl, rec) in _RISK_KEYWORDS.items():
        if keyword in clause_lower:
            return {
                "risk_level": level,
                "explanation": expl,
                "recommendation": rec,
                "clause_type": keyword,
            }
    return {
        "risk_level": "low",
        "explanation": "Standard clause — no significant risk detected.",
        "recommendation": "No changes recommended.",
        "clause_type": "other",
    }


def risk_analyzer_node(state: dict) -> dict:
    """
    Analyses each clause for risk using Ollama LLM + ChromaDB RAG.

    Falls back to keyword matching if Ollama is unavailable.
    During a retry loop, only re-analyzes clauses that were explicitly flagged by the Judge.

    Args:
        state: Current AgentState. Reads "clauses" and "judge_feedback" fields.

    Returns:
        Updated state slice: {
            "analyzed_risks": list[dict],
            "retry_count": int,
            "validation_passed": bool,
        }
    """
    retry_count: int = state.get("retry_count", 0)
    previous_risks: list[dict] = state.get("analyzed_risks", [])
    
    flagged_ids: set[str] = set()

    if retry_count == 0:
        logger.info("[RiskAnalyzer] Agent running — starting initial analysis.")
    else:
        feedback = state.get("judge_feedback", "")
        # Extract clause IDs like "clause_006" from the Judge's feedback
        flagged_ids = set(re.findall(r"clause_\d{3}", feedback))
        
        logger.info(
            "[RiskAnalyzer] Re-analysis (attempt {}) — Judge feedback: {}",
            retry_count,
            feedback,
        )
        logger.info("[RiskAnalyzer] Only re-analyzing {} flagged clause(s).", len(flagged_ids))

    clauses: list[dict] = state.get("clauses", [])
    analyzed_risks: list[dict] = []
    ollama_available: bool | None = None  # Will be determined on first call

    for i, clause in enumerate(clauses):
        clause_text = clause["text"]
        clause_id = clause["clause_id"]

        # -------------------------------------------------------------
        # OPTIMIZATION: If this is a retry and this clause wasn't flagged,
        # skip LLM processing and reuse the previous result to save time.
        # -------------------------------------------------------------
        if retry_count > 0 and clause_id not in flagged_ids:
            prev_result = next((r for r in previous_risks if r["clause_id"] == clause_id), None)
            if prev_result:
                analyzed_risks.append(prev_result)
                continue

        logger.info(
            "[RiskAnalyzer] Analyzing clause {}/{}: {} ({}...)",
            i + 1,
            len(clauses),
            clause_id,
            clause_text[:60],
        )

        analysis: dict | None = None

        # Try Ollama + RAG first (skip if already determined unavailable)
        if ollama_available is not False:
            # Step 1: RAG retrieval
            standard_clauses = _query_chromadb(clause_text)

            # Step 2: LLM analysis (pass the feedback so the LLM knows what to fix if it's a retry)
            # We append the feedback to the prompt for the specific flagged clause
            prompt_text = clause_text
            if retry_count > 0 and clause_id in flagged_ids:
                prompt_text += f"\n\n[CRITICAL JUDGE FEEDBACK TO FIX]: {feedback}"
                
            analysis = _call_ollama(prompt_text, standard_clauses)

            if analysis is None and ollama_available is None:
                ollama_available = False
                logger.warning(
                    "[RiskAnalyzer] Ollama unavailable. Falling back to keyword matching for all clauses."
                )
            elif analysis is not None:
                ollama_available = True

        # Fallback to keywords
        if analysis is None:
            analysis = _keyword_fallback(clause_text)
            matched_rule = f"keyword_match::{analysis['clause_type']}"
        else:
            matched_rule = f"llm_analysis::{analysis.get('clause_type', 'unknown')}"

        analyzed_risks.append({
            "clause_id": clause_id,
            "clause_text": clause_text,
            "risk_level": analysis["risk_level"],
            "explanation": analysis["explanation"],
            "recommendation": analysis["recommendation"],
            "matched_rule": matched_rule,
        })

    high_count = sum(1 for r in analyzed_risks if r["risk_level"] == "high")
    medium_count = sum(1 for r in analyzed_risks if r["risk_level"] == "medium")

    logger.info(
        "[RiskAnalyzer] {} clause(s) analysed. High: {}, Medium: {}, Low: {}. Method: {}",
        len(analyzed_risks),
        high_count,
        medium_count,
        len(analyzed_risks) - high_count - medium_count,
        "Ollama+RAG" if ollama_available else "keyword fallback",
    )

    return {
        "analyzed_risks": analyzed_risks,
        "retry_count": retry_count + 1,
        "validation_passed": False,
    }
