"""
rag/reranker.py
───────────────
Cross-Encoder reranker for the RAG retrieval pipeline.

How it fits in:
  1. ChromaDB does a fast vector-similarity search (wide net, top-10 results)
  2. This module scores every candidate by reading the query and the candidate
     *together* using BAAI/bge-reranker-base — much more accurate than vectors
  3. Only the top-3 highest-scoring candidates are returned to the LLM

The model (~270 MB) is downloaded from HuggingFace on first use and cached
locally. No API key required — it runs 100% locally.

Graceful degradation: if sentence-transformers is not installed or the model
fails to load, the module logs a warning and returns candidates in the original
vector-similarity order. The pipeline continues unaffected.
"""

from __future__ import annotations

from loguru import logger

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

_MODEL_NAME = "BAAI/bge-reranker-base"

# Lazy-loaded singleton — the model is loaded once on first call to rerank(),
# then reused for every subsequent clause. Loading takes ~2–5 seconds the
# first time; subsequent calls are instant.
_model = None


# ─────────────────────────────────────────────
# Internal: model loader
# ─────────────────────────────────────────────


def _load_model():
    """
    Loads the CrossEncoder model on first call; returns cached instance thereafter.

    Returns None if sentence-transformers is not installed or the model fails to
    load, allowing the caller to degrade gracefully.
    """
    global _model  # noqa: PLW0603

    if _model is not None:
        return _model

    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        logger.info(
            "[Reranker] Loading model: {} "
            "(first-time use will download ~270 MB — subsequent starts are instant)",
            _MODEL_NAME,
        )
        _model = CrossEncoder(_MODEL_NAME, max_length=512)
        logger.info("[Reranker] Model loaded and ready.")
        return _model

    except ImportError:
        logger.warning(
            "[Reranker] sentence-transformers is not installed. "
            "Run `uv sync` to enable reranking. Falling back to vector-similarity order."
        )
        return None

    except Exception as e:
        logger.warning(
            "[Reranker] Failed to load model '{}': {}. "
            "Falling back to vector-similarity order.",
            _MODEL_NAME,
            e,
        )
        return None


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """
    Re-ranks a list of candidate clauses by their relevance to the query clause.

    The cross-encoder reads the (query, candidate) pair jointly — unlike vector
    search which embeds them separately — and produces a precise relevance score.

    Args:
        query:      The clause text being analysed by the RiskAnalyzer.
        candidates: Dicts with 'text' and 'metadata' keys, as returned by ChromaDB.
        top_k:      How many candidates to return after reranking.

    Returns:
        Up to top_k candidates sorted by cross-encoder score (most relevant first).
        Returns candidates[:top_k] in original order if reranking is unavailable.
    """
    if not candidates:
        return candidates

    # Nothing to rerank if we have fewer candidates than requested
    if len(candidates) <= top_k:
        return candidates

    model = _load_model()

    if model is None:
        # Reranker unavailable — return top-k from vector-similarity order
        logger.debug("[Reranker] Skipping rerank — returning top-{} by vector order.", top_k)
        return candidates[:top_k]

    try:
        pairs = [(query, c["text"]) for c in candidates]
        raw_scores = model.predict(pairs)  # type: ignore[union-attr]
        scores: list[float] = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)

        ranked = sorted(zip(scores, candidates, strict=False), key=lambda x: x[0], reverse=True)
        result = [c for _, c in ranked[:top_k]]

        best_score  = ranked[0][0]  if ranked else 0.0
        worst_score = ranked[-1][0] if ranked else 0.0
        logger.info(
            "[Reranker] Re-ranked {} → {} candidates "
            "(scores: best={:.4f}, {}th={:.4f})",
            len(candidates),
            len(result),
            best_score,
            top_k,
            ranked[top_k - 1][0] if len(ranked) >= top_k else worst_score,
        )
        return result

    except Exception as e:
        logger.warning(
            "[Reranker] Reranking failed: {}. "
            "Returning top-{} candidates in vector-similarity order.",
            e,
            top_k,
        )
        return candidates[:top_k]
