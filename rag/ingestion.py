"""
rag/ingestion.py
────────────────
CUAD dataset ingestion pipeline.

Downloads the CUAD (Contract Understanding Atticus Dataset) v1 JSON file,
extracts clause-level text with their labels, and loads them into ChromaDB
as the "standard/reference" clause set for RAG comparison.

The CUAD_v1.json follows SQuAD format:
  data → [{"title": ..., "paragraphs": [{"context": ..., "qas": [...]}]}]

Usage:
    uv run python -m rag.ingestion
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request

from loguru import logger

from rag.vectorstore import get_chroma_client, get_or_create_collection

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# How many contracts to load into the reference set
MAX_CONTRACTS: int = 40
# ChromaDB has a batch size limit
BATCH_SIZE: int = 50

# Direct download URL for the CUAD v1 JSON (HuggingFace raw file)
CUAD_JSON_URL = "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json"
CUAD_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".cuad_cache.json")


def _make_doc_id(contract_title: str, clause_type: str, idx: int) -> str:
    """Generate a deterministic document ID for deduplication."""
    raw = f"{contract_title}::{clause_type}::{idx}"
    return hashlib.md5(raw.encode()).hexdigest()


def _download_cuad_json() -> dict:
    """
    Downloads CUAD_v1.json and caches it locally to avoid re-downloading.

    Returns:
        Parsed JSON dict in SQuAD format.
    """
    if os.path.exists(CUAD_CACHE_PATH):
        logger.info("Loading CUAD JSON from local cache: {}", CUAD_CACHE_PATH)
        with open(CUAD_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)

    logger.info("Downloading CUAD_v1.json from HuggingFace (~30MB)...")
    urllib.request.urlretrieve(CUAD_JSON_URL, CUAD_CACHE_PATH)
    logger.info("Download complete. Cached at: {}", CUAD_CACHE_PATH)

    with open(CUAD_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def ingest_cuad() -> int:
    """
    Downloads CUAD v1, extracts answered clause QA pairs, and loads them into ChromaDB.

    The CUAD JSON is structured as:
      data[i].title         → contract name
      data[i].paragraphs[j].qas[k].question → clause type (e.g. "Termination For Convenience")
      data[i].paragraphs[j].qas[k].answers  → list of extracted clause texts

    Returns:
        Number of clauses successfully ingested.
    """
    logger.info("Starting CUAD ingestion (max {} contracts)...", MAX_CONTRACTS)

    cuad_data = _download_cuad_json()
    all_contracts: list[dict] = cuad_data.get("data", [])
    logger.info("CUAD dataset loaded — {} contracts found", len(all_contracts))

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    existing_count = collection.count()
    if existing_count > 0:
        logger.warning(
            "Collection already has {} documents. Skipping ingestion to avoid duplicates. "
            "Delete the .cuad_cache.json and restart to re-ingest.",
            existing_count,
        )
        return existing_count

    # Process up to MAX_CONTRACTS contracts
    contracts_to_process = all_contracts[:MAX_CONTRACTS]
    docs_to_add: list[dict] = []

    for contract in contracts_to_process:
        title: str = contract.get("title", "unknown")
        paragraphs: list[dict] = contract.get("paragraphs", [])

        for paragraph in paragraphs:
            qas: list[dict] = paragraph.get("qas", [])
            for qa in qas:
                question: str = qa.get("question", "").strip()
                answers: list[dict] = qa.get("answers", [])

                # Skip if no answer was found (clause not present in this contract)
                if not answers or not answers[0].get("text", "").strip():
                    continue

                clause_text: str = answers[0]["text"].strip()
                clause_type: str = question

                doc_id = _make_doc_id(title, clause_type, len(docs_to_add))

                docs_to_add.append(
                    {
                        "id": doc_id,
                        "document": clause_text,
                        "metadata": {
                            "contract_title": title,
                            "clause_type": clause_type,
                            "source": "cuad_v1",
                            "char_length": len(clause_text),
                        },
                    }
                )

    if not docs_to_add:
        logger.warning("No clauses extracted from CUAD. Check dataset format.")
        return 0

    logger.info(
        "Extracted {} clause-answer pairs from {} contracts. Starting ChromaDB load...",
        len(docs_to_add),
        len(contracts_to_process),
    )

    # Batch insert into ChromaDB
    total_added = 0
    num_batches = -(-len(docs_to_add) // BATCH_SIZE)  # ceiling division
    for i in range(0, len(docs_to_add), BATCH_SIZE):
        batch = docs_to_add[i : i + BATCH_SIZE]
        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["document"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        total_added += len(batch)
        logger.info(
            "Ingested batch {}/{} — {} docs loaded so far",
            i // BATCH_SIZE + 1,
            num_batches,
            total_added,
        )

    logger.info(
        "CUAD ingestion complete. {} clauses from {} contracts loaded into ChromaDB.",
        total_added,
        len(contracts_to_process),
    )
    return total_added


if __name__ == "__main__":
    count = ingest_cuad()
    print(f"\nDone. {count} clauses ingested into ChromaDB.")
