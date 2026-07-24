"""
rag/ingestion.py
────────────────
CUAD dataset ingestion pipeline.

Downloads the CUAD (Contract Understanding Atticus Dataset), extracts
clause-level text with their labels, and loads them into ChromaDB
as the "standard/reference" clause set for RAG comparison.

Usage:
    uv run python -m rag.ingestion
"""

from __future__ import annotations

import hashlib
import textwrap

from datasets import load_dataset
from loguru import logger

from rag.vectorstore import get_chroma_client, get_or_create_collection

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# How many contracts to load into the reference set
MAX_CONTRACTS: int = 40
# ChromaDB has a batch size limit
BATCH_SIZE: int = 50

# The 41 CUAD clause categories we care about
CUAD_CLAUSE_TYPES: list[str] = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
]


def _make_doc_id(contract_title: str, clause_type: str, idx: int) -> str:
    """Generate a deterministic document ID for deduplication."""
    raw = f"{contract_title}::{clause_type}::{idx}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest_cuad() -> int:
    """
    Downloads CUAD, extracts answered clauses, and loads them into ChromaDB.

    Returns:
        Number of clauses successfully ingested.
    """
    logger.info("Starting CUAD ingestion (max {} contracts)...", MAX_CONTRACTS)

    # Load the CUAD QA dataset
    dataset = load_dataset("theatticusproject/cuad-qa", split="train")
    logger.info("CUAD dataset loaded — {} total QA rows", len(dataset))

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    existing_count = collection.count()
    if existing_count > 0:
        logger.warning(
            "Collection already has {} documents. Skipping ingestion to avoid duplicates. "
            "Delete the collection first if you want to re-ingest.",
            existing_count,
        )
        return existing_count

    # Group by contract title (first MAX_CONTRACTS unique titles)
    seen_titles: set[str] = set()
    docs_to_add: list[dict] = []

    for row in dataset:
        title: str = row.get("title", "unknown")

        if title in seen_titles and len(seen_titles) >= MAX_CONTRACTS:
            continue
        seen_titles.add(title)

        if len(seen_titles) > MAX_CONTRACTS:
            break

        # Each row has a "question" (clause type) and "answers" (extracted text)
        question: str = row.get("question", "")
        answers: dict = row.get("answers", {})
        answer_texts: list[str] = answers.get("text", [])

        # Skip if no answer was found (clause not present in this contract)
        if not answer_texts or not answer_texts[0].strip():
            continue

        clause_text = answer_texts[0].strip()

        # Determine clause type from the question
        clause_type = question.strip()

        doc_id = _make_doc_id(title, clause_type, 0)

        docs_to_add.append(
            {
                "id": doc_id,
                "document": clause_text,
                "metadata": {
                    "contract_title": title,
                    "clause_type": clause_type,
                    "source": "cuad",
                    "char_length": len(clause_text),
                },
            }
        )

    if not docs_to_add:
        logger.warning("No clauses extracted from CUAD. Check dataset format.")
        return 0

    # Batch insert into ChromaDB
    total_added = 0
    for i in range(0, len(docs_to_add), BATCH_SIZE):
        batch = docs_to_add[i : i + BATCH_SIZE]
        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["document"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        total_added += len(batch)
        logger.info("Ingested batch {}/{} ({} docs)", i // BATCH_SIZE + 1, -(-len(docs_to_add) // BATCH_SIZE), total_added)

    logger.info(
        "CUAD ingestion complete. {} clauses from {} contracts loaded into ChromaDB.",
        total_added,
        len(seen_titles),
    )
    return total_added


if __name__ == "__main__":
    count = ingest_cuad()
    print(f"\nDone. {count} clauses ingested into ChromaDB.")
