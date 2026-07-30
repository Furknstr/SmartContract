"""
data/prepare_test_set.py
────────────────────────
Extracts a labeled test set from the CUAD v1 JSON for evaluation.

Uses contracts at indices 40–54 (15 contracts) to avoid overlap with
the first 40 contracts used for ChromaDB ingestion in rag/ingestion.py.

For each contract, extracts:
  - The full paragraph context (raw text of the contract section)
  - All QA pairs that have answers → these are the ground-truth
    "notable/risky" clauses with their CUAD clause type labels.

Output:
  data/test_set/manifest.json         — index of all test contracts
  data/test_set/{sanitized_title}.json — per-contract ground truth

Usage:
    uv run python -m data.prepare_test_set
"""

from __future__ import annotations

import json
import os
import re

from loguru import logger

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Contracts 0–39 are used for ChromaDB reference set (rag/ingestion.py)
TEST_START_INDEX: int = 40
TEST_END_INDEX: int = 55  # exclusive → 15 contracts
TEST_SET_DIR: str = os.path.join(os.path.dirname(__file__), "test_set")

# Path to the cached CUAD JSON (downloaded by rag/ingestion.py)
CUAD_CACHE_PATH: str = os.path.join(os.path.dirname(__file__), "..", "rag", ".cuad_cache.json")

# Also check direct download if cache is not in rag/
CUAD_JSON_URL = "https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json"


def _sanitize_filename(name: str) -> str:
    """Convert a contract title into a safe filename."""
    # Remove/replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = sanitized.strip("_.")
    return sanitized[:120]  # Truncate overly long names


def _load_cuad_json() -> dict:
    """
    Loads the CUAD v1 JSON from the local cache.

    The cache is expected to exist at rag/.cuad_cache.json after running
    the ingestion pipeline (uv run python -m rag.ingestion).

    Returns:
        Parsed CUAD JSON dict in SQuAD format.

    Raises:
        FileNotFoundError: If the cache doesn't exist.
    """
    cache_path = os.path.normpath(CUAD_CACHE_PATH)

    if os.path.exists(cache_path):
        logger.info("Loading CUAD JSON from cache: {}", cache_path)
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    logger.error(
        "CUAD cache not found at {}. Run 'uv run python -m rag.ingestion' first to download the dataset.",
        cache_path,
    )
    raise FileNotFoundError(f"CUAD cache not found at {cache_path}. Run 'uv run python -m rag.ingestion' first.")


def _extract_ground_truth(contract: dict) -> dict:
    """
    Extracts ground-truth annotations from a single CUAD contract.

    Args:
        contract: A single entry from cuad_data["data"], containing
                  "title" and "paragraphs" with QA pairs.

    Returns:
        Dict with:
          - title: contract title
          - raw_text: concatenated paragraph contexts
          - ground_truth_clauses: list of {clause_type, clause_text, answer_start}
          - clause_type_counts: dict of clause_type → count
    """
    title: str = contract.get("title", "unknown")
    paragraphs: list[dict] = contract.get("paragraphs", [])

    raw_text_parts: list[str] = []
    ground_truth_clauses: list[dict] = []
    clause_type_counts: dict[str, int] = {}

    for paragraph in paragraphs:
        context: str = paragraph.get("context", "")
        if context:
            raw_text_parts.append(context)

        qas: list[dict] = paragraph.get("qas", [])
        for qa in qas:
            question: str = qa.get("question", "").strip()
            answers: list[dict] = qa.get("answers", [])
            is_impossible: bool = qa.get("is_impossible", False)

            # Skip if no answer (clause not present in this contract)
            if is_impossible or not answers:
                continue

            # Only take answers that have actual text
            valid_answers = [a for a in answers if a.get("text", "").strip()]
            if not valid_answers:
                continue

            # Use the first answer as the canonical one
            answer = valid_answers[0]
            clause_text: str = answer["text"].strip()
            answer_start: int = answer.get("answer_start", -1)

            ground_truth_clauses.append(
                {
                    "clause_type": question,
                    "clause_text": clause_text,
                    "answer_start": answer_start,
                    "char_length": len(clause_text),
                }
            )

            clause_type_counts[question] = clause_type_counts.get(question, 0) + 1

    return {
        "title": title,
        "raw_text": "\n\n".join(raw_text_parts),
        "ground_truth_clauses": ground_truth_clauses,
        "clause_type_counts": clause_type_counts,
        "total_ground_truth": len(ground_truth_clauses),
    }


def prepare_test_set() -> int:
    """
    Main function: extracts labeled test contracts from CUAD and writes them
    to data/test_set/.

    Returns:
        Number of test contracts prepared.
    """
    cuad_data = _load_cuad_json()
    all_contracts: list[dict] = cuad_data.get("data", [])

    if len(all_contracts) < TEST_END_INDEX:
        logger.warning(
            "CUAD has only {} contracts, but TEST_END_INDEX is {}. Using all available contracts after index {}.",
            len(all_contracts),
            TEST_END_INDEX,
            TEST_START_INDEX,
        )

    test_contracts = all_contracts[TEST_START_INDEX:TEST_END_INDEX]
    logger.info(
        "Selected {} test contracts (indices {}–{}).",
        len(test_contracts),
        TEST_START_INDEX,
        TEST_START_INDEX + len(test_contracts) - 1,
    )

    # Create output directory
    os.makedirs(TEST_SET_DIR, exist_ok=True)

    manifest_entries: list[dict] = []
    total_clauses = 0

    for i, contract in enumerate(test_contracts):
        ground_truth = _extract_ground_truth(contract)
        title = ground_truth["title"]
        safe_name = _sanitize_filename(title)
        filename = f"{safe_name}.json"

        output_path = os.path.join(TEST_SET_DIR, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2, ensure_ascii=False)

        total_clauses += ground_truth["total_ground_truth"]

        manifest_entries.append(
            {
                "index": TEST_START_INDEX + i,
                "title": title,
                "filename": filename,
                "total_ground_truth_clauses": ground_truth["total_ground_truth"],
                "clause_types_found": list(ground_truth["clause_type_counts"].keys()),
                "raw_text_length": len(ground_truth["raw_text"]),
            }
        )

        logger.info(
            "  [{}/{}] {} — {} ground-truth clauses",
            i + 1,
            len(test_contracts),
            title,
            ground_truth["total_ground_truth"],
        )

    # Write manifest
    manifest = {
        "description": "CUAD v1 test set for evaluation",
        "source_indices": f"{TEST_START_INDEX}–{TEST_START_INDEX + len(test_contracts) - 1}",
        "total_contracts": len(test_contracts),
        "total_ground_truth_clauses": total_clauses,
        "contracts": manifest_entries,
    }
    manifest_path = os.path.join(TEST_SET_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(
        "\nTest set prepared: {} contracts, {} total ground-truth clauses.",
        len(test_contracts),
        total_clauses,
    )
    logger.info("Output directory: {}", TEST_SET_DIR)

    return len(test_contracts)


if __name__ == "__main__":
    count = prepare_test_set()
    print(f"\nDone. {count} test contracts prepared in data/test_set/.")
