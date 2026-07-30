"""
evaluation/evaluate_precision_recall.py
───────────────────────────────────────
Evaluates the Smart Contract Audit pipeline against CUAD ground-truth labels.

For each test contract in data/test_set/:
  1. Feeds the raw text through the full LangGraph pipeline.
  2. Compares the system's analyzed_risks against CUAD ground-truth clauses.
  3. Calculates precision, recall, and F1 score.

Matching strategy:
  - Ground-truth clause types are normalized via clause_type_map.py.
  - A system finding is a True Positive if its clause_type matches any
    ground-truth clause type present in the contract.
  - Comparison is at the clause TYPE level (not exact text overlap),
    because clause extraction boundaries will naturally differ.

Usage:
    uv run python -m evaluation.evaluate_precision_recall
    uv run python -m evaluation.evaluate_precision_recall --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from loguru import logger

from api.agents.graph import compiled_graph
from evaluation.clause_type_map import GUARDED_CLAUSE_TYPES, normalize_clause_type

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

TEST_SET_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data", "test_set")
RESULTS_DIR: str = os.path.join(os.path.dirname(__file__), "results")


# ─────────────────────────────────────────────
# Data classes for metrics
# ─────────────────────────────────────────────


@dataclass
class ContractResult:
    """Evaluation result for a single contract."""

    title: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    system_risk_count: int = 0
    ground_truth_count: int = 0
    overall_risk_score: float = 0.0
    tp_details: list[str] = field(default_factory=list)
    fp_details: list[str] = field(default_factory=list)
    fn_details: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class AggregateResult:
    """Aggregate evaluation result across all test contracts."""

    total_contracts: int = 0
    total_true_positives: int = 0
    total_false_positives: int = 0
    total_false_negatives: int = 0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    contracts_with_errors: int = 0
    per_contract: list[dict] = field(default_factory=list)
    per_clause_type: dict[str, dict] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Core evaluation logic
# ─────────────────────────────────────────────


def _run_pipeline_for_evaluation(raw_text: str, document_name: str) -> dict:
    """
    Runs the full LangGraph pipeline with the given raw text.

    Returns the final state dict from the pipeline.
    """
    initial_state: dict = {
        "document_name": document_name,
        "file_bytes": None,
        "file_path": None,
        "page_count": 0,
        "raw_text": raw_text,
        "clauses": [],
        "analyzed_risks": [],
        "validation_passed": False,
        "retry_count": 0,
        "judge_feedback": "",
        "final_report": None,
    }

    return compiled_graph.invoke(initial_state)


def _extract_system_clause_types(analyzed_risks: list[dict]) -> set[str]:
    """
    Extracts the set of clause types the system identified as risky
    (medium or high risk level).

    The matched_rule field has format "llm_analysis::termination" or
    "keyword_match::liability" or "guardrail::rule_id".
    """
    types: set[str] = set()
    for risk in analyzed_risks:
        risk_level = risk.get("risk_level", "low")
        if risk_level in ("medium", "high"):
            matched_rule = risk.get("matched_rule", "")
            if "::" in matched_rule:
                clause_type = matched_rule.split("::", 1)[1]
                types.add(clause_type.lower())
    return types


def _extract_ground_truth_types(ground_truth_clauses: list[dict]) -> set[str]:
    """
    Extracts the set of normalized clause types from CUAD ground truth.

    Only includes types that map to our system's guarded clause types,
    since those are the ones our system is designed to detect.
    """
    types: set[str] = set()
    for clause in ground_truth_clauses:
        cuad_type = clause.get("clause_type", "")
        system_type = normalize_clause_type(cuad_type)
        # Only count clause types that our system actually checks for
        if system_type in GUARDED_CLAUSE_TYPES:
            types.add(system_type)
    return types


def evaluate_contract(test_data: dict) -> ContractResult:
    """
    Evaluates a single contract against its ground truth.

    Args:
        test_data: Dict loaded from a test_set JSON file, containing
                   'title', 'raw_text', 'ground_truth_clauses'.

    Returns:
        ContractResult with precision/recall/F1 and details.
    """
    title = test_data["title"]
    result = ContractResult(title=title)

    # Skip contracts with no raw text
    if not test_data.get("raw_text", "").strip():
        result.error = "Empty raw text"
        logger.warning("[Eval] Skipping '{}': empty raw text.", title)
        return result

    # Run the pipeline
    try:
        logger.info("[Eval] Running pipeline for: {}", title)
        final_state = _run_pipeline_for_evaluation(
            raw_text=test_data["raw_text"],
            document_name=title,
        )
    except Exception as e:
        result.error = str(e)
        logger.error("[Eval] Pipeline failed for '{}': {}", title, e)
        return result

    # Extract what the system found
    analyzed_risks: list[dict] = final_state.get("analyzed_risks", [])
    system_types = _extract_system_clause_types(analyzed_risks)

    # Extract ground truth (normalized to system clause types)
    ground_truth_types = _extract_ground_truth_types(test_data.get("ground_truth_clauses", []))

    # Calculate TP / FP / FN at the clause TYPE level
    true_positives = system_types & ground_truth_types
    false_positives = system_types - ground_truth_types
    false_negatives = ground_truth_types - system_types

    result.true_positives = len(true_positives)
    result.false_positives = len(false_positives)
    result.false_negatives = len(false_negatives)
    result.system_risk_count = len(system_types)
    result.ground_truth_count = len(ground_truth_types)
    result.tp_details = sorted(true_positives)
    result.fp_details = sorted(false_positives)
    result.fn_details = sorted(false_negatives)

    # Get overall risk score from the report
    final_report = final_state.get("final_report", {})
    if final_report:
        result.overall_risk_score = final_report.get("overall_risk_score", 0.0)

    # Precision, recall, F1
    if result.true_positives + result.false_positives > 0:
        result.precision = result.true_positives / (result.true_positives + result.false_positives)

    if result.true_positives + result.false_negatives > 0:
        result.recall = result.true_positives / (result.true_positives + result.false_negatives)

    if result.precision + result.recall > 0:
        result.f1 = (2 * result.precision * result.recall) / (result.precision + result.recall)

    logger.info(
        "[Eval] {} → P={:.2f} R={:.2f} F1={:.2f} (TP={} FP={} FN={})",
        title,
        result.precision,
        result.recall,
        result.f1,
        result.true_positives,
        result.false_positives,
        result.false_negatives,
    )

    return result


def run_evaluation(limit: int | None = None) -> AggregateResult:
    """
    Runs the full evaluation across all test contracts.

    Args:
        limit: If set, only evaluate this many contracts (for quick testing).

    Returns:
        AggregateResult with per-contract and aggregate metrics.
    """
    manifest_path = os.path.join(TEST_SET_DIR, "manifest.json")

    if not os.path.exists(manifest_path):
        logger.error(
            "Test set manifest not found at {}. Run 'uv run python -m data.prepare_test_set' first.",
            manifest_path,
        )
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    contracts = manifest.get("contracts", [])
    if limit:
        contracts = contracts[:limit]

    logger.info(
        "Starting evaluation of {} test contract(s)...",
        len(contracts),
    )

    aggregate = AggregateResult(total_contracts=len(contracts))
    per_type_tp: dict[str, int] = {}
    per_type_fp: dict[str, int] = {}
    per_type_fn: dict[str, int] = {}
    precision_sum = 0.0
    recall_sum = 0.0
    valid_count = 0

    for i, entry in enumerate(contracts):
        filename = entry["filename"]
        contract_path = os.path.join(TEST_SET_DIR, filename)

        sep = "=" * 60
        logger.info("\n{}\n[{}/{}] Evaluating: {}\n{}", sep, i + 1, len(contracts), entry["title"], sep)

        with open(contract_path, encoding="utf-8") as f:
            test_data = json.load(f)

        result = evaluate_contract(test_data)
        aggregate.per_contract.append(asdict(result))

        if result.error:
            aggregate.contracts_with_errors += 1
            continue

        aggregate.total_true_positives += result.true_positives
        aggregate.total_false_positives += result.false_positives
        aggregate.total_false_negatives += result.false_negatives
        precision_sum += result.precision
        recall_sum += result.recall
        valid_count += 1

        # Track per-clause-type metrics
        for ct in result.tp_details:
            per_type_tp[ct] = per_type_tp.get(ct, 0) + 1
        for ct in result.fp_details:
            per_type_fp[ct] = per_type_fp.get(ct, 0) + 1
        for ct in result.fn_details:
            per_type_fn[ct] = per_type_fn.get(ct, 0) + 1

    # ── Macro-average (average of per-contract metrics) ──
    if valid_count > 0:
        aggregate.macro_precision = precision_sum / valid_count
        aggregate.macro_recall = recall_sum / valid_count
        if aggregate.macro_precision + aggregate.macro_recall > 0:
            aggregate.macro_f1 = (2 * aggregate.macro_precision * aggregate.macro_recall) / (
                aggregate.macro_precision + aggregate.macro_recall
            )

    # ── Micro-average (aggregate TP/FP/FN then calculate) ──
    tp_total = aggregate.total_true_positives
    fp_total = aggregate.total_false_positives
    fn_total = aggregate.total_false_negatives

    if tp_total + fp_total > 0:
        aggregate.micro_precision = tp_total / (tp_total + fp_total)
    if tp_total + fn_total > 0:
        aggregate.micro_recall = tp_total / (tp_total + fn_total)
    if aggregate.micro_precision + aggregate.micro_recall > 0:
        aggregate.micro_f1 = (2 * aggregate.micro_precision * aggregate.micro_recall) / (
            aggregate.micro_precision + aggregate.micro_recall
        )

    # ── Per-clause-type breakdown ──
    all_types = set(per_type_tp) | set(per_type_fp) | set(per_type_fn)
    for ct in sorted(all_types):
        tp = per_type_tp.get(ct, 0)
        fp = per_type_fp.get(ct, 0)
        fn = per_type_fn.get(ct, 0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        aggregate.per_clause_type[ct] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }

    return aggregate


# ─────────────────────────────────────────────
# Output generation
# ─────────────────────────────────────────────


def _write_json_report(aggregate: AggregateResult) -> str:
    """Writes the full evaluation results as JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "evaluation_report.json")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_contracts": aggregate.total_contracts,
        "contracts_with_errors": aggregate.contracts_with_errors,
        "aggregate_metrics": {
            "micro": {
                "precision": round(aggregate.micro_precision, 4),
                "recall": round(aggregate.micro_recall, 4),
                "f1": round(aggregate.micro_f1, 4),
            },
            "macro": {
                "precision": round(aggregate.macro_precision, 4),
                "recall": round(aggregate.macro_recall, 4),
                "f1": round(aggregate.macro_f1, 4),
            },
            "totals": {
                "true_positives": aggregate.total_true_positives,
                "false_positives": aggregate.total_false_positives,
                "false_negatives": aggregate.total_false_negatives,
            },
        },
        "per_clause_type": aggregate.per_clause_type,
        "per_contract": aggregate.per_contract,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return path


def _write_summary_markdown(aggregate: AggregateResult) -> str:
    """Writes a human-readable markdown summary."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "summary.md")

    lines: list[str] = [
        "# Evaluation Results — Smart Contract Audit System",
        "",
        f"**Generated at:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Test contracts:** {aggregate.total_contracts}",
        f"**Contracts with errors:** {aggregate.contracts_with_errors}",
        "",
        "---",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Micro | Macro |",
        "|--------|-------|-------|",
        f"| Precision | {aggregate.micro_precision:.4f} | {aggregate.macro_precision:.4f} |",
        f"| Recall | {aggregate.micro_recall:.4f} | {aggregate.macro_recall:.4f} |",
        f"| F1 Score | {aggregate.micro_f1:.4f} | {aggregate.macro_f1:.4f} |",
        "",
        f"**Totals:** TP={aggregate.total_true_positives}, "
        f"FP={aggregate.total_false_positives}, "
        f"FN={aggregate.total_false_negatives}",
        "",
        "---",
        "",
        "## Per Clause Type Breakdown",
        "",
        "| Clause Type | TP | FP | FN | Precision | Recall | F1 |",
        "|-------------|----|----|-----|-----------|--------|-----|",
    ]

    for ct, metrics in sorted(aggregate.per_clause_type.items()):
        lines.append(
            f"| {ct} | {metrics['true_positives']} | {metrics['false_positives']} "
            f"| {metrics['false_negatives']} | {metrics['precision']:.4f} "
            f"| {metrics['recall']:.4f} | {metrics['f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Per Contract Results",
            "",
            "| # | Contract | P | R | F1 | TP | FP | FN | Risk Score |",
            "|---|----------|---|---|----|----|----|----|------------|",
        ]
    )

    for i, contract in enumerate(aggregate.per_contract):
        error = contract.get("error")
        if error:
            lines.append(f"| {i + 1} | {contract['title'][:40]} | — | — | — | — | — | — | ERROR: {error[:30]} |")
        else:
            lines.append(
                f"| {i + 1} | {contract['title'][:40]} "
                f"| {contract['precision']:.2f} | {contract['recall']:.2f} "
                f"| {contract['f1']:.2f} | {contract['true_positives']} "
                f"| {contract['false_positives']} | {contract['false_negatives']} "
                f"| {contract['overall_risk_score']:.4f} |"
            )

    lines.extend(["", "---", ""])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path


def _print_console_summary(aggregate: AggregateResult) -> None:
    """Prints a compact summary table to the console."""
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS — Smart Contract Audit System")
    print("=" * 70)
    print(f"\n  Contracts evaluated: {aggregate.total_contracts}")
    print(f"  Contracts with errors: {aggregate.contracts_with_errors}")
    print(f"\n  {'Metric':<12} {'Micro':>10} {'Macro':>10}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 10}")
    print(f"  {'Precision':<12} {aggregate.micro_precision:>10.4f} {aggregate.macro_precision:>10.4f}")
    print(f"  {'Recall':<12} {aggregate.micro_recall:>10.4f} {aggregate.macro_recall:>10.4f}")
    print(f"  {'F1 Score':<12} {aggregate.micro_f1:>10.4f} {aggregate.macro_f1:>10.4f}")
    print(f"\n  True Positives:  {aggregate.total_true_positives}")
    print(f"  False Positives: {aggregate.total_false_positives}")
    print(f"  False Negatives: {aggregate.total_false_negatives}")
    print("\n" + "=" * 70)


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────


def main() -> None:
    """CLI entry point for running the evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Smart Contract Audit pipeline against CUAD ground truth."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate this many contracts (for quick testing).",
    )
    args = parser.parse_args()

    aggregate = run_evaluation(limit=args.limit)

    # Write outputs
    json_path = _write_json_report(aggregate)
    md_path = _write_summary_markdown(aggregate)
    _print_console_summary(aggregate)

    print(f"\n  Full report:  {json_path}")
    print(f"  Summary:      {md_path}")


if __name__ == "__main__":
    main()
