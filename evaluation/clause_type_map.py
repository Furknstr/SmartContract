"""
evaluation/clause_type_map.py
─────────────────────────────
Mapping from CUAD's 41 fine-grained clause type labels to the
simplified clause types used by the Smart Contract Audit system.

CUAD labels come from the "question" field in each QA pair.
Our system uses: termination, liability, confidentiality, penalty,
governing_law, indemnification, ip_ownership, non_compete, renewal, other.
"""

from __future__ import annotations

# ─────────────────────────────────────────────
# CUAD question → system clause type
# ─────────────────────────────────────────────

CUAD_TO_SYSTEM_TYPE: dict[str, str] = {
    # ── Termination ──────────────────────────
    "Termination For Convenience": "termination",
    "Rofr/Rofo/Rofn": "termination",
    "Change Of Control": "termination",
    "Anti-Assignment": "termination",

    # ── Liability ────────────────────────────
    "Cap On Liability": "liability",
    "Limitation Of Liability": "liability",
    "Warranty Duration": "liability",

    # ── Confidentiality ──────────────────────
    "Post-Termination Services": "confidentiality",
    "Non-Disclosure Agreement": "confidentiality",

    # ── Non-Compete ──────────────────────────
    "Non-Compete": "non_compete",
    "Exclusivity": "non_compete",
    "No-Solicit Of Customers": "non_compete",
    "No-Solicit Of Employees": "non_compete",
    "Covenant Not To Sue": "non_compete",

    # ── Indemnification ──────────────────────
    "Uncapped Liability": "indemnification",
    "Insurance": "indemnification",

    # ── IP Ownership ─────────────────────────
    "Ip Ownership Assignment": "ip_ownership",
    "Joint Ip Ownership": "ip_ownership",
    "License Grant": "ip_ownership",
    "Non-Transferable License": "ip_ownership",

    # ── Governing Law ────────────────────────
    "Governing Law": "governing_law",
    "Jurisdiction": "governing_law",
    "Audit Rights": "governing_law",
    "Most Favored Nation": "governing_law",

    # ── Renewal ──────────────────────────────
    "Renewal Term": "renewal",
    "Auto-Renewal": "renewal",
    "Expiration Date": "renewal",
    "Effective Date": "renewal",
    "Minimum Commitment": "renewal",
    "Volume Restriction": "renewal",

    # ── Penalty ──────────────────────────────
    "Liquidated Damages": "penalty",
    "Price Restrictions": "penalty",

    # ── Other ────────────────────────────────
    "Competitive Restriction Exception": "other",
    "Third Party Beneficiary": "other",
    "Affiliate License-Loss Of IP": "other",
    "Source Code Escrow": "other",
    "Revenue/Profit Sharing": "other",
    "Irrevocable Or Perpetual License": "other",
    "Unlimited/All-You-Can-Eat-License": "other",
}


def normalize_clause_type(cuad_question: str) -> str:
    """
    Maps a CUAD question label to the system's simplified clause type.

    Falls back to 'other' for unknown labels.

    Args:
        cuad_question: The raw question string from the CUAD dataset.

    Returns:
        One of: termination, liability, confidentiality, penalty,
        governing_law, indemnification, ip_ownership, non_compete,
        renewal, other.
    """
    return CUAD_TO_SYSTEM_TYPE.get(cuad_question.strip(), "other")


# System clause types that have guardrail rules (from guardrails/rules.yaml)
GUARDED_CLAUSE_TYPES: set[str] = {
    "termination",
    "penalty",
    "liability",
    "confidentiality",
    "governing_law",
    "renewal",
}
