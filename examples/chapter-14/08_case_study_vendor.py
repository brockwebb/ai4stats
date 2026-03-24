"""
08_case_study_vendor.py
Chapter 14: Evaluating AI Systems for Federal Use

Case B.1: Mock vendor one-pager (AutoCode Pro 2.0).
Rubric scoring walkthrough with commentary.
The vendor one-pager is a pedagogical device -- do not change it.
"""

# ── Configuration ─────────────────────────────────────────────────────────────
VENDOR_ONEPAGER = """
MOCK VENDOR ONE-PAGER: AutoCode Pro 2.0
========================================
AutoCode Pro 2.0 automatically classifies survey industry and occupation
responses to NAICS and SOC standards. Trusted by leading market research firms.

PERFORMANCE:
  - 94.2% accuracy on industry coding (NAICS 2-digit)
  - 87.6% accuracy on occupation coding (SOC major group)
  - Validated on 500,000 coded responses

FEATURES:
  - Cloud-based API (AWS deployment)
  - Batch processing: up to 1 million records per day
  - Returns code + confidence score
  - Dashboard for monitoring

IMPLEMENTATION:
  - 2-week deployment
  - REST API integration
  - 30-day money-back guarantee

PRICING:
  - $0.002 per record
  - Enterprise agreement available

Contact: sales@autocodeproexample.com
"""

# Rubric scoring: score, evidence cited (or absence), and commentary
RUBRIC_SCORES = [
    {
        "dim": 1, "name": "Task fit", "score": 2,
        "evidence": "Addresses a documented operational need (industry and occupation coding).",
        "gap": "No documented needs assessment. Problem defined by vendor, not agency.",
    },
    {
        "dim": 2, "name": "Accuracy", "score": 1,
        "evidence": "94.2% accuracy on industry coding; 87.6% on occupation coding.",
        "gap": (
            "Single number per task. No subgroup breakdown. No 6-digit NAICS accuracy. "
            "No baseline comparison to human coder agreement rates. "
            "Validation data controlled by vendor."
        ),
    },
    {
        "dim": 3, "name": "Reproducibility", "score": 0,
        "evidence": "Not mentioned.",
        "gap": (
            "Cloud API may be stochastic (temperature not specified). "
            "No version pinning mentioned. No output logging described."
        ),
    },
    {
        "dim": 4, "name": "Documentation", "score": 0,
        "evidence": "Not mentioned.",
        "gap": (
            "No model card. No data sheet. No failure analysis. "
            "No architectural description. 'Proprietary' would be expected here."
        ),
    },
    {
        "dim": 5, "name": "Failure modes", "score": 1,
        "evidence": "Returns code + confidence score (supports routing).",
        "gap": (
            "No documented error types. No routing or alert protocol described. "
            "Dashboard mentioned but failure management not specified."
        ),
    },
    {
        "dim": 6, "name": "Human oversight", "score": 0,
        "evidence": "Not mentioned.",
        "gap": (
            "No human review step described. No override capability documented. "
            "Automation bias not addressed."
        ),
    },
    {
        "dim": 7, "name": "Data governance", "score": 0,
        "evidence": "AWS deployment mentioned.",
        "gap": (
            "AWS commercial -- not FedRAMP-authorized. "
            "Title 13 data CANNOT be sent without FedRAMP authorization. "
            "No data use agreement. No retention policy. No legal review path."
        ),
    },
    {
        "dim": 8, "name": "Bias and fairness", "score": 0,
        "evidence": "Not mentioned.",
        "gap": (
            "No subgroup accuracy. No testing on minority-language responses. "
            "No disparate impact analysis."
        ),
    },
    {
        "dim": 9, "name": "Update and drift management", "score": 0,
        "evidence": "Not mentioned.",
        "gap": (
            "No update notification process. No revalidation protocol. "
            "No concept drift monitoring."
        ),
    },
    {
        "dim": 10, "name": "State Fidelity Validity (SFV)", "score": 0,
        "evidence": "Not applicable to a stateless coding API.",
        "gap": (
            "If used in a multi-session analytic pipeline, SFV controls "
            "would be required. Not assessed in the one-pager."
        ),
    },
]


def print_vendor_onepager():
    print(VENDOR_ONEPAGER)


def print_rubric_walkthrough():
    print("Rubric Scoring: AutoCode Pro 2.0")
    print("=" * 65)
    total = 0
    for item in RUBRIC_SCORES:
        total += item["score"]
        score_label = {0: "MISSING", 1: "MINIMUM", 2: "ADEQUATE", 3: "BEST PRACTICE"}
        print(f"\n  Dimension {item['dim']}: {item['name']}")
        print(f"  Score: {item['score']}/3 [{score_label[item['score']]}]")
        print(f"  Evidence: {item['evidence']}")
        print(f"  Gap:      {item['gap']}")

    max_score = 3 * len(RUBRIC_SCORES)
    print()
    print("=" * 65)
    print(f"TOTAL SCORE: {total}/{max_score} ({total / max_score:.0%})")
    print()
    print("RECOMMENDATION: Do NOT deploy.")
    print()
    print("Major gaps in documentation, governance, and oversight.")
    print("The single high score (Task fit = 2) reflects that coding is a")
    print("real need -- not that this product meets federal requirements.")
    print()
    print("Minimum conditions for re-evaluation:")
    print("  - Model card with subgroup accuracy (dimensions 2, 4)")
    print("  - FedRAMP-authorized deployment path (dimension 7)")
    print("  - Bias testing on minority-language responses (dimension 8)")
    print("  - Human oversight design with logged overrides (dimension 6)")
    print("  - Version pinning and reproducibility documentation (dimension 3)")


if __name__ == "__main__":
    print_vendor_onepager()
    print_rubric_walkthrough()
