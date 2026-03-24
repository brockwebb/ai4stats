"""
04_evaluation_rubric.py
Chapter 14: Evaluating AI Systems for Federal Use

10-dimension evaluation rubric for federal AI systems.
Defines each dimension with its core question, minimum requirement,
best practice, and red flags. Prints formatted display.

Dimension 10 (State Fidelity Validity) is covered in full in Chapter 15.
"""

# ── Configuration ─────────────────────────────────────────────────────────────
SCORE_LABELS = {0: "Missing", 1: "Minimum", 2: "Adequate", 3: "Best practice"}

RUBRIC = {
    1: {
        "dimension": "Task fit",
        "question": "Does the AI system address a real, documented operational need?",
        "minimum": "Documented use case with specific operational requirements.",
        "best_practice": (
            "Needs assessment conducted with end users; "
            "alternative solutions evaluated."
        ),
        "red_flags": [
            "Vendor-defined problem statement",
            "No documented baseline performance",
            "Solving a problem the agency does not have",
        ],
    },
    2: {
        "dimension": "Accuracy",
        "question": (
            "Measured against what baseline, on whose data, "
            "at what classification level?"
        ),
        "minimum": "Accuracy reported on independently validated holdout set.",
        "best_practice": (
            "Accuracy by subgroup, task difficulty, and classification granularity; "
            "compared to human coder agreement rates."
        ),
        "red_flags": [
            "Single accuracy number without subgroup breakdown",
            "Vendor-controlled validation data",
            "Accuracy reported at coarser granularity than operational use",
        ],
    },
    3: {
        "dimension": "Reproducibility",
        "question": (
            "Does the same input produce the same output "
            "across calls, versions, and time?"
        ),
        "minimum": "Deterministic output (temperature=0 or equivalent).",
        "best_practice": (
            "Version-pinned model, logged outputs, "
            "documented update validation protocol."
        ),
        "red_flags": [
            "Stochastic outputs without majority voting",
            "Model updates without revalidation",
            "No output logging",
        ],
    },
    4: {
        "dimension": "Documentation",
        "question": (
            "Can an external reviewer understand what the system does "
            "and how it was built?"
        ),
        "minimum": (
            "Model card covering training data, performance, "
            "intended use, and limitations."
        ),
        "best_practice": (
            "Data sheet, failure analysis, bias audit, "
            "architectural decision record."
        ),
        "red_flags": [
            "'Proprietary' used to deflect documentation requests",
            "No model card",
            "No failure analysis",
        ],
    },
    5: {
        "dimension": "Failure modes",
        "question": "What happens when the system is wrong? Who notices? How quickly?",
        "minimum": "Documented error types; monitoring for accuracy drift.",
        "best_practice": (
            "Confidence-based routing to human review; "
            "alert system for systematic degradation."
        ),
        "red_flags": [
            "No error analysis",
            "Silent failures",
            "No monitoring plan",
        ],
    },
    6: {
        "dimension": "Human oversight",
        "question": "Where are the human decision points? Can they be bypassed?",
        "minimum": "Clear identification of where humans are in the loop.",
        "best_practice": (
            "Bounded agency design: AI assists, human decides; "
            "override is easy and logged."
        ),
        "red_flags": [
            "Fully automated pipeline with no human review step",
            "Override capability not documented",
            "Automation bias not addressed in training",
        ],
    },
    7: {
        "dimension": "Data governance",
        "question": "What data does the system ingest, retain, and share?",
        "minimum": "Data use agreement; documented retention and sharing policies.",
        "best_practice": (
            "FedRAMP authorization; Title 13/CIPSEA compliance review; "
            "no retention of survey responses."
        ),
        "red_flags": [
            "Survey responses retained by vendor",
            "No FedRAMP authorization for cloud deployment",
            "Data sharing terms not reviewed by legal counsel",
        ],
    },
    8: {
        "dimension": "Bias and fairness",
        "question": "Has performance been tested across relevant subpopulations?",
        "minimum": "Accuracy by major demographic and linguistic subgroup.",
        "best_practice": (
            "Disparate impact analysis; testing on minority-language responses; "
            "bias mitigation documented."
        ),
        "red_flags": [
            "Only aggregate accuracy reported",
            "No testing on minority-language or ambiguous responses",
            "No disparate impact analysis",
        ],
    },
    9: {
        "dimension": "Update and drift management",
        "question": "How does the system change over time? Who validates updates?",
        "minimum": "Update notification with changelog; revalidation before deployment.",
        "best_practice": (
            "Drift monitoring; periodic revalidation on agency data; "
            "version control for model and prompts."
        ),
        "red_flags": [
            "Model updates without notice",
            "No revalidation protocol",
            "No concept drift monitoring",
        ],
    },
    10: {
        "dimension": "State Fidelity Validity (SFV)",
        "question": (
            "For stateful or agentic systems: does accumulated pipeline state "
            "faithfully represent its actual operational history across sessions?"
        ),
        "minimum": (
            "Session boundaries explicitly managed; "
            "handoff documents for multi-session work."
        ),
        "best_practice": (
            "Config-driven vocabulary; graph-backed decision log; "
            "periodic state reconciliation against canonical record."
        ),
        "red_flags": [
            "No session management for multi-session AI pipelines",
            "Terminology inconsistency across sessions not detected",
            "Accumulated pipeline decisions not logged or auditable",
        ],
        "note": "See Chapter 15 for the full State Fidelity Validity framework.",
    },
}


def print_rubric():
    print("AI System Evaluation Rubric: 10 Dimensions")
    print("=" * 65)
    for dim_num, info in RUBRIC.items():
        print(f"\n{dim_num}. {info['dimension'].upper()}")
        print(f"   Question:      {info['question']}")
        print(f"   Minimum:       {info['minimum']}")
        print(f"   Best practice: {info['best_practice']}")
        flags_str = "; ".join(info["red_flags"])
        print(f"   Red flags:     {flags_str}")
        if "note" in info:
            print(f"   Note:          {info['note']}")

    print()
    print("Scoring scale: 0 = Missing, 1 = Minimum, 2 = Adequate, 3 = Best practice")
    print("Maximum score: 30 points")
    print()
    print("Scoring guidance:")
    print("  0-10 (0-33%):  Not deployable. Major gaps in governance and safety.")
    print("  11-18 (37-60%): Conditional pilot only. Gaps must be addressed first.")
    print("  19-24 (63-80%): Deployable with documented mitigations.")
    print("  25-30 (83-100%): Ready for federal statistical production.")


if __name__ == "__main__":
    print_rubric()
