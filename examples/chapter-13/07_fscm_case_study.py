"""
Chapter 13: Agentic AI for Federal Statistical Operations
Example 07: Federal Survey Concept Mapper (FSCM) case study

ILLUSTRATIVE parameters based on realistic internal experiments.
Specific figures are representative rather than from a published evaluation.
No API calls. Pedagogical display only.
"""

import pandas as pd


# --- Problem description ---
def display_problem():
    print("Federal Survey Concept Mapper: case study (illustrative)")
    print("=" * 65)
    print()
    print("The problem:")
    print("  The Census Bureau operates 46 surveys.")
    print("  ~7,000 questions across those surveys overlap, duplicate,")
    print("  or relate to each other in undocumented ways.")
    print("  Nobody had a comprehensive map of how these questions")
    print("  relate to the Bureau's official concept taxonomy.")
    print()
    print("Why not manual?")
    print("  Hundreds of hours of analyst time. Human coders disagree")
    print("  on edge cases. No audit trail for individual decisions.")
    print()
    print("Why not a fully autonomous agent?")
    print("  Fast, impressive-looking -- and wrong in unpredictable ways.")
    print("  No confidence quantification. No human review for hard cases.")
    print("  Not defensible to subject matter experts or the DRB.")
    print()


# --- Illustrative results ---
def display_results():
    print("Illustrative results (representative parameters):")
    print("-" * 65)

    fscm_results = {
        "Questions processed": "~7,000",
        "Surveys covered": "46",
        "Categorization success rate": "~99% (illustrative)",
        "Model agreement rate (topic level)": "~89% (illustrative)",
        "Cohen's Kappa": "~0.84 (almost perfect; illustrative)",
        "Dual-modal questions (spans two concepts)": "~2-3%",
        "Flagged for human review": "<1%",
        "Total API cost": "~$15 (illustrative order of magnitude)",
        "Total runtime": "~2 hours (illustrative)",
    }

    for metric, value in fscm_results.items():
        print(f"  {metric}: {value}")

    print()
    print("What bounded agency replaced: weeks of manual work and thousands")
    print("of dollars in coder time, with a complete audit trail the manual")
    print("process could not have provided.")
    print()
    print("NOTE: Figures above are illustrative. They represent realistic")
    print("parameters for this class of pipeline, not a published evaluation.")
    print()


# --- Architecture description ---
def display_architecture():
    architecture = {
        "Component": [
            "Question input",
            "Classifier A",
            "Classifier B",
            "Agreement check",
            "Auto-assign path",
            "Dual-modal flag path",
            "Arbitrator (bounded)",
            "Human review queue",
        ],
        "Role": [
            "Survey question text received as input",
            "First LLM call: assigns concept code with confidence",
            "Second LLM call: independent assignment and confidence",
            "If A == B and both >= 0.90, route to auto-assign",
            "High-confidence agreement: code assigned automatically",
            "Both high-confidence but different codes: flag as dual-modal",
            "Disagreement: a third, constrained call resolves ambiguity",
            "Cases too ambiguous for arbitrator: queue for expert review",
        ],
        "Autonomy level": [
            "None (input)",
            "Bounded (classifier only)",
            "Bounded (classifier only)",
            "Rule-based (no model)",
            "Automated (89% of cases)",
            "Automated flag (2-3%)",
            "Bounded agent (most of remainder)",
            "Human (<1%)",
        ],
    }

    df = pd.DataFrame(architecture)
    print("Bounded agency architecture (structured overview):")
    print("-" * 65)
    for _, row in df.iterrows():
        print(f"\n  {row['Component']}")
        print(f"    Role:           {row['Role']}")
        print(f"    Autonomy:       {row['Autonomy level']}")
    print()


# --- Comparison: bounded vs. fully autonomous ---
def display_comparison():
    comparison = {
        "Dimension": [
            "Error detection",
            "Confidence quantification",
            "Edge case handling",
            "Audit trail",
            "Silent failure mode",
            "Defensibility to DRB",
        ],
        "Fully autonomous agent": [
            "None built in",
            "Implicit (no tiers)",
            "Guesses with confidence",
            "Murky",
            "Yes -- errors look like outputs",
            "Hard",
        ],
        "FSCM bounded agency": [
            "Cross-validation catches errors",
            "Explicit confidence tiers",
            "Flags for human review",
            "Every decision documented",
            "No -- ambiguity is surfaced",
            "Yes -- complete audit trail",
        ],
    }

    df = pd.DataFrame(comparison)
    print("Bounded agency vs. fully autonomous agent:")
    print("-" * 65)
    for _, row in df.iterrows():
        print(f"\n  {row['Dimension']}")
        print(f"    Fully autonomous: {row['Fully autonomous agent']}")
        print(f"    Bounded agency:   {row['FSCM bounded agency']}")

    print()
    print("Lesson: Bounded agency takes longer to specify and design.")
    print("The output is auditable and defensible. Fully autonomous is faster")
    print("to build and harder to defend when it fails.")


if __name__ == "__main__":
    display_problem()
    display_results()
    display_architecture()
    display_comparison()
