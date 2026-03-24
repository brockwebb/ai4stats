"""
Chapter 15: Terminology drift detection.
Demonstrates the compute_term_similarity() function, reference definitions,
later usages, drift detection table output, and three-panel drift visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)


def compute_term_similarity(reference_def, later_usage):
    """
    Compute how similar a later usage is to the reference definition.
    Lower similarity suggests potential drift.
    """
    matcher = SequenceMatcher(None, reference_def.lower(), later_usage.lower())
    return matcher.ratio()


# Reference definitions established in session 1
reference_terms = {
    "unit nonresponse": "cases where the entire household refused to participate",
    "item nonresponse": "cases where the household responded but specific items are missing",
    "privacy budget": "epsilon = 0.5, approved by DRB, applied to all noise-infusion steps",
    "income imputation method": "random forest imputation, logistic regression rejected in pilot",
}

# How these terms appear in later sessions (with drift)
later_usages = {
    "unit nonresponse": [
        # Session 2: still correct
        "cases where the entire household refused to participate",
        # Session 3: slight drift
        "overall nonresponse patterns including household refusals",
        # Session 5: drift merged with item nonresponse
        "nonresponse patterns across the survey",
    ],
    "privacy budget": [
        # Session 2: correct
        "epsilon = 0.5 per DRB approval",
        # Session 4 (after revision): correct revision to 1.0
        "epsilon = 1.0 per DRB revision",
        # Session 5: reverted to old value (T4/T5 interaction)
        "epsilon = 0.5 as established in our initial methodology",
    ],
    "income imputation method": [
        # Session 2: correct
        "random forest imputation for income item nonresponse",
        # Session 3: T2 failure - confabulation
        "logistic regression as our primary imputation method",
        # Session 5: T5 carries forward T2 confabulation
        "logistic regression for income imputation",
    ],
}

print("Terminology drift analysis: similarity to session 1 reference definitions")
print("=" * 70)
print(f"{'Term':<30} {'Session 2':>12} {'Session 3':>12} {'Session 5':>12}")
print("-" * 70)

for term, ref_def in reference_terms.items():
    if term in later_usages:
        usages = later_usages[term]
        sims = [compute_term_similarity(ref_def, u) for u in usages]
        sim_str = "  ".join(f"{s:.2f}" for s in sims)
        flag = " <-- DRIFT" if min(sims) < 0.4 else ""
        print(f"{term:<30} {sims[0]:>12.2f} {sims[1]:>12.2f} {sims[2]:>12.2f}{flag}")

print()
print("Interpretation: scores below 0.40 indicate substantial drift from reference definition.")
print("A score of 1.00 means identical; 0.00 means no shared content.")

# Visualize terminology drift over sessions

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

terms_to_plot = ["unit nonresponse", "privacy budget", "income imputation method"]
session_labels = ["Session 2", "Session 3", "Session 5"]
colors = ["#2196F3", "#FF9800", "#F44336"]

for i, term in enumerate(terms_to_plot):
    ref_def = reference_terms[term]
    usages = later_usages[term]
    sims = [compute_term_similarity(ref_def, u) for u in usages]

    ax = axes[i]
    bars = ax.bar(session_labels, sims, color=colors, alpha=0.75, edgecolor="white")
    ax.axhline(y=0.40, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="Drift threshold")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"\"{term}\"", fontsize=9, wrap=True)
    ax.set_ylabel("Similarity to session 1 definition" if i == 0 else "")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Annotate failures
    for j, (bar, sim) in enumerate(zip(bars, sims)):
        if sim < 0.4:
            ax.text(bar.get_x() + bar.get_width() / 2, sim + 0.03,
                   "DRIFT", ha="center", va="bottom", fontsize=8, color="#F44336", fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, sim + 0.03,
                   f"{sim:.2f}", ha="center", va="bottom", fontsize=8)

    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle("Terminology drift detection: similarity to session 1 reference definitions",
            fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig("../assets/diagrams/chapter_15_terminology_drift.png", dpi=120, bbox_inches="tight")
plt.show()
print("Figure: Terminology consistency rates across three key terms over five sessions.")
