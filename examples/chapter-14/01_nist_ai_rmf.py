"""
01_nist_ai_rmf.py
Chapter 14: Evaluating AI Systems for Federal Use

NIST AI Risk Management Framework 1.0 (January 2023)
Visualize the four core functions: GOVERN, MAP, MEASURE, MANAGE.

The AI RMF is the federal government's primary AI governance framework.
It was never rescinded and does not depend on any executive order.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration ─────────────────────────────────────────────────────────────
FIGURE_SIZE = (12, 6)
TITLE = "NIST AI Risk Management Framework 1.0 (January 2023)"
SEED = 2025  # not used here but set for consistency across chapter scripts

FUNCTIONS = [
    {
        "label": "GOVERN",
        "x": 1.5,
        "color": "#4C96D7",
        "description": "Policies, roles, accountability.\nWho is responsible?\nWhat are the rules?",
    },
    {
        "label": "MAP",
        "x": 3.5,
        "color": "#F4A261",
        "description": "Context of use, risk identification.\nWhat is the system for?\nWho is affected?",
    },
    {
        "label": "MEASURE",
        "x": 5.5,
        "color": "#57CC99",
        "description": "Metrics, testing, evaluation.\nHow do we know it works?\nWhat are the error rates?",
    },
    {
        "label": "MANAGE",
        "x": 8.0,
        "color": "#E76F51",
        "description": "Risk treatment, monitoring,\nincident response.\nWhat do we do when it fails?",
    },
]

ANNOTATION_TEXT = (
    "AI RMF is NOT a checklist.\n"
    "It is risk-based: higher-stakes uses\n"
    "require more rigorous application."
)


def draw_rmf_diagram():
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    for fn in FUNCTIONS:
        rect = plt.Rectangle(
            (fn["x"] - 0.9, 1.5), 1.8, 3.2,
            facecolor=fn["color"], alpha=0.3,
            edgecolor=fn["color"], linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(fn["x"], 4.2, fn["label"],
                ha="center", va="bottom", fontsize=14, fontweight="bold",
                color=fn["color"])
        ax.text(fn["x"], 2.0, fn["description"],
                ha="center", va="bottom", fontsize=8, color="black", linespacing=1.5)

    # Arrows between adjacent functions
    arrow_segments = [(2.4, 2.6), (4.4, 4.6), (6.4, 7.1)]
    for x_from, x_to in arrow_segments:
        ax.annotate(
            "", xy=(x_to, 3.0), xytext=(x_from, 3.0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
        )

    ax.text(
        6.0, 5.0, ANNOTATION_TEXT,
        ha="center", fontsize=9, style="italic", color="gray",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    ax.set_title(TITLE, fontsize=12, pad=10)
    plt.tight_layout()
    plt.show()


def print_summary():
    print("NIST AI Risk Management Framework 1.0 — Function Summary")
    print("=" * 60)
    for fn in FUNCTIONS:
        print(f"\n{fn['label']}")
        print(f"  {fn['description'].replace(chr(10), ' | ')}")
    print()
    print("Key insight: the AI RMF does not tell you what to do.")
    print("It provides a structure for asking the right questions at each stage.")
    print("The level of rigor required scales with the stakes of the application.")
    print()
    print("Status: The NIST AI RMF (NIST AI 100-1) was published January 2023.")
    print("It was never rescinded. It remains the federal government's primary")
    print("AI governance framework under any executive order or OMB memorandum.")


if __name__ == "__main__":
    draw_rmf_diagram()
    print_summary()
