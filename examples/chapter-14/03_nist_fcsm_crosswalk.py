"""
03_nist_fcsm_crosswalk.py
Chapter 14: Evaluating AI Systems for Federal Use

NIST AI RMF to FCSM Quality Dimensions: Crosswalk.
Visualize arrows from the four RMF functions to the FCSM dimensions they address.
Print the key textual mappings for quick reference.
"""

import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
FIGURE_SIZE = (12, 5)
TITLE = "NIST AI RMF to FCSM Quality Dimensions: Crosswalk"

NIST_COLORS = {
    "GOVERN": "#4C96D7",
    "MAP": "#F4A261",
    "MEASURE": "#57CC99",
    "MANAGE": "#E76F51",
}

FCSM_DIMS = [
    "Relevance",
    "Accuracy & Reliability",
    "Timeliness",
    "Accessibility",
    "Coherence",
    "Interpretability",
]

# Which FCSM dimensions each RMF function maps to
MAPPING = {
    "GOVERN": ["Interpretability", "Coherence", "Accessibility"],
    "MAP": ["Relevance", "Interpretability"],
    "MEASURE": ["Accuracy & Reliability", "Coherence", "Timeliness"],
    "MANAGE": ["Timeliness", "Accessibility"],
}

# Vertical positions for NIST boxes (left column)
NIST_Y = {"GOVERN": 3.8, "MAP": 2.9, "MEASURE": 2.0, "MANAGE": 1.1}

KEY_MAPPINGS = [
    ("GOVERN", "Interpretability, Coherence",
     "Can users understand and trust the system?"),
    ("MAP", "Relevance",
     "Is this the right tool for this statistical task?"),
    ("MEASURE", "Accuracy, Coherence, Timeliness",
     "Does it meet production requirements?"),
    ("MANAGE", "Timeliness, Accessibility",
     "Can it operate within production constraints?"),
]


def draw_crosswalk():
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # FCSM dimension boxes on right
    fcsm_x = 8.5
    fcsm_y = {dim: 4.2 - i * 0.7 for i, dim in enumerate(FCSM_DIMS)}

    for dim in FCSM_DIMS:
        y = fcsm_y[dim]
        rect = plt.Rectangle(
            (fcsm_x - 1.5, y - 0.25), 3.0, 0.45,
            facecolor="lavender", edgecolor="gray", linewidth=1,
        )
        ax.add_patch(rect)
        ax.text(fcsm_x, y, dim, ha="center", va="center", fontsize=8)

    ax.text(fcsm_x, 4.8, "FCSM Quality Dimensions",
            ha="center", fontsize=9, fontweight="bold", color="purple")

    # NIST function boxes on left
    nist_x = 2.0
    for fn, color in NIST_COLORS.items():
        y = NIST_Y[fn]
        rect = plt.Rectangle(
            (nist_x - 0.8, y - 0.25), 1.6, 0.45,
            facecolor=color, alpha=0.4, edgecolor=color, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(nist_x, y, fn, ha="center", va="center",
                fontsize=9, fontweight="bold")

    ax.text(nist_x, 4.8, "NIST AI RMF Functions",
            ha="center", fontsize=9, fontweight="bold", color="navy")

    # Arrows from NIST functions to FCSM dimensions
    for fn, targets in MAPPING.items():
        color = NIST_COLORS[fn]
        for target in targets:
            if target in fcsm_y:
                ax.annotate(
                    "", xy=(fcsm_x - 1.5, fcsm_y[target]),
                    xytext=(nist_x + 0.8, NIST_Y[fn]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.6),
                )

    ax.set_title(TITLE, fontsize=11, pad=10)
    plt.tight_layout()
    plt.show()


def print_key_mappings():
    print("NIST AI RMF to FCSM: Key Mappings")
    print("=" * 65)
    for fn, dims, rationale in KEY_MAPPINGS:
        print(f"\n  {fn:<10} -> {dims}")
        print(f"             Why: {rationale}")

    print()
    print("Note: 'Interpretability' appears under GOVERN because interpretability")
    print("is a governance requirement -- users must be able to understand and")
    print("explain AI-generated outputs to stakeholders and oversight bodies.")
    print()
    print("Practical use: when evaluating an AI system, use GOVERN questions to")
    print("probe interpretability and coherence, MAP questions to probe relevance,")
    print("MEASURE questions for accuracy and reliability, and MANAGE questions")
    print("for timeliness and operational accessibility.")


if __name__ == "__main__":
    draw_crosswalk()
    print_key_mappings()
