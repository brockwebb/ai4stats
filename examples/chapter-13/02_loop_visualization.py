"""
Chapter 13: Agentic AI for Federal Statistical Operations
Example 02: Observe-decide-act-check loop visualization

Shows two annotation layers: recipe workflow and federal pipeline.
No savefig calls. Pedagogical illustration only.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_loop(ax, positions, colors, title, annotations):
    """Draw the observe-decide-act-check loop with custom annotations."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    step_labels = {
        "OBSERVE": "Take in\ninformation",
        "DECIDE": "Determine\nwhat to do",
        "ACT": "Execute\nthe action",
        "CHECK": "Did it work?\nDone / continue / stop?",
    }

    for step, (x, y) in positions.items():
        box = plt.Rectangle(
            (x - 1.2, y - 0.5), 2.4, 1.0,
            fill=True, facecolor=colors[step], alpha=0.85,
            edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.1, step, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(x, y - 0.22, step_labels[step], ha="center", va="center",
                fontsize=7.5, color="white", style="italic")

    # Arrows between steps
    arrow_props = dict(arrowstyle="-|>", color="black", lw=1.5)
    ax.annotate("", xy=(3.8, 4.5), xytext=(3.2, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(6.8, 4.5), xytext=(6.2, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(5, 2.5), xytext=(8, 3.95), arrowprops=arrow_props)
    ax.annotate("", xy=(2, 3.95), xytext=(3.8, 2),
                arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.5,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(2.1, 3.0, "loop", fontsize=8, color="#666666", style="italic")

    # Annotation callouts
    for step, annotation_text in annotations.items():
        x, y = positions[step]
        ax.text(x, y - 1.1, annotation_text, ha="center", va="top",
                fontsize=7, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDE7",
                          edgecolor="#F9A825", linewidth=0.8))

    ax.set_title(title, fontsize=10, pad=12)


# --- Recipe workflow annotations ---
recipe_annotations = {
    "OBSERVE": "Read recipe step;\ncheck dietary restrictions",
    "DECIDE": "Select recipe;\ngather ingredients",
    "ACT": "Prep and cook",
    "CHECK": "Taste and adjust;\ndone or iterate?",
}

# --- Federal pipeline annotations ---
federal_annotations = {
    "OBSERVE": "Intake survey response;\npull reference codes",
    "DECIDE": "Select coding prompt;\nchoose model / threshold",
    "ACT": "Call LLM;\nassign NAICS code",
    "CHECK": "Confidence >= threshold?\nFlag or accept?",
}

step_positions = {
    "OBSERVE": (2, 4.5),
    "DECIDE": (5, 4.5),
    "ACT": (8, 4.5),
    "CHECK": (5, 2),
}
step_colors = {
    "OBSERVE": "#1976D2",
    "DECIDE": "#7B1FA2",
    "ACT": "#388E3C",
    "CHECK": "#F57C00",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

draw_loop(
    axes[0], step_positions, step_colors,
    "Recipe workflow: observe-decide-act-check annotated",
    recipe_annotations,
)
draw_loop(
    axes[1], step_positions, step_colors,
    "Federal coding pipeline: observe-decide-act-check annotated",
    federal_annotations,
)

plt.suptitle(
    "The observe-decide-act-check loop: same pattern, different contexts",
    fontsize=11, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.show()
