"""
Chapter 13: Agentic AI for Federal Statistical Operations
Example 03: Autonomy dial visualization

Horizontal spectrum from Full Human Control to Fully Autonomous.
Federal Zone marked in the left third.
No savefig calls. Pedagogical illustration only.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_autonomy_dial():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Dial line
    ax.plot([0.5, 9.5], [2.0, 2.0], color="#333333", linewidth=3)
    ax.annotate("", xy=(9.7, 2.0), xytext=(9.2, 2.0),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2))

    # Positions on the dial
    positions = [
        (1.0, "Tool use\nonly", "#1976D2"),
        (3.0, "Propose,\nhuman approves", "#388E3C"),
        (5.5, "Auto-execute\nwith human review", "#FF9800"),
        (7.5, "Exceptions\nescalate to human", "#E64A19"),
        (9.5, "Fully\nautonomous", "#B71C1C"),
    ]

    for x, label, color in positions:
        ax.plot(x, 2.0, "o", color=color, markersize=16, zorder=5)
        ax.text(x, 1.1, label, ha="center", va="top", fontsize=7.5,
                color=color, fontweight="bold")

    # Federal Zone shading (left third: x 0.3 to 4.0)
    ax.add_patch(plt.Rectangle(
        (0.3, 2.25), 3.7, 0.9,
        facecolor="#1976D2", alpha=0.12, edgecolor="#1976D2", linewidth=1.5,
    ))
    ax.text(2.0, 2.7, "Federal Zone", ha="center", va="center",
            fontsize=9, color="#1976D2", fontweight="bold")
    ax.text(2.0, 3.05, "Federal statistical operations belong here",
            ha="center", va="center", fontsize=7.5, color="#1976D2")

    # Axis labels
    ax.text(0.5, 3.55, "FULL HUMAN CONTROL", ha="left", va="top",
            fontsize=9, color="#1976D2", fontweight="bold")
    ax.text(9.5, 3.55, "FULLY AUTONOMOUS", ha="right", va="top",
            fontsize=9, color="#B71C1C", fontweight="bold")

    ax.set_title("The autonomy dial: where federal operations belong",
                 fontsize=11, pad=10)
    plt.tight_layout()
    plt.show()

    print()
    print("Rule of thumb: Move right only when you have to. Stay left when you can.")
    print()
    print("What 'have to' means: specific, auditable decisions where all three hold:")
    print("  1. Decision volume makes human review impractical at production scale")
    print("  2. Errors can be caught and corrected before they reach published data")
    print("  3. Error cost is proportionate to the saved human review burden")
    print()
    print("Federal statistical production rarely satisfies all three conditions for full")
    print("autonomy. Bounded agency (AI proposes, human approves) is the default.")


if __name__ == "__main__":
    draw_autonomy_dial()
