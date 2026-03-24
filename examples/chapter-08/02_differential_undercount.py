"""
02_differential_undercount.py
==============================
Visualizes 2020 Census differential undercount by demographic group using
published Post-Enumeration Survey (PES) estimates.

WHY THIS MATTERS:
    The differential undercount is the clearest example of disparate impact
    in federal statistical production. Positive values mean a group was
    missed at a higher rate than the overall population; negative values
    mean a group was counted more than once (net overcount). Communities
    with a net undercount receive proportionally fewer resources in federal
    funding formulas and less representation in apportionment.

    This chart establishes the real-world stakes before the chapter
    introduces the model-based analysis.

SOURCE:
    U.S. Census Bureau, "2020 Post-Enumeration Survey Estimation Report"
    (November 2022). Table 4: Net Coverage Rates by Race and Hispanic Origin.
    https://www.census.gov/library/publications/2022/dec/G-01.html

OUTPUTS:
    - Displays bar chart (positive = undercount, negative = overcount)
    - Saves ../figures/ch08_differential_undercount.png

REQUIREMENTS:
    Python 3.9+, numpy, matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# 2020 Census PES estimates -- all values in percentage points
# Positive = net undercount; Negative = net overcount
# Source: Census Bureau G-01 report, November 2022
# ---------------------------------------------------------------------------
PES_DATA = [
    ("American Indian/Alaska Native (on reservation)", 5.64, "Undercounted"),
    ("Hispanic",                                        4.99, "Undercounted"),
    ("Black non-Hispanic",                              3.30, "Undercounted"),
    ("Native Hawaiian/Other Pacific Islander",          1.92, "Undercounted"),
    ("White non-Hispanic",                             -1.64, "Overcounted"),
    ("Asian non-Hispanic",                             -2.62, "Overcounted"),
]

FIGURE_TITLE = (
    "2020 Census: Differential undercount by demographic group\n"
    "(Source: Census Bureau Post-Enumeration Survey, November 2022)"
)
XLABEL = "Net undercount rate (%) -- positive = undercounted, negative = overcounted"
FIGURE_DPI = 120
XLIM = (-4.5, 8.0)

# Colors: blue = undercounted (resource loss), red = overcounted
COLOR_UNDER = "#1976D2"
COLOR_OVER = "#F44336"


def build_chart(save_path: Path | None = None) -> None:
    """Draw horizontal bar chart of PES net undercount rates."""
    # Sort by rate ascending so worst undercount appears at top
    data_sorted = sorted(PES_DATA, key=lambda x: x[1])
    labels = [row[0] for row in data_sorted]
    values = [row[1] for row in data_sorted]
    colors = [COLOR_UNDER if v > 0 else COLOR_OVER for v in values]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(XLABEL, fontsize=9)
    ax.set_title(FIGURE_TITLE, fontsize=10)
    ax.set_xlim(*XLIM)

    for bar, val in zip(bars, values):
        offset = 0.12 if val >= 0 else -0.12
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}%",
            va="center",
            ha=ha,
            fontsize=8.5,
        )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    plt.show()


def print_summary() -> None:
    """Print the PES data with a plain-language interpretation."""
    print("2020 Census Post-Enumeration Survey -- Net Coverage Rates")
    print("=" * 62)
    print()
    print(f"{'Group':<50} {'Rate':>8}")
    print("-" * 62)
    for label, rate, direction in sorted(PES_DATA, key=lambda x: -x[1]):
        flag = "(undercounted)" if rate > 0 else "(overcounted)"
        print(f"{label:<50} {rate:>+7.2f}%  {flag}")

    print()
    print("Consequence:")
    print("  Congressional apportionment, federal funding formulas, and")
    print("  redistricting all use Census counts. A 5.6% undercount of")
    print("  American Indians on reservations represents thousands of")
    print("  uncounted people whose communities receive proportionally less")
    print("  representation and fewer federal resources.")
    print()
    print("  When ML models are trained on Census-derived data, they inherit")
    print("  this differential coverage error unless corrective measures are")
    print("  explicitly applied and documented.")


if __name__ == "__main__":
    here = Path(__file__).parent
    figures_dir = here.parent.parent / "figures"
    save_path = figures_dir / "ch08_differential_undercount.png"

    print_summary()
    build_chart(save_path=save_path)
