"""
06_error_analysis.py -- Chapter 12: Error Type Classification

Classifies LLM coding errors into types (adjacent sector, unrelated sector,
refusal) and visualizes the breakdown. Understanding error types is more
actionable than overall accuracy alone.

This script:
- Loads the simulated evaluation dataset (see 03_evaluation_dataset.py)
- Classifies each record as correct, adjacent sector error, unrelated error,
  or refusal
- Produces a visualization of error type breakdown by sector
- Prints an error breakdown table

Standalone: no external data files required. Run with Python 3.9+.
No LLM API calls are made.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from _shared_data import get_full_eval_df, SECTORS, SECTOR_CODES  # noqa: E402

# ---------------------------------------------------------------------------
# Adjacent sector pairs: these represent genuine coding ambiguity.
# When the LLM picks the wrong code in an adjacent pair, the error is
# understandable and may even be defensible on some descriptions.
# When the LLM picks a non-adjacent wrong sector, the error is more severe.
# ---------------------------------------------------------------------------

ADJACENT_PAIRS = [
    frozenset({"54", "51"}),
    frozenset({"54", "52"}),
    frozenset({"62", "81"}),
    frozenset({"61", "92"}),
    frozenset({"44-45", "72"}),
    frozenset({"23", "81"}),
    frozenset({"51", "52"}),
]


def classify_error(human_code, llm_code):
    """
    Classify a single LLM coding result into an error type.

    Returns
    -------
    str : one of "correct", "adjacent_sector", "unrelated_sector", "refusal"
    """
    if llm_code == "UNCLEAR":
        return "refusal"
    if human_code == llm_code:
        return "correct"
    pair = frozenset({human_code, llm_code})
    if pair in ADJACENT_PAIRS:
        return "adjacent_sector"
    return "unrelated_sector"


ERROR_COLORS = {
    "correct":          "steelblue",
    "adjacent_sector":  "gold",
    "unrelated_sector": "tomato",
    "refusal":          "lightgray",
}

ERROR_LABELS = {
    "correct":          "Correct",
    "adjacent_sector":  "Adjacent sector (ambiguous)",
    "unrelated_sector": "Unrelated sector (clear error)",
    "refusal":          "Refusal / UNCLEAR",
}


if __name__ == "__main__":
    df_eval = get_full_eval_df()

    df_eval["error_type"] = df_eval.apply(
        lambda r: classify_error(r["human_sector"], r["llm_sector"]), axis=1
    )

    error_types = ["correct", "adjacent_sector", "unrelated_sector", "refusal"]

    # Overall summary
    error_counts = df_eval["error_type"].value_counts().reindex(error_types, fill_value=0)
    n_total = len(df_eval)

    print("=" * 60)
    print("ERROR TYPE BREAKDOWN -- OVERALL")
    print("=" * 60)
    for etype in error_types:
        count = error_counts[etype]
        pct = count / n_total * 100
        bar = "#" * int(pct / 2)
        print(f"  {ERROR_LABELS[etype]:<35}: {count:>3} ({pct:.1f}%) {bar}")
    n_errors = n_total - error_counts["correct"]
    print()
    print(f"  Total errors: {n_errors} ({n_errors/n_total:.1%})")
    print()
    print("  Adjacent sector errors represent genuine ambiguity -- human coders")
    print("  also disagree on these cases. Unrelated sector errors are the")
    print("  more serious failure: the model missed the sector entirely.")

    # Per-sector breakdown
    print()
    print("=" * 75)
    print("ERROR TYPE BREAKDOWN BY SECTOR")
    print("=" * 75)
    print(f"  {'Code':<8} {'Sector':<40} {'Correct':>8} {'Adjacent':>9} "
          f"{'Unrelated':>10} {'Refusal':>8}")
    print("  " + "-" * 78)
    sector_rows = []
    for code in SECTOR_CODES:
        mask = df_eval["human_sector"] == code
        sub = df_eval[mask]
        row = {"code": code, "name": SECTORS[code], "n": len(sub)}
        for etype in error_types:
            row[etype] = (sub["error_type"] == etype).sum()
        sector_rows.append(row)
        n = row["n"]
        print(
            f"  {code:<8} {SECTORS[code]:<40} "
            f"{row['correct']:>5} ({row['correct']/n:.0%}) "
            f"{row['adjacent_sector']:>5} ({row['adjacent_sector']/n:.0%}) "
            f"{row['unrelated_sector']:>5} ({row['unrelated_sector']/n:.0%}) "
            f"{row['refusal']:>5} ({row['refusal']/n:.0%})"
        )
    sector_df = pd.DataFrame(sector_rows)

    # Stacked bar chart: error type breakdown per sector
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(sector_df))
    for etype in error_types:
        vals = sector_df[etype].values / sector_df["n"].values * 100
        ax.barh(
            sector_df["name"], vals, left=bottom,
            color=ERROR_COLORS[etype], label=ERROR_LABELS[etype],
            edgecolor="white", linewidth=0.5,
        )
        bottom = bottom + vals

    ax.set_xlabel("Percentage of records (%)")
    ax.set_title("LLM coding error types by NAICS sector (simulated)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 105)
    ax.axvline(100, color="gray", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "06_error_analysis.png"),
        dpi=120, bbox_inches="tight",
    )
    print()
    print("Chart saved: 06_error_analysis.png")
    plt.show()

    # Highlight worst sectors by unrelated error rate
    sector_df["unrelated_rate"] = (
        sector_df["unrelated_sector"] / sector_df["n"]
    )
    worst = sector_df.nlargest(3, "unrelated_rate")[
        ["code", "name", "unrelated_sector", "n", "unrelated_rate"]
    ]
    print()
    print("Sectors with highest unrelated-sector error rate:")
    for _, row in worst.iterrows():
        print(f"  {row['code']:<8} {row['name']:<40}: "
              f"{row['unrelated_sector']}/{row['n']} = {row['unrelated_rate']:.0%}")
    print()
    print("These sectors warrant targeted prompt revision or additional few-shot examples.")
