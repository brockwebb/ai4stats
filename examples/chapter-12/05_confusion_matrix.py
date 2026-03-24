"""
05_confusion_matrix.py -- Chapter 12: Confusion Matrix Analysis

Builds raw and normalized confusion matrices for LLM industry coding errors.
Identifies the most common sector confusion pairs.

This script:
- Loads the simulated evaluation dataset (see 03_evaluation_dataset.py)
- Plots raw count and row-normalized confusion matrices side by side
- Extracts and prints the top 5 confusion pairs
- Explains what each confusion pair reveals about model failure modes

Standalone: no external data files required. Run with Python 3.9+.
No LLM API calls are made.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(__file__))
from _shared_data import get_full_eval_df, SECTORS, SECTOR_CODES  # noqa: E402

# Short display labels for matrix axes
SHORT_NAMES = {
    "44-45": "Retail",
    "62":    "Health",
    "61":    "Educ",
    "54":    "ProfSvc",
    "72":    "FoodSvc",
    "23":    "Constr",
    "52":    "Finance",
    "51":    "InfoTech",
    "81":    "OthSvc",
    "92":    "PubAdmin",
}


if __name__ == "__main__":
    df_eval = get_full_eval_df()

    # Keep only records where LLM assigned a known sector code
    valid_mask = df_eval["llm_sector"].isin(SECTOR_CODES)
    df_valid = df_eval[valid_mask].copy()

    # Encode sector codes to integer labels
    le = LabelEncoder()
    le.fit(SECTOR_CODES)
    y_true = le.transform(df_valid["human_sector"])
    y_pred = le.transform(df_valid["llm_sector"])

    n_classes = len(SECTOR_CODES)
    labels_idx = list(range(n_classes))
    names_short = [SHORT_NAMES[c] for c in SECTOR_CODES]

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    # Row-normalize: each row sums to 1 (proportion of true sector coded as X)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw counts", "Normalized (row = true sector)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, ax=ax, cmap="Blues",
            xticklabels=names_short,
            yticklabels=names_short,
            annot=True, fmt=fmt,
            linewidths=0.4, linecolor="lightgray",
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title)
        ax.set_xlabel("LLM-assigned sector")
        ax.set_ylabel("Human-assigned sector (true)")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.suptitle("LLM Industry Coding Confusion Matrix (simulated)", fontsize=13)
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "05_confusion_matrix.png"),
        dpi=120, bbox_inches="tight",
    )
    print("Chart saved: 05_confusion_matrix.png")
    plt.show()

    # Extract confusion pairs (off-diagonal, sorted by count)
    confusion_pairs = []
    for i, true_code in enumerate(SECTOR_CODES):
        for j, pred_code in enumerate(SECTOR_CODES):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((
                    int(cm[i, j]),
                    float(cm_norm[i, j]),
                    true_code,
                    pred_code,
                    SECTORS[true_code],
                    SECTORS[pred_code],
                ))
    confusion_pairs.sort(reverse=True)

    print()
    print("=" * 75)
    print("TOP CONFUSION PAIRS (human code -> LLM-assigned code)")
    print("=" * 75)
    print(f"  {'Rank':<5} {'Count':>6}  {'Rate':>6}  True sector --> LLM sector")
    print("  " + "-" * 70)
    for rank, (count, rate, tc, pc, tname, pname) in enumerate(confusion_pairs[:5], 1):
        print(f"  {rank:<5} {count:>6}  {rate:>6.1%}  {tc} ({tname}) --> {pc} ({pname})")

    print()
    print("Interpretation of top confusion pairs:")
    interpretations = {
        ("54", "51"): (
            "Professional Services miscoded as Information. "
            "IT consulting firms are sector 54; software companies are sector 51. "
            "The employer's primary activity distinguishes them."
        ),
        ("81", "62"): (
            "Other Services miscoded as Health Care. "
            "Day spas, massage therapists can look like healthcare to the model. "
            "The distinction is whether the provider is licensed healthcare."
        ),
        ("81", "54"): (
            "Other Services miscoded as Professional Services. "
            "Independent tradespeople and service providers cross this boundary."
        ),
        ("92", "61"): (
            "Public Administration miscoded as Education. "
            "State education agencies and school boards are NAICS 92, not 61."
        ),
        ("52", "54"): (
            "Finance miscoded as Professional Services. "
            "Financial advisors at independent firms are 52; "
            "independent fee-only advisors at consulting firms are 54."
        ),
    }
    shown = set()
    for count, rate, tc, pc, tname, pname in confusion_pairs[:5]:
        key = (tc, pc)
        if key in interpretations and key not in shown:
            print(f"\n  {tc} -> {pc}: {interpretations[key]}")
            shown.add(key)

    # Diagonal accuracy summary
    diagonal_acc = cm.diagonal() / cm.sum(axis=1)
    worst_idx = np.argmin(diagonal_acc)
    best_idx = np.argmax(diagonal_acc)
    print()
    print(f"  Lowest diagonal accuracy: {SECTOR_CODES[worst_idx]} "
          f"({SECTORS[SECTOR_CODES[worst_idx]]}) = {diagonal_acc[worst_idx]:.0%}")
    print(f"  Highest diagonal accuracy: {SECTOR_CODES[best_idx]} "
          f"({SECTORS[SECTOR_CODES[best_idx]]}) = {diagonal_acc[best_idx]:.0%}")
