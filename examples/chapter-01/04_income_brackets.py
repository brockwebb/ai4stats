"""
Chapter 1 -- Multi-Class Classification: Income Brackets

Loads the synthetic ACS-like survey data, bins continuous income into three
brackets (low / middle / high), and fits a multi-class logistic regression.

What this script demonstrates:
- When discretizing a continuous variable is appropriate vs. when it loses
  information (discussed in comments and printed output)
- Stratified train/test split for multi-class targets
- Macro vs. weighted averaging of precision, recall, and F1 across classes
- Raw and row-normalized confusion matrices for multi-class problems
- How middle-bracket misclassification patterns differ from boundary errors

Income bracket thresholds are chosen for illustration; real survey work would
use policy-defined thresholds (poverty line, median income, etc.).

Prerequisites: run 01_generate_survey_data.py first.

Usage:
    python 04_income_brackets.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = "synthetic_acs_survey.csv"
FEATURES = ["age", "education_years", "hours_per_week", "urban"]
TARGET = "income"
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Bracket boundaries and labels
BINS = [-np.inf, 40_000, 90_000, np.inf]
BRACKET_LABELS = {0: "Low (<$40k)", 1: "Middle ($40k-$90k)", 2: "High (>$90k)"}
BRACKET_KEYS = [0, 1, 2]


def load_and_bin(path):
    """
    Load synthetic survey CSV and bin income into three brackets.

    Returns df, X (features), y (bracket labels as int).
    """
    df = pd.read_csv(path)
    y = pd.cut(
        df[TARGET],
        bins=BINS,
        labels=BRACKET_KEYS,
        right=True,
        include_lowest=True,
    ).astype(int)
    X = df[FEATURES]
    return df, X, y


def print_bracket_distribution(y):
    """Print count and percentage for each income bracket."""
    counts = pd.Series(y).value_counts().sort_index()
    total = len(y)
    print("Income bracket distribution:")
    for k in BRACKET_KEYS:
        n = counts.get(k, 0)
        print(f"  {BRACKET_LABELS[k]:<22}  {n:>5} records  ({n/total:.1%})")
    print()


def print_multiclass_metrics(y_true, y_pred):
    """Print accuracy and both macro and weighted averages of P/R/F1."""
    acc = accuracy_score(y_true, y_pred)
    print(f"  Accuracy:              {acc:.3f}")
    for avg in ("macro", "weighted"):
        prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
        print(f"  Precision ({avg:<8}): {prec:.3f}")
        print(f"  Recall    ({avg:<8}): {rec:.3f}")
        print(f"  F1        ({avg:<8}): {f1:.3f}")
    print()
    print("  Use macro when each bracket deserves equal weight regardless of size.")
    print("  Use weighted when overall accuracy across people matters more.")


def plot_confusion_matrices(y_true, y_pred, tick_labels, title="Income bracket confusion"):
    """
    Plot raw counts and row-normalized confusion matrices side by side.

    Row normalization reveals per-class recall (diagonal = fraction of true
    instances correctly classified for that bracket).
    """
    cm = confusion_matrix(y_true, y_pred, labels=BRACKET_KEYS)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, data, fmt, subtitle in [
        (axes[0], cm, "d", "Raw counts"),
        (axes[1], cm_norm, ".2f", "Row-normalized (recall per bracket)"),
    ]:
        ax.imshow(data, cmap="Blues", vmin=0, vmax=(1 if fmt == ".2f" else None))
        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xlabel("Predicted bracket")
        ax.set_ylabel("True bracket")
        ax.set_xticks(BRACKET_KEYS)
        ax.set_yticks(BRACKET_KEYS)
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")
        ax.set_yticklabels(tick_labels)
        for i in BRACKET_KEYS:
            for j in BRACKET_KEYS:
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load, bin, and inspect
    # ------------------------------------------------------------------
    df, X, y = load_and_bin(CSV_PATH)
    print_bracket_distribution(y)

    # ------------------------------------------------------------------
    # Note on discretization
    # ------------------------------------------------------------------
    # When to discretize:
    #   - Output brackets are required by policy or reporting standards
    #   - Stakeholders need categorical labels for program eligibility decisions
    #   - You are comparing across subgroups and bracket membership is the unit
    #
    # When discretization loses information:
    #   - The continuous variable has meaningful variation within a bracket
    #     (e.g., $39,000 and $41,000 are treated the same as $10,000 and $89,000)
    #   - Downstream analysis needs the continuous value (e.g., imputation for
    #     total household income estimation)
    #   - Bracket boundaries are arbitrary and could be drawn differently
    #
    # In general: keep continuous variables continuous as long as possible.
    # Discretize only at the reporting or decision-making step.

    # ------------------------------------------------------------------
    # 2. Stratified split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows\n")

    # ------------------------------------------------------------------
    # 3. Fit multi-class logistic regression
    # ------------------------------------------------------------------
    clf = LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # ------------------------------------------------------------------
    # 4. Metrics
    # ------------------------------------------------------------------
    print("Multi-class logistic regression metrics:")
    print_multiclass_metrics(y_test, y_pred)

    # ------------------------------------------------------------------
    # 5. Confusion matrices
    # ------------------------------------------------------------------
    tick_labels = [BRACKET_LABELS[k] for k in BRACKET_KEYS]
    plot_confusion_matrices(y_test, y_pred, tick_labels)
    plt.savefig("fig_income_bracket_confusion.png", dpi=120, bbox_inches="tight")

    plt.show()
    print("Done. Figure saved to fig_income_bracket_confusion.png")
