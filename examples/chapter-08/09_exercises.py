"""
09_exercises.py
===============
Setup code and starter scaffolding for Chapter 8 exercises.

Exercise 8.1 -- Subgroup accuracy decomposition by age group
Exercise 8.2 -- Fairness metric conflicts and threshold choice
Exercise 8.3 -- Leadership briefing template

These exercises build on the predictions saved by 01_dataset_and_model.py.
Students modify this code and answer the interpretation questions in the
chapter. The setup code is intentionally minimal -- the interesting work
is in the interpretation, not the computation.

DEPENDENCIES:
    Loads ch08_test_predictions.csv produced by 01_dataset_and_model.py.
    Run 01_dataset_and_model.py first.

REQUIREMENTS:
    Python 3.9+, numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def load_predictions(data_dir: Path) -> pd.DataFrame:
    """Load test-set predictions from 01_dataset_and_model.py."""
    path = data_dir / "ch08_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}\n"
            "Run 01_dataset_and_model.py first."
        )
    return pd.read_csv(path)


def compute_group_metrics(subset: pd.DataFrame) -> dict | None:
    """Return accuracy, TPR, FNR, FPR, precision for a subgroup."""
    if len(subset) < 5:
        return None
    y_true = subset["y_true"].values
    y_pred = subset["y_pred"].values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(subset)
    return {
        "n": n,
        "base_rate": float(y_true.mean()),
        "accuracy": float((tp + tn) / n),
        "TPR": float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
        "FNR": float(fn / (tp + fn)) if (tp + fn) > 0 else np.nan,
        "FPR": float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan,
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan,
        "pred_rate": float(y_pred.mean()),
    }


# ---------------------------------------------------------------------------
# Exercise 8.1: Subgroup accuracy decomposition by age group
# ---------------------------------------------------------------------------

def exercise_8_1(df: pd.DataFrame) -> None:
    """
    Setup code for Exercise 8.1.

    The exercise asks students to:
      - Identify which racial/ethnic group has the highest miss rate
      - Identify which has the lowest accuracy
      - Explain whether the highest-miss-rate group is also the highest base-rate group
      - Extend the decomposition to age groups (setup provided here)
    """
    print("=" * 65)
    print("Exercise 8.1: Subgroup accuracy decomposition")
    print("=" * 65)
    print()

    race_groups = [
        "White non-Hispanic", "Black non-Hispanic", "Hispanic",
        "Asian non-Hispanic", "Other",
    ]

    print("Racial/ethnic group decomposition (pre-computed):")
    print(f"{'Group':<28} {'N':>5} {'Base':>6} {'Acc':>6} {'TPR':>6} {'FNR':>6}")
    print("-" * 65)
    for group in race_groups:
        subset = df[df["race"] == group]
        m = compute_group_metrics(subset)
        if m:
            print(
                f"{group:<28} {m['n']:>5} {m['base_rate']:>6.3f} "
                f"{m['accuracy']:>6.3f} {m['TPR']:>6.3f} {m['FNR']:>6.3f}"
            )

    print()
    print("--- Age group decomposition (starter for Exercise 8.1 extension) ---")

    # Students can modify age bin edges and run the decomposition
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 30, 45, 60, 80],
        labels=["18-30", "31-45", "46-60", "61-80"],
    )
    age_groups = ["18-30", "31-45", "46-60", "61-80"]

    print(f"{'Age group':<12} {'N':>5} {'Base':>6} {'Acc':>6} {'FNR':>6}")
    print("-" * 40)
    for group in age_groups:
        subset = df[df["age_group"] == group]
        m = compute_group_metrics(subset)
        if m:
            print(
                f"{group:<12} {m['n']:>5} {m['base_rate']:>6.3f} "
                f"{m['accuracy']:>6.3f} {m['FNR']:>6.3f}"
            )

    print()
    print("Questions to answer:")
    print("  1. Which group has the highest miss rate (FNR)?")
    print("  2. Is that also the group with the highest base rate?")
    print("  3. What does this tell you about how model errors compound?")


# ---------------------------------------------------------------------------
# Exercise 8.2: Fairness metric conflicts and threshold choice
# ---------------------------------------------------------------------------

def exercise_8_2(df: pd.DataFrame) -> None:
    """
    Setup code for Exercise 8.2.

    The exercise presents pre-computed fairness metrics and asks students to:
      - Identify which criterion is violated
      - Choose which criterion they would optimize and justify the choice
    """
    print()
    print("=" * 65)
    print("Exercise 8.2: Fairness metric conflicts")
    print("=" * 65)
    print()
    print("Pre-computed fairness metrics by racial/ethnic group:")
    print()

    race_groups = [
        "White non-Hispanic", "Black non-Hispanic", "Hispanic",
        "Asian non-Hispanic", "Other",
    ]

    print(
        f"{'Group':<28} {'Pred Rate':>10} {'TPR':>6} {'FPR':>6} {'Precision':>10}"
    )
    print("-" * 68)
    for group in race_groups:
        subset = df[df["race"] == group]
        m = compute_group_metrics(subset)
        if m:
            print(
                f"{group:<28} {m['pred_rate']:>10.3f} {m['TPR']:>6.3f} "
                f"{m['FPR']:>6.3f} {m['precision']:>10.3f}"
            )

    print()
    print("Questions:")
    print("  1. Does the model satisfy demographic parity?")
    print("     (Are positive prediction rates equal across groups?)")
    print("  2. Does the model satisfy equalized odds?")
    print("     (Are TPR and FPR equal across groups?)")
    print("  3. Which criterion would you optimize for a nonresponse")
    print("     prediction model in a federal survey?")
    print("  4. Justify your answer in terms of the cost of false negatives")
    print("     vs. false positives for communities that are already undercounted.")


# ---------------------------------------------------------------------------
# Exercise 8.3: Leadership briefing template
# ---------------------------------------------------------------------------

def exercise_8_3() -> None:
    """
    Template for Exercise 8.3 leadership briefing.

    The exercise scenario: briefing a division chief on a vendor AI system
    for income imputation that shows strong overall accuracy but significant
    subgroup disparities.
    """
    print()
    print("=" * 65)
    print("Exercise 8.3: Leadership briefing template")
    print("=" * 65)
    print()
    print("SCENARIO:")
    print("  You are briefing your division chief on a proposed AI system")
    print("  that automates income imputation for ACS microdata.")
    print()
    print("  The vendor reports:")
    print("    Overall accuracy:                       94%")
    print("    Validation dataset:                     50,000 records")
    print()
    print("  Your subgroup analysis shows:")
    print("    Accuracy for income lowest quintile:    78%")
    print("    Accuracy for income above median:       97%")
    print("    Accuracy for Hispanic households:       81%")
    print("    Accuracy for Black non-Hispanic HHs:    83%")
    print()
    print("BRIEFING OUTLINE (fill in your answers):")
    print()
    print("  1. What does '94% overall accuracy' conceal in this case?")
    print("     [YOUR ANSWER]")
    print()
    print("  2. Who bears the error burden?")
    print("     Is this distribution acceptable for a federal statistical system?")
    print("     [YOUR ANSWER]")
    print()
    print("  3. What additional information would you request from the vendor")
    print("     before a procurement decision?")
    print("     [YOUR ANSWER]")
    print()
    print("  4. What conditions would you require before approving this system")
    print("     for production use?")
    print("     [YOUR ANSWER]")
    print()
    print("  5. How does the impossibility theorem change your evaluation of")
    print("     the vendor's promise to improve the model?")
    print("     [YOUR ANSWER]")
    print()
    print("Key principle: An AI vendor who claims their model is 'fair to everyone'")
    print("either does not understand the impossibility theorem or is not being")
    print("honest with you. The right question is: which errors did they choose")
    print("to minimize, for which groups, and why?")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    here = Path(__file__).parent

    df = load_predictions(here)
    exercise_8_1(df)
    exercise_8_2(df)
    exercise_8_3()
