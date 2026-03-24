"""
08_model_card.py
================
Populates and prints a model card for the Chapter 8 logistic regression
nonresponse model, following the Mitchell et al. (2019) model card format.

WHY MODEL CARDS:
    A model card is the documentation artifact that makes fairness tradeoffs
    explicit and auditable. For federal agencies, it is the record of
    accountability. When a system produces disparate impact, the model card
    shows whether the decision-makers knew about it, documented it, and made
    a deliberate choice about which fairness criterion to prioritize.

    Model cards are not optional analysis. Under OMB Statistical Policy
    Directive 15 and Executive Order provisions on AI equity, documentation
    of subgroup performance is a governance requirement.

REFERENCE:
    Mitchell, M., et al. (2019). "Model Cards for Model Reporting."
    Proceedings of the ACM Conference on Fairness, Accountability, and
    Transparency (FAccT).

DEPENDENCIES:
    Loads ch08_test_predictions.csv produced by 01_dataset_and_model.py.
    Run 01_dataset_and_model.py first.

OUTPUTS:
    - Prints the populated model card to stdout

REQUIREMENTS:
    Python 3.9+, numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from pathlib import Path

RACE_GROUPS = [
    "White non-Hispanic",
    "Black non-Hispanic",
    "Hispanic",
    "Asian non-Hispanic",
    "Other",
]
MIN_GROUP_SIZE = 5


def load_predictions(data_dir: Path) -> pd.DataFrame:
    """Load test-set predictions from 01_dataset_and_model.py."""
    path = data_dir / "ch08_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}\n"
            "Run 01_dataset_and_model.py first."
        )
    return pd.read_csv(path)


def compute_subgroup_range(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    """Return (min, max) of a metric across all racial/ethnic groups."""
    values = []
    for group in RACE_GROUPS:
        subset = df[df["race"] == group]
        if len(subset) < MIN_GROUP_SIZE:
            continue

        y_true = subset["y_true"].values
        y_pred = subset["y_pred"].values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        n = len(subset)

        if metric == "accuracy":
            values.append((tp + tn) / n)
        elif metric == "fnr":
            if (tp + fn) > 0:
                values.append(fn / (tp + fn))

    return min(values), max(values)


def build_model_card(df: pd.DataFrame) -> dict:
    """Compute performance metrics and assemble the model card data structure."""
    y_test = df["y_true"].values
    y_pred = df["y_pred"].values

    overall_acc = accuracy_score(y_test, y_pred)
    overall_prec = precision_score(y_test, y_pred, zero_division=0)
    overall_rec = recall_score(y_test, y_pred, zero_division=0)

    acc_min, acc_max = compute_subgroup_range(df, "accuracy")
    fnr_min, fnr_max = compute_subgroup_range(df, "fnr")

    return {
        "Model identification": {
            "Model name":      "Nonresponse propensity model v1.0",
            "Version date":    "2026-03-21",
            "Intended use":    "Predict nonresponse probability for targeted follow-up in ACS",
            "Out-of-scope":    "Not intended for eligibility decisions or individual scoring",
        },
        "Training data": {
            "Source":                         "ACS PUMS 2019-2023 5-year respondents (synthetic in chapter example)",
            "Known limitation":               "Survivorship bias: nonrespondents absent from training data",
            "Protected characteristics used": "Race, ethnicity, age present as features",
            "Demographic composition":        "See subgroup decomposition table in Section 6",
        },
        "Performance metrics (test set)": {
            "Overall accuracy":     f"{overall_acc:.3f}",
            "Overall precision":    f"{overall_prec:.3f}",
            "Overall recall":       f"{overall_rec:.3f}",
            "Subgroup accuracy":    f"{acc_min:.3f} - {acc_max:.3f} (range across race/ethnicity groups)",
            "Subgroup miss rate":   f"{fnr_min:.3f} - {fnr_max:.3f} (FNR range across race/ethnicity groups)",
            "Fairness criterion":   "Equalized odds (equal TPR across race groups) -- chosen to reduce compounding of undercount",
        },
        "Fairness analysis": {
            "Demographic parity":        "NOT satisfied (prediction rates differ by group)",
            "Equalized odds":            "PARTIALLY satisfied (TPR similar; FPR varies)",
            "Calibration":               "NOT tested (requires probability calibration analysis)",
            "Criterion chosen":          "Equalized miss rates (TPR parity) prioritized",
            "Justification":             "False negatives for undercounted groups compound existing undercount",
        },
        "Limitations and risks": {
            "Primary risk":              "Higher miss rates for small subgroups due to limited training data",
            "Known failure mode":        "Performance degrades for households without fixed addresses",
            "Recommended human review":  "Cases with model confidence < 0.60 in small geographic areas",
        },
        "Governance": {
            "Documentation standard":    "Mitchell et al. (2019) Model Card format",
            "Regulatory basis":          "OMB Statistical Policy Directive 15; EO on Safe/Secure/Trustworthy AI",
            "Approved for production":   "NO -- requires division chief sign-off on fairness criterion choice",
            "Review schedule":           "Annual or when survey methodology changes",
        },
    }


def print_model_card(card: dict) -> None:
    """Print the model card in a readable structured format."""
    print("=" * 68)
    print("MODEL CARD: Nonresponse Propensity Model")
    print("Reference: Mitchell et al. (2019), FAccT")
    print("=" * 68)
    for section, fields in card.items():
        print(f"\n[[ {section} ]]")
        for field, value in fields.items():
            # Wrap long values
            if len(value) > 55:
                print(f"  {field}:")
                print(f"    {value}")
            else:
                print(f"  {field}: {value}")
    print()
    print("=" * 68)
    print("NOTE: This model card was generated from computed metrics.")
    print("A production model card requires human review and sign-off")
    print("before the system is deployed.")
    print("=" * 68)


if __name__ == "__main__":
    here = Path(__file__).parent

    df = load_predictions(here)
    card = build_model_card(df)
    print_model_card(card)
