"""
07_compounding_effect.py
========================
Shows how a high base rate of nonresponse and a high model miss rate compound
to produce a disproportionate coverage gap for specific groups.

WHY COMPOUNDING MATTERS:
    The groups most likely to be nonrespondents (high base rate) are also
    the groups for which the model tends to have higher miss rates, because
    the model has less training data for those groups (survivorship bias).

    The compound risk -- probability of being a nonrespondent AND being missed
    by the model -- is not the sum of these two rates. It is their product.
    A group with 40% nonresponse and 35% miss rate has a 14% compound risk
    of being both a true nonrespondent and going untargeted for follow-up.

    This calculation translates model error rates into an operational
    consequence: uncounted people.

DEPENDENCIES:
    Loads ch08_test_predictions.csv produced by 01_dataset_and_model.py.
    Run 01_dataset_and_model.py first.

OUTPUTS:
    - Prints compounding table: base rate, miss rate, compound risk by group
    - Prints step-by-step cascade narrative

REQUIREMENTS:
    Python 3.9+, numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
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


def compute_compound_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each group, compute base rate, miss rate (FNR), and compound risk.

    Compound risk = base_rate * miss_rate
    = P(nonrespondent) * P(missed | nonrespondent)
    = P(nonrespondent AND missed by model)
    """
    rows = []
    for group in RACE_GROUPS:
        subset = df[df["race"] == group]
        if len(subset) < MIN_GROUP_SIZE:
            continue

        y_true = subset["y_true"].values
        y_pred = subset["y_pred"].values

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        base_rate = float(y_true.mean())
        fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else np.nan
        compound = base_rate * fnr if not np.isnan(fnr) else np.nan

        rows.append({
            "group": group,
            "n": len(subset),
            "base_rate": base_rate,
            "miss_rate_FNR": fnr,
            "compound_risk": compound,
        })

    return pd.DataFrame(rows).set_index("group").sort_values("compound_risk", ascending=False)


def print_compounding_table(compound_df: pd.DataFrame) -> None:
    """Print the compounding table with the highest-risk group flagged."""
    max_compound = compound_df["compound_risk"].max()

    print("Base rate x miss rate compounding by racial/ethnic group:")
    print("=" * 72)
    print()
    print(f"{'Group':<28} {'Base Rate':>10} {'Miss Rate':>10} {'Compound Risk':>14}  {'Flag'}")
    print("-" * 72)
    for group, row in compound_df.iterrows():
        flag = "<-- highest compound risk" if row["compound_risk"] == max_compound else ""
        print(
            f"{group:<28} {row['base_rate']:>10.3f} {row['miss_rate_FNR']:>10.3f} "
            f"{row['compound_risk']:>14.3f}  {flag}"
        )

    print()
    print("Interpretation:")
    print("  Compound risk = P(true nonrespondent) * P(model misses them)")
    print("  = probability this group member is both uncaptured by follow-up")
    print("  targeting AND a true nonrespondent.")
    print()
    print("  Groups with high base rates tend to also have higher miss rates")
    print("  because the model has less training data for them (survivorship")
    print("  bias). The two rates multiply, not add.")


def print_cascade_narrative(compound_df: pd.DataFrame) -> None:
    """Print the 7-step cascade showing how model errors compound operationally."""
    highest_group = compound_df["compound_risk"].idxmax()
    base = compound_df.loc[highest_group, "base_rate"]
    fnr = compound_df.loc[highest_group, "miss_rate_FNR"]
    compound = compound_df.loc[highest_group, "compound_risk"]

    print()
    print("The compounding cascade in nonresponse adjustment:")
    print("=" * 60)
    print(f"  Using group: {highest_group}")
    print()
    print(f"  Step 1: Underlying nonresponse rate is higher ({base:.1%})")
    print(f"  Step 2: Model miss rate (FNR) is higher for this group ({fnr:.1%})")
    print(f"  Step 3: Compound risk = {base:.1%} x {fnr:.1%} = {compound:.1%}")
    print("  Step 4: Follow-up resources not targeted to this group")
    print("  Step 5: Lower follow-up response rate")
    print("  Step 6: Post-survey weights must compensate harder")
    print("  Step 7: Higher variance in estimates for this group")
    print()
    print("  The model error does not stand alone. It compounds the underlying")
    print("  nonresponse problem. Each step makes the next harder to correct.")


if __name__ == "__main__":
    here = Path(__file__).parent

    df = load_predictions(here)
    compound_df = compute_compound_risk(df)
    print_compounding_table(compound_df)
    print_cascade_narrative(compound_df)
