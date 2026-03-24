"""
03_training_data_bias.py
========================
Compares the demographic composition of the full simulated dataset against
the respondent-only subset to illustrate training data bias (survivorship bias).

WHY THIS MATTERS:
    If you train a nonresponse model only on prior respondents, the training
    data systematically underrepresents groups with high nonresponse rates.
    The model then learns less about the populations it is most critical to
    predict correctly. This is survivorship bias applied to federal surveys.

    The representation ratio (respondent share / population share) makes the
    distortion quantitative. A ratio below 1.0 means the training-data-only
    view of the world is missing more of that group than of others.

DEPENDENCIES:
    Loads ch08_test_predictions.csv produced by 01_dataset_and_model.py.
    If that file is absent, regenerates the dataset from scratch.

OUTPUTS:
    - Prints representation ratios by group
    - Saves ../figures/ch08_training_data_bias.png

REQUIREMENTS:
    Python 3.9+, numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RACE_GROUPS = [
    "White non-Hispanic",
    "Black non-Hispanic",
    "Hispanic",
    "Asian non-Hispanic",
    "Other",
]

UNDERREP_THRESHOLD = 0.90   # representation ratio below this flags a group as underrepresented
FIGURE_DPI = 120


def load_or_build_dataset() -> pd.DataFrame:
    """Load the full synthetic dataset, regenerating if needed."""
    here = Path(__file__).parent
    predictions_path = here / "ch08_test_predictions.csv"

    if predictions_path.exists():
        # The CSV only has test rows; rebuild the full dataset for population comparison
        pass  # fall through to regenerate

    # Regenerate full dataset using same parameters as 01_dataset_and_model.py
    import sys
    sys.path.insert(0, str(here))
    from importlib import import_module
    mod = import_module("01_dataset_and_model")
    return mod.build_dataset()


def compute_representation_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compare respondent composition to population composition."""
    respondents = df[df["nonresponse"] == 0]
    rows = []
    for group in RACE_GROUPS:
        pop_frac = (df["race"] == group).mean()
        resp_frac = (respondents["race"] == group).mean()
        ratio = resp_frac / pop_frac if pop_frac > 0 else np.nan
        rows.append({
            "group": group,
            "population_share": pop_frac,
            "respondent_share": resp_frac,
            "representation_ratio": ratio,
            "underrepresented": ratio < UNDERREP_THRESHOLD,
        })
    return pd.DataFrame(rows).set_index("group")


def build_chart(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """Side-by-side bar chart: population composition vs. respondent composition."""
    respondents = df[df["nonresponse"] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: full population
    pop_counts = pd.Series(
        {g: (df["race"] == g).mean() for g in RACE_GROUPS}
    )
    axes[0].bar(
        pop_counts.index,
        pop_counts.values,
        color="#1976D2",
        edgecolor="white",
        alpha=0.8,
    )
    axes[0].set_title("Full dataset composition\n(population)", fontsize=10)
    axes[0].set_ylabel("Fraction of dataset")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylim(0, 0.75)

    # Panel 2: respondents only
    resp_counts = pd.Series(
        {g: (respondents["race"] == g).mean() for g in RACE_GROUPS}
    )
    axes[1].bar(
        resp_counts.index,
        resp_counts.values,
        color="#F44336",
        edgecolor="white",
        alpha=0.8,
    )
    axes[1].set_title(
        "Respondent-only composition\n(training data if trained on respondents only)",
        fontsize=10,
    )
    axes[1].set_ylabel("Fraction of respondents")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylim(0, 0.75)

    plt.suptitle(
        "Training data bias: respondent-only training underrepresents high-nonresponse groups",
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    plt.show()


def print_representation_table(ratios: pd.DataFrame) -> None:
    """Print representation ratios with flags for underrepresented groups."""
    print("Representation ratio (respondent share / population share):")
    print("< 1.0 means this group is underrepresented in respondent-only training data")
    print()
    print(f"{'Group':<28}  {'Pop. share':>10}  {'Resp. share':>11}  {'Ratio':>7}  {'Flag'}")
    print("-" * 75)
    for group, row in ratios.iterrows():
        flag = "<-- underrepresented" if row["underrepresented"] else ""
        print(
            f"{group:<28}  {row['population_share']:>10.3f}  "
            f"{row['respondent_share']:>11.3f}  {row['representation_ratio']:>7.3f}  {flag}"
        )


if __name__ == "__main__":
    here = Path(__file__).parent
    figures_dir = here.parent.parent / "figures"

    df = load_or_build_dataset()
    ratios = compute_representation_ratios(df)
    print_representation_table(ratios)
    build_chart(df, save_path=figures_dir / "ch08_training_data_bias.png")
