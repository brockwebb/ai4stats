"""
06_subgroup_decomposition.py
=============================
Implements a general-purpose subgroup_decomposition function and applies it
to decompose model performance by race/ethnicity and by income quintile.

WHY SUBGROUP DECOMPOSITION:
    Overall accuracy is a weighted average that conceals disparities. A model
    that is 94% accurate overall can be 78% accurate for the subgroup that
    matters most to avoid undercounting. The subgroup decomposition makes
    the distribution of errors visible.

    The miss rate (false negative rate) is the metric most directly connected
    to undercount: a missed nonrespondent is a person the model failed to
    flag for targeted follow-up, increasing the probability they remain
    uncounted.

DEPENDENCIES:
    Loads ch08_test_predictions.csv produced by 01_dataset_and_model.py.
    Run 01_dataset_and_model.py first.

OUTPUTS:
    - Prints decomposition tables by race/ethnicity and income quintile
    - Saves ../figures/ch08_subgroup_decomposition.png

REQUIREMENTS:
    Python 3.9+, numpy, pandas, matplotlib, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path

RACE_GROUPS = [
    "White non-Hispanic",
    "Black non-Hispanic",
    "Hispanic",
    "Asian non-Hispanic",
    "Other",
]
INCOME_GROUPS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
MIN_GROUP_SIZE = 5
FIGURE_DPI = 120


def load_predictions(data_dir: Path) -> pd.DataFrame:
    """Load test-set predictions from 01_dataset_and_model.py."""
    path = data_dir / "ch08_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}\n"
            "Run 01_dataset_and_model.py first."
        )
    return pd.read_csv(path)


def subgroup_decomposition(
    df_results: pd.DataFrame,
    group_col: str,
    groups: list[str],
) -> pd.DataFrame:
    """
    Compute accuracy metrics for each group in `groups`.

    Parameters
    ----------
    df_results : DataFrame with columns y_true, y_pred, y_prob, and group_col
    group_col  : Column name to group by (e.g., "race", "income_quintile")
    groups     : Ordered list of group labels to include

    Returns
    -------
    DataFrame indexed by group with accuracy, TPR, FPR, FNR (miss rate), precision
    """
    rows = []
    for group in groups:
        subset = df_results[df_results[group_col] == group]
        if len(subset) < MIN_GROUP_SIZE:
            continue

        y_true = subset["y_true"].values
        y_pred = subset["y_pred"].values

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        n = len(subset)
        base_rate = float(y_true.mean())
        accuracy = float((tp + tn) / n)
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan
        fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else np.nan   # miss rate
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan

        rows.append({
            "group": group,
            "n": n,
            "base_rate": base_rate,
            "accuracy": accuracy,
            "TPR (recall)": tpr,
            "FPR": fpr,
            "FNR (miss rate)": fnr,
            "precision": prec,
        })

    return pd.DataFrame(rows).set_index("group")


def build_chart(
    decomp_race: pd.DataFrame,
    decomp_income: pd.DataFrame,
    overall_accuracy: float,
    overall_fnr: float,
    save_path: Path | None = None,
) -> None:
    """3-panel chart: accuracy by race, miss rate by race, accuracy by income."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: accuracy by race/ethnicity
    groups_sorted = decomp_race.sort_values("accuracy").index
    acc_values = decomp_race.loc[groups_sorted, "accuracy"]
    bars = axes[0].barh(groups_sorted, acc_values, color="#1976D2", edgecolor="white", alpha=0.8)
    axes[0].axvline(
        overall_accuracy, color="black", linestyle="--",
        linewidth=1.0, label=f"Overall: {overall_accuracy:.3f}"
    )
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Accuracy by racial/ethnic group", fontsize=10)
    axes[0].set_xlim(0.5, 1.0)
    axes[0].legend(fontsize=8)
    for bar, val in zip(bars, acc_values):
        axes[0].text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=8)

    # Panel 2: miss rate (FNR) by race/ethnicity
    groups_sorted2 = decomp_race.sort_values("FNR (miss rate)", ascending=False).index
    fnr_values = decomp_race.loc[groups_sorted2, "FNR (miss rate)"]
    bars2 = axes[1].barh(groups_sorted2, fnr_values, color="#F44336", edgecolor="white", alpha=0.8)
    axes[1].axvline(
        overall_fnr, color="black", linestyle="--",
        linewidth=1.0, label=f"Overall: {overall_fnr:.3f}"
    )
    axes[1].set_xlabel("False Negative Rate (miss rate)")
    axes[1].set_title("Miss rate by racial/ethnic group\n(nonrespondents missed by model)", fontsize=10)
    axes[1].legend(fontsize=8)
    for bar, val in zip(bars2, fnr_values):
        if not np.isnan(val):
            axes[1].text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center", fontsize=8)

    # Panel 3: accuracy by income quintile
    inc_acc = decomp_income["accuracy"]
    bars3 = axes[2].bar(
        INCOME_GROUPS,
        [inc_acc.get(q, np.nan) for q in INCOME_GROUPS],
        color="#FF9800",
        edgecolor="white",
        alpha=0.8,
    )
    axes[2].axhline(overall_accuracy, color="black", linestyle="--",
                    linewidth=1.0, label="Overall")
    axes[2].set_xlabel("Income quintile (Q1=lowest)")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Accuracy by income quintile", fontsize=10)
    axes[2].set_ylim(0.5, 1.0)
    axes[2].legend(fontsize=8)
    for bar, val in zip(bars3, [inc_acc.get(q, np.nan) for q in INCOME_GROUPS]):
        if not np.isnan(val):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.suptitle(
        "Subgroup accuracy decomposition: overall accuracy conceals disparities",
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    plt.show()


if __name__ == "__main__":
    here = Path(__file__).parent
    figures_dir = here.parent.parent / "figures"

    df = load_predictions(here)

    overall_accuracy = accuracy_score(df["y_true"], df["y_pred"])
    overall_fnr = 1.0 - (
        df[df["y_true"] == 1]["y_pred"].sum() / (df["y_true"] == 1).sum()
    )

    print("Subgroup decomposition by race/ethnicity:")
    print("=" * 85)
    decomp_race = subgroup_decomposition(df, "race", RACE_GROUPS)
    print(decomp_race.round(3).to_string())

    max_acc_group = decomp_race["accuracy"].idxmax()
    min_acc_group = decomp_race["accuracy"].idxmin()
    max_acc = decomp_race["accuracy"].max()
    min_acc = decomp_race["accuracy"].min()
    print()
    print(f"Accuracy gap: {max_acc_group} ({max_acc:.3f}) vs. {min_acc_group} ({min_acc:.3f})")
    print(f"  Difference: {max_acc - min_acc:.3f} ({(max_acc - min_acc)*100:.1f} percentage points)")

    print()
    print("Subgroup decomposition by income quintile:")
    print("=" * 85)
    decomp_income = subgroup_decomposition(df, "income_quintile", INCOME_GROUPS)
    print(
        decomp_income[["n", "base_rate", "accuracy", "TPR (recall)", "FNR (miss rate)"]].round(3).to_string()
    )

    build_chart(
        decomp_race,
        decomp_income,
        overall_accuracy,
        overall_fnr,
        save_path=figures_dir / "ch08_subgroup_decomposition.png",
    )
