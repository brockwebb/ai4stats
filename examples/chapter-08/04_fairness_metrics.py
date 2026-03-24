"""
04_fairness_metrics.py
======================
Computes four core fairness metrics by racial/ethnic group and produces
a 4-panel bar chart comparing them across groups.

WHY FOUR METRICS:
    Each metric has a different normative justification:
      - Demographic parity:  equal prediction rates (equal treatment)
      - True positive rate:  equal recall (equal error on actual positives)
      - False positive rate: equal false alarm rate (equal error on negatives)
      - Precision:           equal reliability of positive predictions

    No single metric captures fairness completely. The purpose of showing
    all four together is to make visible that they disagree -- and that
    the disagreement is not a data quality problem but a structural result
    (see 05_impossibility_theorem.py).

DEPENDENCIES:
    Loads ch08_test_predictions.csv produced by 01_dataset_and_model.py.
    Run 01_dataset_and_model.py first.

OUTPUTS:
    - Prints fairness metric table by group
    - Saves ../figures/ch08_fairness_metrics.png

REQUIREMENTS:
    Python 3.9+, numpy, pandas, matplotlib, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

RACE_GROUPS = [
    "White non-Hispanic",
    "Black non-Hispanic",
    "Hispanic",
    "Asian non-Hispanic",
    "Other",
]

GROUP_COLORS = ["#1976D2", "#F44336", "#FF9800", "#4CAF50", "#9C27B0"]
FIGURE_DPI = 120
MIN_GROUP_SIZE = 5   # skip groups too small for reliable estimates


def load_predictions(data_dir: Path) -> pd.DataFrame:
    """Load test-set predictions from 01_dataset_and_model.py."""
    path = data_dir / "ch08_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}\n"
            "Run 01_dataset_and_model.py first."
        )
    return pd.read_csv(path)


def compute_group_metrics(df_group: pd.DataFrame) -> dict | None:
    """
    Compute classification metrics for a single subgroup.

    Returns None if the group has fewer than MIN_GROUP_SIZE observations
    (estimates would be unreliable).
    """
    n = len(df_group)
    if n < MIN_GROUP_SIZE:
        return None

    y_true = df_group["y_true"].values
    y_pred = df_group["y_pred"].values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "n": n,
        "base_rate": float(y_true.mean()),
        "accuracy": float((tp + tn) / n),
        "positive_prediction_rate": float(y_pred.mean()),   # demographic parity
        "true_positive_rate": float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan,
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan,
    }


def compute_all_group_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fairness metrics for every racial/ethnic group."""
    rows = {}
    for group in RACE_GROUPS:
        subset = df[df["race"] == group]
        metrics = compute_group_metrics(subset)
        if metrics:
            rows[group] = metrics
    return pd.DataFrame(rows).T


def print_metrics_table(metrics_df: pd.DataFrame) -> None:
    """Print formatted fairness metric table."""
    print("Fairness metric decomposition by racial/ethnic group:")
    print("=" * 85)
    print()
    header = (
        f"{'Group':<28} {'N':>5} {'Base':>6} {'Acc':>6} "
        f"{'Pred%':>6} {'TPR':>6} {'FPR':>6} {'Prec':>6}"
    )
    print(header)
    print("-" * 85)
    for grp, row in metrics_df.iterrows():
        print(
            f"{grp:<28} {row['n']:>5.0f} {row['base_rate']:>6.3f} "
            f"{row['accuracy']:>6.3f} {row['positive_prediction_rate']:>6.3f} "
            f"{row['true_positive_rate']:>6.3f} {row['false_positive_rate']:>6.3f} "
            f"{row['precision']:>6.3f}"
        )
    print()
    print("Columns: Base=actual nonresponse rate, Pred%=demographic parity numerator,")
    print("         TPR=true positive rate (equalized odds), FPR=false positive rate,")
    print("         Prec=precision (predictive parity)")


def build_chart(metrics_df: pd.DataFrame, save_path: Path | None = None) -> None:
    """4-panel bar chart of fairness metrics across groups."""
    metrics_to_plot = [
        ("positive_prediction_rate", "Positive Prediction Rate\n(Demographic Parity)"),
        ("true_positive_rate",       "True Positive Rate\n(Equalized Odds -- component 1)"),
        ("false_positive_rate",      "False Positive Rate\n(Equalized Odds -- component 2)"),
        ("precision",                "Precision\n(Predictive Parity / Calibration)"),
    ]

    groups = list(metrics_df.index)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics_to_plot):
        values = [metrics_df.loc[g, metric] if g in metrics_df.index else np.nan
                  for g in groups]
        bars = ax.bar(
            range(len(groups)),
            values,
            color=GROUP_COLORS[:len(groups)],
            edgecolor="white",
            alpha=0.8,
        )
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.replace(" ", "\n") for g in groups], fontsize=7.5)
        ax.set_ylabel(label, fontsize=8.5)
        ax.set_ylim(0, 1.0)
        mean_val = np.nanmean(values)
        ax.axhline(
            y=mean_val,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            label=f"Mean: {mean_val:.2f}",
        )
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
        ax.legend(fontsize=8)

    plt.suptitle(
        "Four fairness metrics across racial/ethnic groups\n"
        "Dashed line = overall mean; equal bars = fairness criterion satisfied",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    plt.show()
    print()
    print("Observation: No metric is equal across all groups.")
    print("Different groups bear different error burdens under each definition of fairness.")


if __name__ == "__main__":
    here = Path(__file__).parent
    figures_dir = here.parent.parent / "figures"

    df = load_predictions(here)
    metrics_df = compute_all_group_metrics(df)
    print_metrics_table(metrics_df)
    build_chart(metrics_df, save_path=figures_dir / "ch08_fairness_metrics.png")
