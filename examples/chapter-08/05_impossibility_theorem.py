"""
05_impossibility_theorem.py
============================
Demonstrates the Chouldechova/Kleinberg impossibility theorem by sweeping
decision thresholds for two groups with different base rates (Hispanic vs.
Asian non-Hispanic) and showing that TPR-parity and precision-parity cannot
be simultaneously achieved.

WHY THIS MATTERS:
    The impossibility theorem is not a practical limitation -- it is a
    mathematical proof. When base rates differ across groups, any model
    that achieves calibration (equal precision) must have unequal error
    rates (violating equalized odds), and vice versa.

    Showing this visually -- across every possible threshold, not just the
    default 0.5 -- drives home that the conflict is inescapable. The policy
    question is which type of error to prioritize, and that is a governance
    decision, not a statistical one.

DEPENDENCIES:
    Loads ch08_test_predictions.csv produced by 01_dataset_and_model.py.
    Run 01_dataset_and_model.py first.

OUTPUTS:
    - Prints base rate comparison for the two groups
    - Saves ../figures/ch08_impossibility.png (3-panel threshold sweep)

REQUIREMENTS:
    Python 3.9+, numpy, pandas, matplotlib, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

# Groups chosen because they show the largest base rate difference in our data
GROUP_A = "Hispanic"
GROUP_B = "Asian non-Hispanic"
GROUP_COLORS = {GROUP_A: "#F44336", GROUP_B: "#1976D2"}

THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
N_THRESHOLDS = 50
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


def metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Return TPR, FPR, and precision at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {"tpr": tpr, "fpr": fpr, "precision": prec}


def sweep_thresholds(df: pd.DataFrame) -> dict:
    """Compute metrics across all thresholds for both groups."""
    thresholds = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, N_THRESHOLDS)
    results = {
        GROUP_A: {"threshold": [], "tpr": [], "fpr": [], "precision": []},
        GROUP_B: {"threshold": [], "tpr": [], "fpr": [], "precision": []},
    }

    for thresh in thresholds:
        for group in [GROUP_A, GROUP_B]:
            subset = df[df["race"] == group]
            if len(subset) < 5:
                continue
            m = metrics_at_threshold(
                subset["y_true"].values,
                subset["y_prob"].values,
                thresh,
            )
            results[group]["threshold"].append(thresh)
            results[group]["tpr"].append(m["tpr"])
            results[group]["fpr"].append(m["fpr"])
            results[group]["precision"].append(m["precision"])

    return results


def build_chart(results: dict, save_path: Path | None = None) -> None:
    """3-panel plot: TPR, FPR, precision vs. threshold for both groups."""
    metric_labels = [
        ("tpr",       "True Positive Rate\n(Equalized Odds target 1)"),
        ("fpr",       "False Positive Rate\n(Equalized Odds target 2)"),
        ("precision", "Precision\n(Predictive Parity target)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, (metric, label) in zip(axes, metric_labels):
        for group in [GROUP_A, GROUP_B]:
            r = results[group]
            ax.plot(
                r["threshold"],
                r[metric],
                color=GROUP_COLORS[group],
                linewidth=2,
                label=group,
            )
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel(label, fontsize=9)
        ax.set_xlim(THRESHOLD_MIN, THRESHOLD_MAX)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    plt.suptitle(
        f"Impossibility theorem: {GROUP_A} vs. {GROUP_B}\n"
        "At every threshold, at least one fairness criterion is violated for one group",
        fontsize=9.5,
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    plt.show()


def print_summary(df: pd.DataFrame) -> None:
    """Print base rate comparison and theorem statement."""
    print("The Impossibility Theorem (Chouldechova 2017; Kleinberg et al. 2016)")
    print("=" * 68)
    print()
    print("Theorem: When base rates (actual positive rates) differ across groups,")
    print("it is IMPOSSIBLE to simultaneously satisfy:")
    print("  (1) Calibration: equal precision across groups")
    print("  (2) Equalized odds: equal TPR and FPR across groups")
    print()
    base_a = df[df["race"] == GROUP_A]["y_true"].mean()
    base_b = df[df["race"] == GROUP_B]["y_true"].mean()
    print(f"Base rates in this dataset:")
    print(f"  {GROUP_A:<28}: {base_a:.1%}")
    print(f"  {GROUP_B:<28}: {base_b:.1%}")
    print()
    print(f"The {abs(base_a - base_b):.1%} gap between these groups means no single")
    print("threshold can equalize both TPR and precision simultaneously.")
    print()
    print("This is not a model failure. It is a mathematical constraint.")
    print("The choice of which criterion to optimize is a governance decision.")


if __name__ == "__main__":
    here = Path(__file__).parent
    figures_dir = here.parent.parent / "figures"

    df = load_predictions(here)
    print_summary(df)
    results = sweep_thresholds(df)
    build_chart(results, save_path=figures_dir / "ch08_impossibility.png")
