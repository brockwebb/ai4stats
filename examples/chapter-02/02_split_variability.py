"""
Chapter 2 -- Split Variability
Demonstrates that a single train/test split produces an unreliable AUC estimate.

The same logistic regression model is fit 30 times, each with a different random
split of the same data. The resulting range of AUC values shows that the "performance"
of a model depends substantially on which records land in the test set -- not just on
the model itself.

Key takeaway: if you report only one split, you may be reporting a lucky (or unlucky)
draw. Cross-validation, shown in 03_cross_validation.py, averages over many such splits
to produce a more stable estimate.

Run: python 02_split_variability.py
     (run 01_dataset_setup.py first to generate the dataset)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE    = Path(__file__).resolve().parents[2] / "data" / "synthetic_survey_ch02.csv"
FIGURE_DIR   = Path(__file__).resolve().parents[2] / "figures"
FIGURE_FILE  = FIGURE_DIR / "ch02_split_variability.png"
FEATURES_CLF = ["age", "education_years", "urban", "contact_attempts", "prior_response"]
TARGET_CLF   = "responded"
TEST_SIZE    = 0.20
N_SEEDS      = 30


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load classification features and target from CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run 01_dataset_setup.py first."
        )
    df = pd.read_csv(path)
    return df[FEATURES_CLF], df[TARGET_CLF]


def compute_auc_by_seed(
    X: pd.DataFrame,
    y: pd.Series,
    n_seeds: int = N_SEEDS,
    test_size: float = TEST_SIZE,
) -> list[float]:
    """
    Fit logistic regression on n_seeds different random splits and return AUC per split.

    Parameters
    ----------
    X : DataFrame of features
    y : binary target series
    n_seeds : number of random seeds to try
    test_size : fraction of data held out as test

    Returns
    -------
    list of AUC values, one per seed
    """
    aucs = []
    for seed in range(n_seeds):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        clf = LogisticRegression(max_iter=500, random_state=seed)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, prob))
    return aucs


def plot_variability(aucs: list[float], output_path: Path) -> None:
    """Save a chart of AUC values across seeds with the mean highlighted."""
    mean_auc = np.mean(aucs)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(aucs, "o-", alpha=0.7, color="steelblue", label="AUC (single split)")
    ax.axhline(mean_auc, color="crimson", linestyle="--",
               label=f"Mean AUC = {mean_auc:.3f}")
    ax.set_xlabel("Random seed")
    ax.set_ylabel("AUC")
    ax.set_title("AUC varies across 30 random splits of the same data\n"
                 "(same model, same dataset -- only the split changes)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {output_path}")


def main() -> None:
    X, y = load_data(DATA_FILE)

    # Baseline: the canonical seed-42 split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    clf_base = LogisticRegression(max_iter=500, random_state=42)
    clf_base.fit(X_tr, y_tr)
    auc_seed42 = roc_auc_score(y_te, clf_base.predict_proba(X_te)[:, 1])
    print(f"AUC with seed=42 (canonical split): {auc_seed42:.3f}")

    # Variability across 30 seeds
    aucs = compute_auc_by_seed(X, y)
    print(f"\nAUC across {N_SEEDS} random splits:")
    print(f"  Mean:  {np.mean(aucs):.3f}")
    print(f"  Std:   {np.std(aucs):.3f}")
    print(f"  Range: [{min(aucs):.3f}, {max(aucs):.3f}]")
    print(f"\nThe range ({max(aucs) - min(aucs):.3f}) is pure split randomness, not model variation.")
    print("Cross-validation collapses this by averaging over all possible splits.")

    plot_variability(aucs, FIGURE_FILE)


if __name__ == "__main__":
    main()
