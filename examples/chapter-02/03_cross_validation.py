"""
Chapter 2 -- Cross-Validation Strategies
Demonstrates KFold, StratifiedKFold, and GroupKFold on the synthetic survey dataset.

Three CV strategies are compared:
  - KFold: basic k-fold, no special treatment of classes or groups
  - StratifiedKFold: preserves the class ratio in each fold (preferred for binary outcomes)
  - GroupKFold: keeps all household members in the same fold (required for clustered data)

GroupKFold typically produces a lower (more conservative) AUC estimate because it prevents
within-household information from leaking across folds. That lower number is more honest:
it approximates how the model performs when deployed on genuinely new households.

Also shows cross_validate for multi-metric evaluation and inspecting train vs test scores
as an overfitting diagnostic.

Run: python 03_cross_validation.py
     (run 01_dataset_setup.py first)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold,
    cross_val_score, cross_validate,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE    = Path(__file__).resolve().parents[2] / "data" / "synthetic_survey_ch02.csv"
FEATURES_CLF = ["age", "education_years", "urban", "contact_attempts", "prior_response"]
TARGET_CLF   = "responded"
GROUP_COL    = "household_id"
N_SPLITS     = 5
SEED         = 42


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load features, target, and group labels from CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run 01_dataset_setup.py first."
        )
    df = pd.read_csv(path)
    return df[FEATURES_CLF], df[TARGET_CLF], df[GROUP_COL]


def summarize_scores(name: str, scores: np.ndarray) -> None:
    """Print a compact summary of per-fold CV scores."""
    print(f"\n{name}")
    print(f"  Per-fold AUC: {scores.round(3)}")
    print(f"  Mean: {scores.mean():.3f}   Std: {scores.std():.3f}")
    print(f"  Range: [{scores.min():.3f}, {scores.max():.3f}]")


def main() -> None:
    X, y, groups = load_data(DATA_FILE)
    clf = LogisticRegression(max_iter=500, random_state=SEED)

    # ------------------------------------------------------------------
    # Section 1: Three CV strategies compared
    # ------------------------------------------------------------------
    print("=" * 60)
    print("SECTION 1: KFold vs StratifiedKFold vs GroupKFold")
    print("=" * 60)

    kf  = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    gkf = GroupKFold(n_splits=N_SPLITS)

    scores_kf  = cross_val_score(clf, X, y, cv=kf,  scoring="roc_auc")
    scores_skf = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
    scores_gkf = cross_val_score(clf, X, y, cv=gkf, groups=groups, scoring="roc_auc")

    summarize_scores("KFold (5-fold)", scores_kf)
    summarize_scores("StratifiedKFold (5-fold)", scores_skf)
    summarize_scores("GroupKFold (5-fold)", scores_gkf)

    print(f"\nDifference (KFold - GroupKFold): "
          f"{scores_kf.mean() - scores_gkf.mean():.3f} AUC points")
    print("This gap reflects within-household information leakage in KFold.")

    # ------------------------------------------------------------------
    # Section 2: Multi-metric CV with train/test score comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 2: Multi-metric CV (StratifiedKFold)")
    print("=" * 60)

    cv_results = cross_validate(
        clf, X, y, cv=skf,
        scoring=["roc_auc", "f1", "accuracy"],
        return_train_score=True,
    )

    metrics = ["roc_auc", "f1", "accuracy"]
    rows = []
    for m in metrics:
        rows.append({
            "metric":        m,
            "train_mean":    cv_results[f"train_{m}"].mean(),
            "test_mean":     cv_results[f"test_{m}"].mean(),
            "test_std":      cv_results[f"test_{m}"].std(),
            "train-test gap": (cv_results[f"train_{m}"] - cv_results[f"test_{m}"]).mean(),
        })
    summary = pd.DataFrame(rows).set_index("metric")
    print("\nCross-validation summary (5-fold stratified):")
    print(summary.round(3).to_string())
    print("\nA large train-test gap signals overfitting.")
    print("Both scores low signals underfitting (needs better features or model).")

    # ------------------------------------------------------------------
    # Section 3: Why stratification matters -- class balance per fold
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SECTION 3: Class balance across folds (KFold vs StratifiedKFold)")
    print("=" * 60)

    print("\nKFold -- response rate per fold:")
    for fold_idx, (_, test_idx) in enumerate(kf.split(X, y)):
        rate = y.iloc[test_idx].mean()
        print(f"  Fold {fold_idx + 1}: {rate:.3f}")

    print("\nStratifiedKFold -- response rate per fold:")
    for fold_idx, (_, test_idx) in enumerate(skf.split(X, y)):
        rate = y.iloc[test_idx].mean()
        print(f"  Fold {fold_idx + 1}: {rate:.3f}")

    print("\nStratifiedKFold keeps each fold's class ratio close to the overall rate.")
    print(f"Overall response rate: {y.mean():.3f}")


if __name__ == "__main__":
    main()
