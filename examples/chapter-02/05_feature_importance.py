"""
Chapter 2 -- Feature Importance
Computes and compares coefficient importance and permutation importance for the
nonresponse classification model.

Two complementary methods are shown:

  Coefficient importance
    Logistic regression assigns a coefficient to each feature. The magnitude
    indicates relative importance *within this model*, and the sign shows
    direction. Assumes features are on comparable scales; raw coefficients
    can be misleading when features have very different ranges.

  Permutation importance
    Measures how much AUC drops when one feature's values are randomly
    shuffled (destroying its predictive signal). A large drop means the model
    relies on that feature. A near-zero drop means the feature contributes
    little. Works for any model, not just linear ones.

Limitation of both methods: when two features are highly correlated, their
importance scores split between them. Neither method tells you about causation --
only association within this model on this dataset.

Run: python 05_feature_importance.py
     (run 01_dataset_setup.py first)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE    = Path(__file__).resolve().parents[2] / "data" / "synthetic_survey_ch02.csv"
FIGURE_DIR   = Path(__file__).resolve().parents[2] / "figures"
FEATURES_CLF = ["age", "education_years", "urban", "contact_attempts", "prior_response"]
TARGET_CLF   = "responded"
TEST_SIZE    = 0.20
SEED         = 42
PERM_REPEATS = 20    # number of shuffle repetitions per feature


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run 01_dataset_setup.py first."
        )
    df = pd.read_csv(path)
    return df[FEATURES_CLF], df[TARGET_CLF]


def fit_model(
    X: pd.DataFrame, y: pd.Series
) -> tuple[LogisticRegression, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Fit a logistic regression classifier and return the model plus train/test splits.
    Uses StratifiedKFold split to preserve class balance.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    clf = LogisticRegression(max_iter=500, random_state=SEED)
    clf.fit(X_tr, y_tr)
    return clf, X_tr, X_te, y_tr, y_te


def coefficient_importance(
    clf: LogisticRegression,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return a DataFrame of logistic regression coefficients sorted by magnitude."""
    coefs = clf.coef_[0]
    df = pd.DataFrame({
        "feature":     feature_names,
        "coefficient": coefs,
        "abs_coef":    np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)
    return df


def perm_importance(
    clf: LogisticRegression,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    feature_names: list[str],
    n_repeats: int = PERM_REPEATS,
) -> pd.DataFrame:
    """Return a DataFrame of permutation importances sorted descending."""
    result = permutation_importance(
        clf, X_te, y_te,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=SEED,
    )
    df = pd.DataFrame({
        "feature":         feature_names,
        "importance_mean": result.importances_mean,
        "importance_std":  result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return df


def plot_comparison(
    coef_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a side-by-side comparison of coefficient and permutation importance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Coefficient plot
    colors = ["steelblue" if c > 0 else "salmon" for c in coef_df["coefficient"]]
    ax1.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel("Coefficient (log-odds)")
    ax1.set_title("Coefficient importance\n(logistic regression)")

    # Permutation importance plot (sorted ascending for barh readability)
    perm_sorted = perm_df.sort_values("importance_mean", ascending=True)
    ax2.barh(
        perm_sorted["feature"],
        perm_sorted["importance_mean"],
        xerr=perm_sorted["importance_std"],
        capsize=4,
        color="steelblue",
    )
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_xlabel("Mean AUC decrease")
    ax2.set_title("Permutation importance\n(test set, 20 repeats)")

    fig.suptitle("Feature importance: two views of the same model", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_path}")


def main() -> None:
    X, y = load_data(DATA_FILE)
    clf, X_tr, X_te, y_tr, y_te = fit_model(X, y)

    # ------------------------------------------------------------------
    # Coefficient importance
    # ------------------------------------------------------------------
    print("=" * 60)
    print("COEFFICIENT IMPORTANCE (logistic regression)")
    print("=" * 60)
    coef_df = coefficient_importance(clf, FEATURES_CLF)
    print("\nCoefficients sorted by magnitude:")
    print(coef_df[["feature", "coefficient", "abs_coef"]].to_string(index=False))
    print("\nInterpretation:")
    print("  Positive: feature increases log-odds of responding.")
    print("  Negative: feature decreases log-odds of responding.")
    print("  Magnitude assumes features are on comparable scales.")

    # ------------------------------------------------------------------
    # Permutation importance
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PERMUTATION IMPORTANCE (model-agnostic, test set)")
    print("=" * 60)
    perm_df = perm_importance(clf, X_te, y_te, FEATURES_CLF)
    print("\nMean AUC decrease when each feature is shuffled:")
    print(perm_df.to_string(index=False))
    print("\nInterpretation:")
    print("  Large decrease: the model depends heavily on this feature.")
    print("  Near zero: the model barely uses this feature.")
    print("  Negative (rare): shuffling accidentally improved score -- "
          "feature may be adding noise.")

    # ------------------------------------------------------------------
    # Comparison and caveats
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON AND CAVEATS")
    print("=" * 60)
    print("\nRanking by coefficient magnitude:")
    for _, row in coef_df.iterrows():
        print(f"  {row['feature']:<20} coef={row['coefficient']:+.3f}")
    print("\nRanking by permutation importance:")
    for _, row in perm_df.iterrows():
        print(f"  {row['feature']:<20} mean_decrease={row['importance_mean']:.3f} "
              f"(+/-{row['importance_std']:.3f})")
    print("\nKey caveat: when features are correlated, importance splits across them.")
    print("Neither method implies causation -- only association within this model.")

    # Cross-validation to confirm the model is sound before reporting importance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    auc_scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
    print(f"\nModel CV AUC (5-fold stratified): {auc_scores.mean():.3f} "
          f"+/- {auc_scores.std():.3f}")
    print("Feature importance is only meaningful for a model with reasonable CV performance.")

    plot_comparison(coef_df, perm_df, FIGURE_DIR / "ch02_feature_importance.png")


if __name__ == "__main__":
    main()
