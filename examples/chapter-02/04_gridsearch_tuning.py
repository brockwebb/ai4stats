"""
Chapter 2 -- Hyperparameter Tuning with GridSearchCV
Demonstrates manual alpha search and automated GridSearchCV for Ridge regression
and logistic regression, with a strict train/test discipline.

The central rule: the test set is used exactly once, at the very end, to report
final performance. All hyperparameter choices happen inside cross-validation on the
training set. Peeking at test performance during tuning inflates the reported metric
(data leakage).

Workflow demonstrated:
  1. Split data into train (80%) and test (20%).
  2. Manual alpha search on training set using 5-fold CV.
  3. Automated GridSearchCV on training set.
  4. Evaluate the winning model once on the test set.
  5. Repeat for logistic regression with the C parameter.

Run: python 04_gridsearch_tuning.py
     (run 01_dataset_setup.py first)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import (
    KFold, StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
)
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE    = Path(__file__).resolve().parents[2] / "data" / "synthetic_survey_ch02.csv"
FIGURE_DIR   = Path(__file__).resolve().parents[2] / "figures"
FEATURES_REG = ["age", "education_years", "hours_per_week", "urban"]
FEATURES_CLF = ["age", "education_years", "urban", "contact_attempts", "prior_response"]
TARGET_REG   = "income"
TARGET_CLF   = "responded"
TEST_SIZE    = 0.20
SEED         = 42

RIDGE_ALPHAS = [10, 50, 100, 200, 500, 1000, 2000, 5000]
LOGR_C_VALUES = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run 01_dataset_setup.py first."
        )
    return pd.read_csv(path)


def manual_alpha_search(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    alphas: list[int | float],
    n_splits: int = 5,
) -> tuple[int | float, list[float]]:
    """
    Search Ridge alphas via cross-validation on the training set.

    Returns
    -------
    best_alpha : the alpha with the lowest mean CV MAE
    mean_maes  : list of mean CV MAE for each alpha (same order as alphas)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    mean_maes = []
    for a in alphas:
        scores = cross_val_score(
            Ridge(alpha=a), X_tr, y_tr,
            cv=kf, scoring="neg_mean_absolute_error"
        )
        mean_maes.append(-scores.mean())
    best_alpha = alphas[int(np.argmin(mean_maes))]
    return best_alpha, mean_maes


def plot_alpha_search(
    alphas: list[int | float],
    mean_maes: list[float],
    best_alpha: int | float,
    output_path: Path,
) -> None:
    """Save a semilog plot of CV MAE vs alpha."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(alphas, mean_maes, "o-", color="steelblue")
    ax.axvline(best_alpha, linestyle=":", color="crimson",
               label=f"Best alpha = {best_alpha}")
    ax.set_xlabel("Alpha (log scale)")
    ax.set_ylabel("CV Mean Absolute Error ($)")
    ax.set_title("Ridge regularization: choosing alpha via cross-validation")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {output_path}")


def plot_logreg_search(grid: GridSearchCV, output_path: Path) -> None:
    """Save a semilog plot of CV AUC vs C for logistic regression."""
    cv_df = pd.DataFrame(grid.cv_results_)
    c_vals = cv_df["param_C"].astype(float)
    means  = cv_df["mean_test_score"]
    stds   = cv_df["std_test_score"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(c_vals, means, "o-", color="steelblue", label="Mean CV AUC")
    ax.fill_between(c_vals, means - stds, means + stds, alpha=0.2, color="steelblue")
    ax.axvline(grid.best_params_["C"], linestyle=":", color="crimson",
               label=f"Best C = {grid.best_params_['C']}")
    ax.set_xlabel("C (log scale, inverse regularization strength)")
    ax.set_ylabel("CV AUC")
    ax.set_title("Logistic regression: choosing C via cross-validation")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {output_path}")


def main() -> None:
    df = load_data(DATA_FILE)

    # ------------------------------------------------------------------
    # Ridge regression: tune alpha
    # ------------------------------------------------------------------
    print("=" * 60)
    print("RIDGE REGRESSION: Hyperparameter Tuning (alpha)")
    print("=" * 60)

    X_reg = df[FEATURES_REG]
    y_reg = df[TARGET_REG]

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg, y_reg, test_size=TEST_SIZE, random_state=SEED
    )

    # Manual search
    best_alpha, mean_maes = manual_alpha_search(X_tr_r, y_tr_r, RIDGE_ALPHAS)
    print(f"\nManual search -- Best alpha: {best_alpha}  CV MAE: ${min(mean_maes):,.0f}")

    plot_alpha_search(
        RIDGE_ALPHAS, mean_maes, best_alpha,
        FIGURE_DIR / "ch02_ridge_alpha_search.png"
    )

    # GridSearchCV (automated, same result as manual search above)
    kf_r = KFold(n_splits=5, shuffle=True, random_state=SEED)
    grid_r = GridSearchCV(
        Ridge(),
        {"alpha": RIDGE_ALPHAS},
        cv=kf_r,
        scoring="neg_mean_absolute_error",
        refit=True,
    )
    grid_r.fit(X_tr_r, y_tr_r)
    print(f"\nGridSearchCV -- Best alpha: {grid_r.best_params_['alpha']}")
    print(f"CV MAE (train): ${-grid_r.best_score_:,.0f}")

    # Final evaluation on the held-out test set -- done exactly once
    y_pred_r = grid_r.best_estimator_.predict(X_te_r)
    print(f"\nTest-set evaluation (reported once, after all tuning is complete):")
    print(f"  MAE: ${mean_absolute_error(y_te_r, y_pred_r):,.0f}")
    print(f"  R²:  {r2_score(y_te_r, y_pred_r):.3f}")

    # ------------------------------------------------------------------
    # Logistic regression: tune C
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION: Hyperparameter Tuning (C)")
    print("=" * 60)

    X_clf = df[FEATURES_CLF]
    y_clf = df[TARGET_CLF]

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_clf, y_clf, test_size=TEST_SIZE, random_state=SEED, stratify=y_clf
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid_c = GridSearchCV(
        LogisticRegression(max_iter=500),
        {"C": LOGR_C_VALUES},
        cv=skf,
        scoring="roc_auc",
        refit=True,
    )
    grid_c.fit(X_tr_c, y_tr_c)
    print(f"\nBest C: {grid_c.best_params_['C']}")
    print(f"CV AUC (train): {grid_c.best_score_:.3f}")

    plot_logreg_search(grid_c, FIGURE_DIR / "ch02_logreg_C_search.png")

    y_prob_c = grid_c.best_estimator_.predict_proba(X_te_c)[:, 1]
    print(f"\nTest-set evaluation (reported once):")
    print(f"  AUC: {roc_auc_score(y_te_c, y_prob_c):.3f}")

    # ------------------------------------------------------------------
    # Summary: what each alpha means
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ALPHA SEARCH SUMMARY (Ridge, CV MAE by alpha)")
    print("=" * 60)
    for a, mae in zip(RIDGE_ALPHAS, mean_maes):
        marker = " <-- best" if a == best_alpha else ""
        print(f"  alpha={a:>5}: CV MAE = ${mae:,.0f}{marker}")


if __name__ == "__main__":
    main()
