"""
Chapter 2 -- Exercises: Skeleton and Solutions
Activity setups and complete solutions for the four in-class activities.

Activities:
  9.1 Cross-validate regression and compare splits
  9.2 GridSearchCV for Ridge regression
  9.3 Permutation importance for income regression
  9.4 GroupKFold vs KFold: spot the difference

Each activity has a SKELETON block (what a learner would fill in) and a
SOLUTION block. Run the solutions directly or adapt the skeletons.

Run: python 06_exercises.py
     (run 01_dataset_setup.py first)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold,
    GridSearchCV, cross_val_score, train_test_split,
)
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE    = Path(__file__).resolve().parents[2] / "data" / "synthetic_survey_ch02.csv"
FEATURES_REG = ["age", "education_years", "hours_per_week", "urban"]
FEATURES_CLF = ["age", "education_years", "urban", "contact_attempts", "prior_response"]
TARGET_REG   = "income"
TARGET_CLF   = "responded"
GROUP_COL    = "household_id"
SEED         = 42


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run 01_dataset_setup.py first."
        )
    return pd.read_csv(path)


# ===========================================================================
# Activity 9.1: Cross-validate regression and compare splits
# ===========================================================================

def activity_9_1_skeleton(X_reg: pd.DataFrame, y_reg: pd.Series) -> None:
    """
    SKELETON -- fill in the TODOs.

    Run 5-fold CV on Ridge(alpha=200) for income prediction.
    Report mean and std MAE. Then compare with a single split (seed=42).
    """
    kf5 = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # TODO: use cross_val_score with scoring="neg_mean_absolute_error"
    scores = cross_val_score(
        Ridge(alpha=200), X_reg, y_reg,
        cv=kf5, scoring=...   # TODO
    )
    mae_scores = ...  # TODO: negate the scores

    print(f"5-fold CV MAE: ${mae_scores.mean():,.0f}  +/- ${mae_scores.std():,.0f}")


def activity_9_1_solution(X_reg: pd.DataFrame, y_reg: pd.Series) -> None:
    """SOLUTION for activity 9.1."""
    print("\n--- Activity 9.1 Solution ---")
    kf5 = KFold(n_splits=5, shuffle=True, random_state=SEED)

    scores = cross_val_score(
        Ridge(alpha=200), X_reg, y_reg,
        cv=kf5, scoring="neg_mean_absolute_error"
    )
    mae_scores = -scores

    print(f"5-fold CV MAE: ${mae_scores.mean():,.0f}  +/- ${mae_scores.std():,.0f}")
    print("The std tells you how much the estimate would shift if you used a different subset as test.")
    print("A single split cannot show you this uncertainty -- it gives one number with no context.")

    # Single-split comparison
    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.20, random_state=SEED)
    single_mae = mean_absolute_error(y_te, Ridge(alpha=200).fit(X_tr, y_tr).predict(X_te))
    print(f"Single split MAE (seed=42): ${single_mae:,.0f}")
    print(f"Difference from CV mean:    ${abs(single_mae - mae_scores.mean()):,.0f}")


# ===========================================================================
# Activity 9.2: GridSearchCV for Ridge regression
# ===========================================================================

def activity_9_2_skeleton(X_reg: pd.DataFrame, y_reg: pd.Series) -> None:
    """
    SKELETON -- fill in the TODOs.

    Run GridSearchCV on Ridge for income prediction.
    Search alphas [50, 100, 200, 500, 1000, 2000]. Report best alpha and test MAE.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.20, random_state=SEED)

    param_grid = {"alpha": ...}   # TODO: list the alphas to search
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    grid = GridSearchCV(Ridge(), param_grid, cv=kf, scoring="neg_mean_absolute_error", refit=True)
    grid.fit(X_tr, y_tr)

    print(f"Best alpha: {grid.best_params_['alpha']}")
    final = grid.best_estimator_
    print(f"Test MAE: ${mean_absolute_error(y_te, final.predict(X_te)):,.0f}")


def activity_9_2_solution(X_reg: pd.DataFrame, y_reg: pd.Series) -> None:
    """SOLUTION for activity 9.2."""
    print("\n--- Activity 9.2 Solution ---")
    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.20, random_state=SEED)

    alphas = [50, 100, 200, 500, 1000, 2000]
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        Ridge(), {"alpha": alphas}, cv=kf,
        scoring="neg_mean_absolute_error", refit=True
    )
    grid.fit(X_tr, y_tr)

    print(f"Best alpha: {grid.best_params_['alpha']}")
    print(f"CV MAE (train set): ${-grid.best_score_:,.0f}")
    print(f"Test MAE:           ${mean_absolute_error(y_te, grid.best_estimator_.predict(X_te)):,.0f}")

    # Show the full search table
    cv_df = pd.DataFrame(grid.cv_results_)[["param_alpha", "mean_test_score", "std_test_score"]]
    cv_df["CV_MAE"] = -cv_df["mean_test_score"]
    cv_df["CV_MAE_std"] = cv_df["std_test_score"]
    print("\nFull search results:")
    print(cv_df[["param_alpha", "CV_MAE", "CV_MAE_std"]].to_string(index=False))
    print("\nNote: the minimum CV MAE is not always the best production choice.")
    print("Regularization slightly above the minimum often generalizes better (the '1-SE rule').")


# ===========================================================================
# Activity 9.3: Permutation importance for income regression
# ===========================================================================

def activity_9_3_skeleton(X_reg: pd.DataFrame, y_reg: pd.Series) -> None:
    """
    SKELETON -- fill in the TODOs.

    Fit Ridge(alpha=200) on the income training set.
    Compute permutation importance on the test set. Which feature is most important?
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.20, random_state=SEED)

    model = Ridge(alpha=200).fit(X_tr, y_tr)
    perm = permutation_importance(
        model, X_te, y_te,
        scoring="r2",
        n_repeats=...,     # TODO: choose a number (20 is a good default)
        random_state=SEED
    )

    perm_df = pd.DataFrame({
        "feature":    FEATURES_REG,
        "importance": perm.importances_mean,
    }).sort_values("importance", ascending=False)

    print(perm_df.to_string(index=False))


def activity_9_3_solution(X_reg: pd.DataFrame, y_reg: pd.Series) -> None:
    """SOLUTION for activity 9.3."""
    print("\n--- Activity 9.3 Solution ---")
    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.20, random_state=SEED)

    model = Ridge(alpha=200).fit(X_tr, y_tr)
    perm = permutation_importance(
        model, X_te, y_te,
        scoring="r2",
        n_repeats=20,
        random_state=SEED,
    )

    perm_df = pd.DataFrame({
        "feature":    FEATURES_REG,
        "importance": perm.importances_mean,
        "std":        perm.importances_std,
    }).sort_values("importance", ascending=False)

    print("Permutation importance (mean R² decrease):")
    print(perm_df.to_string(index=False))
    top = perm_df.iloc[0]["feature"]
    print(f"\nMost important feature: {top}")
    print("This matches the data generation process: education_years and age drive income.")
    print("Business case for keeping these: dropping them would noticeably degrade predictions.")
    print("hours_per_week has the smallest effect here and could be a candidate for removal")
    print("if it were costly to collect.")


# ===========================================================================
# Activity 9.4: GroupKFold vs KFold -- spot the difference
# ===========================================================================

def activity_9_4_skeleton(
    X_clf: pd.DataFrame, y_clf: pd.Series, groups: pd.Series
) -> None:
    """
    SKELETON -- fill in the TODOs.

    Run KFold and GroupKFold on the classification task.
    Report mean AUC for each. Which is more optimistic? Why?
    """
    clf = LogisticRegression(max_iter=500)

    kf_scores = cross_val_score(
        clf, X_clf, y_clf,
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="roc_auc"
    )
    gkf_scores = cross_val_score(
        clf, X_clf, y_clf,
        cv=GroupKFold(n_splits=5),
        groups=...,    # TODO: pass the household group labels
        scoring="roc_auc"
    )

    print(f"KFold      mean AUC: {kf_scores.mean():.3f}  std: {kf_scores.std():.3f}")
    print(f"GroupKFold mean AUC: {gkf_scores.mean():.3f}  std: {gkf_scores.std():.3f}")


def activity_9_4_solution(
    X_clf: pd.DataFrame, y_clf: pd.Series, groups: pd.Series
) -> None:
    """SOLUTION for activity 9.4."""
    print("\n--- Activity 9.4 Solution ---")
    clf = LogisticRegression(max_iter=500, random_state=SEED)

    kf_scores = cross_val_score(
        clf, X_clf, y_clf,
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="roc_auc"
    )
    gkf_scores = cross_val_score(
        clf, X_clf, y_clf,
        cv=GroupKFold(n_splits=5),
        groups=groups,
        scoring="roc_auc"
    )

    print(f"KFold      mean AUC: {kf_scores.mean():.3f}  std: {kf_scores.std():.3f}")
    print(f"GroupKFold mean AUC: {gkf_scores.mean():.3f}  std: {gkf_scores.std():.3f}")
    gap = kf_scores.mean() - gkf_scores.mean()
    print(f"\nKFold - GroupKFold gap: {gap:.3f} AUC points")
    print("\nWhy GroupKFold is lower (more conservative):")
    print("  Standard KFold can place household member A in the training fold")
    print("  and member B in the test fold. Because they share characteristics")
    print("  (same address, same household income, similar demographics), the model")
    print("  effectively 'sees' test data during training. This is information leakage.")
    print("  GroupKFold prevents this by keeping the whole household together.")
    print("\nThe gap tells you how much the model was benefiting from household-level")
    print("signals in KFold. In production, every new address is genuinely unknown,")
    print("so GroupKFold's estimate is the honest one to report.")


# ===========================================================================
# Main runner
# ===========================================================================

def main() -> None:
    df = load_data(DATA_FILE)
    X_reg = df[FEATURES_REG]
    y_reg = df[TARGET_REG]
    X_clf = df[FEATURES_CLF]
    y_clf = df[TARGET_CLF]
    groups = df[GROUP_COL]

    print("Running all activity solutions...")
    print("=" * 60)

    activity_9_1_solution(X_reg, y_reg)
    activity_9_2_solution(X_reg, y_reg)
    activity_9_3_solution(X_reg, y_reg)
    activity_9_4_solution(X_clf, y_clf, groups)

    print("\n" + "=" * 60)
    print("All solutions complete.")
    print("To work through the skeletons, call the _skeleton functions")
    print("and fill in the TODOs.")


if __name__ == "__main__":
    main()
