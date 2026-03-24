"""
07_rf_imputation.py
Chapter 7: Imputation Methods for Survey Data

Random Forest imputation: fit a RandomForestRegressor on complete cases, then
predict missing values. Uses stochastic tree selection (draw from one random
tree per observation rather than averaging all trees) to add variability and
avoid the deterministic-imputation variance collapse.

Also shows where RF helps (nonlinear subgroups) and where it does not
(low-count subgroups, auditability constraints).

Requires: base_data.csv produced by 01_dataset_and_missingness.py

Usage:
    python 07_rf_imputation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

INPUT_FILE = "base_data.csv"
RANDOM_SEED = 42
N_ESTIMATORS = 100
PREDICTORS = ["age", "educ", "region", "hours_wk", "fulltime"]


# ---------------------------------------------------------------------------
# Imputation functions
# ---------------------------------------------------------------------------
def rf_impute(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_SEED,
) -> tuple:
    """
    Random Forest imputation with stochastic tree selection.

    For each observation, draw one tree at random from the ensemble and use
    that tree's prediction. This adds variability compared to averaging all
    trees (which would produce a deterministic imputation and collapse variance).

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    predictor_cols : list of str
    n_estimators : int
        Number of trees in the forest.
    random_state : int

    Returns
    -------
    imputed : pd.Series
    rf_model : RandomForestRegressor
    """
    rng = np.random.default_rng(random_state)
    complete = df[target_col].notna()

    X_complete = pd.get_dummies(df.loc[complete, predictor_cols], drop_first=True)
    y_complete = df.loc[complete, target_col]

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        min_samples_leaf=5,
    )
    rf.fit(X_complete, y_complete)

    X_all = pd.get_dummies(df[predictor_cols], drop_first=True)
    X_all = X_all.reindex(columns=X_complete.columns, fill_value=0)

    # Stochastic prediction: each observation draws from one randomly chosen tree
    all_tree_preds = np.array([tree.predict(X_all) for tree in rf.estimators_])
    tree_choice = rng.integers(0, len(rf.estimators_), size=len(df))
    predictions_stoch = all_tree_preds[tree_choice, np.arange(len(df))]

    result = df[target_col].copy()
    result[~complete] = predictions_stoch[~complete]
    return result, rf


def stochastic_regression_impute(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list,
    random_state: int = RANDOM_SEED,
) -> pd.Series:
    """Stochastic linear regression imputation (baseline for comparison)."""
    rng = np.random.default_rng(random_state)
    complete = df[target_col].notna()
    X_complete = pd.get_dummies(df.loc[complete, predictor_cols], drop_first=True)
    y_complete = df.loc[complete, target_col]

    model = LinearRegression()
    model.fit(X_complete, y_complete)

    residuals = y_complete - model.predict(X_complete)
    residual_std = residuals.std()

    X_all = pd.get_dummies(df[predictor_cols], drop_first=True)
    X_all = X_all.reindex(columns=X_complete.columns, fill_value=0)
    predictions = model.predict(X_all)

    n_missing = (~complete).sum()
    noise = rng.normal(0, residual_std, n_missing)
    result = df[target_col].copy()
    result[~complete] = predictions[~complete] + noise
    return result.clip(lower=0)


# ---------------------------------------------------------------------------
# Subgroup comparison
# ---------------------------------------------------------------------------
def subgroup_comparison(df: pd.DataFrame) -> None:
    """
    Compare RF and stochastic regression on two subgroups where their
    behavior differs:
      - Part-time workers: nonlinear income relationship, RF may help
      - Graduate-degree workers: high income, limited training data at the top
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    lims = [0, 150_000]

    subgroups = [
        ("Part-time workers (fulltime=0)", df["missing"] & (df["fulltime"] == 0), [0, 80_000]),
        ("Graduate degree (educ=4)", df["missing"] & (df["educ"] == 4), lims),
    ]

    for ax, (title, mask, plot_lims) in zip(axes, subgroups):
        if mask.sum() < 5:
            ax.set_title(f"{title}\n(insufficient missing records)")
            continue

        mae_reg = mean_absolute_error(
            df.loc[mask, "income_true"], df.loc[mask, "income_reg_stoch"]
        )
        mae_rf = mean_absolute_error(
            df.loc[mask, "income_true"], df.loc[mask, "income_rf"]
        )

        ax.scatter(
            df.loc[mask, "income_true"], df.loc[mask, "income_reg_stoch"],
            alpha=0.6, s=25, color="darkorange", label=f"Stoch. reg (MAE=${mae_reg/1000:.1f}k)",
        )
        ax.scatter(
            df.loc[mask, "income_true"], df.loc[mask, "income_rf"],
            alpha=0.6, s=25, color="mediumpurple", label=f"RF (MAE=${mae_rf/1000:.1f}k)",
        )
        ax.plot(plot_lims, plot_lims, "k--", linewidth=1)
        ax.set_xlabel("True income ($)")
        ax.set_ylabel("Imputed income ($)")
        ax.set_title(f"{title}\nn={mask.sum()} missing")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("07_rf_imputation.png", dpi=120, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Chapter 7: Random Forest Imputation ===\n")

    df = pd.read_csv(INPUT_FILE)
    df["missing"] = df["missing"].astype(bool)

    print("Fitting stochastic regression (baseline) ...")
    df["income_reg_stoch"] = stochastic_regression_impute(df, "income_obs", PREDICTORS)

    print(f"Fitting Random Forest ({N_ESTIMATORS} trees, stochastic prediction) ...")
    df["income_rf"], rf_model = rf_impute(df, "income_obs", PREDICTORS)

    mae_reg = mean_absolute_error(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_reg_stoch"],
    )
    mae_rf = mean_absolute_error(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_rf"],
    )
    true_var = df["income_true"].var()

    print(f"\nOverall MAE comparison:")
    print(f"  Stochastic regression: ${mae_reg:,.0f}")
    print(f"  Random Forest:         ${mae_rf:,.0f}")

    print(f"\nVariance preservation (ratio to true variance):")
    print(f"  Stochastic regression: {df['income_reg_stoch'].var() / true_var:.3f}")
    print(f"  Random Forest:         {df['income_rf'].var() / true_var:.3f}")

    print("\nSubgroup comparison:")
    subgroup_comparison(df)
    print("Figure saved to 07_rf_imputation.png")

    print("\nWhen RF imputation helps:")
    print("  - Complex nonlinear interactions between predictors and target")
    print("  - Large samples (RF needs data to learn patterns reliably)")
    print("  - Multiple variables to impute simultaneously (iterative missForest)")

    print("\nWhen RF imputation does not help (or hurts):")
    print("  - Small samples: RF overfits, linear regression is more stable")
    print("  - Auditability required: RF is hard to explain to OMB or IG reviewers")
    print("  - Variance estimates needed: RF imputation without multiple imputation")
    print("    still understates uncertainty (need Rubin's rules)")
    print("  - Regulatory constraints: documented, reproducible formulas are required")
    print("\nGuideline: if MAE difference between RF and regression is small on your")
    print("data, choose regression. The governance advantage outweighs marginal accuracy.")
