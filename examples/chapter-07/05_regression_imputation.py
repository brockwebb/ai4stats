"""
05_regression_imputation.py
Chapter 7: Imputation Methods for Survey Data

Regression imputation in two forms:
  - Deterministic: predict missing values from regression; no noise added.
    All imputed values lie exactly on the regression plane, understating variance.
  - Stochastic: add a draw from the residual distribution to each prediction.
    Restores the variance that deterministic imputation removes.

The stochastic form has higher MAE than the deterministic form. This is expected
and correct: income is genuinely variable around its predicted value, and a good
imputation method should reflect that variability.

Requires: base_data.csv produced by 01_dataset_and_missingness.py

Usage:
    python 05_regression_imputation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

INPUT_FILE = "base_data.csv"
RANDOM_SEED = 42

PREDICTORS = ["age", "educ", "region", "hours_wk", "fulltime"]


# ---------------------------------------------------------------------------
# Imputation functions
# ---------------------------------------------------------------------------
def regression_impute(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list,
) -> tuple:
    """
    Deterministic regression imputation.

    Fit a linear regression on complete cases (rows where target_col is not
    missing). Predict missing values using the fitted model. Return the imputed
    series and the fitted model.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    predictor_cols : list of str

    Returns
    -------
    imputed : pd.Series
    model : LinearRegression
    X_complete_cols : list
        Column names after one-hot encoding, needed for downstream diagnostics.
    """
    complete = df[target_col].notna()
    X_complete = pd.get_dummies(df.loc[complete, predictor_cols], drop_first=True)
    y_complete = df.loc[complete, target_col]

    model = LinearRegression()
    model.fit(X_complete, y_complete)

    X_all = pd.get_dummies(df[predictor_cols], drop_first=True)
    X_all = X_all.reindex(columns=X_complete.columns, fill_value=0)
    predictions = model.predict(X_all)

    result = df[target_col].copy()
    result[~complete] = predictions[~complete]
    return result, model, list(X_complete.columns)


def stochastic_regression_impute(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list,
    random_state: int = RANDOM_SEED,
) -> tuple:
    """
    Stochastic regression imputation.

    Same as deterministic imputation, but adds a random draw from the empirical
    residual distribution (N(0, sigma_residual)) to each imputed value. This
    restores the within-record variance that deterministic imputation suppresses.

    The clip to non-negative ensures no imputed income is below zero.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    predictor_cols : list of str
    random_state : int

    Returns
    -------
    imputed : pd.Series
    model : LinearRegression
    residual_std : float
    """
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
    result = result.clip(lower=0)
    return result, model, residual_std


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def plot_comparison(df: pd.DataFrame, residual_std: float) -> None:
    """
    Three-panel figure comparing deterministic and stochastic regression:
      Left   -- distribution of imputed values vs. observed
      Center -- true vs. imputed scatter for deterministic
      Right  -- true vs. imputed scatter for stochastic
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    lims = [0, 150_000]

    ax = axes[0]
    ax.hist(
        df.loc[~df["missing"], "income_obs"],
        bins=40, alpha=0.5, color="steelblue", density=True, label="Observed",
    )
    ax.hist(
        df.loc[df["missing"], "income_reg_det"],
        bins=40, alpha=0.5, color="orchid", density=True, label="Deterministic",
    )
    ax.hist(
        df.loc[df["missing"], "income_reg_stoch"],
        bins=40, alpha=0.5, color="darkorange", density=True, label="Stochastic",
    )
    ax.set_xlabel("Income ($)")
    ax.set_title("Stochastic imputation: more realistic spread")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.scatter(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_reg_det"],
        alpha=0.5, s=20, color="orchid",
    )
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("True income ($)")
    ax.set_ylabel("Imputed income ($)")
    ax.set_title("Deterministic: tight band along regression line")

    ax = axes[2]
    ax.scatter(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_reg_stoch"],
        alpha=0.5, s=20, color="darkorange",
    )
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("True income ($)")
    ax.set_ylabel("Imputed income ($)")
    ax.set_title(f"Stochastic: added noise SD = ${residual_std:,.0f}")

    plt.tight_layout()
    plt.savefig("05_regression_imputation.png", dpi=120, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Chapter 7: Regression Imputation ===\n")

    df = pd.read_csv(INPUT_FILE)
    df["missing"] = df["missing"].astype(bool)

    # Deterministic
    df["income_reg_det"], reg_model, enc_cols = regression_impute(
        df, "income_obs", PREDICTORS
    )
    complete_mask = ~df["missing"]
    X_check = pd.get_dummies(df.loc[complete_mask, PREDICTORS], drop_first=True)
    X_check = X_check.reindex(columns=enc_cols, fill_value=0)
    r2 = r2_score(df.loc[complete_mask, "income_obs"], reg_model.predict(X_check))
    print(f"Regression R^2 on complete cases: {r2:.3f}")

    # Stochastic
    df["income_reg_stoch"], _, res_std = stochastic_regression_impute(
        df, "income_obs", PREDICTORS
    )
    print(f"Residual standard deviation:      ${res_std:,.0f}")

    mae_det = mean_absolute_error(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_reg_det"],
    )
    mae_stoch = mean_absolute_error(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_reg_stoch"],
    )
    print(f"\nDeterministic regression MAE: ${mae_det:,.0f}")
    print(f"Stochastic regression MAE:    ${mae_stoch:,.0f}")
    print(
        "\nNote: stochastic has higher MAE than deterministic -- this is expected.\n"
        "Higher MAE does not mean worse imputation. Stochastic correctly reflects\n"
        "that income varies around its predicted value."
    )

    true_var = df["income_true"].var()
    print(f"\nVariance preservation (ratio to true variance):")
    print(f"  True income variance:            {true_var:>15,.0f}")
    print(f"  Deterministic imputation:        {df['income_reg_det'].var():>15,.0f}  ratio = {df['income_reg_det'].var()/true_var:.3f}")
    print(f"  Stochastic imputation:           {df['income_reg_stoch'].var():>15,.0f}  ratio = {df['income_reg_stoch'].var()/true_var:.3f}")

    plot_comparison(df, res_std)
    print("\nFigure saved to 05_regression_imputation.png")
