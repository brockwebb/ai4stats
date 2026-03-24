"""
06_multiple_imputation.py
Chapter 7: Imputation Methods for Survey Data

Multiple imputation with Rubin's combining rules.

Creates M=5 stochastic regression imputed datasets, analyzes each separately
(computes mean income), then combines results using Rubin's rules:
  Q_bar  -- pooled point estimate (mean of M estimates)
  B      -- between-imputation variance (how much estimates differ across datasets)
  W      -- within-imputation variance (average variance from one dataset)
  T      -- total variance (W + (1 + 1/M) * B)
  SE     -- pooled standard error = sqrt(T)

Key insight: B captures uncertainty about the imputed values themselves.
A single imputed dataset ignores B entirely, producing confidence intervals
that are too narrow. Multiple imputation is mandatory when published estimates
must have valid standard errors.

Requires: base_data.csv produced by 01_dataset_and_missingness.py

Usage:
    python 06_multiple_imputation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

INPUT_FILE = "base_data.csv"
M = 5            # Number of imputed datasets. Rubin (1987): M=5-20 is usually sufficient.
RANDOM_SEED = 42
PREDICTORS = ["age", "educ", "region", "hours_wk", "fulltime"]


# ---------------------------------------------------------------------------
# Stochastic regression imputation (single dataset)
# ---------------------------------------------------------------------------
def stochastic_regression_impute(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list,
    random_state: int,
) -> pd.Series:
    """
    Stochastic regression imputation for one imputed dataset.

    Fits a linear regression on complete cases, predicts missing values,
    and adds N(0, sigma_residual) noise to preserve variance.
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
    return result.clip(lower=0)


# ---------------------------------------------------------------------------
# Multiple imputation
# ---------------------------------------------------------------------------
def multiple_imputation(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: list,
    m: int = M,
    random_state: int = RANDOM_SEED,
) -> list:
    """
    Create M imputed datasets via stochastic regression.

    Each dataset is independent: different random seed produces different
    noise draws, yielding different imputed values. The spread across
    datasets reflects uncertainty about the missing data.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    predictor_cols : list of str
    m : int
        Number of imputed datasets.
    random_state : int
        Base seed. Each imputed dataset uses random_state + m * 100.

    Returns
    -------
    list of pd.Series, length m
    """
    return [
        stochastic_regression_impute(
            df, target_col, predictor_cols, random_state=random_state + i * 100
        )
        for i in range(m)
    ]


# ---------------------------------------------------------------------------
# Rubin's combining rules
# ---------------------------------------------------------------------------
def rubins_rules(results: list, n: int) -> dict:
    """
    Combine M analysis results using Rubin's (1987) combining rules.

    Each element of results is a tuple (Q_m, U_m) where:
      Q_m -- point estimate from the m-th imputed dataset
      U_m -- within-imputation variance for that estimate

    Returns a dict with keys: Q_bar, B, W, T, SE

    The formula:
      Q_bar = (1/M) * sum(Q_m)              pooled estimate
      W     = (1/M) * sum(U_m)              within-imputation variance
      B     = (1/(M-1)) * sum((Q_m-Q_bar)^2) between-imputation variance
      T     = W + (1 + 1/M) * B             total variance
      SE    = sqrt(T)
    """
    m = len(results)
    estimates = [q for q, _ in results]
    variances = [u for _, u in results]

    Q_bar = np.mean(estimates)
    W = np.mean(variances)
    B = np.var(estimates, ddof=1)
    T = W + (1 + 1 / m) * B
    SE = np.sqrt(T)

    return {"Q_bar": Q_bar, "B": B, "W": W, "T": T, "SE": SE}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_between_imputation_variability(
    df: pd.DataFrame, imputed_datasets: list
) -> None:
    """
    Overlay histograms of imputed values from each of the M datasets.
    The spread of the distributions illustrates between-imputation variability --
    the uncertainty about what the missing values actually are.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(imputed_datasets)))

    for m_idx, imp in enumerate(imputed_datasets):
        missing_vals = imp[df["missing"]]
        ax.hist(
            missing_vals, bins=35, alpha=0.3, density=True,
            color=colors[m_idx], label=f"Imputation {m_idx + 1}",
        )

    ax.hist(
        df.loc[~df["missing"], "income_obs"],
        bins=35, alpha=0.6, density=True, color="steelblue",
        label="Observed (all datasets)",
    )
    ax.set_xlabel("Income ($)")
    ax.set_title(f"Multiple imputation: {len(imputed_datasets)} versions of the missing values")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig("06_multiple_imputation.png", dpi=120, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Chapter 7: Multiple Imputation and Rubin's Rules ===\n")

    df = pd.read_csv(INPUT_FILE)
    df["missing"] = df["missing"].astype(bool)
    n_total = len(df)

    print(f"Creating {M} imputed datasets ...")
    imputed_datasets = multiple_imputation(df, "income_obs", PREDICTORS, m=M)

    # Analyze each imputed dataset: estimate mean income
    print("\nMean income from each imputed dataset:")
    results = []
    for m_idx, imp in enumerate(imputed_datasets):
        df_m = df.copy()
        df_m["income_imputed"] = imp
        q_m = df_m["income_imputed"].mean()
        # Within-imputation variance of the mean
        u_m = df_m["income_imputed"].var() / n_total
        results.append((q_m, u_m))
        print(f"  Dataset {m_idx + 1}: mean = ${q_m:,.0f}")

    # Apply Rubin's combining rules
    rb = rubins_rules(results, n_total)

    print(f"\nCombined estimate (Rubin's rules, M={M}):")
    print(f"  Pooled mean income (Q_bar):         ${rb['Q_bar']:,.0f}")
    print(f"  Within-imputation variance (W):      {rb['W']:.2f}")
    print(f"  Between-imputation variance (B):     {rb['B']:.2f}")
    print(f"  Total variance (T):                  {rb['T']:.2f}")
    print(f"  SE (single imputation, sqrt(W)):     ${np.sqrt(rb['W']):,.0f}")
    print(f"  SE (multiple imputation, sqrt(T)):   ${rb['SE']:,.0f}")
    print(
        f"  Between-imputation component:        "
        f"{(1 + 1/M) * rb['B'] / rb['T']:.1%} of total variance"
    )

    true_mean = df["income_true"].mean()
    print(f"\n  True population mean:               ${true_mean:,.0f}")
    print(f"  Absolute error of pooled estimate:  ${abs(rb['Q_bar'] - true_mean):,.0f}")

    print(
        "\nInterpretation: the multiple imputation SE is larger than the single-imputation\n"
        "SE because it accounts for uncertainty about the imputed values (the B term).\n"
        "A single imputed dataset produces confidence intervals that are too narrow."
    )

    print(
        f"\nWhy M={M}? Rubin (1987) relative efficiency = (1 + lambda/M)^{{-1}}\n"
        f"where lambda is the fraction of missing information. At M=5 and lambda=0.3,\n"
        f"efficiency is ~94%. Increasing M beyond 10-20 yields diminishing returns."
    )

    plot_between_imputation_variability(df, imputed_datasets)
    print("\nFigure saved to 06_multiple_imputation.png")
