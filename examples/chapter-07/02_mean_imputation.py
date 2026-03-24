"""
02_mean_imputation.py
Chapter 7: Imputation Methods for Survey Data

Demonstrates mean imputation and its principal failure modes:
  - All missing values receive the same imputed value (no individual variation)
  - Imputed distribution spikes at the mean (variance collapse)
  - Correlation between income and other predictors is attenuated

Requires: base_data.csv produced by 01_dataset_and_missingness.py

Usage:
    python 02_mean_imputation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

INPUT_FILE = "base_data.csv"


# ---------------------------------------------------------------------------
# Imputation function
# ---------------------------------------------------------------------------
def impute_mean(series: pd.Series) -> pd.Series:
    """
    Replace missing values with the column-wide observed mean.

    This is the simplest possible imputation strategy. It is rarely
    appropriate for published survey estimates because it collapses all
    variance in the imputed records to zero -- every missing respondent
    gets the same value regardless of their characteristics.

    Parameters
    ----------
    series : pd.Series
        A numeric series that may contain NaN values.

    Returns
    -------
    pd.Series
        Series with NaN replaced by the observed mean.
    """
    return series.fillna(series.mean())


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def plot_mean_imputation_failures(df: pd.DataFrame) -> None:
    """
    Two-panel figure showing:
      Left  -- histogram comparing observed distribution to imputed spike
      Right -- scatter of true vs. imputed income for missing records
               (a flat horizontal line reveals zero individual variation)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    imputed_val = df["income_obs"].mean()

    # Left: distribution comparison
    ax = axes[0]
    ax.hist(
        df.loc[~df["missing"], "income_obs"],
        bins=40, alpha=0.6, color="steelblue", density=True, label="Observed (complete)",
    )
    ax.hist(
        df.loc[df["missing"], "income_mean_imp"],
        bins=40, alpha=0.6, color="tomato", density=True, label="Imputed (mean)",
    )
    ax.axvline(
        imputed_val, color="tomato", linestyle="--", linewidth=2,
        label=f"Imputed value = ${imputed_val:,.0f}",
    )
    ax.set_xlabel("Income ($)")
    ax.set_title("Mean imputation collapses variation")
    ax.legend(fontsize=8)

    # Right: true vs. imputed for missing records
    ax = axes[1]
    ax.scatter(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_mean_imp"],
        alpha=0.5, s=20, color="tomato",
    )
    lims = [0, 150_000]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("True income ($)")
    ax.set_ylabel("Imputed income ($)")
    ax.set_title("Mean imputation: all predictions identical")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("02_mean_imputation.png", dpi=120, bbox_inches="tight")
    plt.show()


def show_correlation_attenuation(df: pd.DataFrame) -> None:
    """
    Demonstrate that mean imputation attenuates the correlation between
    income and education -- a well-documented failure mode.
    """
    corr_obs = df.loc[~df["missing"], ["income_obs", "educ"]].corr().iloc[0, 1]
    corr_imp = df[["income_mean_imp", "educ"]].corr().iloc[0, 1]
    print(f"  Correlation (income, educ) — observed only: {corr_obs:.3f}")
    print(f"  Correlation (income, educ) — after mean imputation: {corr_imp:.3f}")
    print(
        f"  Attenuation: {corr_obs - corr_imp:.3f} points "
        "(imputed records pull the correlation toward zero)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Chapter 7: Mean Imputation ===\n")

    df = pd.read_csv(INPUT_FILE)
    df["missing"] = df["missing"].astype(bool)

    # Apply mean imputation
    df["income_mean_imp"] = impute_mean(df["income_obs"])

    mae = mean_absolute_error(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_mean_imp"],
    )
    print(f"Mean imputation MAE on missing records: ${mae:,.0f}\n")

    true_var = df["income_true"].var()
    imp_var = df["income_mean_imp"].var()
    print(f"True income variance:              {true_var:>15,.0f}")
    print(f"Variance after mean imputation:    {imp_var:>15,.0f}")
    print(f"Variance ratio (imputed / true):   {imp_var / true_var:.3f}  (severe underestimate)\n")

    print("Why mean imputation is rarely appropriate:")
    print("  1. All imputed values are identical -- no individual variation")
    print("  2. Underestimates variance, so standard errors are too small")
    print("  3. Attenuates correlations between income and other variables")
    print("  4. Published estimates based on mean-imputed data understate income inequality\n")

    print("Correlation attenuation:")
    show_correlation_attenuation(df)

    plot_mean_imputation_failures(df)
    print("\nFigure saved to 02_mean_imputation.png")
