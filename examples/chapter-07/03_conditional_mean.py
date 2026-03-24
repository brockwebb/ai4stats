"""
03_conditional_mean.py
Chapter 7: Imputation Methods for Survey Data

Conditional mean imputation: replace missing values with the mean within
strata defined by education level and region. Compares this to unconditional
mean imputation and shows the improvement from conditioning on known predictors.

Requires: base_data.csv produced by 01_dataset_and_missingness.py

Usage:
    python 03_conditional_mean.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

INPUT_FILE = "base_data.csv"


# ---------------------------------------------------------------------------
# Imputation functions
# ---------------------------------------------------------------------------
def impute_mean(series: pd.Series) -> pd.Series:
    """Replace missing values with the column-wide observed mean."""
    return series.fillna(series.mean())


def impute_conditional_mean(
    df: pd.DataFrame, target_col: str, group_cols: list
) -> pd.Series:
    """
    Replace missing values with the group mean, where groups are defined by
    every combination of group_cols.

    If an entire group has no observed values (all missing), falls back to
    the column-wide mean to avoid introducing NaN into the imputed series.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Must contain target_col and all columns in group_cols.
    target_col : str
        Column with missing values to impute.
    group_cols : list of str
        Columns that define the conditioning strata.

    Returns
    -------
    pd.Series
        Imputed series with no missing values.
    """
    result = df[target_col].copy()
    # Transform computes the group mean aligned to every row's index
    group_means = df.groupby(group_cols)[target_col].transform("mean")
    result = result.fillna(group_means)
    # Fallback for strata where every record is missing
    result = result.fillna(df[target_col].mean())
    return result


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------
def compare_distributions(df: pd.DataFrame) -> None:
    """
    Two-panel figure comparing:
      Left  -- distributions of imputed values: mean vs. conditional mean
      Right -- true vs. imputed scatter for missing records under each method
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(
        df.loc[~df["missing"], "income_obs"],
        bins=40, alpha=0.5, color="steelblue", density=True, label="Observed",
    )
    ax.hist(
        df.loc[df["missing"], "income_mean_imp"],
        bins=40, alpha=0.5, color="tomato", density=True, label="Unconditional mean",
    )
    ax.hist(
        df.loc[df["missing"], "income_cond_imp"],
        bins=40, alpha=0.5, color="darkorange", density=True, label="Conditional mean",
    )
    ax.set_xlabel("Income ($)")
    ax.set_title("Conditional mean: more spread than unconditional")
    ax.legend(fontsize=8)

    ax = axes[1]
    lims = [0, 150_000]
    ax.scatter(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_mean_imp"],
        alpha=0.4, s=18, color="tomato", label="Unconditional mean",
    )
    ax.scatter(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_cond_imp"],
        alpha=0.4, s=18, color="darkorange", label="Conditional mean",
    )
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("True income ($)")
    ax.set_ylabel("Imputed income ($)")
    ax.set_title("Conditional mean: better tracking, still band-like")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("03_conditional_mean.png", dpi=120, bbox_inches="tight")
    plt.show()


def strata_diagnostics(df: pd.DataFrame) -> None:
    """
    Print per-stratum (education x region) observed mean vs. imputed mean.
    A well-functioning conditional mean imputation should show close alignment.
    """
    educ_labels = {1: "<HS", 2: "HS", 3: "College", 4: "Grad"}
    print(f"\n{'Educ':<6} {'Region':<12} {'Obs mean':>10} {'Cond imp mean':>14} {'Diff':>8}")
    print("-" * 55)
    for e in [1, 2, 3, 4]:
        for r in ["Northeast", "Midwest", "South", "West"]:
            obs_mask = (df.educ == e) & (df.region == r) & ~df["missing"]
            imp_mask = (df.educ == e) & (df.region == r) & df["missing"]
            if obs_mask.sum() == 0 or imp_mask.sum() == 0:
                continue
            obs_mean = df.loc[obs_mask, "income_obs"].mean()
            cond_mean = df.loc[imp_mask, "income_cond_imp"].mean()
            print(
                f"{educ_labels[e]:<6} {r:<12} "
                f"${obs_mean:>9,.0f} ${cond_mean:>13,.0f} "
                f"${cond_mean - obs_mean:>+7,.0f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Chapter 7: Conditional Mean Imputation ===\n")

    df = pd.read_csv(INPUT_FILE)
    df["missing"] = df["missing"].astype(bool)

    # Unconditional mean for comparison
    df["income_mean_imp"] = impute_mean(df["income_obs"])

    # Conditional mean by education and region
    df["income_cond_imp"] = impute_conditional_mean(
        df, "income_obs", ["educ", "region"]
    )

    mae_mean = mean_absolute_error(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_mean_imp"],
    )
    mae_cond = mean_absolute_error(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_cond_imp"],
    )

    print(f"Unconditional mean MAE: ${mae_mean:,.0f}")
    print(f"Conditional mean MAE:   ${mae_cond:,.0f}")
    print(f"Improvement:            ${mae_mean - mae_cond:,.0f} better from conditioning\n")

    true_var = df["income_true"].var()
    print(f"Variance preservation (ratio of imputed to true variance):")
    print(f"  Unconditional mean: {df['income_mean_imp'].var() / true_var:.3f}")
    print(f"  Conditional mean:   {df['income_cond_imp'].var() / true_var:.3f}")
    print(
        "\nNote: conditional mean still collapses within-cell variance. "
        "Two respondents in the same education/region cell with different ages "
        "or hours worked get identical imputed values."
    )

    print("\nPer-stratum diagnostics (education x region):")
    strata_diagnostics(df)

    compare_distributions(df)
    print("\nFigure saved to 03_conditional_mean.png")
