"""
03_utility_marginal.py
======================
Chapter 9: Synthetic Data Generation for Federal Statistics

Evaluates marginal utility of the synthetic dataset by comparing univariate
distributions and summary statistics between the confidential and synthetic
datasets.

Why this matters:
    Marginal utility is the first and easiest utility check. If the synthetic
    data does not approximately match the confidential data on individual
    variable distributions, it fails before you even check relationships. But
    passing marginal checks does not guarantee analytic validity — that requires
    the regression tests in 05_utility_regression.py.

Usage:
    python 03_utility_marginal.py
    (Requires confidential_microdata.csv and synthetic_data.csv)

Outputs:
    - marginal_comparison.png: 4-panel histogram comparison
    - Summary statistics table printed to stdout

Requirements:
    Python 3.9+, numpy, pandas, matplotlib, scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
import os


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load confidential and synthetic datasets from CSV files."""
    for path in ["confidential_microdata.csv", "synthetic_data.csv"]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run 01_confidential_dataset.py and "
                "02_sequential_synthesis.py first."
            )
    return pd.read_csv("confidential_microdata.csv"), pd.read_csv("synthetic_data.csv")


def print_summary_table(df_conf: pd.DataFrame, df_synth: pd.DataFrame) -> None:
    """Print a side-by-side summary statistics table for numeric variables."""
    print("Summary statistics comparison: confidential vs. synthetic")
    print("=" * 78)
    header = f"{'Variable':<10} {'Stat':<8} {'Confidential':>14} {'Synthetic':>14} {'Diff':>10}"
    print(header)
    print("-" * 78)

    for col in ["age", "educ", "income", "married"]:
        c = df_conf[col]
        s = df_synth[col]
        stats = {
            "mean":   (c.mean(),   s.mean()),
            "std":    (c.std(),    s.std()),
            "median": (c.median(), s.median()),
            "skew":   (skew(c),    skew(s)),
        }
        for stat_name, (cv, sv) in stats.items():
            diff = sv - cv
            print(f"{col:<10} {stat_name:<8} {cv:>14.2f} {sv:>14.2f} {diff:>+10.2f}")
        print()


def plot_marginal_comparison(
    df_conf: pd.DataFrame,
    df_synth: pd.DataFrame,
    output_path: str = "marginal_comparison.png",
) -> None:
    """
    4-panel histogram comparison: age, educ, income, married.

    Each panel overlays confidential (blue) and synthetic (orange) distributions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    plot_specs = [
        ("age",     "Age (years)",        None,    None),
        ("educ",    "Education (years)",  None,    None),
        ("income",  "Income ($)",         0,       200_000),
        ("married", "Married (0/1)",      None,    None),
    ]

    for ax, (col, label, xmin, xmax) in zip(axes.flat, plot_specs):
        c_vals = df_conf[col].values
        s_vals = df_synth[col].values

        if xmin is not None or xmax is not None:
            c_vals = np.clip(c_vals, xmin or c_vals.min(), xmax or c_vals.max())
            s_vals = np.clip(s_vals, xmin or s_vals.min(), xmax or s_vals.max())

        bins = 25 if col != "married" else 3
        ax.hist(c_vals, bins=bins, alpha=0.5, color="steelblue",
                density=True, label=f"Confidential (n={len(df_conf)})")
        ax.hist(s_vals, bins=bins, alpha=0.5, color="tomato",
                density=True, label=f"Synthetic (n={len(df_synth)})")
        ax.set_xlabel(label)
        ax.set_title(f"{col.capitalize()} distribution")
        ax.legend(fontsize=8)

    fig.suptitle("Marginal distribution comparison: confidential vs. synthetic", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.close()


if __name__ == "__main__":
    df_conf, df_synth = load_datasets()
    print(f"Loaded: confidential n={len(df_conf)}, synthetic n={len(df_synth)}")
    print()
    print_summary_table(df_conf, df_synth)
    plot_marginal_comparison(df_conf, df_synth)
    print()
    print("Interpretation:")
    print("  Passing marginal checks is necessary but not sufficient.")
    print("  A synthetic dataset can match univariate distributions exactly")
    print("  while destroying all bivariate correlations. See 04_utility_bivariate.py.")
