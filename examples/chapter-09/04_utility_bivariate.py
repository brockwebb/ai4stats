"""
04_utility_bivariate.py
=======================
Chapter 9: Synthetic Data Generation for Federal Statistics

Evaluates bivariate utility by comparing the correlation structure of the
confidential and synthetic datasets. Produces a three-panel heatmap:
confidential correlations, synthetic correlations, and the difference matrix.

Why this matters:
    Many federal statistical analyses depend on relationships between variables
    (income and education, age and marital status, etc.). A synthetic dataset
    can match marginal distributions while completely destroying bivariate
    correlations. Checking the correlation matrix is the minimum bivariate
    utility check. Large differences in the difference panel indicate which
    relationships the synthesis failed to preserve.

Usage:
    python 04_utility_bivariate.py
    (Requires confidential_microdata.csv and synthetic_data.csv)

Outputs:
    - bivariate_correlation.png: 3-panel correlation heatmap
    - Correlation difference table printed to stdout

Requirements:
    Python 3.9+, numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def compute_numeric_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns only."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].corr()


def print_correlation_diff_table(
    conf_corr: pd.DataFrame, synth_corr: pd.DataFrame
) -> None:
    """Print pairwise correlation differences between confidential and synthetic."""
    cols = conf_corr.columns.tolist()
    print("Pairwise correlation comparison: confidential vs. synthetic")
    print("=" * 68)
    print(f"{'Pair':<22} {'Confidential':>14} {'Synthetic':>14} {'Difference':>12}")
    print("-" * 68)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            cv = conf_corr.loc[c1, c2]
            sv = synth_corr.loc[c1, c2]
            diff = sv - cv
            flag = " <-- check" if abs(diff) > 0.10 else ""
            print(f"{c1 + ' vs ' + c2:<22} {cv:>14.3f} {sv:>14.3f} {diff:>+12.3f}{flag}")


def plot_correlation_heatmaps(
    conf_corr: pd.DataFrame,
    synth_corr: pd.DataFrame,
    output_path: str = "bivariate_correlation.png",
) -> None:
    """
    3-panel correlation heatmap:
        Panel 1: Confidential correlations
        Panel 2: Synthetic correlations
        Panel 3: Absolute difference (synthetic - confidential)
    """
    diff_corr = synth_corr - conf_corr

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    panels = [
        ("Confidential correlations",      conf_corr,  "coolwarm", -1, 1),
        ("Synthetic correlations",          synth_corr, "coolwarm", -1, 1),
        ("Difference (synthetic - conf)",   diff_corr,  "RdBu_r",  -0.4, 0.4),
    ]

    for ax, (title, mat, cmap, vmin, vmax) in zip(axes, panels):
        cols = mat.columns.tolist()
        im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(cols, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(
                    j, i, f"{mat.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(mat.values[i, j]) < 0.7 else "white",
                )
        ax.set_title(title, fontsize=10)

    fig.suptitle("Bivariate utility: correlation matrix comparison", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.close()


if __name__ == "__main__":
    df_conf, df_synth = load_datasets()

    conf_corr = compute_numeric_corr(df_conf)
    synth_corr = compute_numeric_corr(df_synth)

    print_correlation_diff_table(conf_corr, synth_corr)
    print()
    plot_correlation_heatmaps(conf_corr, synth_corr)
    print()
    print("Interpretation:")
    print("  Differences < 0.05 in absolute value are generally acceptable.")
    print("  Differences > 0.10 indicate the synthesis failed to preserve that relationship.")
    print("  Large differences in income-married indicate married was not modeled correctly.")
