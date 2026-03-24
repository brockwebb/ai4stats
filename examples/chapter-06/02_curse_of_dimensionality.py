"""
02_curse_of_dimensionality.py
==============================
Demonstrate the curse of dimensionality using pairwise distance concentration
and a pair plot of a 6-variable ACS subset.

Key ideas illustrated
---------------------
1. As dimensionality grows, the distribution of pairwise Euclidean distances
   narrows relative to its mean.  "Nearest" and "farthest" neighbors become
   nearly indistinguishable -- a problem for any distance-based algorithm
   (K-nearest neighbors, K-means, hot-deck imputation matching).

2. Even with a manageable 15-variable dataset, exhaustive pairwise scatter
   plots (105 pairs) are impractical.  Dimension reduction replaces them with
   2-3 interpretable summary axes.

Prerequisites
-------------
Run 01_synthetic_county_data.py first to generate county_data.csv.

Usage
-----
    python 02_curse_of_dimensionality.py

Requirements: numpy, pandas, matplotlib, scikit-learn
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "county_data.csv"
RANDOM_SEED = 0
DIMENSIONS_TO_COMPARE = [2, 15, 100]
SUBSET_VARS = [
    "median_hh_income",
    "pct_bachelors",
    "pct_poverty",
    "median_age",
    "pop_density_log",
    "pct_broadband",
]
PROFILE_COLORS = {"A_urban": "#e41a1c", "B_suburban": "#377eb8", "C_rural": "#4daf4a"}


def distance_concentration_plot(n_points: int, dimensions: list, seed: int) -> None:
    """
    Plot histograms of pairwise Euclidean distances for uniform random data
    at different dimensionalities.

    Shows that as p increases, the relative spread of distances shrinks --
    the classic distance concentration effect.
    """
    fig, axes = plt.subplots(1, len(dimensions), figsize=(14, 4))

    for ax, p in zip(axes, dimensions):
        rng = np.random.default_rng(seed)
        pts = rng.uniform(0, 1, size=(n_points, p))
        dists = pairwise_distances(pts, metric="euclidean")
        upper = dists[np.triu_indices(n_points, k=1)]

        mean_d = np.mean(upper)
        std_d = np.std(upper)
        cv = std_d / mean_d  # coefficient of variation -- lower = more concentrated

        ax.hist(upper, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(mean_d, color="firebrick", lw=2,
                   label=f"mean={mean_d:.2f}\nCV={cv:.3f}")
        ax.set_title(f"p = {p} dimensions")
        ax.set_xlabel("Euclidean distance")
        ax.set_ylabel("Count" if ax is axes[0] else "")
        ax.legend(fontsize=9)

    plt.suptitle(
        "Distance concentration: pairwise distances in uniform random data\n"
        "CV (std/mean) shrinks as dimensions grow -- 'nearest' and 'farthest' converge",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "02_distance_concentration.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 02_distance_concentration.png")


def pair_plot(df: pd.DataFrame, variables: list) -> None:
    """
    Scatter matrix for a subset of ACS variables, colored by demographic profile.

    With 6 variables there are 15 pairwise plots -- already hard to scan.
    With 15 variables there would be 105 pairs.
    """
    n = len(variables)
    fig, axes = plt.subplots(n, n, figsize=(12, 10))

    for i, vi in enumerate(variables):
        for j, vj in enumerate(variables):
            ax = axes[i][j]
            if i == j:
                ax.hist(df[vi], bins=20, color="gray", edgecolor="white", alpha=0.7)
            else:
                for prof, grp in df.groupby("profile"):
                    ax.scatter(
                        grp[vj],
                        grp[vi],
                        c=PROFILE_COLORS[prof],
                        s=8,
                        alpha=0.5,
                        label=prof if i == 0 and j == 1 else None,
                    )
            if j == 0:
                ax.set_ylabel(vi, fontsize=7)
            if i == n - 1:
                ax.set_xlabel(vj, fontsize=7)
            ax.tick_params(labelsize=6)

    axes[0][1].legend(fontsize=8, title="Profile", loc="upper right")
    plt.suptitle(
        f"Pair plot: {len(variables)} of 15 ACS variables "
        f"({len(variables)*(len(variables)-1)//2} pairs shown)\n"
        "15 variables would require 105 pairs -- impractical for inspection",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "02_pair_plot.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 02_pair_plot.png")


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)
    n_counties = len(df)
    print(f"Loaded {n_counties} counties from {DATA_FILE.name}")

    print("\n--- Distance concentration ---")
    distance_concentration_plot(n_counties, DIMENSIONS_TO_COMPARE, RANDOM_SEED)

    print("\n--- Pair plot (6-variable subset) ---")
    pair_plot(df, SUBSET_VARS)

    print(
        "\nKey takeaway: even 15 variables produce 105 pairs; "
        "at 100 variables that becomes 4,950. "
        "Dimension reduction replaces all pairs with 2-3 interpretable axes."
    )
