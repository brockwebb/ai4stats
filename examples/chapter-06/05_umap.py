"""
05_umap.py
==========
UMAP (Uniform Manifold Approximation and Projection) on county ACS data.

UMAP vs. t-SNE
--------------
- UMAP is generally faster, especially at n > 10,000.
- UMAP preserves more global structure: relative distances *between* clusters
  are more meaningful than in t-SNE, though still not fully interpretable.
- UMAP supports .transform() -- fit on training data and embed new counties
  without refitting.  t-SNE does not.
- Both methods produce non-interpretable axes.  Never label a UMAP axis as
  if it has demographic meaning.

Produces
--------
1. UMAP embedding (n_neighbors=15), colored by profile and by income
2. n_neighbors sensitivity comparison: 5, 15, 50

Prerequisites
-------------
Run 01_synthetic_county_data.py first to generate county_data.csv.
Install umap-learn:  pip install umap-learn

Usage
-----
    python 05_umap.py

Requirements: numpy, pandas, matplotlib, scikit-learn, umap-learn
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "county_data.csv"
RANDOM_SEED = 42
N_NEIGHBORS_VALUES = [5, 15, 50]
MIN_DIST = 0.1
PROFILE_COLORS = {"A_urban": "#e41a1c", "B_suburban": "#377eb8", "C_rural": "#4daf4a"}
FEATURE_COLS = [
    "median_age", "pct_bachelors", "median_hh_income", "pct_poverty",
    "pct_owner_occupied", "pct_employed", "pct_under18", "pct_over65",
    "pct_hispanic", "pct_foreign_born", "pct_renter", "pop_density_log",
    "median_gross_rent", "pct_no_vehicle", "pct_broadband",
]


def import_umap():
    """
    Import UMAP with a clear error message if umap-learn is not installed.

    Returns the UMAP class or None if unavailable.
    """
    try:
        from umap import UMAP as _UMAP
        return _UMAP
    except ImportError:
        return None


def run_umap(X_std: np.ndarray, n_neighbors: int, min_dist: float,
             seed: int, UMAP_cls) -> np.ndarray:
    """
    Fit and return a 2D UMAP embedding.

    Parameters
    ----------
    n_neighbors : int
        Controls local vs. global balance.  Small values (5) emphasize fine
        local structure; large values (50) preserve global topology.
    min_dist : float
        Minimum distance between points in 2D.  Small values produce tighter
        clusters; large values spread points out.
    """
    reducer = UMAP_cls(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(X_std)


def main_embedding_plot(Y: np.ndarray, df: pd.DataFrame, n_neighbors: int) -> None:
    """Two-panel plot: embedding by profile and by income."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for prof, idx_series in df.groupby("profile").groups.items():
        axes[0].scatter(
            Y[idx_series, 0], Y[idx_series, 1],
            c=PROFILE_COLORS[prof], label=prof, s=25, alpha=0.7,
        )
    axes[0].set_title(f"UMAP (n_neighbors={n_neighbors}, colored by profile)")
    axes[0].set_xlabel("UMAP-1 (not interpretable)")
    axes[0].set_ylabel("UMAP-2 (not interpretable)")
    axes[0].legend()

    sc = axes[1].scatter(
        Y[:, 0], Y[:, 1],
        c=df["median_hh_income"], cmap="RdYlGn", s=25, alpha=0.8,
    )
    plt.colorbar(sc, ax=axes[1], label="Median HH Income ($)")
    axes[1].set_title(f"UMAP (n_neighbors={n_neighbors}, colored by income)")
    axes[1].set_xlabel("UMAP-1 (not interpretable)")
    axes[1].set_ylabel("UMAP-2 (not interpretable)")

    plt.tight_layout()
    plt.savefig(
        Path(__file__).parent / f"05_umap_nn{n_neighbors}.png",
        dpi=120, bbox_inches="tight",
    )
    plt.show()
    print(f"Saved: 05_umap_nn{n_neighbors}.png")


def n_neighbors_sensitivity_plot(X_std: np.ndarray, df: pd.DataFrame,
                                 n_neighbors_list: list, min_dist: float,
                                 seed: int, UMAP_cls) -> None:
    """
    Side-by-side UMAP embeddings at multiple n_neighbors values.

    Small n_neighbors: fine local structure, disconnected micro-clusters.
    Large n_neighbors: broad global structure, clusters may merge.
    A robust cluster structure looks similar across a range of n_neighbors.
    """
    fig, axes = plt.subplots(1, len(n_neighbors_list), figsize=(16, 4))

    for ax, nn in zip(axes, n_neighbors_list):
        Y = run_umap(X_std, nn, min_dist, seed, UMAP_cls)
        for prof, idx_series in df.groupby("profile").groups.items():
            ax.scatter(
                Y[idx_series, 0], Y[idx_series, 1],
                c=PROFILE_COLORS[prof], label=prof, s=15, alpha=0.6,
            )
        ax.set_title(f"n_neighbors = {nn}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        if ax is axes[0]:
            ax.legend(fontsize=8)

    plt.suptitle(
        "UMAP sensitivity to n_neighbors\n"
        "Small n_neighbors = local detail; large = global shape",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "05_umap_n_neighbors_comparison.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 05_umap_n_neighbors_comparison.png")


if __name__ == "__main__":
    UMAP_cls = import_umap()
    if UMAP_cls is None:
        print("umap-learn is not installed.")
        print("Install it with:  pip install umap-learn")
        print("Skipping UMAP examples.")
        raise SystemExit(1)

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} counties")

    X_std = StandardScaler().fit_transform(df[FEATURE_COLS].values)

    print("\n--- Main UMAP embedding (n_neighbors=15) ---")
    Y_15 = run_umap(X_std, n_neighbors=15, min_dist=MIN_DIST,
                    seed=RANDOM_SEED, UMAP_cls=UMAP_cls)
    main_embedding_plot(Y_15, df, n_neighbors=15)

    print("\n--- n_neighbors sensitivity comparison ---")
    n_neighbors_sensitivity_plot(X_std, df, N_NEIGHBORS_VALUES,
                                 MIN_DIST, RANDOM_SEED, UMAP_cls)

    print(
        "\nReminder: like t-SNE, UMAP axes are not interpretable.  "
        "Use UMAP for visualization only, not as input to downstream algorithms."
    )
