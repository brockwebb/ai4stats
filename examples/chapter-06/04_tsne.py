"""
04_tsne.py
==========
t-SNE (t-distributed Stochastic Neighbor Embedding) on county ACS data.

Produces
--------
1. t-SNE embedding at perplexity=30, colored by profile and by income
2. Perplexity sensitivity comparison: perplexity = 5, 30, 50

Key cautions demonstrated
--------------------------
- t-SNE axes have no interpretable meaning.  Scale, rotation, and axis labels
  are meaningless -- only relative cluster positions matter.
- Different perplexity values can produce qualitatively different layouts.
  Perplexity controls the effective number of neighbors (try 5-50).
- t-SNE is deterministic given a seed, but two runs with different seeds or
  perplexities are not directly comparable.
- t-SNE does not support transforming new data points.  For out-of-sample
  embedding, use UMAP or PCA.

Prerequisites
-------------
Run 01_synthetic_county_data.py first to generate county_data.csv.

Usage
-----
    python 04_tsne.py

Requirements: numpy, pandas, matplotlib, scikit-learn
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "county_data.csv"
RANDOM_SEED = 42
PERPLEXITIES = [5, 30, 50]
PROFILE_COLORS = {"A_urban": "#e41a1c", "B_suburban": "#377eb8", "C_rural": "#4daf4a"}
FEATURE_COLS = [
    "median_age", "pct_bachelors", "median_hh_income", "pct_poverty",
    "pct_owner_occupied", "pct_employed", "pct_under18", "pct_over65",
    "pct_hispanic", "pct_foreign_born", "pct_renter", "pop_density_log",
    "median_gross_rent", "pct_no_vehicle", "pct_broadband",
]


def run_tsne(X_std: np.ndarray, perplexity: int, seed: int) -> np.ndarray:
    """
    Fit and return a 2D t-SNE embedding.

    init="pca" is strongly recommended -- random init often produces worse
    layouts and is harder to reproduce.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        n_iter=1000,
    )
    return tsne.fit_transform(X_std)


def main_embedding_plot(Y: np.ndarray, df: pd.DataFrame, perplexity: int) -> None:
    """
    Two-panel plot: embedding colored by profile, then colored by income.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for prof, idx_series in df.groupby("profile").groups.items():
        axes[0].scatter(
            Y[idx_series, 0], Y[idx_series, 1],
            c=PROFILE_COLORS[prof], label=prof, s=25, alpha=0.7,
        )
    axes[0].set_title(f"t-SNE (perplexity={perplexity}, colored by profile)")
    axes[0].set_xlabel("tSNE-1 (not interpretable)")
    axes[0].set_ylabel("tSNE-2 (not interpretable)")
    axes[0].legend()

    sc = axes[1].scatter(
        Y[:, 0], Y[:, 1],
        c=df["median_hh_income"], cmap="RdYlGn", s=25, alpha=0.8,
    )
    plt.colorbar(sc, ax=axes[1], label="Median HH Income ($)")
    axes[1].set_title(f"t-SNE (perplexity={perplexity}, colored by income)")
    axes[1].set_xlabel("tSNE-1 (not interpretable)")
    axes[1].set_ylabel("tSNE-2 (not interpretable)")

    plt.tight_layout()
    plt.savefig(
        Path(__file__).parent / f"04_tsne_perp{perplexity}.png",
        dpi=120, bbox_inches="tight",
    )
    plt.show()
    print(f"Saved: 04_tsne_perp{perplexity}.png")


def perplexity_sensitivity_plot(X_std: np.ndarray, df: pd.DataFrame,
                                perplexities: list, seed: int) -> None:
    """
    Side-by-side t-SNE embeddings at multiple perplexity values.

    Low perplexity (5): emphasizes local structure -- tight micro-clusters but
    global layout may be arbitrary.

    High perplexity (50): emphasizes global structure -- clusters merge more but
    relative positions carry more meaning.

    If the three plots look qualitatively similar (same groups separating),
    the cluster structure is robust to hyperparameter choice.
    """
    fig, axes = plt.subplots(1, len(perplexities), figsize=(16, 4))

    for ax, perp in zip(axes, perplexities):
        Y = run_tsne(X_std, perp, seed)
        for prof, idx_series in df.groupby("profile").groups.items():
            ax.scatter(
                Y[idx_series, 0], Y[idx_series, 1],
                c=PROFILE_COLORS[prof], label=prof, s=15, alpha=0.6,
            )
        ax.set_title(f"perplexity = {perp}")
        ax.set_xlabel("tSNE-1")
        ax.set_ylabel("tSNE-2")
        if ax is axes[0]:
            ax.legend(fontsize=8)

    plt.suptitle(
        "t-SNE sensitivity to perplexity\n"
        "Cluster separation should be qualitatively consistent if structure is real",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "04_tsne_perplexity_comparison.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 04_tsne_perplexity_comparison.png")


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} counties")

    X_std = StandardScaler().fit_transform(df[FEATURE_COLS].values)

    print("\n--- Main t-SNE embedding (perplexity=30) ---")
    Y_30 = run_tsne(X_std, perplexity=30, seed=RANDOM_SEED)
    main_embedding_plot(Y_30, df, perplexity=30)

    print("\n--- Perplexity sensitivity comparison ---")
    perplexity_sensitivity_plot(X_std, df, PERPLEXITIES, RANDOM_SEED)

    print(
        "\nReminder: t-SNE axes are meaningless.  Do not use t-SNE coordinates "
        "as input to K-means or other downstream algorithms -- cluster on the "
        "original standardized features instead."
    )
