"""
06_method_comparison.py
========================
Side-by-side comparison of PCA, t-SNE, and UMAP on the same 400-county dataset.

All three methods reduce the same 15-variable standardized feature matrix to 2D.
The comparison illustrates their different trade-offs:

- PCA:   Linear. Axes are interpretable (PC1 = economic gradient, PC2 = urbanicity).
         Clusters may overlap because PCA optimizes for global variance, not separation.
- t-SNE: Nonlinear. Cluster separation is typically clearer.  Axes are meaningless.
         Cannot embed new data without refitting.
- UMAP:  Nonlinear. Similar cluster separation to t-SNE, faster on large datasets.
         Preserves more global structure than t-SNE.  Can embed new data.

Prerequisites
-------------
Run 01_synthetic_county_data.py first to generate county_data.csv.
umap-learn is optional; the plot runs with PCA + t-SNE if UMAP is not installed.

Usage
-----
    python 06_method_comparison.py

Requirements: numpy, pandas, matplotlib, scikit-learn
Optional:     umap-learn
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "county_data.csv"
RANDOM_SEED = 42
TSNE_PERPLEXITY = 30
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
PROFILE_COLORS = {"A_urban": "#e41a1c", "B_suburban": "#377eb8", "C_rural": "#4daf4a"}
FEATURE_COLS = [
    "median_age", "pct_bachelors", "median_hh_income", "pct_poverty",
    "pct_owner_occupied", "pct_employed", "pct_under18", "pct_over65",
    "pct_hispanic", "pct_foreign_born", "pct_renter", "pop_density_log",
    "median_gross_rent", "pct_no_vehicle", "pct_broadband",
]


def try_import_umap():
    """Return UMAP class or None without raising."""
    try:
        from umap import UMAP as _UMAP
        return _UMAP
    except ImportError:
        return None


def compute_embeddings(X_std: np.ndarray, UMAP_cls) -> dict:
    """
    Compute 2D embeddings for all available methods.

    Returns a dict mapping method name to (n_points, 2) array.
    """
    results = {}

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    results["PCA"] = pca.fit_transform(X_std)

    tsne = TSNE(
        n_components=2, perplexity=TSNE_PERPLEXITY,
        learning_rate="auto", init="pca",
        random_state=RANDOM_SEED, n_iter=1000,
    )
    results["t-SNE"] = tsne.fit_transform(X_std)

    if UMAP_cls is not None:
        reducer = UMAP_cls(
            n_components=2, n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST, random_state=RANDOM_SEED,
        )
        results["UMAP"] = reducer.fit_transform(X_std)
    else:
        print("umap-learn not installed -- UMAP panel omitted.")

    return results


def comparison_plot(embeddings: dict, df: pd.DataFrame) -> None:
    """
    Single-row panel with one subplot per method.

    Each panel shows all 400 counties colored by their true demographic profile.
    The comparison highlights how each method trades off interpretability,
    cluster separation, and global structure preservation.
    """
    n_methods = len(embeddings)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, (name, coords) in zip(axes, embeddings.items()):
        for prof, idx_series in df.groupby("profile").groups.items():
            ax.scatter(
                coords[idx_series, 0], coords[idx_series, 1],
                c=PROFILE_COLORS[prof], label=prof, s=20, alpha=0.6,
            )
        ax.set_title(name, fontsize=13)
        ax.set_xlabel(f"{name}-1")
        ax.set_ylabel(f"{name}-2")
        ax.legend(fontsize=8)

        # Annotate axis interpretability
        note = "Axes interpretable" if name == "PCA" else "Axes not interpretable"
        ax.text(0.02, 0.02, note, transform=ax.transAxes,
                fontsize=8, color="gray", style="italic")

    plt.suptitle(
        "Dimension reduction methods compared\n"
        "Same 15 ACS variables, 400 counties -- three different 2D representations",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "06_method_comparison.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 06_method_comparison.png")


if __name__ == "__main__":
    UMAP_cls = try_import_umap()
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} counties")

    X_std = StandardScaler().fit_transform(df[FEATURE_COLS].values)

    print("\nComputing embeddings (t-SNE may take ~30 seconds)...")
    embeddings = compute_embeddings(X_std, UMAP_cls)

    for name, coords in embeddings.items():
        print(f"  {name}: embedding shape {coords.shape}")

    comparison_plot(embeddings, df)
