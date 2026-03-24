"""
07_clustering.py
================
K-means and hierarchical clustering on county ACS data.

Produces
--------
1. K-means elbow curve (inertia) and silhouette scores for k=2..9
2. K-means cluster assignments visualized on PCA score plot
3. Hierarchical clustering dendrogram (60-county sample, Ward linkage)
4. Hierarchical full-dataset cluster assignments
5. Cluster profiling heatmap (normalized variable means by cluster)
6. Adjusted Rand Index (ARI) comparing K-means vs. hierarchical

Key principle: cluster on original features, not on 2D embeddings.
K-means uses Euclidean distance in the original standardized feature space.
Distances in t-SNE / UMAP 2D space are distorted and not suitable for
clustering algorithms.

Prerequisites
-------------
Run 01_synthetic_county_data.py first to generate county_data.csv.

Usage
-----
    python 07_clustering.py

Requirements: numpy, pandas, matplotlib, scikit-learn, scipy
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "county_data.csv"
RANDOM_SEED = 42
K_RANGE = range(2, 10)
DENDROGRAM_SAMPLE_SIZE = 60
FEATURE_COLS = [
    "median_age", "pct_bachelors", "median_hh_income", "pct_poverty",
    "pct_owner_occupied", "pct_employed", "pct_under18", "pct_over65",
    "pct_hispanic", "pct_foreign_born", "pct_renter", "pop_density_log",
    "median_gross_rent", "pct_no_vehicle", "pct_broadband",
]


def elbow_silhouette_plot(X_std: np.ndarray, k_range: range,
                          seed: int) -> tuple:
    """
    Compute and plot inertia (elbow method) and silhouette scores for K-means.

    Returns the best k by silhouette score and the full list of labels.
    The elbow method and silhouette score are heuristics -- they suggest
    a starting point, but domain expertise should inform the final choice.
    """
    inertia = []
    sil_scores = []
    all_labels = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_std)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(X_std, labels))
        all_labels[k] = labels

    best_k = list(k_range)[int(np.argmax(sil_scores))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(k_range, inertia, "o-", color="steelblue")
    axes[0].set_xlabel("Number of clusters k")
    axes[0].set_ylabel("Inertia (within-cluster SSE)")
    axes[0].set_title("Elbow method: look for the bend")
    axes[0].axvline(best_k, color="firebrick", linestyle="--",
                    label=f"Best k = {best_k}")
    axes[0].legend()

    axes[1].plot(k_range, sil_scores, "s-", color="firebrick")
    axes[1].axhline(
        max(sil_scores), color="gray", linestyle="--",
        label=f"Max sil={max(sil_scores):.3f} at k={best_k}",
    )
    axes[1].set_xlabel("Number of clusters k")
    axes[1].set_ylabel("Silhouette score (higher = better)")
    axes[1].set_title("Silhouette scores")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "07_elbow_silhouette.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: 07_elbow_silhouette.png  (best k by silhouette = {best_k})")

    return best_k, all_labels


def cluster_score_plot(scores_2d: np.ndarray, labels: np.ndarray,
                       best_k: int) -> None:
    """
    K-means cluster assignments overlaid on PCA score plot.

    Note: the clusters were computed on the 15-variable standardized feature
    space, not on the 2D PCA coordinates.  The PCA plot is used only for
    visualization.
    """
    colors = plt.cm.tab10(np.linspace(0, 0.8, best_k))
    fig, ax = plt.subplots(figsize=(8, 6))

    for cl in range(best_k):
        mask = labels == cl
        ax.scatter(
            scores_2d[mask, 0], scores_2d[mask, 1],
            c=[colors[cl]], label=f"Cluster {cl}", s=30, alpha=0.7,
        )
    ax.set_title(f"K-means (k={best_k}) clusters on PCA coordinates\n"
                 "(clusters computed on 15 original features, not on PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "07_kmeans_pca.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 07_kmeans_pca.png")


def dendrogram_plot(X_std: np.ndarray, best_k: int, seed: int) -> None:
    """
    Ward linkage dendrogram on a random subsample of counties.

    The dendrogram shows the tree of merges.  You read it top-down:
    the height of each join indicates how dissimilar the merged groups are.
    A horizontal cut at a given height yields the corresponding number of clusters.
    The red dashed line marks the cut that gives best_k clusters.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_std), size=min(DENDROGRAM_SAMPLE_SIZE, len(X_std)),
                     replace=False)
    X_sample = X_std[idx]

    Z = linkage(X_sample, method="ward")

    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(
        Z, ax=ax,
        leaf_rotation=90, leaf_font_size=7,
        color_threshold=Z[-best_k, 2],
    )
    ax.set_title(
        f"Hierarchical clustering dendrogram\n"
        f"(n={len(idx)} sample, Ward linkage, cut at k={best_k})"
    )
    ax.set_xlabel("County index (subsample)")
    ax.set_ylabel("Distance (Ward linkage)")
    ax.axhline(
        Z[-best_k, 2], color="firebrick", linestyle="--",
        label=f"Cut at k={best_k} clusters",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "07_dendrogram.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 07_dendrogram.png")


def cluster_profile_heatmap(df: pd.DataFrame, cluster_col: str,
                             feature_cols: list, best_k: int) -> None:
    """
    Normalized heatmap of variable means by cluster.

    Each cell shows where a cluster sits on each variable (0 = lowest cluster
    mean, 1 = highest cluster mean).  Read across rows to characterize a cluster.
    A cluster with high income, high education, high density, and low poverty
    is clearly urban.  Naming clusters from their profiles is an analyst task.
    """
    cluster_summary = df.groupby(cluster_col)[feature_cols].mean()
    cluster_norm = pd.DataFrame(
        MinMaxScaler().fit_transform(cluster_summary.T).T,
        index=cluster_summary.index,
        columns=cluster_summary.columns,
    )

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(cluster_norm.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(best_k))
    ax.set_yticklabels([f"Cluster {c}" for c in range(best_k)])
    plt.colorbar(im, ax=ax, label="Normalized mean (0=low, 1=high)")
    ax.set_title(
        "Cluster profiles: average ACS variable by cluster (normalized)\n"
        "Read each row to name the cluster; read each column to see which "
        "clusters differ on that variable"
    )
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "07_cluster_profiles.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 07_cluster_profiles.png")


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} counties")

    X_std = StandardScaler().fit_transform(df[FEATURE_COLS].values)

    # K-means selection
    print("\n--- Elbow and silhouette plots ---")
    best_k, all_labels = elbow_silhouette_plot(X_std, K_RANGE, RANDOM_SEED)

    # Final K-means
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    df["kmeans_cluster"] = km_final.fit_predict(X_std)
    print(f"\nK-means cluster distribution:\n"
          f"{df['kmeans_cluster'].value_counts().sort_index().to_string()}")

    # Score plot (PCA coordinates for visualization)
    pca_2 = PCA(n_components=2, random_state=RANDOM_SEED)
    scores_2d = pca_2.fit_transform(X_std)

    print("\n--- Cluster visualization on PCA ---")
    cluster_score_plot(scores_2d, df["kmeans_cluster"].values, best_k)

    print("\n--- Dendrogram ---")
    dendrogram_plot(X_std, best_k, RANDOM_SEED)

    # Hierarchical clustering
    hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    df["hier_cluster"] = hier.fit_predict(X_std)

    ari = adjusted_rand_score(df["kmeans_cluster"], df["hier_cluster"])
    print(f"\nAdjusted Rand Index (K-means vs. hierarchical): {ari:.3f}")
    print("(1.0 = perfect agreement, 0.0 = random chance)")
    if ari > 0.8:
        print("High ARI: both methods agree -- cluster structure is robust.")
    elif ari > 0.5:
        print("Moderate ARI: broad agreement with some boundary disagreement.")
    else:
        print("Low ARI: methods disagree substantially -- inspect both solutions.")

    print("\n--- Cluster profiles heatmap ---")
    cluster_profile_heatmap(df, "kmeans_cluster", FEATURE_COLS, best_k)

    key_vars = ["median_hh_income", "pct_bachelors", "pct_poverty",
                "median_age", "pop_density_log", "pct_broadband"]
    print("\nCluster summary (key variables):")
    print(df.groupby("kmeans_cluster")[key_vars].mean().round(1).to_string())
