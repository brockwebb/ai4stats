"""
03_pca.py
=========
Full PCA pipeline on the synthetic county ACS dataset.

Produces
--------
1. Scree plot + cumulative explained variance
2. PC1 and PC2 loadings heatmap (bar charts)
3. Score plot colored by demographic profile and by income
4. Biplot (scores + loading arrows)
5. PCA-based stratification: 3x3 grid on PC1/PC2 terciles, with variance
   reduction calculation for median household income

Prerequisites
-------------
Run 01_synthetic_county_data.py first to generate county_data.csv.

Usage
-----
    python 03_pca.py

Requirements: numpy, pandas, matplotlib, scikit-learn
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "county_data.csv"
RANDOM_SEED = 42
PROFILE_COLORS = {"A_urban": "#e41a1c", "B_suburban": "#377eb8", "C_rural": "#4daf4a"}
FEATURE_COLS = [
    "median_age", "pct_bachelors", "median_hh_income", "pct_poverty",
    "pct_owner_occupied", "pct_employed", "pct_under18", "pct_over65",
    "pct_hispanic", "pct_foreign_born", "pct_renter", "pop_density_log",
    "median_gross_rent", "pct_no_vehicle", "pct_broadband",
]


def scree_plot(explained: np.ndarray, cumulative: np.ndarray) -> None:
    """
    Plot scree (bar + line) and cumulative explained variance.

    The elbow in the scree plot suggests how many PCs capture the dominant
    structure.  The cumulative plot shows thresholds for 80% and 90% retention.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    n_pcs = len(explained)
    axes[0].bar(range(1, n_pcs + 1), explained, color="steelblue", edgecolor="white")
    axes[0].plot(range(1, n_pcs + 1), explained, "o-", color="firebrick", ms=5)
    axes[0].axhline(0.10, color="gray", linestyle="--", label="10% threshold")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title("Scree plot")
    axes[0].legend()

    axes[1].plot(range(1, n_pcs + 1), cumulative, "s-", color="steelblue")
    axes[1].axhline(0.80, color="firebrick", linestyle="--", label="80% threshold")
    axes[1].axhline(0.90, color="darkorange", linestyle="--", label="90% threshold")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("Cumulative explained variance")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "03_scree_plot.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 03_scree_plot.png")


def loadings_plot(loadings: pd.DataFrame, explained: np.ndarray) -> None:
    """
    Horizontal bar charts showing PC1 and PC2 loadings for each ACS variable.

    Red bars = negative loading (variable inversely related to this PC direction).
    Blue bars = positive loading.
    Variables with large absolute loadings define what the component measures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, col in zip(axes, ["PC1", "PC2"]):
        pc_idx = int(col[-1]) - 1
        vals = loadings[col].sort_values()
        bar_colors = ["firebrick" if v < 0 else "steelblue" for v in vals]
        ax.barh(vals.index, vals.values, color=bar_colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Loading weight")
        ax.set_title(
            f"{col} loadings  ({explained[pc_idx] * 100:.1f}% variance)"
        )

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "03_loadings.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 03_loadings.png")


def score_plot(scores: np.ndarray, df: pd.DataFrame, explained: np.ndarray) -> None:
    """
    Two score plots side by side: one colored by known profile, one by income.

    In a real workflow, the profile column does not exist -- you would color by
    known geographic attributes (region, urbanicity) to interpret what the PCs
    are capturing.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for prof, idx_series in df.groupby("profile").groups.items():
        axes[0].scatter(
            scores[idx_series, 0], scores[idx_series, 1],
            c=PROFILE_COLORS[prof], label=prof, s=25, alpha=0.7,
        )
    axes[0].set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    axes[0].set_title("PCA score plot (colored by demographic profile)")
    axes[0].legend()

    sc = axes[1].scatter(
        scores[:, 0], scores[:, 1],
        c=df["median_hh_income"], cmap="RdYlGn", s=25, alpha=0.8,
    )
    plt.colorbar(sc, ax=axes[1], label="Median HH Income ($)")
    axes[1].set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    axes[1].set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    axes[1].set_title("PCA score plot (colored by income)")

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "03_score_plot.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 03_score_plot.png")


def biplot(scores: np.ndarray, loadings: pd.DataFrame,
           df: pd.DataFrame, explained: np.ndarray) -> None:
    """
    PCA biplot: county scores overlaid with loading arrows for each ACS variable.

    Points close together are demographically similar.
    Arrows pointing in the same direction indicate positively correlated variables.
    An arrow pointing toward a cluster means those counties score high on that variable.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for prof, idx_series in df.groupby("profile").groups.items():
        ax.scatter(
            scores[idx_series, 0], scores[idx_series, 1],
            c=PROFILE_COLORS[prof], label=prof, s=20, alpha=0.5,
        )

    scale = 3.5
    for var, row in loadings.iterrows():
        ax.annotate(
            "", xy=(row["PC1"] * scale, row["PC2"] * scale), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )
        ax.text(
            row["PC1"] * scale * 1.12, row["PC2"] * scale * 1.12,
            var, fontsize=7.5, ha="center", va="center", color="black",
        )

    ax.axhline(0, color="gray", lw=0.5, linestyle="--")
    ax.axvline(0, color="gray", lw=0.5, linestyle="--")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    ax.set_title("PCA biplot: county scores + ACS variable loadings")
    ax.legend(title="Profile", loc="upper right")
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "03_biplot.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 03_biplot.png")


def stratification_analysis(scores: np.ndarray, df: pd.DataFrame) -> None:
    """
    Use PC1 and PC2 terciles as a stratification grid and measure variance reduction.

    Stratification groups geographies by demographic similarity.  If the PCA
    axes capture the main sources of variation, within-stratum variance on key
    outcomes (income, response rate) should be substantially lower than total
    variance -- meaning the stratification is effective.
    """
    df = df.copy()
    df["pc1_stratum"] = pd.qcut(scores[:, 0], q=3,
                                labels=["PC1_low", "PC1_mid", "PC1_high"])
    df["pc2_stratum"] = pd.qcut(scores[:, 1], q=3,
                                labels=["PC2_low", "PC2_mid", "PC2_high"])
    df["pca_stratum"] = df["pc1_stratum"].astype(str) + "_" + df["pc2_stratum"].astype(str)

    total_var = df["median_hh_income"].var()
    within_var = df.groupby("pca_stratum")["median_hh_income"].var().mean()
    reduction = (1 - within_var / total_var) * 100

    print("\n--- PCA-based stratification ---")
    print(f"Number of strata: {df['pca_stratum'].nunique()}")
    print(f"Total income variance:           {total_var:>12,.0f}")
    print(f"Mean within-stratum variance:    {within_var:>12,.0f}")
    print(f"Variance reduction:              {reduction:>11.1f}%")
    print(
        "\nInterpretation: counties in the same PCA-based stratum are demographically "
        "similar, so income variance within each stratum is lower than overall. "
        f"A {reduction:.0f}% variance reduction means the stratification is capturing "
        "real demographic structure."
    )


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} counties")

    X = df[FEATURE_COLS].values
    X_std = StandardScaler().fit_transform(X)

    # Full PCA for scree/variance analysis
    pca_full = PCA(random_state=RANDOM_SEED)
    pca_full.fit(X_std)
    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    n80 = np.searchsorted(cumulative, 0.80) + 1
    n90 = np.searchsorted(cumulative, 0.90) + 1
    print(f"\nPC1 explains {explained[0]*100:.1f}% | PC2 {explained[1]*100:.1f}%")
    print(f"Components for 80% variance: {n80} | for 90%: {n90}")

    print("\n--- Scree plot ---")
    scree_plot(explained, cumulative)

    # 2-component PCA for visualization
    pca_2 = PCA(n_components=2, random_state=RANDOM_SEED)
    scores_2d = pca_2.fit_transform(X_std)

    loadings = pd.DataFrame(
        pca_2.components_.T,
        index=FEATURE_COLS,
        columns=["PC1", "PC2"],
    )

    print("\n--- Loadings ---")
    loadings_plot(loadings, explained)

    print("\nTop PC1 loadings:")
    print(loadings["PC1"].abs().sort_values(ascending=False).head(5).round(3).to_string())

    print("\n--- Score plot ---")
    score_plot(scores_2d, df, explained)

    print("\n--- Biplot ---")
    biplot(scores_2d, loadings, df, explained)

    print("\n--- Stratification analysis ---")
    stratification_analysis(scores_2d, df)
