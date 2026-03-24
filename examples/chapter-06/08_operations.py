"""
08_operations.py
================
Operational applications of dimension reduction and clustering in federal surveys.

Three use cases demonstrated
------------------------------
1. Nonresponse analysis by cluster
   -- Which demographic segments have systematically lower response rates?
   -- These clusters become targets for adaptive design follow-up.

2. Hot-deck imputation classes
   -- Clusters define within-cluster donor pools for hot-deck imputation.
   -- Donors are demographically similar to recipients, reducing imputation bias.

3. Variance reduction from PCA-based stratification
   -- Stratum definitions derived from PCA scores reduce within-stratum variance
      on key survey outcomes (income, response rate).

All three demonstrate the same principle: segment on features (or PCA scores),
then apply the segmentation to an operational decision.  The algorithm clusters;
the analyst names and uses the clusters.

Prerequisites
-------------
Run 01_synthetic_county_data.py first to generate county_data.csv.

Usage
-----
    python 08_operations.py

Requirements: numpy, pandas, matplotlib, scikit-learn
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "county_data.csv"
RANDOM_SEED = 42
N_CLUSTERS = 3  # Use 3 for interpretability in the operational examples
MISSING_FRACTION = 0.10
RESPONSE_RATE_SEED = 99
FEATURE_COLS = [
    "median_age", "pct_bachelors", "median_hh_income", "pct_poverty",
    "pct_owner_occupied", "pct_employed", "pct_under18", "pct_over65",
    "pct_hispanic", "pct_foreign_born", "pct_renter", "pop_density_log",
    "median_gross_rent", "pct_no_vehicle", "pct_broadband",
]


def simulate_response_rates(df: pd.DataFrame, seed: int) -> pd.Series:
    """
    Simulate county-level response rates correlated with demographic profiles.

    Urban counties (high poverty, high density, high renter fraction) tend to
    have lower response rates in real surveys.  This simulation encodes that
    pattern using the known profile labels.
    """
    rng = np.random.default_rng(seed)
    base_rr = {"A_urban": 0.62, "B_suburban": 0.74, "C_rural": 0.68}
    rates = [
        float(np.clip(base_rr[p] + rng.normal(0, 0.05), 0.30, 0.95))
        for p in df["profile"]
    ]
    return pd.Series(rates, index=df.index, name="response_rate")


def nonresponse_analysis(df: pd.DataFrame, cluster_col: str,
                         n_clusters: int) -> None:
    """
    Plot response rates by cluster and income vs. response rate scatter.

    If clusters with lower response rates are demographically coherent
    (e.g., all urban, high-renter), the cluster variable can be used to
    target adaptive design interventions before data collection begins.
    """
    cluster_rr = df.groupby(cluster_col)["response_rate"].agg(["mean", "std"])
    overall_mean = df["response_rate"].mean()
    colors = plt.cm.tab10(np.linspace(0, 0.8, n_clusters))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].bar(
        cluster_rr.index, cluster_rr["mean"],
        yerr=cluster_rr["std"], capsize=5,
        color=[colors[c] for c in range(n_clusters)], edgecolor="white",
    )
    axes[0].axhline(overall_mean, color="firebrick", linestyle="--",
                    label=f"Overall mean = {overall_mean:.3f}")
    axes[0].set_xlabel("K-means cluster")
    axes[0].set_ylabel("Simulated response rate")
    axes[0].set_title("Response rate by demographic cluster\n"
                      "(clusters identify which segments need adaptive follow-up)")
    axes[0].legend()

    for cl in range(n_clusters):
        mask = df[cluster_col] == cl
        axes[1].scatter(
            df.loc[mask, "median_hh_income"],
            df.loc[mask, "response_rate"],
            c=[colors[cl]], label=f"Cluster {cl}", s=20, alpha=0.6,
        )
    axes[1].set_xlabel("Median HH Income ($)")
    axes[1].set_ylabel("Simulated response rate")
    axes[1].set_title("Income vs. response rate by cluster")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "08_nonresponse_analysis.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 08_nonresponse_analysis.png")

    print("\nResponse rate by cluster:")
    print(
        df.groupby(cluster_col)["response_rate"]
        .agg(["mean", "std", "count"])
        .round(3)
        .to_string()
    )
    lowest_cl = cluster_rr["mean"].idxmin()
    print(
        f"\nCluster {lowest_cl} has the lowest response rate "
        f"({cluster_rr.loc[lowest_cl, 'mean']:.3f}).  "
        "Recommendation: prioritize follow-up outreach for counties in this cluster."
    )


def hot_deck_imputation(df: pd.DataFrame, target_var: str,
                        cluster_col: str, missing_frac: float,
                        seed: int) -> None:
    """
    Demonstrate within-cluster hot-deck imputation.

    Hot-deck replaces each missing value with an observed value (a 'donor')
    from the same cluster.  Because all donors share the same cluster assignment,
    they are demographically similar to the recipient -- reducing imputation bias
    compared to drawing donors from the full dataset.

    This is the standard approach used in ACS and CPS imputation.
    Clusters used here are the same K-means clusters from the main analysis.
    """
    rng = np.random.default_rng(seed)
    df_work = df.copy()
    missing_idx = rng.choice(df_work.index, size=int(missing_frac * len(df_work)),
                              replace=False)
    true_values = df_work.loc[missing_idx, target_var].copy()
    df_work.loc[missing_idx, target_var] = np.nan

    # Impute
    imputed_col = df_work[target_var].copy()
    for idx in missing_idx:
        cl = df_work.loc[idx, cluster_col]
        donors = df_work[(df_work[cluster_col] == cl) & df_work[target_var].notna()]
        if len(donors) > 0:
            imputed_col.loc[idx] = donors[target_var].sample(
                1, random_state=int(idx)
            ).values[0]
        else:
            # Fallback: global donor (should be rare with sensible cluster sizes)
            imputed_col.loc[idx] = df_work[target_var].dropna().sample(
                1, random_state=int(idx)
            ).values[0]

    n_imputed = missing_idx.shape[0]
    corr = np.corrcoef(true_values, imputed_col.loc[missing_idx])[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    observed_vals = df_work.loc[~df_work.index.isin(missing_idx), target_var].dropna()
    axes[0].hist(observed_vals, bins=30, alpha=0.6, label="Observed", color="steelblue")
    axes[0].hist(imputed_col.loc[missing_idx], bins=30, alpha=0.6,
                 label="Imputed", color="firebrick")
    axes[0].set_title(
        f"Hot-deck imputation: observed vs. imputed distribution\n"
        f"({n_imputed} values imputed from within-cluster donors)"
    )
    axes[0].set_xlabel(target_var)
    axes[0].legend()

    axes[1].scatter(true_values, imputed_col.loc[missing_idx], alpha=0.7, s=25)
    min_val = min(true_values.min(), imputed_col.loc[missing_idx].min())
    max_val = max(true_values.max(), imputed_col.loc[missing_idx].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect imputation")
    axes[1].set_xlabel(f"True {target_var} (withheld for check)")
    axes[1].set_ylabel(f"Imputed {target_var}")
    axes[1].set_title(f"Imputation accuracy check\nCorrelation with true values: {corr:.3f}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "08_hot_deck_imputation.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 08_hot_deck_imputation.png")
    print(
        f"\nHot-deck summary: {n_imputed} missing values imputed "
        f"using within-cluster donors.  Correlation with true values: {corr:.3f}."
    )


def pca_stratification_variance_reduction(df: pd.DataFrame,
                                           X_std: np.ndarray) -> None:
    """
    Measure variance reduction achieved by PCA-based stratification.

    Creates a 3x3 grid of strata using PC1 and PC2 terciles.
    Variance reduction is measured for median_hh_income and response_rate.

    A large reduction (>30%) means the strata are demographically coherent --
    counties within the same stratum are more similar than counties overall.
    This is the goal of stratification in survey design.
    """
    pca_2 = PCA(n_components=2, random_state=RANDOM_SEED)
    scores_2d = pca_2.fit_transform(X_std)

    df = df.copy()
    df["pc1_stratum"] = pd.qcut(scores_2d[:, 0], q=3,
                                labels=["PC1_low", "PC1_mid", "PC1_high"])
    df["pc2_stratum"] = pd.qcut(scores_2d[:, 1], q=3,
                                labels=["PC2_low", "PC2_mid", "PC2_high"])
    df["pca_stratum"] = df["pc1_stratum"].astype(str) + "_" + df["pc2_stratum"].astype(str)

    print("\n--- PCA-based stratification variance reduction ---")
    for var in ["median_hh_income", "response_rate"]:
        total_var = df[var].var()
        within_var = df.groupby("pca_stratum")[var].var().mean()
        reduction = (1 - within_var / total_var) * 100
        print(f"  {var:<22} | total var = {total_var:>12.1f} | "
              f"within-stratum var = {within_var:>10.1f} | "
              f"reduction = {reduction:.1f}%")


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} counties")

    X_std = StandardScaler().fit_transform(df[FEATURE_COLS].values)

    # Fit K-means on original features
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    df["kmeans_cluster"] = km.fit_predict(X_std)
    print(f"\nK-means cluster distribution (k={N_CLUSTERS}):")
    print(df["kmeans_cluster"].value_counts().sort_index().to_string())

    # Simulate response rates
    df["response_rate"] = simulate_response_rates(df, RESPONSE_RATE_SEED)

    print("\n=== 1. Nonresponse analysis ===")
    nonresponse_analysis(df, "kmeans_cluster", N_CLUSTERS)

    print("\n=== 2. Hot-deck imputation ===")
    hot_deck_imputation(df, "median_hh_income", "kmeans_cluster",
                        MISSING_FRACTION, RANDOM_SEED)

    print("\n=== 3. Variance reduction from PCA stratification ===")
    pca_stratification_variance_reduction(df, X_std)
