"""
09_exercise.py
==============
Chapter 6 in-class activity: 50-county adaptive survey design exercise.

Scenario
--------
You are advising a regional Census office on adaptive survey design for the
next ACS collection cycle.  You have 50 simulated counties described by
8 ACS-style variables and a pre-computed scree plot result showing the first
3 PCs explain 68% of variance.  You also have a pre-computed cluster profile
table for a 4-cluster K-means solution.

Your job is to evaluate a proposed segmentation -- not just run the code.

Exercise questions (answer before running the solution)
-------------------------------------------------------
1. The first 3 PCs explain 68% of variance with an elbow at PC3.
   How many PCs would you use for stratification? Justify your answer.

2. A 4-cluster solution produces these profiles (see CLUSTER_PROFILE_TABLE below).
   Name each cluster.  Which should get priority follow-up resources?

3. A colleague proposes clustering using the t-SNE 2D coordinates instead of
   the original 15 variables.  What is wrong with this approach?

4. The cluster solution changes substantially when you change the random seed.
   What does this tell you?  What would you do?

5. (Optional) Run the full pipeline using the solution code below and verify
   your cluster naming against the computed profiles.

Run the full solution
---------------------
Set SHOW_SOLUTION = True and re-run this file to execute the full pipeline.

Usage
-----
    python 09_exercise.py

Requirements: numpy, pandas, matplotlib, scikit-learn
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 2024
N_ACTIVITY_COUNTIES = 50
SHOW_SOLUTION = True  # Set to False to suppress solution plots during exercise

# Pre-computed cluster profile table (provided to students; solution verifies it)
CLUSTER_PROFILE_TABLE = """
Pre-computed 4-cluster profile (K-means on 8 ACS variables, 50 counties):

Cluster  | median_income | pct_poverty | pop_density_log | pct_broadband | pct_over65
---------|---------------|-------------|-----------------|---------------|-----------
    0    |   ~$72,000    |    ~8%      |      ~7.1       |     ~87%      |   ~12%
    1    |   ~$44,000    |   ~20%      |      ~4.5       |     ~62%      |   ~22%
    2    |   ~$58,000    |   ~13%      |      ~5.5       |     ~76%      |   ~17%
    3    |   ~$38,000    |   ~24%      |      ~3.2       |     ~58%      |   ~26%

Question: Name each cluster.  Which needs the most follow-up resources?
"""

ACTIVITY_FEATURE_COLS = [
    "median_age", "pct_bachelors", "median_income", "pct_poverty",
    "pop_density_log", "pct_renter", "pct_broadband", "pct_over65",
]


def build_activity_dataset(n: int, seed: int) -> pd.DataFrame:
    """
    Generate the 50-county activity dataset with 8 ACS-style variables
    and a simulated response rate.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "county_id":       [f"C{str(i).zfill(2)}" for i in range(n)],
        "median_age":      rng.normal(40, 8, n).clip(25, 65),
        "pct_bachelors":   rng.normal(28, 12, n).clip(5, 65),
        "median_income":   rng.normal(55000, 15000, n).clip(25000, 110000),
        "pct_poverty":     rng.normal(14, 6, n).clip(3, 40),
        "pop_density_log": rng.normal(5, 2, n).clip(1, 9),
        "pct_renter":      rng.normal(35, 12, n).clip(10, 70),
        "pct_broadband":   rng.normal(74, 10, n).clip(40, 97),
        "pct_over65":      rng.normal(17, 5, n).clip(5, 40),
    })
    # Response rate correlated with poverty, density, and broadband access
    noise = rng.normal(0, 0.04, n)
    df["response_rate"] = np.clip(
        0.72
        - 0.003 * (df["pct_poverty"] - 14)
        - 0.001 * (df["pop_density_log"] - 5)
        + 0.001 * (df["pct_broadband"] - 74)
        + noise,
        0.30, 0.95,
    )
    return df


def solution_pipeline(df: pd.DataFrame) -> None:
    """
    Full solution: PCA scree, K-means clustering, cluster profiles, response rates.
    """
    X = df[ACTIVITY_FEATURE_COLS].values
    X_std = StandardScaler().fit_transform(X)

    # Step 1: PCA scree
    pca_full = PCA(random_state=RANDOM_SEED)
    pca_full.fit(X_std)
    ev = pca_full.explained_variance_ratio_
    cumev = np.cumsum(ev)
    n80 = int(np.searchsorted(cumev, 0.80)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(1, len(ev) + 1), ev, color="steelblue", edgecolor="white")
    axes[0].set_title("Scree plot (50-county activity)")
    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("Explained variance ratio")

    axes[1].plot(range(1, len(ev) + 1), cumev, "o-", color="steelblue")
    axes[1].axhline(0.80, color="firebrick", linestyle="--",
                    label=f"{n80} PCs for 80%")
    axes[1].axhline(cumev[2], color="darkorange", linestyle="--",
                    label=f"PC1-3 = {cumev[2]*100:.0f}%")
    axes[1].legend()
    axes[1].set_title("Cumulative variance")
    axes[1].set_xlabel("PC")
    axes[1].set_ylabel("Cumulative")
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "09_exercise_scree.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: 09_exercise_scree.png")
    print(f"PCs for 80% variance: {n80}")
    print(f"PC1-3 combined: {cumev[2]*100:.1f}%  (elbow visible at PC3)")

    # Step 2: K-means k=4
    pca_2 = PCA(n_components=2, random_state=RANDOM_SEED)
    scores_2d = pca_2.fit_transform(X_std)

    km = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=10)
    df = df.copy()
    df["cluster"] = km.fit_predict(X_std)

    clr = {0: "#e41a1c", 1: "#377eb8", 2: "#4daf4a", 3: "#ff7f00"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    # PCA score plot
    for c, grp in df.groupby("cluster"):
        idx = grp.index
        axes[0].scatter(
            scores_2d[idx, 0], scores_2d[idx, 1],
            c=clr[c], label=f"Cluster {c}", s=60, alpha=0.8,
        )
        for i in idx:
            axes[0].text(
                scores_2d[i, 0] + 0.05, scores_2d[i, 1] + 0.05,
                df.loc[i, "county_id"], fontsize=6,
            )
    axes[0].set_title("PCA score plot: 50 counties, k=4 clusters")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=8)

    # Response rate by cluster
    rr_cl = df.groupby("cluster")["response_rate"].agg(["mean", "std"])
    axes[1].bar(
        rr_cl.index, rr_cl["mean"],
        yerr=rr_cl["std"], capsize=5,
        color=[clr[c] for c in rr_cl.index],
    )
    axes[1].axhline(
        df["response_rate"].mean(), color="black", linestyle="--",
        label=f"Overall={df['response_rate'].mean():.2f}",
    )
    axes[1].set_title("Response rate by cluster")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Response rate")
    axes[1].legend()

    # Cluster heatmap
    cl_summary = df.groupby("cluster")[ACTIVITY_FEATURE_COLS].mean()
    cl_norm = pd.DataFrame(
        MinMaxScaler().fit_transform(cl_summary.T).T,
        index=cl_summary.index, columns=cl_summary.columns,
    )
    im = axes[2].imshow(cl_norm.values, aspect="auto", cmap="RdYlGn")
    axes[2].set_xticks(range(len(ACTIVITY_FEATURE_COLS)))
    axes[2].set_xticklabels(ACTIVITY_FEATURE_COLS, rotation=45, ha="right", fontsize=8)
    axes[2].set_yticks(range(4))
    axes[2].set_yticklabels(["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"])
    plt.colorbar(im, ax=axes[2], label="Normalized mean")
    axes[2].set_title("Cluster profiles")

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "09_exercise_solution.png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: 09_exercise_solution.png")

    # Summary
    lowest_cl = rr_cl["mean"].idxmin()
    print(
        f"\nCluster {lowest_cl} has lowest mean response rate "
        f"({rr_cl.loc[lowest_cl, 'mean']:.3f}).  "
        "Concentrate follow-up resources there."
    )
    print("\nCluster profiles (key variables):")
    print(
        df.groupby("cluster")[
            ["median_income", "pct_poverty", "pop_density_log",
             "pct_broadband", "response_rate"]
        ].mean().round(2).to_string()
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 6 Activity: 50-county adaptive design exercise")
    print("=" * 60)

    df_activity = build_activity_dataset(N_ACTIVITY_COUNTIES, RANDOM_SEED)
    print(f"\nActivity dataset: {df_activity.shape}")
    print(df_activity.head().to_string())

    print("\n" + CLUSTER_PROFILE_TABLE)

    if SHOW_SOLUTION:
        print("\n--- Full solution pipeline ---")
        solution_pipeline(df_activity)
    else:
        print(
            "\nSHOW_SOLUTION = False.  Answer the exercise questions, then "
            "set SHOW_SOLUTION = True to run the full pipeline."
        )
