"""
Chapter 3 -- Feature Importance Stability Analysis
===================================================
Tests whether importance rankings are stable across multiple runs:
- Bootstrapped permutation importance (30 seeds): mean rank and 95% CI per feature
- Repeated k-fold (5 folds x 10 seeds = 50 fits): per-fold importance distribution
- SHAP stability: SHAP values on 5 bootstrap samples of training data
- Output: stability table (feature, mean rank, rank std, top-3 frequency)

Dataset: tract-level nonresponse targeting (seed=2025, n=300 tracts).
Same dataset as the chapter activity (Section 10) and 10_tract_exercise.py.

Run: python 05_stability_analysis.py
Requires: pip install shap
"""

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split

warnings.filterwarnings("ignore")

try:
    import shap
except ImportError:
    print("ERROR: shap is not installed.")
    print("Install with:  pip install shap")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Reproduce the tract dataset from the chapter activity (seed=2025, n=300)
# ---------------------------------------------------------------------------
np.random.seed(2025)
n_tracts = 300

tract_data = pd.DataFrame(
    {
        "tract_id": [f"T{str(i).zfill(3)}" for i in range(n_tracts)],
        "pct_renters": np.random.normal(35, 15, n_tracts).clip(5, 90),
        "median_age": np.random.normal(40, 8, n_tracts).clip(22, 70),
        "pct_foreign_born": np.random.normal(15, 10, n_tracts).clip(0, 60),
        "pct_bachelors": np.random.normal(30, 12, n_tracts).clip(5, 75),
        "pop_density_log": np.random.normal(6, 2, n_tracts).clip(1, 10),
        "prior_rr": np.random.normal(0.72, 0.08, n_tracts).clip(0.40, 0.95),
        "contact_attempts": np.random.poisson(2.5, n_tracts).clip(1, 8),
    }
)

logit_tract = (
    -1.5
    + 0.04 * tract_data["pct_renters"]
    + 0.03 * tract_data["pct_foreign_born"]
    - 2.0 * (tract_data["prior_rr"] - 0.72)
    + 0.15 * tract_data["contact_attempts"]
    + np.random.normal(0, 0.4, n_tracts)
)
tract_data["low_response"] = (1 / (1 + np.exp(-logit_tract)) > 0.45).astype(int)

FEATURES = [
    "pct_renters",
    "median_age",
    "pct_foreign_born",
    "pct_bachelors",
    "pop_density_log",
    "prior_rr",
    "contact_attempts",
]

X = tract_data[FEATURES].values
y = tract_data["low_response"].values
feature_names = FEATURES

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Tract dataset: {tract_data.shape}")
print(f"Low-response tracts: {y.mean():.1%}")
print(f"Train: {len(X_train)}  Test: {len(X_test)}")
print()

# ===========================================================================
# 1. Bootstrapped permutation importance (30 random seeds)
#    Each iteration: bootstrap-resample training data, fit RF, compute
#    permutation importance on the held-out test set. Collect each feature's
#    rank within that run.
# ===========================================================================
N_BOOTSTRAP = 30
print(f"Running {N_BOOTSTRAP} bootstrapped permutation importance runs ...")

all_perm_ranks = []

for seed in range(N_BOOTSTRAP):
    rng = np.random.RandomState(seed)
    boot_idx = rng.choice(len(X_train), size=len(X_train), replace=True)

    rf = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train[boot_idx], y_train[boot_idx])

    perm = permutation_importance(
        rf, X_test, y_test, n_repeats=5, random_state=seed, scoring="roc_auc"
    )
    ranks = pd.Series(perm.importances_mean, index=feature_names).rank(
        ascending=False, method="min"
    )
    all_perm_ranks.append(ranks)

rank_df = pd.DataFrame(all_perm_ranks)  # shape: (30, n_features)

mean_rank = rank_df.mean()
std_rank = rank_df.std()
top3_freq = (rank_df <= 3).sum()

stability_table = pd.DataFrame(
    {
        "mean_rank": mean_rank.round(2),
        "rank_std": std_rank.round(2),
        "top3_frequency": top3_freq,
        "top3_pct": (top3_freq / N_BOOTSTRAP * 100).round(0).astype(int),
    }
).sort_values("mean_rank")

print()
print("=" * 70)
print(f"Bootstrapped Permutation Importance Stability (n={N_BOOTSTRAP} runs)")
print("=" * 70)
print(stability_table.to_string())
print()
print("Notes:")
print("  - mean_rank ~1.0, rank_std < 0.5: stable top feature.")
print("  - rank_std > 1.0: ranking fluctuates meaningfully -- do not assert strict")
print("    ordering of features with overlapping confidence intervals.")
print()

# Box plots of rank distributions
fig, ax = plt.subplots(figsize=(10, 5))
rank_df[stability_table.index].boxplot(ax=ax)
ax.set_ylabel("Rank (1 = most important)")
ax.set_title(
    f"Permutation Importance Rank Distribution ({N_BOOTSTRAP} bootstrap runs)\n"
    "Tract Nonresponse Model"
)
ax.invert_yaxis()  # rank 1 at top
plt.tight_layout()
plt.savefig("stability_rank_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: stability_rank_boxplots.png")
print()

# ===========================================================================
# 2. Repeated k-fold stability (5-fold x 10 seeds = 50 total fits)
#    Collect per-fold Gini importances. This mirrors the "practical pattern for
#    federal reports" from the chapter.
# ===========================================================================
N_SEEDS = 10
N_FOLDS = 5
print(
    f"Running repeated k-fold ({N_FOLDS} folds x {N_SEEDS} seeds = "
    f"{N_FOLDS * N_SEEDS} total fits) ..."
)

kfold_ranks = []

for seed in range(N_SEEDS):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for train_idx, _ in kf.split(X_train):
        rf_fold = RandomForestClassifier(
            n_estimators=50,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
        rf_fold.fit(X_train[train_idx], y_train[train_idx])
        gini = pd.Series(rf_fold.feature_importances_, index=feature_names)
        ranks = gini.rank(ascending=False, method="min")
        kfold_ranks.append(ranks)

kfold_rank_df = pd.DataFrame(kfold_ranks)  # shape: (50, n_features)

kfold_summary = pd.DataFrame(
    {
        "mean_rank": kfold_rank_df.mean().round(2),
        "rank_std": kfold_rank_df.std().round(2),
        "top3_frequency": (kfold_rank_df <= 3).sum(),
    }
).sort_values("mean_rank")

print()
print("=" * 70)
print(f"Repeated k-fold Gini Importance Stability ({N_SEEDS}x{N_FOLDS}={N_SEEDS*N_FOLDS} fits)")
print("=" * 70)
print(kfold_summary.to_string())
print()

# ===========================================================================
# 3. SHAP stability across 5 bootstrap samples
#    Fit 5 Random Forests on 5 bootstrap samples of training data. Compute
#    SHAP values for each. Compare the top-3 feature rankings across samples.
# ===========================================================================
N_SHAP_BOOTS = 5
print(f"Computing SHAP stability across {N_SHAP_BOOTS} bootstrap samples ...")

shap_mean_ranks = []

for seed in range(N_SHAP_BOOTS):
    boot_idx = np.random.RandomState(seed).choice(
        len(X_train), size=len(X_train), replace=True
    )
    rf_boot = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )
    rf_boot.fit(X_train[boot_idx], y_train[boot_idx])

    explainer = shap.TreeExplainer(rf_boot)
    sv = explainer.shap_values(X_test)
    if isinstance(sv, list):
        shap_vals_class1 = sv[1]
    else:
        shap_vals_class1 = sv

    mean_abs = pd.Series(
        np.abs(shap_vals_class1).mean(axis=0), index=feature_names
    )
    ranks = mean_abs.rank(ascending=False, method="min")
    shap_mean_ranks.append(ranks)

shap_rank_df = pd.DataFrame(shap_mean_ranks)  # shape: (5, n_features)

shap_stability = pd.DataFrame(
    {
        "mean_rank": shap_rank_df.mean().round(2),
        "rank_std": shap_rank_df.std().round(2),
        "top3_frequency": (shap_rank_df <= 3).sum(),
    }
).sort_values("mean_rank")

print()
print("=" * 70)
print(f"SHAP Importance Rank Stability ({N_SHAP_BOOTS} bootstrap samples)")
print("=" * 70)
print(shap_stability.to_string())
print()

# Heatmap: rows = bootstrap sample, columns = feature, cell = rank
fig, ax = plt.subplots(figsize=(9, 4))
heatmap_data = shap_rank_df[shap_stability.index]
im = ax.imshow(
    heatmap_data.values, aspect="auto", cmap="YlOrRd_r",
    vmin=1, vmax=len(feature_names)
)
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels(shap_stability.index, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(N_SHAP_BOOTS))
ax.set_yticklabels([f"Bootstrap {i + 1}" for i in range(N_SHAP_BOOTS)])
plt.colorbar(im, ax=ax, label="Rank (1 = most important)")
ax.set_title(
    f"SHAP Feature Importance Rank Stability\n({N_SHAP_BOOTS} bootstrap samples, "
    "Tract Nonresponse Model)"
)
for i in range(N_SHAP_BOOTS):
    for j in range(len(feature_names)):
        ax.text(
            j, i, f"{int(heatmap_data.iloc[i, j])}",
            ha="center", va="center", fontsize=11, fontweight="bold",
        )
plt.tight_layout()
plt.savefig("stability_shap_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: stability_shap_heatmap.png")
print()

# ===========================================================================
# 4. Final stability summary table (for methodology documentation)
# ===========================================================================
print("=" * 70)
print("STABILITY SUMMARY TABLE (for methodology appendix)")
print("=" * 70)
print(
    f"Feature importance rankings reported as mean rank +/- std across\n"
    f"{N_BOOTSTRAP} bootstrapped permutation importance runs.\n"
)
print(f"{'Feature':<22} {'Mean rank':>10} {'Rank std':>10} {'Top-3 / 30':>14}")
print("-" * 60)
for feat in stability_table.index:
    row = stability_table.loc[feat]
    print(
        f"{feat:<22} {row['mean_rank']:>10.2f} {row['rank_std']:>10.2f} "
        f"{int(row['top3_frequency']):>6}/{N_BOOTSTRAP}"
    )
print()
print("Interpretation:")
print("  - Features with mean_rank < 2 and rank_std < 0.5 are stable top predictors.")
print("  - Report these as 'strongly supported by the stability analysis'.")
print("  - Features whose 95% rank CI overlaps with another feature's CI should not")
print("    be ranked relative to each other in a policy report.")
print("  - Apply the same principle used for survey estimate confidence intervals:")
print("    importance is an estimate; report it as one.")
print()
print("Saved figures:")
print("  stability_rank_boxplots.png  -- permutation rank distributions (30 runs)")
print("  stability_shap_heatmap.png   -- SHAP rank stability (5 bootstrap samples)")
