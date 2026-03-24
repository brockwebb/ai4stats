"""
Chapter 3 — Computational Scaling of Random Forests
=====================================================
Demonstrates how Random Forest fit time scales with n_estimators, max_depth,
dataset size, and max_features. Provides a timing table and extrapolation
guidance for large federal datasets.

Why this matters for federal work
----------------------------------
Federal datasets are millions of records, not 1,200. The compute profile of
200 trees × depth 15 × 50 features × 3M records is very different from a
laptop experiment. This script shows where the costs actually come from and
how to subsample intelligently while verifying that findings remain valid.

Key outputs
-----------
1. Timing table: fit time vs. n_estimators × max_depth × dataset size.
2. max_features effect: "sqrt" vs. all features.
3. Subsampling validation: importance rankings at 25%, 50%, 75%, 100% of data.
4. Extrapolation note: projecting from n=1,200 to n=3M.

Requirements
------------
Python 3.9+, numpy, pandas, matplotlib, scikit-learn
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reproduce the chapter dataset (seed=42, n=1200)
# ---------------------------------------------------------------------------
np.random.seed(42)
n = 1200

states = np.random.choice(
    ["California", "Texas", "New York", "Florida", "Illinois"],
    size=n, p=[0.20, 0.20, 0.18, 0.17, 0.25],
)
age = np.random.normal(42, 14, n).clip(18, 80).astype(int)
education_years = np.random.choice(
    [9, 12, 14, 16, 18], size=n, p=[0.10, 0.35, 0.20, 0.25, 0.10]
)
hours_per_week = np.random.normal(38, 10, n).clip(0, 80).astype(int)
urban = np.random.binomial(1, 0.72, n)
contact_attempts = np.random.poisson(2, n).clip(1, 7)
prior_response = np.random.binomial(1, 0.68, n)

log_income = (
    9.2
    + 0.04 * (education_years - 12)
    + 0.008 * age
    + 0.003 * hours_per_week
    + np.random.normal(0, 0.35, n)
)
income = np.exp(log_income).clip(10_000, 250_000).astype(int)

logit_nr = (
    -0.5
    + 0.25 * contact_attempts
    - 1.2 * prior_response
    - 0.3 * urban
    + 0.01 * (age - 42)
    + np.random.normal(0, 0.3, n)
)
prob_nonresponse = 1 / (1 + np.exp(-logit_nr))
responded = (np.random.uniform(size=n) > prob_nonresponse).astype(int)

df = pd.DataFrame(
    {
        "state": states,
        "age": age,
        "education_years": education_years,
        "hours_per_week": hours_per_week,
        "urban": urban,
        "contact_attempts": contact_attempts,
        "prior_response": prior_response,
        "income": income,
        "responded": responded,
    }
)

FEATURES = ["age", "education_years", "urban", "contact_attempts", "prior_response"]
X = df[FEATURES].values
y = df["responded"].values
feature_names = FEATURES

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ===========================================================================
# 1. Timing experiment: n_estimators × max_depth × dataset size
# ===========================================================================
N_ESTIMATORS_GRID = [10, 50, 100, 200, 500]
MAX_DEPTH_GRID = [5, 10, 15, None]
DATASET_FRACTIONS = [200 / len(X_train), 500 / len(X_train), 1.0]

print("Running timing experiments (n_estimators × max_depth)...")
print("Using full training set size. This may take a minute.\n")

timing_rows = []

for n_est in N_ESTIMATORS_GRID:
    for max_depth in MAX_DEPTH_GRID:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_depth,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=1,   # single-threaded for reproducible timing
        )
        t0 = time.perf_counter()
        rf.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        timing_rows.append(
            {
                "n_estimators": n_est,
                "max_depth": max_depth if max_depth is not None else "None",
                "fit_time_sec": round(elapsed, 4),
            }
        )

timing_df = pd.DataFrame(timing_rows)
pivot_timing = timing_df.pivot(
    index="n_estimators", columns="max_depth", values="fit_time_sec"
)

print("=" * 60)
print("Fit time (seconds) — n_estimators × max_depth (n=960, 5 features)")
print("=" * 60)
print(pivot_timing.to_string())
print()

# ===========================================================================
# 2. max_features effect: "sqrt" vs. all features
# ===========================================================================
print("max_features effect: sqrt vs. all features (n=200 trees)...")
timing_mf = []
for mf, label in [("sqrt", "sqrt"), (1.0, "all features")]:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        max_features=mf,
        random_state=42,
        n_jobs=1,
    )
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    timing_mf.append({"max_features": label, "fit_time_sec": round(elapsed, 4)})

mf_df = pd.DataFrame(timing_mf)
ratio = mf_df.loc[1, "fit_time_sec"] / mf_df.loc[0, "fit_time_sec"]
print()
print("=" * 60)
print("max_features effect (200 trees, unlimited depth)")
print("=" * 60)
print(mf_df.to_string(index=False))
print(f"\n  'all features' is {ratio:.1f}x slower than 'sqrt'.")
print(
    "  With p=50 features, sqrt(50)≈7. Evaluating all 50 features at every split\n"
    "  is ~7x more work per node. This compounds over depth × n_estimators."
)
print()

# ===========================================================================
# 3. Dataset size scaling
# ===========================================================================
# Show how fit time scales with training set size. Subsample to 200, 500, full.
print("Dataset size scaling experiment...")
SIZE_GRID = [200, 500, len(X_train)]
timing_size = []

for size in SIZE_GRID:
    idx = np.random.RandomState(42).choice(len(X_train), size=size, replace=False)
    Xs, ys = X_train[idx], y_train[idx]
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=1,
    )
    t0 = time.perf_counter()
    rf.fit(Xs, ys)
    elapsed = time.perf_counter() - t0
    timing_size.append({"n_train": size, "fit_time_sec": round(elapsed, 4)})

size_df = pd.DataFrame(timing_size)
print()
print("=" * 60)
print("Dataset size scaling (200 trees, unlimited depth)")
print("=" * 60)
print(size_df.to_string(index=False))
print()

# Fit a linear model to extrapolate: time ~ a * n
times = size_df["fit_time_sec"].values
sizes = size_df["n_train"].values
a = np.polyfit(sizes, times, 1)[0]   # seconds per record
for target_n in [50_000, 100_000, 1_000_000, 3_000_000]:
    est_sec = a * target_n
    est_min = est_sec / 60
    print(
        f"  n={target_n:>10,}:  ~{est_sec:.0f} s  (~{est_min:.1f} min)  "
        "[linear extrapolation, 200 trees, 5 features, n_jobs=1]"
    )
print()
print(
    "NOTE: Real federal datasets (50 features, max_depth=15) will be substantially\n"
    "slower. The extrapolation above is a lower bound for this 5-feature toy problem.\n"
    "With p=50 features and max_depth=15, multiply by ~7x for max_features='sqrt'\n"
    "or ~30x for max_features=1.0 (all features)."
)
print()

# ===========================================================================
# 4. Subsampling validation: do importance rankings stabilize before full n?
# ===========================================================================
# Fit RF on subsamples of training data. Compare importance rankings at each
# size to the full-data ranking. This answers: "At what n can I trust the
# top-3 finding?"
print("Subsampling validation: importance ranking convergence...")
SUBSAMPLE_SIZES = [150, 300, 480, 720, 960]   # 15%, 30%, 50%, 75%, 100% of train

# Reference ranking: full training set
rf_full = RandomForestClassifier(
    n_estimators=200, min_samples_leaf=5, max_features="sqrt",
    random_state=42, n_jobs=-1
)
rf_full.fit(X_train, y_train)
perm_full = permutation_importance(
    rf_full, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc"
)
ref_ranks = pd.Series(perm_full.importances_mean, index=feature_names).rank(
    ascending=False, method="min"
)

subsample_rows = []
for size in SUBSAMPLE_SIZES:
    idx = np.random.RandomState(7).choice(len(X_train), size=size, replace=False)
    Xs, ys = X_train[idx], y_train[idx]
    rf_sub = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, max_features="sqrt",
        random_state=42, n_jobs=-1
    )
    rf_sub.fit(Xs, ys)
    perm_sub = permutation_importance(
        rf_sub, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc"
    )
    sub_ranks = pd.Series(perm_sub.importances_mean, index=feature_names).rank(
        ascending=False, method="min"
    )
    # Spearman rank correlation with full-data ranking
    from scipy.stats import spearmanr
    rho, _ = spearmanr(ref_ranks.values, sub_ranks.values)
    top3_match = int(sum(
        sub_ranks[feat] <= 3 and ref_ranks[feat] <= 3
        for feat in feature_names
    ))
    subsample_rows.append({
        "n_train": size,
        "pct_of_full": f"{size / len(X_train):.0%}",
        "rank_correlation": round(rho, 3),
        "top3_matches": f"{top3_match}/3",
    })

sub_df = pd.DataFrame(subsample_rows)
print()
print("=" * 65)
print("Importance ranking stability vs. subsample size")
print("(rank_correlation: 1.0 = identical ranking to full dataset)")
print("=" * 65)
print(sub_df.to_string(index=False))
print()
print(
    "Practical guidance:\n"
    "  - When rank_correlation >= 0.95 and top3_matches = 3/3, your subsample\n"
    "    size is sufficient for importance analysis.\n"
    "  - Run this check on your actual dataset before committing to a subsample\n"
    "    strategy. 50K–100K records is a common starting point for ACS-scale data."
)

# Summary timing chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(
    timing_df[timing_df["max_depth"] == "None"]["n_estimators"],
    timing_df[timing_df["max_depth"] == "None"]["fit_time_sec"],
    marker="o", label="Unlimited depth",
)
axes[0].plot(
    timing_df[timing_df["max_depth"] == 5]["n_estimators"],
    timing_df[timing_df["max_depth"] == 5]["fit_time_sec"],
    marker="s", linestyle="--", label="max_depth=5",
)
axes[0].set_xlabel("n_estimators")
axes[0].set_ylabel("Fit time (seconds)")
axes[0].set_title("Fit time vs. n_estimators")
axes[0].legend()

axes[1].plot(size_df["n_train"], size_df["fit_time_sec"], marker="o", color="steelblue")
axes[1].set_xlabel("Training set size (n)")
axes[1].set_ylabel("Fit time (seconds)")
axes[1].set_title("Fit time vs. dataset size\n(200 trees, unlimited depth)")

plt.suptitle("Random Forest Computational Scaling", fontsize=12)
plt.tight_layout()
plt.savefig("computational_scaling.png", dpi=150, bbox_inches="tight")
plt.close()
print()
print("Saved: computational_scaling.png")


if __name__ == "__main__":
    pass  # all output produced at module level above
