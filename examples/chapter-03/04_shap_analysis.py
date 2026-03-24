"""
Chapter 3 -- SHAP Analysis for Random Forest Interpretability
=============================================================
Demonstrates SHAP TreeExplainer on a fitted Random Forest:
- SHAP summary plot (beeswarm): global importance with directionality
- SHAP dependence plot for top 2 features
- SHAP waterfall for a single prediction ("why was this tract flagged?")
- Comparison: SHAP vs. Gini vs. permutation importance rankings

Dataset: tract-level nonresponse targeting (seed=2025, n=300 tracts).
This is the same dataset used in the chapter activity (Section 10) and
10_tract_exercise.py. Using the tract dataset here shows how SHAP closes
the gap between aggregate importance and per-record audit exhibits.

Requires: pip install shap
Run: python 04_shap_analysis.py
"""

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import guard: shap is an optional dependency
# ---------------------------------------------------------------------------
try:
    import shap
except ImportError:
    print("ERROR: shap is not installed.")
    print("Install with:  pip install shap")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Reproduce the tract dataset from the chapter activity (seed=2025, n=300)
# Features: pct_renters, median_age, pct_foreign_born, pct_bachelors,
#           pop_density_log, prior_rr, contact_attempts
# Target: low_response (binary, 1 = flagged as low-response tract)
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

X = tract_data[FEATURES]
y = tract_data["low_response"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Tract dataset: {tract_data.shape}")
print(f"Low-response tracts: {y.mean():.1%}")
print(f"Train: {len(X_train)}  Test: {len(X_test)}")
print()

# ===========================================================================
# 1. Fit Random Forest (200 trees)
# ===========================================================================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    oob_score=True,
)
rf.fit(X_train, y_train)
y_prob = rf.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_prob)
print(f"Random Forest test AUC: {test_auc:.3f}")
print(f"OOB accuracy:           {rf.oob_score_:.3f}")
print()

# ===========================================================================
# 2. SHAP TreeExplainer
#    TreeExplainer uses the tree structure directly -- no sampling, exact values.
#    For binary classification, shap_values is a list [class_0_vals, class_1_vals].
#    We use class 1 (low_response = 1 = flagged tract).
# ===========================================================================
print("Computing SHAP values (TreeExplainer) ...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    # Legacy shap API: list of [class_0, class_1]
    shap_vals_class1 = shap_values[1]
    base_value = explainer.expected_value[1]
else:
    # Newer shap API: single array or Explanation object
    shap_vals_class1 = shap_values
    base_value = explainer.expected_value

print(f"SHAP base rate (expected value, class 1): {base_value:.4f}")
print(f"SHAP values shape: {shap_vals_class1.shape}")
print()

# ===========================================================================
# 3. Summary plot (beeswarm)
#    Each dot = one test-set prediction.
#    x-axis = SHAP value: positive pushes toward low_response=1 (flagged),
#             negative pushes toward low_response=0 (not flagged).
#    Color = feature value (red = high, blue = low).
# ===========================================================================
print("Generating SHAP summary plot (beeswarm) ...")
plt.figure(figsize=(9, 5))
shap.summary_plot(shap_vals_class1, X_test, feature_names=FEATURES, show=False)
plt.title(
    "SHAP Summary Plot -- Tract Nonresponse Prediction\n"
    "(each dot = one test tract; x-axis = SHAP value; color = feature value)",
    fontsize=9,
)
plt.tight_layout()
plt.savefig("shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_summary_beeswarm.png")
print()

# ===========================================================================
# 4. Dependence plots for top 2 features by mean |SHAP|
#    Dependence plot: how one feature's SHAP value changes with its raw value.
#    Interaction coloring (color by a second feature) reveals feature interactions.
# ===========================================================================
print("Generating SHAP dependence plots ...")
mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)
top2_idx = np.argsort(mean_abs_shap)[::-1][:2]
top2_features = [FEATURES[i] for i in top2_idx]
print(f"  Top 2 features by mean |SHAP|: {top2_features[0]}, {top2_features[1]}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, feat, interact_feat in zip(axes, top2_features, top2_features[::-1]):
    feat_idx = FEATURES.index(feat)
    interact_idx = FEATURES.index(interact_feat)
    x_vals = X_test[feat].values
    shap_feat = shap_vals_class1[:, feat_idx]
    color_vals = X_test[interact_feat].values

    sc = ax.scatter(
        x_vals,
        shap_feat,
        c=color_vals,
        cmap="coolwarm",
        alpha=0.7,
        s=25,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label=interact_feat)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feat)
    ax.set_ylabel(f"SHAP value for {feat}")
    ax.set_title(f"SHAP dependence: {feat}\n(colored by {interact_feat})", fontsize=9)

plt.suptitle("SHAP Dependence Plots -- Tract Nonresponse Prediction", fontsize=11)
plt.tight_layout()
plt.savefig("shap_dependence_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_dependence_plots.png")
print()

# ===========================================================================
# 5. Waterfall for a single prediction -- "Why was this tract flagged?"
#    Find the test tract with the highest predicted probability of low response.
#    Print the additive decomposition and generate a waterfall bar chart.
# ===========================================================================
high_risk_idx = np.argmax(y_prob)
top_tract_id = tract_data.iloc[X_test.index[high_risk_idx]]["tract_id"]
prediction = y_prob[high_risk_idx]

print("=" * 60)
print(f"SHAP waterfall for most-flagged test tract: {top_tract_id}")
print("=" * 60)
print(f"  Base rate (mean model output, class 1): {base_value:.3f}")
print(f"  Model prediction (nonresponse prob):    {prediction:.3f}")
print()

single_shap = shap_vals_class1[high_risk_idx]
feature_vals = X_test.iloc[high_risk_idx]
total_contribution = single_shap.sum()

print(f"  {'Feature':<22} {'Value':>10} {'SHAP contribution':>20}")
print("  " + "-" * 56)
sorted_idx = np.argsort(np.abs(single_shap))[::-1]
for i in sorted_idx:
    direction = "+" if single_shap[i] >= 0 else ""
    print(
        f"  {FEATURES[i]:<22} {feature_vals.iloc[i]:>10.3f}"
        f"  {direction}{single_shap[i]:>+.4f}"
    )
print("  " + "-" * 56)
print(f"  {'Base rate':>22}              {base_value:>+.4f}")
print(f"  {'Sum of SHAP values':>22}     {total_contribution:>+.4f}")
print(f"  {'Reconstructed prediction':>22}  {base_value + total_contribution:.4f}")
print(f"  {'Model output (actual)':>22}     {prediction:.4f}")
print()
print(
    "  Interpretation: The model started at the population base rate. "
    "Each feature's SHAP value shows how much it pushed the prediction "
    "toward (positive) or away from (negative) flagging as low-response."
)
print()

# Waterfall bar chart
fig, ax = plt.subplots(figsize=(9, 5))
sorted_shap = [(FEATURES[i], single_shap[i], feature_vals.iloc[i]) for i in sorted_idx]
feature_labels = [f"{f} = {v:.2f}" for f, _, v in sorted_shap]
shap_contribs = [s for _, s, _ in sorted_shap]

colors = ["firebrick" if s > 0 else "steelblue" for s in shap_contribs]
y_pos = np.arange(len(feature_labels))

ax.barh(y_pos, shap_contribs, color=colors, edgecolor="white", height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_labels, fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("SHAP value (contribution to flagging probability)")
ax.set_title(
    f"SHAP Waterfall -- Tract {top_tract_id}\n"
    f"Base rate = {base_value:.3f}, Prediction = {prediction:.3f} "
    f"({'FLAGGED' if prediction > 0.5 else 'not flagged'})",
    fontsize=10,
)
plt.tight_layout()
plt.savefig("shap_waterfall_single.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: shap_waterfall_single.png")
print()

# ===========================================================================
# 6. Three-way importance comparison: SHAP vs. Gini vs. permutation
#    When all three methods agree on the top features, that is strong evidence.
#    When they diverge, investigate correlated feature pairs.
# ===========================================================================
# Gini importance: computed on training data during fitting
gini_imp = pd.Series(rf.feature_importances_, index=FEATURES)
gini_rank = gini_imp.rank(ascending=False).astype(int)

# Permutation importance: shuffle each feature on test set, measure AUC drop
perm_result = permutation_importance(
    rf, X_test, y_test, n_repeats=15, random_state=42, scoring="roc_auc", n_jobs=-1
)
perm_imp = pd.Series(perm_result.importances_mean, index=FEATURES)
perm_rank = perm_imp.rank(ascending=False).astype(int)

# SHAP importance: mean absolute SHAP value on test set
shap_imp = pd.Series(np.abs(shap_vals_class1).mean(axis=0), index=FEATURES)
shap_rank = shap_imp.rank(ascending=False).astype(int)

comparison = pd.DataFrame(
    {
        "Feature": FEATURES,
        "Gini_rank": gini_rank.values,
        "Perm_rank": perm_rank.values,
        "SHAP_rank": shap_rank.values,
        "Gini_score": np.round(gini_imp.values, 4),
        "Perm_AUC_drop": np.round(perm_imp.values, 4),
        "SHAP_mean_abs": np.round(shap_imp.values, 4),
    }
).sort_values("SHAP_rank")

print("=" * 60)
print("Three-way feature importance comparison")
print("=" * 60)
print(comparison.to_string(index=False))
print()
print("Notes:")
print("  - Gini: computed on training data; can over-rank correlated features.")
print("  - Permutation: shuffled on test set; AUC drop; reliable for reporting.")
print("  - SHAP: per-prediction attributions aggregated; shows direction and magnitude.")
print("  - When ranks agree across all three, the finding is robust.")
print("  - When Gini diverges from Perm/SHAP, suspect feature correlation.")

# Bar chart
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, col, title in zip(
    axes,
    ["Gini_score", "Perm_AUC_drop", "SHAP_mean_abs"],
    [
        "Gini importance\n(training data)",
        "Permutation importance\n(test AUC drop)",
        "SHAP mean |value|\n(test set)",
    ],
):
    vals = comparison.set_index("Feature")[col].sort_values(ascending=False)
    ax.barh(vals.index[::-1], vals.values[::-1], color="steelblue")
    ax.set_xlabel("Score")
    ax.set_title(title, fontsize=9)

plt.suptitle(
    "Feature Importance: Gini vs. Permutation vs. SHAP (Tract Nonresponse Model)",
    fontsize=10,
)
plt.tight_layout()
plt.savefig("shap_importance_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print()
print("  Saved: shap_importance_comparison.png")
print()
print("All SHAP outputs complete.")
print("Figures saved:")
print("  shap_summary_beeswarm.png   -- global importance with direction")
print("  shap_dependence_plots.png   -- feature-level nonlinear effects")
print("  shap_waterfall_single.png   -- per-record audit exhibit")
print("  shap_importance_comparison.png -- three-way method comparison")
