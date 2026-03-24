"""
Chapter 3 -- Tract-Level Targeting Exercise (Full Solution)
===========================================================
Simulates a real field operations decision: which census tracts should a
survey program prioritize for in-person nonresponse follow-up when the
budget allows visits to only 25% of tracts?

This script generates tract-level synthetic data, fits both a decision tree
and a Random Forest, produces the printable decision rules for the tree model,
computes permutation importance, outputs a prioritized tract ranking, and
generates a SHAP waterfall for the top-ranked flagged tract.

Activity context
----------------
This is the worked solution for the in-chapter activity (Section 10).
Students who want to work the exercise themselves should stop after reading
the activity setup in the chapter and attempt the tasks before running this.

Tasks this script completes
----------------------------
1. Generate tract-level dataset (300 tracts, seed=2025).
2. Fit Decision Tree (depth 3) -- print rules, compute test AUC.
3. Fit Random Forest (200 trees) -- compute test AUC and OOB score.
4. Compute permutation importance -- identify the two strongest predictors.
5. Score all 300 tracts by predicted probability of low response.
6. Output the top 25% (75 tracts) as a prioritized list.
7. SHAP waterfall for the top-ranked flagged tract (audit exhibit).

Interpretation prompt
---------------------
After running this script, consider:
- Why is the top-ranked tract flagged? Walk through the decision tree rules
  to trace its path.
- The Random Forest has higher AUC but no printable rules.
  Write a 2-sentence recommendation for which model to deploy and why.
- Which features would you drop based on the permutation importance table?
  What is the budget case for simpler models?
- Given a budget for 25% of tracts, which model's ranking would you trust more?
  What is the risk of each choice?
- Review the SHAP waterfall: does it match your intuition from the
  permutation importance table? Are the same top features driving this tract?

Requirements
------------
Python 3.9+, numpy, pandas, matplotlib, scikit-learn
SHAP waterfall (step 7): pip install shap
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
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Activity dataset -- tract-level aggregated data (seed=2025, n_tracts=300)
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

# Simulate low-response tracts (harder to reach)
logit_tract = (
    -1.5
    + 0.04 * tract_data["pct_renters"]
    + 0.03 * tract_data["pct_foreign_born"]
    - 2.0 * (tract_data["prior_rr"] - 0.72)
    + 0.15 * tract_data["contact_attempts"]
    + np.random.normal(0, 0.4, n_tracts)
)
tract_data["low_response"] = (1 / (1 + np.exp(-logit_tract)) > 0.45).astype(int)

TRACT_FEATURES = [
    "pct_renters",
    "median_age",
    "pct_foreign_born",
    "pct_bachelors",
    "pop_density_log",
    "prior_rr",
    "contact_attempts",
]

print(f"Tract dataset: {tract_data.shape}")
print(f"Low-response tracts: {tract_data['low_response'].mean():.1%}")
print()

# ===========================================================================
# Step 1: Train/test split (80/20, stratified)
# ===========================================================================
X_act = tract_data[TRACT_FEATURES]
y_act = tract_data["low_response"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_act, y_act, test_size=0.20, random_state=42, stratify=y_act
)

# ===========================================================================
# Step 2: Decision Tree (depth 3) -- rules for methodology documentation
# ===========================================================================
dt_act = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_act.fit(X_tr, y_tr)

print("=" * 60)
print("Decision tree rules (depth=3)")
print("=" * 60)
print("These rules can be attached directly to a methodology memo.")
print()
print(export_text(dt_act, feature_names=TRACT_FEATURES))

dt_auc = roc_auc_score(y_te, dt_act.predict_proba(X_te)[:, 1])
print(f"Decision tree test AUC: {dt_auc:.3f}")
print()

# ===========================================================================
# Step 3: Random Forest (200 trees)
# ===========================================================================
rf_act = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    oob_score=True,
)
rf_act.fit(X_tr, y_tr)
y_prob_act = rf_act.predict_proba(X_te)[:, 1]

rf_auc = roc_auc_score(y_te, y_prob_act)
print(f"Random Forest test AUC:     {rf_auc:.3f}")
print(f"Random Forest OOB accuracy: {rf_act.oob_score_:.3f}")
print()

# ===========================================================================
# Step 4: Permutation importance
# ===========================================================================
perm_act = permutation_importance(
    rf_act, X_te, y_te,
    n_repeats=15,
    random_state=42,
    scoring="roc_auc",
    n_jobs=-1,
)
perm_df = pd.DataFrame(
    {
        "feature": TRACT_FEATURES,
        "importance": perm_act.importances_mean,
        "std": perm_act.importances_std,
    }
).sort_values("importance", ascending=False)

print("=" * 60)
print("Permutation importance (AUC drop on test set)")
print("=" * 60)
print(perm_df.round(4).to_string(index=False))
print()
print(
    f"Top 2 predictors: {perm_df['feature'].iloc[0]} "
    f"and {perm_df['feature'].iloc[1]}"
)
print()

# ===========================================================================
# Step 5: Score all 300 tracts -- produce priority ranking
# ===========================================================================
tract_data["predicted_lr_prob"] = rf_act.predict_proba(X_act)[:, 1]
tract_data_sorted = tract_data.sort_values("predicted_lr_prob", ascending=False)
top25_pct = tract_data_sorted.head(int(0.25 * n_tracts))
cutoff_prob = top25_pct["predicted_lr_prob"].min()

print("=" * 60)
print(f"Field prioritization: top 25% of tracts to visit (n={len(top25_pct)})")
print(f"Probability cutoff: {cutoff_prob:.3f}")
print("=" * 60)
print(
    top25_pct[
        ["tract_id", "predicted_lr_prob", "prior_rr", "pct_renters", "contact_attempts"]
    ]
    .head(15)
    .to_string(index=False)
)
print()

# ===========================================================================
# Step 6: Visualize permutation importance + prioritization histogram
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

perm_df_plot = perm_df.sort_values("importance")
axes[0].barh(
    perm_df_plot["feature"],
    perm_df_plot["importance"],
    xerr=perm_df_plot["std"],
    capsize=4,
    color="steelblue",
    edgecolor="white",
)
axes[0].set_title("Permutation importance (AUC drop)")
axes[0].set_xlabel("Mean AUC decrease when feature is shuffled")

axes[1].hist(
    tract_data["predicted_lr_prob"], bins=30,
    color="lightgray", edgecolor="white", label="All tracts",
)
axes[1].hist(
    top25_pct["predicted_lr_prob"], bins=15,
    color="firebrick", edgecolor="white", alpha=0.7, label="Top 25% to visit",
)
axes[1].axvline(
    cutoff_prob, color="black", linestyle="--",
    label=f"Cutoff = {cutoff_prob:.2f}",
)
axes[1].set_xlabel("Predicted probability of low response")
axes[1].set_ylabel("Number of tracts")
axes[1].set_title("Field prioritization: tracts to visit (top 25%)")
axes[1].legend()

plt.tight_layout()
plt.savefig("tract_prioritization.png", dpi=120, bbox_inches="tight")
print("Tract prioritization chart saved to tract_prioritization.png")
plt.close()

# ===========================================================================
# Step 7: SHAP waterfall for the top-ranked flagged tract (audit exhibit)
#    This answers: "Why is this tract ranked first?"
#    The waterfall decomposes the prediction into per-feature contributions.
#    Required for OMB review or IG audit of a model-assisted survey operation.
# ===========================================================================
try:
    import shap
    _shap_available = True
except ImportError:
    _shap_available = False
    print()
    print("NOTE: SHAP waterfall skipped (shap not installed).")
    print("Install with: pip install shap")
    print("Then re-run to see the per-tract audit exhibit.")

if _shap_available:
    print()
    print("=" * 60)
    print("SHAP waterfall for top-ranked flagged tract (audit exhibit)")
    print("=" * 60)

    # TreeExplainer uses the tree structure directly -- exact, no sampling.
    explainer = shap.TreeExplainer(rf_act)
    shap_values = explainer.shap_values(X_act)
    if isinstance(shap_values, list):
        shap_vals_class1 = shap_values[1]
        base_value = explainer.expected_value[1]
    else:
        shap_vals_class1 = shap_values
        base_value = explainer.expected_value

    # Identify the top-ranked tract by predicted probability
    top_tract_id = tract_data_sorted.iloc[0]["tract_id"]
    top_tract_orig_idx = tract_data.index[
        tract_data["tract_id"] == top_tract_id
    ][0]
    top_shap = shap_vals_class1[top_tract_orig_idx]
    top_pred_prob = tract_data_sorted.iloc[0]["predicted_lr_prob"]

    print(f"Tract: {top_tract_id}")
    print(f"  Predicted probability of low response: {top_pred_prob:.3f}")
    print(f"  Base rate (population average):        {base_value:.3f}")
    print()
    print(f"  {'Feature':<22} {'Value':>10} {'SHAP contribution':>20}")
    print("  " + "-" * 56)

    sorted_shap_idx = np.argsort(np.abs(top_shap))[::-1]
    for i in sorted_shap_idx:
        feat = TRACT_FEATURES[i]
        val = float(X_act[feat].iloc[top_tract_orig_idx])
        sv = top_shap[i]
        direction = "+" if sv >= 0 else ""
        print(
            f"  {feat:<22} {val:>10.3f}  {direction}{sv:>+.4f}"
        )

    reconstructed = base_value + top_shap.sum()
    print("  " + "-" * 56)
    print(f"  {'Base rate':>22}              {base_value:>+.4f}")
    print(f"  {'Sum of SHAP values':>22}     {top_shap.sum():>+.4f}")
    print(f"  {'Reconstructed prediction':>22}  {reconstructed:.4f}")
    print(f"  {'Model output (actual)':>22}     {top_pred_prob:.4f}")
    print()

    # Waterfall bar chart for methodology documentation
    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_shap_data = [
        (
            TRACT_FEATURES[i],
            top_shap[i],
            float(X_act[TRACT_FEATURES[i]].iloc[top_tract_orig_idx]),
        )
        for i in sorted_shap_idx
    ]
    labels = [f"{f} = {v:.2f}" for f, _, v in sorted_shap_data]
    contribs = [s for _, s, _ in sorted_shap_data]
    colors = ["firebrick" if s > 0 else "steelblue" for s in contribs]
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, contribs, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (contribution to flagging probability)")
    ax.set_title(
        f"SHAP Waterfall -- Tract {top_tract_id} (Top-ranked for follow-up)\n"
        f"Base rate = {base_value:.3f}, Prediction = {top_pred_prob:.3f}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("tract_shap_waterfall_top.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("SHAP waterfall saved to tract_shap_waterfall_top.png")
    print()
    print(
        "Interpretation: The waterfall above is the exhibit to include in a\n"
        "methodology report or OMB review. It answers 'why is this tract ranked\n"
        "first?' in per-feature, per-prediction terms that aggregate importance\n"
        "tables cannot provide."
    )

# ===========================================================================
# Summary for methodology documentation
# ===========================================================================
print()
print("=" * 60)
print("Summary for methodology documentation")
print("=" * 60)
print(f"  Model: Random Forest, 200 trees, min_samples_leaf=5")
print(f"  Validation: OOB accuracy = {rf_act.oob_score_:.3f}, test AUC = {rf_auc:.3f}")
print(f"  Decision Tree AUC (for rule documentation): {dt_auc:.3f}")
print(f"  Top predictor: {perm_df['feature'].iloc[0]}")
print(f"  Second predictor: {perm_df['feature'].iloc[1]}")
print(f"  Tracts flagged for follow-up: {len(top25_pct)} of {n_tracts}")
print(f"  Flagging threshold: predicted probability >= {cutoff_prob:.3f}")
