"""
Chapter 4 — Example 07: Interpretability Contrast
==================================================
Side-by-side demonstration of what you can and cannot explain from each
of the four model families trained in example 06.

The central contrast:
- Logistic Regression: one coefficient per feature.  Directly maps to
  odds ratios.  Can be printed in a methodology report.
- Decision Tree: a set of printable if/then rules.  A manager can trace
  any single prediction by hand.
- Random Forest: feature importance scores but no printable rules.  You
  can say "prior_response matters most" but not "here is the exact rule
  for household 1042."
- MLP: weight matrices only.  No coefficient, no rule.  Partial dependence
  plots (PDPs) provide the best available aggregate explanation.

Partial dependence:
PDPs marginalise over all other features to show the effect of one feature
on the predicted probability.  They answer "as contact_attempts increases,
what does the model predict on average?"  They do NOT answer "why did the
model predict 0.72 for this specific household?"  That requires SHAP
values, covered in later chapters.

Outputs
-------
- Console: LR coefficients, DT rules, RF feature importances, MLP weight shapes.
- Plot 1: LR coefficient bar chart.
- Plot 2: PDPs for MLP (one subplot per feature).

Requirements
------------
Python 3.9+, numpy, pandas, matplotlib, scikit-learn
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import partial_dependence

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location(
    "setup",
    os.path.join(os.path.dirname(__file__), "01_dataset_setup.py"),
)
_setup = module_from_spec(_spec)
_spec.loader.exec_module(_setup)

X_clf_train = _setup.X_clf_train
X_clf_test = _setup.X_clf_test
X_clf_train_sc = _setup.X_clf_train_sc
X_clf_test_sc = _setup.X_clf_test_sc
y_clf_train = _setup.y_clf_train
y_clf_test = _setup.y_clf_test
FEATURES_CLF = _setup.FEATURES_CLF

# ---------------------------------------------------------------------------
# 1. Fit all four models
# ---------------------------------------------------------------------------
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_clf_train_sc, y_clf_train)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_clf_train, y_clf_train)

rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                             random_state=42, n_jobs=-1)
rf.fit(X_clf_train, y_clf_train)

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    alpha=0.0001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=15,
    verbose=False,
)
mlp.fit(X_clf_train_sc, y_clf_train)

# ---------------------------------------------------------------------------
# 2. Logistic Regression: coefficient table
# ---------------------------------------------------------------------------
print("=" * 60)
print("Logistic Regression — interpretable coefficients")
print("=" * 60)
coef_df = pd.DataFrame(
    {
        "Feature": FEATURES_CLF,
        "Coefficient": lr.coef_[0].round(3),
        "Odds ratio": np.exp(lr.coef_[0]).round(3),
    }
).sort_values("Coefficient", key=abs, ascending=False)
print(coef_df.to_string(index=False))
print()
print("Read: prior_response has the largest negative coefficient, meaning")
print("respondents with a prior response are much less likely to be nonresponders.")

# ---------------------------------------------------------------------------
# 3. Decision Tree: printable rules
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Decision Tree (depth 3) — printable rules")
print("=" * 60)
print(export_text(dt, feature_names=FEATURES_CLF))

# ---------------------------------------------------------------------------
# 4. Random Forest: feature importance
# ---------------------------------------------------------------------------
print("=" * 60)
print("Random Forest — feature importances (not individual rules)")
print("=" * 60)
fi_df = pd.DataFrame(
    {
        "Feature": FEATURES_CLF,
        "Importance": rf.feature_importances_.round(4),
    }
).sort_values("Importance", ascending=False)
print(fi_df.to_string(index=False))
print("Note: importance shows WHICH features matter, not HOW they affect predictions.")

# ---------------------------------------------------------------------------
# 5. MLP: weight matrix shapes
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Neural Network (100, 50) — weight matrix shapes")
print("=" * 60)
total_params = 0
for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
    layer_name = f"Layer {i+1} (hidden)" if i < len(mlp.coefs_) - 1 else "Output layer"
    print(f"  {layer_name}:")
    print(f"    Weight matrix: {w.shape}  ({w.size:,} parameters)")
    print(f"    Bias vector:   {b.shape}  ({b.size:,} parameters)")
    total_params += w.size + b.size
print(f"  Total trainable parameters: {total_params:,}")
print()
print("  → No single number you can show leadership.")
print("  → No rule you can attach to a methodology report.")
print("  → Partial dependence plots (PDPs) are the best available aggregate explanation.")

# ---------------------------------------------------------------------------
# 6. Coefficient bar chart for LR
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bar_colors = ["firebrick" if c < 0 else "steelblue" for c in coef_df["Coefficient"]]
axes[0].barh(coef_df["Feature"], coef_df["Coefficient"],
             color=bar_colors, edgecolor="white")
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set_xlabel("Logistic regression coefficient")
axes[0].set_title("Logistic Regression: direct coefficient interpretation\n(blue = increases response probability)")
for i, (feat, coef) in enumerate(zip(coef_df["Feature"], coef_df["Coefficient"])):
    axes[0].text(
        coef + (0.02 if coef >= 0 else -0.02),
        i,
        f"{coef:+.3f}",
        va="center",
        ha="left" if coef >= 0 else "right",
        fontsize=8.5,
    )

# ---------------------------------------------------------------------------
# 7. PDPs for the MLP
# ---------------------------------------------------------------------------
n_features = len(FEATURES_CLF)
pdp_axes = [axes[1]] if n_features == 1 else None

# Create a new figure for PDPs
fig2, pdp_axes = plt.subplots(1, n_features, figsize=(16, 4), sharey=False)

for ax, feat in zip(pdp_axes, FEATURES_CLF):
    feat_idx = FEATURES_CLF.index(feat)
    pd_result = partial_dependence(
        mlp,
        X_clf_train_sc,
        features=[feat_idx],
        kind="average",
        grid_resolution=30,
    )
    # Convert standardised grid back to original scale for readability
    feat_mean = X_clf_train[feat].mean()
    feat_std = X_clf_train[feat].std()
    grid_orig = pd_result["grid_values"][0] * feat_std + feat_mean

    ax.plot(grid_orig, pd_result["average"][0], color="darkorchid", lw=2)
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel("Predicted response\nprobability" if feat == FEATURES_CLF[0] else "")
    ax.set_title(feat, fontsize=9)
    ax.tick_params(labelsize=8)

fig2.suptitle(
    "Partial dependence plots (MLP): marginal effect of each feature\n"
    "(averages over all other features — not an individual prediction explanation)",
    fontsize=10,
)
fig2.tight_layout()

plt.figure(fig.number)
axes[0].figure.tight_layout()
axes[0].figure.savefig(
    os.path.join(os.path.dirname(__file__), "07_lr_coefficients.png"),
    dpi=120,
    bbox_inches="tight",
)

fig2.savefig(
    os.path.join(os.path.dirname(__file__), "07_mlp_pdp.png"),
    dpi=120,
    bbox_inches="tight",
)
plt.show()

print("\nPlots saved: 07_lr_coefficients.png, 07_mlp_pdp.png")
