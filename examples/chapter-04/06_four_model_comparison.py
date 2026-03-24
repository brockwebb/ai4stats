"""
Chapter 4 — Example 06: Four-Model Comparison (Part I Capstone)
================================================================
This is the capstone script for Part I.  It refits all four model families —
Logistic Regression, Decision Tree, Random Forest, and MLP — on the same
training split and evaluates them on the same test set.

Why this matters:
Federal analysts are regularly asked to compare model alternatives or to
evaluate a vendor claim that "our neural network outperforms classical
methods."  The honest answer almost always requires running both on the
same data under the same conditions.  Swapping datasets, evaluation periods,
or threshold choices can make any model look best.

Design choices documented here:
- LR and MLP receive standardised features (required for gradient-based
  methods and good practice for LR).
- DT and RF use raw features (tree splits are scale-invariant).
- Decision Tree is capped at depth=3 to maintain interpretability
  (shallow trees can be printed in a methodology report).
- Random Forest uses 200 trees and min_samples_leaf=5 to avoid leaf-level
  overfitting on small cells.
- MLP uses the (100, 50) architecture established in example 02.

Outputs
-------
- Console: full comparison table (accuracy, precision, recall, F1, AUC).
- Plot 1: overlaid ROC curves for all four models.
- Plot 2: grouped bar chart of accuracy, F1, and AUC-ROC.

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Load dataset — both scaled (for LR, MLP) and unscaled (for DT, RF)
# ---------------------------------------------------------------------------
from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location(
    "setup",
    os.path.join(os.path.dirname(__file__), "01_dataset_setup.py"),
)
_setup = module_from_spec(_spec)
_spec.loader.exec_module(_setup)

X_clf_train_sc = _setup.X_clf_train_sc
X_clf_test_sc = _setup.X_clf_test_sc
X_clf_train = _setup.X_clf_train
X_clf_test = _setup.X_clf_test
y_clf_train = _setup.y_clf_train
y_clf_test = _setup.y_clf_test
FEATURES_CLF = _setup.FEATURES_CLF

# ---------------------------------------------------------------------------
# 1. Define models
# ---------------------------------------------------------------------------
MODEL_CONFIGS = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=42),
        True,   # use scaled features
    ),
    (
        "Decision Tree (depth 3)",
        DecisionTreeClassifier(max_depth=3, random_state=42),
        False,
    ),
    (
        "Random Forest (200 trees)",
        RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                               random_state=42, n_jobs=-1),
        False,
    ),
    (
        "MLP (100, 50)",
        MLPClassifier(
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
        ),
        True,
    ),
]

# ---------------------------------------------------------------------------
# 2. Fit all models and collect metrics
# ---------------------------------------------------------------------------
results = []
roc_data = {}  # for ROC curve overlay

for name, model, use_scaled in MODEL_CONFIGS:
    X_tr = X_clf_train_sc if use_scaled else X_clf_train
    X_te = X_clf_test_sc if use_scaled else X_clf_test

    model.fit(X_tr, y_clf_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    results.append(
        {
            "Model": name,
            "Accuracy":  round(accuracy_score(y_clf_test, y_pred), 3),
            "Precision": round(precision_score(y_clf_test, y_pred), 3),
            "Recall":    round(recall_score(y_clf_test, y_pred), 3),
            "F1":        round(f1_score(y_clf_test, y_pred), 3),
            "AUC":       round(roc_auc_score(y_clf_test, y_prob), 3),
        }
    )
    roc_data[name] = (y_prob,)

comparison = pd.DataFrame(results).set_index("Model")

print("Four-model comparison — nonresponse prediction (same train/test split)")
print("=" * 70)
print(comparison.to_string())
print()
print("Key observation:")
best_auc = comparison["AUC"].max()
worst_auc = comparison["AUC"].min()
best_model = comparison["AUC"].idxmax()
print(f"  AUC range: {worst_auc:.3f} – {best_auc:.3f} ({(best_auc - worst_auc):.3f} spread)")
print(f"  Best AUC:  {best_model}")
print()
print("If the spread is < 0.02, the models are essentially equivalent on this data.")
print("The interpretability advantage of LR and DT does not cost measurable accuracy.")

# ---------------------------------------------------------------------------
# 3. ROC curve overlay
# ---------------------------------------------------------------------------
COLORS = ["steelblue", "firebrick", "forestgreen", "darkorchid"]
STYLES = ["-", "--", ":", "-."]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for (name, model, _), color, style in zip(MODEL_CONFIGS, COLORS, STYLES):
    # Reuse the stored probability from roc_data
    X_te = X_clf_test_sc if _ else X_clf_test  # _ == use_scaled

    # Re-predict (model already fitted above)
    y_prob = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_clf_test, y_prob)
    auc = roc_auc_score(y_clf_test, y_prob)
    axes[0].plot(fpr, tpr, color=color, ls=style, lw=2,
                 label=f"{name} ({auc:.3f})")

axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curves — all four models, same test set")
axes[0].legend(fontsize=8.5, title="Model (AUC)")

# ---------------------------------------------------------------------------
# 4. Metric bar chart
# ---------------------------------------------------------------------------
metric_cols = ["Accuracy", "F1", "AUC"]
x = np.arange(len(metric_cols))
w = 0.20

for i, (name, color) in enumerate(zip(comparison.index, COLORS)):
    vals = [comparison.loc[name, m] for m in metric_cols]
    axes[1].bar(x + i * w, vals, width=w, label=name,
                color=color, edgecolor="white", alpha=0.85)

axes[1].set_xticks(x + 1.5 * w)
axes[1].set_xticklabels(metric_cols)
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1)
axes[1].set_title("Key metrics by model")
axes[1].legend(fontsize=8.5)
axes[1].axhline(0.5, color="gray", lw=0.5, linestyle=":")

plt.suptitle(
    "Part I capstone: four model families on the same survey prediction task",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "06_four_model_comparison.png"),
    dpi=120,
    bbox_inches="tight",
)
plt.show()

print(f"\nPlot saved: 06_four_model_comparison.png")
