"""
Chapter 4 — Example 04: Architecture Search
============================================
Systematically compares six MLP hidden-layer configurations to illustrate
that larger architectures do not reliably outperform smaller ones on
tabular survey data of modest size.

The six configurations span two design dimensions:
- Depth: one hidden layer vs. two vs. three.
- Width: 50 to 200 units per layer.

Key finding this script is designed to surface:
On n=1200 survey records, the performance differences across architectures
are typically within 0.5–1 AUC points — well within random variation.
Larger architectures also take more epochs to converge and are more
sensitive to the learning rate.  The right takeaway is not "use the biggest
network" but "more parameters do not compensate for small data."

Outputs
-------
- Console: results table sorted by test AUC-ROC.
- Plot: bar chart of AUC-ROC by architecture.

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

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

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

X_clf_train_sc = _setup.X_clf_train_sc
X_clf_test_sc = _setup.X_clf_test_sc
y_clf_train = _setup.y_clf_train
y_clf_test = _setup.y_clf_test

# ---------------------------------------------------------------------------
# 1. Architecture definitions
#    Six configurations: vary depth (1–3 layers) and width (50–200 units)
# ---------------------------------------------------------------------------
ARCHITECTURES = {
    "(50,)":           (50,),
    "(100,)":          (100,),
    "(50, 50)":        (50, 50),
    "(100, 50)":       (100, 50),
    "(100, 100)":      (100, 100),
    "(100, 50, 25)":   (100, 50, 25),
}

COMMON_KWARGS = dict(
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

# ---------------------------------------------------------------------------
# 2. Fit each architecture and collect metrics
# ---------------------------------------------------------------------------
results = []
for label, sizes in ARCHITECTURES.items():
    model = MLPClassifier(hidden_layer_sizes=sizes, **COMMON_KWARGS)
    model.fit(X_clf_train_sc, y_clf_train)

    train_auc = roc_auc_score(
        y_clf_train, model.predict_proba(X_clf_train_sc)[:, 1]
    )
    test_auc = roc_auc_score(
        y_clf_test, model.predict_proba(X_clf_test_sc)[:, 1]
    )
    n_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)

    results.append(
        {
            "Architecture": label,
            "Parameters": n_params,
            "Epochs": model.n_iter_,
            "Train AUC": round(train_auc, 4),
            "Test AUC": round(test_auc, 4),
            "AUC gap (overfit)": round(train_auc - test_auc, 4),
        }
    )

arch_df = pd.DataFrame(results).sort_values("Test AUC", ascending=False).reset_index(
    drop=True
)

print("Architecture search results (sorted by Test AUC-ROC):")
print(arch_df.to_string(index=False))
print()
best = arch_df.iloc[0]
worst = arch_df.iloc[-1]
print(
    f"Range of test AUC: {worst['Test AUC']:.4f} – {best['Test AUC']:.4f} "
    f"({(best['Test AUC'] - worst['Test AUC']):.4f} spread)"
)
print(
    "Takeaway: differences are small relative to the parameter count increase."
)

# ---------------------------------------------------------------------------
# 3. Bar chart — test AUC by architecture
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(arch_df)))
bars = axes[0].bar(
    arch_df["Architecture"],
    arch_df["Test AUC"],
    color=colors,
    edgecolor="white",
)
axes[0].axhline(
    arch_df["Test AUC"].max(),
    color="firebrick",
    lw=1,
    linestyle="--",
    label=f"Best = {arch_df['Test AUC'].max():.4f}",
)
axes[0].set_ylim(
    max(0.5, arch_df["Test AUC"].min() - 0.02),
    min(1.0, arch_df["Test AUC"].max() + 0.03),
)
axes[0].set_xlabel("Architecture (hidden layer sizes)")
axes[0].set_ylabel("Test AUC-ROC")
axes[0].set_title("Test AUC-ROC by architecture\n(bigger is not always better)")
axes[0].legend()
plt.setp(axes[0].get_xticklabels(), rotation=25, ha="right")

# Train vs. test AUC — shows overfitting gap
x = np.arange(len(arch_df))
w = 0.35
axes[1].bar(x - w / 2, arch_df["Train AUC"], width=w, label="Train AUC", color="steelblue", alpha=0.8)
axes[1].bar(x + w / 2, arch_df["Test AUC"], width=w, label="Test AUC", color="firebrick", alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(arch_df["Architecture"], rotation=25, ha="right")
axes[1].set_ylabel("AUC-ROC")
axes[1].set_title("Train vs. test AUC\n(gap = overfitting signal)")
axes[1].legend()
axes[1].set_ylim(
    max(0.5, arch_df[["Train AUC", "Test AUC"]].min().min() - 0.02),
    min(1.0, arch_df[["Train AUC", "Test AUC"]].max().max() + 0.03),
)

plt.suptitle(
    "Architecture search: six MLP configurations on the same survey dataset",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "04_architecture_search.png"),
    dpi=120,
    bbox_inches="tight",
)
plt.show()

print(f"\nPlot saved: 04_architecture_search.png")
