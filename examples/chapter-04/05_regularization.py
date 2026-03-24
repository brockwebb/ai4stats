"""
Chapter 4 — Example 05: Regularisation (Alpha Sweep)
=====================================================
Demonstrates how the L2 regularisation penalty (alpha) governs the
bias-variance trade-off in MLPClassifier.

What alpha does:
- alpha=0 (no regularisation): weights can grow arbitrarily large.  The
  model memorises the training set.  Train AUC >> Test AUC.
- Large alpha: weights are penalised toward zero, effectively shrinking
  the model.  Extreme values produce underfitting: both Train and Test AUC
  fall.
- The sweet spot is the alpha value where Test AUC peaks.

Why this matters for federal analysts:
Vendors who report only training accuracy may be showing an overfit model.
The train-test AUC gap is the diagnostic signal.  A large gap (> 5 AUC
points) is a warning sign regardless of how impressive the training number
looks.

Outputs
-------
- Console: alpha sweep table showing train AUC, test AUC, and gap.
- Plot: semilog plot of train vs. test AUC across the alpha range.

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
# 1. Alpha sweep — 8 values spanning 5 orders of magnitude
# ---------------------------------------------------------------------------
ALPHAS = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

COMMON_KWARGS = dict(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=15,
    verbose=False,
)

rows = []
for alpha in ALPHAS:
    model = MLPClassifier(alpha=alpha, **COMMON_KWARGS)
    model.fit(X_clf_train_sc, y_clf_train)

    train_auc = roc_auc_score(
        y_clf_train, model.predict_proba(X_clf_train_sc)[:, 1]
    )
    test_auc = roc_auc_score(
        y_clf_test, model.predict_proba(X_clf_test_sc)[:, 1]
    )
    rows.append(
        {
            "alpha": alpha,
            "Train AUC": round(train_auc, 4),
            "Test AUC": round(test_auc, 4),
            "Gap (overfit)": round(train_auc - test_auc, 4),
            "Epochs": model.n_iter_,
        }
    )

alpha_df = pd.DataFrame(rows)

print("Regularisation sweep (MLPClassifier, architecture=(100, 50))")
print("=" * 60)
print(alpha_df.to_string(index=False))
print()

best_row = alpha_df.loc[alpha_df["Test AUC"].idxmax()]
print(f"Best test AUC: {best_row['Test AUC']:.4f} at alpha={best_row['alpha']}")
print()
print("Reading the gap column:")
print("  Large gap (> 0.05): overfitting — increase alpha")
print("  Near-zero gap with low test AUC: underfitting — decrease alpha")

# ---------------------------------------------------------------------------
# 2. Plot train vs. test AUC across alpha
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].semilogx(
    alpha_df["alpha"], alpha_df["Train AUC"],
    "o-", color="steelblue", lw=2, label="Train AUC",
)
axes[0].semilogx(
    alpha_df["alpha"], alpha_df["Test AUC"],
    "s-", color="firebrick", lw=2, label="Test AUC",
)
axes[0].axvline(
    best_row["alpha"],
    color="forestgreen",
    lw=1,
    linestyle="--",
    label=f"Best test AUC (alpha={best_row['alpha']})",
)
axes[0].set_xlabel("alpha (L2 regularisation strength, log scale)")
axes[0].set_ylabel("AUC-ROC")
axes[0].set_title("Bias-variance trade-off:\nunderfitting ← alpha → overfitting")
axes[0].legend(fontsize=9)

# Gap plot
axes[1].semilogx(
    alpha_df["alpha"], alpha_df["Gap (overfit)"],
    "D-", color="darkorchid", lw=2,
)
axes[1].axhline(0, color="black", lw=0.5)
axes[1].axhline(0.05, color="firebrick", lw=1, linestyle=":", label="5-point warning threshold")
axes[1].set_xlabel("alpha (log scale)")
axes[1].set_ylabel("Train AUC − Test AUC")
axes[1].set_title("Overfitting gap\n(> 0.05 is a warning sign)")
axes[1].legend(fontsize=9)

plt.suptitle(
    "L2 regularisation (alpha) controls the overfitting-underfitting balance",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "05_regularization.png"),
    dpi=120,
    bbox_inches="tight",
)
plt.show()

print(f"\nPlot saved: 05_regularization.png")
