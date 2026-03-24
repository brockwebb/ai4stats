"""
Chapter 4 — Example 02: MLP Classification (Nonresponse Prediction)
====================================================================
Trains a Multilayer Perceptron classifier to predict survey nonresponse
using the synthetic dataset from 01_dataset_setup.py.

Key decisions documented here:
- hidden_layer_sizes=(100, 50): two hidden layers.  Wider first layer captures
  more feature interactions; smaller second layer compresses the representation
  before the output.  A later script (04_architecture_search.py) benchmarks
  alternatives.
- early_stopping=True: halts training when the held-out validation loss
  stops improving for n_iter_no_change consecutive epochs.  This is the
  simplest overfitting defence available in sklearn's MLP.
- solver="adam": adaptive learning-rate optimiser.  Suitable for most survey
  datasets.  Alternative: "sgd" with a momentum schedule for larger datasets.

The training curve plot is the primary diagnostic output.  A well-behaved
curve shows steadily decreasing loss that flattens — not a curve that is still
falling steeply at the last epoch (undertrained) or one that oscillates
wildly (learning rate too high).

Outputs
-------
- Console: classification report with accuracy, precision, recall, F1, AUC.
- Plot 1: training loss curve.
- Plot 2: confusion matrix heatmap.

Requirements
------------
Python 3.9+, numpy, pandas, matplotlib, scikit-learn
"""

import sys
import os

# Allow running from repo root or from this directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Re-generate the dataset (same seed = same splits)
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
# 1. Define and train the classifier
# ---------------------------------------------------------------------------
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    alpha=0.0001,            # L2 regularisation (see 05_regularization.py)
    max_iter=500,
    random_state=42,
    early_stopping=True,     # stop if validation loss plateaus
    validation_fraction=0.10,
    n_iter_no_change=15,
    verbose=False,
)
mlp_clf.fit(X_clf_train_sc, y_clf_train)

# ---------------------------------------------------------------------------
# 2. Predictions and metrics
# ---------------------------------------------------------------------------
y_pred = mlp_clf.predict(X_clf_test_sc)
y_prob = mlp_clf.predict_proba(X_clf_test_sc)[:, 1]

acc = accuracy_score(y_clf_test, y_pred)
prec = precision_score(y_clf_test, y_pred)
rec = recall_score(y_clf_test, y_pred)
f1 = f1_score(y_clf_test, y_pred)
auc = roc_auc_score(y_clf_test, y_prob)

print("MLP Classifier — nonresponse prediction")
print("=" * 45)
print(f"Architecture:      {mlp_clf.hidden_layer_sizes}")
print(f"Epochs (early stop): {mlp_clf.n_iter_}")
print(f"Accuracy:          {acc:.3f}")
print(f"Precision:         {prec:.3f}")
print(f"Recall:            {rec:.3f}")
print(f"F1:                {f1:.3f}")
print(f"AUC-ROC:           {auc:.3f}")

# ---------------------------------------------------------------------------
# 3. Training curve — primary convergence diagnostic
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(mlp_clf.loss_curve_, color="steelblue", lw=2, label="Training loss")
axes[0].axvline(
    mlp_clf.n_iter_ - 1,
    color="firebrick",
    lw=1,
    linestyle="--",
    label=f"Early stop (epoch {mlp_clf.n_iter_})",
)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-entropy loss")
axes[0].set_title("Training loss curve")
axes[0].legend()

# ---------------------------------------------------------------------------
# 4. Confusion matrix
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_clf_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Responded", "Nonresponse"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion matrix (MLP)")

plt.suptitle(
    "MLP Classifier: training curve and confusion matrix",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "02_mlp_classification.png"),
    dpi=120,
    bbox_inches="tight",
)
plt.show()

print(f"\nPlot saved: 02_mlp_classification.png")
print("\nNote: compare these metrics to 06_four_model_comparison.py for full context.")
