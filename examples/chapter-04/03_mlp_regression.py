"""
Chapter 4 — Example 03: MLP Regression (Income Prediction)
===========================================================
Trains a Multilayer Perceptron regressor to predict household income.

Regression-specific notes:
- The target (income) is standardised before training.  Without this,
  gradient descent struggles because the output layer weights need to span
  a range of $10 000 – $250 000, far outside the initialisation scale.
- After training, predictions are de-standardised (multiplied by the
  training standard deviation, then shifted by the training mean) before
  computing error metrics in the original dollar scale.
- The parity plot (actual vs. predicted) is the standard regression
  diagnostic.  A well-calibrated model scatters points symmetrically
  around the diagonal.  Systematic below-diagonal bias in the high-income
  range would suggest the model is not capturing the upper tail — a common
  pattern with log-income targets on small datasets.

Outputs
-------
- Console: MAE and R² on the test set.
- Plot: parity plot (actual vs. predicted income, $K) with perfect-prediction line.

Requirements
------------
Python 3.9+, numpy, pandas, matplotlib, scikit-learn
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------------------------
# Load dataset from 01_dataset_setup.py
# ---------------------------------------------------------------------------
from importlib.util import spec_from_file_location, module_from_spec

_spec = spec_from_file_location(
    "setup",
    os.path.join(os.path.dirname(__file__), "01_dataset_setup.py"),
)
_setup = module_from_spec(_spec)
_spec.loader.exec_module(_setup)

X_reg_train_sc = _setup.X_reg_train_sc
X_reg_test_sc = _setup.X_reg_test_sc
y_reg_train_sc = _setup.y_reg_train_sc
y_reg_test = _setup.y_reg_test
y_reg_mean = _setup.y_reg_mean
y_reg_std = _setup.y_reg_std

# ---------------------------------------------------------------------------
# 1. Train the regressor on standardised target
# ---------------------------------------------------------------------------
mlp_reg = MLPRegressor(
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
mlp_reg.fit(X_reg_train_sc, y_reg_train_sc)

# ---------------------------------------------------------------------------
# 2. De-standardise predictions for evaluation in original dollar scale
# ---------------------------------------------------------------------------
y_pred_sc = mlp_reg.predict(X_reg_test_sc)
y_pred = y_pred_sc * y_reg_std + y_reg_mean   # reverse the standardisation

mae = mean_absolute_error(y_reg_test, y_pred)
r2 = r2_score(y_reg_test, y_pred)

print("MLP Regressor — income prediction")
print("=" * 40)
print(f"Architecture:  {mlp_reg.hidden_layer_sizes}")
print(f"Epochs (early stop): {mlp_reg.n_iter_}")
print(f"Test MAE:      ${mae:,.0f}")
print(f"Test R²:       {r2:.3f}")
print()
print("Interpretation:")
print(f"  On average, predictions miss actual income by ${mae/1000:.1f}K.")
print(f"  R²={r2:.3f} means the model explains {r2:.1%} of income variance.")
print("  For context, a naive model predicting the mean always gives R²=0.")

# ---------------------------------------------------------------------------
# 3. Parity plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Parity plot
axes[0].scatter(
    y_reg_test / 1000,
    y_pred / 1000,
    alpha=0.35,
    s=14,
    color="steelblue",
    edgecolors="none",
)
lo = min(y_reg_test.min(), y_pred.min()) / 1000
hi = max(y_reg_test.max(), y_pred.max()) / 1000
axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
axes[0].set_xlabel("Actual income ($K)")
axes[0].set_ylabel("Predicted income ($K)")
axes[0].set_title(f"Parity plot — MLP\nMAE=${mae/1000:.1f}K, R²={r2:.3f}")
axes[0].legend()

# Residual distribution
residuals = (y_pred - y_reg_test) / 1000
axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
axes[1].axvline(0, color="firebrick", lw=1.5, linestyle="--", label="Zero error")
axes[1].set_xlabel("Prediction error ($K)")
axes[1].set_ylabel("Count")
axes[1].set_title("Residual distribution")
axes[1].legend()

plt.suptitle("MLP Regressor: income prediction diagnostics", fontsize=11)
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "03_mlp_regression.png"),
    dpi=120,
    bbox_inches="tight",
)
plt.show()

print(f"\nPlot saved: 03_mlp_regression.png")
print("\nNote: The upper-tail compression is expected — log-income generation")
print("creates a right skew that a small MLP does not fully capture.")
