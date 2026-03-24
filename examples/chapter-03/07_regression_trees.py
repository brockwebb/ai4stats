"""
Chapter 3 — Regression Trees: Income Prediction
================================================
Applies decision tree and Random Forest logic to a continuous target
(household income) using the same synthetic dataset from 01_dataset_setup.py.

The regression tree splits on MSE (mean squared error) within each node
rather than Gini impurity. Everything else — depth control, bootstrap
sampling, feature subsampling — is identical to the classification case.

Key outputs
-----------
1. Decision tree and Random Forest regression metrics (MAE, R^2).
2. Parity plots (actual vs. predicted) for both models.
3. OOB score for the Random Forest regressor.

Interpretation note
-------------------
Regression trees are useful for income imputation tasks where you want
a model that produces realistic conditional distributions without assuming
linearity. The printable rules still apply: a shallow regression tree can
be included in an imputation methodology memo.

Requirements
------------
Python 3.9+, numpy, pandas, matplotlib, scikit-learn
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

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

FEATURES_REG = ["age", "education_years", "hours_per_week", "urban"]
X_reg = df[FEATURES_REG]
y_reg = df["income"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

# ===========================================================================
# 1. Single regression tree
# ===========================================================================
dt_reg = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)
dt_reg.fit(X_reg_train, y_reg_train)

y_pred_tree_reg = dt_reg.predict(X_reg_test)
mae_tree = mean_absolute_error(y_reg_test, y_pred_tree_reg)
r2_tree = r2_score(y_reg_test, y_pred_tree_reg)

# ===========================================================================
# 2. Random Forest regressor
# ===========================================================================
rf_reg = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=5,
    max_features=0.5,  # 50% of features at each split — common for regression
    random_state=42,
    n_jobs=-1,
    oob_score=True,
)
rf_reg.fit(X_reg_train, y_reg_train)
y_pred_rf_reg = rf_reg.predict(X_reg_test)
mae_rf = mean_absolute_error(y_reg_test, y_pred_rf_reg)
r2_rf = r2_score(y_reg_test, y_pred_rf_reg)

# ===========================================================================
# 3. Print metrics
# ===========================================================================
print("=" * 60)
print("Income prediction (regression) — test set")
print("=" * 60)
print(f"  Decision Tree (depth=4):  MAE = ${mae_tree:,.0f},  R² = {r2_tree:.3f}")
print(f"  Random Forest (200 trees): MAE = ${mae_rf:,.0f},  R² = {r2_rf:.3f}")
print(f"  Random Forest OOB R²:      {rf_reg.oob_score_:.3f}")
print()

# ===========================================================================
# 4. Print regression tree rules (for shallow models — useful in methodology docs)
# ===========================================================================
print("Decision Tree regression rules (depth=4):")
print(export_text(dt_reg, feature_names=FEATURES_REG))

# ===========================================================================
# 5. Parity plots
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, name, y_pred in [
    (axes[0], "Decision Tree (depth=4)", y_pred_tree_reg),
    (axes[1], "Random Forest (200 trees)", y_pred_rf_reg),
]:
    ax.scatter(y_reg_test / 1000, y_pred / 1000, alpha=0.4, s=15, color="steelblue")
    lims = [20, 250]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual income ($K)")
    ax.set_ylabel("Predicted income ($K)")
    ax.set_title(name)
    r2 = r2_score(y_reg_test, y_pred)
    mae = mean_absolute_error(y_reg_test, y_pred)
    ax.text(
        0.05, 0.90,
        f"R\u00b2 = {r2:.3f}\nMAE = ${mae / 1000:.1f}K",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    ax.legend(fontsize=9)

plt.suptitle("Income prediction: actual vs. predicted (test set)", fontsize=12)
plt.tight_layout()
plt.savefig("regression_parity_plots.png", dpi=120, bbox_inches="tight")
print("Parity plots saved to regression_parity_plots.png")
plt.close()
