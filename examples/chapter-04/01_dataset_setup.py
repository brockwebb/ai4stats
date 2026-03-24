"""
Chapter 4 — Example 01: Dataset Setup with Scaling
====================================================
Generates the synthetic federal survey dataset used throughout Chapter 4
and prepares it for neural network training via StandardScaler.

The same random seed (42) and n=1200 sample size are used in every
Chapter 4 example so results are directly comparable across scripts.

Key preparation steps beyond Chapters 1-3:
- StandardScaler is fitted on the training split ONLY, then applied to both
  train and test.  Fitting on the full dataset would leak test-set information
  into the scaler statistics (data leakage).
- The regression target (income) is also standardised.  MLPRegressor converges
  much faster when the output range matches the typical weight initialisation
  scale (~0 ± 1) rather than spanning $10 000 – $250 000.

Outputs
-------
Prints dataset shape and a brief summary.  All split/scaled arrays are
defined in module scope so downstream scripts can import them directly::

    from examples.chapter_04.01_dataset_setup import (
        X_clf_train_sc, X_clf_test_sc, y_clf_train, y_clf_test,
        X_reg_train_sc, X_reg_test_sc, y_reg_train, y_reg_test,
        y_reg_mean, y_reg_std, FEATURES_CLF, FEATURES_REG,
    )

Requirements
------------
Python 3.9+, numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Reproducible seed
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N = 1200

# ---------------------------------------------------------------------------
# 2. Synthetic survey data generation
# ---------------------------------------------------------------------------
states = np.random.choice(
    ["California", "Texas", "New York", "Florida", "Illinois"],
    size=N,
    p=[0.20, 0.20, 0.18, 0.17, 0.25],
)
age = np.random.normal(42, 14, N).clip(18, 80).astype(int)
education_years = np.random.choice(
    [9, 12, 14, 16, 18], size=N, p=[0.10, 0.35, 0.20, 0.25, 0.10]
)
hours_per_week = np.random.normal(38, 10, N).clip(0, 80).astype(int)
urban = np.random.binomial(1, 0.72, N)
contact_attempts = np.random.poisson(2, N).clip(1, 7)
prior_response = np.random.binomial(1, 0.68, N)

# Log-normal income with plausible demographic relationships
log_income = (
    9.2
    + 0.04 * (education_years - 12)
    + 0.008 * age
    + 0.003 * hours_per_week
    + np.random.normal(0, 0.35, N)
)
income = np.exp(log_income).clip(10_000, 250_000).astype(int)

# Nonresponse logit: prior response and urban suppres nonresponse probability
logit_nr = (
    -0.5
    + 0.25 * contact_attempts
    - 1.2 * prior_response
    - 0.3 * urban
    + 0.01 * (age - 42)
    + np.random.normal(0, 0.3, N)
)
responded = (np.random.uniform(size=N) > 1 / (1 + np.exp(-logit_nr))).astype(int)

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

# ---------------------------------------------------------------------------
# 3. Feature selection
# ---------------------------------------------------------------------------
FEATURES_CLF = ["age", "education_years", "urban", "contact_attempts", "prior_response"]
FEATURES_REG = ["age", "education_years", "hours_per_week", "urban"]

X_clf = df[FEATURES_CLF]
y_clf = df["responded"]
X_reg = df[FEATURES_REG]
y_reg = df["income"]

# ---------------------------------------------------------------------------
# 4. Train/test split
# ---------------------------------------------------------------------------
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.20, random_state=RANDOM_SEED, stratify=y_clf
)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=RANDOM_SEED
)

# ---------------------------------------------------------------------------
# 5. Feature scaling — fit on train only, apply to both splits
#    Neural networks require features on a common scale.
#    Logistic regression also converges faster with scaled inputs.
#    Decision trees and Random Forests do NOT require scaling.
# ---------------------------------------------------------------------------
scaler_clf = StandardScaler()
X_clf_train_sc = scaler_clf.fit_transform(X_clf_train)
X_clf_test_sc = scaler_clf.transform(X_clf_test)

scaler_reg = StandardScaler()
X_reg_train_sc = scaler_reg.fit_transform(X_reg_train)
X_reg_test_sc = scaler_reg.transform(X_reg_test)

# ---------------------------------------------------------------------------
# 6. Target standardisation for regression
#    Store mean and std so predictions can be de-standardised for evaluation.
# ---------------------------------------------------------------------------
y_reg_mean = y_reg_train.mean()
y_reg_std = y_reg_train.std()
y_reg_train_sc = (y_reg_train - y_reg_mean) / y_reg_std

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Dataset shape:           {df.shape}")
    print(f"Classification features: {FEATURES_CLF}")
    print(f"Regression features:     {FEATURES_REG}")
    print(f"Train size (clf):        {len(X_clf_train)}")
    print(f"Test size  (clf):        {len(X_clf_test)}")
    print(f"Response rate (train):   {y_clf_train.mean():.1%}")
    print(f"Income range (train):    ${y_reg_train.min():,} – ${y_reg_train.max():,}")
    print(f"Income mean (train):     ${y_reg_mean:,.0f}")
    print(f"Income std  (train):     ${y_reg_std:,.0f}")
    print()
    print("Scaled classification features (first row):")
    print(dict(zip(FEATURES_CLF, X_clf_train_sc[0].round(3))))
    print()
    print("Dataset ready for neural network training.")
