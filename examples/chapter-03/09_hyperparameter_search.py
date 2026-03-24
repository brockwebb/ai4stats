"""
Chapter 3 — Hyperparameter Search: RandomizedSearchCV for Random Forest
========================================================================
Demonstrates the end-to-end tuning workflow for a Random Forest classifier.
Uses RandomizedSearchCV with 5-fold cross-validation to explore the
hyperparameter space without touching the held-out test set during tuning.

Protocol — do NOT deviate from this order
------------------------------------------
1. Establish feature set and train/test split.
2. Fit a baseline shallow tree (interpretable, fast).
3. Fit a default Random Forest and compare to baseline.
4. Run randomized hyperparameter search on the train set only.
5. Evaluate the best model on the test set *exactly once*.
6. Report permutation importance from the best model.
7. Attach the shallow tree rules to methodology documentation.

Why randomized search instead of grid search
---------------------------------------------
Grid search evaluates every combination, which is expensive. Randomized
search samples n_iter combinations from a distribution over the parameter
space. With n_iter=24 and cv=5 this is 120 model fits — feasible in a
few minutes on a laptop, and in practice captures most of the benefit of
a full grid search.

Key outputs
-----------
1. Best hyperparameters and best CV AUC-ROC.
2. Tuned model test AUC-ROC.
3. Top 10 candidate configurations from the search.

Requirements
------------
Python 3.9+, numpy, pandas, scikit-learn
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

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

FEATURES_CLF = ["age", "education_years", "urban", "contact_attempts", "prior_response"]
X_clf = df[FEATURES_CLF]
y_clf = df["responded"]

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf
)

# ===========================================================================
# Hyperparameter search space
# ===========================================================================
# Each key maps to a list of candidate values. RandomizedSearchCV samples
# uniformly from each list for each of the n_iter configurations.
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_leaf": [1, 5, 10, 20],
    "max_features": ["sqrt", "log2", 0.5],
}

# ===========================================================================
# Run search (train set only — test set is never touched here)
# ===========================================================================
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True),
    param_distributions=param_dist,
    n_iter=24,        # number of candidate configurations to try
    cv=5,             # 5-fold cross-validation
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
    return_train_score=True,
)
rf_search.fit(X_clf_train, y_clf_train)

print("=" * 60)
print("RandomizedSearchCV results")
print("=" * 60)
print(f"Best hyperparameters: {rf_search.best_params_}")
print(f"Best CV AUC-ROC:      {rf_search.best_score_:.4f}")
print()

# ===========================================================================
# Evaluate the best model on the held-out test set — exactly once
# ===========================================================================
rf_tuned = rf_search.best_estimator_
y_prob_tuned = rf_tuned.predict_proba(X_clf_test)[:, 1]
test_auc = roc_auc_score(y_clf_test, y_prob_tuned)
print(f"Tuned model test AUC-ROC: {test_auc:.4f}")
print()

# ===========================================================================
# Show the top 10 configurations ranked by mean CV AUC
# ===========================================================================
cv_results = pd.DataFrame(rf_search.cv_results_)
top10 = (
    cv_results[["params", "mean_test_score", "std_test_score", "mean_train_score"]]
    .sort_values("mean_test_score", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10["mean_test_score"] = top10["mean_test_score"].round(4)
top10["std_test_score"] = top10["std_test_score"].round(4)
top10["mean_train_score"] = top10["mean_train_score"].round(4)

print("Top 10 configurations by mean CV AUC-ROC:")
for _, row in top10.iterrows():
    print(
        f"  CV AUC={row['mean_test_score']:.4f} (std={row['std_test_score']:.4f})"
        f"  train={row['mean_train_score']:.4f}  params={row['params']}"
    )
