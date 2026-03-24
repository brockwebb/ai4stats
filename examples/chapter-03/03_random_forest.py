"""
Chapter 3 — Random Forest: OOB Scoring, n_estimators Effect, Feature Importance
=================================================================================
Fits a Random Forest classifier on the synthetic survey nonresponse dataset
and demonstrates the three key diagnostic tools: OOB score, the n_estimators
learning curve, and a comparison of Gini vs. permutation feature importance.

Why permutation importance for federal reports
----------------------------------------------
The built-in Gini importance is computed on training data only. It can
over-rank features with many unique values or correlated features. Permutation
importance shuffles each feature on the *test set* and measures how much
AUC-ROC drops. It is more reliable and is the version to cite in methodology
documentation.

Key outputs
-----------
1. Random Forest test-set metrics and OOB score.
2. n_estimators learning curve (performance stabilizes after ~100 trees).
3. Side-by-side Gini vs. permutation importance bar charts.
4. Permutation importance table suitable for a methodology appendix.

Requirements
------------
Python 3.9+, numpy, pandas, matplotlib, scikit-learn
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

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
# 1. Fit Random Forest (200 trees) with OOB scoring
# ===========================================================================
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,       # trees grow fully; forest controls variance via averaging
    min_samples_leaf=5,
    max_features="sqrt",  # sqrt(n_features) per split — standard for classification
    random_state=42,
    n_jobs=-1,
    oob_score=True,       # estimate generalization error from out-of-bag samples
)
rf_clf.fit(X_clf_train, y_clf_train)

y_pred_rf = rf_clf.predict(X_clf_test)
y_prob_rf = rf_clf.predict_proba(X_clf_test)[:, 1]

print("=" * 60)
print("Random Forest (200 trees) — test set performance")
print("=" * 60)
print(f"  OOB accuracy (train-time estimate):  {rf_clf.oob_score_:.3f}")
print(f"  Test accuracy:                       {accuracy_score(y_clf_test, y_pred_rf):.3f}")
print(f"  Test precision:                      {precision_score(y_clf_test, y_pred_rf):.3f}")
print(f"  Test recall:                         {recall_score(y_clf_test, y_pred_rf):.3f}")
print(f"  Test F1:                             {f1_score(y_clf_test, y_pred_rf):.3f}")
print(f"  Test AUC-ROC:                        {roc_auc_score(y_clf_test, y_prob_rf):.3f}")
print()

# ===========================================================================
# 2. n_estimators effect: learning curve
# ===========================================================================
n_trees_range = [1, 5, 10, 25, 50, 100, 200, 500]
oob_scores, test_aucs = [], []

for n_trees in n_trees_range:
    rf_temp = RandomForestClassifier(
        n_estimators=n_trees,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        oob_score=(n_trees > 1),
        n_jobs=-1,
    )
    rf_temp.fit(X_clf_train, y_clf_train)
    test_aucs.append(roc_auc_score(y_clf_test, rf_temp.predict_proba(X_clf_test)[:, 1]))
    oob_scores.append(rf_temp.oob_score_ if n_trees > 1 else float("nan"))

print("Test AUC-ROC by number of trees:")
for n_t, auc in zip(n_trees_range, test_aucs):
    print(f"  n_estimators={n_t:>4}:  AUC = {auc:.4f}")
print()

fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogx(n_trees_range, test_aucs, "o-", color="steelblue", label="Test AUC-ROC")
ax.semilogx(n_trees_range, oob_scores, "s--", color="firebrick", label="OOB accuracy")
ax.set_xlabel("Number of trees (log scale)")
ax.set_ylabel("Score")
ax.set_title("Performance stabilizes after ~100 trees (diminishing returns beyond that)")
ax.legend()
plt.tight_layout()
plt.savefig("rf_n_estimators.png", dpi=120, bbox_inches="tight")
print("n_estimators curve saved to rf_n_estimators.png")
plt.close()

# ===========================================================================
# 3. Gini importance vs. permutation importance
# ===========================================================================
gini_imp = pd.Series(rf_clf.feature_importances_, index=FEATURES_CLF).sort_values(
    ascending=True
)

perm = permutation_importance(
    rf_clf,
    X_clf_test,
    y_clf_test,
    n_repeats=20,
    random_state=42,
    scoring="roc_auc",
    n_jobs=-1,
)
perm_imp = pd.Series(perm.importances_mean, index=FEATURES_CLF).sort_values(ascending=True)
perm_err = pd.Series(perm.importances_std, index=FEATURES_CLF).reindex(perm_imp.index)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

gini_imp.plot(kind="barh", ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title("Feature importance: mean Gini decrease\n(built into the forest, training data)")
axes[0].set_xlabel("Importance")
axes[0].axvline(0, color="black", lw=0.5)

axes[1].barh(
    perm_imp.index,
    perm_imp.values,
    xerr=perm_err.values,
    color="firebrick",
    edgecolor="white",
    capsize=4,
)
axes[1].set_title(
    "Feature importance: permutation (AUC drop)\n(more reliable; computed on test set)"
)
axes[1].set_xlabel("Mean AUC decrease when feature is shuffled")
axes[1].axvline(0, color="black", lw=0.5)

plt.tight_layout()
plt.savefig("feature_importance_comparison.png", dpi=120, bbox_inches="tight")
print("Feature importance chart saved to feature_importance_comparison.png")
plt.close()

print("=" * 60)
print("Permutation importance table (use this in methodology reports)")
print("=" * 60)
perm_table = pd.DataFrame(
    {
        "feature": FEATURES_CLF,
        "mean_auc_drop": perm.importances_mean.round(4),
        "std": perm.importances_std.round(4),
    }
).sort_values("mean_auc_drop", ascending=False)
print(perm_table.to_string(index=False))
