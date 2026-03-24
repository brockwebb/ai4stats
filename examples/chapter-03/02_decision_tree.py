"""
Chapter 3 — Decision Tree: Fitting, Rules, and Overfitting Analysis
====================================================================
Demonstrates the full decision tree workflow on the synthetic survey
dataset from 01_dataset_setup.py. Focus is on the artifacts that
matter for federal work: printed decision rules and the overfitting
depth curve.

Key outputs
-----------
1. Fitted depth-3 tree with train/test accuracy.
2. Text rules via export_text() — copy-pasteable into a methodology report.
3. Visualization of the tree structure (saved to figures/ if run standalone).
4. Gini impurity manual calculation for the first candidate split.
5. Overfitting analysis: train vs. test accuracy across depth 1-19.
6. min_samples_leaf comparison table.

Auditability note
-----------------
The export_text() output is the complete decision logic of the model.
A methodology reviewer does not need to run any code to understand it.
This is a qualitative advantage over logistic regression coefficients
(which require log-odds interpretation) and a large advantage over
neural networks (which have no equivalent printout).

Requirements
------------
Python 3.9+, numpy, pandas, matplotlib, scikit-learn
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

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

from sklearn.model_selection import train_test_split

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf
)

# ===========================================================================
# 1. Fit a shallow decision tree (depth 3)
# ===========================================================================
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_clf_train, y_clf_train)

print("=" * 60)
print("Decision Tree (max_depth=3)")
print("=" * 60)
print(f"  Tree depth:     {dt.get_depth()}")
print(f"  Number of leaves: {dt.get_n_leaves()}")
print(f"  Train accuracy: {dt.score(X_clf_train, y_clf_train):.3f}")
print(f"  Test accuracy:  {dt.score(X_clf_test, y_clf_test):.3f}")
print()

# ===========================================================================
# 2. Export text rules — the methodology attachment
# ===========================================================================
rules = export_text(dt, feature_names=FEATURES_CLF)
print("Decision rules (can be copied into a methodology report):")
print()
print(rules)

# ===========================================================================
# 3. Visualize the tree structure
# ===========================================================================
fig, ax = plt.subplots(figsize=(18, 7))
plot_tree(
    dt,
    feature_names=FEATURES_CLF,
    class_names=["not responded", "responded"],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
    impurity=True,
    proportion=False,
)
ax.set_title("Decision tree (max_depth=3) for nonresponse prediction", fontsize=13)
plt.tight_layout()
plt.savefig("tree_structure.png", dpi=120, bbox_inches="tight")
print("Tree visualization saved to tree_structure.png")
plt.close()

# ===========================================================================
# 4. Manual Gini impurity calculation for the first candidate split
# ===========================================================================

def gini(y: pd.Series) -> float:
    """Compute Gini impurity for a binary label series."""
    if len(y) == 0:
        return 0.0
    p = y.mean()
    return 1 - p**2 - (1 - p) ** 2


mask_left = X_clf_train["contact_attempts"] <= 2
mask_right = ~mask_left
y_train = y_clf_train

g_left = gini(y_train[mask_left])
g_right = gini(y_train[mask_right])
n_left = mask_left.sum()
n_right = mask_right.sum()
n_total = len(y_train)
weighted_gini = (n_left / n_total) * g_left + (n_right / n_total) * g_right

print("=" * 60)
print("Manual Gini calculation — split on contact_attempts <= 2")
print("=" * 60)
print(f"  Left  node: n={n_left}, Gini={g_left:.4f}, "
      f"response rate={y_train[mask_left].mean():.2%}")
print(f"  Right node: n={n_right}, Gini={g_right:.4f}, "
      f"response rate={y_train[mask_right].mean():.2%}")
print(f"  Weighted Gini after split:  {weighted_gini:.4f}")
print(f"  Parent Gini (before split): {gini(y_train):.4f}")
print(f"  Gini reduction:             {gini(y_train) - weighted_gini:.4f}")
print()

# ===========================================================================
# 5. Overfitting analysis: depth 1-19
# ===========================================================================
depths = range(1, 20)
train_scores, test_scores = [], []

for d in depths:
    dt_d = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_d.fit(X_clf_train, y_clf_train)
    train_scores.append(dt_d.score(X_clf_train, y_clf_train))
    test_scores.append(dt_d.score(X_clf_test, y_clf_test))

best_depth = list(depths)[int(np.argmax(test_scores))]
print(f"Best test accuracy at max_depth={best_depth}: {max(test_scores):.3f}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(depths, train_scores, "o-", label="Train accuracy", color="steelblue")
ax.plot(depths, test_scores, "s-", label="Test accuracy", color="firebrick")
ax.set_xlabel("max_depth")
ax.set_ylabel("Accuracy")
ax.set_title("Overfitting: train accuracy climbs while test accuracy levels off or drops")
ax.axvline(3, color="gray", linestyle="--", label="depth=3 (chosen)")
ax.legend()
plt.tight_layout()
plt.savefig("overfitting_curve.png", dpi=120, bbox_inches="tight")
print("Overfitting curve saved to overfitting_curve.png")
plt.close()

# ===========================================================================
# 6. min_samples_leaf comparison
# ===========================================================================
min_leaf_sizes = [1, 5, 10, 20, 50]
results = []
for msl in min_leaf_sizes:
    dt_msl = DecisionTreeClassifier(max_depth=10, min_samples_leaf=msl, random_state=42)
    dt_msl.fit(X_clf_train, y_clf_train)
    results.append(
        {
            "min_samples_leaf": msl,
            "n_leaves": dt_msl.get_n_leaves(),
            "train_acc": round(dt_msl.score(X_clf_train, y_clf_train), 3),
            "test_acc": round(dt_msl.score(X_clf_test, y_clf_test), 3),
        }
    )

print()
print("=" * 60)
print("min_samples_leaf comparison (max_depth=10)")
print("=" * 60)
print(pd.DataFrame(results).to_string(index=False))
