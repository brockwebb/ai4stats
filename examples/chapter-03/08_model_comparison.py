"""
Chapter 3 — Three-Model Comparison: Logistic Regression, Decision Tree, Random Forest
======================================================================================
Evaluates all three classifiers on the *same* held-out test set, making direct
comparison valid. This is the concluding analysis for the classification task.

All three models are fit from scratch using identical train/test splits
(seed=42, stratified 80/20 split).

Key outputs
-----------
1. Comparison table: Accuracy, Precision, Recall, F1, AUC-ROC for all three models.
2. ROC curves overlaid on the same axes.
3. Metric bar chart for visual comparison.

Interpretation for federal analysts
------------------------------------
On this synthetic dataset the three models perform similarly — which is the
expected result for well-behaved tabular data with a clear linear signal.
The choice between them is about auditability, not accuracy:
- Logistic regression: fastest to explain, coefficients have log-odds interpretation.
- Decision tree: printable rules, direct audit trail.
- Random Forest: best AUC, but no single printable rule set.

Requirements
------------
Python 3.9+, numpy, pandas, matplotlib, scikit-learn
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
# 1. Fit all three models
# ===========================================================================

# Logistic regression requires standardized features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_clf_train)
X_test_sc = scaler.transform(X_clf_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_clf_train)
y_pred_lr = lr.predict(X_test_sc)
y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]

dt_best = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_best.fit(X_clf_train, y_clf_train)
y_pred_dt = dt_best.predict(X_clf_test)
y_prob_dt = dt_best.predict_proba(X_clf_test)[:, 1]

rf_clf = RandomForestClassifier(
    n_estimators=200, min_samples_leaf=5, max_features="sqrt",
    random_state=42, n_jobs=-1, oob_score=True,
)
rf_clf.fit(X_clf_train, y_clf_train)
y_pred_rf = rf_clf.predict(X_clf_test)
y_prob_rf = rf_clf.predict_proba(X_clf_test)[:, 1]

# ===========================================================================
# 2. Comparison table
# ===========================================================================
models = {
    "Logistic Regression": (y_pred_lr, y_prob_lr),
    "Decision Tree (d=3)": (y_pred_dt, y_prob_dt),
    "Random Forest (200)": (y_pred_rf, y_prob_rf),
}

rows = []
for name, (y_pred, y_prob) in models.items():
    rows.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_clf_test, y_pred),
            "Precision": precision_score(y_clf_test, y_pred),
            "Recall": recall_score(y_clf_test, y_pred),
            "F1": f1_score(y_clf_test, y_pred),
            "AUC-ROC": roc_auc_score(y_clf_test, y_prob),
        }
    )

comparison = pd.DataFrame(rows).set_index("Model").round(3)
print("=" * 60)
print("Model comparison — held-out test set")
print("=" * 60)
print(comparison.to_string())
print()

# ===========================================================================
# 3. ROC curves + metric bar chart
# ===========================================================================
colors_roc = ["steelblue", "firebrick", "forestgreen"]
line_styles = ["-", "--", ":"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for (name, (y_pred, y_prob)), ls, col in zip(models.items(), line_styles, colors_roc):
    fpr, tpr, _ = roc_curve(y_clf_test, y_prob)
    auc = roc_auc_score(y_clf_test, y_prob)
    axes[0].plot(fpr, tpr, ls=ls, color=col, lw=2, label=f"{name} (AUC={auc:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC curves: all three models, same test set")
axes[0].legend(fontsize=9)

metric_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
x = np.arange(len(metric_cols))
w = 0.25
for i, (name, row) in enumerate(comparison.iterrows()):
    axes[1].bar(
        x + i * w, row[metric_cols], width=w,
        label=name, color=colors_roc[i], edgecolor="white", alpha=0.85,
    )
axes[1].set_xticks(x + w)
axes[1].set_xticklabels(metric_cols, rotation=15)
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1)
axes[1].set_title("Test set metrics by model")
axes[1].legend(fontsize=9)
axes[1].axhline(0.5, color="gray", lw=0.5, linestyle=":")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120, bbox_inches="tight")
print("Comparison chart saved to model_comparison.png")
plt.close()
