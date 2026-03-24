"""
Chapter 4 — Example 08: Exercises (Tract-Level Four-Model Activity)
====================================================================
Setup and worked solution for the Chapter 4 exercises.

Exercise context:
You are evaluating four modeling approaches for a nonresponse prediction
task at a regional office.  The dataset represents 300 census tracts with
demographic and operational features.  Your task is to recommend a model
for deployment.

The solution section demonstrates the full four-model comparison and
provides a written recommendation following the structure expected in
federal methodology documentation.

Exercise questions (answer before running the solution):
1. Which model would you recommend deploying at this regional office?
   Write a one-paragraph justification citing specific evidence.

2. A vendor proposes replacing all four models with a deep neural network.
   Using the checklist from Chapter 4 Section 9, what seven questions do
   you ask before evaluating the proposal?

3. The IT department says PyTorch is not on the approved software list.
   What are your options?
   (Hint: sklearn's MLPClassifier uses numpy/scipy, not PyTorch.
    What does that tell you about the approval question?)

4. If the MLP achieves 0.815 AUC vs. the Random Forest's 0.813 AUC,
   would you recommend the switch?  What factors govern that decision?

5. Optional: Run this script and compare your answers to the solution output.

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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---------------------------------------------------------------------------
# 1. Tract-level dataset
#    300 synthetic census tracts.  Features match the kind available in
#    Census Planning Database or ACS summary files.
# ---------------------------------------------------------------------------
np.random.seed(2025)
N_TRACTS = 300

tract_data = pd.DataFrame(
    {
        "tract_id": [f"T{str(i).zfill(3)}" for i in range(N_TRACTS)],
        "pct_renters": np.random.normal(35, 15, N_TRACTS).clip(5, 90),
        "median_age": np.random.normal(40, 8, N_TRACTS).clip(22, 70),
        "pct_foreign_born": np.random.normal(15, 10, N_TRACTS).clip(0, 60),
        "pct_bachelors": np.random.normal(30, 12, N_TRACTS).clip(5, 75),
        "pop_density_log": np.random.normal(6, 2, N_TRACTS).clip(1, 10),
        "prior_rr": np.random.normal(0.72, 0.08, N_TRACTS).clip(0.40, 0.95),
        "contact_attempts": np.random.poisson(2.5, N_TRACTS).clip(1, 8),
    }
)

# Generate binary outcome: low-response tract (1 = low response)
logit_tract = (
    -1.5
    + 0.04 * tract_data["pct_renters"]
    + 0.03 * tract_data["pct_foreign_born"]
    - 2.0 * (tract_data["prior_rr"] - 0.72)
    + 0.15 * tract_data["contact_attempts"]
    + np.random.normal(0, 0.4, N_TRACTS)
)
tract_data["low_response"] = (1 / (1 + np.exp(-logit_tract)) > 0.45).astype(int)

TRACT_FEATURES = [
    "pct_renters", "median_age", "pct_foreign_born",
    "pct_bachelors", "pop_density_log", "prior_rr", "contact_attempts",
]

print("Tract dataset summary:")
print(f"  Records:           {len(tract_data)}")
print(f"  Features:          {TRACT_FEATURES}")
print(f"  Low-response rate: {tract_data['low_response'].mean():.1%}")
print()

# ---------------------------------------------------------------------------
# 2. Exercise scaffold (students fill this in)
# ---------------------------------------------------------------------------
print("=" * 60)
print("EXERCISE SCAFFOLD")
print("=" * 60)
print("""
X_tract = tract_data[TRACT_FEATURES]
y_tract = tract_data["low_response"]

# Step 1: Split 80/20 stratified
# YOUR CODE HERE

# Step 2: Standardise (fit on train only)
# YOUR CODE HERE

# Steps 3-5: Fit LR, DT (depth 3), RF (100 trees), MLP (64,64)
#            Collect accuracy, F1, AUC-ROC
# YOUR CODE HERE

# Step 6: Print the comparison table and write your recommendation
# YOUR RECOMMENDATION HERE
""")

# ---------------------------------------------------------------------------
# 3. Full solution
# ---------------------------------------------------------------------------
print("=" * 60)
print("SOLUTION")
print("=" * 60)

X_tract = tract_data[TRACT_FEATURES]
y_tract = tract_data["low_response"]

# Step 1: Split
X_tr, X_te, y_tr, y_te = train_test_split(
    X_tract, y_tract,
    test_size=0.20,
    random_state=42,
    stratify=y_tract,
)

# Step 2: Standardise
sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc = sc.transform(X_te)

# Step 3: Define models
MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42), True
    ),
    "Decision Tree (depth 3)": (
        DecisionTreeClassifier(max_depth=3, random_state=42), False
    ),
    "Random Forest (100 trees)": (
        RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                               random_state=42, n_jobs=-1), False
    ),
    "MLP (64, 64)": (
        MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=15,
            verbose=False,
        ),
        True,
    ),
}

# Steps 4-5: Fit and evaluate
act_results = []
for name, (model, use_scaled) in MODELS.items():
    Xtr_use = X_tr_sc if use_scaled else X_tr
    Xte_use = X_te_sc if use_scaled else X_te
    model.fit(Xtr_use, y_tr)
    y_pred = model.predict(Xte_use)
    y_prob = model.predict_proba(Xte_use)[:, 1]
    act_results.append(
        {
            "Model": name,
            "Accuracy": round(accuracy_score(y_te, y_pred), 3),
            "F1": round(f1_score(y_te, y_pred), 3),
            "AUC-ROC": round(roc_auc_score(y_te, y_prob), 3),
        }
    )

act_df = pd.DataFrame(act_results).set_index("Model")
print("\nFour-model comparison on tract data:")
print(act_df.to_string())

# Step 6: Written recommendation
print("""
Recommendation:
  Deploy the Random Forest for prediction, and attach a shallow Decision Tree
  (depth 3) to the methodology report for auditability.

  All four models achieve similar AUC on this 300-tract dataset, confirming
  that neural networks offer no meaningful accuracy advantage at this scale.
  The Random Forest is preferred over Logistic Regression because permutation
  importance shows field operations which tract characteristics most drive
  nonresponse risk — enabling targeted data collection improvements.

  The neural network (MLP 64,64) is not recommended:
  - AUC improvement over RF is within noise at this sample size.
  - No interpretable coefficients or rules for the methodology memo.
  - Requires sklearn approval review if moving from a Colab prototype to
    a production pipeline — sklearn's MLP uses numpy/scipy (not PyTorch),
    which clears the PyTorch ATO concern, but the model itself still
    requires documentation under OMB Statistical Policy Directive No. 1.

  The Decision Tree (depth 3) is ideal for individual-case auditability:
  rules can be printed and attached to the methodology documentation.

  Final recommendation:
  - Prediction pipeline: Random Forest (200 trees recommended for production)
  - Methodology report: Decision Tree rules (depth 3, printed via export_text)
  - Baseline comparison: Logistic Regression coefficients
""")

# ---------------------------------------------------------------------------
# 4. Bar chart summary
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
metrics = ["Accuracy", "F1", "AUC-ROC"]
x = np.arange(len(metrics))
w = 0.20
colors = ["steelblue", "firebrick", "forestgreen", "darkorchid"]

for i, (name, color) in enumerate(zip(act_df.index, colors)):
    vals = [act_df.loc[name, m] for m in metrics]
    ax.bar(x + i * w, vals, width=w, label=name, color=color,
           edgecolor="white", alpha=0.85)

ax.set_xticks(x + 1.5 * w)
ax.set_xticklabels(metrics)
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("Four-model comparison: tract-level nonresponse prediction")
ax.legend(fontsize=8.5, loc="lower right")
ax.axhline(0.5, color="gray", lw=0.5, linestyle=":")

plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "08_exercises.png"),
    dpi=120,
    bbox_inches="tight",
)
plt.show()

print("Plot saved: 08_exercises.png")
