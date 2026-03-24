"""
Chapter 1 -- Exercises and Solutions

Skeleton versions of the in-chapter exercises appear first (clearly marked
with TODO). Full solutions follow in the __main__ block below.

Work through the skeleton versions before looking at the solutions. Each
exercise is self-contained: the only prerequisite is that
01_generate_survey_data.py has been run and synthetic_acs_survey.csv
exists in this directory.

Exercises:
  1. Predict income with two features (age + hours_per_week only)
  2. Ridge regression across train/test split sizes
  3. Classify nonresponse with a non-default threshold
  4. Feature importance from logistic regression coefficients

Usage:
    python 05_exercises.py        # runs solutions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)

CSV_PATH = "synthetic_acs_survey.csv"

# ---------------------------------------------------------------------------
# EXERCISE 1: Predict income with two features
# ---------------------------------------------------------------------------
# Use only age and hours_per_week to predict income.
# Use a 90/10 split (test_size=0.10) and report MAE and R^2.
# Compare to the four-feature model: why is R^2 lower here?
#
# def exercise_1_skeleton(df):
#     X2 = df[["age", "hours_per_week"]]
#     y2 = df["income"]
#     X_tr, X_te, y_tr, y_te = train_test_split(X2, y2, test_size=..., random_state=42)   # TODO
#     model = ...   # TODO
#     model.fit(X_tr, y_tr)
#     y_hat = model.predict(X_te)
#     print(f"MAE: ${mean_absolute_error(y_te, y_hat):,.0f}")
#     print(f"R^2: {r2_score(y_te, y_hat):.3f}")


# ---------------------------------------------------------------------------
# EXERCISE 2: Ridge regression across split sizes
# ---------------------------------------------------------------------------
# Train Ridge(alpha=200) for income using all four regression features.
# Compare test R^2 for train fractions of 60%, 70%, 80%, 90% with random_state=42.
# Plot R^2 vs. train percentage.
#
# def exercise_2_skeleton(X_reg, y_reg):
#     splits = [...]    # TODO: test fractions corresponding to 60/40, 70/30, 80/20, 90/10
#     r2_scores = []
#     for t in splits:
#         X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=t, random_state=42)
#         model = Ridge(alpha=200).fit(X_tr, y_tr)
#         r2_scores.append(r2_score(y_te, model.predict(X_te)))
#     plt.figure(figsize=(6, 4))
#     plt.plot([60, 70, 80, 90], r2_scores, "o-", lw=2)
#     plt.xlabel("Train %")
#     plt.ylabel("R^2 (test)")
#     plt.title("Effect of train/test split on Ridge regression accuracy")
#     plt.grid(True)


# ---------------------------------------------------------------------------
# EXERCISE 3: Classify nonresponse at a non-default threshold
# ---------------------------------------------------------------------------
# Use a fitted LogisticRegression (features below) and set threshold=0.35.
# Print accuracy, precision, recall, and F1.
# Does recall improve compared to 0.50? What is the cost?
#
# FEATURES_CLF = ["age", "education_years", "urban", "contact_attempts", "prior_response"]
# def exercise_3_skeleton(clf, y_clf_proba, y_clf_test):
#     threshold = ...   # TODO: try 0.35
#     pred_adjusted = (y_clf_proba >= threshold).astype(int)
#     print(f"Threshold: {threshold}")
#     print(f"Accuracy:  {accuracy_score(y_clf_test, pred_adjusted):.3f}")
#     print(f"Precision: {precision_score(y_clf_test, pred_adjusted, zero_division=0):.3f}")
#     print(f"Recall:    {recall_score(y_clf_test, pred_adjusted, zero_division=0):.3f}")
#     print(f"F1:        {f1_score(y_clf_test, pred_adjusted, zero_division=0):.3f}")


# ---------------------------------------------------------------------------
# EXERCISE 4: Feature importance from logistic regression coefficients
# ---------------------------------------------------------------------------
# Using a fitted LogisticRegression, plot a horizontal bar chart of its
# coefficients from largest to smallest.
# Which feature most strongly predicts response?
# Which most strongly predicts nonresponse?
#
# def exercise_4_skeleton(clf, feature_names):
#     coef_df = pd.DataFrame({
#         "feature": ...,        # TODO: list of feature names
#         "coefficient": ...,    # TODO: clf.coef_[0]
#     }).sort_values("coefficient")
#     plt.figure(figsize=(6, 3))
#     plt.barh(coef_df["feature"], coef_df["coefficient"])
#     plt.axvline(0, color="k", lw=0.8)
#     plt.xlabel("Coefficient (log-odds scale)")
#     plt.title("Logistic regression: response predictors")
#     plt.tight_layout()


# ===========================================================================
# FULL SOLUTIONS
# ===========================================================================

def exercise_1_solution(df):
    """Two-feature income model. Lower R^2 shows cost of leaving out education."""
    print("=== Exercise 1: Two-feature income model ===")
    X2 = df[["age", "hours_per_week"]]
    y2 = df["income"]
    X_tr, X_te, y_tr, y_te = train_test_split(X2, y2, test_size=0.10, random_state=42)
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    print(f"MAE: ${mean_absolute_error(y_te, y_hat):,.0f}")
    print(f"R^2: {r2_score(y_te, y_hat):.3f}")
    print("Compare to four-feature model: education and urban carry predictive power.")
    print()


def exercise_2_solution(X_reg, y_reg):
    """Ridge R^2 across split sizes. More training data generally helps."""
    print("=== Exercise 2: Ridge across split sizes ===")
    splits = [0.40, 0.30, 0.20, 0.10]   # test fractions for 60/40, 70/30, 80/20, 90/10
    train_pcts = [60, 70, 80, 90]
    r2_scores = []
    for t in splits:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_reg, y_reg, test_size=t, random_state=42
        )
        model = Ridge(alpha=200).fit(X_tr, y_tr)
        r2_scores.append(r2_score(y_te, model.predict(X_te)))
    for pct, r2 in zip(train_pcts, r2_scores):
        print(f"  Train {pct}%: R^2 = {r2:.3f}")
    plt.figure(figsize=(6, 4))
    plt.plot(train_pcts, r2_scores, "o-", lw=2)
    plt.xlabel("Train %")
    plt.ylabel("R^2 (test)")
    plt.title("Effect of train/test split on Ridge regression accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_exercise2_ridge_splits.png", dpi=120, bbox_inches="tight")
    print("Figure saved to fig_exercise2_ridge_splits.png\n")


def exercise_3_solution(df):
    """
    Lower threshold: more nonrespondents caught (higher recall) at cost of
    more false positives (lower precision).
    """
    print("=== Exercise 3: Nonresponse at threshold 0.35 ===")
    features = ["age", "education_years", "urban", "contact_attempts", "prior_response"]
    X = df[features]
    y = df["responded"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_tr, y_tr)
    y_proba = clf.predict_proba(X_te)[:, 1]

    for threshold in [0.50, 0.35]:
        pred = (y_proba >= threshold).astype(int)
        print(f"  Threshold={threshold:.2f}  "
              f"Acc={accuracy_score(y_te, pred):.3f}  "
              f"Prec={precision_score(y_te, pred, zero_division=0):.3f}  "
              f"Recall={recall_score(y_te, pred, zero_division=0):.3f}  "
              f"F1={f1_score(y_te, pred, zero_division=0):.3f}")
    print("Recall increases at 0.35; precision decreases (more unnecessary follow-up).")
    print()


def exercise_4_solution(df):
    """Coefficient bar chart shows which features drive response probability."""
    print("=== Exercise 4: Feature importance from logistic coefficients ===")
    features = ["age", "education_years", "urban", "contact_attempts", "prior_response"]
    X = df[features]
    y = df["responded"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_tr, y_tr)

    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": clf.coef_[0],
    }).sort_values("coefficient")

    print("Coefficients (positive = increases probability of responding):")
    print(coef_df.to_string(index=False))

    plt.figure(figsize=(6, 3))
    plt.barh(coef_df["feature"], coef_df["coefficient"])
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("Coefficient (log-odds scale)")
    plt.title("Logistic regression: response predictors")
    plt.tight_layout()
    plt.savefig("fig_exercise4_coef.png", dpi=120, bbox_inches="tight")
    print("Figure saved to fig_exercise4_coef.png\n")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    REG_FEATURES = ["age", "education_years", "hours_per_week", "urban"]
    X_reg = df[REG_FEATURES]
    y_reg = df["income"]

    exercise_1_solution(df)
    exercise_2_solution(X_reg, y_reg)
    exercise_3_solution(df)
    exercise_4_solution(df)

    plt.show()
