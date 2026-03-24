"""
Chapter 1 -- Classification: Predicting Survey Nonresponse

Loads the synthetic ACS-like survey data and fits a logistic regression model
to predict whether a sampled person responded (responded=1) or did not (0).

What this script demonstrates:
- Stratified train/test split: preserves the response rate in both halves,
  preventing misleadingly easy or hard evaluation sets
- Accuracy, precision, recall, F1, and AUC as classification metrics and
  why each matters differently in a survey context
- Confusion matrix: which kinds of errors are most costly for field operations
- ROC curve: threshold-independent view of the precision-recall trade-off
- Threshold sensitivity: lowering the threshold catches more nonrespondents
  but also generates more unnecessary follow-up contacts
- Odds ratios: translating log-odds coefficients into an interpretable scale

Prerequisites: run 01_generate_survey_data.py first.

Usage:
    python 03_classification_nonresponse.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = "synthetic_acs_survey.csv"
FEATURES = ["age", "education_years", "urban", "contact_attempts", "prior_response"]
TARGET = "responded"
TEST_SIZE = 0.20
RANDOM_STATE = 42
CLASS_LABELS = ["Did not respond", "Responded"]


def load_data(path):
    """Load the synthetic survey CSV and return feature matrix and target."""
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET]
    return df, X, y


def print_metrics(label, y_true, y_pred, y_proba=None):
    """Print classification metrics. AUC requires y_proba."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"  {label}")
    print(f"    Accuracy:  {acc:.3f}")
    print(f"    Precision: {prec:.3f}  (of flagged as responding, fraction that did)")
    print(f"    Recall:    {rec:.3f}  (of true responders, fraction correctly flagged)")
    print(f"    F1:        {f1:.3f}")
    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"    AUC:       {auc:.3f}  (threshold-independent ranking quality)")


def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion matrix"):
    """Plot a labeled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=15)
    ax.set_yticklabels(labels)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, title="ROC curve"):
    """Plot the ROC curve with AUC annotation."""
    auc = roc_auc_score(y_true, y_proba)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def threshold_sensitivity(y_true, y_proba, thresholds):
    """
    Compute accuracy, precision, recall, F1 at each threshold.

    Returns a DataFrame with one row per threshold.
    """
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def plot_threshold_sensitivity(df_thresh):
    """Line chart of classification metrics across thresholds."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in ["accuracy", "precision", "recall", "f1"]:
        ax.plot(df_thresh["threshold"], df_thresh[col], marker="o", label=col)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics at different classification thresholds")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def odds_ratio_table(model, feature_names):
    """Return a DataFrame of coefficients and odds ratios."""
    return pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_[0],
        "odds_ratio": np.exp(model.coef_[0]),
    }).sort_values("coefficient", ascending=False)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load data and split (stratified to preserve response rate)
    # ------------------------------------------------------------------
    df, X, y = load_data(CSV_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"Train response rate: {y_train.mean():.2%}")
    print(f"Test response rate:  {y_test.mean():.2%}\n")

    # ------------------------------------------------------------------
    # 2. Fit logistic regression
    # ------------------------------------------------------------------
    clf = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # 3. Evaluation metrics
    # ------------------------------------------------------------------
    print("Logistic regression at default threshold (0.5):")
    print_metrics("Default threshold", y_test, y_pred, y_proba)

    # ------------------------------------------------------------------
    # 4. Confusion matrix and ROC curve
    # ------------------------------------------------------------------
    plot_confusion_matrix(y_test, y_pred, CLASS_LABELS)
    plt.savefig("fig_confusion_matrix.png", dpi=120, bbox_inches="tight")

    plot_roc_curve(y_test, y_proba)
    plt.savefig("fig_roc_curve.png", dpi=120, bbox_inches="tight")

    # ------------------------------------------------------------------
    # 5. Threshold sensitivity
    # ------------------------------------------------------------------
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    df_thresh = threshold_sensitivity(y_test, y_proba, thresholds)
    print("\nThreshold sensitivity:")
    print(df_thresh.to_string(index=False))

    plot_threshold_sensitivity(df_thresh)
    plt.savefig("fig_threshold_sensitivity.png", dpi=120, bbox_inches="tight")

    # ------------------------------------------------------------------
    # 6. Odds ratios
    # ------------------------------------------------------------------
    or_df = odds_ratio_table(clf, FEATURES)
    print("\nCoefficients and odds ratios (positive coefficient = more likely to respond):")
    print(or_df.to_string(index=False))

    plt.show()
    print("\nDone. Figures saved: fig_confusion_matrix.png, fig_roc_curve.png,")
    print("  fig_threshold_sensitivity.png")
