"""
Chapter 5 Example 5: Comparison Features and Linkage Classifier
================================================================

For each candidate pair produced by blocking, computes a feature vector:
  - name_sim:  normalized string similarity (SequenceMatcher ratio)
  - dob_sim:   DOB agreement (1.0 exact, 0.5 year-only, 0.0 mismatch)
  - addr_sim:  normalized string similarity on address field

Then trains a logistic regression classifier to distinguish true matches
from non-matches, and evaluates it on a held-out test set.

Key output: classification report showing precision, recall, and F1 for
both classes. Feature coefficients show which fields drive the decision.

Plots feature distributions for matches vs. non-matches; the separation
between distributions determines how well a feature discriminates.

Requires: Run 03_synthetic_records.py first (or uses inline generation).

Requirements: numpy, pandas, matplotlib, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
_m3 = import_module("03_synthetic_records")
generate_true_records = _m3.generate_true_records
make_source_a         = _m3.make_source_a
make_source_b         = _m3.make_source_b

_m4 = import_module("04_blocking")
generate_candidate_pairs = _m4.generate_candidate_pairs


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------

def string_similarity(a, b):
    """
    Normalized string similarity using SequenceMatcher (edit-distance based).

    Returns a float in [0, 1]: 0 = no overlap, 1 = identical strings.
    NaN-safe: returns 0.0 if either input is missing.

    Parameters
    ----------
    a, b : str or float
        String values to compare (names, addresses).

    Returns
    -------
    float
    """
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def dob_match(a, b):
    """
    Partial date-of-birth agreement score.

    Returns 1.0 for exact match, 0.5 if only the year matches (useful when
    month/day transpositions are common), and 0.0 otherwise.
    NaN-safe: returns 0.0 if either input is missing.

    Parameters
    ----------
    a, b : str or float
        DOB strings in 'YYYY-MM-DD' format.

    Returns
    -------
    float
    """
    if pd.isna(a) or pd.isna(b):
        return 0.0
    if str(a) == str(b):
        return 1.0
    if str(a)[:4] == str(b)[:4]:
        return 0.5
    return 0.0


def address_similarity(a, b):
    """
    String similarity on address fields (alias for readability).

    Parameters
    ----------
    a, b : str or float

    Returns
    -------
    float
    """
    return string_similarity(a, b)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_comparison_features(source_a, source_b, candidate_pairs):
    """
    Compute a feature vector for every candidate pair.

    Parameters
    ----------
    source_a : pd.DataFrame
    source_b : pd.DataFrame
    candidate_pairs : set of (int, int)
        Index pairs from blocking step.

    Returns
    -------
    pd.DataFrame
        Columns: rec_a, rec_b, true_id_a, true_id_b, name_sim, dob_sim,
        addr_sim, is_match.
    """
    rows = []
    for ai, bi in candidate_pairs:
        row_a = source_a.loc[ai]
        row_b = source_b.loc[bi]
        rows.append({
            "rec_a":      row_a["record_id"],
            "rec_b":      row_b["record_id"],
            "true_id_a":  row_a["true_id"],
            "true_id_b":  row_b["true_id"],
            "name_sim":   string_similarity(row_a["name"],    row_b["name"]),
            "dob_sim":    dob_match(row_a["dob"],             row_b["dob"]),
            "addr_sim":   address_similarity(row_a["address"], row_b["address"]),
            "is_match":   int(row_a["true_id"] == row_b["true_id"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def train_linkage_classifier(comp_df):
    """
    Train a logistic regression classifier to predict match/non-match.

    The classifier is trained on 70% of candidate pairs and evaluated on 30%.
    Class weights are balanced to handle the heavy class imbalance (far more
    non-matches than true matches in the candidate set).

    Parameters
    ----------
    comp_df : pd.DataFrame
        Output of compute_comparison_features().

    Returns
    -------
    tuple
        (fitted LogisticRegression, fitted StandardScaler, test DataFrame,
         y_pred array, y_prob array)
    """
    feature_cols = ["name_sim", "dob_sim", "addr_sim"]
    X = comp_df[feature_cols].values
    y = comp_df["is_match"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(comp_df)),
        test_size=0.30, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=500)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    test_df = comp_df.iloc[idx_test].copy()
    test_df["y_pred"] = y_pred
    test_df["y_prob"] = y_prob

    print("Linkage classifier performance (test set):")
    print(classification_report(y_test, y_pred, target_names=["Non-match", "Match"]))

    coef_df = pd.DataFrame({
        "Feature":     feature_cols,
        "Coefficient": clf.coef_[0],
    })
    print("Feature coefficients (higher = stronger signal toward match):")
    print(coef_df.round(3).to_string(index=False))

    return clf, scaler, test_df, y_pred, y_prob


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_feature_distributions(comp_df):
    """
    Plot histograms of each comparison feature split by match/non-match label.

    Good features show clear separation between the two distributions.
    Overlapping distributions indicate a weak discriminating signal.

    Parameters
    ----------
    comp_df : pd.DataFrame
        Output of compute_comparison_features().
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    features = [
        ("name_sim",  "Name similarity"),
        ("dob_sim",   "DOB match score"),
        ("addr_sim",  "Address similarity"),
    ]
    for ax, (feat, label) in zip(axes, features):
        for match_val, color, lbl in [(0, "firebrick", "Non-match"), (1, "steelblue", "True match")]:
            vals = comp_df[comp_df["is_match"] == match_val][feat]
            ax.hist(vals, bins=20, alpha=0.6, color=color, label=lbl, density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {label}")
        ax.legend(fontsize=8)
    plt.suptitle("Comparison feature distributions: matches vs. non-matches", fontsize=11)
    plt.tight_layout()
    plt.savefig("feature_distributions.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: feature_distributions.png")


if __name__ == "__main__":
    true_df  = generate_true_records(n=200, seed=42)
    source_a = make_source_a(true_df)
    source_b = make_source_b(true_df, n_overlap=180, seed=99)

    candidate_pairs = generate_candidate_pairs(source_a, source_b)
    comp_df = compute_comparison_features(source_a, source_b, candidate_pairs)

    print(f"Candidate pairs: {len(comp_df)}")
    print(f"True matches in candidate set: {comp_df['is_match'].sum()} "
          f"({comp_df['is_match'].mean():.1%})")
    print(f"\nMean feature values by match status:")
    print(comp_df.groupby("is_match")[["name_sim", "dob_sim", "addr_sim"]].mean().round(3))
    print()

    clf, scaler, test_df, y_pred, y_prob = train_linkage_classifier(comp_df)
    plot_feature_distributions(comp_df)
