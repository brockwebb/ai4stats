"""
Chapter 10 — Membership Inference Attack Demonstration
=======================================================

Purpose
-------
This script demonstrates that membership inference attacks are a measurable,
concrete phenomenon — not a theoretical concern. It generates a small synthetic
dataset, trains a RandomForestClassifier, and then shows that training records
can be distinguished from holdout records based on model confidence scores alone.

A membership inference attack exploits the observation that a trained model tends
to produce higher-confidence predictions for records it was trained on (because
it has "memorized" those records to some degree) compared to records it has never
seen. An adversary who can query the model and observe its output probabilities
can use this signal to test whether a specific individual was in the training set.

Why this matters for SDL reviewers
-----------------------------------
If an API exposes a model's prediction probabilities, an adversary does not need
the underlying microdata to run a membership inference attack. They need only the
ability to query the model. This script makes that signal visible on a simple
example. The same signal exists — often stronger — on complex models trained on
real survey or administrative data.

Usage
-----
    python 01_membership_inference_demo.py

Requirements: Python 3.9+, numpy, scikit-learn
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def generate_dataset(n_samples=200, n_features=5, random_state=42):
    """
    Generate a small synthetic binary classification dataset.

    Returns arrays X (features) and y (binary label). The dataset is synthetic
    and contains no real respondent data. It is structured to resemble the kind
    of tabular data a statistical agency might use to train a benefit-eligibility
    or risk-classification model.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        random_state=random_state,
    )
    return X, y


def run_membership_inference_demo():
    """
    Train a classifier and demonstrate the membership inference signal.

    Steps:
    1. Generate 200 synthetic records with 5 features and a binary label.
    2. Split into 150 training records and 50 holdout records.
    3. Train a RandomForestClassifier on the training records.
    4. Compute the maximum class probability (confidence score) for each record
       in both the training set and the holdout set.
    5. Show that training records have systematically higher confidence scores.
    6. Compute the AUC-ROC of a naive membership inference attack that uses the
       confidence score as the attack signal: records above a threshold are
       predicted to be training members, records below are predicted to be holdout.

    Key message
    -----------
    The gap between training and holdout confidence scores is the membership
    inference signal. Even a naive threshold attack achieves AUC-ROC well above
    0.5 on this simple example. On complex models trained on rare or unique
    records, the signal is often much stronger.
    """
    np.random.seed(42)

    print("=" * 60)
    print("Chapter 10 — Membership Inference Attack Demonstration")
    print("=" * 60)
    print()

    # Step 1: Generate synthetic dataset
    X, y = generate_dataset(n_samples=200, n_features=5, random_state=42)
    print(f"Dataset: {X.shape[0]} records, {X.shape[1]} features, binary label")

    # Step 2: Split into training (150) and holdout (50)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=50, train_size=150, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} records")
    print(f"Holdout set:  {X_holdout.shape[0]} records")
    print()

    # Step 3: Train a RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print("Model trained: RandomForestClassifier (100 trees)")
    print()

    # Step 4: Compute max-confidence (highest class probability) for each record
    # This is what an adversary observes when querying the model's probability output.
    train_probs = clf.predict_proba(X_train)
    holdout_probs = clf.predict_proba(X_holdout)

    # Max confidence = the probability assigned to whichever class the model favors.
    # Training records tend to receive higher max-confidence because the model has
    # seen and (partially) memorized them.
    train_max_conf = train_probs.max(axis=1)
    holdout_max_conf = holdout_probs.max(axis=1)

    # Step 5: Compare confidence distributions
    print("Membership inference signal (max prediction confidence):")
    print(f"  Mean max-confidence, training records: {train_max_conf.mean():.4f}")
    print(f"  Mean max-confidence, holdout records:  {holdout_max_conf.mean():.4f}")
    gap = train_max_conf.mean() - holdout_max_conf.mean()
    print(f"  Gap (training - holdout):              {gap:.4f}")
    print()

    # Step 6: Naive membership inference attack — use max confidence as the attack signal.
    # Label training records as 1 (member) and holdout records as 0 (non-member).
    # The attack score is the max confidence score for each record.
    # AUC-ROC measures how well this score separates members from non-members.
    attack_labels = np.concatenate([
        np.ones(len(X_train)),   # training records are "members"
        np.zeros(len(X_holdout)) # holdout records are "non-members"
    ])
    attack_scores = np.concatenate([train_max_conf, holdout_max_conf])
    attack_auc = roc_auc_score(attack_labels, attack_scores)

    print("Naive membership inference attack results:")
    print(f"  Attack signal: max prediction confidence")
    print(f"  AUC-ROC: {attack_auc:.4f}  (0.5 = no signal, 1.0 = perfect attack)")
    print()

    # Interpretation
    print("-" * 60)
    print("Interpretation")
    print("-" * 60)
    if attack_auc >= 0.70:
        signal_level = "strong"
    elif attack_auc >= 0.60:
        signal_level = "moderate"
    else:
        signal_level = "weak"

    print(
        f"This naive attack achieves AUC-ROC = {attack_auc:.4f}, "
        f"indicating a {signal_level} membership inference signal."
    )
    print()
    print(
        "This signal exists because the model assigns higher confidence to records "
        "it was trained on. An adversary with API access who can observe prediction "
        "probabilities can exploit this signal without ever seeing the training data."
    )
    print()
    print(
        "SDL implication: any API that exposes prediction probabilities from a model "
        "trained on confidential data is a potential membership inference vector. "
        "Rate limiting slows the attack; it does not eliminate the signal. Differential "
        "privacy applied during training (see Chapter 9) is the primary technical "
        "mitigation."
    )
    print()
    print(
        "Note: this demonstration uses a small, simple dataset. On high-capacity "
        "models (deep neural networks, LLMs) trained on rare or unique records, "
        "the membership inference signal is typically much stronger."
    )


if __name__ == "__main__":
    run_membership_inference_demo()
