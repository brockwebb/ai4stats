"""
Chapter 3 — Dataset Setup
=========================
Reproduces the synthetic survey dataset used throughout Chapter 3
(Decision Trees and Random Forests). The same seed and parameters
are used in all subsequent scripts so results are directly comparable.

Dataset design
--------------
- n=1200 synthetic respondents with state, age, education, hours worked,
  urban indicator, contact attempts, and prior response history.
- Income is log-normally distributed with realistic predictors.
- The binary 'responded' outcome is generated from a logistic model,
  making prior_response and contact_attempts the dominant drivers —
  matching what a federal nonresponse follow-up analyst would expect.

Outputs
-------
Prints dataset shape, train/test sizes, and response rate to stdout.
No files are written; other scripts import the setup logic directly.

Requirements
------------
Python 3.9+, numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Seed and sample size — must match all other chapter-03 scripts
# ---------------------------------------------------------------------------
SEED = 42
N = 1200

# ---------------------------------------------------------------------------
# Feature columns used for each task
# ---------------------------------------------------------------------------
FEATURES_CLF = ["age", "education_years", "urban", "contact_attempts", "prior_response"]
FEATURES_REG = ["age", "education_years", "hours_per_week", "urban"]


def build_dataset(seed: int = SEED, n: int = N) -> pd.DataFrame:
    """
    Generate the synthetic survey dataset.

    Parameters
    ----------
    seed : int
        NumPy random seed for reproducibility.
    n : int
        Number of synthetic survey respondents.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: state, age, education_years,
        hours_per_week, urban, contact_attempts, prior_response,
        income, responded.
    """
    rng = np.random.default_rng(seed)

    states = rng.choice(
        ["California", "Texas", "New York", "Florida", "Illinois"],
        size=n,
        p=[0.20, 0.20, 0.18, 0.17, 0.25],
    )
    age = rng.normal(42, 14, n).clip(18, 80).astype(int)
    education_years = rng.choice(
        [9, 12, 14, 16, 18], size=n, p=[0.10, 0.35, 0.20, 0.25, 0.10]
    )
    hours_per_week = rng.normal(38, 10, n).clip(0, 80).astype(int)
    urban = rng.binomial(1, 0.72, n)
    contact_attempts = rng.poisson(2, n).clip(1, 7)
    prior_response = rng.binomial(1, 0.68, n)

    log_income = (
        9.2
        + 0.04 * (education_years - 12)
        + 0.008 * age
        + 0.003 * hours_per_week
        + rng.normal(0, 0.35, n)
    )
    income = np.exp(log_income).clip(10_000, 250_000).astype(int)

    logit_nr = (
        -0.5
        + 0.25 * contact_attempts
        - 1.2 * prior_response
        - 0.3 * urban
        + 0.01 * (age - 42)
        + rng.normal(0, 0.3, n)
    )
    prob_nonresponse = 1 / (1 + np.exp(-logit_nr))
    responded = (rng.uniform(size=n) > prob_nonresponse).astype(int)

    return pd.DataFrame(
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


def make_splits(df: pd.DataFrame):
    """
    Create train/test splits for both the classification and regression tasks.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_dataset().

    Returns
    -------
    tuple
        (X_clf_train, X_clf_test, y_clf_train, y_clf_test,
         X_reg_train, X_reg_test, y_reg_train, y_reg_test)
    """
    X_clf = df[FEATURES_CLF]
    y_clf = df["responded"]
    X_reg = df[FEATURES_REG]
    y_reg = df["income"]

    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.20, random_state=SEED, stratify=y_clf
    )
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=SEED
    )

    return (
        X_clf_train, X_clf_test, y_clf_train, y_clf_test,
        X_reg_train, X_reg_test, y_reg_train, y_reg_test,
    )


# ---------------------------------------------------------------------------
# Main — run standalone to confirm setup
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # NOTE: The original chapter uses np.random.seed() (legacy API).
    # We replicate that here so numeric outputs match the book exactly.
    np.random.seed(SEED)
    n = N

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

    (
        X_clf_train, X_clf_test, y_clf_train, y_clf_test,
        X_reg_train, X_reg_test, y_reg_train, y_reg_test,
    ) = make_splits(df)

    print(f"Dataset shape:            {df.shape}")
    print(f"Response rate:            {df['responded'].mean():.1%}")
    print(f"Classification train:     {X_clf_train.shape}")
    print(f"Classification test:      {X_clf_test.shape}")
    print(f"Regression train:         {X_reg_train.shape}")
    print(f"Regression test:          {X_reg_test.shape}")
    print()
    print(df.head())
