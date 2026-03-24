"""
Chapter 1 -- Generate Synthetic ACS-Like Survey Data

Creates a synthetic dataset mimicking an ACS person record file.
Features include state, age, education, hours worked, urban/rural status,
contact attempt history, prior response history, income, and a binary
responded flag.

Income is generated from a log-normal distribution with education, age, and
hours worked as predictors. Nonresponse probability is modeled as a logistic
function of contact attempts, prior response, urban status, and age.

Run this first before the other chapter-01 scripts:
    python 01_generate_survey_data.py

Output: synthetic_acs_survey.csv in this directory

The dataset is reproducible (seed=42) and contains no real person records.
"""

import numpy as np
import pandas as pd


def generate_survey_data(n=1200, seed=42):
    """
    Generate a synthetic ACS-like survey person record file.

    Parameters
    ----------
    n : int
        Number of person records to generate. Default 1200.
    seed : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: state, age, education_years, hours_per_week,
        urban, contact_attempts, prior_response, income, responded.
    """
    np.random.seed(seed)

    # Geographic and demographic features
    states = np.random.choice(
        ["California", "Texas", "New York", "Florida", "Illinois"],
        size=n, p=[0.20, 0.20, 0.18, 0.17, 0.25]
    )
    age = np.random.normal(42, 14, n).clip(18, 80).astype(int)
    education_years = np.random.choice(
        [9, 12, 14, 16, 18], size=n, p=[0.10, 0.35, 0.20, 0.25, 0.10]
    )
    hours_per_week = np.random.normal(38, 10, n).clip(0, 80).astype(int)
    urban = np.random.binomial(1, 0.72, n)            # 1 = urban tract
    contact_attempts = np.random.poisson(2, n).clip(1, 7)
    prior_response = np.random.binomial(1, 0.68, n)   # responded in prior cycle

    # Income: log-normal to capture right skew typical of wage distributions.
    # Each additional year of education adds ~4% to log-income; age and hours
    # contribute smaller increments. Residual variance (~0.35) reflects unobserved
    # factors (occupation, employer, region).
    log_income = (
        9.2
        + 0.04 * (education_years - 12)
        + 0.008 * age
        + 0.003 * hours_per_week
        + np.random.normal(0, 0.35, n)
    )
    income = np.exp(log_income).clip(10_000, 250_000).astype(int)

    # Nonresponse: higher contact attempts and absent prior response => harder
    # to reach. The logistic function converts log-odds to a probability.
    logit_nr = (
        -0.5
        + 0.25 * contact_attempts
        - 1.2 * prior_response
        - 0.3 * urban
        + 0.01 * (age - 42)
        + np.random.normal(0, 0.3, n)
    )
    responded = (np.random.uniform(size=n) > 1 / (1 + np.exp(-logit_nr))).astype(int)

    return pd.DataFrame({
        "state": states,
        "age": age,
        "education_years": education_years,
        "hours_per_week": hours_per_week,
        "urban": urban,
        "contact_attempts": contact_attempts,
        "prior_response": prior_response,
        "income": income,
        "responded": responded,
    })


if __name__ == "__main__":
    df = generate_survey_data()
    df.to_csv("synthetic_acs_survey.csv", index=False)
    print(f"Saved {len(df)} records to synthetic_acs_survey.csv")
    print(f"Response rate: {df['responded'].mean():.1%}")
    print(f"Nonresponse rate: {1 - df['responded'].mean():.1%}")
    print("\nSummary statistics:")
    print(df.describe().round(1))
    print("\nResponse rate by state:")
    print(df.groupby("state")["responded"].mean().round(3))
