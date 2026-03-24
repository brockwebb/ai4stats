"""
Chapter 2 -- Dataset Setup
Reproduces the synthetic ACS-like survey dataset from Chapter 1 and adds
a household_id column for GroupKFold demonstrations.

The dataset simulates 1,200 survey respondents with characteristics drawn
from distributions roughly matching American Community Survey patterns.
household_id clusters every 4 consecutive records into the same household,
giving 300 distinct households for use in grouped cross-validation.

Output: data/synthetic_survey_ch02.csv

Run: python 01_dataset_setup.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N = 1200
HOUSEHOLDS = N // 4        # 300 four-person households
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_survey_ch02.csv"

STATE_NAMES = ["California", "Texas", "New York", "Florida", "Illinois"]
STATE_PROBS = [0.20, 0.20, 0.18, 0.17, 0.25]

EDUCATION_LEVELS = [9, 12, 14, 16, 18]       # years of schooling
EDUCATION_PROBS  = [0.10, 0.35, 0.20, 0.25, 0.10]

# Income model parameters (log-linear)
INCOME_INTERCEPT      = 9.2
INCOME_EDUC_COEF      = 0.04    # per year above 12
INCOME_AGE_COEF       = 0.008
INCOME_HRS_COEF       = 0.003
INCOME_NOISE_STD      = 0.35
INCOME_MIN            = 10_000
INCOME_MAX            = 250_000

# Nonresponse logit parameters
NR_INTERCEPT          = -0.5
NR_CONTACT_COEF       = 0.25
NR_PRIOR_COEF         = -1.2
NR_URBAN_COEF         = -0.3
NR_AGE_COEF           = 0.01
NR_NOISE_STD          = 0.3


def build_dataset(seed: int = SEED, n: int = N) -> pd.DataFrame:
    """
    Generate the synthetic ACS-like survey dataset.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n : int
        Number of records.

    Returns
    -------
    pd.DataFrame
        Survey dataset with demographic features, income target, and
        binary response indicator.
    """
    rng = np.random.default_rng(seed)

    states            = rng.choice(STATE_NAMES, size=n, p=STATE_PROBS)
    age               = rng.normal(42, 14, n).clip(18, 80).astype(int)
    education_years   = rng.choice(EDUCATION_LEVELS, size=n, p=EDUCATION_PROBS)
    hours_per_week    = rng.normal(38, 10, n).clip(0, 80).astype(int)
    urban             = rng.binomial(1, 0.72, n)
    contact_attempts  = rng.poisson(2, n).clip(1, 7)
    prior_response    = rng.binomial(1, 0.68, n)

    # Household clustering: 4 consecutive records share the same household_id.
    # This mimics household surveys where multiple family members appear in the
    # same sample frame.
    household_id = np.repeat(np.arange(n // 4), 4)[:n]

    # Income target (log-linear model)
    log_income = (
        INCOME_INTERCEPT
        + INCOME_EDUC_COEF * (education_years - 12)
        + INCOME_AGE_COEF  * age
        + INCOME_HRS_COEF  * hours_per_week
        + rng.normal(0, INCOME_NOISE_STD, n)
    )
    income = np.exp(log_income).clip(INCOME_MIN, INCOME_MAX).astype(int)

    # Binary response indicator (logit model)
    logit_nr = (
        NR_INTERCEPT
        + NR_CONTACT_COEF * contact_attempts
        + NR_PRIOR_COEF   * prior_response
        + NR_URBAN_COEF   * urban
        + NR_AGE_COEF     * (age - 42)
        + rng.normal(0, NR_NOISE_STD, n)
    )
    prob_nonresponse = 1 / (1 + np.exp(-logit_nr))
    responded = (rng.uniform(size=n) > prob_nonresponse).astype(int)

    return pd.DataFrame({
        "state":            states,
        "age":              age,
        "education_years":  education_years,
        "hours_per_week":   hours_per_week,
        "urban":            urban,
        "contact_attempts": contact_attempts,
        "prior_response":   prior_response,
        "household_id":     household_id,
        "income":           income,
        "responded":        responded,
    })


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataset()

    print(f"Dataset shape:      {df.shape}")
    print(f"Response rate:      {df['responded'].mean():.1%}")
    print(f"Unique households:  {df['household_id'].nunique()}")
    print(f"Income range:       ${df['income'].min():,} -- ${df['income'].max():,}")
    print(f"Median income:      ${df['income'].median():,.0f}")
    print()
    print(df.head(8).to_string(index=False))

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
