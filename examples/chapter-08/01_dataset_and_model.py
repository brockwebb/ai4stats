"""
01_dataset_and_model.py
=======================
Creates a synthetic ACS-like nonresponse dataset and fits a logistic regression
model to predict nonresponse propensity.

WHY THIS APPROACH:
    Real ACS PUMS data cannot ship with this repository, but the educational
    point -- that nonresponse is differentially distributed across demographic
    groups -- holds in any realistic synthetic dataset. The nonresponse
    probability is constructed so that race/ethnicity, income, age, education,
    and region all contribute, replicating the structural patterns documented
    in Census Bureau operations research.

    Saving test predictions and group labels lets downstream scripts (02-09)
    run independently without re-fitting the model each time.

OUTPUTS:
    - Prints dataset summary and per-group nonresponse rates
    - Saves test set predictions to ch08_test_predictions.csv (same directory)

REQUIREMENTS:
    Python 3.9+, numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants -- all tunable parameters live here, not buried in logic
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_RECORDS = 2000
TEST_SIZE = 0.30
LOGISTIC_MAX_ITER = 500

RACE_GROUPS = [
    "White non-Hispanic",
    "Black non-Hispanic",
    "Hispanic",
    "Asian non-Hispanic",
    "Other",
]
RACE_PROBS = [0.60, 0.13, 0.18, 0.06, 0.03]

# Median income by group, approximating ACS PUMS patterns
INCOME_BASE = {
    "White non-Hispanic": 65_000,
    "Black non-Hispanic": 43_000,
    "Hispanic": 40_000,
    "Asian non-Hispanic": 78_000,
    "Other": 48_000,
}

# Education distribution by group, approximating ACS PUMS educational attainment
EDUCATION_LEVELS = ["Less than high school", "High school", "Some college", "Bachelor+"]
EDU_PROBS_BY_RACE = {
    "White non-Hispanic": [0.06, 0.27, 0.30, 0.37],
    "Black non-Hispanic": [0.11, 0.30, 0.30, 0.29],
    "Hispanic":           [0.24, 0.28, 0.28, 0.20],
    "Asian non-Hispanic": [0.04, 0.15, 0.20, 0.61],
    "Other":              [0.12, 0.28, 0.30, 0.30],
}

# Nonresponse logit coefficients -- these control fairness differences in the model
# The positive coefficients for Black and Hispanic groups reflect documented patterns
# of differential nonresponse in federal surveys (see Census operational research)
NONRESPONSE_INTERCEPT = -1.5
NONRESPONSE_COEFS = {
    "Black non-Hispanic": 0.8,
    "Hispanic": 0.9,
    "Other": 0.3,
}
INCOME_COEF = -0.015          # higher income -> lower nonresponse
AGE_SENIOR_COEF = 0.008       # older adults less likely to respond online
EDUCATION_LOWHIGH_COEF = 0.4  # less than HS increases nonresponse
SOUTH_COEF = 0.5              # regional effect documented in NRFU research
NOISE_STD = 0.5


def build_dataset(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Construct synthetic ACS-like dataset with nonresponse labels."""
    rng = np.random.default_rng(seed)

    race = rng.choice(RACE_GROUPS, size=N_RECORDS, p=RACE_PROBS)
    age = rng.normal(42, 14, N_RECORDS).clip(18, 80).astype(int)

    income = np.array([
        rng.lognormal(
            np.log(INCOME_BASE[r] * rng.uniform(0.85, 1.15)),
            0.4,
        )
        for r in race
    ]).clip(10_000, 250_000)

    education = np.array([
        rng.choice(EDUCATION_LEVELS, p=EDU_PROBS_BY_RACE[r])
        for r in race
    ])

    hours_per_week = rng.integers(0, 80, N_RECORDS)
    urban = rng.choice([0, 1], size=N_RECORDS, p=[0.20, 0.80])
    contact_attempts = rng.integers(1, 6, N_RECORDS)
    prior_response = rng.choice([0, 1], size=N_RECORDS, p=[0.30, 0.70])
    region = rng.choice(
        ["Northeast", "Midwest", "South", "West"],
        size=N_RECORDS,
        p=[0.18, 0.21, 0.38, 0.23],
    )

    # Build nonresponse probability from known structural factors
    logit = np.full(N_RECORDS, NONRESPONSE_INTERCEPT, dtype=float)
    for group, coef in NONRESPONSE_COEFS.items():
        logit += coef * (race == group).astype(float)
    logit += INCOME_COEF * (income / 1_000)
    logit += AGE_SENIOR_COEF * np.maximum(0, 65 - age)
    logit += EDUCATION_LOWHIGH_COEF * (education == "Less than high school").astype(float)
    logit += SOUTH_COEF * (region == "South").astype(float)
    logit += rng.normal(0, NOISE_STD, N_RECORDS)

    nonresponse_prob = 1 / (1 + np.exp(-logit))
    nonresponse = (rng.uniform(0, 1, N_RECORDS) < nonresponse_prob).astype(int)

    df = pd.DataFrame({
        "race": race,
        "age": age,
        "income": income,
        "education": education,
        "hours_per_week": hours_per_week,
        "urban": urban,
        "contact_attempts": contact_attempts,
        "prior_response": prior_response,
        "region": region,
        "nonresponse": nonresponse,
        "income_quintile": pd.qcut(income, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]),
    })
    return df


def fit_model(df: pd.DataFrame) -> dict:
    """
    Encode features, split train/test, fit logistic regression.

    Returns a dict with model, test indices, predictions, and aligned DataFrame.
    """
    df_model = df.copy()

    le_race = LabelEncoder()
    le_edu = LabelEncoder()
    le_region = LabelEncoder()

    df_model["race_enc"] = le_race.fit_transform(df_model["race"])
    df_model["edu_enc"] = le_edu.fit_transform(df_model["education"])
    df_model["region_enc"] = le_region.fit_transform(df_model["region"])
    df_model["income_log"] = np.log1p(df_model["income"])

    features = ["age", "income_log", "race_enc", "edu_enc", "region_enc"]
    X = df_model[features].values
    y = df_model["nonresponse"].values

    indices = np.arange(len(df_model))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    df_test = df_model.iloc[test_idx].copy().reset_index(drop=True)
    df_test["y_true"] = y_test
    df_test["y_pred"] = y_pred
    df_test["y_prob"] = y_prob

    return {
        "model": clf,
        "df_test": df_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def print_summary(df: pd.DataFrame, results: dict) -> None:
    """Print dataset summary and model performance."""
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    y_prob = results["y_prob"]

    print("Chapter 8: Bias, Fairness, and Equity in Federal AI/ML")
    print("=" * 55)
    print()
    print(f"Synthetic ACS-like dataset: {N_RECORDS:,} records")
    print(f"Overall nonresponse rate:   {df['nonresponse'].mean():.1%}")
    print()
    print("Nonresponse rate by racial/ethnic group:")
    for group in RACE_GROUPS:
        mask = df["race"] == group
        rate = df.loc[mask, "nonresponse"].mean()
        count = mask.sum()
        print(f"  {group:<28}: {rate:.1%}  (n={count})")

    print()
    print("Logistic regression nonresponse model (test set):")
    print(f"  Overall accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Overall precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"  Overall recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"  Overall AUC-ROC:   {roc_auc_score(y_test, y_prob):.3f}")
    print()
    print("Overall metrics look acceptable. The key question is how they")
    print("decompose by subgroup -- see scripts 04 and 06.")


def save_test_predictions(df_test: pd.DataFrame, out_dir: Path) -> None:
    """Save test-set predictions so other scripts can load them without re-fitting."""
    out_path = out_dir / "ch08_test_predictions.csv"
    df_test.to_csv(out_path, index=False)
    print(f"\nTest predictions saved: {out_path}")


if __name__ == "__main__":
    here = Path(__file__).parent

    df = build_dataset()
    results = fit_model(df)
    print_summary(df, results)
    save_test_predictions(results["df_test"], here)
