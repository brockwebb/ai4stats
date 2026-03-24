"""
05_utility_regression.py
========================
Chapter 9: Synthetic Data Generation for Federal Statistics

Tests analytic validity by fitting the same regression model on both the
confidential and synthetic datasets and comparing coefficient estimates.

Why this matters:
    The most important utility question for a specific analysis is: "If I run
    my regression on synthetic data, do I recover approximately the same
    coefficients as I would on the confidential data?" This is called analytic
    validity or regression utility. It is more demanding than marginal or even
    bivariate utility — a synthetic dataset can match correlation matrices but
    still produce misleading regression coefficients if the multivariate
    structure is off.

    Federal statistical agencies increasingly publish analytic validity results
    alongside synthetic data products to document which analyses are supported.

Model:
    income ~ age + educ + region (OLS)

Usage:
    python 05_utility_regression.py
    (Requires confidential_microdata.csv and synthetic_data.csv)

Outputs:
    - Coefficient recovery table printed to stdout

Requirements:
    Python 3.9+, numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import os


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load confidential and synthetic datasets from CSV files."""
    for path in ["confidential_microdata.csv", "synthetic_data.csv"]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run 01_confidential_dataset.py and "
                "02_sequential_synthesis.py first."
            )
    return pd.read_csv("confidential_microdata.csv"), pd.read_csv("synthetic_data.csv")


def encode_region(df: pd.DataFrame, le: LabelEncoder = None) -> tuple[np.ndarray, LabelEncoder]:
    """Encode region as integer for use in OLS. Returns encoded array and fitted encoder."""
    if le is None:
        le = LabelEncoder()
        encoded = le.fit_transform(df["region"])
    else:
        encoded = le.transform(df["region"])
    return encoded, le


def fit_income_model(df: pd.DataFrame, region_le: LabelEncoder) -> dict:
    """
    Fit OLS: income ~ age + educ + region.

    Returns a dict with intercept, coefficients, and R-squared.
    """
    region_enc, _ = encode_region(df, region_le)
    X = np.column_stack([df["age"].values, df["educ"].values, region_enc])
    y = df["income"].values
    model = LinearRegression()
    model.fit(X, y)
    return {
        "intercept": model.intercept_,
        "age_coef":    model.coef_[0],
        "educ_coef":   model.coef_[1],
        "region_coef": model.coef_[2],
        "r_squared":   model.score(X, y),
    }


def print_coefficient_table(conf_results: dict, synth_results: dict) -> None:
    """Print coefficient recovery comparison table."""
    print("Analytic validity: regression coefficient recovery")
    print("Model: income ~ age + educ + region (OLS)")
    print("=" * 75)
    print(f"{'Parameter':<18} {'Confidential':>14} {'Synthetic':>14} {'Abs Diff':>10} {'% Diff':>8} {'Status':>8}")
    print("-" * 75)

    params = [
        ("Intercept ($)",   "intercept"),
        ("Age ($/yr)",      "age_coef"),
        ("Education ($/yr)", "educ_coef"),
        ("Region",          "region_coef"),
    ]

    for label, key in params:
        cv = conf_results[key]
        sv = synth_results[key]
        abs_diff = abs(sv - cv)
        pct_diff = abs_diff / abs(cv) * 100 if cv != 0 else float("nan")
        status = "Good" if pct_diff < 15 else "Check"
        print(f"{label:<18} {cv:>14,.1f} {sv:>14,.1f} {abs_diff:>10,.1f} {pct_diff:>7.1f}% {status:>8}")

    print("-" * 75)
    print(f"{'R-squared':<18} {conf_results['r_squared']:>14.3f} {synth_results['r_squared']:>14.3f}")

    print()
    print("Interpretation:")
    print("  'Good'  = coefficient within 15% of confidential value.")
    print("  'Check' = deviation exceeds 15%; analyses using this coefficient may be biased.")
    print()
    print("Sequential synthesis was designed to preserve income ~ age + educ + region.")
    print("Coefficients should be well-recovered because these variables were modeled.")


if __name__ == "__main__":
    df_conf, df_synth = load_datasets()

    # Fit shared region encoder on confidential data so both models use the same encoding
    le = LabelEncoder()
    le.fit(df_conf["region"])

    conf_results = fit_income_model(df_conf, le)
    synth_results = fit_income_model(df_synth, le)

    print_coefficient_table(conf_results, synth_results)
