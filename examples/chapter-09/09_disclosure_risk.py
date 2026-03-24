"""
09_disclosure_risk.py
=====================
Chapter 9: Synthetic Data Generation for Federal Statistics

Assesses disclosure risk for the synthetic dataset by checking two types
of disclosure:

1. Identity disclosure: what fraction of synthetic records are "too close"
   to a specific confidential record on quasi-identifying variables?

2. Attribute disclosure: can an adversary infer a sensitive attribute
   (income) from the combination of quasi-identifiers (age, educ, region)?

Why this matters:
    Utility metrics tell you whether the synthetic data is useful. Disclosure
    risk metrics tell you whether it is safe. Both are required before a
    federal agency releases a synthetic dataset. A synthesis that passes all
    utility checks can still present unacceptable disclosure risk if it
    reproduces rare combinations of characteristics.

    This is a simplified illustration. Formal disclosure risk assessment
    at agencies involves more sophisticated methods, independent review, and
    regulatory sign-off.

Usage:
    python 09_disclosure_risk.py
    (Requires confidential_microdata.csv and synthetic_data.csv)

Outputs:
    - Identity disclosure rates printed to stdout
    - Attribute disclosure analysis printed to stdout

Requirements:
    Python 3.9+, numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
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


def identity_disclosure_rate(
    df_conf: pd.DataFrame,
    df_synth: pd.DataFrame,
    quasi_identifiers: list[str],
    n_sample: int = 200,
    random_state: int = 2025,
) -> dict:
    """
    Estimate identity disclosure rate by exact-match on quasi-identifiers.

    For each sampled synthetic record, check whether any confidential record
    has exactly the same values on all quasi-identifier columns. A match does
    not mean the synthetic record IS that real person, but it means the
    combination is not unique — a potential re-identification risk if the
    adversary knows the real person's quasi-identifiers.

    Parameters
    ----------
    quasi_identifiers : list of str
        Column names to use as quasi-identifiers.
    n_sample : int
        Number of synthetic records to check (full dataset can be slow).

    Returns
    -------
    dict with match_rate, n_matches, n_checked
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(df_synth), size=min(n_sample, len(df_synth)), replace=False)
    synth_sample = df_synth.iloc[idx].reset_index(drop=True)

    n_matches = 0
    for _, synth_row in synth_sample.iterrows():
        mask = pd.Series([True] * len(df_conf))
        for col in quasi_identifiers:
            mask = mask & (df_conf[col] == synth_row[col])
        if mask.sum() >= 1:
            n_matches += 1

    return {
        "quasi_identifiers": quasi_identifiers,
        "n_checked": len(synth_sample),
        "n_matches": n_matches,
        "match_rate": n_matches / len(synth_sample),
    }


def attribute_disclosure_analysis(
    df_conf: pd.DataFrame,
    df_synth: pd.DataFrame,
    quasi_identifiers: list[str],
    sensitive_var: str,
    n_sample: int = 200,
    tolerance_frac: float = 0.10,
    random_state: int = 2025,
) -> dict:
    """
    Assess attribute disclosure: can an adversary infer a sensitive variable
    from quasi-identifiers using the synthetic dataset?

    Procedure:
        1. For each sampled confidential record, find synthetic records that
           match on quasi-identifiers.
        2. Compute the mean sensitive value for those matching synthetic records.
        3. Check whether that estimate is within tolerance of the true value.

    This simulates an adversary who knows a target's quasi-identifiers and
    uses the synthetic data to estimate their income.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(df_conf), size=min(n_sample, len(df_conf)), replace=False)
    conf_sample = df_conf.iloc[idx].reset_index(drop=True)

    n_accurate = 0
    n_matchable = 0
    errors = []

    for _, conf_row in conf_sample.iterrows():
        mask = pd.Series([True] * len(df_synth))
        for col in quasi_identifiers:
            mask = mask & (df_synth[col] == conf_row[col])
        matching = df_synth[mask]

        if len(matching) == 0:
            continue

        n_matchable += 1
        estimated = matching[sensitive_var].mean()
        true_val = conf_row[sensitive_var]
        rel_error = abs(estimated - true_val) / max(abs(true_val), 1)
        errors.append(rel_error)

        if rel_error <= tolerance_frac:
            n_accurate += 1

    return {
        "n_checked": len(conf_sample),
        "n_matchable": n_matchable,
        "n_accurate": n_accurate,
        "accuracy_rate": n_accurate / n_matchable if n_matchable > 0 else float("nan"),
        "mean_relative_error": float(np.mean(errors)) if errors else float("nan"),
        "tolerance_frac": tolerance_frac,
    }


if __name__ == "__main__":
    df_conf, df_synth = load_datasets()
    print(f"Loaded: confidential n={len(df_conf)}, synthetic n={len(df_synth)}")
    print()

    # Identity disclosure: age + educ + region
    print("Identity disclosure assessment")
    print("=" * 55)
    for qi_set in [
        ["age", "educ", "region"],
        ["age", "educ", "region", "married"],
    ]:
        result = identity_disclosure_rate(df_conf, df_synth, qi_set, n_sample=200)
        print(f"Quasi-identifiers: {', '.join(qi_set)}")
        print(f"  Synthetic records checked: {result['n_checked']}")
        print(f"  Matched to a confidential record: {result['n_matches']}")
        print(f"  Match rate: {result['match_rate']:.1%}")
        print()

    print("Interpretation:")
    print("  Matching on demographic variables alone is expected — many people")
    print("  share the same age, education, and region. What matters is whether")
    print("  the combination identifies a UNIQUE individual. Check rare cells.")
    print()

    # Attribute disclosure: can income be inferred from quasi-identifiers?
    print("Attribute disclosure assessment (inferring income from demographics)")
    print("=" * 65)
    attr_result = attribute_disclosure_analysis(
        df_conf, df_synth,
        quasi_identifiers=["region", "educ"],
        sensitive_var="income",
        n_sample=200,
        tolerance_frac=0.10,
    )
    print(f"Quasi-identifiers: region + educ")
    print(f"Sensitive variable: income")
    print(f"Tolerance: within {attr_result['tolerance_frac']:.0%} of true value")
    print(f"Confidential records sampled: {attr_result['n_checked']}")
    print(f"Records with matching synthetic rows: {attr_result['n_matchable']}")
    print(f"Accurate estimates (within tolerance): {attr_result['n_accurate']}")
    print(f"Accuracy rate: {attr_result['accuracy_rate']:.1%}")
    print(f"Mean relative error: {attr_result['mean_relative_error']:.3f}")
    print()
    print("Interpretation:")
    print("  High accuracy rate on broad quasi-identifiers reflects the known")
    print("  income-education-region relationship, not synthesis failure.")
    print("  The risk question is: does the synthetic data give an adversary")
    print("  more information about a SPECIFIC person than they already had?")
    print("  Formal disclosure risk assessment compares synthetic to suppressed")
    print("  public-use microdata to answer that question rigorously.")
