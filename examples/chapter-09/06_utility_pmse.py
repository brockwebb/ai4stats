"""
06_utility_pmse.py
==================
Chapter 9: Synthetic Data Generation for Federal Statistics

Computes the propensity score MSE (pMSE) as a formal global utility metric.
The idea: train a classifier to distinguish confidential records from synthetic
records. If the synthetic data is statistically indistinguishable, the
classifier should perform no better than random guessing.

Why this matters:
    Marginal and bivariate checks are useful but require analyst judgment on
    which variables and pairs to examine. pMSE provides a single number that
    summarizes how distinguishable the synthetic data is from the confidential
    data across all features simultaneously. It is interpretable:
        pMSE = 0.000 means the classifier cannot tell them apart (ideal)
        pMSE = 0.250 means perfect discrimination (synthesizer failed)

    pMSE is increasingly reported in federal synthetic data documentation.

Formula:
    pMSE = mean( (P(record is synthetic) - 0.5)^2 )
    measured over all records in the combined dataset.

Usage:
    python 06_utility_pmse.py
    (Requires confidential_microdata.csv and synthetic_data.csv)

Outputs:
    - pMSE value and interpretation printed to stdout
    - Classifier accuracy printed as a secondary check

Requirements:
    Python 3.9+, numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
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


def prepare_features(df: pd.DataFrame, region_le: LabelEncoder) -> np.ndarray:
    """Encode region and return numeric feature matrix."""
    region_enc = region_le.transform(df["region"])
    return np.column_stack([
        df["age"].values,
        df["educ"].values,
        region_enc,
        df["income"].values,
        df["married"].values,
    ])


def compute_pmse(
    df_conf: pd.DataFrame,
    df_synth: pd.DataFrame,
    random_state: int = 2025,
) -> dict:
    """
    Compute pMSE utility metric.

    Procedure:
        1. Label confidential records 0, synthetic records 1.
        2. Stack them into a combined dataset.
        3. Standardize features.
        4. Train logistic regression to predict the label.
        5. Compute pMSE = mean((predicted_prob_synthetic - 0.5)^2).

    A low pMSE means the classifier assigns probabilities close to 0.5 for all
    records, indicating it cannot distinguish real from synthetic.

    Returns
    -------
    dict with keys: pmse, classifier_accuracy, n_conf, n_synth
    """
    le = LabelEncoder()
    le.fit(df_conf["region"])

    n_use = min(len(df_conf), len(df_synth))
    rng = np.random.default_rng(random_state)

    # Sample equal numbers from each dataset to avoid class imbalance
    conf_idx = rng.choice(len(df_conf), size=n_use, replace=False)
    synth_idx = rng.choice(len(df_synth), size=n_use, replace=False)

    X_conf = prepare_features(df_conf.iloc[conf_idx], le)
    X_synth = prepare_features(df_synth.iloc[synth_idx], le)

    X = np.vstack([X_conf, X_synth])
    y = np.array([0] * n_use + [1] * n_use)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_scaled, y)

    probs = clf.predict_proba(X_scaled)[:, 1]
    pmse = float(np.mean((probs - 0.5) ** 2))
    accuracy = float(clf.score(X_scaled, y))

    return {
        "pmse": pmse,
        "classifier_accuracy": accuracy,
        "n_conf": n_use,
        "n_synth": n_use,
    }


def interpret_pmse(pmse: float, accuracy: float) -> str:
    """Return a human-readable interpretation of the pMSE value."""
    if pmse < 0.005:
        quality = "Excellent — synthetic data is nearly indistinguishable from confidential data."
    elif pmse < 0.020:
        quality = "Good — classifier has limited ability to distinguish datasets."
    elif pmse < 0.050:
        quality = "Moderate — noticeable differences exist; check which variables drive them."
    else:
        quality = "Poor — synthetic data is clearly distinguishable; synthesis needs improvement."

    return (
        f"pMSE = {pmse:.6f}\n"
        f"  Range: 0.000 (ideal) to 0.250 (perfectly distinguishable)\n"
        f"  Classifier accuracy: {accuracy:.3f}  (0.50 = random, 1.00 = perfect discrimination)\n"
        f"  Assessment: {quality}"
    )


if __name__ == "__main__":
    df_conf, df_synth = load_datasets()
    print(f"Loaded: confidential n={len(df_conf)}, synthetic n={len(df_synth)}")
    print()

    results = compute_pmse(df_conf, df_synth, random_state=2025)
    print("Propensity Score Utility (pMSE) Assessment")
    print("=" * 55)
    print(interpret_pmse(results["pmse"], results["classifier_accuracy"]))
    print()
    print("Formula: pMSE = mean( (P(record is synthetic) - 0.5)^2 )")
    print()
    print("Note: pMSE is a global metric. A good pMSE does not guarantee")
    print("that every specific analysis is valid. Use regression tests")
    print("(05_utility_regression.py) for analysis-specific validation.")
