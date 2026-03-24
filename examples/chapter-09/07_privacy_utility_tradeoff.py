"""
07_privacy_utility_tradeoff.py
==============================
Chapter 9: Synthetic Data Generation for Federal Statistics

Demonstrates the privacy-utility tradeoff using a KNN synthesizer with
varying k. Lower k means the synthesizer copies nearest neighbors more
faithfully (higher utility, higher disclosure risk). Higher k means heavy
smoothing (lower disclosure risk, less utility).

Why this matters:
    Every synthesis decision implicitly trades privacy for utility. There is no
    free lunch: you cannot simultaneously achieve perfect statistical fidelity
    and strong privacy guarantees. This tradeoff is not a technical problem to
    be solved; it is a policy choice about which analyses matter most and how
    much privacy risk is acceptable.

    The KNN synthesizer is a clean demonstration because the parameter k is
    directly interpretable: "how many real people's records does this synthetic
    record draw from?"

Privacy proxy:
    Nearest-neighbor distance ratio (NNDR): for each synthetic record, compute
    the ratio of the distance to its first nearest neighbor in the confidential
    data to the distance to its second nearest neighbor.
    Low NNDR (close to 0) means the synthetic record is very close to a specific
    real person's record — high disclosure risk.

Usage:
    python 07_privacy_utility_tradeoff.py
    (Requires confidential_microdata.csv and synthetic_data.csv)

Outputs:
    - Privacy-utility tradeoff table printed to stdout
    - privacy_utility_tradeoff.png: scatter plot

Requirements:
    Python 3.9+, numpy, pandas, matplotlib, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load confidential and synthetic datasets from CSV files."""
    for path in ["confidential_microdata.csv", "synthetic_data.csv"]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run 01_confidential_dataset.py first."
            )
    return pd.read_csv("confidential_microdata.csv"), pd.read_csv("synthetic_data.csv")


def knn_synthesize_income(
    df_conf: pd.DataFrame,
    k: int,
    n_synth: int = 600,
    random_state: int = 2025,
) -> pd.DataFrame:
    """
    Synthesize income using KNN regression on (age, educ, region).

    KNN with k=1 copies the nearest real record's income almost exactly.
    KNN with large k averages many neighbors, smoothing the distribution
    but losing rare patterns.

    The synthetic covariate rows are drawn randomly from the confidential
    dataset to keep the comparison fair.
    """
    rng = np.random.default_rng(random_state)
    le = LabelEncoder()
    region_enc = le.fit_transform(df_conf["region"])

    X = np.column_stack([df_conf["age"].values, df_conf["educ"].values, region_enc])
    y = df_conf["income"].values

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X, y)

    # Draw synthetic covariate rows from confidential data
    idx = rng.integers(0, len(df_conf), size=n_synth)
    X_synth = X[idx]
    pred_income = knn.predict(X_synth)

    # Add modest residual noise so the synthetic income is not deterministic
    residuals = y - knn.predict(X)
    noise = rng.normal(0, residuals.std() * 0.3, n_synth)
    synth_income = np.clip(pred_income + noise, 5000, 300_000).astype(int)

    return pd.DataFrame({
        "age":    df_conf["age"].values[idx],
        "educ":   df_conf["educ"].values[idx],
        "region": df_conf["region"].values[idx],
        "income": synth_income,
    })


def compute_pmse_simple(
    df_conf: pd.DataFrame,
    df_synth: pd.DataFrame,
    random_state: int = 2025,
) -> float:
    """Compute pMSE for income-only comparison (simplified for tradeoff plot)."""
    le = LabelEncoder()
    le.fit(df_conf["region"])
    region_enc_c = le.transform(df_conf["region"])
    region_enc_s = le.transform(df_synth["region"])

    X_conf = np.column_stack([df_conf["age"], df_conf["educ"], region_enc_c, df_conf["income"]])
    X_synth = np.column_stack([df_synth["age"], df_synth["educ"], region_enc_s, df_synth["income"]])

    n_use = min(len(df_conf), len(df_synth))
    rng = np.random.default_rng(random_state)
    c_idx = rng.choice(len(X_conf), n_use, replace=False)
    s_idx = rng.choice(len(X_synth), n_use, replace=False)

    X = np.vstack([X_conf[c_idx], X_synth[s_idx]])
    y = np.array([0] * n_use + [1] * n_use)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=500, random_state=random_state)
    clf.fit(X_scaled, y)
    probs = clf.predict_proba(X_scaled)[:, 1]
    return float(np.mean((probs - 0.5) ** 2))


def compute_nndr(
    df_conf: pd.DataFrame,
    df_synth: pd.DataFrame,
    n_sample: int = 200,
    random_state: int = 2025,
) -> float:
    """
    Compute mean nearest-neighbor distance ratio (NNDR) as a privacy proxy.

    For each synthetic record, find the two nearest confidential neighbors.
    NNDR = dist_to_1st_neighbor / dist_to_2nd_neighbor.
    Low NNDR means the synthetic record is disproportionately close to one
    real record — a disclosure risk signal.
    """
    le = LabelEncoder()
    le.fit(df_conf["region"])

    X_conf = np.column_stack([
        df_conf["age"].values,
        df_conf["educ"].values,
        le.transform(df_conf["region"]),
        df_conf["income"].values / 10_000,  # scale to comparable range
    ])
    X_synth = np.column_stack([
        df_synth["age"].values,
        df_synth["educ"].values,
        le.transform(df_synth["region"]),
        df_synth["income"].values / 10_000,
    ])

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_synth), size=min(n_sample, len(X_synth)), replace=False)
    X_sample = X_synth[idx]

    nbrs = NearestNeighbors(n_neighbors=2)
    nbrs.fit(X_conf)
    distances, _ = nbrs.kneighbors(X_sample)

    # Avoid divide-by-zero when both neighbors are equidistant
    d1 = distances[:, 0]
    d2 = distances[:, 1]
    nndr = d1 / np.where(d2 > 0, d2, 1e-9)
    return float(np.mean(nndr))


if __name__ == "__main__":
    df_conf, _ = load_datasets()
    print(f"Loaded confidential microdata: n={len(df_conf)}")
    print()

    k_values = [1, 10, 50]
    results = []

    for k in k_values:
        df_synth_k = knn_synthesize_income(df_conf, k=k, random_state=2025)
        pmse = compute_pmse_simple(df_conf, df_synth_k, random_state=2025)
        nndr = compute_nndr(df_conf, df_synth_k, random_state=2025)
        results.append({"k": k, "pmse": pmse, "nndr": nndr})
        print(f"k={k:>3}: pMSE={pmse:.5f}  NNDR={nndr:.3f}")

    print()
    print("Privacy-utility tradeoff table")
    print("=" * 60)
    print(f"{'k':>4} {'pMSE (utility)':>16} {'NNDR (privacy proxy)':>22} {'Interpretation':>14}")
    print("-" * 60)
    for r in results:
        if r["k"] == 1:
            interp = "High risk"
        elif r["k"] == 10:
            interp = "Balanced"
        else:
            interp = "Over-smoothed"
        print(f"{r['k']:>4} {r['pmse']:>16.5f} {r['nndr']:>22.3f} {interp:>14}")

    print()
    print("pMSE: lower = better utility (harder to distinguish from confidential)")
    print("NNDR: higher = more privacy (synthetic records not too close to any one real record)")
    print("The tradeoff: k=1 achieves good utility but at high privacy cost.")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 5))
    pmse_vals = [r["pmse"] for r in results]
    nndr_vals = [r["nndr"] for r in results]
    k_labels  = [str(r["k"]) for r in results]

    ax.scatter(nndr_vals, pmse_vals, s=120, color=["tomato", "darkorange", "steelblue"], zorder=5)
    for nndr, pmse, k in zip(nndr_vals, pmse_vals, k_labels):
        ax.annotate(f"k={k}", (nndr, pmse), textcoords="offset points", xytext=(8, 4), fontsize=10)

    ax.set_xlabel("NNDR (privacy proxy — higher = more privacy)")
    ax.set_ylabel("pMSE (utility — lower = better utility)")
    ax.set_title("Privacy-utility tradeoff: KNN synthesizer with varying k")
    ax.invert_xaxis()  # Left = less privacy; right = more privacy

    plt.tight_layout()
    plt.savefig("privacy_utility_tradeoff.png", dpi=150, bbox_inches="tight")
    print("\nSaved figure: privacy_utility_tradeoff.png")
    plt.close()
