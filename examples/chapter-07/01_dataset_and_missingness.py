"""
01_dataset_and_missingness.py
Chapter 7: Imputation Methods for Survey Data

Creates a synthetic ACS-like dataset (800 records), introduces MAR missingness
on income, visualizes the three missingness mechanisms (MCAR, MAR, MNAR),
demonstrates complete-case bias, and saves the base dataset.

Usage:
    python 01_dataset_and_missingness.py

Outputs:
    base_data.csv   -- 800-record dataset with income_obs (MAR missingness)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_RECORDS = 800
RANDOM_SEED = 42
OUTPUT_FILE = "base_data.csv"

REGION_PREMIUMS = {
    "Northeast": 5000,
    "Midwest": -2000,
    "South": -1000,
    "West": 4000,
}
REGION_PROBS = [0.18, 0.22, 0.38, 0.22]  # Northeast, Midwest, South, West
EDUC_PROBS = [0.15, 0.30, 0.35, 0.20]    # <HS, HS diploma, Some college/BA, Graduate

# MAR missingness: probability of skipping income rises with education level
MAR_BASE_PROB = 0.04
MAR_EDUC_COEF = 0.06  # additional miss probability per education unit


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def generate_acs_like_dataset(n: int, seed: int) -> pd.DataFrame:
    """
    Generate a synthetic ACS-like dataset with continuous income as the
    target variable. Income is a linear function of age, education, region,
    and hours worked per week plus Gaussian noise.

    Returns a DataFrame with columns:
        age, educ, region, fulltime, hours_wk,
        income_true, income_obs, missing
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(22, 68, n)
    # educ: 1=<HS, 2=HS diploma, 3=Some college/BA, 4=Graduate degree
    educ = rng.choice([1, 2, 3, 4], n, p=EDUC_PROBS)
    region = rng.choice(
        ["Northeast", "Midwest", "South", "West"], n, p=REGION_PROBS
    )
    fulltime = rng.binomial(1, 0.65, n)
    hours_wk = np.where(
        fulltime,
        rng.normal(42, 7, n),
        rng.normal(22, 8, n),
    )
    hours_wk = np.clip(hours_wk, 4, 70).astype(int)

    region_adj = np.array([REGION_PREMIUMS[r] for r in region])
    income_true = (
        15000
        + 5500 * educ
        + 400 * (age - 22)
        + region_adj
        + 1200 * (hours_wk - 20)
        + rng.normal(0, 9000, n)
    )
    income_true = np.clip(income_true, 0, 200_000).astype(int)

    # Introduce MAR missingness: higher-education respondents skip income more
    miss_prob = MAR_BASE_PROB + MAR_EDUC_COEF * educ
    missing_mask = rng.random(n) < miss_prob
    income_obs = income_true.astype(float)
    income_obs[missing_mask] = np.nan

    return pd.DataFrame({
        "age": age,
        "educ": educ,
        "region": region,
        "fulltime": fulltime,
        "hours_wk": hours_wk,
        "income_true": income_true,
        "income_obs": income_obs,
        "missing": missing_mask,
    })


# ---------------------------------------------------------------------------
# Missingness mechanism visualization (toy dataset, n=300)
# ---------------------------------------------------------------------------
def visualize_mechanisms(seed: int) -> None:
    """
    Plot the three missingness mechanisms (MCAR, MAR, MNAR) side by side
    on a small toy dataset. Shows which records are observed vs. missing
    and prints the observed-mean bias for each mechanism.
    """
    rng = np.random.default_rng(seed)
    n = 300
    age = rng.integers(25, 65, n)
    educ = rng.choice([1, 2, 3, 4], n, p=EDUC_PROBS)
    income_true = (
        20_000 + 3_000 * educ + 800 * (age - 25) + rng.normal(0, 8_000, n)
    )
    income_true = np.clip(income_true, 5_000, 150_000)

    mcar_mask = rng.random(n) < 0.20
    mar_prob = 0.05 + 0.07 * educ
    mar_mask = rng.random(n) < mar_prob
    mnar_prob = np.where(income_true > 60_000, 0.35, 0.08)
    mnar_mask = rng.random(n) < mnar_prob

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    mechanisms = [("MCAR", mcar_mask), ("MAR", mar_mask), ("MNAR", mnar_mask)]

    for ax, (label, mask) in zip(axes, mechanisms):
        ax.scatter(
            age[~mask], income_true[~mask],
            alpha=0.4, s=15, color="steelblue", label="Observed",
        )
        ax.scatter(
            age[mask], income_true[mask],
            alpha=0.5, s=15, color="tomato", label="Missing", marker="x",
        )
        ax.set_title(f"{label} ({mask.sum()} missing, {mask.mean():.0%})")
        ax.set_xlabel("Age")
        ax.legend(markerscale=1.5, fontsize=8)

    axes[0].set_ylabel("Income ($)")
    fig.suptitle(
        "Three missingness mechanisms: same dataset, different reasons for missing",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("01_mechanisms.png", dpi=120, bbox_inches="tight")
    plt.show()

    print("\nMissing counts and observed-mean bias by mechanism:")
    true_mean = income_true.mean()
    for label, mask in mechanisms:
        obs_mean = income_true[~mask].mean()
        print(
            f"  {label}: {mask.sum():3d} missing | "
            f"observed mean = ${obs_mean:,.0f} | "
            f"true mean = ${true_mean:,.0f} | "
            f"bias = ${obs_mean - true_mean:+,.0f}"
        )


# ---------------------------------------------------------------------------
# Complete-case bias demonstration
# ---------------------------------------------------------------------------
def show_complete_case_bias(df: pd.DataFrame) -> None:
    """
    Demonstrate that deleting incomplete records understates mean income
    when the missingness mechanism is MAR (higher-education respondents skip more).
    """
    n = len(df)
    n_missing = df["missing"].sum()
    true_mean = df["income_true"].mean()
    cc_mean = df["income_obs"].mean()  # pandas skips NaN by default

    print(f"\nDataset summary: {n} records, {n_missing} missing income ({n_missing/n:.1%})")
    print("\nMissing rate by education level:")
    educ_labels = {1: "< HS", 2: "HS diploma", 3: "Some college/BA", 4: "Graduate"}
    for e in [1, 2, 3, 4]:
        sub = df[df.educ == e]
        print(f"  Educ {e} ({educ_labels[e]:<20}): {sub['missing'].mean():.1%} missing")

    print(f"\nTrue mean income:                 ${true_mean:,.0f}")
    print(f"Complete-case mean (n={n-n_missing}): ${cc_mean:,.0f}")
    print(f"Bias from complete-case analysis: ${cc_mean - true_mean:+,.0f}")
    print(
        "\nInterpretation: because higher-education (higher-income) respondents skip more,\n"
        "complete-case analysis over-represents lower-education, lower-income respondents\n"
        "and understates average income."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Chapter 7: Dataset and Missingness ===\n")

    # 1. Visualize mechanisms on toy data
    print("1. Visualizing three missingness mechanisms ...")
    visualize_mechanisms(seed=RANDOM_SEED)

    # 2. Build the full ACS-like dataset used throughout the chapter
    print("\n2. Generating 800-record ACS-like dataset ...")
    df = generate_acs_like_dataset(N_RECORDS, RANDOM_SEED)

    # 3. Show complete-case bias
    print("\n3. Complete-case bias demonstration:")
    show_complete_case_bias(df)

    # 4. Save for downstream scripts
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDataset saved to {OUTPUT_FILE}")
    print(f"Columns: {list(df.columns)}")
