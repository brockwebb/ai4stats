"""
01_confidential_dataset.py
==========================
Chapter 9: Synthetic Data Generation for Federal Statistics

Generates a synthetic confidential microdata dataset representing the kind of
record-level data a federal statistical agency might hold but cannot release
directly. Demonstrates three traditional disclosure avoidance methods
(top-coding, noise infusion, and data swapping) and their effect on the income
distribution.

Why this matters:
    Before agencies can discuss synthetic data, practitioners need to understand
    why traditional disclosure avoidance methods are imperfect. Each method
    introduces systematic bias or destroys specific analytical relationships.
    This script makes those tradeoffs concrete.

Usage:
    python 01_confidential_dataset.py

Outputs:
    - Summary statistics printed to stdout
    - traditional_da_comparison.png saved to current directory
    - confidential_microdata.csv saved to current directory (used by other scripts)

Requirements:
    Python 3.9+, numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_confidential_dataset(seed: int = 2025, n: int = 600) -> pd.DataFrame:
    """
    Generate a synthetic confidential microdata dataset.

    The dataset mimics a simplified public-use file from a federal statistical
    agency: demographic variables linked to income. The income model includes
    realistic positive returns to education and age, plus regional variation.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n : int
        Number of records to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: age, educ, region, married, income.
    """
    np.random.seed(seed)

    # Age: roughly working-age population
    age = np.random.normal(42, 14, n)
    age = np.clip(np.round(age), 18, 80).astype(int)

    # Education: years of schooling with realistic probabilities
    educ = np.random.choice(
        [9, 12, 14, 16, 18],
        size=n,
        p=[0.10, 0.35, 0.20, 0.25, 0.10],
    )

    # Region: four Census regions with approximate population shares
    region = np.random.choice(
        ["Northeast", "South", "Midwest", "West"],
        size=n,
        p=[0.20, 0.35, 0.25, 0.20],
    )

    # Marital status: slightly more likely married overall
    married = np.random.binomial(1, 0.52, n)

    # Income: log-normal with systematic effects for age, education, and region
    region_premium = {
        "Northeast": 5000,
        "South": -2000,
        "Midwest": 0,
        "West": 3000,
    }
    log_income_base = (
        10.5
        + 0.03 * (age - 40)
        + 0.06 * (educ - 12)
        + np.array([region_premium[r] for r in region]) / 50000
        + np.random.normal(0, 0.45, n)
    )
    income = np.exp(log_income_base).astype(int)
    income = np.clip(income, 5000, 300000)

    return pd.DataFrame(
        {
            "age": age,
            "educ": educ,
            "region": region,
            "married": married,
            "income": income,
        }
    )


def apply_traditional_da(df: pd.DataFrame) -> dict:
    """
    Apply three traditional disclosure avoidance methods to the income variable.

    Returns a dictionary mapping method name to transformed income array.
    """
    income = df["income"].values.copy()

    # Method 1: Top-coding — cap income at $150,000
    # Rationale: high earners are identifiable; replace their exact value with a cap.
    # Effect: removes the right tail; underestimates high-income statistics.
    topcoded = np.minimum(income, 150_000)

    # Method 2: Noise addition — add +/- 5% uniform random noise
    # Rationale: makes it harder to match records to external data.
    # Effect: introduces random errors analysts cannot remove; biases regression coefficients.
    noisy = income * (1 + np.random.uniform(-0.05, 0.05, len(income)))
    noisy = np.clip(noisy, 0, None).astype(int)

    # Method 3: Record swapping — swap income between 10% of records
    # Rationale: breaks the link between income and quasi-identifiers.
    # Effect: destroys income-demographic correlations for swapped records.
    swapped = income.copy()
    n_swap = int(0.10 * len(income))
    swap_indices = np.random.choice(len(income), size=n_swap * 2, replace=False)
    for i in range(n_swap):
        a, b = swap_indices[i], swap_indices[i + n_swap]
        swapped[a], swapped[b] = swapped[b], swapped[a]

    return {
        "Confidential (true)": income,
        "Top-coded (>$150K)": topcoded,
        "Noise (+/- 5%)": noisy,
        "Swapped (10%)": swapped,
    }


def print_da_summary(income_methods: dict) -> None:
    """Print mean and standard deviation for each disclosure avoidance method."""
    true_mean = income_methods["Confidential (true)"].mean()
    true_std = income_methods["Confidential (true)"].std()

    print("Impact of traditional disclosure avoidance methods on income statistics")
    print("=" * 70)
    print(f"{'Method':<25} {'Mean':>12} {'Bias':>12} {'Std Dev':>12}")
    print("-" * 70)
    for name, values in income_methods.items():
        mean = values.mean()
        bias = mean - true_mean
        std = values.std()
        print(f"{name:<25} ${mean:>10,.0f} ${bias:>+10,.0f} ${std:>10,.0f}")

    print()
    print("Interpretation:")
    print("  Top-coding systematically depresses the mean for high earners.")
    print("  Noise addition inflates standard deviation and biases individual records.")
    print("  Swapping breaks income-demographic correlations without obvious mean shift.")
    print()
    print("None of these methods provides formal privacy guarantees.")
    print("Each introduces distortions that vary by the analysis being conducted.")


def plot_da_comparison(income_methods: dict, output_path: str = "traditional_da_comparison.png") -> None:
    """Create a 2x2 panel comparison of income distributions under each method."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bins = np.linspace(0, 200_000, 60)
    colors = ["steelblue", "tomato", "darkorange", "seagreen"]
    true_income = income_methods["Confidential (true)"]

    for ax, (name, values), color in zip(axes.flat, income_methods.items(), colors):
        ax.hist(true_income, bins=bins, alpha=0.35, color="steelblue", density=True, label="True")
        ax.hist(values, bins=bins, alpha=0.55, color=color, density=True, label=name)
        ax.set_xlabel("Income ($)")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 200_000)

    fig.suptitle("Traditional disclosure avoidance methods alter the income distribution", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Generating synthetic confidential microdata (n=600, seed=2025)...")
    df = generate_confidential_dataset(seed=2025, n=600)

    print(f"\nDataset shape: {df.shape}")
    print("\nFirst 8 records:")
    print(df.head(8).to_string(index=False))

    print("\nSummary statistics:")
    print(df.describe().round(1).to_string())

    print("\nRegion distribution:")
    print(df["region"].value_counts().to_string())

    # Save confidential dataset for use by other scripts
    df.to_csv("confidential_microdata.csv", index=False)
    print("\nSaved: confidential_microdata.csv")
    print("NOTE: In a real agency, this file would NEVER be released publicly.")

    # Traditional DA comparison
    income_methods = apply_traditional_da(df)
    print_da_summary(income_methods)
    plot_da_comparison(income_methods)
