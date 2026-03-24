"""
01_synthetic_county_data.py
===========================
Generate a synthetic county-level ACS-style dataset for use in Chapter 6
dimension reduction and clustering examples.

Design
------
Three latent demographic profiles drive the data:
  - A_urban    : high income, high education, dense, high renters (30% of counties)
  - B_suburban : moderate income, homeowners, mid-density          (40% of counties)
  - C_rural    : lower income, older population, sparse            (30% of counties)

15 ACS-style variables are generated with profile-specific means and realistic
noise. Synthetic state FIPS codes and county IDs are assigned to support
geographic joins in downstream examples.

Output
------
Writes county_data.csv to the same directory as this script.

Usage
-----
    python 01_synthetic_county_data.py

Requirements: numpy, pandas
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_COUNTIES = 400
RANDOM_SEED = 42
OUTPUT_FILE = Path(__file__).parent / "county_data.csv"

PROFILE_PROBS = {"A_urban": 0.30, "B_suburban": 0.40, "C_rural": 0.30}

# ACS variable column names (order matches the per-profile spec below)
FEATURE_COLS = [
    "median_age",
    "pct_bachelors",
    "median_hh_income",
    "pct_poverty",
    "pct_owner_occupied",
    "pct_employed",
    "pct_under18",
    "pct_over65",
    "pct_hispanic",
    "pct_foreign_born",
    "pct_renter",
    "pop_density_log",
    "median_gross_rent",
    "pct_no_vehicle",
    "pct_broadband",
]

# ---------------------------------------------------------------------------
# Per-profile variable distributions
# Each entry: (mean, std, clip_low, clip_high)  -- None = no clip
# ---------------------------------------------------------------------------
PROFILE_SPECS = {
    "A_urban": {
        "median_age":         (36,     3,      None, None),
        "pct_bachelors":      (42,     8,      10,   80),
        "median_hh_income":   (72000,  12000,  None, None),
        "pct_poverty":        (10,     3,      2,    30),
        "pct_owner_occupied": (48,     8,      20,   80),
        "pct_employed":       (64,     4,      40,   80),
        "pct_under18":        (20,     3,      10,   35),
        "pct_over65":         (13,     3,      5,    30),
        "pct_hispanic":       (22,     10,     2,    70),
        "pct_foreign_born":   (18,     6,      2,    50),
        "pct_renter":         (52,     8,      20,   80),
        "pop_density_log":    (7.5,    0.8,    None, None),
        "median_gross_rent":  (1350,   200,    None, None),
        "pct_no_vehicle":     (15,     5,      2,    40),
        "pct_broadband":      (82,     5,      60,   97),
    },
    "B_suburban": {
        "median_age":         (39,     4,      None, None),
        "pct_bachelors":      (32,     8,      10,   70),
        "median_hh_income":   (62000,  10000,  None, None),
        "pct_poverty":        (12,     4,      3,    30),
        "pct_owner_occupied": (68,     8,      30,   90),
        "pct_employed":       (61,     4,      40,   80),
        "pct_under18":        (23,     3,      10,   40),
        "pct_over65":         (16,     4,      5,    35),
        "pct_hispanic":       (12,     8,      1,    50),
        "pct_foreign_born":   (10,     5,      1,    35),
        "pct_renter":         (32,     7,      10,   60),
        "pop_density_log":    (5.5,    1.0,    None, None),
        "median_gross_rent":  (1050,   180,    None, None),
        "pct_no_vehicle":     (7,      3,      1,    25),
        "pct_broadband":      (78,     6,      55,   96),
    },
    "C_rural": {
        "median_age":         (44,     5,      None, None),
        "pct_bachelors":      (18,     6,      5,    50),
        "median_hh_income":   (46000,  9000,   None, None),
        "pct_poverty":        (18,     5,      4,    40),
        "pct_owner_occupied": (75,     7,      40,   95),
        "pct_employed":       (56,     5,      35,   75),
        "pct_under18":        (21,     3,      10,   38),
        "pct_over65":         (20,     5,      8,    40),
        "pct_hispanic":       (8,      7,      1,    40),
        "pct_foreign_born":   (5,      3,      0.5,  20),
        "pct_renter":         (25,     6,      10,   55),
        "pop_density_log":    (3.0,    0.9,    None, None),
        "median_gross_rent":  (760,    130,    None, None),
        "pct_no_vehicle":     (6,      3,      1,    22),
        "pct_broadband":      (68,     8,      40,   90),
    },
}


def make_county(profile: str, rng: np.random.Generator) -> dict:
    """Generate one county row from the given demographic profile."""
    spec = PROFILE_SPECS[profile]
    row = {}
    for var, (mean, std, lo, hi) in spec.items():
        value = rng.normal(mean, std)
        if lo is not None or hi is not None:
            value = float(np.clip(value, lo, hi))
        row[var] = value
    return row


def build_dataset(n_counties: int, seed: int) -> pd.DataFrame:
    """
    Build and return the synthetic county dataset.

    Parameters
    ----------
    n_counties : int
        Number of county rows to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per county, 15 ACS feature columns plus profile, state_fips,
        and county_id.
    """
    rng = np.random.default_rng(seed)
    profiles = list(PROFILE_SPECS.keys())
    probs = [PROFILE_PROBS[p] for p in profiles]
    profile_labels = rng.choice(profiles, size=n_counties, p=probs)

    rows = [make_county(p, rng) for p in profile_labels]
    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    df["profile"] = profile_labels

    # Assign synthetic state FIPS codes proportionally
    state_blocks = (
        ["06"] * 60   # California
        + ["48"] * 80   # Texas
        + ["36"] * 50   # New York
        + ["17"] * 50   # Illinois
        + ["39"] * 40   # Ohio
        + ["12"] * 40   # Florida
        + ["37"] * 30   # North Carolina
        + ["47"] * 50   # Tennessee
    )
    padded = (state_blocks + ["99"] * (n_counties - len(state_blocks)))[:n_counties]
    df["state_fips"] = padded
    df["county_id"] = [
        f"{df['state_fips'].iloc[i]}{str(i % 100).zfill(3)}"
        for i in range(n_counties)
    ]

    return df


if __name__ == "__main__":
    df = build_dataset(N_COUNTIES, RANDOM_SEED)

    print(f"Dataset shape: {df.shape}")
    print(f"\nProfile distribution:\n{df['profile'].value_counts().to_string()}")
    print(f"\nSummary statistics (first 5 features):")
    print(df[FEATURE_COLS[:5]].describe().round(1).to_string())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")
