"""
Chapter 5 Example 3: Synthetic Record Generation
=================================================

Generates two synthetic datasets (source_a.csv and source_b.csv) representing
the same 200 people as they appear in two different data systems: a survey
(clean-ish) and an administrative file (noisier).

Noise functions model real-world data quality issues:
- Name typos, abbreviations (first initial only), and missing values
- Date of birth transpositions, year-off errors, and missing values
- Address abbreviations ("Street" -> "St") and missing values

Source B contains records for 180 of the 200 people in Source A (20 are absent)
plus 20 new entities not found in Source A. This mirrors operational reality:
administrative files are never a perfect superset of survey rosters.

Outputs
-------
source_a.csv : 200 records (survey, relatively clean)
source_b.csv : 200 records (admin, noisy; 180 overlap + 20 new)

Requirements: numpy, pandas
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Noise functions
# ---------------------------------------------------------------------------

def add_noise(name, p_typo=0.25, p_abbrev=0.20, p_missing=0.10, rng=None):
    """
    Add realistic noise to a person name field.

    Applies one transformation drawn in order: missing -> abbreviation -> typo.
    Only one transformation is applied per call.

    Parameters
    ----------
    name : str
        Clean full name, e.g. "Maria Garcia".
    p_typo : float
        Probability of introducing a single-character typo.
    p_abbrev : float
        Probability of abbreviating first name to initial ("M. Garcia").
    p_missing : float
        Probability of returning NaN (field not recorded).
    rng : numpy.random.Generator, optional
        Random generator for reproducibility. Created fresh if None.

    Returns
    -------
    str or float
        Noisy name string, or NaN if missing.
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < p_missing:
        return np.nan
    if rng.random() < p_abbrev:
        parts = name.split()
        return parts[0][0] + ". " + " ".join(parts[1:]) if len(parts) > 1 else name
    if rng.random() < p_typo and len(name) > 3:
        pos = rng.integers(1, len(name) - 1)
        chars = list(name)
        chars[pos] = rng.choice(list("aeiourstln"))
        return "".join(chars)
    return name


def add_date_noise(dob, p_transpose=0.10, p_year_off=0.08, p_missing=0.08, rng=None):
    """
    Add noise to a date-of-birth string in 'YYYY-MM-DD' format.

    Parameters
    ----------
    dob : str
        Clean DOB string, e.g. "1985-04-12".
    p_transpose : float
        Probability of transposing month and day.
    p_year_off : float
        Probability of shifting year by +/- 1.
    p_missing : float
        Probability of returning NaN.
    rng : numpy.random.Generator, optional
        Random generator. Created fresh if None.

    Returns
    -------
    str or float
        Noisy DOB string, or NaN if missing.
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < p_missing:
        return np.nan
    y, m, d = dob.split("-")
    if rng.random() < p_year_off:
        y = str(int(y) + rng.choice([-1, 1]))
    if rng.random() < p_transpose:
        m, d = d, m
    return f"{y}-{m}-{d}"


def add_address_noise(addr, p_abbrev=0.35, p_missing=0.12, rng=None):
    """
    Add noise to a street address by abbreviating road type or dropping it.

    Parameters
    ----------
    addr : str
        Clean address string, e.g. "123 Main Street, Austin TX".
    p_abbrev : float
        Probability of replacing full road type with abbreviation.
    p_missing : float
        Probability of returning NaN.
    rng : numpy.random.Generator, optional
        Random generator. Created fresh if None.

    Returns
    -------
    str or float
        Noisy address string, or NaN if missing.
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < p_missing:
        return np.nan
    replacements = {
        " Street":    " St",
        " Avenue":    " Ave",
        " Road":      " Rd",
        " Boulevard": " Blvd",
        " Drive":     " Dr",
        " Lane":      " Ln",
    }
    if rng.random() < p_abbrev:
        for full, abbr in replacements.items():
            addr = addr.replace(full, abbr)
    return addr


# ---------------------------------------------------------------------------
# Record generation
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer",
    "Michael", "Linda", "David", "Barbara", "William", "Susan",
    "Richard", "Jessica", "Joseph", "Sarah", "Thomas", "Karen",
    "Charles", "Lisa", "Maria", "Carlos", "Wei", "Mei", "Fatima",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
    "Wilson", "Anderson", "Taylor", "Thomas", "Moore", "Jackson",
    "Martin", "Lee", "Thompson", "White", "Harris", "Clark", "Lewis",
]
STREETS = [
    "Main Street", "Oak Avenue", "Elm Road", "Park Boulevard",
    "Maple Drive", "Cedar Lane", "Birch Street", "Pine Avenue",
]
CITIES = ["Austin", "Houston", "Los Angeles", "Chicago", "Miami"]
STATES = ["TX", "CA", "NY", "FL", "IL", "OH", "PA", "GA"]


def generate_true_records(n=200, seed=42):
    """
    Generate N true person records (clean ground-truth entities).

    Parameters
    ----------
    n : int
        Number of people to generate.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: true_id, name, dob, address.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        fname  = rng.choice(FIRST_NAMES)
        lname  = rng.choice(LAST_NAMES)
        year   = rng.integers(1940, 2000)
        month  = rng.integers(1, 13)
        day    = rng.integers(1, 29)
        num    = rng.integers(1, 1000)
        street = rng.choice(STREETS)
        city   = rng.choice(CITIES)
        state  = rng.choice(STATES)
        rows.append({
            "true_id": i,
            "name":    f"{fname} {lname}",
            "dob":     f"{year}-{month:02d}-{day:02d}",
            "address": f"{num} {street}, {city} {state}",
        })
    return pd.DataFrame(rows)


def make_source_a(true_df):
    """
    Create Source A (survey): all 200 records, minimal noise.

    Parameters
    ----------
    true_df : pd.DataFrame
        Ground-truth records from generate_true_records().

    Returns
    -------
    pd.DataFrame
        Columns: record_id, true_id, name, dob, address.
    """
    df = true_df.copy()
    df["record_id"] = ["A" + str(i).zfill(3) for i in range(len(df))]
    return df[["record_id", "true_id", "name", "dob", "address"]]


def make_source_b(true_df, n_overlap=180, seed=99):
    """
    Create Source B (admin): 180 overlapping records with noise + 20 new entities.

    Parameters
    ----------
    true_df : pd.DataFrame
        Ground-truth records.
    n_overlap : int
        How many of the 200 true entities appear in Source B.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: record_id, true_id, name, dob, address.
    """
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(true_df), size=n_overlap, replace=False)

    rows = []
    for idx in selected:
        row = true_df.iloc[idx].to_dict()
        row["record_id"] = "B" + str(idx).zfill(3)
        row["name"]    = add_noise(row["name"], rng=rng)
        row["dob"]     = add_date_noise(row["dob"], rng=rng)
        row["address"] = add_address_noise(row["address"], rng=rng)
        rows.append(row)

    n_new = len(true_df) - n_overlap
    for j in range(n_new):
        fname  = rng.choice(FIRST_NAMES)
        lname  = rng.choice(LAST_NAMES)
        street = rng.choice(STREETS)
        rows.append({
            "true_id":   len(true_df) + j,
            "record_id": f"B{len(true_df) + j}",
            "name":      f"{fname} {lname}",
            "dob":       (f"{rng.integers(1940,2000)}-"
                          f"{rng.integers(1,13):02d}-"
                          f"{rng.integers(1,29):02d}"),
            "address":   f"{rng.integers(1,1000)} {street}, SomeCity TX",
        })

    df = pd.DataFrame(rows).reset_index(drop=True)
    return df[["record_id", "true_id", "name", "dob", "address"]]


if __name__ == "__main__":
    true_df = generate_true_records(n=200, seed=42)
    source_a = make_source_a(true_df)
    source_b = make_source_b(true_df, n_overlap=180, seed=99)

    source_a.to_csv("source_a.csv", index=False)
    source_b.to_csv("source_b.csv", index=False)

    print(f"Source A (survey):  {len(source_a)} records -> source_a.csv")
    print(f"Source B (admin):   {len(source_b)} records -> source_b.csv")
    print(f"Overlap (true matches): 180 pairs")
    print(f"\nSample from Source A (clean):")
    print(source_a.head(3).to_string(index=False))
    print(f"\nSample from Source B (noisy):")
    # Show rows that correspond to the first 3 true_ids for comparison
    print(source_b[source_b["true_id"].isin([0, 1, 2])].to_string(index=False))
