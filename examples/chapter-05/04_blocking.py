"""
Chapter 5 Example 4: Blocking Strategies
=========================================

Demonstrates two blocking strategies that reduce the record comparison space
before scoring:

  1. Soundex-like blocking on last name: pairs records whose last names
     encode to the same four-character phonetic code.
  2. Birth-year blocking: pairs records with the same birth year.

A union of both blocking keys is used so that a pair qualifies for comparison
if it passes *either* key. This increases recall (fewer missed true matches)
at the cost of slightly more candidate pairs.

Key metric: reduction ratio.
  - Full cross-product: 200 x 200 = 40,000 comparisons
  - After blocking: typically 1,000-2,000 (>95% reduction)

Run 03_synthetic_records.py first to generate source_a.csv and source_b.csv,
or this script will generate them inline.

Requirements: numpy, pandas
"""

import numpy as np
import pandas as pd

# Import noise functions and generators from example 3 if available,
# otherwise define inline.
try:
    from _03_synthetic_records import generate_true_records, make_source_a, make_source_b
except ImportError:
    # Inline fallback so this file is standalone
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from importlib import import_module
    _m = import_module("03_synthetic_records")
    generate_true_records = _m.generate_true_records
    make_source_a = _m.make_source_a
    make_source_b = _m.make_source_b


# ---------------------------------------------------------------------------
# Blocking key functions
# ---------------------------------------------------------------------------

def soundex(name):
    """
    Simplified Soundex phonetic blocking key for a person's last name.

    Returns a four-character code: first letter of last name + three digits
    encoding consonant groups. Used to group records that sound alike even
    when spelled differently (e.g. "Garcia" and "Garsia").

    Parameters
    ----------
    name : str or float
        Full name string "First Last". NaN-safe: returns "X000".

    Returns
    -------
    str
        Four-character blocking key, e.g. "G620".
    """
    if pd.isna(name):
        return "X000"
    name = str(name).upper()
    parts = name.split()
    lname = parts[-1] if parts else name
    if not lname:
        return "X000"
    code = [lname[0]]
    mapping = {
        "BFPV":   "1",
        "CGJKQSXYZ": "2",
        "DT":     "3",
        "L":      "4",
        "MN":     "5",
        "R":      "6",
    }
    for ch in lname[1:].upper():
        for letters, digit in mapping.items():
            if ch in letters:
                code.append(digit)
                break
    code = code[:4]
    while len(code) < 4:
        code.append("0")
    return "".join(code)


def dob_year(dob):
    """
    Extract the four-digit birth year from a 'YYYY-MM-DD' string.

    Used as a coarse blocking key: records born in different years are
    unlikely to be the same person (barring data entry errors).

    Parameters
    ----------
    dob : str or float
        DOB string. NaN-safe: returns "0000".

    Returns
    -------
    str
        Four-character year, e.g. "1985".
    """
    if pd.isna(dob):
        return "0000"
    return str(dob)[:4]


# ---------------------------------------------------------------------------
# Candidate pair generation
# ---------------------------------------------------------------------------

def generate_candidate_pairs(source_a, source_b):
    """
    Generate candidate pairs using the union of Soundex and birth-year blocks.

    A pair (i, j) is included if source_a[i] and source_b[j] share the same
    Soundex key OR the same birth year. Using a union (rather than
    intersection) increases recall: a true match is captured as long as at
    least one blocking key agrees, even if the other field is noisy.

    Parameters
    ----------
    source_a : pd.DataFrame
        Must have columns: name, dob.
    source_b : pd.DataFrame
        Must have columns: name, dob.

    Returns
    -------
    set of (int, int)
        Set of (index_in_A, index_in_B) candidate pairs.
    """
    source_a = source_a.copy()
    source_b = source_b.copy()
    source_a["block_soundex"] = source_a["name"].apply(soundex)
    source_a["block_year"]    = source_a["dob"].apply(dob_year)
    source_b["block_soundex"] = source_b["name"].apply(soundex)
    source_b["block_year"]    = source_b["dob"].apply(dob_year)

    candidate_pairs = set()

    # Soundex blocking
    for key in source_a["block_soundex"].unique():
        a_idx = source_a[source_a["block_soundex"] == key].index.tolist()
        b_idx = source_b[source_b["block_soundex"] == key].index.tolist()
        for ai in a_idx:
            for bi in b_idx:
                candidate_pairs.add((ai, bi))

    # Birth-year blocking
    for key in source_a["block_year"].unique():
        a_idx = source_a[source_a["block_year"] == key].index.tolist()
        b_idx = source_b[source_b["block_year"] == key].index.tolist()
        for ai in a_idx:
            for bi in b_idx:
                candidate_pairs.add((ai, bi))

    return candidate_pairs


def report_blocking_statistics(source_a, source_b, candidate_pairs):
    """
    Print blocking reduction statistics and recall estimate.

    Parameters
    ----------
    source_a : pd.DataFrame
        Source A records.
    source_b : pd.DataFrame
        Source B records.
    candidate_pairs : set
        Output of generate_candidate_pairs().
    """
    total_possible = len(source_a) * len(source_b)
    n_candidates   = len(candidate_pairs)
    reduction      = (1 - n_candidates / total_possible) * 100

    # How many of the 180 true matches fall in the candidate set?
    true_match_ids = set(
        source_b[source_b["true_id"].isin(source_a["true_id"])]["true_id"]
    )
    true_pairs_in_candidates = 0
    for ai, bi in candidate_pairs:
        if source_a.loc[ai, "true_id"] == source_b.loc[bi, "true_id"]:
            true_pairs_in_candidates += 1
    n_true_matches = sum(
        1 for tid in source_a["true_id"] if tid in true_match_ids
    )
    recall = true_pairs_in_candidates / n_true_matches if n_true_matches else 0

    print("Blocking statistics")
    print("=" * 40)
    print(f"  Total possible pairs (no blocking): {total_possible:,}")
    print(f"  Candidate pairs after blocking:     {n_candidates:,}")
    print(f"  Reduction:                          {reduction:.1f}%")
    print(f"  True matches in candidate set:      {true_pairs_in_candidates} / {n_true_matches}")
    print(f"  Blocking recall:                    {recall:.1%}")
    print()
    print("The blocking step eliminates pairs that almost certainly do not match.")
    print("Recall measures what fraction of true matches survive; anything below")
    print("~99% means the blocking strategy is losing matches before scoring begins.")


if __name__ == "__main__":
    true_df  = generate_true_records(n=200, seed=42)
    source_a = make_source_a(true_df)
    source_b = make_source_b(true_df, n_overlap=180, seed=99)

    candidate_pairs = generate_candidate_pairs(source_a, source_b)
    report_blocking_statistics(source_a, source_b, candidate_pairs)
