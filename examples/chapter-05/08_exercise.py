"""
Chapter 5 Example 8: Admin Record Linkage Exercise and Solution
===============================================================

Activity setup and full solution for the in-chapter exercise.

Scenario
--------
You receive two administrative extracts for the same county:
  - admin_a: 150-record public assistance roll (clean)
  - admin_b: 180-record healthcare enrollment file (noisy; 150 overlap + 30 new)

Your task: link the two files to identify the 150 people who appear in both.

Exercise questions (prose, not code)
-------------------------------------
1. You receive a linked file with this cluster size distribution:
   [1: 180 records, 2: 52 clusters, 3: 8 clusters, 4+: 3 clusters].
   Which clusters would you flag for manual review? Why?

2. The match rate for Hispanic surnames is 14% lower than for non-Hispanic
   surnames. What might explain this? What would you recommend to the
   program manager?

3. Your blocking strategy reduces 40,000 possible pairs to 1,200 candidates.
   The match rate is 82%. Estimate how many true matches were missed by the
   blocking step.

4. Optional: Run the full pipeline below and compare your solution to the
   reference implementation.

Solution to Q3:
   - 1,200 candidates x 82% match rate = 984 accepted matches
   - The total possible true matches in this scenario = 150
   - If recall within the candidate set is ~95%, then ~142 of 150 true
     matches are in the 1,200 candidates
   - Approximately 8 true matches were eliminated by blocking
   - This estimate depends on blocking recall, which can be measured against
     a gold standard or estimated from sampling

Requirements: numpy, pandas, networkx, scikit-learn
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from difflib import SequenceMatcher
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
_m3 = import_module("03_synthetic_records")
_m4 = import_module("04_blocking")
_m5 = import_module("05_comparison_scoring")

add_noise       = _m3.add_noise
add_date_noise  = _m3.add_date_noise
FIRST_NAMES     = _m3.FIRST_NAMES
LAST_NAMES      = _m3.LAST_NAMES
soundex         = _m4.soundex
dob_year        = _m4.dob_year
string_similarity = _m5.string_similarity
dob_match         = _m5.dob_match


# ---------------------------------------------------------------------------
# Activity dataset generation
# ---------------------------------------------------------------------------

def build_exercise_datasets(n_match=150, n_b_extra=30, seed=2025):
    """
    Build admin_a and admin_b datasets for the linkage exercise.

    Parameters
    ----------
    n_match : int
        Number of people who appear in both files (true matches).
    n_b_extra : int
        Number of additional records in admin_b with no match in admin_a.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (admin_a DataFrame, admin_b DataFrame, true_match_pairs list of (id_a, id_b))
    """
    rng = np.random.default_rng(seed)

    names_a, names_b = [], []
    dobs_a,  dobs_b  = [], []
    ids_a,   ids_b   = [], []
    true_pairs       = []

    for i in range(n_match):
        fname = rng.choice(FIRST_NAMES)
        lname = rng.choice(LAST_NAMES)
        name_clean = f"{fname} {lname}"
        dob_clean  = (f"{rng.integers(1950, 1995)}-"
                      f"{rng.integers(1, 13):02d}-"
                      f"{rng.integers(1, 29):02d}")

        names_a.append(name_clean)
        dobs_a.append(dob_clean)
        ids_a.append(f"A{i:03d}")

        names_b.append(add_noise(name_clean, p_typo=0.20, p_abbrev=0.25, rng=rng))
        dobs_b.append(add_date_noise(dob_clean, p_transpose=0.08, p_year_off=0.05, rng=rng))
        ids_b.append(f"B{i:03d}")
        true_pairs.append((f"A{i:03d}", f"B{i:03d}"))

    # Extra records in B with no corresponding record in A
    for j in range(n_b_extra):
        fname = rng.choice(FIRST_NAMES)
        lname = rng.choice(LAST_NAMES)
        names_b.append(f"{fname} {lname}")
        dobs_b.append(
            f"{rng.integers(1950, 1995)}-{rng.integers(1, 13):02d}-01"
        )
        ids_b.append(f"BX{j:02d}")

    admin_a = pd.DataFrame({"id": ids_a, "name": names_a, "dob": dobs_a})
    admin_b = pd.DataFrame({"id": ids_b, "name": names_b, "dob": dobs_b})
    return admin_a, admin_b, true_pairs


# ---------------------------------------------------------------------------
# Full pipeline solution
# ---------------------------------------------------------------------------

def run_full_pipeline(admin_a, admin_b, true_pairs):
    """
    Full record linkage pipeline: blocking -> comparison -> scoring -> clustering.

    Parameters
    ----------
    admin_a : pd.DataFrame
    admin_b : pd.DataFrame
    true_pairs : list of (str, str)
        Known ground-truth match pairs for evaluation.

    Returns
    -------
    dict with pipeline results
    """
    # Step 1: Blocking
    admin_a = admin_a.copy()
    admin_b = admin_b.copy()
    admin_a["soundex"] = admin_a["name"].apply(soundex)
    admin_a["year"]    = admin_a["dob"].apply(dob_year)
    admin_b["soundex"] = admin_b["name"].apply(soundex)
    admin_b["year"]    = admin_b["dob"].apply(dob_year)

    cands = set()
    for key in admin_a["soundex"].unique():
        for ai in admin_a[admin_a["soundex"] == key].index:
            for bi in admin_b[admin_b["soundex"] == key].index:
                cands.add((ai, bi))
    for key in admin_a["year"].unique():
        for ai in admin_a[admin_a["year"] == key].index:
            for bi in admin_b[admin_b["year"] == key].index:
                cands.add((ai, bi))

    print(f"Step 1 - Blocking")
    print(f"  Total possible pairs:    {len(admin_a) * len(admin_b):,}")
    print(f"  Candidate pairs:         {len(cands):,}")
    reduction = (1 - len(cands) / (len(admin_a) * len(admin_b))) * 100
    print(f"  Reduction:               {reduction:.1f}%")

    # Step 2: Comparison features + ground truth
    true_pair_set = set(true_pairs)
    comp_rows = []
    for ai, bi in cands:
        ra = admin_a.loc[ai]
        rb = admin_b.loc[bi]
        ns = string_similarity(ra["name"], rb["name"])
        ds = dob_match(ra["dob"], rb["dob"])
        is_m = int((ra["id"], rb["id"]) in true_pair_set)
        comp_rows.append({
            "id_a": ra["id"], "id_b": rb["id"],
            "name_sim": ns, "dob_sim": ds, "is_match": is_m,
        })
    cdf = pd.DataFrame(comp_rows)

    print(f"\nStep 2 - Comparison features")
    print(f"  True matches in candidates: {cdf['is_match'].sum()} / {len(true_pairs)}")
    blocking_recall = cdf["is_match"].sum() / len(true_pairs)
    print(f"  Blocking recall:            {blocking_recall:.1%}")

    # Step 3: Simple rule-based classifier (threshold approach)
    cdf["predicted"] = ((cdf["name_sim"] >= 0.60) & (cdf["dob_sim"] >= 0.50)).astype(int)
    accepted = cdf[cdf["predicted"] == 1]
    tp = (accepted["is_match"] == 1).sum()
    fp = (accepted["is_match"] == 0).sum()
    fn = cdf["is_match"].sum() - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"\nStep 3 - Rule-based scoring (name_sim >= 0.60 AND dob_sim >= 0.50)")
    print(f"  Accepted matches:  {len(accepted)}")
    print(f"  True positives:    {tp}")
    print(f"  False positives:   {fp}")
    print(f"  False negatives:   {fn}")
    print(f"  Precision:         {precision:.3f}")
    print(f"  Recall:            {recall:.3f}")

    # Step 4: Linkage graph
    G_act = nx.Graph()
    G_act.add_nodes_from(admin_a["id"], source="A")
    G_act.add_nodes_from(admin_b["id"], source="B")
    for _, row in accepted.iterrows():
        G_act.add_edge(row["id_a"], row["id_b"])

    # Step 5: Cluster analysis
    comps = list(nx.connected_components(G_act))
    dist  = Counter([len(c) for c in comps])

    print(f"\nStep 4-5 - Linkage graph and clusters")
    print(f"  Cluster size distribution:")
    for sz, cnt in sorted(dist.items()):
        print(f"    Size {sz}: {cnt} clusters")
    size2 = dist.get(2, 0)
    print(f"\n  Clusters of size 2 (expected {len(true_pairs)}): {size2}")
    print(f"  Recovery rate: {size2 / len(true_pairs):.1%}")

    return {
        "n_candidates":      len(cands),
        "blocking_recall":   blocking_recall,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "cluster_dist": dict(dist),
        "n_size2": size2,
    }


if __name__ == "__main__":
    admin_a, admin_b, true_pairs = build_exercise_datasets(
        n_match=150, n_b_extra=30, seed=2025
    )

    print("Exercise datasets")
    print("=" * 40)
    print(f"  admin_a: {len(admin_a)} records (public assistance roll)")
    print(f"  admin_b: {len(admin_b)} records (healthcare enrollment)")
    print(f"  True matches: {len(true_pairs)}")
    print(f"\nSample from admin_a (clean):")
    print(admin_a.head(3).to_string(index=False))
    print(f"\nSample from admin_b (noisy):")
    print(admin_b.head(3).to_string(index=False))
    print()

    results = run_full_pipeline(admin_a, admin_b, true_pairs)
