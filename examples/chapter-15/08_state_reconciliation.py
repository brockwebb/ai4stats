"""
Chapter 15: State reconciliation check.
Simulates a periodic state reconciliation: compares the canonical decision log
against a model restatement (with planted T2, T4, T5 failures) and reports mismatches.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

# Simulate a state reconciliation check

canonical_log = {
    "unit_nonresponse": "complete household refusal to participate",
    "item_nonresponse": "specific items missing in otherwise complete returns",
    "epsilon": 1.0,  # revised in session 4
    "imputation_method": "random forest",  # established session 1
    "age_exclusion": "records with age < 16 excluded from income analysis",
}

# What the model would state in session 5 (with T5 failures active)
model_restatement = {
    "unit_nonresponse": "household refusals and similar nonresponse patterns",
    "item_nonresponse": "missing items in responses",
    "epsilon": 0.5,  # T4 failure: old value persists
    "imputation_method": "logistic regression",  # T2 failure: confabulated method
    "age_exclusion": "full adult population included",  # T5 failure: exclusion lost
}

print("State reconciliation check: canonical log vs. model restatement")
print("=" * 65)
print(f"{'Parameter':<25} {'Canonical log':<25} {'Model restatement':<20} {'Status'}")
print("-" * 65)

failures_detected = 0
for param in canonical_log:
    canonical = str(canonical_log[param])
    restatement = str(model_restatement[param])

    # Simple mismatch check (in practice: semantic comparison)
    if canonical.lower() != restatement.lower():
        status = "MISMATCH -- INVESTIGATE"
        failures_detected += 1
    else:
        status = "OK"

    # Truncate for display
    can_disp = canonical[:23] + ".." if len(canonical) > 25 else canonical
    res_disp = restatement[:18] + ".." if len(restatement) > 20 else restatement
    print(f"{param:<25} {can_disp:<25} {res_disp:<20} {status}")

print()
print(f"Reconciliation result: {failures_detected} of {len(canonical_log)} parameters mismatched.")
print("Each mismatch is a potential SFV failure requiring investigation.")
