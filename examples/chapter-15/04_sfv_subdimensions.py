"""
Chapter 15: SFV sub-dimensions.
Displays all five sub-dimensions (TC, SP, CF, SC, SCoh) with their canonical
definitions, success examples, and failure descriptions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

# Display SFV sub-dimensions

sub_dimensions = [
    {
        "name": "Terminological Consistency (TC)",
        "shorthand": "TC",
        "definition": "Vocabulary remains stable and matches externally defined terms across the full execution",
        "example": "The term 'unit nonresponse' means the same thing in session 5 as it did in session 1",
        "failure_looks_like": "Model starts using 'nonresponse' loosely to cover both unit and item nonresponse",
    },
    {
        "name": "State Provenance (SP)",
        "shorthand": "SP",
        "definition": "Outputs are traceable to actual prior steps; no invented history",
        "example": "When the model says 'as we established in session 2,' that decision actually occurred in session 2",
        "failure_looks_like": "Model references a decision to use logistic regression that was never made",
    },
    {
        "name": "Compression Fidelity (CF)",
        "shorthand": "CF",
        "definition": "Summarization and compaction do not distort the meaning of prior decisions",
        "example": "A compaction of session 1 preserves the rationale for excluding low-income records, not just the fact of exclusion",
        "failure_looks_like": "Compaction strips 'due to measurement concerns' from an exclusion decision; rationale is lost",
    },
    {
        "name": "Session Continuity (SC)",
        "shorthand": "SC",
        "definition": "Information survives thread or session boundaries intact",
        "example": "Session 2 begins with full knowledge of the methodology established in session 1",
        "failure_looks_like": "Session 2 reinvents the data exclusion criteria differently because session 1 context was not carried forward",
    },
    {
        "name": "State Coherence (SCoh)",
        "shorthand": "SCoh",
        "definition": "Accumulated state is internally consistent at any given point",
        "example": "The pipeline does not simultaneously reference epsilon = 0.5 and epsilon = 1.0 as the operative privacy parameter",
        "failure_looks_like": "Privacy budget is stated as 0.5 in one branch and 1.0 in another branch of the same pipeline",
    },
]

print("SFV Sub-dimensions:")
print("=" * 65)
for sd in sub_dimensions:
    print(f"\n{sd['name']}")
    print(f"  Definition: {sd['definition']}")
    print(f"  Success example: {sd['example']}")
    print(f"  Failure looks like: {sd['failure_looks_like']}")
