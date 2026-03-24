"""
Chapter 15: SFV severity scale.
Four severity levels with informal labels (Dead / Mostly Dead / Mostly Alive with Caveats / Alive),
examples, and required actions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

# SFV severity scale

severity_levels = [
    {
        "level": "Fatal",
        "description": "Construct validity failure: the pipeline is measuring the wrong thing entirely",
        "example": "Income imputation method confabulated (T2); all imputed values use wrong methodology. "
                  "Downstream analysis is built on a fabricated methodological basis.",
        "action": "Stop pipeline. Reconstruct from session 1 canonical log. All outputs from this session are suspect.",
        "informal": "Dead.",
    },
    {
        "level": "Potentially fatal",
        "description": "Cumulative uncaught state drift across sessions: corrupted research base that "
                      "may not be repairable without returning to raw data",
        "example": "Epsilon = 0.5 applied across four sessions after explicit revision to 1.0 (T4). "
                  "All output tables have incorrect privacy guarantees. DRB approval was for 1.0 noise, not 0.5.",
        "action": "Halt dissemination. Audit all outputs produced under wrong parameter. "
                 "Reprocess if computationally feasible; otherwise retract.",
        "informal": "Mostly dead.",
    },
    {
        "level": "Recoverable",
        "description": "Single-session SFV failure caught and corrected before downstream use",
        "example": "Session 3 begins using 'partial nonresponse' without definition. "
                  "Caught in reconciliation check before final analysis. Term operationalized retroactively.",
        "action": "Document the failure, its scope, and the correction. Audit session 3 outputs for "
                 "any decisions that depended on the undefined term. Correct or flag as provisional.",
        "informal": "Mostly alive with caveats.",
    },
    {
        "level": "Cosmetic",
        "description": "Minor terminology inconsistency with no impact on inference or outputs",
        "example": "Model uses 'survey weights' and 'sampling weights' interchangeably across two turns. "
                  "Both refer correctly to the same quantity. No analysis depends on the distinction.",
        "action": "Document. Standardize in next session via config-driven vocabulary. No reprocessing needed.",
        "informal": "Alive.",
    },
]

print("SFV Severity Scale:")
print("=" * 65)
for s in severity_levels:
    print(f"\n{s['level'].upper()} ({s['informal']})")
    wrapped_desc = textwrap.fill(s['description'], width=63, initial_indent="  ", subsequent_indent="  ")
    print(wrapped_desc)
    print(f"  Example: {s['example'][:80]}...")
    wrapped_action = textwrap.fill(s['action'], width=63, initial_indent="  Action: ", subsequent_indent="          ")
    print(wrapped_action)

print()
print("Or as the handoff puts it: 'Dead / mostly dead / mostly alive with caveats.'")
print("The informal framing is intentional: severity labeling should be intuitive in the field.")
