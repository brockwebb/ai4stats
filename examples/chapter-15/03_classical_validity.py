"""
Chapter 15: Classical validity types and their limits.
Displays the four classical validity types, their core questions,
what they assume about the instrument, and how LLM pipelines violate those assumptions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

# Display classical validity types and their instrument assumptions

classical_validity = [
    {
        "type": "Construct validity",
        "question": "Are you measuring what you claim to measure?",
        "instrument_assumption": "The instrument is defined and stable",
        "lLM_violation": "Context buffer changes the operative construct mid-execution",
    },
    {
        "type": "Internal validity",
        "question": "Are causal inferences warranted?",
        "instrument_assumption": "The instrument does not change during measurement",
        "lLM_violation": "Accumulated state mutates; confounders introduced mid-pipeline",
    },
    {
        "type": "External validity",
        "question": "Do findings generalize beyond study conditions?",
        "instrument_assumption": "Instrument behaves consistently across contexts",
        "lLM_violation": "Session restarts produce different operative states from the same prompt",
    },
    {
        "type": "Statistical conclusion validity",
        "question": "Are statistical inferences warranted?",
        "instrument_assumption": "Instrument produces consistent, interpretable measurements",
        "lLM_violation": "Terminology drift means statistical quantities may reference different constructs across sessions",
    },
]

print("Classical validity types and their limits in AI pipelines:")
print("=" * 65)
for v in classical_validity:
    print(f"\n{v['type']}")
    print(f"  Core question:     {v['question']}")
    print(f"  Assumes:           {v['instrument_assumption']}")
    print(f"  LLM pipeline breaks this because:")
    print(f"    {v['lLM_violation']}")
