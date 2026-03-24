"""
Chapter 15: Three-layer reproducibility problem.
Illustrates the three layers of the AI reproducibility problem,
with SFV markers identifying which layer SFV addresses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

# Illustrate the three layers of the reproducibility problem

layers = {
    "Layer 1: Stochastic outputs": {
        "description": "Same prompt -> different outputs across runs",
        "classical_solution": "Set temperature=0, seed random state",
        "limitation": "Partial fix; many APIs do not guarantee determinism",
        "addressed_by_sfv": False,
    },
    "Layer 2: Prompt sensitivity": {
        "description": "Minor prompt variation -> substantially different outputs",
        "classical_solution": "Version prompts; test prompt stability",
        "limitation": "Helps but does not address accumulated state",
        "addressed_by_sfv": False,
    },
    "Layer 3: State accumulation failures": {
        "description": "Accumulated context degrades, drifts, or is lost",
        "classical_solution": "No classical solution exists",
        "limitation": "This is the SFV problem; requires a new validity framework",
        "addressed_by_sfv": True,
    },
}

print("Three layers of the AI reproducibility problem:")
print("=" * 60)
for layer, details in layers.items():
    sfv_marker = " <-- SFV addresses this" if details["addressed_by_sfv"] else ""
    print(f"\n{layer}{sfv_marker}")
    print(f"  Problem:   {details['description']}")
    print(f"  Partial fix: {details['classical_solution']}")
    print(f"  Limit:     {details['limitation']}")
