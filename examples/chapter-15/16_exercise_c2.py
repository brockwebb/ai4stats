"""
Chapter 15: Exercise C.2 starter framework.
Countermeasure mapping for failures identified in Exercise C.1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

# Exercise C.2: Starter -- countermeasure mapping

exercise_c2 = {
    "T1 (session 2, turn 1 seed)": {
        "prevention_possible": True,
        "preventive_countermeasure": "(your answer: which countermeasure?)",
        "earliest_detection": "(your answer: which detection strategy?)",
    },
    "T2 (session 3, turn 1)": {
        "prevention_possible": True,
        "preventive_countermeasure": "(your answer)",
        "earliest_detection": "(your answer)",
    },
    "T4 (session 4, turn 2)": {
        "prevention_possible": True,
        "preventive_countermeasure": "(your answer)",
        "earliest_detection": "(your answer)",
    },
    "T5 (session 5)": {
        "prevention_possible": True,
        "preventive_countermeasure": "(your answer)",
        "minimum_recovery_path": "(your answer: what must be reconstructed?)",
    },
}

print("Exercise C.2: Countermeasure analysis framework")
for failure, analysis in exercise_c2.items():
    print(f"\n{failure}")
    for k, v in analysis.items():
        print(f"  {k}: {v}")
