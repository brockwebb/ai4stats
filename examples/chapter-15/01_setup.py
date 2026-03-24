"""
Chapter 15: Capstone - Reproducible AI-Assisted Research
Setup: imports, seed, chapter identification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

# Reproducibility
np.random.seed(42)

print("Chapter 15: Capstone - Reproducible AI-Assisted Research")
print("=" * 55)
print()
print("Core concept: State Fidelity Validity (SFV)")
print("  The degree to which an AI-assisted pipeline preserves")
print("  the accuracy and integrity of its accumulated state")
print("  across sequential operations.")
