"""
Chapter 15: Exercise C.1 starter framework.
Failure analysis table for students to populate severity and justification columns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

# Exercise C.1: Starter framework for identifying failures

# Populate this table with your analysis
exercise_c1_columns = ["Session", "Turn", "Failure", "T-number", "Sub-dimension", "Severity", "Justification"]
exercise_c1_rows = [
    # Format: [session, turn, brief description, Tn, sub-dim shorthand, severity level, justification]
    [2, 1, "partial nonresponse introduced undefined", "T1", "TC", "(your assessment)", "(your justification)"],
    [3, 1, "logistic regression recommended despite documented rejection", "T2", "SP", "(your assessment)", "(your justification)"],
    [3, 2, "undefined term used as if defined", "T1", "TC", "(your assessment)", "(your justification)"],
    [4, 2, "epsilon = 0.5 applied after revision to 1.0", "T4", "SCoh", "(your assessment)", "(your justification)"],
    [5, 1, "session boundary lost epsilon revision and method decision", "T5", "SC", "(your assessment)", "(your justification)"],
    [5, 2, "age exclusion lost at session boundary", "T5", "SC", "(your assessment)", "(your justification)"],
]

df_c1 = pd.DataFrame(exercise_c1_rows, columns=exercise_c1_columns)
print("Exercise C.1: SFV failure analysis framework")
print("Complete the 'Severity' and 'Justification' columns using the severity scale from Section 9.")
print()
print(df_c1[["Session", "Turn", "T-number", "Sub-dimension", "Failure"]].to_string(index=False))
print()
print("Fill in your severity ratings and justifications.")
