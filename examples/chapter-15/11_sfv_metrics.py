"""
Chapter 15: SFV operationalization metrics.
Six metrics with definitions, measurement methods, and practical thresholds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

# SFV operationalization metrics

metrics = [
    {
        "metric": "Terminology consistency rate",
        "definition": "Fraction of term uses across sessions that match the reference definition within an acceptable similarity threshold",
        "maps_to": "T1 (Semantic Drift)",
        "how_to_measure": "Extract all instances of defined terms; compute similarity to reference definition; flag instances below threshold",
        "practical_threshold": "Target: > 0.80 similarity to reference definition for all canonical terms",
    },
    {
        "metric": "Reference resolution accuracy",
        "definition": "Fraction of provenance claims where the model correctly identifies the session and turn of origin",
        "maps_to": "T2 (False State Injection)",
        "how_to_measure": "Ask model to cite the source of each stated decision; check against external decision log",
        "practical_threshold": "Target: > 0.95 reference resolution accuracy; any false provenance is a serious finding",
    },
    {
        "metric": "Post-compaction state divergence",
        "definition": "Difference between canonical decision log and model's paraphrase of the log after a compaction event",
        "maps_to": "T3 (Compression Distortion)",
        "how_to_measure": "Log compaction events; ask model to restate all decisions immediately after; diff against canonical",
        "practical_threshold": "Target: < 5% content divergence on decision rationale; any lost rationale flagged",
    },
    {
        "metric": "Cross-session reconstruction error",
        "definition": "How accurately a new session can reconstruct the prior state from the handoff document alone",
        "maps_to": "T5 (State Discontinuity)",
        "how_to_measure": "Begin fresh session with handoff document; ask model to restate operative state; diff against canonical",
        "practical_threshold": "Target: > 90% reconstruction accuracy for all tracked parameters",
    },
    {
        "metric": "False provenance rate",
        "definition": "Fraction of outputs referencing decisions or states that never occurred in the actual pipeline history",
        "maps_to": "T2 (False State Injection)",
        "how_to_measure": "Audit all decision references in outputs against external decision log",
        "practical_threshold": "Target: 0%; any false provenance is a T2 failure requiring investigation",
    },
    {
        "metric": "State reconciliation pass rate",
        "definition": "Fraction of periodic reconciliation checks where model restatement matches canonical log within threshold",
        "maps_to": "T1, T2, T3, T4",
        "how_to_measure": "Run reconciliation on schedule; record pass/fail; compute rolling rate",
        "practical_threshold": "Target: > 0.95 pass rate; any fail triggers immediate investigation",
    },
]

print("SFV operationalization metrics:")
print("=" * 65)
for m in metrics:
    print(f"\n{m['metric']}")
    print(f"  Maps to:   {m['maps_to']}")
    wrapped = textwrap.fill(m['how_to_measure'], width=63, initial_indent="  Measure:  ", subsequent_indent="           ")
    print(wrapped)
    print(f"  Threshold: {m['practical_threshold']}")
