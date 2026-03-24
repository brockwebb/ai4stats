"""
Chapter 15: Countermeasure coverage heatmap.
Coverage matrix (countermeasures x threats) visualized as a heatmap.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

fig, ax = plt.subplots(figsize=(10, 5))

threat_labels = ["T1\nSemantic\nDrift", "T2\nFalse State\nInjection",
                "T3\nCompression\nDistortion", "T4\nState Supersession\nFailure",
                "T5\nState\nDiscontinuity"]
cm_labels = [
    "Config-driven vocab",
    "Graph-backed ontology",
    "TEVV validation loops",
    "Handoff documents",
    "Documentation-as-traceability",
    "Multi-model triangulation",
    "Periodic state reconciliation",
]

# Coverage matrix: rows = countermeasures, cols = threats (T1-T5)
coverage = np.array([
    [1, 0, 0, 0, 0],  # Config-driven vocab: T1
    [1, 1, 0, 0, 0],  # Graph-backed ontology: T1, T2
    [0, 1, 1, 0, 0],  # TEVV loops: T2, T3
    [0, 0, 0, 0, 1],  # Handoff docs: T5
    [0, 0, 1, 1, 0],  # Documentation-as-traceability: T3, T4
    [0, 1, 0, 0, 0],  # Multi-model triangulation: T2
    [1, 1, 1, 1, 0],  # Periodic reconciliation: T1, T2, T3, T4
])

# Display as heatmap
im = ax.imshow(coverage, cmap="Blues", aspect="auto", vmin=0, vmax=1.5)

ax.set_xticks(range(5))
ax.set_xticklabels(threat_labels, fontsize=9)
ax.set_yticks(range(len(cm_labels)))
ax.set_yticklabels(cm_labels, fontsize=9)

for r in range(coverage.shape[0]):
    for c in range(coverage.shape[1]):
        if coverage[r, c] == 1:
            ax.text(c, r, "yes", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

ax.set_title("Engineering countermeasures vs. SFV threats", fontsize=11, pad=12)
ax.set_xlabel("Threat", fontsize=10)
ax.set_ylabel("Countermeasure", fontsize=10)

plt.tight_layout()
plt.savefig("../assets/diagrams/chapter_15_countermeasures.png", dpi=120, bbox_inches="tight")
plt.show()

print()
# Count coverage
threat_coverage = coverage.sum(axis=0)
print("Threat coverage by number of available countermeasures:")
for i, (t, count) in enumerate(zip(["T1", "T2", "T3", "T4", "T5"], threat_coverage)):
    print(f"  {t}: {int(count)} countermeasure(s) available")
