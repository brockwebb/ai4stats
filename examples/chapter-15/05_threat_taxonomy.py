"""
Chapter 15: SFV threat taxonomy (T1-T5).
Displays the five canonical threats with their mechanisms, detection difficulty,
and severity ranges.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

# SFV threat taxonomy overview

threats = [
    {
        "number": "T1",
        "name": "Semantic Drift",
        "mechanism": "Terminology mutates across turns or sessions without explicit redefinition",
        "detection_difficulty": "Moderate -- output is fluent but terms mean different things",
        "severity_range": "Cosmetic to potentially fatal depending on how central the drifted term is",
    },
    {
        "number": "T2",
        "name": "False State Injection",
        "mechanism": "System confabulates 'memory' of decisions or agreements never established",
        "detection_difficulty": "High -- false claims reference plausible pipeline events",
        "severity_range": "Potentially fatal -- analysis proceeds on fabricated methodological basis",
    },
    {
        "number": "T3",
        "name": "Compression Distortion",
        "mechanism": "Context window compaction silently strips nuance or alters meaning of prior decisions",
        "detection_difficulty": "High -- compaction is automatic; researcher may not know it occurred",
        "severity_range": "Recoverable to potentially fatal depending on what rationale is lost",
    },
    {
        "number": "T4",
        "name": "State Supersession Failure",
        "mechanism": "Outdated information persists and influences output despite being explicitly superseded",
        "detection_difficulty": "Moderate -- requires tracking which decisions were revised",
        "severity_range": "Potentially fatal -- analysis applies obsolete parameters as if current",
    },
    {
        "number": "T5",
        "name": "State Discontinuity",
        "mechanism": "Session boundary loss drops accumulated context; new session operates on partial history",
        "detection_difficulty": "Moderate -- visible as inconsistency across sessions if checked",
        "severity_range": "Potentially fatal -- entire prior methodology may be lost or reinvented",
    },
]

print("SFV Threat Taxonomy:")
print("=" * 65)
for t in threats:
    print(f"\n{t['number']}: {t['name']}")
    print(f"  Mechanism: {t['mechanism']}")
    print(f"  Detection: {t['detection_difficulty']}")
    print(f"  Severity:  {t['severity_range']}")
