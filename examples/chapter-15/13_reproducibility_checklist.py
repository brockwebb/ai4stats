"""
Chapter 15: 8-item reproducibility checklist.
Each item includes SFV dimension mapping, pass criterion, and fail consequence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

# 8-item reproducibility checklist

checklist = [
    {
        "item": 1,
        "check": "Is the vocabulary externally defined?",
        "detail": "All domain-specific terms used in the pipeline are defined in a document "
                 "external to the AI context window, not just 'what the model said'.",
        "sfv_dimension": "Terminological Consistency (TC)",
        "pass_criterion": "Yes: a vocabulary document or config file exists and is loaded at session start.",
        "fail_consequence": "T1 (Semantic Drift) risk is unmitigated. Terms will mutate across sessions.",
    },
    {
        "item": 2,
        "check": "Are session boundaries explicitly managed?",
        "detail": "Handoff documents are created at every session end and loaded at every session start. "
                 "Session continuity is explicit, not assumed.",
        "sfv_dimension": "Session Continuity (SC)",
        "pass_criterion": "Yes: a structured handoff document exists for every completed session.",
        "fail_consequence": "T5 (State Discontinuity) is the default. Each new session starts from near-zero.",
    },
    {
        "item": 3,
        "check": "Is there a canonical log of decisions that can be diffed against the pipeline's operative state?",
        "detail": "An external record (not the AI context) documents every methodological decision "
                 "with rationale. This log is the ground truth for reconciliation checks.",
        "sfv_dimension": "State Provenance (SP)",
        "pass_criterion": "Yes: a decision log exists and is updated at every decision point.",
        "fail_consequence": "T2 (False State Injection) failures cannot be detected without an external reference.",
    },
    {
        "item": 4,
        "check": "Are compaction events logged and auditable?",
        "detail": "When the context window is compacted or summarized, this event is logged. "
                 "A post-compaction reconciliation check verifies that no decision rationale was lost.",
        "sfv_dimension": "Compression Fidelity (CF)",
        "pass_criterion": "Yes: compaction events are logged; post-compaction reconciliation is run.",
        "fail_consequence": "T3 (Compression Distortion) is invisible. Rationale is silently lost.",
    },
    {
        "item": 5,
        "check": "Is there a periodic reconciliation protocol?",
        "detail": "At defined intervals (every N turns, at session boundaries, before major analysis steps), "
                 "the pipeline's stated operative state is checked against the canonical log.",
        "sfv_dimension": "State Coherence (SCoh)",
        "pass_criterion": "Yes: a reconciliation schedule is defined and followed.",
        "fail_consequence": "Failures accumulate silently. T4 and T1-T3 drift may not be detected before output.",
    },
    {
        "item": 6,
        "check": "Can an external reviewer trace any output to the specific decisions that produced it?",
        "detail": "For any number, code, or recommendation in the final output, a reviewer can follow "
                 "the provenance chain back to the specific session turn and decision that produced it.",
        "sfv_dimension": "State Provenance (SP)",
        "pass_criterion": "Yes: provenance is documented for all substantive outputs.",
        "fail_consequence": "The pipeline is not auditable. External review (DRB, OMB, OIG) cannot verify methodology.",
    },
    {
        "item": 7,
        "check": "Does the pipeline's evaluation framework include SFV metrics, not just accuracy metrics?",
        "detail": "Evaluation of the pipeline includes at minimum: terminology consistency rate, "
                 "state reconciliation pass rate, and post-compaction divergence for any session "
                 "that included a compaction event.",
        "sfv_dimension": "All sub-dimensions",
        "pass_criterion": "Yes: SFV metrics are tracked and reported alongside accuracy metrics.",
        "fail_consequence": "Accuracy metrics are necessary but not sufficient. A pipeline can be "
                           "accurate on average and still have fatal SFV failures in specific branches.",
    },
    {
        "item": 8,
        "check": "Are NIST AI RMF TEVV practices applied to the pipeline as a whole, not just individual model outputs?",
        "detail": "TEVV (Test, Evaluation, Verification, and Validation) covers the full pipeline: "
                 "data inputs, intermediate pipeline states, session boundaries, and final outputs. "
                 "Individual model output evaluation is not sufficient.",
        "sfv_dimension": "All sub-dimensions",
        "pass_criterion": "Yes: TEVV scope includes pipeline-level state fidelity, not just model-level accuracy.",
        "fail_consequence": "Model-level TEVV may pass while pipeline-level SFV failures corrupt results.",
    },
]

print("Reproducibility Checklist for AI-Assisted Federal Research")
print("=" * 65)
print()
for item in checklist:
    print(f"Item {item['item']}: {item['check']}")
    print(f"  SFV dimension:  {item['sfv_dimension']}")
    print(f"  Pass criterion: {item['pass_criterion']}")
    print()

print("A pipeline that answers 'no' to items 1-3 has unmitigated risk for the most dangerous SFV threats.")
print("A pipeline that answers 'yes' to all 8 items does not guarantee zero SFV failures, but it has")
print("the monitoring infrastructure to detect and correct them before they reach final output.")
