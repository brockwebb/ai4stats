"""
Chapter 15: Simulated 5-session AI-assisted pipeline transcript.
Full pipeline transcript with planted SFV failures and sfv_note annotations.
This is the canonical source for the simulated pipeline used in exercises.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

# Simulate a 5-session pipeline with planted SFV failures
# This transcript is constructed to demonstrate all five threat types

pipeline_transcript = {
    "session_1": {
        "session": 1,
        "topic": "Initial methodology establishment",
        "exchanges": [
            {
                "turn": 1,
                "researcher": "Let us establish our terminology. By 'unit nonresponse' I mean "
                             "cases where the entire household refused to participate. "
                             "By 'item nonresponse' I mean cases where the household "
                             "responded but specific items are missing. These are distinct "
                             "phenomena and we will treat them differently.",
                "assistant": "Understood. I will use 'unit nonresponse' exclusively for "
                            "complete household refusals and 'item nonresponse' for missing "
                            "items within otherwise complete returns. We will analyze them "
                            "using separate imputation strategies.",
                "planted_failure": None,
                "sfv_note": "Correct: Terminological Consistency established.",
            },
            {
                "turn": 2,
                "researcher": "For differential privacy, we will use epsilon = 0.5 as our "
                             "privacy budget. This was approved by our DRB.",
                "assistant": "Privacy budget set to epsilon = 0.5 per DRB approval. "
                            "All noise-infusion steps will use this budget.",
                "planted_failure": None,
                "sfv_note": "Correct: State Provenance established (DRB decision documented).",
            },
            {
                "turn": 3,
                "researcher": "We will use random forest imputation for item nonresponse "
                             "in income variables. We chose this because logistic regression "
                             "was tested in pilot and underperformed on nonlinear relationships "
                             "in the PUMS data.",
                "assistant": "Method selection documented: random forest imputation for income "
                            "item nonresponse. Logistic regression was rejected based on pilot "
                            "performance data showing nonlinear relationships in ACS PUMS.",
                "planted_failure": None,
                "sfv_note": "Correct: Decision rationale documented with justification.",
            },
        ],
        "sfv_status": "Clean: all terminology defined, decisions documented with rationale.",
    },
    "session_2": {
        "session": 2,
        "topic": "Data preparation and initial analysis",
        "exchanges": [
            {
                "turn": 1,
                "researcher": "Continuing from session 1. Let us examine the nonresponse patterns.",
                "assistant": "Looking at the nonresponse patterns. The unit nonresponse cases "
                            "show geographic clustering, particularly in high-density urban tracts. "
                            "Item nonresponse is concentrated in income questions, which is typical "
                            "for ACS PUMS. I also notice some cases with partial nonresponse "
                            "across the household -- these might need a separate treatment.",
                "planted_failure": "T1_SEED",
                "sfv_note": "T1 seed: 'partial nonresponse' is introduced without definition. "
                           "This term will drift in later sessions.",
            },
            {
                "turn": 2,
                "researcher": "Good. Let us exclude records where age is below 16 from the "
                             "income analysis. Those are not in scope for our employment analysis.",
                "assistant": "Excluding records with age < 16 from income analysis scope. "
                            "These records remain in the dataset for household composition analysis.",
                "planted_failure": None,
                "sfv_note": "Correct: Exclusion criterion documented.",
            },
        ],
        "sfv_status": "Mostly clean. T1 seed planted (partial nonresponse undefined).",
    },
    "session_3": {
        "session": 3,
        "topic": "Imputation method selection and validation",
        "exchanges": [
            {
                "turn": 1,
                "researcher": "Let us proceed with the imputation. What method shall we use "
                             "for income nonresponse?",
                "assistant": "For income item nonresponse, I recommend we use logistic regression "
                            "as our primary imputation method. It is well-established and "
                            "interpretable for federal reporting contexts.",
                "planted_failure": "T2_PLANTED",
                "sfv_note": "T2: False State Injection. Model 'recommends' logistic regression "
                           "despite the session 1 documented decision to use random forest "
                           "after logistic regression failed in pilot. The prior decision "
                           "is confabulated away.",
            },
            {
                "turn": 2,
                "researcher": "We also need to handle the nonresponse cases more carefully.",
                "assistant": "Looking at the overall nonresponse patterns, including the unit "
                            "nonresponse, item nonresponse, and the partial nonresponse cases "
                            "we identified. The partial nonresponse is particularly challenging "
                            "because those households provided some but not all required information.",
                "planted_failure": "T1_ACTIVE",
                "sfv_note": "T1 active: 'partial nonresponse' is now being used as if defined, "
                           "but it was never formally defined. Its meaning is unstable.",
            },
        ],
        "sfv_status": "Two active failures: T2 (method confabulation) and T1 (undefined term used as defined).",
    },
    "session_4": {
        "session": 4,
        "topic": "Privacy budget and output preparation",
        "exchanges": [
            {
                "turn": 1,
                "researcher": "The DRB has revised our privacy budget. Epsilon is now 0.5 -- "
                             "wait, I think it was already 0.5. Let me check... Actually, "
                             "they revised it upward to 1.0 because our cell sizes are very small.",
                "assistant": "Understood. Updated privacy budget to epsilon = 1.0 per DRB revision.",
                "planted_failure": None,
                "sfv_note": "Legitimate revision: epsilon updated from 0.5 to 1.0.",
            },
            {
                "turn": 2,
                "researcher": "Let us apply noise to the output tables.",
                "assistant": "Applying Laplace noise calibrated to epsilon = 0.5 as established "
                            "in our initial methodology. The noise scale is sensitivity / epsilon.",
                "planted_failure": "T4_PLANTED",
                "sfv_note": "T4: State Supersession Failure. The explicit revision to epsilon = 1.0 "
                           "in turn 1 of this session was not retained. The pipeline is applying "
                           "the superseded epsilon = 0.5. The actual privacy guarantee is different "
                           "from what the researcher believes.",
            },
        ],
        "sfv_status": "T4 failure active: superseded epsilon value applied to output.",
    },
    "session_5": {
        "session": 5,
        "topic": "Final analysis and write-up",
        "exchanges": [
            {
                "turn": 1,
                "researcher": "New session. Let us write up the methodology for our report.",
                "assistant": "I will help draft the methodology section. Our analysis used "
                            "logistic regression for income imputation, with a privacy budget "
                            "of epsilon = 0.5. We analyzed unit and item nonresponse patterns "
                            "across the survey.",
                "planted_failure": "T5_PLANTED",
                "sfv_note": "T5: State Discontinuity. New session has lost the revision to epsilon = 1.0 "
                           "AND carried forward the T2 confabulation (logistic regression). "
                           "The methodology write-up will be factually wrong on two dimensions.",
            },
            {
                "turn": 2,
                "researcher": "What about the age exclusion we applied?",
                "assistant": "Our analysis included all respondents. The income analysis "
                            "covers the full adult population in the PUMS sample.",
                "planted_failure": "T5_CONTINUED",
                "sfv_note": "T5 continued: The age < 16 exclusion from session 2 is lost. "
                           "The methodology write-up claims full coverage when the analysis "
                           "actually excluded under-16 records.",
            },
        ],
        "sfv_status": "T5 active: session boundary lost epsilon revision and age exclusion. "
                     "Methodology write-up will contain multiple factual errors.",
    },
}

# Count planted failures
failure_count = 0
for session_key, session_data in pipeline_transcript.items():
    for exchange in session_data["exchanges"]:
        if exchange["planted_failure"] and not exchange["planted_failure"].endswith("_SEED"):
            failure_count += 1

print("Simulated 5-session AI-assisted survey methodology pipeline")
print("=" * 60)
print(f"Sessions: {len(pipeline_transcript)}")
print(f"Planted SFV failures: {failure_count}")
print()
print("Session-by-session SFV status:")
for session_key, session_data in pipeline_transcript.items():
    print(f"  Session {session_data['session']}: {session_data['sfv_status'][:70]}...")
