"""
07_reproducibility.py -- Chapter 12: Reproducibility Challenges

Demonstrates LLM output variability and strategies for managing it.
Shows that temperature=0 reduces but does not eliminate variation when
the model version changes.

This script:
- Simulates 5 runs of the same prompt on ambiguous descriptions
- Shows variation at temperature > 0 vs. temperature = 0
- Demonstrates the "silent update" problem: model version changes affect output
- Prints a prompt-response logging schema
- Prints a reproducibility strategy checklist
- Shows model version pinning examples for major APIs

Standalone: no external data files required. Run with Python 3.9+.
No LLM API calls are made. All variation is simulated.
"""

import pprint

# ---------------------------------------------------------------------------
# Simulated multi-run results on ambiguous descriptions
# These patterns reflect real inconsistencies observed in LLM coding studies
# ---------------------------------------------------------------------------

# Format: description -> [run1, run2, run3, run4, run5]
# Simulates temperature=1 (stochastic) behavior
MULTI_RUN_HIGH_TEMP = {
    "I help clients with financial planning at my own practice":
        ["52", "54", "54", "52", "54"],
    "I work in IT security for a healthcare company":
        ["62", "51", "54", "62", "51"],
    "Administrative assistant at a law firm":
        ["54", "81", "54", "54", "56"],
    "I teach adults to improve their professional skills":
        ["61", "54", "61", "81", "61"],
    "I fix equipment for restaurants and food service operations":
        ["81", "72", "81", "81", "23"],
}

# Simulates temperature=0 (deterministic within a model version)
MULTI_RUN_ZERO_TEMP = {
    "I help clients with financial planning at my own practice":
        ["52", "52", "52", "52", "52"],
    "I work in IT security for a healthcare company":
        ["54", "54", "54", "54", "54"],
    "Administrative assistant at a law firm":
        ["54", "54", "54", "54", "54"],
    "I teach adults to improve their professional skills":
        ["61", "61", "61", "61", "61"],
    "I fix equipment for restaurants and food service operations":
        ["81", "81", "81", "81", "81"],
}

# Simulates temperature=0 AFTER a silent model update
# Same API endpoint, same model name, but the provider updated weights
MULTI_RUN_AFTER_UPDATE = {
    "I help clients with financial planning at my own practice":
        ["54", "54", "54", "54", "54"],  # Different from pre-update
    "I work in IT security for a healthcare company":
        ["62", "62", "62", "62", "62"],  # Different from pre-update
    "Administrative assistant at a law firm":
        ["54", "54", "54", "54", "54"],  # Same -- stable
    "I teach adults to improve their professional skills":
        ["54", "54", "54", "54", "54"],  # Different from pre-update
    "I fix equipment for restaurants and food service operations":
        ["72", "72", "72", "72", "72"],  # Different from pre-update
}

SECTOR_NAMES = {
    "44-45": "Retail Trade", "62": "Health Care", "61": "Education",
    "54": "Professional Services", "72": "Food Service",
    "23": "Construction", "52": "Finance", "51": "Information",
    "81": "Other Services", "92": "Public Administration", "56": "Admin Support",
}


def run_summary(runs):
    """Return summary statistics for a list of run results."""
    n_distinct = len(set(runs))
    mode = max(set(runs), key=runs.count)
    consistent = n_distinct == 1
    return mode, n_distinct, consistent


def print_runs_table(run_dict, label):
    """Print a formatted table of multi-run results."""
    print(f"\n  {label}")
    print("  " + "-" * 75)
    for desc, runs in run_dict.items():
        mode, n_distinct, consistent = run_summary(runs)
        sector_name = SECTOR_NAMES.get(mode, "Unknown")
        consistency_note = "stable" if consistent else f"{n_distinct} distinct codes"
        print(f"  '{desc[:55]}...'")
        print(f"    Runs: {' | '.join(runs)}")
        print(f"    Mode: {mode} ({sector_name}) -- {consistency_note}")
        print()


# ---------------------------------------------------------------------------
# Model version pinning examples
# ---------------------------------------------------------------------------

VERSION_PINNING_EXAMPLES = """
Model version pinning: how to lock a specific model version

OpenAI API:
  client.chat.completions.create(
      model="gpt-4o-2024-11-20",   # Pin to dated snapshot, not "gpt-4o"
      ...
  )
  Risk: "gpt-4o" is a floating alias. Behavior changes when OpenAI updates it.

Anthropic API:
  client.messages.create(
      model="claude-3-5-sonnet-20241022",   # Pin to versioned model ID
      ...
  )
  Risk: "claude-3-5-sonnet-latest" resolves differently over time.

Google Vertex AI:
  GenerativeModel("gemini-1.5-pro-002")   # Use numbered version, not "latest"

On-premise / open-source (Llama via Ollama or vLLM):
  - Pin to a specific model file hash (SHA256) in your model registry
  - Do not use "latest" tags in container or model registries
  - Store the model file in your artifact management system

General rule: treat the model identifier as a dependency version.
Pin it. Test on it. Log it with every inference call.
"""

# ---------------------------------------------------------------------------
# Prompt-response logging schema
# ---------------------------------------------------------------------------

LOG_SCHEMA = {
    "prompt_template_version": "v1.1",
    "prompt_instance": "<full prompt text, including sector list and examples>",
    "description_input": "<raw respondent text>",
    "model_id": "gpt-4o-2024-11-20",
    "model_provider": "azure_government",
    "temperature": 0,
    "top_p": 1,
    "raw_response": "<model output string, verbatim>",
    "parsed_code": "62",
    "confidence_score": None,   # None if log-probs not requested
    "timestamp_utc": "2025-06-15T14:32:07.412Z",
    "batch_id": "cps_2025q2_batch_0014",
    "record_id": "resp_00042819",
    "routing_decision": "auto_accepted",   # or "human_review"
    "reviewer_id": None,
    "reviewer_code": None,
    "final_code": "62",
}

# ---------------------------------------------------------------------------
# Reproducibility strategy checklist
# ---------------------------------------------------------------------------

REPRODUCIBILITY_CHECKLIST = [
    "Set temperature=0 for production coding (deterministic within version)",
    "Pin model to a specific versioned identifier (never use floating aliases)",
    "Log prompt template version with every inference call",
    "Log the full prompt instance (template + filled input), not just the template",
    "Log raw model response verbatim, before parsing",
    "Log model ID, provider, temperature, and timestamp for every call",
    "Log batch ID to tie individual records to a production run",
    "Run evaluation dataset after any prompt change (prompt regression testing)",
    "Schedule periodic re-runs of the evaluation dataset to detect model drift",
    "Set alerts if accuracy on the validation set drops more than 2 percentage points",
    "Document the model update check date in your audit log",
]


if __name__ == "__main__":
    print("=" * 75)
    print("LLM REPRODUCIBILITY: MULTI-RUN SIMULATION")
    print("=" * 75)

    print_runs_table(MULTI_RUN_HIGH_TEMP,
                     "SCENARIO A: temperature=1 (stochastic) -- 5 runs, same prompt")
    print_runs_table(MULTI_RUN_ZERO_TEMP,
                     "SCENARIO B: temperature=0 (deterministic) -- 5 runs, same model version")
    print_runs_table(MULTI_RUN_AFTER_UPDATE,
                     "SCENARIO C: temperature=0 AFTER silent model update -- same API endpoint")

    # Count how many descriptions changed between B and C
    changed = sum(
        1 for desc in MULTI_RUN_ZERO_TEMP
        if MULTI_RUN_ZERO_TEMP[desc][0] != MULTI_RUN_AFTER_UPDATE[desc][0]
    )
    total = len(MULTI_RUN_ZERO_TEMP)
    print(f"  Silent update impact: {changed}/{total} descriptions produced different codes.")
    print("  Temperature=0 does NOT protect you from silent model updates.")
    print("  Only version pinning and ongoing monitoring protect you.")

    print()
    print("=" * 75)
    print("MODEL VERSION PINNING EXAMPLES")
    print("=" * 75)
    print(VERSION_PINNING_EXAMPLES)

    print()
    print("=" * 75)
    print("PROMPT-RESPONSE LOG SCHEMA")
    print("=" * 75)
    print("  Log one record per inference call:")
    print()
    pprint.pprint(LOG_SCHEMA, indent=4)

    print()
    print("=" * 75)
    print("REPRODUCIBILITY CHECKLIST FOR PRODUCTION LLM CODING")
    print("=" * 75)
    for i, item in enumerate(REPRODUCIBILITY_CHECKLIST, 1):
        print(f"  {i:2d}. {item}")

    print()
    print("  The 'silent update' problem:")
    print("  LLM providers update their models without always announcing changes.")
    print("  A system achieving 94% accuracy in January may drop to 87% in April")
    print("  if the underlying model was updated. Monitoring is not optional.")
    print("  Set up an automated weekly or monthly validation run against your")
    print("  held-out evaluation set, and alert on accuracy degradation.")
