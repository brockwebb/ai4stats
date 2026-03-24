"""
02_prompt_design.py -- Chapter 12: Prompt Design for Industry Coding

Demonstrates how to construct prompts for LLM-based industry coding.
Covers zero-shot and few-shot prompt structures, prompt versioning,
and the template vs. instance distinction.

This script:
- Defines a reusable build_coding_prompt() function
- Constructs a zero-shot prompt (no examples)
- Constructs a few-shot prompt (3 example pairs)
- Prints both prompts for pedagogical illustration
- Prints a prompt version log schema

Standalone: no external data files required. Run with Python 3.9+.
No LLM API calls are made. Prompts are displayed only.
"""

from datetime import date

# ---------------------------------------------------------------------------
# Prompt template version registry
# ---------------------------------------------------------------------------
# Prompts are code. Track them like code.

PROMPT_REGISTRY = {
    "v1.0": {
        "version": "v1.0",
        "date": "2025-01-15",
        "author": "survey_methods_team",
        "notes": "Initial zero-shot prompt. Accuracy 78% on validation set.",
        "model_tested_on": "gpt-4o-2024-11-20",
    },
    "v1.1": {
        "version": "v1.1",
        "date": "2025-03-01",
        "author": "survey_methods_team",
        "notes": (
            "Added few-shot examples targeting 54 vs 51 confusion. "
            "Accuracy 83% on validation set."
        ),
        "model_tested_on": "gpt-4o-2024-11-20",
    },
}

# ---------------------------------------------------------------------------
# NAICS 2-digit sector reference (embedded in prompt)
# ---------------------------------------------------------------------------

NAICS_SECTORS = """11 - Agriculture, Forestry, Fishing and Hunting
21 - Mining, Quarrying, Oil and Gas Extraction
22 - Utilities
23 - Construction
31-33 - Manufacturing
42 - Wholesale Trade
44-45 - Retail Trade
48-49 - Transportation and Warehousing
51 - Information
52 - Finance and Insurance
53 - Real Estate and Rental and Leasing
54 - Professional, Scientific, and Technical Services
55 - Management of Companies and Enterprises
56 - Administrative and Support and Waste Management Services
61 - Educational Services
62 - Health Care and Social Assistance
71 - Arts, Entertainment, and Recreation
72 - Accommodation and Food Services
81 - Other Services (except Public Administration)
92 - Public Administration"""

# ---------------------------------------------------------------------------
# Prompt construction functions
# ---------------------------------------------------------------------------


def build_coding_prompt(description, classification_scheme="NAICS",
                        few_shot_examples=None, prompt_version="v1.0"):
    """
    Construct an industry coding prompt for an LLM.

    Parameters
    ----------
    description : str
        Free-text industry or occupation description to code.
    classification_scheme : str
        Name of the classification system (default: "NAICS").
    few_shot_examples : list of (str, str) or None
        Optional list of (description, code) example pairs to include.
        If None, produces a zero-shot prompt.
    prompt_version : str
        Version identifier for the prompt template (for logging).

    Returns
    -------
    str
        Formatted prompt string ready to send to an LLM.

    Notes
    -----
    No API call is made. This function constructs the prompt text only.
    The prompt_version argument is used for logging and regression tracking.
    """
    system_instruction = (
        f"You are an expert industry coder for a federal statistical agency. "
        f"Your task is to assign the most appropriate {classification_scheme} "
        f"sector code to the following business or work description.\n\n"
        f"Respond with ONLY the 2-digit sector code and its name, nothing else.\n"
        f"Format: \"XX - Sector Name\"\n\n"
        f"If the description is too vague to code, respond with: \"UNCLEAR\"\n\n"
        f"{classification_scheme} 2-digit sectors:\n"
        f"{NAICS_SECTORS}\n"
    )

    if few_shot_examples:
        system_instruction += "\nExamples:\n"
        for ex_text, ex_code in few_shot_examples:
            system_instruction += f'  Description: "{ex_text}"\n  Code: {ex_code}\n\n'

    system_instruction += f'\nDescription to code: "{description}"\nCode: '
    return system_instruction


# ---------------------------------------------------------------------------
# Demonstration: zero-shot vs. few-shot
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES_V1 = [
    (
        "I work at a grocery store",
        "44-45 - Retail Trade",
    ),
    (
        "I'm an engineer at a software company",
        "51 - Information",
    ),
    (
        "I teach kindergarten at a public school",
        "61 - Educational Services",
    ),
]

FEW_SHOT_EXAMPLES_V2_TARGETED = [
    # Targeted at 54 vs 51 confusion
    (
        "Management consultant helping companies with IT strategy",
        "54 - Professional, Scientific, and Technical Services",
        # Note: primary activity is consulting; the client happens to need IT help
    ),
    (
        "Software engineer at a company that builds accounting software",
        "51 - Information",
        # Note: the EMPLOYER is a software company; occupation does not change NAICS
    ),
    (
        "I run payroll and HR at a regional bank",
        "52 - Finance and Insurance",
        # Note: NAICS codes the employer's industry (banking), not the employee's function
    ),
]

TARGET_DESCRIPTION = "I manage a team at a hospital that handles patient billing"


def print_separator(title):
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


if __name__ == "__main__":
    # --- Zero-shot prompt ---
    print_separator("ZERO-SHOT PROMPT (no examples)")
    zero_shot = build_coding_prompt(
        TARGET_DESCRIPTION,
        few_shot_examples=None,
        prompt_version="v1.0",
    )
    print(zero_shot)
    print()
    print("  [No examples provided. Model must rely on pre-training knowledge only.]")
    print(f"  [Prompt version: v1.0 | Simulated response: '62 - Health Care and Social Assistance']")

    # --- Few-shot prompt (general examples) ---
    print_separator("FEW-SHOT PROMPT v1.1 (3 general examples)")
    few_shot = build_coding_prompt(
        TARGET_DESCRIPTION,
        few_shot_examples=FEW_SHOT_EXAMPLES_V1,
        prompt_version="v1.1",
    )
    print(few_shot)
    print()
    print("  [3 general examples prepended. Helps orient the model to output format.]")
    print(f"  [Prompt version: v1.1 | Simulated response: '62 - Health Care and Social Assistance']")

    # --- Few-shot prompt (targeted at known confusions) ---
    print_separator("FEW-SHOT PROMPT v1.2 (3 targeted examples for 54 vs 51 confusion)")
    targeted_examples = [(ex[0], ex[1]) for ex in FEW_SHOT_EXAMPLES_V2_TARGETED]
    few_shot_targeted = build_coding_prompt(
        "I do cybersecurity work for a consulting firm",
        few_shot_examples=targeted_examples,
        prompt_version="v1.2",
    )
    print(few_shot_targeted)
    print()
    print("  [Targeted examples teach the model the NAICS employer-industry rule.]")
    print("  [Expected response: '54 - Professional, Scientific, and Technical Services']")

    # --- Prompt version log schema ---
    print_separator("PROMPT VERSION LOG SCHEMA (for production logging)")
    log_entry_template = {
        "prompt_template_version": "v1.1",
        "prompt_instance": "<full prompt text here>",
        "description_input": "<raw respondent text>",
        "model_id": "gpt-4o-2024-11-20",
        "temperature": 0,
        "raw_response": "<model output string>",
        "parsed_code": "62",
        "confidence_score": 0.91,
        "timestamp_utc": "2025-06-15T14:32:00Z",
        "batch_id": "cps_2025q2_batch_0014",
        "reviewer_id": None,  # None if auto-accepted; coder ID if human reviewed
    }
    import pprint
    print("  Each coding call should produce one log entry:")
    print()
    pprint.pprint(log_entry_template, indent=4)

    # --- Prompt registry ---
    print_separator("PROMPT VERSION REGISTRY (v1.0 and v1.1 shown)")
    for version, meta in PROMPT_REGISTRY.items():
        print(f"\n  Version: {version}")
        for k, v in meta.items():
            print(f"    {k}: {v}")
