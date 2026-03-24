"""
10_case_study_chatgpt.py
Chapter 14: Evaluating AI Systems for Federal Use

Case B.3: ChatGPT for agency use -- per-dimension assessment.
Illustrates how to apply the 10-dimension rubric to a general-purpose
AI tool when the director asks to 'evaluate ChatGPT for our statistical work.'

Data governance note: Azure Government OpenAI and AWS GovCloud AI services
provide FedRAMP-authorized paths for agencies needing cloud AI services.
The commercial ChatGPT API remains non-compliant for protected data.
"""

# ── Configuration ─────────────────────────────────────────────────────────────
SCENARIO = """
SCENARIO: Evaluating ChatGPT (commercial API) for Internal Agency Use
======================================================================
Your agency director asks you to 'evaluate ChatGPT for use in
our statistical work.' Apply the NIST-FCSM rubric.

Immediate problem: 'ChatGPT for our statistical work' is not a use case.
The rubric requires a specific task before it can be applied.
The evaluation below assumes a hypothetical general-purpose deployment
and shows why that framing is itself an evaluation failure.
"""

# Per-dimension assessment
ASSESSMENT = [
    {
        "dimension": "Task fit",
        "rating": "UNCLEAR",
        "notes": (
            "ChatGPT is a general-purpose tool. 'For our statistical work' is not "
            "a use case. Evaluation cannot proceed without a specific task definition."
        ),
    },
    {
        "dimension": "Accuracy",
        "rating": "CONTEXT-DEPENDENT",
        "notes": (
            "High for general text tasks; unknown for your specific statistical domain. "
            "Cannot evaluate without a defined task and evaluation dataset."
        ),
    },
    {
        "dimension": "Reproducibility",
        "rating": "POOR",
        "notes": (
            "Non-deterministic at default temperature. "
            "Model updates change behavior without advance notice. "
            "Standard API does not support version pinning of model behavior."
        ),
    },
    {
        "dimension": "Documentation",
        "rating": "PARTIAL",
        "notes": (
            "OpenAI publishes system cards, but these are generic, not task-specific. "
            "No agency-specific model card exists or can be created from vendor materials."
        ),
    },
    {
        "dimension": "Failure modes",
        "rating": "UNDOCUMENTED for agency use",
        "notes": (
            "Known hallucination risks. No task-specific failure analysis available. "
            "Error rates on federal statistical tasks are unknown."
        ),
    },
    {
        "dimension": "Human oversight",
        "rating": "DEPENDS on use",
        "notes": (
            "Depends entirely on how the agency uses it. "
            "If humans accept outputs without review, effectively no oversight. "
            "No built-in override logging."
        ),
    },
    {
        "dimension": "Data governance",
        "rating": "CRITICAL CONCERN",
        "notes": (
            "Commercial API. Data may be used for training unless Enterprise tier. "
            "Not FedRAMP-authorized. "
            "Title 13 data CANNOT be sent to the commercial ChatGPT API. "
            "Azure Government OpenAI (FedRAMP High) and AWS GovCloud AI services "
            "provide compliant paths for agencies needing cloud AI services. "
            "The commercial ChatGPT API remains non-compliant for protected data."
        ),
    },
    {
        "dimension": "Bias and fairness",
        "rating": "PARTIALLY DOCUMENTED",
        "notes": (
            "OpenAI publishes usage policies and some bias evaluations. "
            "Agency-specific bias testing has not been done and cannot be required."
        ),
    },
    {
        "dimension": "Update and drift",
        "rating": "POOR",
        "notes": (
            "Model updates are frequent and not announced in advance. "
            "No version pinning in standard API. "
            "Behavior at time of evaluation may not match behavior at time of use."
        ),
    },
    {
        "dimension": "State Fidelity Validity (SFV)",
        "rating": "NOT APPLICABLE for single-session; POOR for multi-session",
        "notes": (
            "For agentic or multi-session use: no native session management, "
            "no handoff documents, no canonical decision log. "
            "See Chapter 15 for SFV requirements."
        ),
    },
]

RECOMMENDATION = """
Recommendation:
  1. Identify specific use cases first. 'Evaluate ChatGPT' is not evaluable.

  2. For any use involving survey data or agency records:
     - Azure Government OpenAI (FedRAMP authorized) is the compliant path,
       not the commercial ChatGPT API.
     - AWS GovCloud AI services provide an alternative FedRAMP-authorized path.

  3. For tasks not involving protected data:
     - Validate on your specific task, not on OpenAI benchmarks.
     - Establish reproducibility protocol (temperature, version pinning).

  4. Develop agency guidelines for which task types are approved,
     which require legal review, and which are prohibited.
     'The director asked us to use it' is not a governance framework.
"""


def print_chatgpt_evaluation():
    print(SCENARIO)

    print("Per-Dimension Assessment")
    print("=" * 70)
    for item in ASSESSMENT:
        print(f"\n  {item['dimension']:<30} [{item['rating']}]")
        print(f"    {item['notes']}")

    print()
    print(RECOMMENDATION)


if __name__ == "__main__":
    print_chatgpt_evaluation()
