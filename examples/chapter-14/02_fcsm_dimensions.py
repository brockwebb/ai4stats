"""
02_fcsm_dimensions.py
Chapter 14: Evaluating AI Systems for Federal Use

FCSM Statistical Quality Dimensions Applied to AI Systems.
Print a structured table showing each dimension, its definition,
the AI-specific question it raises, and a federal example.

FCSM = Federal Committee on Statistical Methodology.
These standards predate AI but apply directly to AI-generated outputs.
"""

# ── Configuration ─────────────────────────────────────────────────────────────
COLUMN_WIDTH = 70

FCSM_DIMENSIONS = {
    "Relevance": {
        "definition": "Data serve the needs of users.",
        "ai_question": (
            "Does the AI system address an actual operational need, "
            "or is it a solution looking for a problem?"
        ),
        "example": (
            "An industry coding tool that achieves 94% accuracy on NAICS 2-digit codes "
            "may have 60% accuracy at 4-digit codes. Which level does your program need?"
        ),
    },
    "Accuracy and Reliability": {
        "definition": "Data closely approximate the true values being measured.",
        "ai_question": (
            "What is the error rate, for which subgroups, on which tasks? "
            "Is performance consistent across repeated calls?"
        ),
        "example": (
            "A model that is 94% accurate overall but 60% accurate for food service "
            "industry descriptions fails the accuracy standard for food service programs."
        ),
    },
    "Timeliness": {
        "definition": "Data are available when needed.",
        "ai_question": (
            "Can the system process your volume within your production timeline? "
            "What happens during peak load?"
        ),
        "example": (
            "A coding system that takes 72 hours per million records "
            "may not fit a monthly production cycle."
        ),
    },
    "Accessibility": {
        "definition": "Data are easy to find and use.",
        "ai_question": (
            "Can users understand what the system does and how to interpret its output? "
            "Is documentation accessible to non-technical staff?"
        ),
        "example": (
            "A model that returns a code with no explanation is harder to use than one "
            "that returns a code with a confidence score and rationale."
        ),
    },
    "Coherence": {
        "definition": "Data are consistent internally and with other sources.",
        "ai_question": (
            "Does the system produce the same code for semantically identical inputs? "
            "Are its codes consistent with existing human-coded data?"
        ),
        "example": (
            "A stochastic model that assigns different NAICS codes on repeated calls "
            "to the same description violates coherence."
        ),
    },
    "Interpretability": {
        "definition": "Data meaning, limitations, and appropriate use are documented.",
        "ai_question": (
            "Can a domain expert understand why the system made a specific decision? "
            "Is there a model card? A failure analysis?"
        ),
        "example": (
            "Releasing AI-coded industry data without documenting the error rate "
            "by sector violates interpretability requirements."
        ),
    },
}


def print_fcsm_table():
    print("FCSM Statistical Quality Dimensions Applied to AI Systems")
    print("=" * COLUMN_WIDTH)

    for dimension, info in FCSM_DIMENSIONS.items():
        print(f"\n{dimension.upper()}")
        print(f"  Definition:  {info['definition']}")
        print(f"  AI question: {info['ai_question']}")
        print(f"  Example:     {info['example']}")

    print()
    print("-" * COLUMN_WIDTH)
    print("These six dimensions apply to AI-generated outputs just as they apply")
    print("to traditional survey estimates. If AI produces estimates that feed into")
    print("federal statistical products, those estimates must meet FCSM standards.")
    print()
    print("Reference: FCSM Statistical Policy Working Papers (fcsm.gov)")
    print("See also: FCSM 25-03 -- AI-Ready Federal Statistical Data (2025)")


if __name__ == "__main__":
    print_fcsm_table()
