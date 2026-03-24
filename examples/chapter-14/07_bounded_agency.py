"""
07_bounded_agency.py
Chapter 14: Evaluating AI Systems for Federal Use

Bounded agency: three levels of human involvement.
Display each level with a definition, appropriate context,
federal example, and design implications.

Cross-reference: Chapter 13 covers bounded agency design
principles in depth, including the autonomy dial.
"""

# ── Configuration ─────────────────────────────────────────────────────────────
DIVIDER_WIDTH = 65

LEVELS = [
    {
        "level": 1,
        "name": "HUMAN-IN-THE-LOOP",
        "definition": "Human reviews every AI output before any action is taken.",
        "when": "High stakes, low volume, novel situations.",
        "example": (
            "AI suggests a NAICS code; human coder approves before logging. "
            "Appropriate for new program launches or unusual industry categories."
        ),
        "implication": (
            "Highest quality assurance. Most labor-intensive. "
            "Preserves full human accountability for every decision."
        ),
    },
    {
        "level": 2,
        "name": "HUMAN-ON-THE-LOOP",
        "definition": (
            "AI operates automatically; human monitors and can intervene."
        ),
        "when": "High volume, moderate stakes, established accuracy baseline.",
        "example": (
            "AI codes automatically; human reviews flagged low-confidence cases. "
            "Appropriate after a validated pilot demonstrates acceptable accuracy."
        ),
        "implication": (
            "Scales well. Requires confidence-based routing and drift monitoring. "
            "Override rate is a health metric: zero overrides may mean humans "
            "have stopped reading the AI's output."
        ),
    },
    {
        "level": 3,
        "name": "HUMAN-OUT-OF-THE-LOOP",
        "definition": "AI operates fully autonomously with no human review step.",
        "when": "Low stakes, reversible actions, demonstrated reliability.",
        "example": (
            "For federal statistical production: RARELY appropriate. "
            "May apply to internal formatting or metadata tasks, not to coded outputs."
        ),
        "implication": (
            "Fastest and cheapest. Highest risk. "
            "'The AI decided' is not an acceptable explanation to a "
            "Congressional inquiry or an OMB audit. "
            "Require documented justification before approving this level."
        ),
    },
]

FEDERAL_CONTEXT = """
Federal Context Implications:
  - OMB, Congress, and the public expect human accountability for agency decisions.
  - 'The AI decided' is not an acceptable explanation to a Congressional inquiry.
  - Human override must be easy, visible, and logged.
  - Override rates are a health metric: if humans never override, they may have
    stopped reading the AI's outputs (automation bias).
  - The appropriate level depends on the stakes of the specific decision,
    not on what is technologically possible.
"""

DESIGN_IMPLICATION = """
Core Design Implication:
  A bounded agency system surfaces the AI's reasoning, not just its conclusion.

  BETTER:
    'This description was coded as NAICS 54 (Professional Services) because
     the phrase "management consulting" matches sector 54 patterns with 0.87
     confidence. Review queue: similar descriptions coded differently in prior
     months flagged for human review.'

  WORSE:
    '54'

  The AI's transparency is what makes human oversight meaningful.
  An opaque system forces the human to either trust blindly or re-do the work.

Cross-reference: Chapter 13 covers the full bounded agency design framework,
including the autonomy dial and design patterns for federal statistical contexts.
"""


def print_bounded_agency():
    print("Bounded Agency: Three Levels of Human Involvement")
    print("=" * DIVIDER_WIDTH)

    for level in LEVELS:
        print(f"\n  Level {level['level']}: {level['name']}")
        print(f"    Definition:   {level['definition']}")
        print(f"    When to use:  {level['when']}")
        print(f"    Example:      {level['example']}")
        print(f"    Implication:  {level['implication']}")

    print()
    print(FEDERAL_CONTEXT)
    print(DESIGN_IMPLICATION)


if __name__ == "__main__":
    print_bounded_agency()
