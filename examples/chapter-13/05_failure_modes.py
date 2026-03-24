"""
Chapter 13: Agentic AI for Federal Statistical Operations
Example 05: Failure mode taxonomy

Maps the Microsoft AI Red Team taxonomy to design principles.
Pedagogical display only. No API calls.
"""

failure_modes = [
    {
        "category": "Misalignment",
        "description": "System pursues a goal that is misspecified or different from what was intended.",
        "federal_example": (
            "Coding pipeline optimizes for throughput (more codes per hour) rather than "
            "accuracy, because the prompt specified speed but not quality."
        ),
        "violated_principle": "Principle 1: Good judgment upfront",
        "detection": "Track accuracy metrics, not just throughput. Audit a sample of output.",
    },
    {
        "category": "Actions outside intended scope",
        "description": "Agent takes actions the designer never intended to authorize.",
        "federal_example": (
            "Pipeline given broad 'edit data' permission auto-corrects respondent answers "
            "instead of flagging them for human review."
        ),
        "violated_principle": "Principle 2: Agency requires governance",
        "detection": "Audit log of all actions taken. Review any writes to production data.",
    },
    {
        "category": "Cascading failures",
        "description": "Error in one step propagates unchecked through downstream steps.",
        "federal_example": (
            "An error in occupation coding (step 3) affects industry imputation (step 5), "
            "which affects weighting (step 7), which affects published estimates (step 10)."
        ),
        "violated_principle": "Principle 5 (Design for uncertainty) and Principle 6 (Digestible chunks)",
        "detection": "Checkpoints between stages. Validate output of each stage independently.",
    },
    {
        "category": "Organizational knowledge loss",
        "description": "Logic is buried in model behavior rather than in auditable rules.",
        "federal_example": (
            "After the LLM coding system is deployed, nobody knows why it makes specific "
            "decisions. When it fails, there is nothing to debug except the entire model."
        ),
        "violated_principle": "Principle 3 (Most problems do not need agents) and Principle 4 (Specification is the skill)",
        "detection": "Require reasoning traces for every decision. Document specification separately.",
    },
    {
        "category": "Accountability gaps",
        "description": "No clear record of who or what made a specific decision, or why.",
        "federal_example": (
            "Auditor asks: why was this household's income imputed to X? No audit trail "
            "exists. The pipeline made the decision; the model has no memory; no logs kept."
        ),
        "violated_principle": "Principle 4: Specification is the skill (audit requirements not specified)",
        "detection": "Mandatory decision logging. Every output traceable to the step that produced it.",
    },
    {
        "category": "Human-in-the-loop bypass",
        "description": "Designed checkpoints are skipped in practice.",
        "federal_example": (
            "Coders required to review low-confidence cases, but the review interface "
            "approves in bulk because individual review is tedious. Oversight exists on "
            "paper but not in practice."
        ),
        "violated_principle": "Principle 5: Design for uncertainty (oversight not designed to be practical)",
        "detection": "Track override rates. If coders never override the model, that is a warning sign.",
    },
]


def display_failure_modes(modes):
    col_widths = {
        "category": 30,
        "violated_principle": 48,
        "detection": 55,
    }

    print("Failure mode taxonomy (from Microsoft AI Red Team) mapped to design principles")
    print("=" * 75)

    for fm in modes:
        print(f"\n{fm['category'].upper()}")
        print(f"  Description:        {fm['description']}")
        print(f"  Federal example:    {fm['federal_example'][:80]}...")
        print(f"  Violates:           {fm['violated_principle']}")
        print(f"  Detection:          {fm['detection']}")

    print()
    print("=" * 75)
    print("Pattern: Every documented failure mode traces back to at least one")
    print("of the six design principles being ignored. The principles are not")
    print("cautionary tales -- they are a checklist for the design review.")


if __name__ == "__main__":
    display_failure_modes(failure_modes)
