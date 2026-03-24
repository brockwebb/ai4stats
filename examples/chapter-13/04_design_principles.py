"""
Chapter 13: Agentic AI for Federal Statistical Operations
Example 04: Six design principles and the meta-principle visualization

Preserves Brock's exact principle formulations.
No savefig calls. Pedagogical illustration only.
"""

import matplotlib.pyplot as plt


# Six design principles: exact formulations preserved
principles = [
    {
        "number": 1,
        "name": "Good judgment upfront",
        "one_liner": "Design quality bounds output quality",
        "explanation": (
            "AI amplifies your process. A poorly specified survey coding prompt produces "
            "bad codes at scale, regardless of model capability. Time spent on design is "
            "not wasted; it determines the ceiling on output quality."
        ),
        "federal_example": (
            "A NAICS coding pipeline with a vague prompt ('code this occupation') will "
            "produce inconsistently formatted and error-prone codes. A prompt that specifies "
            "the taxonomy version, the confidence format, the language for flagging edge "
            "cases, and the escalation condition produces auditable, consistent output."
        ),
        "failure_if_ignored": (
            "Misinterpretation of instructions, agent misalignment, hallucinated codes "
            "that look plausible but are wrong."
        ),
    },
    {
        "number": 2,
        "name": "Agency requires governance",
        "one_liner": "Less agency is often better",
        "explanation": (
            "Giving an agent more authority is not an improvement; it is a tradeoff. "
            "More agency means more flexibility AND less predictability. Start with the "
            "least authority that accomplishes the task."
        ),
        "federal_example": (
            "An imputation pipeline that auto-selects donor pools without human review "
            "can introduce bias that nobody catches. Constraining the pipeline to propose "
            "donor pools -- with a human reviewing stratification decisions -- costs a small "
            "amount of speed and prevents a difficult-to-detect systematic error."
        ),
        "failure_if_ignored": (
            "Actions outside intended scope, user harm from excessive autonomy, cascading "
            "errors when an unconstrained decision propagates through downstream steps."
        ),
    },
    {
        "number": 3,
        "name": "Most problems do not need agents",
        "one_liner": "Simple solutions beat complex ones",
        "explanation": (
            "The hype cycle wants everything to be agentic. Good engineering does not. "
            "An agent adds value only when: the task has genuine variability that cannot "
            "be pre-scripted, decisions must be made at scale, and the cost of human "
            "attention exceeds the cost of imperfect automation."
        ),
        "federal_example": (
            "A regex-based address parser for standardized Census address formats may "
            "outperform an LLM-based one and is fully auditable. A lookup table that maps "
            "common occupation phrases to SOC codes is faster, cheaper, and more reproducible "
            "than an LLM for high-frequency, well-defined descriptions."
        ),
        "failure_if_ignored": (
            "Organizational knowledge loss (logic buried in an opaque model rather than "
            "in an auditable rule set), unnecessary attack surface, dependency on vendor "
            "infrastructure for tasks that could be internal."
        ),
    },
    {
        "number": 4,
        "name": "Specification is the skill",
        "one_liner": "Clarity beats capability",
        "explanation": (
            "You do not need to code to work with agents effectively. You need to think "
            "clearly: what exactly do you want, what constraints apply, what does success "
            "look like, and what should happen when things go wrong. A precisely specified "
            "prompt from a methodologist outperforms a vague prompt from an AI enthusiast, "
            "regardless of model."
        ),
        "federal_example": (
            "The Federal Survey Concept Mapper succeeded because the specification was "
            "precise: exact taxonomy version, confidence tier definitions, dual-modal "
            "flagging rules, escalation conditions. The capability was secondary to the "
            "specification."
        ),
        "failure_if_ignored": (
            "Incorrect permissions (model interprets vague instructions as broad authority), "
            "accountability gaps (nobody can explain why a specific decision was made), "
            "transparency failures."
        ),
    },
    {
        "number": 5,
        "name": "Design for uncertainty",
        "one_liner": "Plan for failure, not just success",
        "explanation": (
            "Things will go wrong. The question is whether you planned for it. Every step "
            "in a pipeline needs a defined failure path. An agent that guesses when uncertain "
            "is worse than one that stops and asks."
        ),
        "federal_example": (
            "A survey coding pipeline that receives an occupation description in a language "
            "the model has never seen should have a defined failure path: route to a bilingual "
            "coder, apply a low-confidence flag, or assign an 'undetermined' code rather "
            "than producing a plausible but wrong code with high apparent confidence."
        ),
        "failure_if_ignored": (
            "Human-in-the-loop bypass (model proceeds when it should stop), cascading "
            "failures (error in step 1 propagates to steps 2-10), denial of service "
            "(unconstrained pipeline exhausts resources trying to process unsolvable cases)."
        ),
    },
    {
        "number": 6,
        "name": "Digestible chunks",
        "one_liner": "Focused beats sprawling",
        "explanation": (
            "Context windows have hard limits. Model performance degrades before those "
            "limits. A pipeline broken into discrete steps with defined inputs and outputs "
            "outperforms one massive prompt trying to do everything. Decompose complex "
            "tasks into stages. Let each stage do one thing well."
        ),
        "federal_example": (
            "A survey processing pipeline divided into: (1) cleaning, (2) item imputation, "
            "(3) occupation coding, (4) industry coding, (5) validation, each with checkpoints "
            "between stages, is more auditable and more robust than a single 'process the "
            "survey' prompt that tries to do everything at once."
        ),
        "failure_if_ignored": (
            "Resource exhaustion, loss of data provenance (which step produced which "
            "output?), hallucinations from overloaded context, subtle degradation that "
            "produces fluent but wrong output."
        ),
    },
]


def display_principles(principles_list):
    print("Six design principles for agentic AI in federal statistics")
    print("=" * 70)
    for p in principles_list:
        print(f"\n{p['number']}. {p['name'].upper()}: {p['one_liner']}")
        print(f"   {p['explanation'][:90]}...")
        print(f"   Federal example: {p['federal_example'][:80]}...")
        print(f"   If ignored: {p['failure_if_ignored'][:75]}...")


def draw_meta_principle():
    """Visualize the meta-principle: AI amplifies your process."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    # Bad process side
    ax.add_patch(plt.Rectangle(
        (0.3, 1.5), 2.5, 1.4,
        facecolor="#FFEBEE", edgecolor="#F44336", linewidth=1.5, alpha=0.9,
    ))
    ax.text(1.55, 2.6, "Bad process", ha="center", fontsize=9,
            fontweight="bold", color="#C62828")
    ax.text(1.55, 2.1, "Chaotic workflow\nVague requirements\nNo checkpoints",
            ha="center", fontsize=7.5, color="#C62828")

    ax.text(3.35, 2.25, "+ AI =", ha="center", fontsize=12,
            fontweight="bold", color="#333333")

    ax.add_patch(plt.Rectangle(
        (4.3, 1.5), 2.7, 1.4,
        facecolor="#FFEBEE", edgecolor="#F44336", linewidth=2, alpha=0.9,
    ))
    ax.text(5.65, 2.6, "Faster bad\noutcomes", ha="center", fontsize=9,
            fontweight="bold", color="#C62828")

    # Good process side
    ax.add_patch(plt.Rectangle(
        (0.3, 0.3), 2.5, 0.85,
        facecolor="#E8F5E9", edgecolor="#388E3C", linewidth=1.5, alpha=0.9,
    ))
    ax.text(1.55, 0.75, "Good process", ha="center", fontsize=9,
            fontweight="bold", color="#2E7D32")

    ax.text(3.35, 0.75, "+ AI =", ha="center", fontsize=12,
            fontweight="bold", color="#333333")

    ax.add_patch(plt.Rectangle(
        (4.3, 0.3), 2.7, 0.85,
        facecolor="#E8F5E9", edgecolor="#388E3C", linewidth=2, alpha=0.9,
    ))
    ax.text(5.65, 0.75, "Faster good outcomes", ha="center", fontsize=9,
            fontweight="bold", color="#2E7D32")

    # Summary box
    ax.text(8.2, 1.75, "AI amplifies\nyour process.\nWhat it multiplies\nis up to you.",
            ha="center", va="center", fontsize=9, color="#333333",
            bbox=dict(boxstyle="round", facecolor="#FFF9C4",
                      edgecolor="#F9A825", linewidth=1.2))

    ax.set_title("The meta-principle: AI amplifies your process", fontsize=11, pad=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    display_principles(principles)
    print()
    draw_meta_principle()
