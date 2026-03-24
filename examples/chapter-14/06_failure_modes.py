"""
06_failure_modes.py
Chapter 14: Evaluating AI Systems for Federal Use

Six common failure modes in AI system evaluation.
Structured display: what each mode is, how to spot it,
and how to counter it. For use by federal evaluators
when reviewing vendor presentations and pilot proposals.
"""

# ── Configuration ─────────────────────────────────────────────────────────────
DIVIDER_WIDTH = 65

FAILURE_MODES = [
    {
        "name": "Demo-ware",
        "description": (
            "The system performs impressively on curated demonstrations "
            "but has no production capability at the required scale."
        ),
        "tell": (
            "Vendor shows a live demo but cannot provide a pilot deployment "
            "or access to a sandbox environment on your actual data."
        ),
        "counter": (
            "Request a 30-day pilot on a random sample of your production data. "
            "Measure accuracy, throughput, and failure rate independently."
        ),
    },
    {
        "name": "Benchmark gaming",
        "description": (
            "High accuracy on a published benchmark dataset that does not "
            "reflect your specific use case, data characteristics, or error modes."
        ),
        "tell": (
            "Vendor cites a published paper or competition result as primary evidence."
        ),
        "counter": (
            "Ask for results on your data. Run independent validation. "
            "Compare to your existing human coder agreement rates as baseline."
        ),
    },
    {
        "name": "Cherry-picked examples",
        "description": (
            "Vendor presents cases where the system performs well, "
            "omitting systematic failure categories."
        ),
        "tell": (
            "Examples all come from the easy, common cases. "
            "No mention of error rates on ambiguous or rare cases."
        ),
        "counter": (
            "Ask specifically: 'Show me 10 cases where your system fails. "
            "What are the common patterns?' A vendor who cannot answer this "
            "has not done a failure analysis."
        ),
    },
    {
        "name": "Opacity as a feature",
        "description": (
            "Vendor uses 'proprietary algorithm' to deflect documentation "
            "and audit requests."
        ),
        "tell": (
            "No model card, no data sheet, no architecture description. "
            "'We cannot share that' applied to fundamental design questions."
        ),
        "counter": (
            "Require documentation as a procurement condition. "
            "NIST AI RMF and federal procurement standards support this."
        ),
    },
    {
        "name": "Automation bias in evaluation",
        "description": (
            "Evaluators defer to the AI system's confidence scores without "
            "independent verification, treating high confidence as evidence of accuracy."
        ),
        "tell": (
            "Evaluation report says 'the model was 94% confident' as evidence "
            "of correctness, without checking against ground truth."
        ),
        "counter": (
            "Calibration analysis: does confidence=0.9 actually mean 90% accuracy? "
            "Overconfident errors are more dangerous than underconfident ones."
        ),
    },
    {
        "name": "The Dunning-Kruger evaluation gap",
        "description": (
            "Evaluators know enough about AI to be impressed by capability "
            "demonstrations, but not enough to identify missing safeguards."
        ),
        "tell": (
            "Evaluation focuses entirely on 'does it work?' and accepts vendor "
            "claims on reproducibility, bias, and documentation without verification."
        ),
        "counter": (
            "Use this rubric. Ask for documentation before the demo. "
            "Bring in an independent technical reviewer who has not seen the demo."
        ),
    },
]


def print_failure_modes():
    print("Common Failure Modes in AI System Evaluation")
    print("=" * DIVIDER_WIDTH)
    for i, fm in enumerate(FAILURE_MODES, start=1):
        print(f"\n{i}. {fm['name'].upper()}")
        print(f"   What it is:  {fm['description']}")
        print(f"   Tell-tale:   {fm['tell']}")
        print(f"   Countermove: {fm['counter']}")

    print()
    print("-" * DIVIDER_WIDTH)
    print("Pattern across all six modes:")
    print("  The fundamental failure is accepting vendor claims without")
    print("  independent evidence. These modes are not AI problems --")
    print("  they are procurement and evaluation discipline problems.")
    print("  The 10-dimension rubric (04_evaluation_rubric.py) forces")
    print("  the questions that each failure mode exploits.")


if __name__ == "__main__":
    print_failure_modes()
