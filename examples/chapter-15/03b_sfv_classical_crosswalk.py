"""
Chapter 15 — SFV Threat-to-Classical-Validity Crosswalk
examples/chapter-15/03b_sfv_classical_crosswalk.py

Maps each of the five SFV threats to the classical validity type it
primarily degrades, with the mechanism and secondary targets.

Run: python examples/chapter-15/03b_sfv_classical_crosswalk.py
"""

crosswalk = [
    {
        "threat": "T1: Semantic Drift",
        "primary_validity": "Construct Validity",
        "mechanism": (
            "Terminology mutates mid-pipeline; operative construct changes "
            "without redefinition. The pipeline is no longer measuring what "
            "it defined at the outset."
        ),
        "secondary": ["Statistical Conclusion Validity"],
        "secondary_note": (
            "Drifted terms cause statistical quantities to reference different "
            "constructs across sessions."
        ),
    },
    {
        "threat": "T2: False State Injection",
        "primary_validity": "Internal Validity",
        "mechanism": (
            "Confabulated decision history breaks the causal chain. Inferences "
            "at step N rest on a methodological record that never occurred."
        ),
        "secondary": ["Construct Validity"],
        "secondary_note": (
            "A confabulated method choice changes what is being measured."
        ),
    },
    {
        "threat": "T3: Compression Distortion",
        "primary_validity": "Statistical Conclusion Validity",
        "mechanism": (
            "Compaction strips caveats and collapses conditional findings into "
            "unconditional ones. Downstream statistical inferences operate on "
            "distorted premises."
        ),
        "secondary": ["Internal Validity"],
        "secondary_note": (
            "Stripped rationale removes the basis for design decisions, "
            "introducing uncontrolled confounds."
        ),
    },
    {
        "threat": "T4: State Supersession Failure",
        "primary_validity": "Internal Validity",
        "mechanism": (
            "A persisting outdated parameter is a systematic confound the "
            "researcher believes was controlled. The analysis uses one value "
            "while the methodology claims another."
        ),
        "secondary": ["Statistical Conclusion Validity"],
        "secondary_note": (
            "Wrong parameter values produce wrong statistical quantities."
        ),
    },
    {
        "threat": "T5: State Discontinuity",
        "primary_validity": "External Validity",
        "mechanism": (
            "Findings are bound to the specific execution context. They do not "
            "generalize even to a re-run of the same pipeline with a session "
            "restart."
        ),
        "secondary": ["Internal Validity"],
        "secondary_note": (
            "Lost context may drop design decisions that controlled for confounds."
        ),
    },
]

print("SFV Threat-to-Classical-Validity Crosswalk")
print("=" * 72)
print()
print(f"{'Threat':<34} {'Primary Target':<30} {'Secondary Target(s)'}")
print("-" * 72)
for row in crosswalk:
    secondary = ", ".join(row["secondary"])
    print(f"{row['threat']:<34} {row['primary_validity']:<30} {secondary}")
print()

print("Primary mapping details")
print("-" * 72)
for row in crosswalk:
    print(f"\n{row['threat']}")
    print(f"  Primary:   {row['primary_validity']}")
    print(f"  Mechanism: {row['mechanism']}")
    print(f"  Secondary: {', '.join(row['secondary'])}")
    print(f"             {row['secondary_note']}")

print()
print("Theoretical summary")
print("-" * 72)
print(
    "Each classical validity type assumes a stable instrument. SFV threats are\n"
    "precisely the ways a stateful instrument can change -- and each mode of\n"
    "change maps to a classical validity failure that the framework assumed away.\n"
    "\n"
    "SFV does not compete with the classical types. It guards the assumption\n"
    "they all share."
)
