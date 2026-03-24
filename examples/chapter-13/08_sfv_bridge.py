"""
Chapter 13: Agentic AI for Federal Statistical Operations
Example 08: SFV threats in agentic pipelines

Maps each SFV threat to its manifestation in an agentic coding pipeline
and the countermeasure provided by bounded agency design.
Pedagogical display only. No API calls.
"""

sfv_threats = {
    "T1: Semantic Drift": {
        "threat_description": (
            "Key terms shift meaning across sessions or pipeline stages without "
            "explicit acknowledgment."
        ),
        "agentic_manifestation": (
            "Terminology established in session 1 (e.g., 'unit nonresponse') mutates "
            "across the multi-session pipeline without explicit redefinition. DECIDE "
            "stages use drifted terminology; decisions are wrong."
        ),
        "loop_stage": "DECIDE stage",
        "mitigation_in_bounded_agency": (
            "Config-driven vocabulary: terms defined in an external config file, "
            "injected at each session start. No session defines its own terminology "
            "from scratch."
        ),
    },
    "T2: False State Injection": {
        "threat_description": (
            "Pipeline incorporates incorrect information from a hallucination, tool "
            "error, or flawed intermediate result, treating it as valid prior state."
        ),
        "agentic_manifestation": (
            "Pipeline 'remembers' a decision to use method X that was never made. "
            "Subsequent DECIDE stages proceed on a fabricated basis."
        ),
        "loop_stage": "DECIDE stage",
        "mitigation_in_bounded_agency": (
            "Decision log: external record of every decision. Periodic state "
            "reconciliation diffed against the log catches injected states."
        ),
    },
    "T3: Compression Distortion": {
        "threat_description": (
            "Context compaction strips rationale from prior decisions. Pipeline "
            "continues on a degraded representation."
        ),
        "agentic_manifestation": (
            "Context compaction strips rationale from an exclusion decision. Later "
            "DECIDE stages cannot distinguish justified from unjustified exclusions."
        ),
        "loop_stage": "OBSERVE stage",
        "mitigation_in_bounded_agency": (
            "Documentation-as-traceability: rationale written to an external document "
            "before compaction can strip it. The pipeline re-ingests the document "
            "at the next session."
        ),
    },
    "T4: State Supersession Failure": {
        "threat_description": (
            "A superseded decision or corrected value persists alongside its replacement. "
            "Pipeline operates on contradictory state."
        ),
        "agentic_manifestation": (
            "Privacy budget (epsilon) revised in session 4, but the ACT stage in "
            "session 5 applies the old value. Correct decision, wrong parameter."
        ),
        "loop_stage": "ACT stage",
        "mitigation_in_bounded_agency": (
            "Parameter audit: after any revision, verify the new value is operative "
            "in all pipeline stages before the next action. Include parameter version "
            "in every log entry."
        ),
    },
    "T5: State Discontinuity": {
        "threat_description": (
            "Session boundary causes partial or complete loss of accumulated pipeline state."
        ),
        "agentic_manifestation": (
            "New session starts without session 1 methodology decisions. The OBSERVE "
            "stage begins from near-zero state; all downstream decisions are affected."
        ),
        "loop_stage": "OBSERVE stage",
        "mitigation_in_bounded_agency": (
            "Handoff documents: explicit state serialization at every session boundary. "
            "The next session's first action is to ingest the handoff and confirm state "
            "before taking any action."
        ),
    },
}


def display_sfv_bridge(threats):
    print("SFV threats in the agentic pipeline context")
    print("=" * 75)
    print()
    print("Chapter 15 (Capstone) formalizes the full SFV framework. This table")
    print("identifies where each SFV threat becomes acute in a coding pipeline.")
    print()

    for threat_name, details in threats.items():
        print(f"{threat_name}")
        print(f"  Threat:           {details['threat_description'][:75]}...")
        print(f"  In pipeline:      {details['agentic_manifestation'][:80]}...")
        print(f"  Loop stage:       {details['loop_stage']}")
        print(f"  Mitigation:       {details['mitigation_in_bounded_agency'][:80]}...")
        print()

    print("=" * 75)
    print("Key insight: agentic pipelines create state, and that state must be")
    print("managed explicitly across sessions and stages. SFV provides the")
    print("validity framework for verifying that management actually works.")
    print()
    print("See Chapter 15 (Capstone), Section 10 for the operational reproducibility")
    print("checklist that applies these mitigations across a pipeline's full lifecycle.")


if __name__ == "__main__":
    display_sfv_bridge(sfv_threats)
