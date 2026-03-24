"""
09_case_study_sfv.py
Chapter 14: Evaluating AI Systems for Federal Use

Case B.2: LLM comment summarizer -- SFV risk analysis.
Illustrates how each of the five SFV threats (T1-T5) manifests
in a multi-session LLM survey comment summarization pipeline.

Cross-reference: Chapter 15 covers the full SFV framework,
threat taxonomy (T1-T5), and engineering countermeasures.
"""

# ── Configuration ─────────────────────────────────────────────────────────────
DIVIDER_WIDTH = 65

SCENARIO = """
SCENARIO: LLM-Powered Respondent Comment Summarizer
====================================================
A vendor proposes an LLM system that 'reads' open-ended survey
comments and produces summary statistics (themes, sentiment,
key concerns). The system runs in multi-session mode: it
accumulates results across the annual survey cycle, building
a running summary that updates each month.

Standard rubric concerns: apply dimensions 1-9 as usual.

SFV concerns specific to this system are shown below.
"""

SFV_THREATS = [
    {
        "code": "T1",
        "name": "Semantic Drift",
        "illustration": (
            "In month 1, 'housing cost burden' is defined as >30% of income. "
            "By month 7, the model has begun using the term to include both the "
            "30%+ threshold AND respondents who 'mention housing costs.' These are "
            "different things. The cumulative summary is now internally inconsistent."
        ),
        "risk": "Annual summary conflates two different definitions without flagging it.",
        "countermeasure": "Config-driven vocabulary with locked definitions per survey cycle.",
    },
    {
        "code": "T2",
        "name": "False State Injection",
        "illustration": (
            "The model 'remembers' a decision to exclude single-person households "
            "from the theme analysis that was never actually made. All summaries "
            "produced after month 4 silently undercount single-person household concerns."
        ),
        "risk": "Fabricated methodological decision treated as fact in downstream analysis.",
        "countermeasure": "Canonical decision log with human-verified entries only.",
    },
    {
        "code": "T3",
        "name": "Compression Distortion",
        "illustration": (
            "Monthly compaction summarizes 'we excluded respondents who indicated "
            "English was not their primary language due to translation reliability "
            "concerns' as 'non-English excluded.' By year-end, the rationale is gone. "
            "The annual summary does not mention this exclusion at all."
        ),
        "risk": (
            "Methodological exclusion becomes invisible; "
            "published summary misrepresents coverage."
        ),
        "countermeasure": (
            "Require preservation of rationale strings, not just decisions, "
            "in compaction summaries."
        ),
    },
    {
        "code": "T4",
        "name": "State Supersession Failure",
        "illustration": (
            "Month 3 codebook revision changed the definition of 'food insecurity.' "
            "The pipeline never received this update. "
            "Months 4-12 use the old definition."
        ),
        "risk": (
            "Published annual summary uses an inconsistent definition "
            "across months without disclosure."
        ),
        "countermeasure": "Active state reconciliation when codebook changes occur.",
    },
    {
        "code": "T5",
        "name": "State Discontinuity",
        "illustration": (
            "A staffing change in month 6 resulted in a new analyst restarting the "
            "LLM session. The new session does not have the accumulated context from "
            "months 1-5. The month 6 report is methodologically inconsistent "
            "with all prior months but this is not flagged."
        ),
        "risk": "Incoherent longitudinal analysis passed to publication without detection.",
        "countermeasure": "Handoff documents that serialize full pipeline state at session close.",
    },
]

CONCLUSION = """
Evaluation Conclusion:
  This system requires explicit SFV controls before it can produce
  defensible survey summary statistics for publication.

  Without config-driven vocabulary, session management, and a canonical
  decision log, the annual summary cannot be trusted to reflect a coherent
  methodological process.

  These are not AI problems per se. They are state management and
  provenance problems that arise specifically because the pipeline
  operates across multiple sessions over time.

See Chapter 15 for the full State Fidelity Validity framework,
engineering countermeasures, and validation criteria.
"""


def print_sfv_case_study():
    print(SCENARIO)
    print("SFV Threat Analysis")
    print("=" * DIVIDER_WIDTH)

    for threat in SFV_THREATS:
        print(f"\n{threat['code']} - {threat['name']}")
        print(f"  Illustration:    {threat['illustration']}")
        print(f"  Risk:            {threat['risk']}")
        print(f"  Countermeasure:  {threat['countermeasure']}")

    print()
    print(CONCLUSION)


if __name__ == "__main__":
    print_sfv_case_study()
