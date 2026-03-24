"""
Chapter 13: Agentic AI for Federal Statistical Operations
Example 09: Exercise starter code

Students complete the TODO sections. No API calls.
"""


# ---------------------------------------------------------------------------
# Exercise 1: Loop mapping
# Trace a hypothetical NAICS coding pipeline through the four phases
# of the observe-decide-act-check loop.
# ---------------------------------------------------------------------------

print("Exercise 13.1: Map the NAICS coding pipeline to the loop")
print("=" * 65)
print()
print("Scenario: Your division is automating initial NAICS coding for the")
print("Economic Census. The system receives open-text business descriptions")
print("and assigns a preliminary 6-digit NAICS code. Human coders review")
print("flagged cases before codes enter the published microdata.")
print()

exercise_1_loop = {
    "OBSERVE": {
        "input": "TODO: what does the pipeline take in?",
        "action": "TODO: what processing happens at this stage?",
        "output": "TODO: what information is ready for DECIDE?",
        "human_involvement": "TODO: is a human involved here? why or why not?",
    },
    "DECIDE": {
        "input": "TODO: what information does DECIDE receive?",
        "action": "TODO: what decision is made?",
        "output": "TODO: what is produced for ACT?",
        "human_involvement": "TODO: human checkpoint here? justify your choice.",
    },
    "ACT": {
        "input": "TODO: what does ACT receive?",
        "action": "TODO: what is written to the database?",
        "output": "TODO: what record is created?",
        "audit_log": "TODO: what must be logged for auditability? (be specific)",
    },
    "CHECK": {
        "input": "TODO: what does CHECK evaluate?",
        "action": "TODO: what validation is run?",
        "output": "TODO: pass / flag for human review / escalate? what triggers each?",
        "human_involvement": "TODO: required human checkpoint here?",
    },
}

for stage, fields in exercise_1_loop.items():
    print(f"{stage}:")
    for key, value in fields.items():
        print(f"  {key}: {value}")
    print()

print("Questions to answer after completing the table:")
print("  1. At which stage would you place a human checkpoint? Justify with reference")
print("     to decision volume, error catchability, and error cost proportionality.")
print("  2. What information at the ACT stage is the minimum needed for an auditor")
print("     to reconstruct why a specific code was assigned to a specific record?")
print()


# ---------------------------------------------------------------------------
# Exercise 2: Autonomy dial rating
# Rate a described system on the autonomy dial with justification.
# ---------------------------------------------------------------------------

print("=" * 65)
print("Exercise 13.2: Rate the autonomy dial")
print("=" * 65)
print()
print("System description:")
print("  The occupation coding pipeline from Exercise 1 is now in production.")
print("  After 6 months, the team proposes removing the human review step for")
print("  cases with confidence >= 0.95, citing 99.1% accuracy on a 500-record")
print("  validation sample and a backlog of 50,000 unprocessed records.")
print()
print("Your task:")
print("  1. On the autonomy dial, where is the CURRENT system positioned?")
print("     (Tool use only / Propose, human approves / Auto-execute with human review /")
print("      Exceptions escalate / Fully autonomous)")
print()
print("     Current position: TODO")
print("     Justification:    TODO")
print()
print("  2. Where would the PROPOSED change move it?")
print()
print("     New position:     TODO")
print("     Justification:    TODO")
print()
print("  3. Apply the three conditions for moving right:")
print("     - Decision volume: TODO (does it justify the change?)")
print("     - Error catchability: TODO (can errors be caught before publication?)")
print("     - Error cost proportionality: TODO (is the risk acceptable?)")
print()
print("  4. What specific new risk does the proposed change introduce?")
print()
print("     New risk:         TODO")
print()


# ---------------------------------------------------------------------------
# Exercise 3: Design principle evaluation
# Apply all six principles to a described agentic system.
# ---------------------------------------------------------------------------

print("=" * 65)
print("Exercise 13.3: Apply the six design principles")
print("=" * 65)
print()
print("System description:")
print("  A colleague proposes an 'autonomous data quality agent' that:")
print("  - Ingests paradata from the current collection wave")
print("  - Identifies interviews with anomalous duration, item nonresponse")
print("    patterns, or geographic inconsistencies")
print("  - Automatically flags those interviews for removal from published microdata")
print("  - Generates a removal rationale in plain language for each flagged case")
print()

principle_evaluations = [
    {
        "number": 1,
        "name": "Good judgment upfront",
        "status": "TODO: satisfied / violated / need more info",
        "reason": "TODO: explain your reasoning",
        "modification": "TODO: what change would address any violation?",
    },
    {
        "number": 2,
        "name": "Agency requires governance",
        "status": "TODO: satisfied / violated / need more info",
        "reason": "TODO: explain your reasoning",
        "modification": "TODO: what change would address any violation?",
    },
    {
        "number": 3,
        "name": "Most problems do not need agents",
        "status": "TODO: satisfied / violated / need more info",
        "reason": "TODO: explain your reasoning",
        "modification": "TODO: what change would address any violation?",
    },
    {
        "number": 4,
        "name": "Specification is the skill",
        "status": "TODO: satisfied / violated / need more info",
        "reason": "TODO: explain your reasoning",
        "modification": "TODO: what change would address any violation?",
    },
    {
        "number": 5,
        "name": "Design for uncertainty",
        "status": "TODO: satisfied / violated / need more info",
        "reason": "TODO: explain your reasoning",
        "modification": "TODO: what change would address any violation?",
    },
    {
        "number": 6,
        "name": "Digestible chunks",
        "status": "TODO: satisfied / violated / need more info",
        "reason": "TODO: explain your reasoning",
        "modification": "TODO: what change would address any violation?",
    },
]

for p in principle_evaluations:
    print(f"Principle {p['number']}: {p['name']}")
    print(f"  Status:       {p['status']}")
    print(f"  Reason:       {p['reason']}")
    print(f"  Modification: {p['modification']}")
    print()

print("Recommended modifications (minimum three):")
print("  1. TODO")
print("  2. TODO")
print("  3. TODO")
print()
print("After recommending modifications, reconsider: is this still best")
print("implemented as an agent, or does the constrained version look more")
print("like a rule-based filter? (Principle 3 applies here.)")
