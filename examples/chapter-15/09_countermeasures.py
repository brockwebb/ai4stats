"""
Chapter 15: Engineering countermeasures.
Seven countermeasures with threat mappings, mechanism descriptions,
and Seldon implementation notes. Seldon references must be preserved exactly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

# Engineering countermeasures: what each addresses

countermeasures = [
    {
        "name": "Config-driven vocabulary",
        "addresses": ["T1"],
        "mechanism": "Terms are defined in configuration files external to the context window. "
                    "The pipeline queries the config for term definitions rather than relying "
                    "on what is retained in context.",
        "seldon_implementation": "YAML config files define all project-specific terminology. "
                                "Terms are injected into each session from config, not from prior context.",
    },
    {
        "name": "Graph-backed ontology",
        "addresses": ["T1", "T2"],
        "mechanism": "Concepts exist in a knowledge graph (e.g., Neo4j) as persistent entities. "
                    "The pipeline queries the graph for concept definitions and relationships; "
                    "it does not rely on the context window to remember them.",
        "seldon_implementation": "Neo4j stores project concepts, decisions, and their provenance. "
                                "Cypher queries retrieve current operative state on demand.",
    },
    {
        "name": "TEVV validation loops",
        "addresses": ["T2", "T3"],
        "mechanism": "Outputs are validated against an external source of truth before entering "
                    "the research base. Validation catches confabulations (T2) and compaction "
                    "distortions (T3) before they propagate.",
        "seldon_implementation": "Validation scripts compare model outputs against canonical "
                                "decision records. Discrepancies are flagged before results are registered.",
    },
    {
        "name": "Handoff documents",
        "addresses": ["T5"],
        "mechanism": "At every session boundary, a structured document serializes the full "
                    "operative state: all active decisions, parameters, terminology, and "
                    "intermediate findings. New sessions load this document before any analysis.",
        "seldon_implementation": "seldon closeout generates a structured handoff. seldon briefing "
                                "loads it at session start. Session continuity is explicit, not assumed.",
    },
    {
        "name": "Documentation-as-traceability",
        "addresses": ["T3", "T4"],
        "mechanism": "Every decision is written to an external document with full rationale before "
                    "any compaction can strip it. Architectural decision records (ADs) ensure "
                    "that revised decisions are tracked alongside their predecessors.",
        "seldon_implementation": "AD-013 and related architectural decision records store rationale "
                                "externally. Revisions create new ADs that reference and supersede prior ones.",
    },
    {
        "name": "Multi-model triangulation",
        "addresses": ["T2"],
        "mechanism": "Key decisions or provenance claims are validated by independently prompting "
                    "a second model instance with the same history. Agreement between models "
                    "provides convergent evidence; disagreement flags potential confabulation.",
        "seldon_implementation": "Critical findings are cross-checked across Claude, Gemini, and GPT "
                                "using independently seeded prompts. Convergent results are registered; "
                                "divergent results require human review.",
    },
    {
        "name": "Periodic state reconciliation",
        "addresses": ["T1", "T2", "T3", "T4"],
        "mechanism": "At regular intervals, the system is asked to restate all active decisions, "
                    "parameters, and terminology. The restatement is diffed against the canonical "
                    "external log. Discrepancies trigger investigation before they propagate.",
        "seldon_implementation": "seldon reconcile runs this check at session boundaries and on demand. "
                                "Output is a structured diff between canonical log and model restatement.",
    },
]

print("Engineering countermeasures for SFV threats:")
print("=" * 65)
for cm in countermeasures:
    threats_addressed = ", ".join(cm["addresses"])
    print(f"\n{cm['name']} (addresses: {threats_addressed})")
    wrapped = textwrap.fill(cm["mechanism"], width=65, initial_indent="  Mechanism: ", subsequent_indent="  ")
    print(wrapped)
