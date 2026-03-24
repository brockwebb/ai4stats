"""
Chapter 13: Agentic AI for Federal Statistical Operations
Example 01: Core vocabulary definitions

Pedagogical display only. No API calls.
"""

vocabulary = [
    {
        "term": "Tool",
        "definition": "A single discrete operation an agent can invoke.",
        "key_question": "What can it do?",
        "federal_example": "API call to Census geocoder; database lookup of prior-year codes.",
    },
    {
        "term": "Agent",
        "definition": "An entity that does work within a workflow.",
        "key_question": "What is doing the work?",
        "federal_example": "An LLM-based classifier processing occupation descriptions.",
    },
    {
        "term": "Workflow",
        "definition": "A structure: defined sequence or graph of steps to accomplish a goal.",
        "key_question": "What is the process?",
        "federal_example": "Survey collection, processing, editing, weighting, publication.",
    },
    {
        "term": "Agentic",
        "definition": "Behavior in which an AI system exercises granted decision-making authority.",
        "key_question": "How much can it adapt?",
        "federal_example": "Pipeline that adjusts its own confidence threshold based on batch error rate.",
    },
    {
        "term": "Agency",
        "definition": "Granted decision-making authority; conferred by design, not inherent to the model.",
        "key_question": "What decisions can it make?",
        "federal_example": "Authority to assign a 4-digit NAICS code without human review.",
    },
    {
        "term": "Bounded agency",
        "definition": "Agentic AI operating within explicit constraints with defined human oversight.",
        "key_question": "Who reviews the decisions?",
        "federal_example": "AI proposes NAICS code; human coder approves flagged cases.",
    },
    {
        "term": "Autonomy dial",
        "definition": "A spectrum from fully human-controlled to fully autonomous AI action.",
        "key_question": "Where does authority sit?",
        "federal_example": "Federal statistical operations belong on the left (bounded) side.",
    },
    {
        "term": "Observe-decide-act-check loop",
        "definition": "The universal pattern: observe state, decide action, act, check result, repeat.",
        "key_question": "What is the pipeline doing right now?",
        "federal_example": "Read survey batch, select coding prompt, call LLM, verify confidence >= threshold.",
    },
]

def display_vocabulary(entries):
    print("Core vocabulary for agentic AI in federal statistics")
    print("=" * 70)
    for entry in entries:
        print(f"\nTerm:           {entry['term']}")
        print(f"Definition:     {entry['definition']}")
        print(f"Key question:   {entry['key_question']}")
        print(f"Federal use:    {entry['federal_example']}")
    print()
    print("=" * 70)
    print("Critical clarification:")
    print("  Agency is a design choice. You decide how much decision-making")
    print("  authority to grant. The LLM does not grant itself authority.")
    print("  When someone proposes an 'AI agent for automated coding,' ask:")
    print("    - What workflow does it operate within?")
    print("    - What agency is being granted?")
    print("    - What tools does it have access to?")
    print("    - Under what conditions does it escalate?")
    print("  A proposal that cannot answer these questions is not ready for design review.")


if __name__ == "__main__":
    display_vocabulary(vocabulary)
