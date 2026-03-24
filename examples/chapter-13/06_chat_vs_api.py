"""
Chapter 13: Agentic AI for Federal Statistical Operations
Example 06: Chat interface vs. API-based pipeline comparison

Pedagogical display only. No API calls.
"""

import textwrap


comparison = {
    "What it is": {
        "Chat (e.g., Claude.ai, ChatGPT)": (
            "Conversational interface; single session; no persistent state."
        ),
        "API-based pipeline": (
            "Programmatic access; version-controlled prompts; logged outputs; reproducible."
        ),
    },
    "Good for": {
        "Chat (e.g., Claude.ai, ChatGPT)": (
            "Exploration, one-shot questions, drafting text, brainstorming."
        ),
        "API-based pipeline": (
            "Reproducible workflows, batch processing, auditable research, published statistics."
        ),
    },
    "Reproducible?": {
        "Chat (e.g., Claude.ai, ChatGPT)": (
            "No. Conversation history is not versioned. Re-running the same prompt "
            "may produce different output."
        ),
        "API-based pipeline": (
            "Yes, with version-pinned models, logged inputs and outputs, seeded randomness."
        ),
    },
    "Auditable?": {
        "Chat (e.g., Claude.ai, ChatGPT)": (
            "No. No audit trail. Export of conversation is manual and informal."
        ),
        "API-based pipeline": (
            "Yes, with structured logging, decision traces, and input/output records."
        ),
    },
    "Privacy-compliant for microdata?": {
        "Chat (e.g., Claude.ai, ChatGPT)": (
            "Depends on data classification and FedRAMP authorization. Sending microdata "
            "through a public chat interface is not appropriate."
        ),
        "API-based pipeline": (
            "Requires FedRAMP authorization and data governance controls. Can be made "
            "compliant with proper architecture."
        ),
    },
    "Appropriate for published statistics?": {
        "Chat (e.g., Claude.ai, ChatGPT)": (
            "No. Not reproducible, not auditable, not defensible to a Disclosure Review Board."
        ),
        "API-based pipeline": (
            "Potentially yes, with proper governance, logging, and human review."
        ),
    },
}

chat_appropriate = [
    "Exploring a new dataset to generate hypotheses",
    "Drafting sections of a methodology report for human editing",
    "Getting a quick explanation of an unfamiliar statistical method",
    "Brainstorming approaches to a research problem",
    "Reviewing your own work for clarity or logical gaps",
    "Learning about a new regulatory or technical standard",
]

api_required = [
    "Batch processing of survey responses for publication",
    "Automated coding that will be incorporated into microdata",
    "Imputation that produces values entering official estimates",
    "Any output that goes into a published statistical product",
    "Any analysis that must be reproducible by an external reviewer",
    "Any decision process that requires an audit trail",
]


def display_comparison(comp):
    print("Chat interfaces vs. API-based pipelines")
    print("=" * 75)
    for dimension, values in comp.items():
        print(f"\n{dimension}:")
        for modality, description in values.items():
            label = modality[:35]
            wrapped = textwrap.fill(
                description, width=58, initial_indent="    ", subsequent_indent="    "
            )
            print(f"  [{label}]")
            print(wrapped)


def display_use_lists(chat, api):
    print()
    print("When chat is appropriate:")
    for item in chat:
        print(f"  + {item}")
    print()
    print("When API-based pipeline is required:")
    for item in api:
        print(f"  ! {item}")
    print()
    print("The line is reproducibility and auditability.")
    print("If it goes into published statistics, it needs a pipeline, not a conversation.")


if __name__ == "__main__":
    display_comparison(comparison)
    display_use_lists(chat_appropriate, api_required)
