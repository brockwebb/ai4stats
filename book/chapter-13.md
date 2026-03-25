---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Chapter 13 - Agentic AI for Federal Statistical Operations

```{admonition} A note on tool stability
:class: warning
The specific tools, frameworks, and vendor products for building agentic AI systems change rapidly. What does not change: the vocabulary for describing agents, the design principles for building them responsibly, and the failure modes that occur when those principles are ignored. This chapter teaches the stable parts. When a specific tool is mentioned, treat it as an illustration, not a recommendation.
```

```{contents}
:depth: 2
```

## Learning goals

By the end of this chapter, you will be able to:

1. Define the core vocabulary: workflow, agent, agency, agentic, tool
2. Trace any agentic pipeline through the observe-decide-act-check loop
3. Explain why autonomy is a dial, not a switch, and why federal operations belong on the left side of that dial
4. Apply all six design principles to proposed federal statistical AI workflows
5. Identify failure modes (from the Microsoft AI Red Team taxonomy) and connect each to a violated design principle
6. Distinguish chat-based AI use from API-based pipeline construction, and explain when each is appropriate
7. Evaluate when an agentic approach is justified versus when a simpler solution is better
8. Recognize that agentic pipelines are the context where the SFV threats from Chapter 15 become most acute

---

## 1. Setup

The import dependencies and session header for this chapter are in `examples/chapter-13/`. See any script there for the full import block.

---

## 2. The vocabulary problem

Walk into any meeting about AI adoption in a federal agency today and you will hear: "We should build an AI agent for this." Rarely does everyone in the room pictures the same thing.

*Agent* is one of the most overloaded terms in technology. It has meant different things in economics, multi-agent systems research, software engineering, and marketing copy. When there is no shared vocabulary, teams talk past each other. Requirements are vague because nobody agrees on what is being built. Accountability is unclear because nobody agreed on what the system was supposed to do.

The definitions below come from the Agents 101 course developed for federal practitioners, synthesizing vocabulary from vendor agent guides (Anthropic, 2024; OpenAI, 2025; Google, 2024) into precise terms for federal use. Use them precisely. `examples/chapter-13/01_vocabulary.py` prints the full structured vocabulary table.

| Term | Definition | Key question | Federal example |
|------|-----------|--------------|-----------------|
| Workflow | Structure, container, sequence of steps | What is the process? | Survey collection, processing, editing, weighting, publication |
| Agent | Entity that does work | What is doing the work? | An LLM-based classifier processing occupation descriptions |
| Agency | Granted decision-making authority | What decisions can it make? | Authority to assign a 4-digit NAICS code without human review |
| Agentic | Behavior where agency is exercised | How much can it adapt? | Pipeline that adjusts its own confidence threshold based on batch error rate |
| Tool | Single discrete operation | What can it do? | API call to Census geocoder, database lookup of prior-year codes |

The most important clarification: *agency is a design choice*. You decide how much decision-making authority to grant. The LLM does not grant itself authority.

```{admonition} Why the vocabulary matters in practice
:class: tip
When someone proposes "an AI agent for automated coding," the vocabulary gives you the right questions to ask:

- *What workflow* does this agent operate within? (What are the steps before and after?)
- *What agency* is being granted? (What decisions can it make without asking a human?)
- *What tools* does it have access to? (Can it write to the production database? Query external APIs?)
- *Under what conditions does it stay in the loop versus escalate?*

A proposal that cannot answer these questions is not ready for a design review, let alone procurement.
```

These definitions matter operationally. OMB M-24-10 (OMB, 2024) required each federal agency to designate a Chief AI Officer (CAIO) and maintain an AI use case inventory. OMB M-25-21 (OMB, 2025) rescinded and replaced M-24-10 but retained the CAIO requirement, assigning the CAIO specific responsibilities including certifying waivers and approving certain high-impact AI uses. When your CAIO asks "is this system an AI agent?", the vocabulary above provides a precise answer: a system with no granted agency is a tool; a system with granted decision-making authority within a defined scope is an agent. The use case inventory needs this distinction.

---

## 3. The loop: observe-decide-act-check

All agentic behavior follows the same underlying pattern. Recognizing this pattern makes it easier to design pipelines, identify where things can go wrong, and specify where human checkpoints belong.

The loop has four stages: *Observe* (take in information), *Decide* (determine what to do), *Act* (execute the action), and *Check* (did it work? done / continue / stop?). The Check stage is the human checkpoint: done, flag, or escalate. When Check triggers another pass, control returns to Observe with new information about the pipeline's state.

See `examples/chapter-13/02_loop_visualization.py` for the loop diagram as a matplotlib figure.

### 3.1 A familiar example: the recipe workflow

Before applying the loop to federal statistics, here is a simpler case that makes the pattern visible.

A recipe pipeline from the Agents 101 course materials illustrates how explicit decision criteria and hard stops work at low stakes. The pipeline proceeds through five stages. In the Decide stage, the model selects a recipe using explicit criteria (rating, source credibility, prep time) with no human involvement. At the allergen check, a hard stop triggers human involvement: when a conflict is detected, the model stops, explains the conflict, and asks the user to decide whether to substitute or choose another recipe. The Check stage asks whether anything is already in the pantry and regenerates accordingly.

What to notice: decision points have explicit criteria; hard stops require human involvement; the Check stage feeds back into the pipeline; every step has a defined outcome including failure paths. See `examples/chapter-13/02_loop_visualization.py` for the full annotated version.

### 3.2 The loop in a federal statistical pipeline

An automated survey coding pipeline maps onto the loop as follows. The Observe stage receives a batch of open-ended occupation descriptions from ACS collection, pulls prior-year codes as reference, and checks batch quality indicators. The Decide stage selects the coding scheme (SOC 2018 or NAICS 2022), selects the model, and sets the confidence threshold (adjusting for high-risk subpopulations). The Act stage runs the LLM-based classifier, assigns codes with confidence scores, and flags low-confidence cases for routing. The Check stage compares against a human-coded sample, evaluates whether the error rate is within acceptable tolerance, checks subgroup accuracy (see Chapter 12), and either approves, adjusts parameters for a re-run, or escalates to human review.

The loop is not sequential and linear. The Check phase feeds back into Observe with new information about batch quality and may cycle multiple times before a batch is released. See `examples/chapter-13/02_loop_visualization.py`.

### 3.3 Agent patterns: single, chain, and reviewer

Not every agentic pipeline is a single agent running one loop. Once you understand the loop, three structural patterns emerge, each suited to different problems.

A *single agent* is one model, one task, one loop iteration. It receives an input, observes, decides, acts, checks, and returns a result. An LLM classifier that takes a single occupation description and returns a NAICS code with a confidence score is a single agent. The pattern is simple, auditable, and appropriate for well-defined tasks where the input and output are both narrow.

An *agent chain* is a sequence of single agents where the output of one feeds the input of the next. Each agent in the chain has its own defined scope and constraints. A three-stage pipeline that first cleans an open-text response, then codes it to a NAICS taxonomy, then validates the code against a reference table is an agent chain. Each stage does one thing well, with a defined checkpoint between stages. This is Principle 6 (Digestible chunks) in structural form.

An *agent-as-reviewer* puts a second model instance in the role of checking the first model's output. The reviewer does not take independent action; it evaluates and either endorses or flags the primary agent's decision. This pattern trades speed for accuracy: running two models is slower and more expensive than one, but the cross-validation surfaces disagreements that neither model would catch alone. The Federal Survey Concept Mapper (Section 8) uses exactly this pattern: two classifiers run independently, their outputs are compared, and disagreements trigger an arbitrator or human review.

| Pattern | Structure | Federal example | When to use |
|---------|-----------|-----------------|-------------|
| Single agent | One model, one loop | Code a single occupation description | Narrow, well-defined task; prototype or low-volume use |
| Agent chain | Sequential agents; output of A feeds B | Clean, then code, then validate survey responses | Complex task decomposed into auditable stages |
| Agent-as-reviewer | Primary model plus reviewer model | Dual-classifier NAICS coding with agreement check | High-stakes decisions requiring cross-validation |

---

## 4. Autonomy is a dial, not a switch

The most consequential design decision in any agentic pipeline is: how much authority does the system exercise on its own?

This is not a binary choice between "full automation" and "human does everything." It is a continuous dial with many positions. Moving right (more autonomy) offers speed and scale. It also means less predictability, harder auditability, and higher consequences when things go wrong.

The five positions on the dial run from *fully human* (AI provides information only) through *AI proposes, human approves before action*, then *AI acts, human reviews after the fact*, then *AI acts, exceptions escalate to human*, and finally *fully autonomous, no human review*. Federal statistical operations belong in the first two positions.

See `examples/chapter-13/03_autonomy_dial.py` for the autonomy dial visualization.

Rule of thumb: move right only when you have to. Stay left when you can.

"Have to" means specific, auditable decisions where: the decision volume makes human review impractical; errors can be caught and corrected before they reach published data; and the cost of an error is proportionate to the saved human review burden. Federal statistical production almost never satisfies all three conditions for full autonomy. Bounded agency -- AI proposes, human approves -- is the appropriate default for anything that goes into published statistics.

### 4.1 The cost-benefit of moving right

Every step right on the autonomy dial trades predictability for speed. For federal statistical production, the predictability side of this tradeoff carries extra weight:

- *Accountability:* Federal agencies are legally accountable for the accuracy and fairness of their statistical products. An autonomous system that makes an error does not absorb the accountability; the agency does.
- *Auditability:* Published statistics must be reproducible and auditable. A pipeline where the AI made decisions without logging its rationale is not auditable.
- *Public trust:* Federal statistics are used for congressional apportionment, resource allocation, and policy. Errors in these outputs have political and legal consequences. The cost of an error is not just statistical; it is institutional.

None of this means "never automate." It means "automate with explicit constraints, human checkpoints, and audit trails." That is exactly what bounded agency provides.

---

## 5. The six design principles

These six principles do not change as tools evolve. They are grounded in what makes agentic systems fail, documented by the Microsoft AI Red Team (2025) and consistent with good engineering practice across domains.

See `examples/chapter-13/04_design_principles.py` for the full structured output with federal examples and failure consequences for each principle.

1. **Good judgment upfront** -- Design quality bounds output quality. AI amplifies your process. A poorly specified survey coding prompt produces bad codes at scale, regardless of model capability. Time spent on design is not wasted; it determines the ceiling on output quality. *If ignored:* misinterpretation of instructions, agent misalignment, hallucinated codes that look plausible but are wrong.

2. **Agency requires governance** -- Less agency is often better. Giving an agent more authority is a tradeoff: more flexibility and less predictability. Start with the least authority that accomplishes the task. *If ignored:* actions outside intended scope, user harm from excessive autonomy, cascading errors when an unconstrained decision propagates through downstream steps.

3. **Most problems do not need agents** -- Simple solutions beat complex ones. An agent adds value only when: the task has genuine variability that cannot be pre-scripted, decisions must be made at scale, and the cost of human attention exceeds the cost of imperfect automation. A regex-based address parser or a lookup table for common occupation phrases may outperform an LLM and will always be more auditable. *If ignored:* organizational knowledge loss, unnecessary attack surface, dependency on vendor infrastructure.

4. **Specification is the skill** -- Clarity beats capability. You do not need to code to work with agents effectively. You need to think clearly: what exactly do you want, what constraints apply, what does success look like, and what should happen when things go wrong. A precisely specified prompt from a methodologist outperforms a vague prompt from an AI enthusiast, regardless of model. *If ignored:* incorrect permissions, accountability gaps, transparency failures.

5. **Design for uncertainty** -- Plan for failure, not just success. Things will go wrong. Every step in a pipeline needs a defined failure path. An agent that guesses when uncertain is worse than one that stops and asks. A survey coding pipeline that receives a description in a language the model has never seen should route to a bilingual coder, apply a low-confidence flag, or assign an 'undetermined' code -- not produce a plausible but wrong code with high apparent confidence. *If ignored:* human-in-the-loop bypass, cascading failures, denial of service.

6. **Digestible chunks** -- Focused beats sprawling. Context windows have hard limits. Model performance degrades before those limits. A pipeline broken into discrete steps with defined inputs and outputs outperforms one massive prompt trying to do everything. Decompose complex tasks into stages; let each stage do one thing well. *If ignored:* resource exhaustion, loss of data provenance, hallucinations from overloaded context, subtle degradation.

The meta-principle: AI amplifies your process. A bad process plus AI produces faster bad outcomes. A good process plus AI produces faster good outcomes. What it multiplies is up to you.

### 5.5 The prompt-as-agent pattern

One of the most liberating realizations in agentic AI design is this: you do not need a framework to build an agent. A well-specified system prompt is itself an agent specification.

The prompt defines everything that matters: the role the model plays, the constraints on its behavior, the tools it can use, the failure paths when something goes wrong, and the success criteria for a correct output. If those elements are clear in the prompt, the model executes them. If they are vague, the model fills the gaps with its own judgment, and you have just delegated design decisions to the model without realizing it. This is why Principle 4 (Specification is the skill) is not just about quality; it is about control.

For federal statistical applications, this means a methodologist with deep subject matter knowledge but no programming background can author the core of an agentic system. The specification is the work. A senior occupation coder who knows the NAICS taxonomy, the edge cases, the escalation conditions, and the acceptable confidence thresholds has everything needed to write an effective agent specification. The engineering layer that wraps it (API calls, logging, routing) is secondary to getting the specification right.

Here is a structural template for an occupation coding agent prompt:

```{code-block} text
Role: NAICS 2022 occupation coding assistant (Economic Census).
Taxonomy: NAICS 2022 only.
Output: naics_code | confidence [high/medium/low] | rationale | escalate [true/false]
Confidence: high = unambiguous; medium = best with alternatives; low = multiple defensible codes.
Escalate when: confidence is low; multiple industries; non-English input; fewer than 5 words.
Constraints: never assign a code when confidence is low -- escalate instead; do not modify input text.
```

The prompt above is the agent. It specifies the taxonomy version, the output format, the confidence tiers, and the escalation conditions. A developer can wrap it in five lines of API call code. The methodologist who knows NAICS is the one who can actually write it correctly.

---

## 6. Failure modes

The Microsoft AI Red Team (2025) has documented a taxonomy of failure modes in agentic AI systems. Almost every failure traces back to ignoring one of the six design principles.

See `examples/chapter-13/05_failure_modes.py` for the full taxonomy printed with federal examples and detection strategies. The table below summarizes the six failure categories.

| Failure mode | Description | Design principle violated |
|---|---|---|
| Misalignment | System pursues a goal that is misspecified or different from what was intended | 1 (Good judgment upfront) |
| Actions outside intended scope | Agent takes actions the designer never intended to authorize | 2 (Agency requires governance) |
| Cascading failures | Error in one step propagates unchecked through downstream steps | 5 (Design for uncertainty) and 6 (Digestible chunks) |
| Organizational knowledge loss | Logic is buried in model behavior rather than in auditable rules | 3 (Most problems do not need agents) and 4 (Specification is the skill) |
| Accountability gaps | No clear record of who or what made a specific decision, or why | 4 (Specification is the skill) -- audit requirements not specified |
| Human-in-the-loop bypass | Designed checkpoints are skipped in practice | 5 (Design for uncertainty) -- oversight not designed to be practical |

Federal examples illustrate each: a coding pipeline optimizing for throughput instead of accuracy (misalignment); a pipeline given broad 'edit data' permission that auto-corrects respondent answers (scope); an occupation coding error in step 3 that propagates through weighting to published estimates (cascading); a deployed system where nobody can explain why a specific decision was made (knowledge loss); an audit with no decision trail (accountability); and coders required to review low-confidence cases who approve in bulk because the interface makes individual review tedious (bypass).

These failure modes are not hypothetical. Chapter 10's analysis of the specification gap, enforcement gap, and impact gap in federal SDL governance applies equally to agentic AI deployments. An agentic pipeline that processes confidential data without SDL-appropriate constraints is a specification gap in action.

---

## 7. Chat vs. API: drawing the line

Every federal statistician who has used a chat interface for exploratory analysis has used AI effectively. That is not the same as using AI in production.

See `examples/chapter-13/06_chat_vs_api.py` for the full comparison. The key dimensions are summarized below.

| Dimension | Chat (e.g., Claude.ai, ChatGPT) | API-based pipeline |
|---|---|---|
| What it is | Conversational interface; single session; no persistent state | Programmatic access; version-controlled prompts; logged outputs; reproducible |
| Good for | Exploration, one-shot questions, drafting text, brainstorming | Reproducible workflows, batch processing, auditable research, published statistics |
| Reproducible? | No. Conversation history is not versioned. Re-running the same prompt may produce different output. | Yes, with version-pinned models, logged inputs/outputs, seeded randomness. |
| Auditable? | No. No audit trail. Export of conversation is manual and informal. | Yes, with structured logging, decision traces, and input/output records. |
| Privacy-compliant? | Depends on data classification and FedRAMP authorization. Sending microdata through a public chat interface is not appropriate. | Requires FedRAMP authorization and data governance controls. Can be made compliant with proper architecture. |
| Appropriate for published statistics? | No. Not reproducible, not auditable, not defensible to DRB. | Potentially yes, with proper governance, logging, and human review. |

```{admonition} The governance conversation about API access
:class: note
Many federal environments restrict or prohibit API access to commercial AI models. This is a governance constraint, not a technical limitation. The appropriate response is not to route around it using chat interfaces. It is to engage the governance process.

Chapter 14 (Evaluating AI Systems for Federal Use) provides the evaluation framework for making the case. If a workflow genuinely requires API access to be reproducible and auditable, that argument should be made through the agency's AI governance process, supported by the NIST AI RMF (NIST, 2023) and FCSM quality standards.

Using chat interfaces to substitute for API pipelines does not solve the governance problem. It creates a reproducibility problem on top of the governance problem.
```

### 7.1 When chat is appropriate

Chat interfaces remain valuable tools for specific purposes. The key is matching the tool to the task.

Chat is appropriate for: exploring a new dataset to generate hypotheses; drafting sections of a methodology report for human editing; getting a quick explanation of an unfamiliar statistical method; brainstorming approaches to a research problem; reviewing your own work for clarity or logical gaps; and learning about a new regulatory or technical standard.

An API-based pipeline is required for: batch processing of survey responses for publication; automated coding that will be incorporated into microdata; imputation that produces values entering official estimates; any output that goes into a published statistical product; any analysis that must be reproducible by an external reviewer; and any decision process that requires an audit trail.

The line is reproducibility and auditability. If it goes into published statistics, it needs a pipeline, not a conversation.

API access is not free. Before proposing an agentic pipeline, estimate the per-record API cost at your production volume. Chapter 12's cost analysis framework applies directly here. A pipeline that processes three million records at one cent per record costs $30,000 per run -- still far cheaper than human coding for many programs, but not negligible in a federal budget context where the procurement process for cloud AI services may take 6 to 12 months.

---

## 8. Case study: Federal Survey Concept Mapper

```{admonition} Source note
:class: note
The results in this case study are from an unpublished internal evaluation conducted by the author (Webb, unpublished internal evaluation, U.S. Census Bureau). The architecture and design principles are the pedagogically important elements.
```

The Federal Survey Concept Mapper is a project that demonstrates bounded agency at federal scale. It is described here as an existence proof, not as a template. The specific architecture is less important than the design choices it illustrates.

The problem: the Census Bureau operates 46 surveys. Approximately 7,000 questions across those surveys overlap, duplicate, or relate to each other in undocumented ways. Nobody had a comprehensive map of how these questions relate to the Bureau's official concept taxonomy. Manual coding would have required hundreds of hours, and human coders would have disagreed on edge cases with no audit trail for individual decisions. A fully autonomous agent would be fast and impressive-looking -- and wrong in unpredictable ways, with no confidence quantification, no human review for hard cases, and no defensibility to subject matter experts or DRB.

The bounded agency solution used two classifiers running independently. When both agreed with confidence above 0.90, the question was auto-assigned. When both agreed but the confidence indicated the question spanned two concepts, it was auto-flagged as dual-modal. When the classifiers disagreed, a bounded arbitrator handled the case. When the arbitrator found the case too ambiguous, it routed to human review.

Illustrative results from this architecture (agreement measured using Cohen's kappa; Cohen, 1960; Landis & Koch, 1977 interpretation scale):

| Metric | Value |
|---|---|
| Questions processed | ~7,000 |
| Surveys covered | 46 |
| Categorization success rate | ~99% |
| Model agreement rate (topic level) | ~89% |
| Cohen's Kappa | ~0.84 (almost perfect) |
| Dual-modal questions (spans two concepts) | ~2-3% |
| Flagged for human review | <1% |
| Total cost | ~$15 in API calls |
| Total runtime | ~2 hours |

What $15 and 2 hours replaced: weeks of work and thousands of dollars in manual coder time, with a complete audit trail the manual process could not have provided.

The bounded agency architecture compared to a fully autonomous alternative:

| Dimension | Fully autonomous agent | FSCM bounded agency |
|---|---|---|
| Error detection | None built in | Cross-validation catches errors |
| Confidence quantification | Implicit (no tiers) | Explicit confidence tiers |
| Edge case handling | Guesses with confidence | Flags for human review |
| Audit trail | Murky | Every decision documented |
| Silent failure mode | Yes -- errors look like outputs | No -- ambiguity is surfaced |
| Defensibility to DRB | Hard | Yes -- complete audit trail |

Lesson: bounded agency is slower to design. The output is defensible. "Always start with a small sample run. It is a do-loop until done right."

See `examples/chapter-13/07_fscm_case_study.py` for the full case study output and architecture diagram.

---

## 9. Agent specification template

Before any agentic pipeline goes to design review, the specification should be complete enough to fill in every blank in the following template. If you cannot fill in every blank, the design is not ready.

```{code-block} text
As a [ROLE], I want the agent to [NARROW OUTCOME] so that [BUSINESS VALUE].

BOUNDARIES
  Allowed to see:    [data sources and systems -- be specific]
  Allowed to do:     [propose flags / assign codes / query reference tables]
  Must NEVER:        [write to production database / contact respondents]

UNCERTAINTY HANDLING
  If unsure:         [flag / use default / stop -- never guess silently]
  Low confidence:    [route to human review / assign 'undetermined' code]
  Malformed input:   [defined failure path -- not 'try anyway']

HUMAN CHECKPOINTS + AUDITABILITY
  Review before:     [actions requiring prior human approval]
  Review after:      [outputs requiring ex-post review]
  Log every decision: inputs / output / confidence / model version / timestamp

SUCCESS CRITERIA
  Primary metric:    [what does 'working correctly' look like?]
  Subgroup check:    [accuracy by demographic group -- see Chapter 12]
  Unacceptable:      [what triggers rollback or escalation?]
```

Reviewing agent behavior (checklist):
- Did it stay within its job description and data boundaries?
- Is the action correct in context?
- Is the rationale understandable?
- Would I sign my name under this action?

If any answer is "no," the example is a failure mode. Collect these. They become your test cases.

---

## 10. Agentic pipelines and State Fidelity Validity (bridge to Chapter 15)

Agentic pipelines are the context where the SFV threats (Chapter 15) become most acute. Every observe-decide-act-check cycle potentially updates the pipeline's working context. Multi-session pipelines accumulate state across sessions. And the state that accumulates includes exactly the kinds of decisions (methodology choices, parameter values, exclusion criteria) that the SFV threat taxonomy is designed to protect.

The five SFV threats map directly onto stages of the agentic loop. See `examples/chapter-13/08_sfv_bridge.py` for the full structured mapping with countermeasures.

| SFV threat | Agentic context | Loop stage affected | Countermeasure |
|---|---|---|---|
| T1: Semantic Drift | Terminology established in session 1 mutates across the multi-session pipeline without explicit redefinition | DECIDE: decisions made using drifted terminology are wrong | Config-driven vocabulary: terms defined externally, injected at each session start |
| T2: False State Injection | Pipeline 'remembers' a decision to use method X that was never made; subsequent DECIDE stages proceed on fabricated basis | DECIDE: false premise produces wrong action | Decision log: external record of every decision; periodic state reconciliation diffed against log |
| T3: Compression Distortion | Context compaction strips rationale from exclusion decision; later DECIDE stages cannot distinguish justified from unjustified exclusions | OBSERVE: compressed state provides incomplete information for decision | Documentation-as-traceability: rationale written to external document before compaction can strip it |
| T4: State Supersession Failure | Privacy budget (epsilon) revised in session 4, but ACT stage in session 5 applies the old value | ACT: correct decision, wrong parameter | Parameter audit: after any revision, verify new value is operative before next action |
| T5: State Discontinuity | New session starts without session 1 methodology; the OBSERVE stage begins from near-zero state | OBSERVE: incomplete picture drives all downstream decisions | Handoff documents: explicit state serialization at every session boundary |

Chapter 15 (Capstone) formalizes this framework. This chapter identifies the pipeline context where SFV threats are most acute. The key insight: agentic pipelines create state, and that state must be managed.

The reproducibility checklist in Chapter 15, Section 10, provides the operational tool for verifying that an agentic pipeline maintains state fidelity across its full operational lifecycle.

```{admonition} From Chapter 12 to Chapter 13 to Chapter 15
:class: note
Chapter 12 (LLMs for Survey Operations) showed what a single-session LLM-based pipeline looks like. Chapter 13 (this chapter) extends that to multi-step, potentially multi-session pipelines where the pipeline exercises decision-making authority. Chapter 15 (Capstone) provides the validity framework for ensuring those pipelines maintain methodological integrity across their full operational history.

The progression: tool (Chapter 12) to workflow (Chapter 13) to trustworthy process (Chapter 15).
```

---

## 11. Exercises

### Exercise 13.1: Map a federal task to the loop

**Scenario:** Your division is proposing to automate the initial coding of occupation descriptions in the Economic Census. The system would receive open-text responses and assign a preliminary 6-digit NAICS code. Human coders review flagged cases.

**Task:**
1. Map this task to the observe-decide-act-check loop. For each stage, specify: what is the input, what action is taken, what is the output, and who is involved?
2. Identify at least two decision points where human checkpoints should exist. Justify your placement.
3. What information should be logged at the ACT stage to ensure auditability?

Starter code for mapping your answers into structured form is in `examples/chapter-13/09_exercises.py`.

### Exercise 13.2: Rate the autonomy dial

**Task:** Using the occupation coding pipeline from Exercise 13.1:

1. On the autonomy dial, where would you position the proposed system? (Fully human / AI proposes-human approves / AI acts-human reviews / AI with exceptions / Fully autonomous)
2. Justify your position with reference to the three conditions that justify moving right on the dial: decision volume, catchability of errors, and proportionality of error cost.
3. What specific change to the pipeline design would allow you to move one step right? What new risk would that introduce?

### Exercise 13.3: Apply the six principles

**Scenario:** A colleague proposes building an "autonomous data quality agent" that:
- Ingests paradata from the current collection wave
- Identifies interviews with anomalous duration, item nonresponse patterns, or geographic inconsistencies
- Automatically flags those interviews for removal from the published microdata

**Task:** Using the six design principles, evaluate this proposal. For each principle, identify whether the proposal satisfies it, violates it, or requires more information to determine. Recommend at least three specific modifications that would make the system appropriate for federal statistical production.

Starter code with the six-principle evaluation scaffold is in `examples/chapter-13/09_exercises.py`.

---

## 12. Key takeaways

1. *Shared vocabulary prevents misalignment.* Workflow, agent, agency, agentic, and tool have precise meanings. Use them precisely before any design discussion.

2. *The observe-decide-act-check loop is universal.* Learn it once, apply it everywhere. Recognizing where each stage happens reveals where checkpoints belong.

3. *Autonomy is a dial.* Federal statistical operations belong on the left side. Move right only when you can demonstrate that the decision volume, error catchability, and error cost proportionality all justify it.

4. *Most problems do not need agents.* A lookup table, a regex, or a simple script may outperform an LLM for well-defined, high-frequency tasks -- and will always be more auditable.

5. *The six design principles are not cautionary tales.* They are how you build systems that work. Every documented failure mode in agentic AI traces back to violating one of them.

6. *Chat is for exploration. API is for production.* If it goes into published statistics, it needs a reproducible, logged, auditable pipeline -- not a conversation.

7. *Agentic pipelines create state.* That state must be managed explicitly across sessions and stages. Chapter 15 (SFV) formalizes why this is a validity requirement, not just good practice.

8. *The progression from Chapter 11 (transformer architecture) through Chapter 12 (LLM application) to this chapter (agentic pipeline design) to Chapter 14 (AI system evaluation) is deliberate.* Each chapter adds a layer: understanding the model, using the model, governing the pipeline, and evaluating the system.

```{admonition} How to explain agentic AI to leadership
:class: dropdown

**On what 'AI agent' actually means:**
"An AI agent is a system that has been granted authority to make decisions within a defined scope. The critical words are 'granted' and 'defined scope.' We decide what authority it has. We decide the scope. A well-designed agent does not exceed its brief. An agent that is not well-designed may take actions we never intended to authorize."

**On why bounded agency rather than full automation:**
"Full automation is fast and impressive. It is also wrong in ways that are hard to predict and harder to audit. When a human coder makes an error, we have a clear accountability path: who made the decision, when, with what information. When an autonomous system makes an error across 100,000 cases, the audit question becomes: who is responsible? The answer has to be us, because we built and deployed the system. Bounded agency (AI proposes, human approves) means we maintain that accountability without giving up the productivity benefits."

**On why this matters for published statistics:**
"Our statistics are used for congressional apportionment, federal funding allocation, and national economic measurement. An error is not just a statistical mistake; it can redirect billions of dollars in resources or misrepresent the size of a congressional district. The standard of care for building the pipelines that produce those statistics must match the stakes. An agentic system that cannot be audited should not be in that pipeline."
```
