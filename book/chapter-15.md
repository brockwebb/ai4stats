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

# Chapter 15 - Capstone: Reproducible AI-Assisted Research

```{contents}
:depth: 2
```

## Learning goals

By the end of this chapter, you will be able to:

1. Explain why classical validity types are insufficient for AI-assisted research pipelines
2. Define State Fidelity Validity (SFV) and its five sub-dimensions
3. Identify the five SFV threats (T1-T5) in realistic pipeline scenarios
4. Apply the SFV evaluation framework to assess a multi-session AI-assisted analysis
5. Recognize engineering countermeasures and evaluate which threats each addresses
6. Connect SFV to the NIST AI RMF TEVV framework introduced in Chapter 14
7. Use the reproducibility checklist to evaluate AI-assisted federal research

---

## 1. Setup

This chapter is a synthesis chapter. Its purpose is to integrate the methods and frameworks from every preceding chapter into a coherent, defensible approach to AI-assisted federal research. The demonstration code in this chapter is minimal by design: it illustrates one specific measurement problem (terminology drift) rather than introducing a new machine learning method. Full scripts are in `examples/chapter-15/`.

See `examples/chapter-15/01_setup.py` for imports and reproducibility configuration.

---

## 2. The reproducibility problem in AI-assisted research

Every introductory methods course teaches reproducibility the same way: same data plus same code equals same results. Document your data sources, version your code, seed your random number generators, and another researcher can reproduce your findings exactly. This is good advice. It is also insufficient.

AI-assisted research pipelines introduce a complication that classical reproducibility frameworks do not address. The issue is not stochasticity, though that matters. The issue is that an AI-assisted research pipeline does not have a fixed instrument.

### 2.1 What is the instrument?

In classical survey methodology, the instrument is the questionnaire. It is designed once, tested, and administered. Every respondent receives the same questions in the same order. If the questionnaire changes mid-fielding, that is a catastrophic methodological failure.

In an AI-assisted pipeline, the instrument is the combination of (fixed model weights) plus (mutable context buffer). The context buffer is the conversation history, the accumulated decisions, the terminology established across prior sessions, the intermediate findings recorded in the pipeline's working state. This composite is the instrument. And unlike a questionnaire, it changes with every interaction.

This is not a flaw unique to AI. Any stateful research process has this vulnerability. A long-running human analysis team experiences the same problem through personnel turnover, informal knowledge, undocumented decisions, and terminology drift across team members. LLMs make it acute because the context buffer is both the instrument and the working memory, and it is subject to automated modification in ways that are invisible to the researcher.

### 2.2 Three layers of the problem

The reproducibility challenge in AI-assisted research has three distinct layers, each requiring a different response. See `examples/chapter-15/02_three_layers.py` for the full display.

Layer 1 is stochastic outputs: the same prompt can produce different outputs across runs. The partial fix is setting temperature to zero and seeding the random state, but many APIs do not guarantee determinism even then.

Layer 2 is prompt sensitivity: minor prompt variations produce substantially different outputs. Versioning prompts and testing for stability helps, but does not address accumulated state.

Layer 3 is state accumulation failures: the accumulated context degrades, drifts, or is lost across a multi-session pipeline. No classical solution exists for this layer. This is the SFV problem.

### 2.3 The instrument is a moving target

Here is a concrete example. Suppose you are conducting a multi-session AI-assisted analysis of survey nonresponse patterns. In session 1, you establish that you will use "unit nonresponse" to mean any case where the entire household refused to participate. In session 3, the model begins using "nonresponse" to refer to both unit and item nonresponse without distinguishing them. By session 5, your analysis is conflating two distinct phenomena. The output looks fluent. The pipeline is running. Nothing visibly breaks. But the methodology has silently degraded.

This is not a bug in the model. It is a structural vulnerability in any stateful research process where the accumulated working context is mutable, subject to drift, and not independently audited.

---

## 3. Classical validity types and their limits

Before introducing State Fidelity Validity, it is worth understanding what classical validity frameworks assume about research instruments, and why those assumptions fail in AI-assisted pipelines.

### 3.1 The four classical types

Each classical validity type has a core question, an assumption about the instrument, and a specific way that AI-assisted pipelines violate that assumption. See `examples/chapter-15/03_classical_validity.py` for the full display.

The four types and their violations:

- *Construct validity* (are you measuring what you claim?) assumes the instrument is defined and stable. The context buffer changes the operative construct mid-execution.
- *Internal validity* (are causal inferences warranted?) assumes the instrument does not change during measurement. Accumulated state mutates; confounders introduced mid-pipeline.
- *External validity* (do findings generalize?) assumes the instrument behaves consistently across contexts. Session restarts produce different operative states from the same prompt.
- *Statistical conclusion validity* (are statistical inferences warranted?) assumes consistent, interpretable measurements. Terminology drift means statistical quantities may reference different constructs across sessions.

### 3.2 The shared assumption

Each classical validity type assumes the instrument is defined, stable, and consistent. A questionnaire cannot spontaneously reword its own questions. A laboratory scale cannot decide to measure in different units from run to run. A regression model, once estimated, does not change its coefficients.

An AI-assisted research pipeline violates all four assumptions. The context buffer changes with every interaction. Terms established in turn 10 may be silently overwritten by turn 40. Decisions made in session 1 may not exist in session 2. Compaction events may strip the rationale from data exclusion decisions.

This is not an argument against AI-assisted research. It is an argument for a validity framework that directly tests the assumption that classical frameworks leave implicit: that the instrument maintained the integrity of its accumulated state.

### 3.3 Positioning State Fidelity Validity

State Fidelity Validity is not a replacement for the classical types. It is a continuous dimension of measurement quality that the classical framework leaves implicit. Every stateful pipeline has some degree of state degradation -- the question is not whether SFV failures exist, but whether they remain within acceptable limits for the inferential claims being made. When state degradation exceeds those limits, claims about construct, internal, external, and statistical conclusion validity are on shaky ground. SFV is how you audit and manage that risk, not how you eliminate it.

The analogy: SFV is to AI-assisted pipelines what instrument calibration is to laboratory equipment. A miscalibrated scale does not just affect measurement precision. It undermines every claim that depends on the measurements. Similarly, a pipeline with poor state fidelity does not just introduce noise. It corrupts the methodological record that every downstream inference depends on.

---

### 3.4 How SFV threats map to classical validity failures

The precondition argument is not abstract. Each SFV threat has a primary home in the classical validity framework -- a specific validity type it degrades through a specific mechanism. The table below maps the five threats defined in Section 5 to their primary classical targets. See `examples/chapter-15/03b_sfv_classical_crosswalk.py` for the full display.

| SFV Threat | Primary Classical Validity Degraded | Mechanism |
|------------|--------------------------------------|----------|
| T1: Semantic Drift | Construct Validity | If terminology mutates mid-pipeline, the operative construct changes. The pipeline is no longer measuring what it defined at the outset. |
| T2: False State Injection | Internal Validity | Confabulated decision history breaks the causal chain. Inferences at step N rest on a methodological record that never occurred. |
| T3: Compression Distortion | Statistical Conclusion Validity | Compaction strips caveats and collapses conditional findings into unconditional ones. Statistical inferences downstream operate on distorted premises. |
| T4: State Supersession Failure | Internal Validity | A persisting outdated parameter is a systematic confound the researcher believes was controlled. The analysis uses one value while the methodology claims another. |
| T5: State Discontinuity | External Validity | Findings are bound to the specific execution context. They do not generalize even to a re-run of the same pipeline with a session restart. |

These mappings are not exclusive. T2 also degrades construct validity (a confabulated method choice changes what is being measured). T3 also degrades internal validity (stripped rationale removes the basis for a design decision). T1 can cascade into statistical conclusion validity if drifted terms cause statistical quantities to reference different constructs across sessions.

But the primary mappings reveal why SFV was invisible to the classical framework. Each classical validity type assumes a stable instrument. The four types collectively test whether the instrument measures the right thing (construct), whether design choices support causal inference (internal), whether findings transfer (external), and whether statistical inferences hold (statistical conclusion). None of them test whether the instrument changed during the measurement process, because traditional instruments do not change. SFV threats are precisely the ways a stateful instrument can change -- and each mode of change maps to a classical validity failure that the framework assumed away.

This is the theoretical contribution: SFV does not compete with the classical types. It guards the assumption they all share.

---

## 4. State Fidelity Validity: definition and sub-dimensions

### 4.1 Formal definition

*State Fidelity Validity* is the degree to which an AI-assisted research or analytic pipeline preserves the accuracy and integrity of its accumulated internal state (decisions, terminology, methodology, and intermediate findings) across sequential operations, such that inferences at step N remain warranted by the actual history of steps 1 through N-1, rather than by degraded, distorted, confabulated, or selectively retained versions of that history.

Abbreviation: SFV. Always capitalized, no periods.

SFV is not binary. A pipeline does not "have" or "lack" state fidelity in the way a study design has or lacks random assignment. State fidelity is a continuous quantity, and the practical question is always whether degradation stays within tolerances acceptable for the claims being made. This is analogous to statistical quality control: a manufacturing process does not aim for zero variance; it aims for variance within specification limits. The stochastic nature of AI-assisted tools imposes a stochastic tax on every pipeline that uses them. SFV provides the framework for quantifying that tax, monitoring it, and determining when it has exceeded the threshold at which downstream inferences are no longer defensible. The severity scale in Section 9 and the operationalization metrics in Section 8 are the tools for making that determination.

```{admonition} Why "validity" and not "reliability"?
:class: note
This distinction matters. Reliability means consistency: a reliable instrument produces the same result repeatedly. An AI pipeline can reliably produce the same wrong provenance claim across multiple runs. That is not a reliability problem; it is a validity problem. The pipeline is not measuring what it claims to measure, because its operative state has diverged from the actual history of the research.

State degradation alters the construct and the methodology mid-execution. That is a validity failure.
```

### 4.2 The five sub-dimensions

SFV is not a single dimension. It decomposes into five testable sub-dimensions, each corresponding to a distinct kind of state failure. See `examples/chapter-15/04_sfv_subdimensions.py` for the full display with success examples and failure descriptions.

| Shorthand | Full Name | Definition |
|-----------|-----------|------------|
| TC | Terminological Consistency | Vocabulary remains stable and matches externally defined terms across the full execution |
| SP | State Provenance | Outputs are traceable to actual prior steps; no invented history |
| CF | Compression Fidelity | Summarization and compaction do not distort the meaning of prior decisions |
| SC | Session Continuity | Information survives thread or session boundaries intact |
| SCoh | State Coherence | Accumulated state is internally consistent at any given point |

The failure mode for each sub-dimension is distinct. TC fails when the model begins using "nonresponse" loosely to cover both unit and item nonresponse. SP fails when the model references a decision to use logistic regression that was never made. CF fails when a compaction strips "due to measurement concerns" from an exclusion decision, leaving only the fact of exclusion without the rationale. SC fails when session 2 reinvents data exclusion criteria because session 1 context was not carried forward. SCoh fails when the pipeline simultaneously references epsilon = 0.5 and epsilon = 1.0 as the operative privacy parameter.

---

## 5. The threat taxonomy (T1-T5)

The SFV threat taxonomy provides a structured vocabulary for categorizing state fidelity failures. Each threat has a canonical name, number, and a distinct failure mechanism. See `examples/chapter-15/05_threat_taxonomy.py` for the full taxonomy with detection difficulty and severity ranges.

### 5.1 The simulated pipeline transcript

The following section describes a realistic five-session AI-assisted survey analysis constructed to demonstrate all five threat types. The full data structure is in `examples/chapter-15/06_pipeline_transcript.py`. Readers working through the exercises should run that script to load the transcript.

Session 1 establishes clean methodology: "unit nonresponse" means complete household refusals; "item nonresponse" means specific items missing in otherwise complete returns; privacy budget is epsilon = 0.5 per DRB approval; income imputation uses random forest after logistic regression failed in pilot.

Session 2 is mostly clean but plants a T1 seed: the term "partial nonresponse" is introduced informally without a definition. This term will drift in later sessions.

Session 3 contains two active failures. In turn 1, the model recommends logistic regression for income imputation as if no prior method decision existed (T2: False State Injection). In turn 2, "partial nonresponse" is used as if formally defined (T1 active: undefined term treated as defined).

Session 4 contains a T4 failure. The DRB revises the privacy budget to epsilon = 1.0 in turn 1, but in turn 2 the model applies the superseded epsilon = 0.5 to the output tables. The actual privacy guarantee differs from what the researcher believes.

Session 5 is a new session boundary. The session opens having lost: the epsilon revision to 1.0 (T4's origin), the T2 confabulation is carried forward as if fact, and the age < 16 exclusion criterion from session 2. The methodology write-up produced in this session will be factually wrong on three dimensions. T5: State Discontinuity.

### 5.2 T1: Semantic Drift (terminology mutation)

T1 is the most common SFV failure and the easiest to miss. Terminology drifts gradually across turns or sessions without any explicit redefinition. The output remains fluent. The researcher may not notice because the drifted term is close enough to the original to seem correct.

In the transcript above, "partial nonresponse" was introduced informally in session 2 and used as if formally defined in session 3. By session 5, it appears in the methodology write-up without ever having been operationalized.

The script `examples/chapter-15/07_terminology_drift.py` demonstrates a `compute_term_similarity()` function that measures how similar later usages are to reference definitions, producing the drift detection table and three-panel visualization below.

```{code-block} python
def compute_term_similarity(reference_def, later_usage):
    """
    Compute how similar a later usage is to the reference definition.
    Lower similarity suggests potential drift.
    """
    matcher = SequenceMatcher(None, reference_def.lower(), later_usage.lower())
    return matcher.ratio()
```

The function returns a similarity score between 0.0 (no shared content) and 1.0 (identical). Scores below 0.40 indicate substantial drift. In the simulated pipeline, "income imputation method" drops from 1.0 in session 2 to well below 0.40 in session 3 (T2 confabulation) and session 5 (T5 carries the confabulation forward).

### 5.3 T2: False State Injection (confabulated decisions)

T2 is the most dangerous SFV failure. The pipeline generates a confident claim about a decision that was never made. Unlike hallucination about external facts (which a researcher might fact-check), T2 produces false claims about the pipeline's own operational history. The researcher has no external source to check against; they may simply trust that the model accurately remembers what was decided.

In the simulated transcript, session 3 recommends logistic regression for income imputation as if no method decision had been made. In reality, logistic regression was explicitly tested and rejected in session 1. The confabulation is fluent, specific, and plausible. It would be easy to miss if the researcher did not have an independent log of session 1 decisions.

### 5.4 T3: Compression Distortion (compaction strips rationale)

T3 occurs when automated context compaction (or intentional summarization) preserves the conclusion of a decision but drops the rationale. The distinction matters: a decision without rationale is not auditable. An external reviewer cannot assess whether the exclusion of a record category was justified if the reason has been compressed out of the pipeline's working state.

In a typical example: "exclude records where income is below poverty threshold due to measurement concerns" compacts to "low-income records excluded." The rationale (measurement concerns) is lost. A reviewer who discovers the exclusion cannot assess whether it was appropriate. The decision itself may even be reversed in a later session without realizing the original justification existed.

### 5.5 T4: State Supersession Failure (old values persist)

T4 occurs when a decision is explicitly revised but the old value persists in the pipeline's operative state. This is particularly dangerous for quantitative parameters: privacy budgets, thresholds, model hyperparameters, sample weights. The revised value may be acknowledged in one turn and forgotten in the next, producing analyses that use obsolete parameters while the researcher believes the revision is in effect.

In the transcript, epsilon was explicitly revised from 0.5 to 1.0 in session 4, turn 1. In session 4, turn 2, the model applies the superseded epsilon = 0.5. The analysis is labeled with a 1.0 privacy guarantee but computed with 0.5 noise.

### 5.6 T5: State Discontinuity (session boundary drops context)

T5 is the broadest failure mode. Every time a session ends and a new one begins, the accumulated state must be explicitly transferred or it is lost. In the simulated transcript, session 5 begins with no knowledge of: the epsilon revision (T4's origin), the age exclusion criterion, and the method decision log from session 1. Session 5 will produce a methodology write-up that is factually wrong on at least three dimensions.

The failure is structural, not a model error. Without explicit mechanisms to carry state across session boundaries, discontinuity is the default.

---

## 6. Detecting state fidelity failures

The fundamental challenge of SFV failures is that they are *latent*. The output looks fluent. The pipeline produces results. Nothing visibly breaks. A researcher who does not explicitly check for state fidelity failures may not discover them until the work is in review or, worse, after publication.

### 6.1 Detection strategies

See `examples/chapter-15/05_threat_taxonomy.py` and `examples/chapter-15/08_state_reconciliation.py` for the full detection strategy display. The key strategies by threat type:

For T1 (Semantic Drift): periodically extract all instances of defined terms and compare against reference definitions; ask the system to define each term and diff against the session 1 definition.

For T2 (False State Injection): for any stated decision, ask the system to cite the specific turn where it was made; maintain an external log of decisions and periodically ask the model to restate all decisions and diff.

For T3 (Compression Distortion): log compaction events; after each compaction, ask the model to restate prior decisions and check for missing rationale; require that every decision statement includes a "why" component.

For T4 (State Supersession Failure): maintain an external log of all revisions; after any update, verify the new value is operative by asking the model to state current values.

For T5 (State Discontinuity): at every session end, serialize the full operative state to a structured document; begin every new session by loading the handoff document and asking the model to confirm its understanding.

### 6.2 Periodic state reconciliation

The most generalizable detection strategy is *periodic state reconciliation*: at regular intervals (every N turns, every session boundary, before any major analysis step), ask the system to restate its current understanding of all active decisions, parameters, and terminology. Then diff that restatement against your canonical external log.

This works because it makes state failures visible before they propagate. A single reconciliation check that catches a T4 failure (epsilon reverting to 0.5) prevents that failure from appearing in the final output.

The script `examples/chapter-15/08_state_reconciliation.py` simulates this check against the session 5 model state (with T2, T4, and T5 failures active), producing output like:

```{code-block} text
State reconciliation check: canonical log vs. model restatement
=================================================================
Parameter                 Canonical log             Model restatement    Status
epsilon                   1.0                       0.5                  MISMATCH -- INVESTIGATE
imputation_method         random forest             logistic regression  MISMATCH -- INVESTIGATE
age_exclusion             records with age < 16...  full adult populat.. MISMATCH -- INVESTIGATE

Reconciliation result: 3 of 5 parameters mismatched.
```

Three failures surface immediately on a simple parameter audit. Each mismatch is a potential SFV failure requiring investigation before outputs are trusted.

---

## 7. Engineering countermeasures

Identifying SFV threats is necessary but not sufficient. The goal is to build pipelines where state fidelity failures are prevented where possible and detected early where prevention fails.

The engineering countermeasures below represent implementable approaches, not theoretical proposals. Seldon, the traceability system developed for the ai4stats research project, implements all of them. The point is not that you must use Seldon; the point is that these countermeasures are real, operational, and field-tested. They are an existence proof.

See `examples/chapter-15/09_countermeasures.py` for the full countermeasure display including Seldon implementation notes.

| Countermeasure | Addresses | Mechanism |
|----------------|-----------|-----------|
| Config-driven vocabulary | T1 | Terms defined in config files; injected from config at session start, not from prior context |
| Graph-backed ontology | T1, T2 | Concepts exist in Neo4j; pipeline queries the graph rather than relying on context window memory |
| TEVV validation loops | T2, T3 | Outputs validated against external source of truth before entering research base |
| Handoff documents | T5 | Full operative state serialized at session end; loaded at session start. `seldon closeout` / `seldon briefing` |
| Documentation-as-traceability | T3, T4 | Decisions written to external documents with rationale before compaction can strip them |
| Multi-model triangulation | T2 | Key provenance claims validated by independent model instance; divergence flags confabulation |
| Periodic state reconciliation | T1, T2, T3, T4 | Regular restatement diffed against canonical log. `seldon reconcile` |

The heatmap in `examples/chapter-15/10_countermeasure_heatmap.py` visualizes coverage across all five threats. T5 has only one countermeasure (handoff documents), making it the highest-risk threat in terms of single-point-of-failure. T1 and T2 have the broadest countermeasure coverage.

---

## 8. SFV operationalization: metrics for your pipeline

SFV is not just a conceptual framework. It is operationalizable. Each sub-dimension corresponds to metrics that can be computed in practice. See `examples/chapter-15/11_sfv_metrics.py` for the full metrics display.

The six metrics and their threat mappings:

- *Terminology consistency rate*: fraction of term uses that match the reference definition within a similarity threshold. Maps to T1. Target: > 0.80 similarity for all canonical terms.
- *Reference resolution accuracy*: fraction of provenance claims where the model correctly identifies the origin session and turn. Maps to T2. Target: > 0.95; any false provenance is a serious finding.
- *Post-compaction state divergence*: difference between canonical log and model's paraphrase after a compaction event. Maps to T3. Target: < 5% content divergence on decision rationale.
- *Cross-session reconstruction error*: how accurately a new session reconstructs prior state from the handoff document alone. Maps to T5. Target: > 90% reconstruction accuracy.
- *False provenance rate*: fraction of outputs referencing decisions that never occurred. Maps to T2. Target: 0%; any false provenance triggers investigation.
- *State reconciliation pass rate*: fraction of periodic reconciliation checks where model restatement matches canonical log. Maps to T1, T2, T3, T4. Target: > 0.95 pass rate.

These metrics are not just academic proposals. They are the same metrics that would satisfy the NIST AI RMF Measure function and TEVV requirements for a federal AI deployment. The confabulation risk category in NIST AI 600-1 maps to T2 (False State Injection) in SFV terms: confabulation about external facts (hallucination) and confabulation about internal pipeline history are distinct failure modes requiring distinct countermeasures. If you are already doing TEVV for your pipeline's model outputs, adding SFV metrics extends that framework to cover the pipeline's accumulated state.

---

## 9. The severity scale

Not all SFV failures are equal. The severity scale provides a practical tool for prioritizing which failures require immediate action and which are acceptable with monitoring. See `examples/chapter-15/12_severity_scale.py` for the full display.

```{code-block} python
severity_levels = [
    {"level": "Fatal",            "informal": "Dead."},
    {"level": "Potentially fatal","informal": "Mostly dead."},
    {"level": "Recoverable",      "informal": "Mostly alive with caveats."},
    {"level": "Cosmetic",         "informal": "Alive."},
]
```

The four levels in practice:

*Fatal* (Dead): construct validity failure. The pipeline is measuring the wrong thing entirely. Example: income imputation method confabulated (T2); all imputed values use wrong methodology; downstream analysis is built on a fabricated methodological basis. Action: stop the pipeline, reconstruct from the session 1 canonical log, treat all outputs as suspect.

*Potentially fatal* (Mostly dead): cumulative uncaught state drift across sessions, corrupting the research base. Example: epsilon = 0.5 applied across four sessions after explicit revision to 1.0 (T4). All output tables have incorrect privacy guarantees; DRB approval was for 1.0 noise, not 0.5. Action: halt dissemination, audit all outputs produced under the wrong parameter, reprocess or retract.

*Recoverable* (Mostly alive with caveats): single-session failure caught and corrected before downstream use. Example: "partial nonresponse" used without definition, caught in a reconciliation check before final analysis, term operationalized retroactively. Action: document the failure, its scope, and the correction; audit affected outputs; correct or flag as provisional.

*Cosmetic* (Alive): minor terminology inconsistency with no impact on inference. Example: "survey weights" and "sampling weights" used interchangeably across two turns, both referring correctly to the same quantity. Action: document; standardize in next session via config-driven vocabulary; no reprocessing needed.

```{admonition} Severity in practice: the 48-hour rule
:class: tip
For federal statistical production, a useful heuristic: if the failure would prevent you from defending the methodology to your data review board within 48 hours, it is at minimum potentially fatal. If you would be comfortable explaining the failure in a methodology footnote, it is recoverable or cosmetic. Fatal failures are those you would not want the data review board to discover after dissemination.
```

---

## 10. Reproducibility checklist for AI-assisted federal research

The following checklist operationalizes SFV into a concrete, usable tool. It is designed for federal statisticians who use AI pipelines for any part of their research workflow, from data cleaning through analysis to write-up. See `examples/chapter-15/13_reproducibility_checklist.py` for the full display.

TEVV (Test, Evaluation, Verification, and Validation) is standard systems engineering practice defined in NIST AI RMF 1.0. It does not depend on any specific executive order; it is a measurement framework applicable to any federal AI deployment.

Each item follows the same structure (check question, SFV dimension, pass criterion, fail consequence). The first two items illustrate the pattern:

```{code-block} text
Item 1: Is the vocabulary externally defined?
  SFV dimension:  Terminological Consistency (TC)
  Pass criterion: Yes: a vocabulary document or config file exists, loaded at session start.
  Fail:           T1 (Semantic Drift) risk is unmitigated. Terms will mutate across sessions.

Item 2: Are session boundaries explicitly managed?
  SFV dimension:  Session Continuity (SC)
  Pass criterion: Yes: a structured handoff document exists for every completed session.
  Fail:           T5 (State Discontinuity) is the default. Each new session starts from near-zero.
```

Items 3 through 8 follow the same structure and cover State Provenance (SP), Compression Fidelity (CF), State Coherence (SCoh), output traceability (SP), SFV metrics in evaluation, and TEVV scope. See `examples/chapter-15/13_reproducibility_checklist.py` for the complete list.

A pipeline that answers "no" to items 1 through 3 has unmitigated risk for the most dangerous SFV threats. A pipeline that answers "yes" to all eight items does not guarantee zero SFV failures, but it has the monitoring infrastructure to detect and correct them before they reach final output.

```{admonition} How to use this checklist in practice
:class: tip
This checklist is most useful at two points: before beginning a new AI-assisted analysis (to verify that your infrastructure is in place), and as part of peer review or DRB preparation (to demonstrate that your methodology is auditable). It is not a certification scheme; it is a structured conversation starter. A thoughtful "no" that leads to a mitigation plan is more valuable than a reflexive "yes."
```

---

## 11. Connecting it all: from Chapter 1 to here

This is the capstone chapter. Its purpose is to integrate the full course arc into a coherent methodological framework. Every preceding chapter taught a specific skill or method. This section synthesizes them.

See `examples/chapter-15/14_course_arc.py` for the full course arc table and the polar wheel visualization.

| Group | Chapters | Theme | SFV Connection |
|-------|----------|-------|----------------|
| Foundations | 1-4 | Knowing what you are working with | Data provenance; unit of analysis clarity |
| Core Methods | 5-8 | Knowing what tools are available | Method selection rationale: decisions that must survive session boundaries |
| Advanced Methods | 6-8 | Knowing the limits of the tools | Exclusion decisions; imputation method choices with documented rationale |
| Privacy and Synthetic Data | 9-10 | Knowing what is at stake | Privacy budget as a state variable; T4 directly threatens privacy guarantees |
| Language Models | 11-13 | Knowing how the instrument works | The LLM is the context window; understanding the instrument is precondition for evaluating its fidelity |
| Governance | 14 | Knowing how to assess fitness for use | NIST AI RMF, FCSM Quality Standards, TEVV; SFV metrics fit within the TEVV measurement framework |
| Capstone | 15 | Knowing whether your pipeline maintained integrity | SFV IS this chapter: the validity framework that makes AI-assisted research defensible |

### 11.1 The argument for bounded agency, restated

Every chapter in this course has returned to the same design principle: AI assists, humans decide. This is not a preference. It is a methodological requirement.

State Fidelity Validity provides the technical justification for why human oversight of AI-assisted pipelines is not optional. If state fidelity failures are latent, fluent, and often invisible until an external reviewer checks the methodology, then human oversight is the primary mechanism for detecting them. A pipeline in which humans never review the accumulated state, never run reconciliation checks, and never audit terminology is a pipeline in which SFV failures will accumulate undetected.

Bounded agency is not about distrust of AI tools. It is about the methodological necessity of independent verification in any research process where the instrument can degrade silently.

```{admonition} SFV and the NIST AI RMF
:class: note
State Fidelity Validity fits within the NIST AI RMF Measure function. Specifically, it extends TEVV to cover pipeline-level state, not just model-level outputs. The eight-item reproducibility checklist maps directly to TEVV requirements: testing (item 7), evaluation (items 1, 5, 6), verification (items 3, 4), and validation (items 2, 8).

If your agency already requires NIST AI RMF compliance for AI deployments, adding SFV metrics does not require a new framework. It is an extension of the measurement practices you are already committed to.
```

---

## 12. Exercises

### Exercise C.1: Identify SFV failures in a simulated transcript

The simulated 5-session transcript presented in Section 5 contains planted SFV failures from all five threat types.

**Task:** Using the pipeline transcript in `examples/chapter-15/06_pipeline_transcript.py`, answer the following:

1. For each planted failure marked in `sfv_note`, identify the threat type (T1-T5) and the specific sub-dimension it violates (TC, SP, CF, SC, SCoh).
2. Assign a severity level (fatal, potentially fatal, recoverable, cosmetic) to each failure. Justify your severity rating.
3. Which failure would you address first if you discovered it at the start of session 5? Why?

**Expected output:** A structured table with columns: session, turn, failure type, sub-dimension, severity, justification.

See `examples/chapter-15/15_exercise_c1.py` for the starter framework with pre-populated failure descriptions and sub-dimension assignments. Students complete the severity and justification columns.

### Exercise C.2: Propose countermeasures

**Task:** For the failures identified in Exercise C.1:

1. For each failure, identify at least one engineering countermeasure (from Section 7) that would have *prevented* it, if any.
2. For failures that could not be prevented, identify the detection strategy (from Section 6) that would have caught it earliest.
3. For the T5 failure at session 5, estimate the scope of reprocessing required to restore the pipeline to a known-good state. What is the minimum recovery path?

**Reflection question:** Which of the five threat types is most amenable to prevention? Which is most amenable to detection only? Explain your reasoning.

See `examples/chapter-15/16_exercise_c2.py` for the starter countermeasure mapping framework.

### Exercise C.3: SFV assessment for an AI-assisted imputation pipeline

**Context:** Your division is proposing to use an AI assistant to support the multiple imputation workflow introduced in Chapter 7. The workflow will span multiple sessions over approximately three weeks, with up to five analysts using the same pipeline at different points. The final output will be published microdata with a federal statistical agency's imprimatur.

**Task:** Write a 1-page SFV assessment for this pipeline using the reproducibility checklist from Section 10 as your structure. Your assessment should:

1. Evaluate the pipeline against all 8 checklist items, rating each as pass, partial, or fail with brief justification.
2. Identify the two highest-priority SFV risks for this specific context (multi-analyst, multi-session, published microdata).
3. Recommend the minimum countermeasures required before this pipeline would be suitable for production use.
4. State the conditions under which you would approve this pipeline for dissemination (not just for internal analysis).

**Note on scope:** Your assessment is for the AI-assisted pipeline, not for the imputation method itself. Assume the random forest imputation method is validated and appropriate. The question is whether the pipeline that uses it maintains state fidelity across a three-week, multi-analyst workflow.

---

## 13. Key takeaways for federal statisticians

1. AI-assisted research introduces validity threats that classical methodology does not address. Same data plus same prompt does not guarantee same operative state across sessions.

2. The instrument in an AI-assisted pipeline is (fixed model weights + mutable context buffer). The context window IS the instrument. State fidelity is whether that instrument maintained its integrity.

3. State Fidelity Validity (SFV) names five distinct threats: Semantic Drift (T1), False State Injection (T2), Compression Distortion (T3), State Supersession Failure (T4), and State Discontinuity (T5).

4. SFV failures are latent. The output looks fluent. Nothing visibly breaks. Detection requires explicit monitoring infrastructure: canonical logs, reconciliation checks, and session handoff documents.

5. SFV is a continuous dimension of measurement quality for stateful pipelines. Each SFV threat primarily degrades a specific classical validity type: T1 threatens construct validity, T2 and T4 threaten internal validity, T3 threatens statistical conclusion validity, and T5 threatens external validity. The question is not whether state degradation exists, but whether it remains within acceptable limits for the inferential claims being made.

6. Engineering countermeasures exist and are implementable. Config-driven vocabulary, graph-backed ontologies, TEVV validation loops, handoff documents, documentation-as-traceability, multi-model triangulation, and periodic state reconciliation each address specific threats.

7. The reproducibility checklist provides a practical evaluation tool for federal statisticians assessing any AI-assisted research pipeline before dissemination.

8. Bounded agency is not just a design preference. SFV provides the methodological justification for why human oversight of AI-assisted pipelines is necessary, not optional. Humans are the primary detection mechanism for latent state failures.

```{admonition} How to explain SFV to leadership
:class: dropdown

**On why classical reproducibility is not enough:**
"Classical reproducibility means same data, same code, same results. That still applies. But in an AI-assisted pipeline, the 'instrument' is not just the code: it is the code plus the accumulated working context. If that context degrades silently across sessions, the results can look perfectly reproducible while the methodology has actually changed. State Fidelity Validity is the framework for checking whether the methodology maintained its integrity."

**On the threat taxonomy:**
"There are five ways an AI pipeline's accumulated state can fail. Terminology can drift without anyone redefining it. The system can 'remember' decisions that were never made. Automated summarization can strip the rationale from a data exclusion. A revised parameter can revert to its old value. A new session can lose everything the prior session established. Each of these has a name in the SFV taxonomy, and each has a countermeasure."

**On the investment in countermeasures:**
"The countermeasures are not exotic. They are structured documentation practices: write decisions to external logs before the AI can forget them, create handoff documents at session boundaries, periodically ask the system to restate its understanding and check that against your log. Seldon implements all of these as an engineering system. But even a researcher without Seldon can implement most of them with a structured notebook and discipline."

**On bounded agency and SFV:**
"We keep saying AI assists, humans decide. State Fidelity Validity explains why that is not just a policy preference. It is a methodological requirement. The AI cannot independently audit its own state. It does not know when its terminology has drifted or when it has confabulated a decision. A human who never checks the accumulated state is not exercising judgment; they are just countersigning the AI's output. The checklist gives us a structured way to verify that human oversight is actually happening."
```

---

*This chapter concludes the ai4stats course. The arc from Chapter 1 to Chapter 15 is the arc from "how do I use these tools" to "how do I trust the process that uses these tools." Both questions matter. Federal statistical work requires both.*
