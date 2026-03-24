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

# Chapter 14 - Evaluating AI Systems for Federal Use

```{contents}
:depth: 1
```

```{admonition} Who is this for?
This chapter is for everyone who has completed this course. Every preceding chapter taught a technical method. This one asks: how do you decide whether to *use* one? Federal statisticians are not just practitioners. They are stewards of public trust. When a vendor pitches an AI product, when a colleague proposes a machine learning pipeline, when your director asks whether the agency is "using AI responsibly," you need a framework for evaluation that goes beyond checking whether the accuracy number looks acceptable. This chapter provides that framework.
```

```{admonition} Why this matters for federal statistics
:class: tip
The federal government is deploying AI systems in consequential domains: benefits eligibility, tax fraud detection, survey coding, demographic estimation, and economic forecasting. Most of these deployments are evaluated primarily on whether they work, not on whether they are trustworthy, reproducible, explainable, or appropriately overseen.

The specific executive orders and memoranda that mandate federal AI governance have changed across administrations -- EO 14110 (2023) was rescinded in January 2025 and replaced by EO 14179; OMB M-24-10 was replaced by M-25-21 in April 2025. But the underlying evaluation needs are durable. The NIST AI Risk Management Framework (January 2023) remains the federal government's primary AI governance framework -- it was never rescinded. FCSM quality standards apply regardless of which executive order is in effect. Federal statisticians who can evaluate AI systems rigorously will be indispensable to their agencies under any policy regime.
```

See `examples/chapter-14/` for all visualization and display code referenced in this chapter.

## 1. Setup

This chapter is prose-first. The supporting code in `examples/chapter-14/` is primarily visualization and structured display -- not machine learning. The main deliverable is the evaluation framework and rubric.

Run `examples/chapter-14/01_nist_ai_rmf.py` through `examples/chapter-14/11_exercise_radar.py` to generate the diagrams and structured outputs referenced in each section.

## 2. The evaluation problem

A vendor presents your agency with an AI-powered survey coding system. Their one-pager says:

> *"Our system achieves 94.2% accuracy on industry coding, reducing human coder hours by 60%. The model was trained on 2 million coded responses and validated on an independent holdout set. Deployment takes 2 weeks."*

What questions do you ask?

Most evaluators focus on accuracy, cost, and timeline. They rarely ask the questions that determine whether a system is actually deployable in a federal statistical program. What does "94.2% accuracy" mean? Accuracy on whose data? At what classification granularity -- NAICS 2-digit sectors or 6-digit detailed codes? Measured against human coders or against another automated system? What happens when the model is wrong? Who notices, and how quickly? Can the accuracy be independently verified, or must you trust the vendor? What happens when survey response patterns change because a new industry emerges or a question is reworded?

Then there are the governance questions that have nothing to do with the model's statistical performance. Does the system retain survey responses? What data use agreement applies? Is the system FedRAMP-authorized if it runs in a cloud environment? Has it been tested on minority-language responses? Can human coders override the system, and is the override logged? Who validates the model when the vendor releases an update?

Most AI evaluation failures are not technical. They are failures of governance and skepticism. The rest of this chapter provides a systematic framework for asking all of these questions.

*Forward reference: Chapter 15 introduces State Fidelity Validity, the validity framework for AI systems that accumulate decisions across multiple sessions. Section 6 of this chapter introduces SFV as Dimension 10 of the evaluation rubric.*

## 3. NIST AI Risk Management Framework

The National Institute of Standards and Technology published the AI Risk Management Framework (AI RMF 1.0, NIST AI 100-1) in January 2023. It is the federal government's primary framework for responsible AI development and deployment. The AI RMF was never rescinded -- it does not derive its authority from any executive order and remains in effect under any administration.

### 3.1 The four functions

The AI RMF organizes AI risk management into four core functions: GOVERN, MAP, MEASURE, and MANAGE. These are not sequential steps -- they are ongoing activities that operate in parallel throughout an AI system's lifecycle.

- *GOVERN* establishes the policies, roles, and accountability structures that determine how AI risk is overseen. Who is responsible for this system? What rules govern its use? How are decisions documented?
- *MAP* establishes context -- what the system is for, who is affected, and what risks are relevant. This is where you determine whether a system is appropriate for a given use case before any technical evaluation begins.
- *MEASURE* assesses AI risks through metrics, testing, and evaluation. This is the TEVV (Test, Evaluation, Verification, and Validation) function. It asks: how do we know the system works? What are the error rates, and for whom?
- *MANAGE* treats identified risks through monitoring, incident response, and ongoing oversight. What do we do when the system fails? How do we detect degradation before it affects published outputs?

The AI RMF is not a checklist. It is risk-based: higher-stakes uses require more rigorous application of each function. A system that routes internal email requires a lighter GOVERN structure than a system that codes survey responses for published microdata. See `examples/chapter-14/01_nist_ai_rmf.py` for a diagram of the four-function structure.

### 3.2 GOVERN: who is accountable?

GOVERN asks the foundational accountability questions. Is there a designated AI accountability structure in the agency? OMB guidance (M-25-21, which replaced M-24-10 in April 2025) requires a Chief AI Officer in each agency. Are roles and responsibilities documented -- who can approve AI deployments, who can suspend them? Are policies in place for human oversight, specifying which decisions require human review and how overrides are logged?

Federal-specific GOVERN questions go beyond the generic framework. Has legal counsel reviewed data use agreements? Is the system FedRAMP-authorized if cloud-based? Does it touch Title 13 or CIPSEA-protected data? Has OMB Statistical Policy Directive No. 1 been reviewed? Is the system covered by the agency's AI use case inventory, as required under federal AI governance guidance?

The CAIO requirement and the AI use case inventory requirement both survived the transition from M-24-10 to M-25-21, suggesting they are durable governance expectations that agencies should treat as permanent infrastructure rather than administration-specific mandates. The specific regulatory language changed; the accountability structure it describes did not.

See `examples/chapter-14/01_nist_ai_rmf.py` for the full GOVERN function description.

## 4. FCSM statistical quality standards

The Federal Committee on Statistical Methodology defines six quality dimensions for federal statistical data. These predate AI but apply directly to AI-generated outputs. If an AI system produces estimates, codes, or classifications that feed into a federal statistical product, those outputs must meet FCSM standards.

The six dimensions are relevance, accuracy and reliability, timeliness, accessibility, coherence, and interpretability. Applied to AI systems, each dimension generates a distinct set of evaluation questions. A system that achieves 94% accuracy overall but performs at 60% for food service descriptions fails the accuracy standard for programs that need food service industry data. A stochastic model that assigns different NAICS codes on repeated calls to the same description fails the coherence standard. A system that returns a code with no rationale fails the accessibility standard for non-technical staff who must act on the code.

The interpretability dimension carries particular weight in federal statistical programs. Releasing AI-coded data without documenting the error rate by sector, the training data characteristics, or the failure analysis is a violation of the interpretability requirement -- even if the aggregate accuracy is high.

See `examples/chapter-14/02_fcsm_dimensions.py` for the full table with AI-specific questions and federal examples for each dimension.

## 5. The NIST-FCSM mapping

The AI RMF functions and the FCSM quality dimensions are not independent frameworks. They address complementary questions and map to each other in consistent ways. Federal statisticians who understand this crosswalk can apply both frameworks together without doubling their evaluation effort.

GOVERN maps primarily to interpretability, coherence, and accessibility. GOVERN asks who is accountable and whether users can understand and explain system outputs to oversight bodies. These are interpretability and coherence questions. MAP maps to relevance -- the fundamental question of whether the system addresses an actual operational need. MEASURE maps to accuracy, coherence, and timeliness, covering the core quality dimensions that determine whether a system meets production requirements. MANAGE maps to timeliness and accessibility, asking whether the system can operate within production constraints over time.

Note that interpretability appears under GOVERN, not only under MEASURE. This placement reflects that interpretability is a governance requirement: users must be able to explain AI-generated outputs to stakeholders and oversight bodies, not merely verify that the system performed accurately on a test set. See `examples/chapter-14/03_nist_fcsm_crosswalk.py` for the visualization and key mapping text.

A recent FCSM working paper extends this framework in an important direction. FCSM 25-03, *AI-Ready Federal Statistical Data: An Extension of Communicating Data Quality* (2025), addresses how AI systems consume federal data -- the upstream direction. Where this chapter focuses on how federal statisticians evaluate AI systems as outputs, FCSM 25-03 addresses how federal data producers should prepare their data so that AI systems can use it reliably. The two documents are complementary: good AI system evaluation requires understanding both the quality of the system being evaluated and the quality of the data it will consume.

The pedagogical crosswalk presented in this chapter maps four RMF functions to six FCSM quality dimensions. A comprehensive systematic version mapping all 11 FCSM 20-04 dimensions to all 72 NIST AI RMF subcategories -- with 104 bidirectional edges -- is available in Webb (2026).

### 5.1 TEVV: Test, Evaluation, Verification, and Validation

TEVV is the NIST AI RMF term for the systematic assessment activities that fall under the MEASURE function. All four components are required for federal AI use; each addresses a distinct question.

*Testing* asks whether the system performs as specified on defined test cases. *Evaluation* asks whether performance meets the requirements for the specific operational context. *Verification* asks whether the system was built correctly -- does it implement the design specification? *Validation* asks whether the right system was built -- does it address the actual operational need? Vendors typically provide testing results. Evaluation, verification, and validation are the agency's responsibility and cannot be outsourced to the vendor.

### 5.2 NIST AI 600-1: Generative AI Profile

In July 2024, NIST released NIST AI 600-1, the Generative Artificial Intelligence Profile, as a companion resource to the AI RMF. AI 600-1 identifies 12 risk categories specific to generative AI systems, including confabulation (hallucination), information integrity risks, harmful bias, data privacy concerns, and others not present in traditional supervised ML systems. It provides more than 200 suggested actions mapped to the GOVERN-MAP-MEASURE-MANAGE structure, giving agencies a specific starting point for applying the RMF to LLM-based tools.

For federal statisticians evaluating generative AI tools -- LLM-powered coding systems, comment summarizers, report drafting assistants -- AI 600-1 provides the most specific NIST guidance currently available. The confabulation and information integrity categories are directly relevant to the SFV threat taxonomy introduced in Chapter 15. Note that the AI RMF is currently in active revision; updated RMF 1.1 guidance is expected through 2026. Agencies should monitor NIST AI publication updates accordingly.

## 6. The evaluation rubric: a practical tool

The following 10-dimension rubric operationalizes the NIST-FCSM framework into questions that a federal statistician can ask when evaluating any AI system proposal. Each dimension has a minimum requirement, a best practice, and specific red flags that indicate the system is not ready for federal statistical deployment.

Scores run from 0 (dimension missing entirely from the system or its documentation) to 3 (best practice met). The maximum score is 30. Systems scoring below 10 are not deployable; systems scoring 10-18 may be piloted only with documented mitigations for each gap. See `examples/chapter-14/04_evaluation_rubric.py` for the full rubric definition.

The 10 dimensions are:

1. *Task fit.* Does the AI system address a real, documented operational need? Red flag: the vendor defines the problem.
2. *Accuracy.* Measured against what baseline, on whose data, at what classification level? Red flag: a single aggregate number without subgroup breakdown.
3. *Reproducibility.* Does the same input produce the same output across calls, versions, and time? Red flag: stochastic outputs without majority voting; model updates without revalidation.
4. *Documentation.* Can an external reviewer understand what the system does and how it was built? Red flag: "proprietary" used to deflect documentation requests; no model card.
5. *Failure modes.* What happens when the system is wrong? Who notices? How quickly? Red flag: no error analysis; silent failures.
6. *Human oversight.* Where are the human decision points? Can they be bypassed? Red flag: fully automated pipeline with no human review step; override not documented.
7. *Data governance.* What data does the system ingest, retain, and share? Red flag: survey responses retained by vendor; no FedRAMP authorization for cloud deployment.
8. *Bias and fairness.* Has performance been tested across relevant subpopulations? Red flag: only aggregate accuracy reported; no testing on minority-language responses.
9. *Update and drift management.* How does the system change over time? Who validates updates? Red flag: model updates without notice; no revalidation protocol.
10. *State Fidelity Validity (SFV).* For stateful or agentic systems: does accumulated pipeline state faithfully represent its actual operational history across sessions? Red flag: no session management for multi-session pipelines; terminology inconsistency not detected. See Chapter 15 for the full State Fidelity Validity framework.

The rubric's value is not the final score -- it is the questions the rubric forces you to ask before any deployment decision. A vendor who cannot answer dimension 4 (documentation) questions has not done a failure analysis. That is disqualifying regardless of the accuracy number.

See `examples/chapter-14/05_radar_chart.py` for the `radar_chart()` function and a hypothetical vendor scoring visualization showing how rubric gaps appear visually.

## 7. Common failure modes in AI evaluation

Beyond gaps in rubric dimensions, evaluators encounter specific patterns of misleading or insufficient AI system presentations. Six failure modes appear consistently in federal AI procurement contexts.

*Demo-ware* is a system that performs impressively on curated demonstrations but has no production capability at the required scale. The tell is that the vendor shows a live demo but cannot provide a pilot deployment on your actual data. The countermove is to request a 30-day pilot on a random sample of production data with independently measured accuracy.

*Benchmark gaming* is high accuracy on a published benchmark dataset that does not reflect your use case, data characteristics, or error modes. The tell is that the vendor cites a published paper or competition result as primary evidence. The countermove is to run independent validation on your data and compare to your human coder agreement rates as baseline.

*Cherry-picked examples* means the vendor presents cases where the system performs well and omits systematic failure categories. The tell is that all examples come from the easy, common cases with no mention of error rates on ambiguous or rare cases. The countermove: ask the vendor to show you ten cases where the system fails and describe the common patterns. A vendor who cannot answer that question has not done a failure analysis.

*Opacity as a feature* is the use of "proprietary algorithm" to deflect documentation and audit requests. The tell is no model card, no data sheet, no architecture description, and "we cannot share that" applied to fundamental design questions. The countermove is to require documentation as a procurement condition. NIST AI RMF and federal procurement standards support this requirement.

*Automation bias in evaluation* is when evaluators defer to the AI system's confidence scores without independent verification. The tell is an evaluation report that says "the model was 94% confident" as evidence of correctness without checking against ground truth. The countermove is calibration analysis: does confidence=0.9 actually mean 90% accuracy? Overconfident errors are more dangerous than underconfident ones.

*The Dunning-Kruger evaluation gap* is when evaluators know enough about AI to be impressed by capability demonstrations but not enough to identify missing safeguards. The tell is an evaluation that focuses entirely on "does it work?" and accepts vendor claims on reproducibility, bias, and documentation without verification. The countermove is this rubric: ask for documentation before the demo, and bring in an independent technical reviewer who has not seen the demo.

See `examples/chapter-14/06_failure_modes.py` for the full structured display.

### 7.1 Procurement context

OMB M-25-22, *Driving Efficient Acquisition of Artificial Intelligence in Government* (2025), provides federal AI procurement guidance that directly affects how agencies acquire AI systems. FAR (Federal Acquisition Regulation) reform is underway under EO 14275 (April 2025), affecting how AI systems are contractually specified and acquired. Federal statisticians who evaluate AI systems should understand that evaluation requirements can be written into contracts -- this is not just a post-award activity.

The 10-dimension evaluation rubric maps directly to procurement. Dimension 4 (documentation -- model card, data sheet, failure analysis) is addressable through contract deliverables. Dimension 7 (data governance -- FedRAMP authorization, data use agreements, retention policies) is addressable through acquisition clauses and data handling terms. Dimension 8 (bias and fairness -- subgroup testing, disparate impact analysis) can be required as a condition of contract award. Federal statisticians who participate in procurement decisions should bring the rubric to the acquisition table, not only to the technical evaluation.

## 8. Bounded agency: the design principle

*Bounded agency* means that AI systems in high-stakes contexts should operate with constrained autonomy and persistent human oversight. The AI provides analysis, recommendations, and automation; humans make consequential decisions and remain accountable for them.

Three levels of human involvement define the spectrum from full human control to full autonomy. *Human-in-the-loop* means a human reviews every AI output before any action is taken. This is appropriate for high-stakes, low-volume, or novel situations -- for example, when a new program is launching or when unusual industry categories appear in survey data. *Human-on-the-loop* means the AI operates automatically but a human monitors and can intervene, typically through confidence-based routing that flags low-confidence cases for human review. This is appropriate after a validated pilot demonstrates acceptable accuracy on production data. *Human-out-of-the-loop* means the AI operates fully autonomously with no human review step. For federal statistical production, this is rarely appropriate. It may apply to internal formatting or metadata tasks, but not to coded outputs that feed published microdata.

The federal context adds institutional constraints that go beyond the general design principle. OMB, Congress, and the public expect human accountability for agency decisions. "The AI decided" is not an acceptable explanation to a Congressional inquiry or an OMB audit. Human override must be easy, visible, and logged. Override rates are a health metric: if human coders never override the AI, either the system is performing perfectly (unlikely) or coders have stopped reading its outputs (dangerous). Both possibilities require investigation.

A bounded agency system surfaces the AI's reasoning, not just its conclusion. The response "This description was coded as NAICS 54 (Professional Services) because the phrase 'management consulting' matches sector 54 patterns with 0.87 confidence" is more useful than "54" alone. The AI's transparency is what makes human oversight meaningful. An opaque system forces the human to either trust blindly or re-do the work.

*Cross-reference: Chapter 13 covers the full bounded agency design framework, including the autonomy dial and specific design patterns for federal statistical contexts.*

See `examples/chapter-14/07_bounded_agency.py` for the three-level display and design implication text.

## 9. Case studies

### Case B.1: Evaluating a survey coding automation tool

The following mock vendor one-pager is reproduced from `examples/chapter-14/08_case_study_vendor.py` as a pedagogical device. It is deliberately designed to illustrate common presentation patterns in AI vendor marketing -- strong on performance claims, weak on governance documentation.

```
MOCK VENDOR ONE-PAGER: AutoCode Pro 2.0
========================================
AutoCode Pro 2.0 automatically classifies survey industry and occupation
responses to NAICS and SOC standards. Trusted by leading market research firms.

PERFORMANCE:
  - 94.2% accuracy on industry coding (NAICS 2-digit)
  - 87.6% accuracy on occupation coding (SOC major group)
  - Validated on 500,000 coded responses

FEATURES:
  - Cloud-based API (AWS deployment)
  - Batch processing: up to 1 million records per day
  - Returns code + confidence score
  - Dashboard for monitoring

IMPLEMENTATION:
  - 2-week deployment
  - REST API integration
  - 30-day money-back guarantee

PRICING:
  - $0.002 per record
  - Enterprise agreement available

Contact: sales@autocodeproexample.com
```

Applying the 10-dimension rubric to this one-pager yields a score of approximately 4/30 (13%). The scoring walkthrough:

Dimension 1 (Task fit) scores 2: AutoCode addresses a real operational need, but the problem is defined by the vendor, not the agency, and no needs assessment is documented.

Dimension 2 (Accuracy) scores 1: Two accuracy numbers are provided, but neither includes subgroup breakdown, 6-digit NAICS performance, or comparison to human coder agreement rates. The validation data is vendor-controlled.

Dimensions 3 (Reproducibility), 4 (Documentation), 6 (Human oversight), 7 (Data governance), 8 (Bias and fairness), 9 (Update and drift management), and 10 (SFV) all score 0. None are mentioned in the one-pager. The AWS cloud deployment without FedRAMP authorization is a disqualifying data governance gap for any system touching Title 13 data.

Dimension 5 (Failure modes) scores 1: a confidence score is returned, which enables routing, but no routing or alert protocol is described.

Recommendation: Do not deploy. The single high score reflects that coding is a real operational need -- not that this product meets federal requirements.

Run `examples/chapter-14/08_case_study_vendor.py` for the full rubric walkthrough with commentary on each dimension.

### Case B.2: Evaluating an LLM "comment summarizer" for SFV risks

A vendor proposes an LLM system that reads open-ended survey comments and produces summary statistics (themes, sentiment, key concerns). The system runs in multi-session mode: it accumulates results across the annual survey cycle, building a running summary that updates each month.

Standard rubric concerns apply -- evaluate dimensions 1-9 as for any AI system. But this system introduces a category of risk that the standard rubric dimensions do not fully capture: the risk that accumulated pipeline state diverges from its actual operational history. This is where State Fidelity Validity applies.

Five specific SFV threats arise in a multi-session comment summarizer:

T1 (Semantic Drift): In month 1, "housing cost burden" is defined as households spending more than 30% of income on housing. By month 7, the system has drifted to using the term for respondents who mention housing costs at all, without the 30% threshold. The cumulative summary is internally inconsistent, and the drift is difficult to detect after the fact.

T2 (False State Injection): The system "remembers" a decision to exclude single-person households from the theme analysis that was never actually made. All summaries produced after month 4 silently undercount single-person household concerns.

T3 (Compression Distortion): Monthly compaction reduces "we excluded respondents who indicated English was not their primary language due to translation reliability concerns" to "non-English excluded." By year-end, the rationale is gone. The annual summary does not mention the exclusion at all.

T4 (State Supersession Failure): A month 3 codebook revision changed the definition of "food insecurity." The pipeline never received this update. Months 4-12 use the old definition, but the annual summary presents the data as though a single consistent definition was applied throughout.

T5 (State Discontinuity): A staffing change in month 6 resulted in a new analyst restarting the LLM session. The new session lacks the accumulated context from months 1-5. The month 6 report is methodologically inconsistent with all prior months, but this is not flagged.

This system requires explicit SFV controls before it can produce defensible survey summary statistics for publication. Without config-driven vocabulary, session management, and a canonical decision log, the annual summary cannot be trusted to reflect a coherent methodological process.

*Cross-reference: Chapter 15 covers the full T1-T5 threat taxonomy, engineering countermeasures, and validation criteria for SFV.*

Run `examples/chapter-14/09_case_study_sfv.py` for the full structured analysis.

### Case B.3: Evaluating ChatGPT for agency use

Your agency director asks you to "evaluate ChatGPT for use in our statistical work." Apply the NIST-FCSM rubric.

The immediate problem is that "ChatGPT for our statistical work" is not a use case. The rubric requires a specific task before it can be applied meaningfully. This framing is itself an evaluation failure: it places the burden on the evaluator to define what "statistical work" means, when that definition should come from the program office before evaluation begins.

Applying the rubric to a hypothetical general-purpose deployment:

Task fit is unclear. Accuracy is context-dependent and unknown for specific statistical tasks. Reproducibility is poor: the commercial API is non-deterministic by default, and model updates change behavior without advance notice. Documentation is partial -- OpenAI publishes system cards, but these are generic, not task-specific. Failure modes are undocumented for agency use; hallucination risks are known but task-specific error rates are not. Human oversight depends entirely on how the agency uses the tool. Update and drift management is poor: model updates are frequent, not announced in advance, and the standard API does not support version pinning.

The most significant concern is data governance. The commercial ChatGPT API may use submitted data for training unless the Enterprise tier is engaged. It is not FedRAMP-authorized. Title 13 data cannot be sent to the commercial ChatGPT API under any circumstances. Azure Government OpenAI and AWS GovCloud AI services provide FedRAMP-authorized paths for agencies needing cloud AI services -- these are the compliant alternatives for any use involving protected data. The commercial ChatGPT API remains non-compliant for protected data regardless of tier.

For tasks not involving protected data, evaluate ChatGPT on your specific task rather than on OpenAI benchmarks, establish a reproducibility protocol (temperature, version pinning), and develop agency guidelines specifying which task types are approved, which require legal review, and which are prohibited.

Run `examples/chapter-14/10_case_study_chatgpt.py` for the full per-dimension assessment.

## 10. Exercises

```{admonition} Exercise B.1: Score the vendor one-pager
:class: tip

Using the 10-dimension evaluation rubric, score AutoCode Pro 2.0 from Case B.1. For each dimension, record your score (0-3), cite the specific evidence (or absence of evidence) from the one-pager, and note what additional information you would need to raise the score.

Create a radar chart of your scores. Which dimensions are the weakest? Use the starter code in `examples/chapter-14/11_exercise_radar.py` to produce your chart.
```

```{admonition} Exercise B.2: Three vendor questions
:class: tip

Based on your rubric scoring from Exercise B.1, write three specific questions you would ask the AutoCode Pro vendor in a follow-up meeting. For each question:

1. State the question precisely.
2. Explain which rubric dimension it addresses.
3. Describe what an *acceptable* answer looks like.
4. Describe what answer would immediately disqualify the vendor.

Example: "Question: 'What is your accuracy on 6-digit NAICS codes for food service establishments?' Dimension: Accuracy. Acceptable answer: >80% with subgroup breakdown. Disqualifying answer: 'We only report at 2-digit.'"
```

```{admonition} Exercise B.3: FCSM mapping
:class: tip

For the AutoCode Pro system, map each of the six FCSM quality dimensions to the evidence in the one-pager:

1. Which FCSM dimensions are addressed in the vendor one-pager?
2. Which are entirely missing?
3. For the dimensions that are addressed, is the evidence sufficient for a federal statistical program?

Write a one-paragraph recommendation to your program director: should AutoCode Pro 2.0 be piloted, rejected outright, or conditionally evaluated?
```

## 11. Key takeaways

- *Most AI evaluation failures are not technical; they are failures of governance and skepticism.* A system that looks impressive in a demo may be missing documentation, failure analysis, bias testing, data governance controls, and human oversight design. The rubric catches these gaps before deployment.
- *NIST AI RMF and FCSM quality standards are complementary, not competing.* The AI RMF provides a risk-based governance structure. FCSM provides the domain-specific quality requirements. Federal statisticians need both.
- *Accuracy is necessary but not sufficient.* A 94% accurate system that is not reproducible, not documented, not tested for bias, not designed for human override, and not compliant with federal data governance requirements is not deployable in a federal statistical program regardless of its accuracy.
- *Federal AI governance requirements are durable even as specific policies change.* The NIST AI RMF has never been rescinded. The CAIO requirement and AI use case inventory requirement survived the transition from M-24-10 to M-25-21. FCSM standards apply regardless of which executive order is in effect. Evaluators who internalize the underlying requirements -- not just the specific citations -- will remain effective across policy regimes.
- *Bounded agency is a design principle, not a limitation.* AI systems that assist human decisions produce better outcomes in high-stakes federal contexts than systems that replace human judgment. Design for assistance: surface reasoning, make override easy, log overrides, monitor override rates. Cross-reference Chapter 13 for bounded agency design patterns.
- *State Fidelity Validity applies to stateful and agentic AI systems.* For any AI pipeline that accumulates decisions across sessions -- a multi-session LLM analysis, a continuous coding system, an agentic data pipeline -- SFV is the validity framework for asking whether the pipeline's accumulated state faithfully represents its actual operational history. Cross-reference Chapter 15 for the full framework.
- *The Dunning-Kruger gap is real in AI evaluation.* People who have used AI tools confidently often evaluate AI systems on surface features rather than on governance, documentation, and failure mode analysis. This rubric is designed to force the questions that confident non-experts do not ask.
- *Transparency is a procurement requirement, not a favor from the vendor.* NIST AI RMF, federal AI governance requirements, and FCSM standards all support requiring documentation before approval. Proprietary algorithms do not exempt vendors from federal AI governance requirements.

```{admonition} How to explain these methods to leadership
:class: dropdown

**On why evaluation goes beyond accuracy:**
"An accuracy number from a vendor is not an evaluation. It is a marketing claim. We need to know: accuracy on whose data, at what classification granularity, for which subpopulations, measured how? We also need to know whether the system is reproducible, whether it is compliant with our data governance requirements, and whether human coders can override it when it is wrong. The NIST AI Risk Management Framework gives us the structure to ask all of these questions systematically before any deployment decision. That framework was not rescinded with the last administration's executive orders -- it remains in effect and is the federal government's primary AI governance standard."

**On the FCSM connection:**
"The quality standards FCSM defines for federal statistical data -- relevance, accuracy, timeliness, accessibility, coherence, and interpretability -- apply to AI-generated outputs just as they apply to survey estimates. If an AI system produces industry codes that go into our published microdata, those codes must meet the same standards as human-coded data. The NIST-FCSM crosswalk I am proposing gives us a way to apply both frameworks together."

**On bounded agency:**
"We are not arguing against AI tools. We are arguing that in federal statistical production, AI should assist human judgment, not replace it. The system should surface its reasoning, make human review easy, log when humans override its output, and alert us when its accuracy is degrading. If human coders are never overriding the AI, that is a warning sign: either the AI is actually perfect (unlikely) or coders have stopped reading its output (dangerous)."

**On the vendor evaluation process:**
"Before any procurement decision, I recommend we require a 30-day pilot on a random sample of our actual production data, with independent accuracy measurement by our own coders. We require a model card, a data sheet, a failure analysis, and documentation of the data governance controls. These are standard requirements under the NIST AI RMF and federal AI governance guidance. A vendor who cannot provide them should not be considered for a federal statistical deployment."

**On policy changes:**
"The executive orders and OMB memoranda governing federal AI have changed, and will likely change again. The evaluation framework in this chapter is not contingent on any specific policy document. The NIST AI RMF, FCSM quality standards, and FedRAMP requirements are the durable foundations. If leadership asks whether our evaluation process is still current after a policy change, the answer is yes -- because we built it on the standards that have persisted across administrations, not on citations to specific executive orders."
```
