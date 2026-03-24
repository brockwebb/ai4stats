# Glossary

```{glossary}
attention mechanism
: A neural network component that allows each element in a sequence to selectively weight information from all other elements. The core innovation of transformer architectures. Attention weights show which tokens the model considers most relevant for a given prediction.

autocoding
: Automated assignment of standardized codes (NAICS, SOC, ICD-10) to free-text responses using rule-based systems or machine learning. Autocoding rate is the fraction of records the system codes without human review.

agentic
: Behavior in which an AI system exercises granted decision-making authority within a workflow. Agentic is a property of behavior, not of the system itself.

agency
: Granted decision-making authority. Agency is conferred by a system designer, not inherent to a model. Granting less agency is often the better design choice.

agent
: An entity that does work within a workflow. In agentic AI, the agent is typically an LLM operating with granted authority over a defined set of actions.

agent chain
: A pipeline pattern where multiple agents execute sequentially, with the output of one feeding the input of the next. Each agent has its own scope and constraints.

autoregressive
: A generation strategy where each token is predicted one at a time, conditioned on all previously generated tokens. The standard approach for transformer-based language models (GPT, Claude, Gemini). Produces high-quality text but is inherently sequential.

autonomy dial
: A conceptual spectrum from fully human-controlled (left) to fully autonomous AI action (right). Federal statistical operations should generally stay on the left side, where AI proposes and humans approve.

bounded agency
: The design principle that AI systems with explicit constraints, human oversight, and defined scope outperform fully autonomous systems in high-stakes contexts. The operating model for responsible agentic AI in federal statistics.

calibration
: A fairness property: for every predicted probability p, the fraction of true positives is p; and this holds equally across groups. When base rates differ across groups, calibration and equalized odds cannot both be satisfied simultaneously.

Chief AI Officer (CAIO)
: The senior official designated by each federal agency under OMB guidance to oversee AI governance, risk management, and use case inventory. Originally required by M-24-10 (2024); the requirement persisted under M-25-21 (April 2025), which replaced M-24-10.

cognitive offloading
: The delegation of analytical tasks to AI systems, reducing direct human engagement with the underlying reasoning. A risk to intellectual rigor in AI-assisted research pipelines.

compaction
: Automated summarization or truncation of an LLM's context window to fit token limits. A source of Compression Distortion (T3) in the SFV threat taxonomy.

confidence-based routing
: Directing high-confidence LLM outputs to automatic acceptance and low-confidence outputs to human review, based on a configurable threshold. The routing decision is made per record; the threshold is a policy choice that determines the accuracy-automation tradeoff. Requires that the LLM produce calibrated confidence scores alongside its classifications.

Compression Distortion (T3)
: An SFV threat: compaction or summarization causes a pipeline to lose nuance, qualifications, or specificity from earlier state. The pipeline continues operating on a degraded representation of prior decisions.

Compression Fidelity (CF)
: An SFV sub-dimension: the degree to which context window management (summarization, compaction, chunking) preserves the accuracy of prior state rather than introducing distortion.

confidential model
: A trained model whose weights encode information from confidential microdata and which is therefore subject to disclosure review before release or external access. A proposed SDL category analogous to confidential datasets.

context window
: The token buffer containing an LLM's current operative state. In SFV terms, the context window IS the instrument. Its integrity is a validity condition.

cross-language accuracy gap
: The difference in classifier performance between the best-performing and worst-performing languages in a multilingual classification task. For LLM-based classifiers, gaps of 10 to 40 percentage points have been documented. Do not assume English-language accuracy generalizes to other languages in federal survey data.

demographic parity
: A fairness property: the positive prediction rate is equal across groups, regardless of base rates. Simple to compute but ignores the fact that base rates may legitimately differ.

differential privacy
: A formal mathematical framework that quantifies the privacy loss from including any single individual's data in a statistical release. Parameterized by epsilon: smaller epsilon means more privacy but less accuracy.

diffusion language model
: A language model that generates text by iteratively refining an entire sequence in parallel (from noise toward coherent text), rather than predicting one token at a time left-to-right. An emerging alternative to autoregressive transformers, exemplified by the Mercury model family.

disclosure avoidance
: Methods used by statistical agencies to protect confidentiality of respondent data before public release. Also called Statistical Disclosure Limitation (SDL). See also: *statistical disclosure limitation* in this glossary.

enforcement gap
: In SDL governance, the gap between written policies and actual operational enforcement. Example: FSRDC output review procedures exist on paper but do not explicitly cover trained models or API endpoints.

epsilon
: The privacy loss parameter in differential privacy. Smaller epsilon means more privacy but less accuracy. The choice of epsilon is a governance decision with consequences for data utility.

equalized odds
: A fairness property: true positive and false positive rates are equal across groups. Focuses on error rates rather than prediction rates. Cannot be simultaneously satisfied with calibration when base rates differ.

few-shot prompting
: Including example input-output pairs in a prompt to guide the model's classification behavior. Contrast with zero-shot prompting (no examples). In industry coding, few-shot examples illustrate the precise distinction the model needs to make -- for example, showing that NAICS codes the employer's industry, not the respondent's occupation. Typically two to five examples are sufficient; more may degrade performance by consuming context window capacity.

False State Injection (T2)
: An SFV threat: the pipeline incorporates incorrect information from a hallucination, tool error, or flawed intermediate result, treating it as valid prior state. The pipeline then builds further analysis on a false foundation.

FCSM
: Federal Committee on Statistical Methodology. Sets quality standards for federal statistical products, including relevance, accuracy, timeliness, accessibility, interpretability, and coherence.

FCSM 25-03
: *AI-Ready Federal Statistical Data: An Extension of Communicating Data Quality* (2025). FCSM working paper on preparing federal statistical data for reliable AI consumption. Extends the FCSM quality framework to address how AI systems ingest and use federal data. Complements this book's focus on how federal statisticians evaluate AI systems as outputs.

fine-tuning
: Adapting a pre-trained model to a specific task by training it further on task-specific labeled data. For survey text coding, this means taking a general language model (e.g., BERT) and training it on agency-labeled industry or occupation descriptions.

handoff document
: Explicit serialization of accumulated pipeline state for session continuity. An engineering countermeasure to State Discontinuity (T5) in the SFV threat taxonomy.

impact gap
: In SDL governance, the gap between disclosure assessments and actual changes to engineering practice. Transparency documentation that does not constrain system design produces paperwork, not privacy protection.

imputation
: The process of filling in missing values in survey data using statistical methods. Hot-deck imputation, regression imputation, and multiple imputation are the most common approaches in federal surveys.

membership inference attack
: An attack that tests whether a specific individual's data were used to train a model. Success reveals participation in a dataset, which can disclose sensitive information (e.g., presence in a disease cohort or hospital system).

NIOCCS
: NIOSH Industry and Occupation Computerized Coding System. A free web-based application that uses machine learning to assign NAICS and SOC codes to free-text industry and occupation descriptions. In production since 2014; adopted ML in 2021. Has coded 150+ million records.

model card
: A standardized documentation format for trained models that reports intended use, performance metrics, training data characteristics, ethical considerations, and limitations. Proposed by Mitchell et al. (2019). Agencies should require model cards as part of AI procurement and deployment review.

model inversion attack
: An attack in which an adversary repeatedly queries a model to reconstruct sensitive features of training data or approximate individual training records.

model poisoning
: Deliberate manipulation of a model's training data to introduce backdoors, biases, or targeted misclassification behaviors that are invisible during standard evaluation but activate on specific inputs. A supply chain security threat for any model whose training data and process are not fully auditable.

model provenance
: The documented origin and transformation history of a trained model, including training data sources, training process, organizational governance, and any fine-tuning or post-processing applied. Analogous to data provenance in federal statistics. Essential for supply chain risk evaluation.

model version pinning
: Locking an API call or inference configuration to a specific dated model version identifier to prevent silent behavior changes from vendor updates. For example, using "gpt-4o-2024-11-20" instead of the floating alias "gpt-4o". Required for reproducibility in production LLM systems. Treat the model identifier as a software dependency version: pin it, test it, and document when you upgrade.

NIST AI 600-1
: The Generative Artificial Intelligence Profile, a companion resource to the AI RMF published by NIST in July 2024. Identifies 12 risk categories specific to generative AI systems -- including confabulation, information integrity risks, harmful bias, and data privacy concerns -- and maps more than 200 suggested actions to the GOVERN-MAP-MEASURE-MANAGE structure. For federal statisticians evaluating generative AI tools, this profile provides the most specific NIST guidance currently available.

NIST AI RMF
: The Artificial Intelligence Risk Management Framework (NIST AI 100-1), published January 2023. A voluntary, sector-agnostic framework for managing AI risks through four functions: GOVERN, MAP, MEASURE, and MANAGE. The primary federal AI governance framework; not tied to any executive order and never rescinded. Higher-stakes AI uses require more rigorous application of each function.

observe-decide-act-check loop
: The universal pattern underlying all agent behavior: (1) observe current state, (2) decide on an action, (3) act, (4) check whether the result is acceptable. The loop repeats until a terminal condition is reached.

predictive parity
: A fairness property: the positive predictive value (precision) is equal across groups. Among those predicted positive, the fraction truly positive is the same regardless of group membership.

prompt regression testing
: Re-running an evaluation dataset after prompt changes to verify that accuracy has not degraded. Analogous to software regression testing: a prompt change that improves performance on one sector may degrade performance on another. The evaluation dataset built at initial deployment is the test suite for all subsequent prompt development.

prompt template
: The reusable structure of an LLM prompt, with placeholders for variable input. The template is versioned and logged as a methodology artifact; the instance (template plus filled input) is logged per inference call. Treat prompt templates as code: version them, test changes against the evaluation dataset, and document what changed and why.

prompt-as-agent
: The design pattern where a well-specified system prompt serves as the complete agent specification, defining role, constraints, tools, failure paths, and success criteria without requiring external framework code.

restricted API
: A model endpoint trained on confidential data and subject to SDL access controls (rate limiting, logging, query auditing, DP noise). A proposed SDL release category.

Semantic Drift (T1)
: An SFV threat: key terms shift meaning across sessions or pipeline stages without explicit acknowledgment. The pipeline continues as if the terms are stable, introducing inconsistency that is difficult to detect.

Session Continuity (SC)
: An SFV sub-dimension: the degree to which state accumulated in prior sessions is accurately and completely preserved and accessible in subsequent sessions.

small language model (SLM)
: A language model with roughly 0.5-7 billion parameters, designed for domain-specific tasks where a fine-tuned small model outperforms a general-purpose large model. SLMs can be quantized and deployed on agency hardware without API dependencies.

specification gap
: In SDL governance, the gap between the scope of confidentiality goals and the precision of SDL definitions. Example: SDL policies define "release" in terms of microdata and tables but do not specify whether model weights or API endpoints are releases.

State Coherence (SCoh)
: An SFV sub-dimension: the degree to which all current active state elements are mutually consistent and free of internal contradiction. Contradictions in active state are T4 (State Supersession Failure) violations.

State Discontinuity (T5)
: An SFV threat: a session boundary causes partial or complete loss of accumulated pipeline state. The next session must reconstruct state from an incomplete record, introducing error or loss of prior decisions.

State Fidelity Validity (SFV)
: The degree to which an AI-assisted research or analytic pipeline preserves the accuracy and integrity of its accumulated internal state across sequential operations. A validity framework developed for multi-session AI pipelines in high-stakes domains.

State Provenance (SP)
: An SFV sub-dimension: the degree to which the origin and transformation history of key state elements can be traced. Without provenance, it is impossible to audit why a pipeline reached a given conclusion.

State Supersession Failure (T4)
: An SFV threat: a superseded decision, retracted finding, or corrected value persists in the pipeline alongside its replacement. The pipeline operates on contradictory state without detecting the conflict.

statistical disclosure limitation
: The set of methods used to reduce the risk that individuals, firms, or organizations can be re-identified from released statistics, microdata, or (in the AI era) trained models and API endpoints. Also called statistical disclosure control (SDC) or disclosure avoidance.

TEVV
: Test, Evaluation, Verification, and Validation. The NIST AI RMF term for systematic AI system assessment. All four components are required for federal AI use; each addresses a distinct question about system trustworthiness.

Terminological Consistency (TC)
: An SFV sub-dimension: the degree to which key technical terms retain consistent meaning across all sessions and pipeline stages. Terminological drift is T1 (Semantic Drift) in the SFV threat taxonomy.

tokenization
: The process of converting text into a sequence of discrete units (tokens) that a model can process. Character-level, word-level, and subword (BPE, WordPiece) are the main approaches. Subword tokenization is standard in modern language models.

quantization
: A compression technique that reduces model precision (e.g., from 32-bit to 4-bit weights) to decrease memory footprint and inference cost while preserving most of the model's accuracy. AWQ (Activation-aware Weight Quantization) is currently recommended over pruning for small language models.

tool
: In agentic AI, a single discrete operation that an agent can invoke: a search, a calculation, a database query, a code execution. Tools are the atomic actions available to the agent within its granted scope.

transformer
: A neural network architecture based on self-attention mechanisms, introduced in 2017. The foundation of BERT, GPT, and all modern large language models. Uses no recurrence or convolution; processes entire sequences in parallel.

workflow
: A structure: a defined sequence or graph of steps to accomplish a goal. Workflows can be executed by humans, AI, or combinations. Not all workflows involve agents; not all agents are part of complex workflows.
```

*Note: This is a starter glossary. Each chapter should add terms as they are introduced. Build the full glossary incrementally as chapters are written or revised.*
