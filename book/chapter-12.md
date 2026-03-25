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

# Chapter 12 - Large Language Models for Survey Operations

```{contents}
:depth: 1
```

```{admonition} Who is this for?
If you have completed Chapter 11 (Transformers), you understand what an LLM is at the architectural level. This chapter is about what you can *do* with one in federal survey work. No API keys, no local models, no GPU required. The entire chapter uses simulated output that illustrates real evaluation methodology. All example code lives in `examples/chapter-12/`.
```

```{admonition} Why this matters for federal statistics
:class: tip
Statistical agencies classify millions of open-ended text responses every year. Industry descriptions get coded to NAICS. Occupation descriptions get coded to SOC. Cause-of-death text gets coded to ICD-10. Geographic descriptions get geocoded. Currently this work is done by teams of trained human coders and rule-based autocoding systems developed over decades at significant cost.

Large language models can perform these tasks with accuracy that sometimes rivals trained human coders, at a fraction of the marginal cost. But LLMs introduce new problems: inconsistency across repeated calls, stochastic outputs, privacy risks when responses are sent to commercial APIs, and no formal uncertainty estimates. Every statistical program evaluating or using LLMs for operational text coding needs a systematic evaluation methodology. This chapter provides one.
```

## 1. Setup

```{code-block} python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, accuracy_score
```

No LLM API calls are made in this chapter. All code examples use simulated data that reproduces the statistical patterns observed in published studies. See `examples/chapter-12/` for fully runnable scripts.

## 2. The coding problem in federal statistics

Every major federal survey collects open-ended text responses that must be assigned standardized codes before analysis. The scale is large. The Current Population Survey and the American Community Survey together process roughly six million industry and occupation descriptions per year. The National Death Index processes nearly three million cause-of-death text strings. The Occupational Employment and Wage Statistics program processes over one million occupation descriptions.

These numbers matter because they determine what is economically feasible. Human coders working at 400 to 800 descriptions per day represent a substantial operational cost. Rule-based autocoding systems -- the current standard -- achieve 60 to 80 percent automation rates, leaving the remainder to human review. Improving that automation rate, or reducing the human review burden on the remaining fraction, has direct cost implications for every agency operating at this scale.

See `examples/chapter-12/01_coding_problem.py` for a full table of federal coding programs and concrete examples of the ambiguity that makes the problem difficult.

### 2.1 A concrete example: industry coding

Industry coding is harder than it looks. The assignment rule is that NAICS codes the *industry of the employer*, not the occupation of the respondent. This matters because many descriptions describe what the respondent does, not what the employer makes or sells. "I do IT for a bank" gets coded to Finance and Insurance (NAICS 52), not Information (NAICS 51), because the employer is a bank. A respondent who says "I'm a nurse at a clinic" presents ambiguity: a clinic could be an outpatient office (NAICS 621) or a hospital outpatient department (NAICS 622), and the word "clinic" alone does not resolve it.

Adjacent sector confusion is the dominant error pattern in both human and LLM coding. Professional Services (54) and Information (51) share many workers whose descriptions mention technology. Health Care (62) and Other Services (81) share service workers whose descriptions mention patients or clients. Understanding which confusions are genuinely ambiguous versus which are clear errors is essential for evaluating a coding system.

### 2.2 Federal automated coding: a brief history

Automated coding in federal statistical programs predates machine learning by decades. The Census Bureau developed deterministic coding systems for industry and occupation in the 1980s using keyword dictionaries and hierarchical matching rules. The National Center for Health Statistics built ACME (Automated Classification of Medical Entities) for cause-of-death coding starting in the 1960s. These rule-based systems reduced human review burden substantially but required constant manual maintenance as language and industries evolved.

The current generation of systems adds machine learning on top of rules. NIOSH introduced the NIOSH Industry and Occupation Computerized Coding System (NIOCCS) in 2014, adopting ML-based coding in 2021; it has since processed more than 100 million records (CDC/NIOSH, 2022). The Census Bureau's automated coding uses a combination of exact matching, probabilistic matching, and classifier models trained on decades of human-coded data. The Bureau of Labor Statistics OEWS program uses a similar hybrid approach for occupation coding.

LLMs are the latest approach, not the first. The advantage they offer is generalization: an LLM can handle novel descriptions and informal language without requiring explicit rule maintenance. The risk is that they introduce different failure modes than rule-based systems, and those failure modes are less transparent. This chapter is about measuring those failure modes systematically. For NIOCCS specifically, see Chapter 11.

## 3. How LLM-based coding works

LLM-based coding uses the model's language understanding to classify text without training a task-specific model. The key tool is the *prompt*: a structured text input that tells the model what to do, provides the classification scheme, and optionally provides examples.

### 3.1 Prompt design principles

A coding prompt has three required components: a role instruction that tells the model it is coding for a federal statistical agency, the classification scheme it should use, and the output format specification. A fourth optional component is few-shot examples -- pairs of (description, correct code) that illustrate the classification rule.

The zero-shot prompt (no examples) works reasonably well for clear cases. The few-shot prompt (two to five examples) significantly improves performance on confusable sectors, particularly when the examples are chosen to illustrate the precise distinction the model needs to make. For the 54 versus 51 confusion, examples showing that NAICS codes the employer's industry rather than the employee's task are more useful than examples that simply show more retail or health care cases.

See `examples/chapter-12/02_prompt_design.py` for the `build_coding_prompt()` function, zero-shot and few-shot prompt examples, and a logging schema.

A short illustrative example of the prompt structure:

```{code-block} python
def build_coding_prompt(description, few_shot_examples=None):
    prompt = (
        "You are an expert industry coder for a federal statistical agency.\n"
        "Assign the most appropriate NAICS 2-digit sector code.\n"
        "Respond with ONLY: XX - Sector Name\n\n"
        "NAICS sectors: ...\n"
    )
    if few_shot_examples:
        for text, code in few_shot_examples:
            prompt += f'  Description: "{text}" -> {code}\n'
    prompt += f'\nDescription: "{description}"\nCode: '
    return prompt
```

### 3.2 Prompt management for production

Prompts are code. They should be versioned, tested, and logged with the same discipline you would apply to any software component that produces a published statistic.

*Prompt versioning* means every prompt template has an identifier (v1.0, v1.1, ...) with a changelog explaining what changed and why. When you change the prompt, you document the previous accuracy and the expected change. When a production run is audited, the prompt version is part of the audit record.

*Prompt regression testing* means that every time you change the prompt, you run the full evaluation dataset against the new prompt before deploying it. This is directly analogous to software regression testing. A prompt change that improves accuracy on the target confusion pair may degrade accuracy on a different sector you were not watching. The evaluation dataset is your test suite. See `examples/chapter-12/02_prompt_design.py` for an example version registry.

*Prompt-model interaction* means that a prompt optimized for one model may perform differently on another. If your agency standardizes on a FedRAMP-authorized version of one model but evaluates on a different model's API, the evaluation results may not transfer. Always run final evaluations on the model you will deploy.

*Template versus instance*: the template is the prompt structure with placeholders (the versioned artifact). The instance is the template with a specific description filled in (the per-record artifact). Log both. The template version identifies the methodology; the instance is what the model actually received. Industry guidance recommends managing prompts as versioned configurations with change logs, running regression test suites across prompt versions, and monitoring per-version quality metrics to detect silent degradations (Anthropic, 2024; OpenAI, 2025; Google, 2024).

## 4. Building the evaluation dataset

Before you can evaluate an LLM coding system, you need a ground-truth evaluation dataset: a set of descriptions where the correct code is known, ideally because they were coded by trained human coders using your agency's standard procedures. For this chapter, we use a simulated 200-record dataset that reproduces the accuracy patterns and confusion structure reported in published studies.

The dataset covers ten NAICS 2-digit sectors with 20 descriptions each. Simulated LLM responses include realistic per-sector accuracy variation, adjacent-sector confusion patterns, and a two percent refusal rate (responses of "UNCLEAR"). The seed is fixed at 2025 for reproducibility.

See `examples/chapter-12/03_evaluation_dataset.py` for the full dataset and simulation code.

Key design notes on the simulation:
- Per-sector accuracy ranges from 74% (Other Services) to 92% (Retail Trade), consistent with published benchmarks for GPT-4 class models on NAICS 2-digit coding. *(Note: These accuracy figures are simulated based on reported ranges in published studies and vendor-reported performance. No peer-reviewed NAICS coding benchmark for large language models on a standardized public gold standard existed as of early 2026. The pedagogical purpose is evaluation methodology, not benchmark reporting.)*
- Confusion patterns follow the documented structure: Professional Services (54) most often confused with Information (51), Health Care (62) confused with Other Services (81), Public Administration (92) confused with Educational Services (61).
- Refusals are rare (2%) but present. A production system must handle them explicitly.

## 5. Evaluation: agreement metrics

Accuracy alone is an insufficient measure for multi-class coding systems with imbalanced classes. A system that always predicts the most common sector would achieve whatever that sector's base rate is, with zero coding ability. Cohen's kappa (Cohen, 1960) corrects for chance agreement and is the standard metric for inter-coder reliability comparisons in survey research.

### 5.1 Overall accuracy and Cohen's kappa

See `examples/chapter-12/04_agreement_metrics.py` for the full computation. The key results from the simulated dataset:

- Overall accuracy on coded records (excluding refusals): approximately 83%
- Cohen's kappa: approximately 0.81 (substantial to almost perfect agreement)
- Refusal rate: 2%

The kappa interpretation table that every coding evaluation report should include:

| Range | Interpretation |
|-------|----------------|
| 0.81 -- 1.00 | Almost perfect agreement |
| 0.61 -- 0.80 | Substantial agreement |
| 0.41 -- 0.60 | Moderate agreement |
| 0.21 -- 0.40 | Fair agreement |
| 0.00 -- 0.20 | Slight agreement (near chance) |

A kappa below 0.61 — "moderate" or lower on the Landis and Koch (1977) scale — would indicate the system is not suitable for operational use without a high human review rate.

### 5.2 Per-sector accuracy

Sector-level accuracy reveals where the system needs improvement. In the simulated results, Other Services (81) and Professional Services (54) are the lowest-performing sectors, consistent with published findings. Both involve heterogeneous employer types that share surface language with adjacent sectors.

See `examples/chapter-12/04_agreement_metrics.py` for the per-sector bar chart and human-human comparison. Published studies report human-human kappa values in the substantial-to-almost-perfect range (roughly 0.6-0.8+) for broad occupation and industry groupings (Landis & Koch, 1977 interpretation scale). This is the practical ceiling: no automated system should be expected to exceed it, because some cases are genuinely ambiguous even to trained coders.

### 5.3 Cost and throughput analysis

LLM coding at federal scale is primarily an economic decision. The accuracy question is whether the system meets quality thresholds; the cost question is whether it does so more efficiently than alternatives.

Human coder costs for industry and occupation coding are estimated at $0.50 to $2.00 per record (author's engineering estimate based on federal coder salary scales and typical throughput rates of 400 to 800 records per day), accounting for training, productivity, quality assurance overhead, and supervision. The lower end represents experienced coders on straightforward tasks; the upper end reflects complex cases requiring specialist knowledge.

As an early 2026 illustrative snapshot (verify current pricing before budgeting): end-to-end classification on a typical 1,000-token record costs on the order of fractions of a cent using compact or mini-tier models, and up to roughly one cent per record using flagship frontier models. Both represent a dramatic reduction from human coder costs of $0.50 to $2.00 per record. For a more detailed cost-performance framing, see the model selection section in Chapter 11.

The break-even calculation is simple: at a 30 percent human review rate (typical for a 95 percent accuracy threshold with a large frontier model), total cost is approximately API cost plus 0.30 times human review cost per record. At a human review cost of $1.00 per record, that is roughly $0.30 per record -- still substantially cheaper than full human coding at $0.50 to $2.00.

The more important planning parameter is throughput. Federal production coding runs are typically *batch* operations with latency tolerance measured in hours, not milliseconds. Use batch API endpoints rather than synchronous per-call endpoints; they are cheaper and scale better. Plan for 1 to 6 hour turnaround per batch. Capacity planning should estimate records per batch, batches per week, and human review capacity for the residual fraction.

See `examples/chapter-12/08_hybrid_workflow.py` for the cost-performance table and break-even analysis.

| Model tier | Cost / 1K records | Est. accuracy (2-digit) | Human review rate (95% target) | Effective cost / accepted record |
|------------|------------------|------------------------|-------------------------------|----------------------------------|
| Large frontier | $2.50 -- $8.00 | 82 -- 88% | ~25 -- 35% | $0.012 -- $0.040 |
| Mid-size | $0.30 -- $0.80 | 75 -- 83% | ~30 -- 45% | $0.005 -- $0.015 |
| Small open-source (on-premise) | ~$0.05 compute | 65 -- 78% | ~40 -- 55% | $0.001 -- $0.005 |
| Human coder only | N/A | 91 -- 93% | 100% | $0.50 -- $2.00 |

*Early 2026 illustrative snapshot. Verify current pricing with vendors.*

## 6. Error analysis: understanding failure modes

Accuracy tells you how often the system is right. Error analysis tells you *why* it is wrong and what to do about it. The three error types in LLM industry coding are:

*Adjacent sector errors*: the LLM assigned a neighboring sector that represents genuine coding ambiguity. A consultant who "provides IT strategy to clients" could be Professional Services (54) if the primary activity is consulting, or Information (51) if the employer is a software firm. Both human coders and LLMs make errors on these cases, and the error rate on them is an upper bound set by the ambiguity of the descriptions themselves.

*Unrelated sector errors*: the LLM assigned a sector that shares no reasonable overlap with the true sector. These are the concerning errors because they indicate a failure to understand the description, not a judgment call on an ambiguous case.

*Refusals*: the LLM returned "UNCLEAR" or a non-code response. Refusals must be routed to human review. A high refusal rate may indicate the prompt is ambiguous about what to do with difficult cases, or that the model is poorly calibrated for this task.

See `examples/chapter-12/06_error_analysis.py` for the classification logic and stacked bar chart by sector. The sectors with the highest unrelated error rates are the candidates for targeted prompt revision.

## 7. Reproducibility challenges

LLMs are *stochastic*. The same prompt sent to the same model twice may return different codes, particularly for ambiguous descriptions. This is not acceptable for published statistics, and it is not acceptable for the reproducibility standards that federal agencies are subject to.

### 7.1 Temperature and majority voting

Setting temperature to zero produces deterministic output *within a model version*. This is necessary but not sufficient. Run five identical calls with temperature=0 and you will get identical results. But that determinism disappears when the model version changes. See `examples/chapter-12/07_reproducibility.py` for a simulation of this effect.

Majority voting -- running the same prompt k times and taking the mode -- reduces variance at temperature > 0 but increases cost by a factor of k. For production coding, temperature=0 with version pinning is usually more practical.

### 7.2 Model version pinning

Model version pinning means locking your API calls to a specific dated model identifier, not a floating alias. "gpt-4o" is a floating alias that resolves to whatever version the provider has current. "gpt-4o-2024-11-20" is a specific model version. These are different artifacts with different behavior.

Vendor update schedules are not synchronized with your evaluation cycles. An evaluation run that establishes 94 percent accuracy on a specific model version is not valid for a different model version. Pin the version. Treat it as a dependency. When you upgrade, re-run the evaluation.

For on-premise open-source models, pinning means tracking the model file hash (SHA256) in your artifact management system and storing the model file rather than referencing a container tag.

### 7.3 The silent update problem

Vendors update models without always announcing the change or preserving behavioral compatibility. A system running at 94 percent accuracy today may run at 87 percent next quarter if the underlying model was silently updated. Monitoring is not optional for any production LLM coding system.

Set up a scheduled evaluation run -- weekly or monthly -- against your held-out validation set. Alert if accuracy drops more than two percentage points from the established baseline. The evaluation infrastructure you built for initial deployment is also your ongoing monitoring infrastructure.

### 7.4 Prompt-response logging

Every production inference call should produce a structured log record. See `examples/chapter-12/07_reproducibility.py` for the recommended log schema. The minimum fields are: prompt template version, full prompt instance, raw model response (verbatim), parsed code, model identifier, temperature, timestamp, batch ID, and routing decision (auto-accepted or human review). These logs are your reproducibility record. Without them, you cannot demonstrate to an auditor what the system actually did.

## 8. Privacy and security considerations

Survey responses sent to an LLM API leave the federal security boundary. Whether that is permissible depends on the legal authority under which the data were collected, the data's sensitivity classification, and the authorization status of the receiving service.

For data collected under Title 13, CIPSEA, or the Privacy Act, sending records to an unapproved commercial API is a legal violation, not just a risk management concern. The appropriate deployment paths are a FedRAMP-authorized cloud service with a data use agreement (moderate risk, requires legal review), or an on-premise open-source model running on agency hardware (lowest risk, highest setup cost). See `examples/chapter-12/` for a full privacy risk assessment of each approach.

The de-identification option -- removing PII before sending descriptions to an external API -- deserves a specific caveat: removing employer names and location information to reduce PII risk may also remove the context that makes the descriptions codeable. A description of "I manage the store" loses its NAICS identity when the employer name is removed. Test de-identified accuracy against full-text accuracy before assuming the approach is viable.

### 8.1 Multilingual considerations

The ACS, NHIS, and other major federal surveys collect responses in multiple languages. Spanish-language responses are particularly common in industry and occupation items. LLM accuracy varies significantly by language. Models trained predominantly on English text perform worse on Spanish, Mandarin, Vietnamese, and other languages that appear in federal survey data.

Non-English responses require separate evaluation. Recent text-classification research shows that LLM classifiers often perform materially worse in lower-resource languages, with cross-language accuracy gaps of 10 to 40 percentage points depending on the setting (Batatia et al., 2025). Do not assume English-language accuracy generalizes to Spanish or other languages represented in federal survey data. Depending on the language distribution of your survey population, you may need separate prompts, separate evaluation datasets, or separate deployment decisions for each language. Some languages may require a different model entirely.

## 9. Designing a hybrid human-LLM workflow

The hybrid workflow is almost always better than either human-only or LLM-only approaches. The core mechanism is *confidence-based routing*: auto-accept LLM assignments above a confidence threshold, route everything below the threshold to human review.

Most LLM APIs can return log-probabilities alongside the text response, which can be used to construct a confidence score. Higher log-probability responses are more likely to be correct. The threshold analysis in `examples/chapter-12/08_hybrid_workflow.py` shows that at approximately the 0.85 to 0.90 confidence threshold (on simulated data), accuracy on auto-accepted records exceeds 95 percent while keeping the human review queue at 25 to 35 percent of total volume.

The recommended production workflow for federal statistics:

1. LLM codes all records and assigns confidence scores
2. Records at confidence >= threshold: auto-accept the LLM code
3. Records at moderate confidence (below threshold): route to human review with LLM suggestion visible
4. Records flagged "UNCLEAR": full human coding (LLM output optionally shown as reference)
5. Five to ten percent random sample of auto-accepted records: human audit to monitor accuracy drift

The threshold itself is a policy decision that depends on the consequence of error for your specific program. A two-digit NAICS code used for descriptive tabulations tolerates more error than a six-digit code used for regulatory classification.

## 10. When to use LLMs and when not to

The decision of whether to use LLM coding for a specific application depends on volume, accuracy requirements, data sensitivity, and available infrastructure. LLMs have clear advantages for high-volume tasks with moderate classification depth and informal input text. They have clear disadvantages for fine-grained classification (6-digit NAICS), legally sensitive applications where audit trails are required, and highly sensitive data where on-premise infrastructure is not available.

| Use case | Use LLM | Use traditional / human |
|----------|---------|------------------------|
| Volume | High (> 100K records) | Low (manual feasible) |
| Ambiguity level | Moderate (needs judgment) | Low (clear rules exist) |
| Required consistency | Moderate | High (legal / regulatory) |
| Classification depth | 2-digit sector | 6-digit industry |
| Data sensitivity | De-identified or FedRAMP | High PII: on-prem only |
| Multilingual | Consider separate evaluation | Rule-based may fail |
| First pass vs. final | First pass + human review | Final coded output |

For a structured evaluation tool covering all relevant dimensions, see the 10-dimension rubric in Chapter 14.

## 11. In-class activity

```{admonition} Activity: analyze the worst-performing sectors
:class: tip

Using the simulated evaluation dataset in `examples/chapter-12/09_activity.py`:

1. Run the script and identify the two sectors with the lowest accuracy. Examine the miscoded examples. Do the errors make sense? Are they genuinely ambiguous, or clear misses?

2. For the worst-performing sector, write a revised few-shot prompt that includes examples specifically designed to address the failure pattern you observed. No LLM call needed: just write the prompt text. Would your examples help? Why or why not?

3. Find the confidence threshold that achieves 95 percent accuracy on auto-coded records while maximizing the automation rate. How many records (out of 200) would require human review?

**Extension:** The dataset contains only 2-digit sector codes. How would accuracy likely change at the 3-digit subsector level? What additional information in the description would help the model resolve the ambiguity?
```

See `examples/chapter-12/09_activity.py` for starter code with TODO markers.

## 12. Key takeaways for survey methodology

- *LLMs are a tool for text classification, not a replacement for classification systems.* They excel at mapping ambiguous natural language to standard codes, handling abbreviations and misspellings, and generalizing across description types. They do not replace the classification scheme itself, the subject matter knowledge of human coders, or the audit infrastructure that federal statistics requires.
- *Accuracy at the 2-digit sector level (roughly 82 to 88 percent for large frontier models — a simulated range based on published vendor reports rather than a standardized public benchmark — as of early 2026) is not the same as accuracy at the 6-digit NAICS level.* Published studies show sharp accuracy drops at finer classification levels. Evaluate at the level your program actually uses.
- *Cohen's kappa is more informative than raw accuracy for unbalanced classification tasks.* A model that always predicts the most common sector on an imbalanced dataset will achieve that sector's base rate as its accuracy. Kappa corrects for chance agreement.
- *Reproducibility is a first-class requirement in federal statistics.* LLM outputs are stochastic. Temperature=0 reduces but does not eliminate variability across API versions. Every production LLM coding system must log model version, prompt template version, and raw outputs. This is not optional.
- *Privacy constraints determine which deployment options are available.* Survey responses containing PII cannot be sent to commercial APIs without FedRAMP authorization and a data use agreement. On-premise deployment is the most secure option and may be the only option for Title 13 data.
- *The hybrid human-LLM workflow is almost always better than either alone.* Auto-accept high-confidence, high-accuracy cases. Route uncertain cases to human review with the LLM suggestion visible. Audit auto-accepted cases at a 5 to 10 percent rate to catch accuracy drift over time.
- *This chapter fits into the chapter arc: transformer architecture (Chapter 11) to LLM application (Chapter 12) to agentic pipeline design (Chapter 13) to AI system governance and evaluation (Chapter 14).* Understanding attention mechanisms from Chapter 11 explains why LLMs handle descriptions with complex context dependencies better than simple pattern matching. Chapter 13 extends these ideas to multi-step AI workflows. Chapter 14 provides the governance and evaluation framework for deploying any of these approaches in federal operations.

```{admonition} How to explain these methods to leadership
:class: dropdown

**On why LLMs matter for survey coding:**
"We spend approximately X million dollars per year on human coders for industry and occupation classification. These are skilled jobs that require substantial training. Large language models can now match trained coder accuracy on roughly 80 to 85 percent of cases at the 2-digit level. If we can automate 70 percent of the coding workload and direct human attention to the 30 percent of ambiguous cases, we reduce costs and may improve turnaround time significantly."

**On the risks:**
"LLMs are not deterministic. The same description sent to the same model twice can get different codes. That is not acceptable for published statistics. We are designing a system that pins the model version, logs every prompt and response, and sends a random audit sample to human review. Before deployment, we will validate that the auto-coded records meet accuracy thresholds established in our quality standards."

**On the privacy question:**
"Survey responses may contain personally identifiable information. Sending them to a commercial cloud service without a formal data use agreement and FedRAMP authorization would violate our legal obligations. We are evaluating FedRAMP-authorized cloud options and on-premise open-source models as the two compliant paths. We will not use an unapproved external API regardless of the potential cost savings."

**On the connection to AI strategy:**
"Industry and occupation coding is a concrete, high-value, low-risk starting point for LLM adoption in our program. The task is well-defined, the output is easily auditable (a code is right or wrong), and we have decades of human-coded data to use for validation. This is the kind of use case we should tackle before moving to less structured applications like narrative generation or data analysis assistance."
```
