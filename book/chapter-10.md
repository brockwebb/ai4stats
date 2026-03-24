# Chapter 10 - Statistical Disclosure Limitation in the Age of AI

> "A model trained on our microdata knows things about our respondents. We need to apply the same care to releasing that model as we apply to releasing the data it was trained on."

```{admonition} Who is this for?
Anyone involved in disclosure review, data governance, or AI deployment decisions at a statistical agency. No ML prerequisites; this chapter is about governance, not implementation. Builds on Chapter 9's synthetic data and differential privacy concepts.
```

```{admonition} Why this matters for federal statistics
All principal statistical agencies are being pushed to adopt AI while modernizing SDL under resource constraints. Trained models, model weights, and API endpoints are new disclosure channels that existing SDL frameworks do not adequately address. The federal statistical system has the infrastructure to close these gaps — but only if it chooses to.
```

---

## Learning Goals

By the end of this chapter, you should be able to:

- Explain why trained models, model weights, and API endpoints constitute new disclosure channels requiring SDL review
- Describe at least two concrete attack types (membership inference, model inversion) in plain language, without mathematical prerequisites
- Identify the three structural gaps in current federal SDL governance: specification, enforcement, and impact
- Apply the SDL evaluation checklist to assess readiness of a proposed AI deployment
- Articulate why model weights trained on confidential data should be classified as restricted statistical data under extended SDL definitions

---

## 1. This Is a Federal Statistical System Problem

It is tempting to frame AI and SDL as a Census Bureau challenge, since the 2020 Census disclosure avoidance system attracted the most public attention. That framing is too narrow. Every principal statistical agency faces the same challenge, and the federal statistical infrastructure that links them creates system-level risks that no single agency can address alone.

Consider the breadth of confidential data across the system. The Bureau of Labor Statistics maintains linked Current Population Survey records, Job Openings and Labor Turnover Survey microdata, and longitudinal youth surveys spanning decades. A model trained on linked CPS records and deployed as an internal research tool creates a disclosure channel that BLS's existing SDL policies — written in an era of published tables and public-use files — simply do not address.

The National Center for Health Statistics links survey records to electronic health records and mortality data. Health-risk models trained on those linked records can expose not just individual attributes but the fact of participation in a disease cohort. The NCHS data stewardship program is rigorous, but its formal policies say nothing about what happens when a neural network learns to predict mortality risk from linked microdata.

The Bureau of Economic Analysis works with firm-level national accounts data under a legal confidentiality framework that explicitly prohibits release in identifiable form. BEA's lawyers have thought carefully about what "identifiable form" means for tables and microdata files. They have not been required to think about what it means for model weights trained on firm-level data.

The National Agricultural Statistics Service operates one of the federal system's strongest enclave environments: a data lab with virtual enclave controls, a Data Access and Disclosure Review Board, and explicit output review. The NASS enclave was designed for researchers producing tables and estimates. It was not designed to govern the export of trained models from that environment.

The shared infrastructure connecting these agencies amplifies the stakes. The Federal Committee on Statistical Methodology sets quality and confidentiality standards that propagate across all principal agencies. OMB guidance on statistical policy shapes every agency's disclosure review. The Federal Statistical Research Data Center network connects agency data to research institutions through a common governance model. A gap in FCSM or OMB guidance on AI and SDL is not one agency's problem; it is everyone's problem simultaneously.

The central argument of this chapter is straightforward: trained models and the weights that encode them are a new category of statistical asset. They require SDL review. That review needs to happen across all principal agencies, not just the ones that have already attracted public attention.

---

## 2. New Disclosure Risks That Traditional SDL Did Not Anticipate

Traditional SDL was designed around a specific threat model. A motivated adversary has access to external data — voter rolls, commercial databases, public records — and attempts to link those external records to information disclosed in published statistics. SDL techniques such as cell suppression, data swapping, rounding, and noise injection were calibrated to defeat that adversary.

AI changes the attack surface in ways the traditional threat model does not anticipate. None of what follows requires advanced technical knowledge to understand. These are governance concepts, not implementation details.

### Model Inversion Attacks

A model inversion attack works by treating a trained model as a database that can be queried into revealing its training data. An adversary sends repeated queries to the model, observes the outputs, and uses patterns in those outputs to reconstruct sensitive features that were present in the training records.

The practical implication for federal statistics is concrete. Suppose an agency trains a model to predict benefit eligibility from survey microdata and exposes it through an API. An adversary with access to that API does not need the underlying microdata. They can query the model systematically and, over many queries, reconstruct enough about the training distribution to infer sensitive individual-level information. Suppression rules on the original data do not protect against this; those rules were applied before the data reached the model.

### Membership Inference Attacks

A membership inference attack addresses a different question: was this specific individual included in the training data? If an attacker can answer that question reliably, they have disclosed participation in a dataset — and participation itself can be sensitive information.

The example that makes this concrete for health data: if an agency trains a model on records linked to a disease registry and an adversary can determine that a specific person's record was in the training data, the adversary now knows that person has the disease. The model never outputs that fact directly. But the membership inference attack extracts it from the model's behavior.

This is not a theoretical concern. Membership inference attacks have been demonstrated on commercially deployed models, on medical AI systems, and on models trained with standard machine learning pipelines. The companion scripts in `examples/chapter-10/` demonstrate the attack on a simple simulated dataset so you can observe the disclosure signal directly.

### Leakage from Synthetic Outputs

Chapter 9 covered differential privacy and synthetic data generation in detail. One implication worth reinforcing here: synthetic data generated by a generative model is not automatically safe. If the generative model memorizes rare or unique records from the training data, it may reproduce those records — or close approximations — in synthetic output. The synthetic label provides false assurance.

This is particularly relevant for health and administrative record linkages where rare combinations of attributes (a very specific occupation, age, geographic area, and health condition) make individuals effectively unique. A generative model trained on such data can memorize and reproduce those rare combinations. SDL review of synthetic data needs to account for this.

### Memorization in Neural Networks

Large neural networks, including the language models now being evaluated for federal applications, can memorize training data verbatim. Research has demonstrated that LLMs will reproduce rare or unique records when prompted in specific ways, even records that appear in training data only once. For a model trained on confidential microdata, this means individual records can potentially be extracted from the model through careful prompting.

The key insight for SDL governance is that memorization is not a bug to be patched; it is a structural property of high-capacity models trained on large datasets. The mitigation is differential privacy during training (covered in Chapter 9) and strict access controls on the deployed model. But those mitigations need to be applied deliberately, which requires someone to ask the question.

The unifying point across all four attack types: trained models and API endpoints must be treated as potential disclosure channels on par with microdata releases. The SDL review process should begin when training data is selected, not when results are published.

For the mechanics of differential privacy and synthetic data generation that underlie some of these mitigations, see Chapter 9.

---

## 3. Three Gaps in Current SDL Governance for AI

Federal SDL governance is not absent. FCSM standards, OMB guidance, agency-level disclosure review, and the FSRDC governance framework collectively represent decades of serious institutional investment. The problem is not that the infrastructure does not exist; it is that the infrastructure has three structural gaps that allow AI-era risks to pass through unaddressed.

### The Specification Gap

SDL policies across federal agencies define "release" primarily in terms of microdata files, published tables, and public-use data products. That definition was adequate when those were the only things agencies released. It is no longer adequate.

When a researcher completes work in an FSRDC and exports results, the FSRDC coordinator reviews those outputs before release. The review process is rigorous and well-documented. The question the review process was not designed to answer is: what happens when the "output" is a trained model? The FSRDC governance documents do not specify whether model weights count as outputs subject to review, or whether they are simply IT artifacts that follow standard data handling rules. That ambiguity is the specification gap.

The consequence is that policies undershoot their own confidentiality goals. An agency with a genuine commitment to respondent confidentiality and a well-staffed disclosure review function may nonetheless be releasing trained models — or permitting researchers to export them — without SDL review, simply because the definition of "release" was never updated to include them.

Closing the specification gap requires a definitional change: model weights, embeddings, and high-capacity API endpoints trained on confidential data should be classified as statistical outputs subject to disclosure review. That classification does not require new enforcement machinery; it requires extending the existing definitions.

### The Enforcement Gap

Federal SDL has strong policies on paper and uneven enforcement in practice. This is not a critique of disclosure review staff, who are generally skilled and dedicated. It is a structural observation: the gap between what policies require and what practices actually implement tends to widen when new technology appears faster than policy guidance can be updated.

Two examples illustrate the gap.

BLS's FSRDC coordinators review all researcher outputs before release. That review process is one of the strongest in the federal statistical system. But the definition of "outputs" that coordinators review does not explicitly include trained models. A researcher who trains a model on linked CPS microdata and wishes to export it faces review procedures calibrated for tables and estimates. The reviewer may flag it or may not; the policy does not specify.

NASS's data lab has a virtual enclave with strict access controls and an explicit Data Access and Disclosure Review Board. NASS has invested more in enclave governance than most federal statistical agencies. And yet NASS's published enclave policies do not include explicit model governance provisions. A trained geospatial model incorporating confidential farm-level data faces no explicitly documented review requirement before export.

Closing the enforcement gap requires two things: explicit procedural requirements that name models and APIs as objects of review, and audit mechanisms that verify those requirements are being met. Neither requires new institutions; both require deliberate policy updates.

### The Impact Gap

The third gap is the hardest to close because it is cultural rather than procedural. Agencies produce disclosure impact assessments. Those assessments describe potential risks. They are often thorough and well-reasoned. And they frequently have little effect on what engineers actually build.

When a disclosure impact assessment identifies a risk but does not include a requirement to apply a specific mitigation, the assessment becomes documentation rather than a design constraint. An API with no differential privacy budget, no rate limiting, and no query auditing may have a disclosure impact assessment that accurately describes the membership inference risk it poses. The assessment satisfies the paperwork requirement. The risk remains.

The deeper problem is that documentation and design are decoupled. SDL review happens at the end of the pipeline, after engineering decisions have been made. Requiring documentation of risks without requiring mitigation of those risks produces what might fairly be called AI transparency theater: agencies document inputs and outputs but do not actually constrain how models can leak respondent information.

Closing the impact gap requires coupling documentation requirements to design requirements. An agency should not be able to satisfy its SDL obligations for an AI system by documenting a risk and moving on. Satisfying SDL obligations for a high-risk AI system should require demonstrable mitigation: a DP guarantee with a specified epsilon, a tested membership inference rate below a specified threshold, or an explicit governance decision that the risk is acceptable and why.

---

## 4. Model Weights as Restricted Statistical Data

This section describes a normative proposal, not current practice. Federal SDL does not currently classify model weights as a category of restricted data. The argument here is that it should.

The core argument is an extension of existing SDL logic. Federal statistical agencies already apply disclosure review to synthetic microdata generated from confidential records. The synthetic data is a derived product; it encodes information about the source data and therefore requires review before release. That classification is well-established and uncontroversial.

Model weights are also a derived product. A neural network trained on confidential microdata encodes information about that data in its parameters. The encoding is less direct than synthetic data — you cannot simply read a record out of the weights — but the information is there, extractable through the attack methods described in section 2. If synthetic microdata from a survey is subject to SDL review, a model trained on the same survey data should be too.

The logical extension of existing practice points to a new category: "confidential model." A confidential model is a trained model whose weights encode information from confidential microdata and which is therefore subject to disclosure review before release or external access. The review process would be analogous to the review applied to other derived statistical products: an assessment of the disclosure risk posed by the weights, with a determination about whether the model can be released, released with restrictions, or held in a restricted environment.

Connecting this to existing infrastructure makes the proposal more tractable. Agencies already license access to high-risk derived data files under controlled environments. Some synthetic data requires a verification server rather than direct release. The FSRDC network already operates a system of controlled access to confidential data. Classifying certain trained models as confidential under that same infrastructure does not require building new institutions; it requires extending existing classifications.

Three governance questions follow directly from this framing, and they are questions FCSM and OMB are positioned to address:

1. Who signs off on exporting a model trained in an FSRDC? The answer should be as explicit as the sign-off required for exporting tables.
2. Is there a concept of a "confidential model" analogous to a confidential dataset, with associated access restrictions and handling requirements?
3. Should FSRDC governance adopt a system-wide policy that any multi-agency model trained in an FSRDC is presumptively confidential unless a disclosure review determines otherwise?

These are not rhetorical questions. They are governance gaps with tractable answers, if the institutions choose to provide them.

---

## 5. Are APIs and Query Systems "Releases"?

The FCSM and NCES data access framework organizes statistical data products into tiers: public-use files, licensed microdata, secure research data centers, and query-based access systems. Most SDL guidance treats query systems as lower-risk because they provide controlled, per-query access rather than wholesale data release. Per-query noise, rate limiting, and output review can, in principle, limit cumulative disclosure.

The key question for the AI era is whether that lower-risk classification still holds when the query system is a high-capacity model endpoint.

A traditional query system returns a sum, a mean, or a count from a database. The SDL risk is bounded and well-understood. A model endpoint trained on confidential data is a different kind of query system. Each query to the model can yield probabilistic outputs encoding information about training records. An adversary with unrestricted access to the endpoint can execute membership inference and model inversion attacks iteratively. Over many queries, they can extract substantially more information than any single query reveals. Rate limiting and per-query review do not prevent this; they only slow it.

The question this raises for FCSM is whether existing SDL definitions cover this case. Two specific questions warrant formal attention:

1. Under current SDL definitions, does exposing a high-capacity model API from within a statistical agency — to researchers, to partner agencies, or to the public — count as a "release"? If not, the endpoint is effectively outside the SDL review process regardless of what it can disclose.

2. If the current answer is no, should FCSM and OMB revise guidance so that any model endpoint trained on confidential data is deemed a release unless it meets explicit safeguards — differential privacy with a documented epsilon, rate limiting, query auditing, and documented membership inference testing?

The framing matters: agencies that have never considered their model APIs as "releases" will continue not to subject them to SDL review until the definition changes. Definitional clarity is a prerequisite for enforcement.

---

## 6. Closing the Gaps — Concrete Proposals

The three gaps described in section 3 each have concrete remedies. What follows is a set of proposals framed as actions FCSM, OMB, and agencies could actually adopt — not aspirational principles, but specific governance changes.

### Closing the Specification Gap

The most direct path is definitional. SDL policies need explicit language classifying model weights, embeddings, and high-capacity API endpoints trained on confidential data as statistical outputs subject to disclosure review. That language should introduce two new release categories: "confidential model" and "restricted API."

Agencies should also be required to state explicitly what privacy threat models their SDL review addresses. A disclosure impact assessment that does not mention membership inference or model inversion has not assessed the disclosure risks of a trained model; it has assessed the disclosure risks of a table or a public-use file and applied that assessment by analogy. Requiring explicit threat model specification forces reviewers to engage with the actual risk.

### Closing the Enforcement Gap

Three enforcement mechanisms would materially close the gap between policy and practice.

Cross-agency audits for high-risk statistical models should become a standard FCSM activity, analogous to the peer review and replication processes that exist for statistical methods. An audit would test a sample of models trained on confidential data for membership inference vulnerability and model inversion risk, using documented attack protocols. The results would inform both agency practice and guidance updates.

Pre-deployment gates should be mandatory: any model trained on restricted data must pass documented privacy and security review before exposure outside a secure enclave. "Documented" means a written record of what was tested, what thresholds were applied, and who signed off. The review should be a prerequisite for deployment, not a post-hoc documentation exercise.

Vendor and researcher partnership contracts should embed SDL and AI requirements as standard terms. When a contractor builds an AI system using agency data, the contract should specify what privacy testing is required, what access controls must be implemented, and who retains governance authority over model weights.

### Closing the Impact Gap

Closing the impact gap requires coupling disclosure documentation to design requirements.

Transparency disclosures should be aligned with the forums that can act on them. Technical disclosure reports on membership inference testing belong in front of FSRDC review boards, not in public-facing documents that satisfy a paperwork requirement without reaching the people who can change engineering decisions.

Documentation mandates should be linked to technique requirements. If an agency documents a membership inference risk for a deployed API, that documentation should trigger a requirement to either implement a mitigation or document an explicit governance decision that the risk is acceptable at the current level.

Outcome-based metrics would give agencies a way to track progress: attack simulation results, near-breach incidents, and respondent trust surveys provide measurable indicators of whether SDL is working, not just whether paperwork has been filed.

---

## 7. Cross-Agency Longitudinal SDL (Forward-Looking)

One emerging challenge deserves brief attention even though federal practice has not yet converged on approaches to address it.

Several agencies are exploring shared synthetic longitudinal panels as a way to make linked longitudinal data more accessible without exposing raw microdata. The Cornell-Census verification server model, which allows researchers to validate results against real data after developing methods on synthetic data, points toward a promising architecture. Differential privacy for continual releases, where privacy budgets are managed across ongoing longitudinal data collections rather than single-point releases, is an active research area with direct federal applications.

The governance challenge for these shared resources is that no single agency owns them. A synthetic panel built from BLS and Census linked data, released through a shared verification server, requires governance that spans both agencies and the FSRDC network. FCSM is the natural home for that governance. Cross-agency steering through FCSM and FSRDC governance structures already exists; extending it to cover shared AI-enabled data products is the logical next step.

This is forward-looking. The infrastructure does not yet exist at scale. But the governance conversations should begin before the technical decisions are made, not after.

---

```{figure} images/ch10_sdl_risk_decision_tree.png
:name: fig-ch10-sdl-risk-tree
:width: 100%
:alt: SDL Risk Classification Decision Tree for AI Deployments at Federal Statistical Agencies

SDL Risk Classification Decision Tree for AI Deployments at Federal Statistical Agencies. Decision nodes assess data sensitivity, model memorization capacity, access mode, and privacy controls. Terminal nodes map to four risk tiers with corresponding review requirements. Framework alignment annotations reference NIST AI RMF, NIST AI 600-1, FCSM Data Protection Toolkit, and OMB M-24-10.
```

## 8. SDL Evaluation Checklist for AI Deployments

This checklist is designed for SDL reviewers evaluating a proposed AI deployment involving confidential statistical data. It is organized by phase of the deployment lifecycle. Not every question applies to every deployment; the checklist should be used to identify which questions are relevant, not as a compliance form to be completed mechanically.

**Before training:**
- What data will be used for training? At what level of aggregation?
- Is this data classified as confidential, restricted, or public-use?
- Has a privacy budget been established? What is the approved epsilon (if DP is required)?
- Has training data provenance been documented? Which surveys, linkages, and vintages?
- Has the training purpose been approved under the applicable data use agreement?

**Model architecture:**
- What is the model's capacity? Is it a simple rule-based system, a shallow tree, a deep neural network, or a large language model?
- Does the architecture include components (embeddings, attention weights) that could be extracted and used for linkage?
- Has the architecture been reviewed for memorization risk of rare records?
- Are differential privacy training techniques (DP-SGD or equivalent) required given data sensitivity?

**Before deployment:**
- Has the model been tested for membership inference vulnerability? What attack success rate was observed?
- Has the model been tested for model inversion risk?
- If DP was applied during training, what is the guaranteed epsilon? Is this documented?
- Has a disclosure impact assessment been completed that explicitly addresses AI-era attack types?
- Who has sign-off authority for deployment? Is this documented?

**Access control:**
- Will the model be accessible outside a secure enclave? If so, what controls are in place?
- Is external access rate-limited? What are the per-user and per-time-window limits?
- Are all queries logged? How long are logs retained? Who can access them?
- Is query auditing active — that is, is someone actually reviewing logs for anomalous query patterns?

**Ongoing monitoring:**
- Is there a documented process for re-evaluating disclosure risk as the model is updated?
- Is there a process for responding to newly discovered attacks that may apply to deployed models?
- Is there a point of contact responsible for SDL compliance for this specific deployment?
- Is the deployment included in any cross-agency audit scope?

---

## Key Takeaways

- AI systems trained on confidential data are disclosure channels, not merely IT assets. Membership inference attacks and model inversion attacks are concrete, demonstrated risks, not theoretical concerns.
- The federal statistical system has existing SDL infrastructure — FCSM standards, FSRDC governance, agency disclosure review functions — that can be extended to address AI-era risks. The goal is extension, not replacement.
- Three structural gaps (specification, enforcement, and impact) provide a diagnostic framework for evaluating any agency's readiness. Each gap has tractable remedies.
- Model weights trained on confidential microdata should be classified as restricted statistical data, subject to the same disclosure review applied to synthetic data and other derived statistical products.
- API endpoints trained on confidential data should be treated as releases under SDL definitions unless they meet explicit safeguards: differential privacy with a documented epsilon, rate limiting, query auditing, and documented membership inference testing.
- Closing the gaps requires concrete governance changes — definitional updates, pre-deployment gates, contractual requirements, and outcome-based metrics — not just additional documentation.
- Cross-agency longitudinal data products require cross-agency governance; FCSM and the FSRDC network are the natural institutions to provide it.
- For the mechanics of differential privacy and synthetic data generation that underlie some of these mitigations, see Chapter 9. For fairness and bias frameworks applicable to the same systems, see Chapter 8. For evaluation frameworks, see Chapter 14. For State Fidelity Validity as a validity framework for AI-assisted research pipelines, see Chapter 15.

---

```{dropdown} How to explain this to leadership
Plain language talking points:

"We have spent decades explaining why we cannot publish individual records. The same logic applies to AI: a model trained on our microdata has absorbed information about our respondents. If we release that model, or its outputs, without the same care we apply to our data releases, we are effectively publishing information we have promised to protect."

"The good news is that we already have the infrastructure. The FSRDC network, FCSM standards, and OMB guidance give us the foundation. We need to extend those frameworks to cover models, not invent new ones."

"The three gaps — specification, enforcement, and impact — give us a practical diagnosis. Which parts of our SDL policy do not mention models? Where are our enforcement mechanisms weakest? Are our disclosure assessments actually changing what our engineers build?"
```

---

## Exercises

1. **Scenario — BLS deployment review.** A BLS analyst proposes training a model on linked CPS microdata and deploying it as an internal API accessible to BLS staff. Using the SDL evaluation checklist in section 8, identify at least three governance questions that must be answered before deployment. For each question, explain what is at stake if it is not addressed before the model goes live.

2. **Classification — Extended SDL definitions.** For each of the following, determine whether it should be classified as a "release" under the proposed expanded SDL definitions described in sections 4 and 5. Briefly justify each determination.
   - (a) Model weights exported from an FSRDC by a researcher who trained the model on confidential linked microdata.
   - (b) An internal-only API with rate limiting, serving a model trained on restricted survey data, accessible only to cleared agency staff.
   - (c) A synthetic dataset generated by a neural network trained on confidential health records, released as a public-use file.
   - (d) Published summary statistics from a differential privacy query system with documented epsilon.

3. **Gap analysis — Agency SDL posture.** Select one principal statistical agency (BLS, NCHS, BEA, NASS, or Census). Using publicly available documentation — agency websites, FSRDC governance documents, FCSM working papers, agency privacy impact assessments — identify one concrete example of each gap (specification, enforcement, impact) in their current AI and SDL posture. For each gap, propose one specific change that the agency or FCSM could make to begin closing it.
