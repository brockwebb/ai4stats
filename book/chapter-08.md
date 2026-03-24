# Chapter 8 - Bias, Fairness, and Equity in Federal AI/ML

## Learning goals

By the end of this chapter, you will be able to:

1. Distinguish between statistical bias and algorithmic bias, and explain why both matter in federal statistical production
2. Identify sources of bias at each stage of an ML pipeline: training data, feature selection, label quality, model choice, and evaluation metrics
3. Define and interpret four core fairness metrics: demographic parity, equalized odds, calibration, and predictive parity
4. Explain why no model can simultaneously satisfy all fairness criteria when base rates differ across groups
5. Apply subgroup accuracy decomposition to a classification model and interpret the results
6. Recognize where bias enters federal-specific workflows: nonresponse adjustment, imputation, coverage estimation, and automated coding
7. Document fairness tradeoffs as a governance requirement, not optional analysis
8. Evaluate vendor fairness claims using a structured checklist
9. Identify OMB and executive order requirements relevant to AI equity documentation

---

## Why this matters for federal statistics

When a model is 94% accurate overall, who bears the 6% error burden?

This question is not rhetorical. Federal surveys use machine learning models to predict nonresponse, impute missing income values, code occupation descriptions, and flag records for quality review. If those models perform differently across demographic groups, the resulting statistics are systematically less accurate for some communities than for others. That inaccuracy flows downstream into congressional apportionment, federal funding formulas, and redistricting.

The 2020 Census Post-Enumeration Survey documented the stakes directly:

| Group | Net undercount rate |
|-------|---------------------|
| American Indian/Alaska Native (on reservation) | +5.64% (undercounted) |
| Hispanic | +4.99% (undercounted) |
| Black non-Hispanic | +3.30% (undercounted) |
| Native Hawaiian/Other Pacific Islander | +1.92% (undercounted) |
| White non-Hispanic | -1.64% (overcounted) |
| Asian non-Hispanic | -2.62% (overcounted) |

Positive values mean the group was missed at a higher rate than average. Negative values mean the group was counted more than once.

Source: Census Bureau Post-Enumeration Survey, G-01 report, November 2022.

The differential undercount is not the result of malice. It is the result of measurement processes that are harder to execute in some communities: language barriers, housing instability, historical distrust of government data collection, and address-based sampling frames that undercount non-traditional housing. When you build an ML model on top of this data, the model learns from a picture of the population that already encodes differential coverage error.

See `examples/chapter-08/02_differential_undercount.py` for the full bar chart visualization.

---

## 1. What "bias" means (and does not mean)

The word "bias" means different things to statisticians and to computer scientists working on fairness. Both meanings matter in federal statistical work, and they are not the same thing.

### 1.1 Statistical bias: systematic error in an estimator

Statistical bias is a property of an estimator. An estimator is *biased* if its expected value differs from the true parameter it is estimating. This is a mathematical property. A biased estimator is not "unfair" in any moral sense; it simply does not produce the right answer on average.

If you estimate mean household income using only survey respondents, and nonrespondents have systematically lower incomes, your estimator is biased -- the estimate will be too high on average.

### 1.2 Algorithmic bias: disparate impact across groups

Algorithmic bias refers to systematic differences in model performance or impact across demographic groups. A model exhibits algorithmic bias when its errors, predictions, or outputs fall disproportionately on specific populations.

Critically: *a statistically unbiased estimator can still produce disparate impact.* A model that produces correct estimates on average can still be systematically worse for Black households than for white households. Overall accuracy is not a sufficient guarantee of fairness.

### 1.3 Why the distinction matters for practitioners

Federal statisticians often encounter both types simultaneously. A nonresponse weighting model can be designed to produce unbiased national estimates (statistical bias controlled) while still systematically underperforming for Hispanic households (algorithmic bias present). The two properties are independent, and both require explicit attention.

The practical implication: *controlling for one type of bias does not control for the other.* Evaluating a model requires checking both.

---

## 2. Sources of bias in ML pipelines

Bias can enter a machine learning pipeline at many points. Understanding where it enters is the prerequisite for addressing it.

### 2.1 Training data bias

If your training data underrepresents a population, the model learns less about that population. For nonresponse prediction, this is a direct problem: you have training labels only for people who participated in prior surveys. People who did not participate are, by definition, missing from your training data. You are building a model to predict absence from a dataset that only contains presence.

This is *survivorship bias*: the training data is structurally filtered to include only the cases that survived a selection process (survey participation), and that filter is correlated with the outcome you are trying to predict.

The representation ratio makes this quantitative. If Hispanic households make up 18% of the population but only 14% of survey respondents, the representation ratio is 0.78 -- meaning the respondent-only training data sees 22% fewer Hispanic households than the actual population. The model trained on this data has less information about Hispanic households than about any other group, at exactly the moment when accurate prediction for that group matters most.

See `examples/chapter-08/03_training_data_bias.py` for the side-by-side composition comparison.

### 2.2 Feature selection bias

Including geography as a feature encodes historical segregation patterns into the model. Excluding geography hides spatial disparities. There is no neutral choice. Address-based features correlate with race/ethnicity due to residential segregation, even if race/ethnicity is not explicitly included as a feature.

Detection: check the correlation between model features and protected characteristics. If a feature like zip code is highly correlated with race/ethnicity, the model is effectively using race as a predictor even when it is not in the feature set.

### 2.3 Label quality bias

Human coders are inconsistent, and that inconsistency is not random across populations. Occupation coding errors are higher for non-standard English descriptions. A model trained on human-coded training data learns the coders' inconsistencies. If coders were less consistent for a particular population, the model will be less accurate for that population, and the evaluation metrics may not flag this if accuracy is only reported overall.

Detection: measure inter-rater agreement by the demographic characteristics of the text being coded. A 10-point gap in inter-rater agreement by language group indicates a training data quality problem that the model cannot fix.

### 2.4 Model choice bias

Decision trees make large-leaf majority-class decisions that apply to entire subgroups. A decision tree might assign a single imputed income value to all Hispanic households in a census tract, regardless of individual variation within that group, because the leaf node is defined by the majority pattern. The within-group variation that a logistic regression or random forest would preserve is discarded.

Detection: compare subgroup accuracy across model types -- logistic regression vs. decision tree vs. random forest. If accuracy gaps are larger for tree-based models, that is evidence of model-choice bias.

### 2.5 Evaluation metric bias

Overall accuracy is a weighted average. If Group A makes up 60% of the dataset and Group B makes up 6%, overall accuracy is dominated by Group A performance. A model that achieves 94% accuracy overall can be 72% accurate for Group B, and the 94% headline number will never reveal this.

Detection: subgroup decomposition of every reported metric. This is not optional analysis -- it is the minimum evaluation requirement for any model in federal statistical production.

---

## 3. Fairness metrics: what they measure and why they conflict

There is no single definition of "fairness." Different definitions measure different properties, and each has a different normative justification. Understanding the definitions is prerequisite to knowing which one applies to your problem.

### 3.1 The four core metrics

**Demographic parity** requires that the positive prediction rate -- the fraction of people the model predicts as nonrespondents -- is equal across groups.

- Formula: P(ŷ=1 | group=A) = P(ŷ=1 | group=B)
- Normative basis: equal treatment. The model acts on each group at the same rate.
- Limitation: ignores whether predictions are correct. A model that randomly flags equal fractions from each group satisfies demographic parity while being useless.
- Federal context: would require that automated coding flags equal fractions from each group for human review.

**Equalized odds** requires that both the true positive rate (recall) and the false positive rate are equal across groups.

- Formula: P(ŷ=1 | y=1, A) = P(ŷ=1 | y=1, B) AND P(ŷ=1 | y=0, A) = P(ŷ=1 | y=0, B)
- Normative basis: equal error rates. The model makes equally costly mistakes for each group.
- Limitation: can require degrading performance for better-performing groups. Hard to achieve when base rates differ substantially.
- Federal context: would require that the nonresponse prediction model is equally sensitive and equally specific across racial groups.

**Calibration** requires that among all cases predicted with confidence p, the fraction of true positives is p, for every group.

- Formula: P(y=1 | ŷ_prob = p, A) = P(y=1 | ŷ_prob = p, B) = p
- Normative basis: equal reliability. A 70% confidence prediction means 70% probability for every group.
- Limitation: does not constrain which groups receive positive predictions. A perfectly calibrated model can still systematically over-predict for one group.
- Federal context: would require that when the model says "80% likely to be nonrespondent," that probability is accurate across all groups.

**Predictive parity** requires that among those predicted as positive, the fraction who are truly positive (precision) is equal across groups.

- Formula: P(y=1 | ŷ=1, A) = P(y=1 | ŷ=1, B)
- Normative basis: equal reliability of positive predictions. Being flagged means the same probability of the outcome across groups.
- Limitation: related to calibration but at a single threshold. Satisfying predictive parity at one threshold does not guarantee it at others.
- Federal context: would require that follow-up contact resources triggered by a positive prediction are equally well-targeted across groups.

### 3.2 Pre-computed fairness metrics for the Chapter 8 nonresponse model

The following table shows computed results from the logistic regression nonresponse model described in `examples/chapter-08/01_dataset_and_model.py` and `04_fairness_metrics.py`. The model was trained on 2,000 synthetic ACS-like records (see script for parameters).

| Group | N | Base Rate | Accuracy | Pred. Rate | TPR | FPR | Precision |
|-------|---|-----------|----------|------------|-----|-----|-----------|
| White non-Hispanic | 349 | 0.29 | 0.74 | 0.28 | 0.57 | 0.17 | 0.59 |
| Black non-Hispanic | 77 | 0.44 | 0.68 | 0.39 | 0.62 | 0.27 | 0.71 |
| Hispanic | 109 | 0.47 | 0.66 | 0.43 | 0.67 | 0.30 | 0.73 |
| Asian non-Hispanic | 35 | 0.20 | 0.77 | 0.17 | 0.43 | 0.07 | 0.50 |
| Other | 20 | 0.40 | 0.65 | 0.35 | 0.63 | 0.25 | 0.72 |

*Pred. Rate = positive prediction rate (demographic parity numerator); TPR = true positive rate (equalized odds); FPR = false positive rate; Precision = predictive parity.*

The pattern is consistent with what the impossibility theorem predicts: no metric is equal across all groups. Groups with higher base rates (Hispanic, Black non-Hispanic) have higher TPR but also higher FPR. Groups with lower base rates (Asian non-Hispanic) have lower TPR and lower FPR. The model cannot simultaneously equalize all four columns.

For the full figure, run `examples/chapter-08/04_fairness_metrics.py`.

---

## 4. The impossibility theorem

Here is the most important result in algorithmic fairness research: you *cannot* simultaneously satisfy all fairness criteria when base rates differ across groups.

This was proven independently by Chouldechova (2017) and Kleinberg, Mullainathan, and Raghavan (2016). The result is not a conjecture or a practical limitation. It is a mathematical theorem.

**Intuition:** If Group A has a 20% nonresponse rate and Group B has 40%, any model that is equally accurate for both groups must either:

- Predict at different rates (violating demographic parity), *or*
- Have different error rates across groups (violating equalized odds), *or*
- Have different precision across groups (violating calibration)

You must choose. The policy question is: which errors are most consequential?

The threshold sweep in `examples/chapter-08/05_impossibility_theorem.py` makes this concrete: across every decision threshold from 0.10 to 0.90, the Hispanic and Asian non-Hispanic groups -- which have the largest base rate difference in this dataset -- cannot both achieve equal TPR and equal precision simultaneously. Lowering the threshold to improve recall for the high-base-rate group increases that group's FPR. Raising it to reduce FPR reduces recall. At every threshold, at least one fairness criterion is violated for at least one group.

```{admonition} How to explain the impossibility theorem to leadership
:class: dropdown

"There is no such thing as a neutral algorithm. Every model that predicts outcomes across demographic groups makes implicit choices about which type of error is acceptable. The impossibility theorem tells us that we cannot simultaneously minimize all types of error for all groups when those groups have different underlying rates. What we can do is identify which errors are most consequential for our specific application, measure whether our model's errors fall disproportionately on specific groups, document our choices explicitly, and build human review into the pipeline for cases where the stakes are highest. An AI vendor who claims their model is fair to everyone either does not understand the impossibility theorem or is not being honest with you."
```

### 4.1 What the impossibility theorem implies for federal practice

The theorem does not mean fairness is unachievable. It means fairness requires a choice. The choice is a governance decision, not a technical one.

For each application, decision-makers must ask:

- *Which type of error has the most serious consequences?* A false negative (failing to identify a likely nonrespondent who then does not respond) has a different cost than a false positive (spending follow-up resources on someone who would have responded anyway).
- *Are those costs symmetric across groups?* If false negatives are more costly for communities that are already undercounted, minimizing false negatives for those groups takes priority, even if it requires accepting higher false positive rates for those groups.
- *Who decides?* This is a policy question, not a statistical one. Statisticians can calculate the tradeoffs. They cannot decide which tradeoffs are acceptable. That requires division chief sign-off and documentation.

---

## 5. Subgroup accuracy decomposition

The minimum requirement for any model used in federal statistical production is that its performance metrics are reported separately for each relevant demographic subgroup. Overall accuracy is not a sufficient summary.

### 5.1 Pre-computed subgroup decomposition results

The following table shows the decomposition from `examples/chapter-08/06_subgroup_decomposition.py`:

**By race/ethnicity:**

| Group | N | Base Rate | Accuracy | TPR | FNR (miss rate) | Precision |
|-------|---|-----------|----------|-----|-----------------|-----------|
| White non-Hispanic | 349 | 0.29 | 0.74 | 0.57 | 0.43 | 0.59 |
| Black non-Hispanic | 77 | 0.44 | 0.68 | 0.62 | 0.38 | 0.71 |
| Hispanic | 109 | 0.47 | 0.66 | 0.67 | 0.33 | 0.73 |
| Asian non-Hispanic | 35 | 0.20 | 0.77 | 0.43 | 0.57 | 0.50 |
| Other | 20 | 0.40 | 0.65 | 0.63 | 0.37 | 0.72 |

**By income quintile:**

| Quintile | N | Base Rate | Accuracy | TPR | FNR (miss rate) |
|----------|---|-----------|----------|-----|-----------------|
| Q1 (lowest) | 121 | 0.52 | 0.62 | 0.71 | 0.29 |
| Q2 | 120 | 0.40 | 0.67 | 0.60 | 0.40 |
| Q3 | 120 | 0.32 | 0.72 | 0.55 | 0.45 |
| Q4 | 120 | 0.23 | 0.77 | 0.48 | 0.52 |
| Q5 (highest) | 109 | 0.16 | 0.82 | 0.38 | 0.62 |

The FNR (miss rate) is the most operationally significant metric here. A missed nonrespondent is a person the model failed to flag for targeted follow-up, increasing the probability they remain uncounted. The groups with the highest miss rates are not random -- they are the groups with the highest underlying nonresponse rates, which is exactly the compounding effect described in Section 6.

### 5.2 Interpreting the decomposition

The income quintile decomposition reveals a pattern that the overall accuracy number (approximately 71%) completely conceals: accuracy is monotonically increasing from lowest to highest income quintile. The model is most accurate for households that need the least intervention and least accurate for households that are hardest to reach.

This is not surprising given the training data structure. High-income households have higher survey response rates, so the model sees more of them during training and learns their patterns better. Low-income households, who are disproportionately nonrespondents, are systematically underrepresented in the training data.

For the visualization, see `examples/chapter-08/06_subgroup_decomposition.py`.

---

## 6. Bias in federal-specific workflows

The general ML bias framework applies with particular force to several specific federal statistical workflows.

### 6.1 Nonresponse adjustment

Weighting models that underperform for hard-to-reach populations amplify existing undercounts. The compounding mechanism works as follows:

1. Underlying nonresponse rate is higher for historically undercounted groups
2. Model has a higher miss rate (FNR) for those groups -- less training data, survivorship bias
3. Follow-up resources are not targeted to those groups
4. Lower follow-up response rate
5. Post-survey weights must compensate harder
6. Higher variance in estimates for those groups
7. In the 2020 Census: 5.0% undercount for the Hispanic population

The compound risk -- the probability that a given person is both a true nonrespondent and missed by the model -- is the product of the base rate and the miss rate, not their sum. A group with a 45% nonresponse rate and a 35% model miss rate has a 15.75% compound risk of being both uncounted and untargeted. See `examples/chapter-08/07_compounding_effect.py` for the full calculation by group.

### 6.2 Imputation donor pools

Hot-deck imputation (Chapter 7) assigns missing values by drawing from a pool of donors with similar characteristics. If the donor pool is primarily composed of majority-group members for minority-group recipients, the imputed values may not reflect the actual distributions for that group.

The connection to pipeline integrity: if you made a deliberate decision to stratify donor pools by racial/ethnic group to address this problem, that decision must survive into the downstream synthetic data generation step (Chapter 9). If session loss or context compaction drops the "stratified donor pools" rationale, the downstream step may silently revert to unstratified pools, reintroducing the bias the imputation step was designed to mitigate.

### 6.3 Automated survey coding

LLM-based occupation coding (Chapter 12) may perform differently on occupation descriptions written in non-standard English, with code-switching, or in languages other than English. Research on automated coding systems consistently shows a pattern:

| Description type | Approximate coding accuracy |
|-----------------|----------------------------|
| Standard English, common occupation | ~95% |
| Standard English, uncommon occupation | ~82% |
| Non-standard English, common occupation | ~78% |
| Non-standard English, uncommon occupation | ~61% |
| Spanish-English code-switching | ~68% |
| Spanish (monolingual) | ~55% |

*These are illustrative rates from published research patterns, not computed from this chapter's dataset.*

A coding system that performs well on standard English descriptions introduces systematic miscoding for communities with different linguistic patterns. The resulting occupational statistics are biased in ways that are invisible when only overall accuracy is reported.

### 6.4 Synthetic data generation

Synthetic data generation (Chapter 9) can silently underrepresent tail distributions. If the generative model does not see enough members of a small subgroup to learn their joint distribution accurately, the synthetic population may effectively round that subgroup toward the majority pattern. Statistical analyses on synthetic data would then underestimate the diversity within and between small groups -- an invisible form of bias that propagates through every downstream analysis.

---

## 7. Evaluating a vendor's fairness claims

When a vendor presents an AI system for federal procurement, overall accuracy is insufficient evidence of fairness. Use this checklist:

**Did they report subgroup accuracy?**
If the vendor only reports overall accuracy, that is a red flag. Demand a subgroup decomposition by every OMB Statistical Policy Directive 15 race/ethnicity category and by income strata. If they have not computed it, they have not evaluated their system for the use case you care about.

**Which fairness metric did they optimize?**
Every model implicitly chooses a fairness criterion. If the vendor does not know which one they optimized -- or if they claim the model is "fair to all groups" without qualification -- they either do not understand the impossibility theorem or are not being forthcoming. The correct answer specifies a criterion (e.g., "we optimized TPR parity across race groups at the 0.5 threshold"), the justification, and the alternative criteria that were considered and deprioritized.

**Did they test on the target population or a convenience sample?**
A model validated on one year's CPS respondents has not been validated on ACS nonrespondents. Domain shift -- the difference between the validation population and the deployment population -- is a common source of silent failure after procurement.

**Do base rates differ across groups in your data?**
If yes, the impossibility theorem applies to the vendor's model exactly as it applies to yours. No model can simultaneously equalize all fairness criteria under these conditions. Ask the vendor which criteria they chose and which they did not.

**What happens when demographics shift?**
Population composition changes over time. A model calibrated on 2019 survey data may be miscalibrated for a 2026 deployment population. Ask whether the vendor has evaluated performance under demographic shift and what the retraining cadence is.

---

## 8. OMB and executive order requirements

Fairness documentation is not optional analysis. Federal agencies operate under specific governance requirements:

**OMB Statistical Policy Directive 15** establishes the standard race/ethnicity categories for federal data collection. Any AI model used in federal statistical production must be evaluated against these categories. Reporting performance only for "White" and "non-White" is not compliant with SPD-15.

**Executive Order on Safe, Secure, and Trustworthy Artificial Intelligence** (October 2023) and subsequent OMB guidance require that agencies document the equity implications of AI systems before deployment, maintain records of bias evaluations, and establish processes for ongoing monitoring. The documentation requirement applies to AI tools used in administrative functions as well as statistical production.

**The practical implication:** a model card (Section 9) or equivalent documentation is not a nicety. It is the record of accountability. When a system produces disparate impact -- and given the impossibility theorem, every system does in some sense -- the model card shows whether decision-makers knew about it, documented the tradeoff, and made a deliberate governance choice. Absence of documentation is not neutrality; it is an undocumented choice.

---

## 9. What to do about it

The impossibility theorem means there is no universal solution. But not all responses are equal.

**Measure it.** Compute subgroup decomposition for every metric on every model in production. At minimum: accuracy, TPR, FPR, and precision by every OMB SPD-15 race/ethnicity category and by income strata. You cannot manage what you do not measure. "Overall 94% accurate" tells you nothing about who bears the 6% error burden.

**Choose your fairness criterion explicitly.** Document which fairness metric the pipeline is optimized for and why. The minimum artifact is a written decision memo that identifies the metric, the justification, and the alternative metrics that were considered and deprioritized. The impossibility theorem guarantees you are making a choice. Making it implicitly means you have not examined it.

**Evaluate the cost of errors asymmetrically.** For each group, assess what the consequence is if a true nonrespondent is missed (false negative) vs. if a likely respondent is flagged for follow-up (false positive). Error costs are often asymmetric. Missing a nonrespondent from an already-undercounted group may have higher downstream consequences than missing one from an overcounted group.

**Document tradeoffs.** Dimension 8 of the evaluation rubric (Chapter 14) requires bias/fairness documentation. The SFV framework (Chapter 15) requires that fairness decisions persist across the pipeline. The minimum artifact is a model card that documents training data composition, subgroup performance, and the fairness criterion chosen.

**Build human review into high-stakes decisions.** Cases involving small subgroups, high-stakes classifications, or low model confidence should require human review. Confidence-based routing -- cases below a threshold go to human coders -- is the operational implementation of bounded agency. Record override rates by demographic group; systematic overrides signal a model that is not working as intended.

---

## 10. Model card template

A model card is the documentation artifact that makes fairness tradeoffs explicit and auditable. The structure below follows Mitchell et al. (2019) with additions for the federal statistical context.

**Model identification**
- Model name and version
- Date of last validation
- Intended use (specific survey, specific task)
- Out-of-scope uses (what the model should not be used for)

**Training data**
- Source dataset and time period
- Known limitations (survivorship bias, underrepresentation)
- Which protected characteristics are present in the training data
- Demographic composition of training set

**Performance metrics (test set)**
- Overall accuracy, precision, recall
- Subgroup accuracy range across OMB SPD-15 categories
- Subgroup miss rate (FNR) range
- Fairness criterion optimized and at which threshold

**Fairness analysis**
- Which criteria are satisfied (or partially satisfied)
- Which criteria are violated and why
- The explicit governance choice: which criterion was prioritized and the justification

**Limitations and risks**
- Known failure modes
- Populations for which performance is weakest
- Recommended confidence thresholds for human review routing

**Governance**
- Documentation standard used
- Regulatory basis (OMB SPD-15, Executive Order provisions)
- Approval status and approving official
- Review schedule and retraining cadence

For a populated example using the Chapter 8 nonresponse model, run `examples/chapter-08/08_model_card.py`.

---

## 11. Exercises

### Exercise 8.1: Subgroup accuracy decomposition

The following table shows pre-computed subgroup results from the Chapter 8 logistic regression nonresponse model (see `examples/chapter-08/06_subgroup_decomposition.py`):

| Group | N | Base Rate | Accuracy | TPR | FNR (miss rate) |
|-------|---|-----------|----------|-----|-----------------|
| White non-Hispanic | 349 | 0.29 | 0.74 | 0.57 | 0.43 |
| Black non-Hispanic | 77 | 0.44 | 0.68 | 0.62 | 0.38 |
| Hispanic | 109 | 0.47 | 0.66 | 0.67 | 0.33 |
| Asian non-Hispanic | 35 | 0.20 | 0.77 | 0.43 | 0.57 |
| Other | 20 | 0.40 | 0.65 | 0.63 | 0.37 |

**Interpretation questions:**

1. Which group has the highest miss rate (FNR)? Which has the lowest accuracy?
2. Is the group with the highest miss rate also the group with the highest base rate of nonresponse? What does this pattern tell you about how errors compound?
3. The model has an overall accuracy of approximately 71%. How does the table change your interpretation of that number?
4. For a nonresponse model used in survey operations, which type of error is more costly: a false negative (missing a true nonrespondent) or a false positive (flagging a likely respondent)? Does your answer change depending on which group the error falls on?

For an extension exercise using age group decomposition, see `examples/chapter-08/09_exercises.py` (Exercise 8.1 section).

### Exercise 8.2: Fairness metric conflicts

The following table shows pre-computed fairness metrics from the same model:

| Group | Pred. Rate | TPR | FPR | Precision |
|-------|------------|-----|-----|-----------|
| White non-Hispanic | 0.28 | 0.57 | 0.17 | 0.59 |
| Black non-Hispanic | 0.39 | 0.62 | 0.27 | 0.71 |
| Hispanic | 0.43 | 0.67 | 0.30 | 0.73 |
| Asian non-Hispanic | 0.17 | 0.43 | 0.07 | 0.50 |
| Other | 0.35 | 0.63 | 0.25 | 0.72 |

**Questions:**

1. Does the model satisfy demographic parity? (Are positive prediction rates equal across groups?)
2. Does the model satisfy equalized odds? (Are TPR and FPR equal across groups?)
3. Which fairness criterion would you optimize for a nonresponse prediction model in a federal survey? Write two to three sentences justifying your choice in terms of the cost of false negatives vs. false positives for communities that are already undercounted.
4. The impossibility theorem says you cannot simultaneously satisfy all criteria. Given your answer to question 3, which criteria are you explicitly deprioritizing, and what would you write in the model card to document this choice?

### Exercise 8.3: Leadership briefing

**Scenario:** You are briefing your division chief on a proposed AI system that automates income imputation for ACS microdata. The vendor reports:

- Overall accuracy: 94%
- Validation dataset: 50,000 records from the prior year's survey

You have just completed a subgroup analysis showing:

- Accuracy for households with income in the lowest quintile: 78%
- Accuracy for households with income above median: 97%
- Accuracy for Hispanic households: 81%
- Accuracy for Black non-Hispanic households: 83%

**Questions to address in your briefing:**

1. What does "94% overall accuracy" conceal in this case?
2. Who bears the error burden? Is this distribution acceptable for a federal statistical system?
3. What additional information should you request from the vendor before a procurement decision? (Use the vendor fairness checklist from Section 7.)
4. What conditions would you require before approving this system for production use?
5. How does the impossibility theorem change your evaluation of the vendor's promise to "improve the model to eliminate disparities"?

---

## 12. Key takeaways

Statistical bias and algorithmic bias are related but distinct. A statistically unbiased estimator can still produce systematically worse outcomes for specific demographic groups. Evaluating a model requires checking both.

Bias enters ML pipelines at every stage: training data composition, feature selection, label quality, model choice, and evaluation metrics. Diagnosing bias requires examining each stage separately.

Four fairness metrics -- demographic parity, equalized odds, calibration, and predictive parity -- each measure different properties. No single metric captures "fairness" completely.

The impossibility theorem (Chouldechova 2017; Kleinberg et al. 2016): when base rates differ across groups, no model can simultaneously satisfy all fairness criteria. The choice of which criterion to optimize is a governance decision, not a technical one.

Subgroup accuracy decomposition is the minimum evaluation requirement for any model in federal statistical production. "Overall 94% accurate" is not a sufficient evaluation.

Federal-specific workflows -- nonresponse adjustment, imputation, automated coding, synthetic data -- each have known failure modes that require specific fairness audits. Methods from Chapters 1-7 can all produce biased outputs when base rates differ across groups.

Model cards and fairness documentation are governance requirements, not optional analysis. OMB Statistical Policy Directive 15 and Executive Order provisions on AI equity require documentation of subgroup performance before deployment. Every model encodes choices about which errors to minimize for which groups. Making those choices explicit is the only defensible position.

```{admonition} Connections to other chapters
:class: note
- *Chapter 1 (Regression/Classification):* The logistic regression model in this chapter is the same class of model introduced there. Accuracy metrics from Chapter 1 are now evaluated by subgroup.
- *Chapter 7 (Imputation):* Hot-deck donor pool stratification decisions should be documented as state that must persist through downstream steps.
- *Chapter 9 (Synthetic Data):* Synthetic generation may underrepresent tail distributions, compounding underrepresentation of small groups.
- *Chapter 10 (Differential Privacy):* Small populations pay higher privacy costs under differential privacy mechanisms.
- *Chapter 12 (LLMs):* Automated coding accuracy should be decomposed by linguistic characteristics of the response text.
- *Chapter 14 (Evaluation rubric):* Dimension 8 of the rubric requires bias/fairness documentation for any federal AI system.
- *Chapter 15 (SFV):* Fairness decisions are state. If the decision to use stratified donor pools is lost at a session boundary or stripped by compaction, downstream steps may reintroduce the bias the imputation step was designed to mitigate.
```
