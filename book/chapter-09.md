# Chapter 9 - Synthetic Data Generation for Federal Statistics

```{admonition} Who is this for?
If you have worked through Chapters 1 and 8, you are ready. Chapter 1 introduced regression modeling. Chapter 7 showed how imputation uses fitted models to fill in missing values. Synthetic data generation extends both ideas: you fit a model to confidential microdata, then generate entirely new records from that model rather than filling in gaps. The concepts are closely related, but the stakes are different. This chapter is self-contained for readers who want to evaluate a synthetic data product without running the code.
```

```{admonition} Why this matters for federal statistics
:class: tip
Every federal statistical agency that releases microdata faces the same fundamental tension: the public has a right to access information they paid to collect, but the individuals who provided that information have a right to privacy. Traditional disclosure avoidance methods (cell suppression, data swapping, top-coding, noise infusion) reduce this tension by distorting the real data. Synthetic data offers a different answer: do not release the real data at all. Release statistically faithful imitations.

Synthetic microdata is not a future research direction. The Census Bureau has produced synthetic data products for over two decades. OnTheMap (LEHD) is synthetic. The Survey of Income and Program Participation Synthetic Beta (Abowd, Benedetto & Stinson, 2006) was one of the first federal synthetic data products. The 2020 Census Disclosure Avoidance System introduced differential privacy at scale, generating significant methodological debate. Every senior statistician in the federal system needs a working understanding of what synthetic data is, how it is evaluated, and what its limitations are.
```

## Learning goals

After working through this chapter, you should be able to:

- Explain what synthetic data is and why agencies use it instead of traditional disclosure avoidance
- Distinguish fully synthetic from partially synthetic data products
- Describe how sequential regression synthesis works at a conceptual level
- Apply a standard evaluation checklist to a synthetic data product
- Interpret utility metrics (marginal, bivariate, pMSE, regression)
- Explain the privacy-utility tradeoff and why it cannot be fully resolved
- Describe the Laplace mechanism and what epsilon means in practice
- Recognize when a synthetic dataset is not adequate for a given analysis
- Know when to request FSRDC access instead of relying on synthetic data

## 1. The disclosure avoidance problem

Statistical agencies collect detailed information about individuals: income, race, health conditions, business revenues, employment status. Publishing this information at the record level (as public-use microdata files) creates direct disclosure risk. Even when names are removed, combinations of variables can uniquely identify individuals.

### 1.1 Traditional approaches and their limitations

Before synthetic data, agencies used several techniques to protect released microdata. Each has known limitations.

**Top-coding** caps sensitive values at a threshold (for example, all incomes above $150,000 are reported as "$150,000+"). This removes the right tail of high-income distributions and systematically biases estimates for high earners.

**Noise infusion** adds random errors to record values. An analyst working with a noise-infused file cannot know how much their specific estimate was affected, and the errors propagate into regression coefficients in ways that are difficult to correct.

**Data swapping** exchanges values between records with similar characteristics. It breaks the link between swapped values and other variables, destroying correlations that may be exactly what an analyst needs.

The table below shows illustrative results from applying each method to a simulated income variable with mean $62,000.

| Method | Mean income | Bias | Std deviation |
|---|---|---|---|
| Confidential (true) | $62,000 | -- | $31,400 |
| Top-coded (>$150K) | $58,900 | -$3,100 | $28,200 |
| Noise (+/- 5%) | $62,100 | +$100 | $32,800 |
| Swapped (10%) | $62,000 | ~$0 | $31,400 |

The mean bias from top-coding is modest, but the distributional distortion is severe: the entire upper tail is truncated. Noise infusion inflates variance. Swapping leaves the mean intact but destroys income-demographic correlations for the swapped records. Analysts working with these files cannot know how much the methods affected their specific analysis.

Synthetic data offers an alternative: instead of distorting the real records, you generate entirely new records that were never associated with any real person.

See `examples/chapter-09/01_confidential_dataset.py` for a working demonstration with visualizations.

## 2. Fully synthetic vs. partially synthetic

The two main architectures for synthetic microdata differ in how much of the released dataset is generated.

**Fully synthetic data** replaces every record with a new, generated record. No released record corresponds to any real person. Privacy protection is stronger because there is no direct link between released records and confidential records. Preserving utility is harder because the synthesizer must capture the entire joint distribution, including rare combinations. OnTheMap (LEHD Origin-Destination Employment Statistics) is a fully synthetic federal product.

**Partially synthetic data** preserves the real records but replaces only sensitive variables with synthesized values. The released record still corresponds to a real respondent; only the sensitive columns are replaced. This is easier to calibrate for utility (most variables are untouched), but the privacy guarantee is weaker because an adversary can link released records back to real people through the unsynthesized variables. The SIPP Synthetic Beta replaced income and program participation variables while preserving demographic structure.

The choice between them is a policy decision as much as a technical one. Both approaches require formal disclosure risk assessment before release.

## 3. How synthetic data is generated: the sequential approach

The most widely used synthesis method in federal statistics is sequential regression synthesis (Raghunathan, Reiter & Rubin, 2003), also called parametric sequential synthesis. The idea: model each variable conditionally on previously synthesized variables, then sample new values from those models. No confidential record values are copied directly.

### 3.1 The synthesis algorithm

```{code-block} text
Sequential regression synthesis algorithm:

For each variable v in synthesis order:
  1. Fit model: v ~ previously_synthesized_variables  (on confidential data)
  2. Predict: v_hat for synthetic records using their synthesized predictors
  3. Add noise: sample from residual distribution to preserve variability
  4. Store: synthetic v values become predictors for the next variable

The confidential data is used only for model fitting, never copied.
```

The synthesis order matters. Variables earlier in the chain become predictors for later variables; variables later in the chain depend on what comes before them. A variable that is omitted entirely from the synthesis cannot have its correlations preserved.

For the examples in this chapter, the synthesis order is:

1. **age** -- sampled from a fitted normal distribution (no predictors)
2. **educ** | age -- multinomial logistic regression
3. **region** | age, educ -- multinomial logistic regression
4. **income** | age, educ, region -- linear regression on log scale plus residual noise
5. **married** | age, educ, income -- logistic regression

See `examples/chapter-09/02_sequential_synthesis.py` for the full implementation.

### 3.2 Why log-scale for income?

Income distributions are right-skewed. Fitting a linear model directly on income concentrates predictive power on the high-income tail, and the residuals violate normality assumptions. Modeling log(income) and exponentiating the prediction produces a more realistic synthesis that preserves the skew without extreme outliers. This is a practical choice with direct effects on analytic validity; it is worth documenting in synthesis methodology reports.

## 4. Evaluating synthetic data utility

Generating synthetic data is the easy part. Evaluating whether it is useful is the hard part. A synthetic dataset that passes a visual check may still fail for specific analytic purposes. Evaluation should proceed in layers: marginal, bivariate, and analytic.

### 4.1 Marginal utility

Check whether the univariate distributions match. The table below shows summary statistics for a 600-record synthetic dataset generated by sequential synthesis.

| Variable | Conf. Mean | Synth. Mean | Conf. Std | Synth. Std |
|---|---|---|---|---|
| age | 42.0 | 42.1 | 13.8 | 13.9 |
| educ | 14.1 | 14.0 | 2.7 | 2.6 |
| income | $62,000 | $61,400 | $31,400 | $30,800 |
| married | 0.52 | 0.51 | 0.50 | 0.50 |

Marginal agreement is necessary but not sufficient. A synthesizer that draws each variable independently from its marginal distribution would match this table perfectly while destroying every correlation.

### 4.2 Bivariate utility: correlation preservation

The correlation matrix comparison is the standard bivariate utility check. The difference panel (synthetic minus confidential) shows which pairwise relationships were preserved.

| Pair | Confidential | Synthetic | Difference |
|---|---|---|---|
| age vs educ | 0.08 | 0.07 | -0.01 |
| age vs income | 0.48 | 0.46 | -0.02 |
| age vs married | 0.21 | 0.19 | -0.02 |
| educ vs income | 0.52 | 0.50 | -0.02 |
| educ vs married | 0.14 | 0.13 | -0.01 |
| income vs married | 0.31 | 0.28 | -0.03 |

Differences below 0.05 are generally acceptable. The synthesis above was designed to preserve all five relationships; they are all well-recovered. If married had been omitted from the model, the income-married difference would approach the full confidential correlation (0.31), not 0.03.

See `examples/chapter-09/04_utility_bivariate.py` for the full correlation heatmap.

### 4.3 Analytic validity: the regression test

The most important utility test for a specific analysis: if you run your regression on the synthetic data, do you recover approximately the same coefficients as on the confidential data?

The table below shows coefficient recovery for the model income ~ age + educ + region.

| Parameter | Confidential | Synthetic | % Difference | Status |
|---|---|---|---|---|
| Intercept | $8,200 | $8,450 | 3.0% | Good |
| Age ($/yr) | $1,240 | $1,210 | 2.4% | Good |
| Education ($/yr) | $4,800 | $4,730 | 1.5% | Good |
| Region | $1,100 | $1,070 | 2.7% | Good |
| R-squared | 0.423 | 0.408 | -- | -- |

These coefficients are well-recovered because income was modeled as a function of age, educ, and region in the synthesis. If you ran a different regression -- say, income on health status, which was not synthesized -- you would not expect valid results.

See `examples/chapter-09/05_utility_regression.py` for the full comparison.

### 4.4 The pMSE global utility metric

A formal global utility metric (Snoke et al., 2018): train a classifier to distinguish confidential records from synthetic records. If it cannot do better than random guessing, the synthetic data is statistically indistinguishable along the dimensions the classifier can detect.

```{code-block} text
pMSE = mean( (P(record is synthetic) - 0.5)^2 )

Range: 0.000 (ideal -- classifier guesses randomly)
       0.250 (worst case -- classifier perfectly separates datasets)
```

For the sequential synthesis described above, the pMSE is approximately 0.003 -- well below the threshold that would suggest concern. A pMSE above 0.020 warrants investigation into which variables drive the distinguishability.

See `examples/chapter-09/06_utility_pmse.py` for the full implementation.

## 5. The privacy-utility tradeoff

More faithful synthesis increases utility but also increases disclosure risk. This tension is fundamental and cannot be fully resolved; it can only be managed.

### 5.1 The KNN illustration

A KNN synthesizer makes the tradeoff concrete. With k=1, every synthetic income value is the nearest neighbor's income in the confidential data. With k=50, it is the average of 50 neighbors. The table below shows the tradeoff:

| k | pMSE (utility) | NNDR (privacy proxy) | Interpretation |
|---|---|---|---|
| 1 | 0.003 | 0.12 | High utility, high disclosure risk |
| 10 | 0.006 | 0.38 | Balanced tradeoff |
| 50 | 0.022 | 0.71 | More privacy, reduced utility |

The nearest-neighbor distance ratio (NNDR) measures how close synthetic records are to specific confidential records. Low NNDR means a synthetic record could be nearly identical to one real person's record -- a disclosure concern. High NNDR means synthetic records blend information from many real records -- safer but less precise.

The sequential regression approach used in this chapter sits somewhere between k=10 and k=50 in typical practice: it does not copy nearest neighbors directly, but it does memorize the marginal distributions of rare combinations.

See `examples/chapter-09/07_privacy_utility_tradeoff.py` for the full analysis.

### 5.2 Differential privacy: a formal approach

Differential privacy (DP; Dwork & Roth, 2014) provides mathematical bounds on privacy loss. The core guarantee: adding or removing any single person's record changes the probability of any output by at most a factor of e^epsilon.

```{code-block} text
Epsilon (privacy budget):
  epsilon = 0.1:  Very strong privacy. Heavy noise. Results less precise.
  epsilon = 1.0:  Moderate privacy. Reasonable accuracy for large counts.
  epsilon = 10.0: Weak privacy. Results close to true values.
```

The Laplace mechanism achieves differential privacy by adding noise drawn from Laplace(0, sensitivity/epsilon) to aggregate statistics. Sensitivity measures how much one person's data could change the statistic (for a count query, sensitivity = 1).

The figure produced by `examples/chapter-09/08_differential_privacy.py` shows the noise distributions for three epsilon values applied to a count of 50,000 individuals. At epsilon=0.1, the noisy count might be off by thousands. At epsilon=10.0, it is accurate to within tens.

The accuracy-epsilon tradeoff for a count statistic with sensitivity 1:

| Epsilon | Noise Scale | Mean Abs Error | 95th percentile Error |
|---|---|---|---|
| 0.10 | 10.0 | 10.0 | 23.0 |
| 0.50 | 2.0 | 2.0 | 4.6 |
| 1.00 | 1.0 | 1.0 | 2.3 |
| 5.00 | 0.2 | 0.2 | 0.5 |
| 10.00 | 0.1 | 0.1 | 0.2 |
| 17.14 | 0.06 | 0.06 | 0.1 |

## 6. The 2020 Census DAS debate: what happened and why it matters

The 2020 Census was the first large-scale production use of differential privacy in federal statistics. Understanding what happened -- and why it generated controversy -- is essential context for any senior statistician.

### 6.1 What the Bureau did

The Census Bureau replaced its traditional disclosure avoidance system with a new Disclosure Avoidance System (DAS) based on formal differential privacy (Abowd, 2018). Rather than swapping records or suppressing cells, the DAS:

1. Computed true population counts at all geographic levels
2. Added calibrated noise with a total privacy-loss budget of epsilon = 19.61 for the redistricting data product, comprising epsilon = 17.14 for the persons file and epsilon = 2.47 for the housing unit data (U.S. Census Bureau, 2021)
3. Used a post-processing algorithm (TopDown) to ensure consistency across geographic levels (state counts must sum to national totals, county counts must sum to state totals, etc.)

The result was a set of published tables in which block-level counts differed from true counts due to the noise injection.

### 6.2 What users objected to

The core complaint from states and localities was about block-level accuracy. Redistricting data -- which requires accurate population counts at the census block level -- showed anomalies: zero counts for blocks with real population, and nonzero counts for uninhabited blocks. Some states filed legal challenges arguing the data could not be used for redistricting as required by federal law.

A secondary complaint was about transparency. Traditional data swapping had known (if not always documented) effects that practitioners had developed intuition about over decades. The DAS introduced a new error model that was mathematically rigorous but unfamiliar, and the Bureau's initial documentation was difficult to parse.

### 6.3 The underlying governance tradeoff

The Bureau made defensible choices under real constraints:

- The 2010 census disclosure avoidance system had been shown to have serious vulnerabilities — database reconstruction attacks could recover most individual records from published tables (Abowd, 2018). The vulnerability was known; the DAS was a genuine response to it.
- Any disclosure avoidance method involves tradeoffs between accuracy and privacy. The difference with DP is that the tradeoff is explicit and auditable. Traditional methods obscured the tradeoff.
- The privacy budget allocation was a policy decision, and the Bureau was transparent about what it was. Users disagreed with the allocation, not the principle.

The lesson for federal statisticians is not that DP was wrong, but that formal methods require formal communication strategies. Users who have relied on a data product for decades need to understand not just what changed, but why the previous approach was untenable. Technical documentation alone is insufficient.

## 7. Sequential synthesis vs. generative AI

Generative AI methods (GANs, variational autoencoders, and fine-tuned language models) are increasingly proposed as alternatives to sequential synthesis for microdata.

### 7.1 What GANs and similar models offer

Generative adversarial networks and similar deep learning approaches can learn complex, nonlinear joint distributions that sequential regression synthesis approximates imperfectly. They can capture higher-order interactions without requiring the analyst to specify a synthesis order. For high-dimensional data with many variables, they can outperform parametric sequential synthesis on global utility metrics.

### 7.2 Why federal agencies move slowly

The federal context creates constraints that raw statistical performance does not resolve:

- **Auditability.** Sequential regression synthesis is fully auditable: every modeling decision is documented, every coefficient can be inspected, every synthesis step can be traced. A GAN is a neural network. Explaining to an OMB reviewer or a congressional oversight committee why a GAN produced a specific distribution is not tractable.
- **Mode collapse.** GANs are prone to mode collapse -- a failure mode in which the generator learns to produce a narrow range of outputs rather than the full distribution. Detecting mode collapse requires exactly the kind of utility testing described in this chapter.
- **Validation burden.** DP mechanisms have mathematical privacy guarantees. GANs do not. Membership inference attacks can sometimes recover information about training records from GAN outputs. Demonstrating safety requires empirical evaluation that is more uncertain than a formal guarantee.
- **Reproducibility.** A sequential synthesis model fit on the same data with the same seed produces the same synthetic dataset. GAN training is sensitive to initialization and architecture choices in ways that complicate reproducibility.

The practical rule for federal agencies: use sequential synthesis (or CART synthesis) when the synthesis methodology needs to be explainable, auditable, and defensible. Monitor research in generative AI synthesis, but adopt new methods through formal evaluation processes, not because the method is newer.

## 8. Evaluating a synthetic data product

When a colleague, vendor, or research partner presents you with a synthetic data product, apply this checklist before using it for analysis.

```{admonition} Evaluation checklist for synthetic data products
:class: tip

**Method and scope**
- What synthesis method was used? (Sequential regression? CART? GAN? DP mechanism?)
- What variables were included in the synthesis model? (Unmodeled variables lose correlations)
- What was the synthesis order? (Earlier variables become predictors for later ones)
- Was differential privacy applied? If so, what is the total epsilon and how was it allocated?

**Utility documentation**
- What utility metrics were reported? (Marginal only? Bivariate? Regression tests? pMSE?)
- What analyses was the synthesis explicitly validated for?
- Was utility tested for your specific analysis?
- What is the sample size? (Small synthetic datasets have high variance even if the synthesis is good)

**Privacy documentation**
- What disclosure risk assessment was performed?
- Were identity and attribute disclosure risks evaluated?
- What quasi-identifier combinations were tested?
- Was the assessment conducted by the agency or an independent reviewer?

**Access pathway**
- Is FSRDC access available for analyses the synthetic data cannot support?
- What is the application timeline for FSRDC access?
- Has the agency published a list of analyses the synthetic data is NOT valid for?
```

## 9. Evaluating disclosure risk

Utility without privacy is just releasing the confidential data. Agencies must balance both. Two types of disclosure risk are routinely assessed.

**Identity disclosure** occurs when a synthetic record can be linked to a specific real person through quasi-identifiers (age, education, region, etc.). In most synthesis methods, many real people share any given combination of demographic variables, so this rate is naturally low. The risk rises when the synthetic data preserves rare combinations of variables that uniquely identify specific individuals.

**Attribute disclosure** occurs when an adversary can infer the value of a sensitive variable (income, health status) from the quasi-identifiers of a known target person. The synthetic data can enable attribute disclosure even if no synthetic record directly corresponds to the target person, simply by preserving the statistical relationship between quasi-identifiers and the sensitive variable.

The table below shows illustrative identity disclosure rates for the example synthesis. Exact-match rates on categorical quasi-identifiers are inherently high because many real people share the same demographic profile -- this is expected and not alarming. The concern is narrow cells where a combination is rare enough to approach uniqueness.

| Quasi-identifiers | Synthetic records checked | Exact matches to a real record | Match rate |
|---|---|---|---|
| age + educ + region | 200 | 185 | 92.5% |
| age + educ + region + married | 200 | 178 | 89.0% |

These high match rates reflect demographic overlap in the population, not synthesis failure. Formal disclosure risk assessment compares these rates to what would be expected under a random population model, and evaluates whether the synthesis adds information an adversary did not already have.

See `examples/chapter-09/09_disclosure_risk.py` for a worked implementation.

## 10. When synthetic data is not enough

Synthetic data expands access to research-quality microdata, but it does not replace controlled access for all analyses.

### 10.1 Limitations of synthesis

Every synthesis preserves what was modeled. The following situations reliably exceed what sequential synthesis can provide:

- **Small subpopulations.** A synthesis trained on national data averages over rare demographic combinations. Estimates for a specific tribe, disability category, or occupation code may have little resemblance to the true confidential values.
- **Variables not synthesized.** If a sensitive variable was not included in the synthesis model, it cannot be analyzed from the synthetic data. This is not a failure of the data -- it is a design choice -- but it means analysts must check whether their variables of interest were synthesized.
- **Rare combinations.** Even if all variables were synthesized, rare combinations of values (income above $200,000, education beyond graduate degree, specific industry-occupation cell) may have been smoothed away by the synthesis model.
- **Publication-quality estimates.** For estimates that will appear in congressional testimony, regulatory filings, or peer-reviewed publications, the uncertainty introduced by synthesis may be unacceptable. Synthetic data is excellent for exploratory analysis; for production estimates, consider whether the confidential data is required.

### 10.2 The Federal Statistical Research Data Center network

The Federal Statistical Research Data Center (FSRDC) network provides secure, controlled access to confidential federal microdata for qualified researchers. The network includes over 30 sites at universities and federal facilities. Researchers apply for access, undergo background checks, and analyze data in secure computing environments. Output must be reviewed for disclosure risk before it leaves the enclave.

The practical pattern for federal statistical research:

1. Use synthetic data for exploration: build models, test code, identify the right variables, check sample sizes
2. Use synthetic data for preliminary analysis: understand the distribution of your outcome, tune analytical approaches
3. Apply for FSRDC access when you need publication-quality estimates on the full confidential data
4. Use the FSRDC for analyses involving rare subpopulations, unmodeled variables, or sensitive combinations

The Census Bureau's Virtual Data Enclave and similar mechanisms at BLS, NCHS, and other agencies provide related controlled-access pathways. The specific application process varies by agency and dataset; check the FSRDC website and your agency's microdata access office.

The right question is not "is synthetic data good enough?" but "what analyses can I do on synthetic data, and what requires the enclave?" Using synthetic data for exploration and the FSRDC for publication is not a failure -- it is the intended use case.

## 11. Limitations and honest assessment

Synthetic data has known, predictable failure modes. Users and analysts should understand them.

**Unmodeled relationships are not preserved.** A synthesis that does not include married in the model will produce synthetic data in which income and marital status are approximately independent -- even if the confidential data shows a strong positive correlation. This is not a bug. It is a design characteristic. The synthesis only knows what you told it.

The pre-computed results below show the correlation loss when married is synthesized from its marginal distribution rather than from a model conditioned on income.

| Correlation | Confidential | Synthesis WITHOUT married model | Synthesis WITH married model |
|---|---|---|---|
| income vs married | 0.31 | 0.02 | 0.28 |

The correlation drops from 0.31 to 0.02 when married is not modeled -- essentially destroyed. Adding a single logistic regression step (married ~ age + educ + income) recovers it to 0.28.

**Implication for data users:** Before analyzing any relationship in synthetic data, confirm that both variables were included in the synthesis model and that the specific relationship was in the utility validation documentation. If it was not, the synthetic data may give wrong answers for that analysis.

**Synthesis order creates asymmetric preservation.** Variables earlier in the synthesis chain are better preserved than variables later in the chain, because earlier variables are conditioned on more information. In the five-variable synthesis above, income predictions depend on age, educ, and region -- all of which are already synthesized -- so the income model has good predictors. The married model also has good predictors. If the synthesis order were reversed and income came last, conditioned on married, the income-married relationship would be preserved differently.

**Outliers and rare combinations may be smoothed away.** Parametric sequential synthesis fits models to the distribution of common cases. Records in sparse parts of the covariate space (high income, unusual education level, rare region) may be poorly approximated.

See `examples/chapter-09/10_limitations.py` for a working demonstration of correlation loss.

## 12. Activity: analyze the synthesis

```{admonition} Discussion questions
:class: tip

The pre-computed results in Section 11 show that the income-married correlation drops from 0.31 to 0.02 when married is synthesized independently of income.

**Question 1.** A user complains that the synthetic data shows no relationship between income and marital status. Using what you know about sequential synthesis, explain why. What would have to change in the synthesis design to fix this?

**Question 2.** The pMSE for the base synthesis (without the married model) is 0.003. After adding the married step, it rises to 0.005. Is this acceptable? What does the increase tell you? What additional checks would you run to confirm the synthesis is still adequate?

**Question 3.** An analyst needs to study the income-marital status relationship for a congressional report. The synthesis without the married model gives a correlation near zero. Should they use the synthetic data or request FSRDC access? Justify your answer, including what additional information you would want before making the recommendation.
```

```{dropdown} Optional coding exercise

Extend the synthesis to include married as a modeled variable and verify that the income-married correlation is restored.

The approach: after synthesizing income, fit a logistic regression of married on age, educ, and income (using confidential data). Apply it to synthetic records to generate synthetic married values. Re-run the correlation check and confirm the income-married correlation is near 0.28 rather than 0.02.

Starter code and full solution: `examples/chapter-09/11_exercise.py`
```

## 13. Key takeaways for survey methodology

- Synthetic data is an active tool, not a research curiosity. OnTheMap, the SIPP Synthetic Beta, and the 2020 Census Disclosure Avoidance System are real production systems. Knowing what they are and how they work is part of being a federal statistician.
- Synthesis does not equal disclosure elimination. A poorly designed synthesizer can produce records nearly identical to real individuals. Formal disclosure risk assessment is required before public release.
- Utility must be demonstrated, not assumed. Sequential synthesis preserves the relationships it explicitly models. Relationships not in the model are typically not preserved. Use the evaluation checklist in Section 8 before analyzing any synthetic data product.
- The privacy-utility tradeoff is real and irreducible. More noise means more privacy and less utility. Any agency that chooses a privacy budget is making a policy decision about which analyses they are willing to degrade.
- Differential privacy is the gold standard for formal guarantees. The Laplace and Gaussian mechanisms provide mathematical bounds on privacy loss. But small epsilon means large noise, and the 2020 Census experience showed that users notice and object when small-area accuracy suffers.
- Sequential regression synthesis is accessible and explainable. It requires only standard regression and classification tools. The mechanism is transparent: each variable is modeled conditionally on the previous ones. This is a major advantage over black-box generative models when you need to explain the method to oversight bodies.
- The FSRDC network exists for a reason. When synthetic data utility is insufficient for a given analysis, approved researchers can access the confidential data under disclosure agreements. Synthetic data expands access but does not replace controlled access for high-stakes analyses.
- Chapters 10 and 15 extend these ideas. Chapter 10 covers differential privacy in more depth. Chapter 15 introduces State Fidelity Validity, a framework for evaluating whether AI-assisted research pipelines preserve the inferential properties of the original data.

```{admonition} How to explain these methods to leadership
:class: dropdown

**On why synthetic data matters:**
"We want to give the public access to our microdata so researchers can study the questions Congress cares about. But we cannot release data that could identify individual respondents. That would violate Title 13 and destroy survey trust. Synthetic data lets us release statistically faithful imitations. Researchers get the patterns; individual respondents get privacy."

**On utility and limitations:**
"Synthetic data is not the real data. It preserves the statistical relationships we built the synthesis to preserve. If a researcher wants to study something the synthesis was not designed for, the synthetic data may give wrong answers. We publish documentation of what analyses are valid, and we maintain the Federal Statistical Research Data Center network for analyses that require the real data."

**On the 2020 Census controversy:**
"The 2020 Census used a new privacy system based on differential privacy. This provides formal mathematical guarantees that were not possible with older methods. The tradeoff is that it adds noise, and that noise is larger for small geographies. Some states and localities argued the noise made redistricting data unreliable. There is no method that provides perfect privacy and perfect accuracy simultaneously. The Bureau was transparent about the tradeoff; what they underestimated was how much help users would need to interpret a new error model."

**On GANs and AI-based synthesis:**
"You may hear about generative adversarial networks or large language models being used for synthetic data. These are more powerful than the sequential regression approach in some settings, but they are also harder to validate and explain. For regulatory and audit purposes, we prefer methods where we can trace the synthesis decisions and demonstrate the privacy-utility tradeoffs explicitly. We monitor research in this area but adopt new methods carefully."
```
