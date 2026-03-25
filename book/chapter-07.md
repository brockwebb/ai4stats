# Chapter 7 - Imputation Methods for Survey Data

> Full runnable code for all examples is in `examples/chapter-07/`.

```{admonition} Who is this for?
If you have worked through Chapters 1-6 (Parts I and II basics), you are ready. This chapter covers a topic that every federal statistical program encounters: what to do when respondents leave questions blank. You will learn four imputation strategies, understand how to choose among them, and develop the vocabulary to evaluate imputation decisions made by others.
```

```{admonition} Why this matters for federal statistics
:class: tip
Missing data is not an exception in federal surveys. It is a constant. The American Community Survey has item nonresponse rates of 5-20% for income items. The Current Population Survey uses hot-deck imputation across its monthly sample of roughly 100,000 person records, with allocation rates ranging from under 1% for demographic items to around 10% for earnings variables. The National Health Interview Survey imputes health insurance status for a substantial fraction of respondents.

What you do about missing values determines whether your published estimates are trustworthy. Deleting incomplete records introduces bias. Filling in the mean underestimates variance. Hot-deck and regression imputation, done carefully, produce estimates that are defensible to Congress, OMB, and the public.

This chapter gives you the conceptual vocabulary to evaluate imputation decisions -- not just accept them.
```

## Learning goals

By the end of this chapter you should be able to:

- Distinguish the three missingness mechanisms and explain why the mechanism matters for method selection
- Identify the principal failure mode of each imputation method
- Apply the decision framework to select an appropriate method for a given survey situation
- Explain imputation decisions to a non-technical leadership audience
- Evaluate an imputed dataset using plausibility diagnostics

## 1. The missing data problem in surveys

Every survey produces missing values. A respondent skips an income question. An interviewer fails to record a field. A respondent says "I don't know." A record fails an edit check and the value is suppressed before publication.

The question is never *whether* data will be missing. It is *how* you handle it.

### 1.1 Three types of missingness

The statistical literature distinguishes three mechanisms (Rubin, 1976; Little & Rubin, 2019). Understanding which one you have changes what you can safely do.

**Missing Completely At Random (MCAR)**

The probability that a value is missing has nothing to do with the value itself or any other variable in the dataset. A questionnaire page is accidentally omitted from printing for 2% of forms, affecting respondents randomly.

*Consequence*: MCAR is the friendliest case. Complete-case analysis (dropping missing rows) gives unbiased estimates, though you lose sample size. MCAR is rare in practice and almost impossible to verify.

**Missing At Random (MAR)**

The probability of missingness depends on *observed* variables, but not on the missing value itself, once you condition on those observed variables. Higher-income respondents are more likely to skip the income question, but once you know their education level, the skip probability no longer depends on their actual income.

*Consequence*: MAR allows valid imputation using the observed variables. Most survey nonresponse is assumed MAR, though this assumption cannot be tested from the data alone. The MAR assumption is a modeling choice, not a fact you can verify.

**Missing Not At Random (MNAR)**

The probability of missingness depends on the missing value itself. Respondents with very high incomes are more likely to skip the income question *because* their income is high.

*Consequence*: MNAR is the hardest case. No imputation method can fully correct for it without external information (administrative records, follow-up surveys). Sensitivity analyses are essential when MNAR is plausible.

The figure below illustrates all three on the same synthetic dataset. With MCAR, missing records (red) are scattered randomly. With MAR, higher-education respondents (who earn more) are missing more -- the missing pattern is systematic but explainable from observed variables. With MNAR, missing records cluster at the top of the income range -- the missing value itself predicts whether it is missing.

```{code-block} python
# See examples/chapter-07/01_dataset_and_missingness.py for the full visualization.
# The key pattern: MCAR shows no systematic difference between missing and observed.
# MAR shows systematic differences that disappear after conditioning on education.
# MNAR shows differences that cannot be removed by conditioning on any observed variable.
```

**Observed-mean bias from each mechanism** (from the example dataset, n=300):

| Mechanism | Missing rate | Observed mean | True mean | Bias |
|---|---|---|---|---|
| MCAR | 20% | ~$48,100 | ~$48,500 | ~-$400 (negligible) |
| MAR | 25% | ~$46,300 | ~$48,500 | ~-$2,200 (upward-ed respondents missing) |
| MNAR | 22% | ~$41,900 | ~$48,500 | ~-$6,600 (high earners absent) |

The MCAR bias is negligible because the missing values are a random draw from the full distribution. MAR introduces moderate bias. MNAR introduces severe and unrecoverable bias.

### 1.2 Why you cannot just drop incomplete records

When the missingness mechanism is MAR, deleting incomplete records produces a biased sample. For income data where higher-education respondents skip more often, complete-case analysis over-represents lower-education respondents. Because education and income are strongly correlated, the resulting mean understates true average income.

In the ACS-like example dataset used throughout this chapter (800 records, ~14% income missing under MAR):

- True mean income: ~$59,400
- Complete-case mean (dropping missing): ~$57,100
- Bias from complete-case: ~$-2,300 (3.9% understatement)

This is not a small rounding error. For a survey that informs federal program eligibility thresholds, a 4% understatement of average income has real consequences. See `examples/chapter-07/01_dataset_and_missingness.py` for the full demonstration.

## 2. Simple imputation methods

### 2.1 Mean imputation

The simplest approach: replace every missing value with the overall mean of observed values.

```{code-block} python
# Mean imputation: one line in pandas
df["income_imp"] = df["income_obs"].fillna(df["income_obs"].mean())
```

Mean imputation has a fundamental flaw: every missing record gets the same value, regardless of who they are. The distribution of imputed values is a spike at the mean. The variance of the imputed dataset is lower than the true variance. Correlations between income and other variables are attenuated because the imputed records pull every relationship toward zero.

From the example dataset:
- Unconditional mean MAE: ~$19,400
- Variance ratio (imputed / true): ~0.87 (13% variance underestimate)
- Correlation between income and education after imputation: lower than observed-only correlation

Mean imputation is essentially never appropriate for published survey estimates. It may be useful as a quick exploratory placeholder, but its statistical properties are poor enough that it should be flagged clearly as such. See `examples/chapter-07/02_mean_imputation.py`.

### 2.2 Conditional mean imputation

Better: impute within subgroups defined by observed variables. If education predicts income, impute with the education-and-region-specific mean.

```{code-block} python
# Conditional mean: pandas groupby + fillna
df["income_cond_imp"] = (
    df.groupby(["educ", "region"])["income_obs"]
    .transform("mean")
)
df["income_cond_imp"] = df["income_cond_imp"].fillna(df["income_obs"].mean())
```

Conditioning reduces bias because the imputed value reflects the respondent's education and region -- key predictors of income. Compared to unconditional mean imputation:

| Method | MAE | Variance preserved | Key limitation |
|---|---|---|---|
| Unconditional mean | ~$19,400 | Poor | All imputed values identical |
| Conditional mean (educ x region) | ~$16,800 | Partial | Within-cell variance still zero |

The improvement is real. Two respondents in the same education-and-region cell who differ in age and hours worked still receive the same imputed value. Within-cell variance is zero. See `examples/chapter-07/03_conditional_mean.py`.

## 3. Hot-deck imputation

Hot-deck imputation is the standard at Census and most federal statistical agencies. The idea: for each record with a missing value, find a similar complete record (the *donor*) and copy its value.

The name comes from the era of physical punched cards: the "hot deck" was the stack of recently processed cards that served as donors.

### 3.1 The logic: imputation classes and donor selection

The key challenge is defining "similar." One principled approach: cluster respondents on their observed variables to define *imputation classes*, then within each class, draw a donor at random.

```{code-block} python
# Hot-deck: find donor from same imputation class
def hot_deck_impute(target_class, donor_pool, variable):
    donors = donor_pool[
        (donor_pool["imputation_class"] == target_class) &
        (donor_pool[variable].notna())
    ]
    return donors[variable].sample(1).iloc[0]   # draw one donor at random
```

KMeans clustering on standardized observed variables (age, education, hours worked, region) creates clusters where respondents are similar across all available dimensions. The donor is a real person from the same cluster -- their income is guaranteed to be a real, plausible value.

See `examples/chapter-07/04_hot_deck.py` for the full implementation including cluster construction, donor assignment, and traceability demonstration.

### 3.2 Why hot-deck is the Census standard

Hot-deck has several properties that make it attractive for federal statistics (see Andridge & Little, 2010, for a comprehensive review):

- *Imputed values are always plausible.* Because donors are real respondents, you never impute a negative income or an income that does not correspond to any real person.
- *The method is transparent and auditable.* You can trace each imputed value back to a specific donor record. When an IG review asks "where did this imputed value come from?", hot-deck provides a clean answer.
- *It preserves univariate distributions.* The distribution of imputed values mirrors the donor pool -- no artificial spike at the mean.
- *It handles mixed data types naturally.* You can hot-deck categorical variables (marital status, employment status) just as easily as continuous variables.

The main limitation: hot-deck does not use all available predictor information efficiently. Two records in the same cluster get the same donor pool, even if they differ on continuous variables within the cluster. Regression imputation addresses this.

## 4. Regression-based imputation

Regression imputation fits a prediction model on complete cases, then uses that model to predict missing values.

### 4.1 Deterministic regression imputation

Fit a linear regression on complete cases. Predict missing values using the fitted model. The prediction uses every predictor simultaneously, so age, education, region, hours worked, and full-time status all inform the imputed value.

The critical problem: all imputed values lie exactly on the regression plane. No individual variation remains. This understates variance for the same reason mean imputation does -- every missing person is assigned the "expected" income for someone with their characteristics, with no acknowledgment that real people differ from their expected values.

### 4.2 Stochastic regression imputation

Add a random draw from the empirical residual distribution to each imputed value. This restores the variance that deterministic imputation suppresses.

```{code-block} python
# Stochastic regression: predict + noise
residual_std = (y_observed - model.predict(X_observed)).std()
imputed_values = model.predict(X_missing) + np.random.normal(0, residual_std, n_missing)
```

The stochastic form has higher MAE than the deterministic form. This is expected and correct: income is genuinely variable around its predicted value, and a good imputation method should reflect that variability. Higher MAE here is not worse imputation -- it is more honest imputation.

From the example dataset:
- Deterministic regression MAE: ~$14,200 (appears accurate, but variance is wrong)
- Stochastic regression MAE: ~$16,800 (higher MAE, but variance is preserved)
- Deterministic variance ratio: ~0.91 (understates variance by 9%)
- Stochastic variance ratio: ~0.99 (near-perfect variance preservation)

See `examples/chapter-07/05_regression_imputation.py`.

## 5. Multiple imputation and Rubin's rules

Even stochastic regression imputation understates uncertainty. When you create one imputed dataset and analyze it, your standard errors are too small because they treat the imputed values as if they were observed.

*Multiple imputation* addresses this by creating *M* completed datasets (typically M=5 to 20), analyzing each one separately, and combining the results.

### 5.1 Rubin's combining rules

When you have M imputed datasets and want a single estimate of, say, mean income:

1. Compute the estimate in each of the M datasets separately.
2. Average the M estimates. That is your point estimate.
3. The standard error has two parts: within-imputation uncertainty (how uncertain the estimate is given the filled-in data) and between-imputation uncertainty (how much the estimate changes depending on which imputed values you got). Combine them using Rubin's formula.

```{code-block} python
Q_bar = np.mean([q for q, _ in results])          # pooled estimate
B = np.var([q for q, _ in results], ddof=1)        # between-imputation variance
W = np.mean([se**2 for _, se in results])          # within-imputation variance
T = W + (1 + 1/M) * B                              # total variance
SE = np.sqrt(T)                                     # pooled standard error
```

The between-imputation component *B* is what a single imputation misses. It captures the fact that you are uncertain about the imputed values themselves. A single imputed dataset ignores *B* and produces confidence intervals that are too narrow.

### 5.2 Multiple imputation for published estimates

**Why M=5 is usually sufficient.** Rubin (1987) showed that the relative efficiency of M imputations is approximately (1 + λ/M)^{-1}, where λ is the fraction of missing information. At M=5 and λ=0.3 (30% of variance attributable to missing data), efficiency is 94%. Increasing to M=20 raises efficiency to 98.5% -- a marginal gain that rarely justifies the computational cost.

**How to analyze.** Run your substantive analysis (regression, mean estimation, classification) on each of the M imputed datasets separately. Then combine results using Rubin's rules. Never pool the M imputed datasets into one large dataset and analyze once -- that ignores between-imputation variance and produces the same understatement of uncertainty as single imputation.

**Software.** SAS PROC MIANALYZE, R `mice` package, Python (manual Rubin's rules implementation as in `examples/chapter-07/06_multiple_imputation.py`). The `fancyimpute` library provides additional Python options.

**OMB requirement.** Statistical Policy Directive 1 requires agencies to document imputation procedures in published methodology reports. If you use multiple imputation for a published estimate, the methodology report must describe the imputation model, the value of M, and the combining rules used.

```{admonition} Rubin's rules in plain language
:class: note
Creating five imputed datasets is not five guesses at the same answer. It is five independent realizations of a plausible dataset, each consistent with what is known about the respondent. The spread across the five estimates tells you how much your conclusions depend on which imputed values you happened to draw. When that spread is large, your data has substantial missing-data uncertainty. Rubin's formula quantifies that uncertainty and incorporates it into your published standard errors.

Do not average the imputed variables across datasets before analysis. This collapses the between-imputation variability and defeats the purpose of multiple imputation.
```

## 6. ML-based imputation

Random Forest imputation treats imputation as a prediction problem: fit a random forest on complete cases, then predict missing values. The iterative version, missForest (Stekhoven & Bühlmann, 2012), alternates across variables until convergence.

### 6.1 When RF imputation helps

RF imputation outperforms regression when:
- There are complex nonlinear interactions between predictors and the target variable
- The sample is large enough for the forest to learn patterns reliably (generally n > 500 complete cases)
- Multiple variables need imputation simultaneously (iterative missForest alternates across variables)

From the example dataset, overall RF imputation reduces MAE modestly compared to stochastic regression. The advantage is most visible for part-time workers, where the relationship between hours worked and income is nonlinear and regression underperforms.

### 6.2 When RF imputation does not help

RF imputation is a poor choice when:
- *The sample is small.* Random forests overfit on small samples; linear regression is more stable.
- *Auditability is required.* "We used an ensemble of 200 decision trees with complex interactions" is harder to explain to OMB or an IG reviewer than "we predicted income from age, education, and region using a linear model."
- *Valid variance estimates are needed.* RF imputation without multiple imputation still understates uncertainty. Single-imputation RF is no better than single-imputation regression in this respect.
- *Regulatory constraints require documented formulas.* Some agencies require the imputation model to be reproducible from a documented specification. A random forest is not easily specified in a methodology report in the same way a regression equation is.

See `examples/chapter-07/07_rf_imputation.py` for the implementation and subgroup comparison.

## 7. Choosing an imputation method: a decision framework

The right imputation method depends on your specific situation. Work through these questions in order.

**1. How much data is missing?**

- Less than 5%: mean or conditional mean may be acceptable for exploratory work. For published estimates, hot-deck is still preferable.
- 5-20%: use hot-deck or stochastic regression. Document the method and validate distributions.
- More than 20%: requires careful modeling and sensitivity analysis. Consider whether the variable is publishable at all, or whether a missing-data caveat is warranted.

**2. What is the missingness mechanism?**

- MCAR: most methods work. Complete-case analysis is unbiased (though inefficient).
- MAR: you must condition on the predictors of missingness. A model that does not include the variables that drive missingness will produce biased imputations.
- MNAR: no standard imputation method produces unbiased estimates. Sensitivity analysis (pattern-mixture models, tipping point analysis) is required. State the limitation explicitly in the methodology report.

**3. What variables predict the missing values?**

- Strong predictors available (R² > 0.4 on complete cases) → regression or RF imputation. The model is doing real work.
- Weak predictors → hot-deck within strata. A regression model with poor fit is not better than a well-constructed hot-deck.

**4. Do you need valid variance estimates?**

- Yes (published estimates, confidence intervals, hypothesis tests) → multiple imputation with Rubin's combining rules. Single imputation always understates uncertainty.
- No (internal data quality check, exploratory analysis) → single imputation may be acceptable.

**5. Is auditability required?**

- Yes (OMB review, IG inquiry, congressional testimony) → hot-deck or regression. You can explain the logic and trace individual decisions.
- No → RF imputation is available if it materially improves accuracy.

**6. Are you imputing continuous or categorical variables?**

- Continuous: all methods apply.
- Categorical: hot-deck handles this naturally. Regression requires a different model form (logistic regression, multinomial). RF handles mixed types but with the same auditability limitations.

## 8. The explainability advantage of simpler methods

When a program director, OMB reviewer, or congressional staffer asks "how did you handle missing income data?", the answer matters as much as the method.

*Hot-deck explanation*: "We replaced the missing income value with the reported income of a similar respondent from the same age-and-education-and-region group. Every imputed value is a real income from a real respondent. We can identify exactly who donated each value."

*Regression imputation explanation*: "We predicted income from the respondent's age, education level, region, and hours worked using a regression model fit on complete cases. The model explains about 65% of income variation. We added random noise to preserve the natural spread in income."

*Random Forest imputation explanation*: "We used an ensemble of 200 decision trees with complex interactions among age, education, region, hours worked, and employment status to predict income. The method produces lower prediction errors than regression on part-time workers, where the income relationship is nonlinear."

The first two explanations are accessible to non-statisticians, documentable in a methodology report, and defensible to a skeptical reviewer. The third is technically accurate but harder to defend in a regulatory context.

When the performance difference between methods is small -- and on well-specified survey data, it often is -- the simpler method wins on governance grounds. This is not a limitation of the analyst. It is a feature of sound methodology. Unexplainable accuracy gains are not gains worth having.

## 9. Evaluating imputation quality

### 9.1 The diagnostic toolkit

**Density overlay by subgroup.** For each major demographic group (education, age band, race, region), plot the distribution of observed values and the distribution of imputed values. If the imputed distribution looks substantially different from the observed distribution in a subgroup, the imputation model is failing for that group. This is the most important visual diagnostic in survey imputation.

```{code-block} python
def plausibility_diagnostics(original, imputed, variable, group_col=None):
    """Check: mean shift, variance ratio, distribution overlap by group."""
```

The full implementation is in `examples/chapter-07/08_diagnostics.py`.

**Mean and variance check.** After imputation, the mean of the full imputed dataset should be close to the observed-only mean (adjusting for the MAR mechanism). The variance ratio (imputed dataset variance / true variance) should be close to 1.0. Ratios substantially below 1.0 indicate variance collapse.

**Correlation preservation.** Check that correlations between the imputed variable and key predictors are not attenuated. Mean imputation always attenuates correlations; regression and hot-deck should preserve them.

### 9.2 The reviewer's checklist

When someone presents imputed data for your review, work through these questions:

1. *What was the missing rate?* More than 30% on key variables is a caution flag. Very high missing rates on a variable may indicate the variable is not reliably measured.
2. *What method was used? Is it documented in the methodology report?*
3. *Were subgroup distributions checked?* Density overlays by demographic group -- are imputed values plausible for each group?
4. *Were variance estimates properly adjusted?* Rubin's combining rules for multiply-imputed data; naive standard errors from single imputation understate uncertainty.
5. *Was sensitivity to the MAR assumption tested?* Pattern-mixture models, tipping point analysis.
6. *Can individual imputed values be traced to donors?* Hot-deck: yes, by construction. RF: no.

### 9.3 Method comparison summary

| Method | MAE | Variance preserved | Auditability | Best when |
|---|---|---|---|---|
| Mean imputation | High | No | High | Never (variance collapse) |
| Conditional mean | Medium | Partial | High | <5% missing, weak predictors |
| Hot-deck | Medium | Yes | High | Standard survey practice |
| Stochastic regression | Low-Medium | Yes | Medium | MAR, strong predictors |
| Multiple imputation (M=5) | Low | Yes | Medium | Published estimates needing valid SEs |
| Random Forest | Lowest | Yes | Low | Large data, complex interactions |

## 10. Activity: imputation on hours_wk

Rather than implementing imputation from scratch, focus on evaluation and decision-making.

**Setup.** The `examples/chapter-07/09_exercise.py` script introduces 15% MAR missingness into `hours_wk` (hours worked per week), runs three imputation methods, and produces a results table and density overlay.

**Pre-computed results** (from the example dataset, n=800, ~15% missing):

| Method | MAE (hours/wk) | Var ratio |
|---|---|---|
| Conditional mean | ~3.8 | 0.74 |
| Hot-deck | ~4.1 | 0.96 |
| Stochastic regression | ~3.6 | 0.98 |

**Discussion questions (attempt before running the solution):**

1. Which method would you recommend for operational use? Justify your recommendation using the decision framework above. Consider: the missing rate, the mechanism (MAR on `fulltime`), the available predictors, and whether published standard errors are needed.

2. A colleague suggests mean imputation because it is simple and everyone understands it. Write a 3-sentence response explaining why this is problematic for a published survey estimate.

3. The density overlay for part-time workers shows RF imputation systematically overestimating `hours_wk`. What does this suggest about the training data for that subgroup? What would you investigate?

4. The stochastic regression has the lowest MAE but the hot-deck has a better variance ratio. Under what circumstances would you choose hot-deck over stochastic regression despite the lower accuracy?

**Optional:** Run `python examples/chapter-07/09_exercise.py` to reproduce the results and compare your reasoning to the code.

## 11. Key takeaways for survey methodology

- *Missing data is not random unless you verify it.* MCAR allows complete-case analysis without bias. MAR allows imputation using observed predictors. MNAR cannot be fully corrected without external data. Understand the mechanism before choosing an approach.

- *Mean imputation is almost always wrong.* It underestimates variance, attenuates correlations, and produces standard errors that are too small. Flag any analysis that uses mean imputation for a published estimate.

- *Hot-deck is the federal standard for a reason.* Imputed values are always real observed values. The method is auditable, produces plausible distributions, and handles categorical variables naturally. Its limitation is efficiency within each imputation class.

- *Stochastic regression imputation is more efficient than hot-deck* when the relationship between predictors and the target is well-modeled by regression. The stochastic noise component is not optional -- it is necessary to preserve the variance of the imputed variable.

- *Multiple imputation is the correct approach when you care about uncertainty.* Analyses of a singly-imputed dataset produce confidence intervals that are too narrow. For published estimates, use multiply-imputed data and Rubin's combining rules.

- *ML imputation is powerful but demands transparency.* Random Forest imputation outperforms regression when relationships are nonlinear. But it is harder to explain, harder to reproduce, and does not naturally provide variance estimates. When performance differences are small, choose the method you can defend.

- *Always compare imputed distributions to observed distributions.* Density overlays within subgroups are the standard diagnostic. An imputed distribution that looks nothing like the observed distribution in a subgroup is a warning sign that the model is failing for that group.

```{admonition} How to explain these methods to leadership
:class: dropdown

**On why imputation is necessary:**
"When a survey respondent leaves a question blank, we have two choices: delete the entire record or fill in a plausible value. Deleting records is not neutral -- it systematically removes certain types of respondents and biases our estimates. Imputation lets us keep those records by filling in values that are consistent with what similar respondents reported."

**On hot-deck imputation:**
"Our income question is missing for about 12% of respondents. We fill in each missing value by finding a respondent who looks similar -- same education level, same region, similar age and hours worked -- and copying their reported income. The value we use is always a real income from a real respondent. We can trace every imputed value back to its source."

**On multiple imputation and uncertainty:**
"When we impute a missing value, we are making a best guess. A single best guess understates our uncertainty. Multiple imputation acknowledges this by creating five versions of the filled-in dataset, each with slightly different imputed values. Our published standard errors reflect both the sampling uncertainty and the uncertainty about the imputed values. This is the approach recommended by OMB Statistical Policy Directive 1."

**On ML-based imputation:**
"Random Forest imputation can produce more accurate predictions than regression-based methods, especially when the relationship between income and its predictors is complex and nonlinear. We are evaluating it for select variables where we have evidence the linear model is misspecified. For any method we adopt operationally, we document the model specification, validation results, and comparison to the current method."

**On when to be skeptical of imputed data:**
"No imputation method can recover information that was never collected. If a variable has 40% missing and the reason for missingness is related to the variable itself (MNAR), the imputed estimates will be biased regardless of the method. For variables with high nonresponse rates or likely MNAR mechanisms, imputed estimates should be published with appropriate caveats and subject to sensitivity analysis."
```
