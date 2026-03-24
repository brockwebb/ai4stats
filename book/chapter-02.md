# Chapter 2 - Cross-Validation and Model Selection

> Evaluate models honestly, tune hyperparameters without peeking at the test set, and explain which features drive predictions.

> Full runnable code for all examples is in `examples/chapter-02/`.

```{admonition} Who is this for?
If you finished Chapter 1, you are ready. We reuse the same synthetic ACS-like dataset.
You should be comfortable with train/test splits and basic scikit-learn model fitting.
```

```{admonition} Why this matters for federal statistics
:class: tip
A single train/test split can be misleading: a lucky split inflates performance, an unlucky one
deflates it. Cross-validation gives you a more stable estimate of how a model will perform on
genuinely new data. In survey operations, this matters because models trained on one data
collection cycle are applied to the next. Overfit models fail quietly in production.

Equally important: if you cannot explain to a program manager *which* respondent
characteristics drive a nonresponse prediction, the model will not get used.
Feature importance tools turn a black box into a defensible methodology.
```

## Learning goals

- Explain why a single holdout split gives a noisy performance estimate.
- Distinguish KFold, StratifiedKFold, and GroupKFold and know when each applies.
- Interpret a GridSearchCV results table and identify the risk of choosing the minimum CV error.
- Read a feature importance chart and explain its limitations to a non-technical audience.
- Know what to say when leadership asks "which variables matter most?"

```{admonition} How to use this chapter
:class: tip
Read sections 1 through 7 at your own pace. Section 8 is a reviewer's checklist -- use it
when evaluating a model evaluation report someone else produced. Sections 9 and 10 are
interpretation exercises: pre-computed results are shown, and your job is to read them
critically and draw conclusions. All runnable code is in `examples/chapter-02/`.
```

---

## 1. Dataset and setup

This chapter reuses the synthetic ACS-like survey dataset from Chapter 1.
Using the same dataset across chapters lets you see how methods build on each other:
Chapter 1 fit a model and evaluated it once; this chapter evaluates it more rigorously.

The dataset contains 1,200 simulated survey respondents with demographic features, an income
target, and a binary response indicator. One addition in Chapter 2: a `household_id` column
that assigns every four consecutive records to the same household. This cluster structure
is used in Section 3.2 to demonstrate GroupKFold.

To regenerate the dataset:

```{code-block} python
# examples/chapter-02/01_dataset_setup.py
python examples/chapter-02/01_dataset_setup.py
# Output: data/synthetic_survey_ch02.csv
```

The dataset has these columns:

| Column | Type | Description |
|---|---|---|
| state | categorical | Five states, weighted toward Illinois |
| age | integer | 18-80, mean 42 |
| education_years | integer | 9, 12, 14, 16, or 18 years |
| hours_per_week | integer | 0-80, mean 38 |
| urban | binary | 1 if urban address, 0 if rural |
| contact_attempts | integer | 1-7 attempts before current cycle |
| prior_response | binary | 1 if responded in prior cycle |
| household_id | integer | 0-299, four records per household |
| income | integer | Annual income in dollars |
| responded | binary | 1 if responded this cycle (target) |

### 1.1 Feature distributions

The five classification features (age, education_years, urban, contact_attempts, prior_response)
span different scales and types. Age and education_years are continuous; urban and prior_response
are binary; contact_attempts is a count bounded at 7. Because scales differ, raw coefficient
magnitudes are not directly comparable across features -- this is why permutation importance
(Section 6) is the preferred summary for a non-technical audience.

### 1.2 Pairwise correlations

The features in this dataset are mostly independent by design. The correlation matrix below
shows no pairs above 0.10 in absolute value, which means feature importance scores here are
unlikely to be distorted by multicollinearity. In real survey data, correlated features are
common (education and income, age and prior response) and must be interpreted more carefully.

| | age | education_years | urban | contact_attempts | prior_response |
|---|---|---|---|---|---|
| age | 1.00 | 0.02 | -0.01 | 0.03 | -0.01 |
| education_years | 0.02 | 1.00 | 0.04 | -0.02 | 0.02 |
| urban | -0.01 | 0.04 | 1.00 | -0.03 | 0.01 |
| contact_attempts | 0.03 | -0.02 | -0.03 | 1.00 | -0.08 |
| prior_response | -0.01 | 0.02 | 0.01 | -0.08 | 1.00 |

```{admonition} What to look for
Highly correlated features carry redundant information. In this dataset,
`contact_attempts` and `prior_response` are both related to response propensity
but measure different things. Correlations near 1.0 or -1.0 between two features
suggest one may be dropped without losing much information. The modest -0.08
between these two here is not a concern.
```

---

## 2. Why a single split is not enough

When you split data into train and test once, the test set is a random sample of your data.
A different random split gives a different test set, and therefore a different performance number --
even when the model and the data are identical. This is not measurement error in the usual sense.
It is genuine sampling uncertainty.

To make this concrete: the same logistic regression model, fit 30 times with 30 different random
80/20 splits of the same 1,200 records, produces AUC values that vary by roughly 0.04 points from
the lowest to the highest draw. Whether you report 0.77 or 0.81 depends entirely on which 240
records happened to land in your test set.

```{code-block} python
# examples/chapter-02/02_split_variability.py shows this directly.
# Key output (illustrative -- actual values depend on your run):
# AUC across 30 seeds: mean=0.782  std=0.011
# Range: [0.757, 0.806]
```

Cross-validation collapses this variability by averaging over many splits. Instead of one number
from one draw, you get a distribution. The mean is a more stable estimate of true performance;
the standard deviation tells you how uncertain that estimate is.

```{admonition} Key insight
The range observed across 30 splits is not a flaw in the model. It is information: it tells you
how much performance you can expect to vary on genuinely new data just from sample composition.
If you report only one split, you may be reporting a lucky (or unlucky) draw -- and you have
no way to know which.
```

---

## 3. K-Fold cross-validation

Instead of one split, KFold divides the data into *k* equal parts (folds).
Each fold serves as the test set exactly once. The model trains on the remaining *k-1* folds.
The final score is the average across all *k* test folds, with the standard deviation indicating
how consistent performance was across folds.

```{code-block} python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

kf = KFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=500)
scores = cross_val_score(clf, X_clf, y_clf, cv=kf, scoring="roc_auc")

# scores is an array of 5 per-fold AUC values
# scores.mean() is the estimate you report
# scores.std() tells you how stable that estimate is
```

```{admonition} Choosing k
- k=5 is a common default: five rounds, each training on 80% of the data.
- k=10 is more stable but slower; use for small datasets.
- For very large datasets, k=3 may be sufficient.
- Leave-one-out (LOO) is an extreme case useful for tiny samples but expensive.
```

### 3.1 Stratified K-Fold for imbalanced outcomes

For classification tasks with imbalanced classes, standard KFold can concentrate all the rare
class into one fold, making that fold much harder and the others artificially easy. The fix is
`StratifiedKFold`, which preserves the class ratio in each fold.

In survey nonresponse prediction, the class is typically imbalanced: response rates of 70-80%
mean nonrespondents are the minority class. Stratification ensures every fold has approximately
the same fraction of nonrespondents, which makes the per-fold scores directly comparable.

Use `StratifiedKFold` as the default CV strategy for any binary survey outcome.

### 3.2 Group K-Fold: keeping household clusters together

In most household surveys, multiple people share the same address. If household members A and B
are split across different CV folds, the model sees characteristics of household A during training
and then evaluates partly on household A during testing -- the test fold is not genuinely new data.
This is information leakage, and it makes the model look better than it will perform in production.

`GroupKFold` prevents this by keeping all members of the same household in the same fold.
The `groups` argument takes the household identifier.

```{code-block} python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
scores_group = cross_val_score(
    clf, X_clf, y_clf,
    cv=gkf,
    groups=df["household_id"],   # keeps household members together
    scoring="roc_auc"
)
```

```{admonition} When to use GroupKFold
Use `GroupKFold` whenever your data has a clustering structure that could cause leakage:
- Household surveys: multiple people per address
- Establishment surveys: multiple employees per employer
- Area probability samples: multiple households per primary sampling unit (PSU)
- Longitudinal surveys: multiple waves per respondent

Failing to account for clustering can make a model look better than it really is.
In production, it will underperform because the groups in the test dataset are genuinely new.
```

### 3.3 When to use which CV strategy

| Data structure | Recommended strategy |
|---|---|
| Independent records, balanced classes | KFold |
| Independent records, imbalanced classes | StratifiedKFold |
| Clustered data (households, PSUs, employers) | GroupKFold |
| Time-ordered data (longitudinal panels, waves) | TimeSeriesSplit |

When in doubt between KFold and StratifiedKFold, use StratifiedKFold. For survey data with any
clustering structure, GroupKFold is almost always the right choice for a final reported estimate.

### 3.4 Compare CV strategies side by side

The pre-computed table below shows what these three strategies produce on the same classification
task. GroupKFold gives the lowest mean AUC and the widest standard deviation -- both are expected:

| Strategy | Mean AUC | Std AUC |
|---|---|---|
| KFold (5-fold) | 0.782 | 0.018 |
| StratifiedKFold (5-fold) | 0.779 | 0.014 |
| GroupKFold (5-fold) | 0.764 | 0.022 |

The 0.018 gap between KFold and GroupKFold is the premium the model was collecting from
within-household leakage. GroupKFold's 0.764 is the honest number to report to a program manager
asking "how well does this model generalize to genuinely new households?"

```{admonition} Train vs test score gap
When using `cross_validate` with `return_train_score=True`, compare the train and test means.
If train AUC is 0.91 and test AUC is 0.76, the model is memorizing training data rather than
learning generalizable patterns (overfitting). If both are 0.62, the model is underfitting --
it needs better features or a different model family.
```

---

## 4. Hyperparameter tuning with GridSearchCV

Hyperparameters are settings you choose before training: regularization strength, tree depth,
number of neighbors. They are not learned from data -- they are constraints you impose on the
learning process.

The central rule is simple: *the test set is used exactly once, at the very end.*
If you try multiple hyperparameter values and pick the one that maximizes test performance,
you have implicitly trained on the test set. The reported number is optimistic. This is data
leakage through the back door.

`GridSearchCV` automates the correct workflow: it performs all hyperparameter search entirely
within the training data, using cross-validation. The test set is never touched until the winning
model is selected and you are ready to report final performance.

The correct workflow:

1. Split into train (80%) and test (20%).
2. All tuning and model selection happens on the train set using CV.
3. Pick the best hyperparameter value.
4. Evaluate the winning model once on the test set.
5. Report that number. Never go back.

### 4.1 Manual alpha search (Ridge regression)

For Ridge regression, `alpha` controls regularization strength: larger alpha shrinks coefficients
more aggressively, which reduces overfitting but also reduces the model's ability to fit the data.
The optimal alpha balances these forces.

The table below shows CV mean absolute error (MAE) across a range of alpha values for income
prediction. The best alpha sits in a flat valley -- performance is similar for several nearby
values, which means the choice is not sensitive. When the curve has a sharp minimum, the choice
matters more and the uncertainty around it is larger.

| Alpha | CV MAE ($) |
|---|---|
| 10 | 18,420 |
| 50 | 18,380 |
| 100 | 18,310 |
| 200 | 18,290 |
| 500 | 18,350 |
| 1000 | 18,480 |

Alpha=200 wins by a small margin. Note that the difference between alpha=100 and alpha=500 is
less than $100 MAE on an income prediction task -- operationally negligible.

```{code-block} python
# examples/chapter-02/04_gridsearch_tuning.py
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    Ridge(),
    {"alpha": [10, 50, 100, 200, 500, 1000]},
    cv=kf,
    scoring="neg_mean_absolute_error",
    refit=True,
)
grid.fit(X_tr, y_tr)         # X_tr is training data only -- test set untouched
final_model = grid.best_estimator_
# Evaluate exactly once:
test_mae = mean_absolute_error(y_te, final_model.predict(X_te))
```

```{admonition} Key rule: never tune on the test set
The test set is used exactly once, at the very end, to estimate real-world performance.
If you tune hyperparameters by looking at test performance, you have implicitly used it
for training and the reported performance is optimistic. GridSearchCV enforces this discipline.
```

### 4.2 The risk of always choosing the minimum CV error

A common mistake is mechanically selecting the hyperparameter value with the lowest CV error.
Two things can go wrong:

*Overfitting to the folds.* With a small grid, the minimum may reflect noise rather than
signal. Choosing a value one step away from the minimum -- if it has nearly the same CV error
with smaller standard deviation -- is often more defensible.

*Ignoring interpretability.* A simpler model (larger alpha for Ridge; smaller C for logistic
regression) is more regularized and less sensitive to individual records. If the CV error
difference is small, the simpler model is easier to explain and more robust to distribution shift.

---

## 5. Scoring new records with the final model

After CV tuning, the final model can score new sampled units before fieldwork begins. The model
produces a probability, not a decision. Human supervisors review flagged records, confirm that
follow-up is appropriate, and approve resource allocation.

```{code-block} python
# Score a new batch of 20 sampled records
new_records["predicted_prob_respond"] = final_clf.predict_proba(new_records)[:, 1]
new_records["flag_for_follow_up"]     = (new_records["predicted_prob_respond"] < 0.5).astype(int)
```

```{admonition} Bounded agency reminder
The model produces a ranked list, not a decision. Human supervisors review flagged records,
confirm that follow-up is appropriate, and approve resource allocation. The model assists;
humans decide. This framing also satisfies most agency requirements for AI oversight.
```

---

## 6. Feature importance

Understanding which features drive predictions is as important as the prediction itself.
Two complementary methods exist: coefficient importance and permutation importance. Each answers
a slightly different question. For a non-technical audience, lead with permutation importance.

### 6.1 Feature importance: what it can and cannot tell you

*Coefficients* give direction (positive or negative association) and magnitude within the model.
For logistic regression, a positive coefficient means the feature increases the log-odds of the
outcome. The ranking by coefficient magnitude tells you what the model is relying on -- but only
if features are on comparable scales. A coefficient of 0.5 on a binary feature (0/1) and a
coefficient of 0.01 on age (18-80) are not directly comparable: the age coefficient applies
across a 62-point range.

*Permutation importance* measures actual contribution to performance. Shuffle one feature's values
at random, destroying any predictive signal it carries, and measure how much performance drops.
A large drop means the model depended on that feature. A near-zero drop means the feature was
not contributing much to predictions, regardless of what the coefficient says.

*What neither method can tell you:*

- Causation. A feature associated with nonresponse in this model may be a proxy for something
  not in the dataset. Prior response history predicts future response, but the mechanism may be
  an unmeasured attitudinal factor correlated with both.

- Performance on future data with different distributions. Importance rankings are relative to
  the current training data. If the population changes, the rankings may shift.

- Reliability when features are correlated. When two features carry overlapping information,
  their importance scores split between them. Permutation importance handles this better than
  coefficients, but both can mislead when correlations are strong.

### 6.2 Coefficient importance (logistic regression)

```{code-block} python
# examples/chapter-02/05_feature_importance.py
coef_df = pd.DataFrame({
    "feature":     FEATURES_CLF,
    "coefficient": final_clf.coef_[0],
    "abs_coef":    np.abs(final_clf.coef_[0]),
}).sort_values("abs_coef", ascending=False)
```

In the nonresponse model, `prior_response` has the largest absolute coefficient (negative:
prior respondents are more likely to respond again, so nonresponse propensity is reduced) and
`contact_attempts` is the second-largest (positive: records requiring many contacts are harder
to convert).

### 6.3 Permutation importance (model-agnostic)

The pre-computed permutation importance table below shows mean AUC decrease when each feature
is shuffled 20 times on the test set. The standard deviation across shuffle repetitions gives a
confidence interval on the importance estimate.

| Feature | Mean AUC Decrease | Std |
|---|---|---|
| prior_response | 0.142 | 0.008 |
| contact_attempts | 0.098 | 0.006 |
| urban | 0.021 | 0.004 |
| age | 0.018 | 0.005 |
| education_years | 0.009 | 0.003 |

Prior response history is by far the most important feature: removing it would cost 14 AUC
points. Education years has the smallest effect at 0.009 -- statistically distinguishable from
zero, but operationally marginal.

```{admonition} Coefficient vs permutation importance
Coefficients show the direction and magnitude of each feature's effect within the model.
Permutation importance shows each feature's actual contribution to model performance.
They often agree, but when correlated features are present, permutation importance is
more reliable because it accounts for how the model uses features together.

When presenting results to a program manager, lead with permutation importance.
It answers the question "if we didn't have this variable, how much worse would the predictions be?"
```

```{admonition} How to explain feature importance to leadership
:class: tip
"The model found that prior response history is the strongest predictor of whether someone
will respond in the current cycle. Contact attempts also matter: records that already required
many contacts are more likely to need follow-up again. Age and urban status have smaller effects."

"We verified this by temporarily removing each variable from the model and measuring how much
accuracy dropped. This gives us confidence that the important variables are genuinely informative,
not just correlated with something else."
```

---

## 7. Quick reference

```{admonition} Cross-validation patterns
- `KFold(n_splits=5, shuffle=True, random_state=42)` -- basic k-fold
- `StratifiedKFold(n_splits=5, ...)` -- preserves class balance in each fold (use for classification)
- `GroupKFold(n_splits=5)` -- keeps groups (households, PSUs) in the same fold
- `cross_val_score(model, X, y, cv=kf, scoring="roc_auc")` -- returns per-fold scores
- `cross_validate(model, X, y, cv=kf, scoring=[...], return_train_score=True)` -- multiple metrics
```

```{admonition} Hyperparameter tuning
- `GridSearchCV(model, param_grid, cv=kf, scoring=..., refit=True)`
- Access results: `grid.best_params_`, `grid.best_score_`, `grid.best_estimator_`
- Rule: tune on train only; evaluate on test once at the end
```

```{admonition} Feature importance
- Coefficients: `model.coef_[0]` for logistic; `model.coef_` for linear
- Permutation: `permutation_importance(model, X_test, y_test, scoring=..., n_repeats=20)`
- Prefer permutation importance for model-agnostic, robust estimates
```

---

## 8. Red flags in a model evaluation report

When reviewing a model evaluation report -- whether from a contractor, a colleague, or an
automated system -- look for these signals before accepting the reported performance numbers.

*Was cross-validation used, or just one train/test split?*
A single split produces a point estimate with unknown uncertainty. Ask: "what is the standard
deviation across folds?" If that number is not reported, the evaluation may be unreliable.

*Was the test set touched during tuning?*
If the report describes choosing a hyperparameter based on test performance, the test set was
used for training. Reported performance is optimistic. The correct workflow is GridSearchCV or
equivalent: all tuning inside training data only.

*Were groups and clusters respected in the CV?*
For household survey data, GroupKFold is the appropriate strategy. Standard KFold allows
within-household leakage and overstates performance. Ask: "how were household members handled
during cross-validation?"

*Is the train-test AUC gap suspiciously large?*
A gap larger than 0.05-0.10 AUC points typically signals overfitting. The model is memorizing
training records rather than learning generalizable patterns. It will likely underperform in
production as the population drifts.

*What was the class balance? Was StratifiedKFold used?*
If the response rate is not close to 50%, unstratified CV folds will have uneven class ratios.
This inflates variance across folds and makes per-fold metrics incomparable.

*Are confidence intervals reported, or just point estimates?*
A mean AUC of 0.782 is not very informative without knowing whether the standard deviation
is 0.003 (stable) or 0.025 (highly variable). Demand the distribution, not just the center.

---

## 9. In-class activities

The activities below use pre-computed outputs. Your job is to interpret the results and draw
conclusions -- the same skill you will need when reviewing a model evaluation report someone
else produced.

### 9.1 Interpreting CV strategy results

A colleague has run three CV strategies on the same nonresponse classification task and reported
the following:

| Strategy | Mean AUC | Std AUC |
|---|---|---|
| KFold (5-fold) | 0.782 | 0.018 |
| StratifiedKFold (5-fold) | 0.779 | 0.014 |
| GroupKFold (5-fold) | 0.764 | 0.022 |

*Question:* Which CV strategy gives the most conservative estimate? Why is that the one you
would report to leadership as the production performance estimate?

*Optional coding exercise:* Reproduce this table using `examples/chapter-02/03_cross_validation.py`.

```{dropdown} Answer
GroupKFold gives the most conservative estimate (0.764), and it is the right one to report for
household survey data.

Standard KFold allows people from the same household to appear in both training and test folds.
Because household members share characteristics (same address, similar demographics, potentially
similar attitudes), the model effectively sees "partial" test data during training. This leakage
inflates performance.

GroupKFold prevents this by keeping all four members of each household in the same fold.
The 0.018 gap between KFold (0.782) and GroupKFold (0.764) is the leakage premium. In
production, every new address will be genuinely unknown, so 0.764 is the honest estimate.

The wider standard deviation for GroupKFold (0.022 vs 0.014 for StratifiedKFold) is also
expected: because whole households move together, the folds are more heterogeneous, and
per-fold performance varies more. This is not a flaw -- it is an accurate reflection of
how much performance depends on which households end up in the test fold.
```

---

### 9.2 Interpreting a GridSearchCV results table

A GridSearchCV run on Ridge regression for income prediction produced these results:

| Alpha | CV MAE ($) |
|---|---|
| 10 | 18,420 |
| 50 | 18,380 |
| 100 | 18,310 |
| 200 | 18,290 |
| 500 | 18,350 |
| 1000 | 18,480 |

*Question:* What alpha would you choose and why? What is the risk of always choosing the alpha
with the minimum CV error?

*Optional coding exercise:* Reproduce this table using `examples/chapter-02/04_gridsearch_tuning.py`.

```{dropdown} Answer
Alpha=200 has the minimum CV MAE ($18,290). It is the technically correct choice.

However, note that alpha=100 produces nearly the same error ($18,310 -- a difference of only $20).
The difference is operationally negligible: both models predict income within $18,300 of the
true value on average. If interpretability or stability matters, either choice is defensible.

The risk of always choosing the minimum: CV MAE is estimated from data, so there is sampling
variance in the estimate itself. The true best alpha might be 100 or 500 -- you cannot know
from one run. A principled alternative is the "one-standard-error rule": choose the most
regularized model (largest alpha) whose CV error is within one standard error of the minimum.
This tends to produce simpler, more robust models.

Also note: the curve is flat between alpha=50 and alpha=500. The choice in this region has
little practical consequence. A flat tuning curve is actually reassuring -- it means the model
is not sensitive to your hyperparameter choice.
```

---

### 9.3 Interpreting a permutation importance table

The nonresponse model produces the following permutation importance results on the held-out
test set:

| Feature | Mean AUC Decrease | Std |
|---|---|---|
| prior_response | 0.142 | 0.008 |
| contact_attempts | 0.098 | 0.006 |
| urban | 0.021 | 0.004 |
| age | 0.018 | 0.005 |
| education_years | 0.009 | 0.003 |

*Question:* Which feature would you drop if you had to reduce the feature set? What is the
business case? What should you be cautious about?

*Optional coding exercise:* Reproduce this table using `examples/chapter-02/05_feature_importance.py`.

```{dropdown} Answer
Education_years is the weakest feature (mean AUC decrease = 0.009). If there were a cost
to collecting or validating this variable, it would be a candidate for removal. The model
would lose roughly 0.009 AUC points -- a small but real degradation.

Business case for dropping it: simpler models are easier to explain, audit, and maintain.
Fewer features also mean fewer data linkage operations if education is derived from a
separate source.

What to be cautious about: education_years may be important for equity reasons even if
it contributes little to overall AUC. A model that ignores education could behave differently
across education groups in ways that are not visible in the aggregate metric. Always check
subgroup performance before removing a demographic feature.

Also note the standard deviations. Education_years (std=0.003) is clearly distinguishable
from zero -- the feature does contribute. Age (0.018 +/- 0.005) and urban (0.021 +/- 0.004)
are both meaningful. The ordering of age and urban is close enough that their ranking could
reverse in a different sample.
```

---

### 9.4 GroupKFold vs KFold: interpreting the gap

A model comparison report shows the following side-by-side results:

| CV Strategy | Mean AUC | Std AUC |
|---|---|---|
| KFold (5-fold) | 0.782 | 0.018 |
| GroupKFold (5-fold, by household_id) | 0.764 | 0.022 |

*Question:* Why does GroupKFold give a lower and more variable estimate? What does the
0.018 gap tell you about the data? Which number would you include in a technical report?

*Optional coding exercise:* Reproduce this comparison using `examples/chapter-02/03_cross_validation.py`.

```{dropdown} Answer
GroupKFold is lower because it eliminates within-household information leakage.
Standard KFold allows person A and person B from the same household to appear in different folds.
They share characteristics (address, neighborhood, household income, possibly attitudes).
The model implicitly learns household-level signals during training and is partially "tested"
on data it has already seen in a different form. GroupKFold removes this advantage by ensuring
all household members land in the same fold.

The 0.018 gap (0.782 - 0.764) is the leakage premium: how much better the model appears to
perform when clustering is ignored. This gap is data-dependent -- a dataset with stronger
within-household similarity would show a larger gap.

The wider standard deviation (0.022 vs 0.018) is also expected. When whole households move
together, folds become more heterogeneous: one fold might have an unusual concentration of
urban multi-person households. This fold-to-fold variation is real and GroupKFold correctly
exposes it.

For a technical report: always report the GroupKFold number for household survey data.
You may include the KFold number for comparison if you want to quantify the clustering effect,
but the GroupKFold estimate is the one that corresponds to production deployment, where every
new household is genuinely unknown.
```

---

## Key takeaways for survey methodology

```{admonition} What you learned
1. A single train/test split produces a noisy performance estimate. Cross-validation averages over many splits for a more reliable number.
2. `StratifiedKFold` preserves the class balance in each fold and should always be used for imbalanced survey outcomes like nonresponse.
3. `GroupKFold` prevents data leakage from household or PSU clustering. It typically gives a more conservative (realistic) performance estimate than standard KFold.
4. `GridSearchCV` automates hyperparameter tuning entirely within the training set, keeping the test set clean for final evaluation.
5. Permutation importance is more reliable than coefficient magnitude for understanding which variables genuinely contribute to predictions.
6. Explainability is not optional in survey operations. A model that cannot be explained to a program manager will not be approved for production use.
```

```{admonition} How to explain these methods to leadership
:class: tip
"We evaluated the model using cross-validation rather than a single split. This gives a
more honest estimate of performance on future data, not just the particular sample we happened
to train on."

"We also verified that the model was not taking advantage of the fact that multiple people
from the same household appeared in both training and evaluation data. By keeping household
members together during evaluation, we confirmed the model generalizes to genuinely new addresses."

"The most important predictors of nonresponse are prior response history and number of contact
attempts. We confirmed this using permutation importance: temporarily removing each variable
from the model and measuring how much performance dropped. This tells us these variables are
genuinely informative, not just correlated with something else."

"The threshold we use to flag records for follow-up was chosen based on field budget constraints,
not just statistical performance. We can adjust it if operational priorities change."
```
