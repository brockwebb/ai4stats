# Chapter 3 - Decision Trees and Random Forests

> Interpretable models for classification and regression. You can print the rules and attach them to a methodology report.

> Full runnable code for all examples is in `examples/chapter-03/`.

```{admonition} Who is this for?
If you finished Chapters 1 and 2 (regression/classification and cross-validation), you are ready.
This chapter continues with the same synthetic survey dataset -- you will compare the models you built in Chapter 1 to tree-based models on the exact same task.
```

## Learning goals

- Explain how a decision tree makes splits using Gini impurity and entropy.
- Grow a decision tree on survey data and visualize the learned rules.
- Control tree depth to avoid overfitting.
- Train a Random Forest and explain why ensembles outperform single trees.
- Extract and plot feature importance from a Random Forest.
- Compare Random Forest to the logistic regression model from Chapter 1 on the same task.
- Articulate why interpretability matters in federal survey programs.
- Explain SHAP values and use them to justify individual predictions to non-technical reviewers.
- Assess the stability of feature importance rankings across repeated runs.
- Estimate computational costs and know when to subsample for large federal datasets.

```{admonition} Why this matters for federal statistics
:class: tip
Federal survey programs are accountable. Every modeling decision that affects data products, nonresponse follow-up targeting, or imputation flags can be challenged by program managers, OMB reviewers, or the public. Decision trees are popular in government statistics precisely because:

- The learned rules can be printed, read, and attached to a methodology report.
- Feature importance answers "which variables most predict nonresponse?" in plain language.
- Tree models handle mixed data types (age, education, binary flags) without preprocessing.
- Random Forests are robust to outliers and missing patterns without hand-tuning.

You must be able to explain *why* the model flagged a household or geography, and tree-based models let you do that.
```

---

## 1. Setup: the same dataset from Chapter 1

This chapter uses the identical synthetic survey dataset from Chapter 1 (`np.random.seed(42)`, `n=1200`). Using the same data is intentional: by the end of this chapter you will have three models (logistic regression, decision tree, Random Forest) all evaluated on the same test set, making direct comparison valid.

The dataset contains 1,200 synthetic survey respondents with five classification features: age, education years, urban indicator, contact attempts, and prior response history. The binary outcome is whether the respondent completed the survey. The data generation process encodes prior response history and contact attempts as the dominant predictors — matching what a real nonresponse analyst would expect to find.

See `examples/chapter-03/01_dataset_setup.py` for the full generation code and train/test splits.

---

## 2. What is a decision tree?

A decision tree makes predictions by asking a sequence of yes/no questions about the input features. Each internal node tests one feature against a threshold. Each path from root to leaf represents a decision rule. The leaf contains the prediction.

```
Is contact_attempts > 3?
 ├── Yes → Is prior_response = 0?
 │          ├── Yes → Predict: did NOT respond  (leaf)
 │          └── No  → Predict: responded        (leaf)
 └── No  → Predict: responded                   (leaf)
```

A federal survey analyst can read this, audit it, attach it to a methodology memo, and explain it to a program manager. That is why it matters.

### 2.1 How the tree chooses splits

At each node, the algorithm searches for the feature and threshold that best separates the two classes. For classification it minimizes **Gini impurity** (Breiman et al., 1984):

$$G = 1 - \sum_{k} p_k^2$$

where $p_k$ is the proportion of class $k$ in a node. A pure node (all one class) has $G = 0$. A maximally mixed node has $G = 0.5$. The algorithm picks the split that reduces Gini the most (weighted by node sizes).

Alternatively, **entropy** (Quinlan, 1986; $H = -\sum_k p_k \log_2 p_k$) measures the same thing in information-theoretic terms. Gini is slightly faster to compute; in practice the results are nearly identical.

For regression trees, the criterion is **mean squared error** of the target within each node.

### 2.2 A tiny manual example

To see Gini in action, consider a single candidate split: "Does `contact_attempts <= 2`?" We can compute the weighted Gini impurity of the two resulting groups by hand:

```{code-block} python
def gini(y):
    if len(y) == 0:
        return 0.0
    p = y.mean()
    return 1 - p**2 - (1-p)**2

mask_left = X_clf_train["contact_attempts"] <= 2
g_left  = gini(y_train[mask_left])
g_right = gini(y_train[~mask_left])
weighted = (mask_left.sum() / len(y_train)) * g_left \
         + (~mask_left).sum() / len(y_train)) * g_right
```

Running this on the training split produces output like:

```
Split: contact_attempts <= 2
  Left  node: n=493, Gini=0.4271, response rate=75.46%
  Right node: n=467, Gini=0.4643, response rate=63.38%
  Weighted Gini after split: 0.4449
  Parent Gini (before split): 0.4553
  Gini reduction: 0.0104
```

The algorithm repeats this search across every feature and every possible threshold, then picks the split with the largest Gini reduction. The full implementation is in `examples/chapter-03/02_decision_tree.py`.

---

## 3. Growing a decision tree on survey data

### 3.1 Fit and visualize

A depth-3 tree answers at most three questions before reaching a prediction. This produces a model shallow enough to print and read.

```{code-block} python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_clf_train, y_clf_train)

print(f"Tree depth:       {dt.get_depth()}")
print(f"Number of leaves: {dt.get_n_leaves()}")
print(f"Train accuracy:   {dt.score(X_clf_train, y_clf_train):.3f}")
print(f"Test accuracy:    {dt.score(X_clf_test, y_clf_test):.3f}")
```

### 3.2 The printable rules: the chapter's signature artifact

`sklearn.tree.export_text()` converts the fitted tree into a text representation that can be copied directly into a methodology report. This is the qualitative advantage of a decision tree over every other model type discussed in this book. A reviewer does not need to run any code to understand it.

```
Decision rules (depth=3 — nonresponse prediction):

|--- prior_response <= 0.50
|   |--- contact_attempts <= 2.50
|   |   |--- urban <= 0.50
|   |   |   |--- class: 0
|   |   |--- urban >  0.50
|   |   |   |--- class: 1
|   |--- contact_attempts >  2.50
|   |   |--- class: 0
|--- prior_response >  0.50
|   |--- age <= 45.50
|   |   |--- class: 1
|   |--- age >  45.50
|   |   |--- class: 1
```

Reading the tree left-to-right: the first split is on `prior_response`. Households that did not respond previously (`<= 0.50`) are further split by `contact_attempts` and then `urban`. Households with a prior response history go to the right branch, where age determines the final prediction. Every leaf has a clear provenance: you can explain exactly which combination of conditions led to a specific household being flagged.

```{admonition} Auditability
:class: tip
The text output above is the complete decision logic of the model. A reviewer does not need to run any code to understand it. This is a qualitative advantage over logistic regression coefficients (which require log-odds interpretation) and a large advantage over neural networks (which have no equivalent printout).
```

---

## 4. Controlling tree growth: the overfitting problem

A tree with no constraints will memorize the training data perfectly. Every leaf would contain exactly one record. That tree would fail on any new data because it learned noise, not patterns. This is **overfitting**.

The standard diagnostic is a depth curve: fit trees from depth 1 through 19 and plot train accuracy vs. test accuracy. The pattern is consistent: train accuracy climbs monotonically, while test accuracy peaks early — often at depth 3 or 4 — and then levels off or drops.

At depth 3 on this dataset, train accuracy is approximately 0.76 and test accuracy is approximately 0.74. By depth 15, train accuracy exceeds 0.95 while test accuracy has dropped back toward 0.72. The tree has memorized the training noise.

The key controls available in scikit-learn:

- `max_depth`: hard limit on depth. Start at 3–5 for interpretable trees.
- `min_samples_leaf`: no leaf can have fewer than this many training records. Prevents memorization of rare subgroups.
- `min_samples_split`: minimum records required to attempt a split.
- `max_features`: fraction of features to consider at each split (also used in Random Forests).

The full depth curve and `min_samples_leaf` comparison are in `examples/chapter-03/02_decision_tree.py`.

---

## 5. Random Forest: an ensemble of trees

A single tree is unstable: small changes in the training data produce very different trees. A **Random Forest** (Breiman, 2001) fixes this by training many trees on different bootstrap samples of the data and combining their predictions.

Two sources of randomness:

1. **Bootstrap sampling**: each tree sees a different random 63% of the training records (with replacement). The other 37% — the out-of-bag (OOB) sample — is used for internal validation without a separate hold-out split.
2. **Feature subsampling**: at each split, only a random subset of features is considered. This decorrelates the trees so their errors do not all go in the same direction.

The final prediction is the majority vote (classification) or mean (regression) across all trees. This averaging smooths out the variance of individual trees while keeping their low bias.

Running 200 trees on the same classification task from Chapter 1:

```
Random Forest (200 trees) — test set performance:
  OOB accuracy (train-time estimate):  0.768
  Test accuracy:                       0.771
  Test precision:                      0.784
  Test recall:                         0.893
  Test F1:                             0.835
  Test AUC-ROC:                        0.813
```

The OOB accuracy closely tracks the test accuracy, confirming that the forest is not overfitting and that no separate validation set was needed to monitor training. This is an especially useful property when labeled data is limited.

### 5.1 Effect of number of trees

Performance stabilizes after approximately 100 trees. Beyond 200 trees, gains in test AUC are negligible. This means there is little reason to use 500 or 1,000 trees in most federal applications — 200 trees is sufficient for tabular survey data at this scale. The `n_estimators` learning curve is produced in `examples/chapter-03/03_random_forest.py`.

### 5.2 Feature importance

Random Forests support two types of feature importance:

**Gini (mean decrease in impurity)** is computed during training. At each split, the model records how much Gini impurity decreases. Features that appear at many high-level splits accumulate large scores. This is fast but can over-rank features with many unique values or correlated features, because it is computed on the training data only.

**Permutation importance** (Breiman, 2001) shuffles each feature on the *test set* and measures how much performance (here, AUC-ROC) drops. A feature that the model truly relies on will cause a large drop when shuffled. A feature that the model learned as a proxy for something else may show a small drop even if Gini importance is high.

For federal reports, use permutation importance. It is defensible: you can explain exactly what the number means ("when we shuffled `prior_response`, AUC dropped by 0.08, the largest drop of any feature"). Gini importance should be considered a diagnostic tool for model development, not the final reported number.

The side-by-side comparison is in `examples/chapter-03/03_random_forest.py`.

```{admonition} Gini importance vs. permutation importance
:class: note
Use permutation importance for any number you put in a methodology report. Prior response history is consistently the strongest predictor of survey nonresponse (Groves & Couper, 1998). It is computed on the test set, it has a clear operational meaning (AUC drop when the feature is removed from the model), and it is not inflated by correlated features. Gini importance is a useful internal diagnostic during model development.
```

---

## SHAP: Explaining Individual Predictions

Gini importance and permutation importance are *global* measures -- they tell you which features matter on average across all predictions. For real policy decisions ("Why was tract T042 flagged for follow-up?"), you need *local*, per-prediction explanations. SHAP (SHapley Additive exPlanations; Lundberg & Lee, 2017) provides this.

### What SHAP is

Each SHAP value represents one feature's contribution to pushing a single prediction above or below the base rate. The values are grounded in cooperative game theory (Shapley, 1953), which gives them a theoretical guarantee that no other additive attribution method has: the contributions are fair, consistent, and sum exactly to the difference between the prediction and the base rate.

### Why it matters for policy

Gini and permutation importance answer "which features matter overall?" SHAP answers "why did the model make THIS prediction for THIS record?" The second question is what a program manager, OMB reviewer, or FOIA request actually asks.

A model that cannot answer "why this household?" is harder to defend in an IG audit or a congressional inquiry, even if its aggregate AUC is excellent. SHAP closes that gap for Random Forests.

### Three SHAP outputs explained

**Summary plot (beeswarm):** Every dot is one prediction. The x-axis shows the SHAP value -- how much that feature pushed the prediction toward nonresponse (positive) or toward response (negative). Color shows the feature value (red = high, blue = low). This gives you both importance and direction simultaneously. "High `contact_attempts` consistently pushes predictions toward nonresponse" is visible at a glance.

**Dependence plot:** Shows how one feature's SHAP value changes as the feature value changes. Nonlinear effects are visible here that coefficient plots cannot show. An interaction coloring (color by a second feature) reveals feature interactions -- for example, whether high `contact_attempts` matters more in urban or rural tracts.

**Waterfall/force plot for a single record:** The most operationally useful output. For a specific flagged tract:

```
SHAP waterfall for tract T042:
  Base rate (population average): 35% nonresponse probability
  prior_rr = 0.54        →  +8.2%  (low prior response pushes toward nonresponse)
  contact_attempts = 5   →  +6.1%  (high attempts pushes toward nonresponse)
  pct_renters = 48.3     →  +3.9%  (high renter share pushes toward nonresponse)
  pct_foreign_born = 12  →  +1.1%
  median_age = 38        →  -0.8%
  pop_density_log = 7.2  →  -0.4%
  pct_bachelors = 22     →  -0.3%
  ─────────────────────────────────
  Model prediction:          52.8% nonresponse probability  →  FLAGGED
```

This is the answer to "why is tract T042 flagged?" that a program manager, OMB reviewer, or FOIA request requires. No other standard feature importance method provides it at this level of specificity.

### SHAP vs. Gini vs. permutation importance

The three methods often agree on the top features but can diverge when features are correlated:

| Feature | Gini rank | Permutation rank | SHAP rank |
|---|---|---|---|
| prior_rr | 1 | 1 | 1 |
| contact_attempts | 2 | 2 | 2 |
| pct_renters | 3 | 3 | 3 |
| pct_foreign_born | 5 | 4 | 4 |
| pop_density_log | 4 | 5 | 5 |
| median_age | 6 | 6 | 6 |
| pct_bachelors | 7 | 7 | 7 |

When rankings agree across all three methods, you have strong evidence the finding is real. When they disagree, investigate the correlation structure -- two correlated features may split importance between them in ways that shuffle their relative ranks.

```{admonition} SHAP for audits
:class: tip
When preparing for OMB review or an IG audit of a model-assisted survey operation, include a SHAP waterfall for a representative flagged case in the methodology report. "Here is exactly why the model flagged this tract, in units the model actually used" is a more defensible exhibit than "here is the overall feature importance ranking."
```

Full SHAP analysis code is in `examples/chapter-03/04_shap_analysis.py`. Requires: `pip install shap`.

---

## Stability: Can You Trust the Importance Rankings?

Run permutation importance twice with different random seeds. The first and second most important features swap. Is `prior_response` really more important than `contact_attempts`, or did you get a lucky draw?

This is not an edge case. Permutation importance is stochastic -- it shuffles features randomly, and the resulting AUC drop is a noisy estimate. Feature pairs with similar true importance will have confidence intervals that overlap, meaning their rankings are genuinely uncertain.

### The solution: repeat and report intervals

Instead of running importance once and reporting a ranked list, run it 30 times with different seeds. Report the mean rank and the 95% confidence interval for each feature:

| Feature | Mean rank | Rank std | Top-3 frequency (30 runs) |
|---|---|---|---|
| prior_response | 1.2 | 0.4 | 30/30 |
| contact_attempts | 2.1 | 0.7 | 29/30 |
| urban | 3.8 | 1.0 | 21/30 |
| age | 4.3 | 1.1 | 8/30 |
| education_years | 4.6 | 0.9 | 2/30 |

The first two features have stable rankings -- their confidence intervals are narrow and they are top-3 in virtually every run. Urban is more uncertain: it is top-3 in 21 of 30 runs but occasionally drops to 4th. Age and education are genuinely lower-ranked; their intervals overlap each other substantially.

**The policy implication:** Reporting "`prior_response` is the strongest predictor" is well-supported. Reporting "age is more important than education" is not -- they are statistically indistinguishable given the noise in the estimate.

### Practical pattern for federal reports

Run 5-fold cross-validation with 5-10 different random seeds. Collect per-fold permutation importances (50 total measurements per feature). Report mean +/- one standard deviation. If two features' intervals overlap, do not claim a strict ranking between them.

This is the same principle that governs confidence intervals on survey estimates. The analysis is a sample-based estimate; report it as one.

### SHAP stability

SHAP values are deterministic for a fitted model (given the same input), but the model itself varies with training data. Running SHAP on 5 bootstrap samples of the training data shows whether the top-feature findings are stable or artifact-dependent. The full demonstration is in `examples/chapter-03/05_stability_analysis.py`.

---

## Computational Cost: Trees Don't Scale for Free

The examples in this chapter use 1,200 records. Federal datasets are millions of records. Understanding how Random Forest compute scales with data size, depth, and number of trees is essential before deploying these methods in production.

### How cost grows

At each split, the algorithm evaluates `max_features` candidate features times all possible split thresholds for those features. With depth $d$, each tree makes up to $2^d - 1$ splits. With $T$ trees and $N$ records, the total work is approximately:

$$\text{cost} \approx T \times N \times \text{max\_features} \times d$$

**Why `max_features="sqrt"` matters:** The default for classification is `sqrt(n_features)`. With 50 features, this means each split evaluates 7 features instead of 50 -- a 7x reduction in split computation. The decorrelation benefit (why ensembles work) is the statistical reason; the compute reduction is the practical reason.

### Observed scaling on this dataset (n=1,200)

| n_estimators | max_depth | Fit time (sec) |
|---|---|---|
| 50 | 5 | 0.08 |
| 100 | 10 | 0.22 |
| 200 | 15 | 0.61 |
| 200 | None | 0.84 |
| 500 | 15 | 1.52 |

At 3 million records (roughly 2,500x larger), extrapolated fit times at depth 15 with 200 trees would be approximately 25 minutes. At depth None (unconstrained), substantially longer.

### Subsampling strategy for large datasets

When fitting 200 trees on millions of records is too slow for iterative development:

1. **Subsample with stratification.** Preserve class balance and subgroup representation. `train_test_split` with `stratify=y` handles this.
2. **Verify importance rankings are stable at your subsample size.** Use the bootstrap stability analysis (Section above) on subsamples of 25K, 50K, 100K. When the top-3 features stop changing, you have sufficient data for the importance analysis.
3. **A common pattern:** Subsample to 50K-100K for exploration and hyperparameter search. Fit the final model on full data for the production deployment.
4. **Statistical power for importance:** The goal is not to use the minimum sample -- it is to use the sample size where importance rankings stabilize. That is a data-dependent question, not a fixed rule.

```{admonition} Federal-scale guidance
:class: note
For a dataset of 3 million records with 50 features: use `max_features="sqrt"`, `max_depth` between 10-15, `n_estimators=200`. Subsample to 100K for hyperparameter search. Fit final model on full data overnight. Total pipeline is feasible on a standard workstation; GPU is not required for Random Forests.
```

Full timing experiments and extrapolation code are in `examples/chapter-03/06_computational_scaling.py`.

---

## 6. Regression tree: predicting income

The same tree logic applies to continuous targets. The split criterion becomes MSE (mean squared error) within each leaf rather than Gini impurity. Everything else — depth control, bootstrap sampling, feature subsampling — is identical to the classification case.

A depth-4 regression tree and a 200-tree Random Forest regressor on the income prediction task:

```
Income prediction (regression) — test set:
  Decision Tree (depth=4):  MAE = $18,420,  R² = 0.312
  Random Forest (200 trees): MAE = $14,890,  R² = 0.487
  Random Forest OOB R²:      0.471
```

The gap is larger here than in the classification task. The regression tree is constrained to a small number of unique predicted values (one per leaf), which limits its ability to capture the continuous range of incomes. The Random Forest averages across many trees, producing smoother predictions and substantially lower MAE.

For income imputation tasks, the Random Forest regressor is the stronger choice on predictive grounds. If the methodology requires printable rules (for example, to explain which respondents were imputed using which model cell), a shallow regression tree attached alongside the Random Forest provides that documentation.

Parity plots and full code are in `examples/chapter-03/07_regression_trees.py`.

---

## 7. Comparison: Logistic Regression vs. Decision Tree vs. Random Forest

We can now compare all three classifiers from Chapters 1 through 3 on the *same test set*.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.754 | 0.771 | 0.881 | 0.822 | 0.782 |
| Decision Tree (depth 3) | 0.742 | 0.758 | 0.878 | 0.814 | 0.759 |
| Random Forest (200 trees) | 0.771 | 0.784 | 0.893 | 0.835 | 0.813 |

The three models perform similarly. This is the expected result for well-behaved tabular data where the true signal is reasonably linear. On this dataset, the choice between models is not primarily about accuracy — it is about what you need to document.

The Random Forest has the best AUC. Logistic regression has interpretable coefficients (log-odds). The decision tree has printable rules. All three perform within a margin that would not be operationally meaningful in most survey applications.

ROC curves, metric bar charts, and the full comparison code are in `examples/chapter-03/08_model_comparison.py`.

### When to choose a tree model for federal work

The right model depends on what the output needs to support, not just which one has the highest AUC:

- **Need printable rules for a methodology report?** Use a shallow decision tree (`max_depth=3` or `4`). The rule set goes in the methodology appendix. Individual records can be traced through the tree by hand.
- **Need best prediction accuracy on tabular data?** Use a Random Forest. It consistently outperforms single trees on held-out data through variance reduction.
- **Need both?** Use the Random Forest for prediction and a shallow decision tree for documentation. Train both. Report both AUC numbers. Include the decision tree rules as the methodology exhibit.
- **Need interpretable coefficients and log-odds?** Use logistic regression (Chapter 1). It is the right default when the relationship is approximately linear and you need effect sizes per feature.
- **Working with text or images?** Those are covered in later chapters (Part IV). Tree models are for tabular data.
- **Need per-case explanations for auditors or FOIA?** → Random Forest + SHAP. Provides individual-record waterfall plots showing exactly why the model made each prediction.

### The Random Forest black-box trade-off

A single decision tree produces a printed rule set. A Random Forest of 200 trees does not — you cannot print 200 trees and attach them to a methodology memo.

What you gain is better predictions. What you lose is the audit trail at the individual-rule level. Whether that trade-off is worth it depends on the use case:

**When the trade-off is worth it:** Resource allocation decisions with human review downstream. The model ranks tracts by risk. A field manager reviews the top-ranked tracts and decides which ones to visit. The model informs a human decision; it does not make a final determination. In this case, predictive accuracy matters more than individual-rule auditability, and the Random Forest is the right choice.

**When the trade-off is not worth it:** Automated decisions, decisions that must be explained record-by-record, situations where OMB or a congressional committee may ask "why was this specific household flagged," or cases where the model output will be challenged by respondents or auditors. In these cases, use a shallow decision tree and accept the modest accuracy trade-off. The audit trail is not optional.

A useful heuristic: if you need to explain the model to a lawyer or a respondent, use the decision tree. If you need to explain it to a program manager choosing between resource allocation strategies, either model works.

---

## 8. Auditing a tree-based model

When reviewing a tree-based model built by someone else — or preparing your own for peer review — these are the questions to ask:

- **What was `max_depth`?** Deep trees memorize noise. A depth-20 tree on 1,000 records has almost certainly overfit. Ask to see the train vs. test accuracy curve.
- **Was OOB or cross-validation used for validation, or just one split?** A single train/test split can be unlucky. OOB scoring or 5-fold CV is more reliable.
- **Were feature importances computed on the test set (permutation) or just from training (Gini)?** Gini importances can mislead. Ask for permutation importances.
- **Is the training data representative of the deployment population?** A model trained on 2020 data deployed in 2024 may encounter a shifted distribution. This is especially important for geographic or demographic subgroups.
- **Were subgroup performance checks done?** Check accuracy and recall by state, urban/rural, demographic group, or any subgroup that the program cares about. Aggregate AUC can hide poor performance on minority subgroups.
- **Was the threshold for "flag" vs. "don't flag" chosen deliberately, or defaulted to 0.5?** The default threshold of 0.5 is almost never the right choice for imbalanced outcomes. The threshold should be chosen based on the cost of false positives vs. false negatives in the specific program context.
- **Were importance rankings tested for stability across multiple runs?** Single-run importance rankings have substantial noise. Ask for mean +/- std across repeated runs.
- **Were individual predictions explained using SHAP or similar?** Aggregate importance is insufficient for case-specific audit questions. Ask for a SHAP waterfall on a representative flagged case.

---

## 9. End-to-end Random Forest workflow

This is the pattern to follow in practice.

```{admonition} End-to-end recipe
1. Establish your feature set and train/test split (same as Chapter 1).
2. Fit a baseline: single shallow tree — interpretable, gets you most of the way.
3. Fit a Random Forest with default settings and compare.
4. Tune with cross-validated search. Do NOT look at the test set during tuning.
5. Evaluate on the test set exactly once.
6. Report permutation importance as your feature importance table.
7. Attach the shallow tree's rules to the methodology documentation.
```

The hyperparameter search uses `RandomizedSearchCV` to explore combinations of `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features` using 5-fold cross-validation. With `n_iter=24`, this evaluates 24 randomly sampled configurations — 120 total model fits — and returns the best-performing set of hyperparameters without ever touching the test set.

Full search implementation is in `examples/chapter-03/09_hyperparameter_search.py`.

---

## 10. Activity: tract-level nonresponse targeting

You are advising a field operations team that wants to prioritize which census tracts to target with in-person nonresponse follow-up. Their budget allows them to visit only 25% of tracts. They want a model that identifies which tracts have the highest probability of not responding, and they want to understand which factors drive that prediction.

The dataset has 300 synthetic tracts with features: percent renters, median age, percent foreign born, percent with a bachelor's degree, log population density, prior response rate, and average contact attempts. The outcome is a binary indicator for low-response tracts.

**The setup** (same data used in the solution — run `examples/chapter-03/10_tract_exercise.py` to see all output):

```{code-block} python
np.random.seed(2025)
n_tracts = 300

tract_data = pd.DataFrame({
    "tract_id":         [f"T{str(i).zfill(3)}" for i in range(n_tracts)],
    "pct_renters":      np.random.normal(35, 15, n_tracts).clip(5, 90),
    "median_age":       np.random.normal(40, 8, n_tracts).clip(22, 70),
    "pct_foreign_born": np.random.normal(15, 10, n_tracts).clip(0, 60),
    "pct_bachelors":    np.random.normal(30, 12, n_tracts).clip(5, 75),
    "pop_density_log":  np.random.normal(6, 2, n_tracts).clip(1, 10),
    "prior_rr":         np.random.normal(0.72, 0.08, n_tracts).clip(0.40, 0.95),
    "contact_attempts": np.random.poisson(2.5, n_tracts).clip(1, 8),
})
```

**Your tasks:**

1. The decision tree rules for the tract model (depth 3) are printed below. Why is the top-ranked tract flagged? Walk through the tree step by step and identify which branch it follows.

```
|--- prior_rr <= 0.68
|   |--- contact_attempts <= 3.50
|   |   |--- pct_renters <= 42.10
|   |   |   |--- class: 0
|   |   |--- pct_renters >  42.10
|   |   |   |--- class: 1
|   |--- contact_attempts >  3.50
|   |   |--- class: 1
|--- prior_rr >  0.68
|   |--- pct_foreign_born <= 28.50
|   |   |--- class: 0
|   |--- pct_foreign_born >  28.50
|   |   |--- class: 1
```

Decision tree test AUC: 0.847. Random Forest test AUC: 0.891.

2. The Random Forest has higher AUC (0.891 vs. 0.847) but no printable rules. Write a 2-sentence recommendation for which model to deploy and why, given that the field team must justify the prioritization list to a program manager.

3. The permutation importance table for the Random Forest (tract model) is:

| Feature | Mean AUC drop | Std |
|---|---|---|
| prior_rr | 0.1823 | 0.0241 |
| contact_attempts | 0.0614 | 0.0183 |
| pct_renters | 0.0312 | 0.0154 |
| pct_foreign_born | 0.0198 | 0.0121 |
| pop_density_log | 0.0041 | 0.0089 |
| median_age | 0.0018 | 0.0076 |
| pct_bachelors | -0.0007 | 0.0065 |

Which features would you drop from a simplified model? What is the budget case for using fewer features?

3a. The stability analysis for the tract model shows that `prior_rr` is top-3 in 30/30 bootstrap runs, `contact_attempts` is top-3 in 27/30, and `pct_renters` is top-3 in 19/30. How would you report the feature importance ranking to leadership? Would you present a strict ordered list, or report it differently?

4. Given a budget for visiting 25% of tracts, which model's ranking would you trust more for making that prioritization decision -- the decision tree or the Random Forest? What is the operational risk of each choice?

5. *Optional:* Run `examples/chapter-03/10_tract_exercise.py` to reproduce all outputs and verify your answers.

---

## Key takeaways for survey methodology

- **Interpretability is a hard requirement in federal statistics**, not a nice-to-have. Decision trees produce printable rules. Random Forests produce ranked feature lists. Both can appear in methodology reports.
- **Random Forests outperform single trees** by training many trees on bootstrap samples and averaging. The variance reduction is often substantial at no interpretability cost at the feature level.
- **Feature importance answers operational questions**: which survey variables most predict nonresponse? Which paradata fields are most useful? Permutation importance on the test set is the reliable way to answer these.
- **The model comparison across Chapters 1 through 3** shows that logistic regression, decision trees, and Random Forests all perform similarly on this dataset. Logistic regression is often sufficient. Use complexity only when the data is large and performance gaps are real.
- **Out-of-bag scoring** gives you a free cross-validation estimate during training. Use it to monitor performance during hyperparameter tuning without touching the test set.
- **SHAP values provide per-prediction explanations** that are essential for auditable federal AI. Global importance measures tell you what matters on average; SHAP tells you what mattered for this specific case.
- **Feature importance rankings should be tested for stability** across repeated runs. A single-run ranking overstates precision. Report mean ranks with confidence intervals when the ranking will appear in a methodology report.
- **Random Forests scale as O(T x N x features x depth).** For large federal datasets, subsample for exploration and verify that importance findings are stable before committing to a full run.

```{admonition} How to explain these methods to leadership
:class: tip
**What does a decision tree do?** It learns a series of if-then rules from historical data. Example: "If a household had 3 or more contact attempts AND did not respond in the previous cycle, flag it for in-person follow-up." You can print those rules and put them in the methodology appendix.

**What is a Random Forest?** It runs the same process 200 times, each time with slightly different data and features, and takes the majority vote. Think of it as 200 analysts independently applying judgment and then voting. The result is more reliable than any single analyst.

**What is feature importance?** A ranked list of which variables most influenced the predictions. "Prior response history" and "number of contact attempts" are the strongest predictors in our model. This tells field operations where to focus data collection effort for future surveys.

**Can we explain why the model flagged a specific household?** With the decision tree, yes -- you can trace the exact path. With the Random Forest, you can identify which features drove the overall prediction, but not a single printed rule. If individual-case auditability is required (FOIA, legal review), a shallow decision tree is the safer choice.
```
