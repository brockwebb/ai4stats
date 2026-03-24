# Chapter 9 Examples: Synthetic Data Generation for Federal Statistics

Standalone Python scripts demonstrating the concepts in Chapter 9.
Run them in order: script 01 generates the data files that later scripts depend on.

## Requirements

Python 3.9+, plus:
```
pip install numpy pandas matplotlib scikit-learn scipy
```

## Scripts

| Script | Description |
|--------|-------------|
| `01_confidential_dataset.py` | Generate synthetic confidential microdata (n=600). Demonstrate three traditional disclosure avoidance methods: top-coding, noise addition, and data swapping. Save `confidential_microdata.csv`. |
| `02_sequential_synthesis.py` | Implement full sequential regression synthesis (age, educ, region, income, married). Compare confidential vs. synthetic summary statistics. Save `synthetic_data.csv`. |
| `03_utility_marginal.py` | Marginal utility evaluation: 4-panel histogram comparison and summary statistics table (mean, std, median, skew) for all variables. |
| `04_utility_bivariate.py` | Bivariate utility: correlation matrix comparison as a 3-panel heatmap (confidential, synthetic, difference). |
| `05_utility_regression.py` | Analytic validity test: fit `income ~ age + educ + region` on both datasets; compare coefficient recovery. |
| `06_utility_pmse.py` | Propensity score MSE (pMSE) global utility metric. Stack datasets, train classifier, measure distinguishability. |
| `07_privacy_utility_tradeoff.py` | KNN synthesizer with k=1, 10, 50. Compute pMSE (utility) and nearest-neighbor distance ratio (privacy proxy). Plot the tradeoff. |
| `08_differential_privacy.py` | Laplace mechanism demonstration. Show noise distributions for epsilon=0.1, 1.0, 10.0. Accuracy vs. epsilon table. 2020 Census DAS context. |
| `09_disclosure_risk.py` | Identity disclosure rate (quasi-identifier matching) and attribute disclosure (inferring income from demographics). |
| `10_limitations.py` | Demonstrate that omitting `married` from the synthesis model destroys the income-married correlation. |
| `11_exercise.py` | Exercise setup and full solution: extend synthesis to include `married`; verify correlation restoration. |

## Data dependencies

```
01 --> confidential_microdata.csv --> 02, 03, 04, 05, 06, 07, 09, 10, 11
02 --> synthetic_data.csv         --> 03, 04, 05, 06, 09
```

Scripts 07, 08, 10, and 11 generate their own synthetic data internally and do not
require `synthetic_data.csv`.
