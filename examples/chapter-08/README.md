# Chapter 8 Examples: Bias, Fairness, and Equity in Federal AI/ML

Standalone Python scripts demonstrating the concepts in Chapter 8.
Each script is independently runnable and self-documented.

## Scripts

| File | Description |
|------|-------------|
| `01_dataset_and_model.py` | Builds a synthetic ACS-like nonresponse dataset (2,000 records) and fits a logistic regression model. Saves test predictions to `ch08_test_predictions.csv` for use by downstream scripts. |
| `02_differential_undercount.py` | Visualizes 2020 Census Post-Enumeration Survey (PES) net undercount estimates by demographic group. Uses published estimates; no model required. |
| `03_training_data_bias.py` | Compares full-population composition to respondent-only composition; computes representation ratios to quantify survivorship bias in training data. |
| `04_fairness_metrics.py` | Computes four fairness metrics (demographic parity, true positive rate, false positive rate, precision) by racial/ethnic group and produces a 4-panel bar chart. |
| `05_impossibility_theorem.py` | Sweeps decision thresholds from 0.10 to 0.90 for two groups with different base rates (Hispanic vs. Asian non-Hispanic) to show that TPR-parity and precision-parity cannot be simultaneously achieved. |
| `06_subgroup_decomposition.py` | General-purpose `subgroup_decomposition` function; applies it by race/ethnicity and by income quintile. Produces 3-panel accuracy/miss-rate visualization. |
| `07_compounding_effect.py` | Calculates how a high base rate of nonresponse and a high model miss rate multiply to produce compound coverage risk. Prints the 7-step compounding cascade. |
| `08_model_card.py` | Populates and prints a model card following the Mitchell et al. (2019) format, including governance fields required by OMB SPD-15 and executive order provisions. |
| `09_exercises.py` | Setup and starter scaffolding for Exercises 8.1 (age group decomposition), 8.2 (fairness metric conflicts), and 8.3 (leadership briefing template). |

## Running Order

Run `01_dataset_and_model.py` first -- it saves `ch08_test_predictions.csv`
which is required by scripts 03-09.

```bash
python 01_dataset_and_model.py
python 02_differential_undercount.py
python 03_training_data_bias.py
python 04_fairness_metrics.py
python 05_impossibility_theorem.py
python 06_subgroup_decomposition.py
python 07_compounding_effect.py
python 08_model_card.py
python 09_exercises.py
```

## Requirements

- Python 3.9+
- numpy
- pandas
- matplotlib
- scikit-learn

Install with: `pip install numpy pandas matplotlib scikit-learn`

## Data

All scripts use synthetic data generated in `01_dataset_and_model.py`.
No real microdata required. The 2020 Census PES estimates in
`02_differential_undercount.py` use published aggregate statistics
(not microdata) from the Census Bureau G-01 report (November 2022).
