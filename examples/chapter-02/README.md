# Chapter 2 Code Examples: Cross-Validation and Model Selection

Run in order:

1. `01_dataset_setup.py` -- reproduces the Chapter 1 dataset + household clusters
2. `02_split_variability.py` -- demonstrates why a single split is unreliable
3. `03_cross_validation.py` -- KFold, StratifiedKFold, GroupKFold
4. `04_gridsearch_tuning.py` -- hyperparameter search without touching the test set
5. `05_feature_importance.py` -- coefficient and permutation importance
6. `06_exercises.py` -- activity solutions

## Requirements

Python 3.9+, numpy, pandas, matplotlib, scikit-learn

## Quick start

```
pip install numpy pandas matplotlib scikit-learn
python 01_dataset_setup.py
python 02_split_variability.py
python 03_cross_validation.py
python 04_gridsearch_tuning.py
python 05_feature_importance.py
python 06_exercises.py
```

Generated figures are saved to `figures/` at the repo root.
The dataset CSV is saved to `data/synthetic_survey_ch02.csv`.
