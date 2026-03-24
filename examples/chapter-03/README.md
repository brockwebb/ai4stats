# Chapter 3 Code Examples: Decision Trees and Random Forests

Run in order:
1. `01_dataset_setup.py` — dataset setup and splits
2. `02_decision_tree.py` — decision tree fitting, rules, overfitting analysis
3. `03_random_forest.py` — Random Forest, OOB, feature importance
4. `04_shap_analysis.py` — SHAP explanations (requires `pip install shap`)
5. `05_stability_analysis.py` — importance stability analysis (30 bootstrap runs)
6. `06_computational_scaling.py` — timing experiments, scaling extrapolation
7. `07_regression_trees.py` — tree-based regression
8. `08_model_comparison.py` — three-model comparison (LR, DT, RF)
9. `09_hyperparameter_search.py` — RandomizedSearchCV
10. `10_tract_exercise.py` — tract-level targeting exercise and solution (includes SHAP)

## Requirements
Python 3.9+, numpy, pandas, matplotlib, scikit-learn
SHAP scripts (04, 05, 10): `pip install shap`
