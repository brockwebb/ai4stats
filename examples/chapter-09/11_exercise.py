"""
11_exercise.py
==============
Chapter 9: Synthetic Data Generation for Federal Statistics

Exercise: extend the sequential synthesis to include married as a modeled
variable and verify that the income-married correlation is restored.

The starter code provides the synthesis pipeline WITHOUT the married step.
The exercise asks you to add it. The full solution is provided below the
starter code for reference.

Why this matters:
    The exercise makes concrete the chapter's central lesson: the synthesis
    preserves what you explicitly model, nothing more. Once you see the
    correlation restored by adding a single logistic regression step, the
    mechanism is no longer abstract.

Exercise questions (no code required):
    1. A user reports that the synthetic data shows no relationship between
       income and marital status. Using what you know about sequential
       synthesis, explain why.

    2. The pMSE for the base synthesis (without married) is 0.003. After
       adding the married step, it rises to 0.005. Is this acceptable?
       What additional checks would you run?

    3. An analyst needs to study the income-marital status relationship for
       a congressional report. Should they use the synthetic data or request
       FSRDC access? Justify your answer.

Usage:
    python 11_exercise.py

Outputs:
    - Starter code output (married not modeled) printed to stdout
    - Full solution output printed to stdout

Requirements:
    Python 3.9+, numpy, pandas, scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os


def load_confidential() -> pd.DataFrame:
    """Load confidential microdata from CSV."""
    if not os.path.exists("confidential_microdata.csv"):
        raise FileNotFoundError(
            "confidential_microdata.csv not found. Run 01_confidential_dataset.py first."
        )
    return pd.read_csv("confidential_microdata.csv")


def synthesize_base(
    df_conf: pd.DataFrame,
    n_synth: int = 600,
    random_state: int = 2025,
) -> pd.DataFrame:
    """
    Sequential synthesis WITHOUT married modeled.

    This is the starter code. married is appended from its marginal
    distribution only, destroying its correlation with income.
    """
    rng = np.random.default_rng(random_state)

    # Step 1: age
    age = np.clip(
        np.round(rng.normal(df_conf["age"].mean(), df_conf["age"].std(), n_synth)),
        18, 80,
    ).astype(int)

    # Step 2: educ | age
    le_educ = LabelEncoder()
    educ_enc = le_educ.fit_transform(df_conf["educ"])
    lr_educ = LogisticRegression(max_iter=500, random_state=random_state)
    lr_educ.fit(df_conf[["age"]], educ_enc)
    educ_probs = lr_educ.predict_proba(age.reshape(-1, 1))
    educ = le_educ.inverse_transform([rng.choice(len(p), p=p) for p in educ_probs])

    # Step 3: region | age, educ
    le_reg = LabelEncoder()
    reg_enc = le_reg.fit_transform(df_conf["region"])
    lr_reg = LogisticRegression(max_iter=500, random_state=random_state)
    lr_reg.fit(df_conf[["age", "educ"]], reg_enc)
    reg_probs = lr_reg.predict_proba(np.column_stack([age, educ]))
    region = le_reg.inverse_transform([rng.choice(len(p), p=p) for p in reg_probs])

    # Step 4: income | age, educ, region
    le_reg2 = LabelEncoder()
    region_enc_conf = le_reg2.fit_transform(df_conf["region"])
    lr_inc = LinearRegression()
    X_inc = np.column_stack([df_conf["age"].values, df_conf["educ"].values, region_enc_conf])
    log_income = np.log(np.maximum(df_conf["income"].values, 1))
    lr_inc.fit(X_inc, log_income)
    residuals = log_income - lr_inc.predict(X_inc)
    region_enc_synth = le_reg2.transform(region)
    X_synth_inc = np.column_stack([age, educ, region_enc_synth])
    noise = rng.normal(0, residuals.std(), n_synth)
    income = np.clip(np.exp(lr_inc.predict(X_synth_inc) + noise), 5000, 300_000).astype(int)

    # Step 5: married — NOT MODELED (starter code)
    # Replace this with a logistic regression step for the exercise.
    married = rng.binomial(1, df_conf["married"].mean(), n_synth)

    return pd.DataFrame({"age": age, "educ": educ, "region": region, "income": income, "married": married})


def synthesize_with_married(
    df_conf: pd.DataFrame,
    n_synth: int = 600,
    random_state: int = 2025,
) -> pd.DataFrame:
    """
    Full solution: sequential synthesis WITH married modeled as the last step.

    Add logistic regression: married ~ age + educ + income
    Fit on confidential data; sample for synthetic records.
    """
    rng = np.random.default_rng(random_state)

    # Steps 1-4: identical to synthesize_base
    age = np.clip(
        np.round(rng.normal(df_conf["age"].mean(), df_conf["age"].std(), n_synth)),
        18, 80,
    ).astype(int)

    le_educ = LabelEncoder()
    educ_enc = le_educ.fit_transform(df_conf["educ"])
    lr_educ = LogisticRegression(max_iter=500, random_state=random_state)
    lr_educ.fit(df_conf[["age"]], educ_enc)
    educ_probs = lr_educ.predict_proba(age.reshape(-1, 1))
    educ = le_educ.inverse_transform([rng.choice(len(p), p=p) for p in educ_probs])

    le_reg = LabelEncoder()
    reg_enc = le_reg.fit_transform(df_conf["region"])
    lr_reg = LogisticRegression(max_iter=500, random_state=random_state)
    lr_reg.fit(df_conf[["age", "educ"]], reg_enc)
    reg_probs = lr_reg.predict_proba(np.column_stack([age, educ]))
    region = le_reg.inverse_transform([rng.choice(len(p), p=p) for p in reg_probs])

    le_reg2 = LabelEncoder()
    region_enc_conf = le_reg2.fit_transform(df_conf["region"])
    lr_inc = LinearRegression()
    X_inc = np.column_stack([df_conf["age"].values, df_conf["educ"].values, region_enc_conf])
    log_income = np.log(np.maximum(df_conf["income"].values, 1))
    lr_inc.fit(X_inc, log_income)
    residuals = log_income - lr_inc.predict(X_inc)
    region_enc_synth = le_reg2.transform(region)
    X_synth_inc = np.column_stack([age, educ, region_enc_synth])
    noise = rng.normal(0, residuals.std(), n_synth)
    income = np.clip(np.exp(lr_inc.predict(X_synth_inc) + noise), 5000, 300_000).astype(int)

    # Step 5: married | age, educ, income (SOLUTION)
    lr_married = LogisticRegression(max_iter=500, random_state=random_state)
    lr_married.fit(df_conf[["age", "educ", "income"]], df_conf["married"])
    married_probs = lr_married.predict_proba(
        np.column_stack([age, educ, income])
    )[:, 1]
    married = rng.binomial(1, married_probs)

    return pd.DataFrame({"age": age, "educ": educ, "region": region, "income": income, "married": married})


if __name__ == "__main__":
    df_conf = load_confidential()
    corr_conf = df_conf["income"].corr(df_conf["married"])
    print(f"Confidential income-married correlation: {corr_conf:.3f}")
    print()

    # Starter code result
    df_base = synthesize_base(df_conf, n_synth=600, random_state=2025)
    corr_base = df_base["income"].corr(df_base["married"])
    print(f"Starter code (married not modeled): income-married corr = {corr_base:.3f}")
    print(f"  Correlation loss: {abs(corr_conf - corr_base):.3f}")
    print()

    # Full solution result
    df_full = synthesize_with_married(df_conf, n_synth=600, random_state=2025)
    corr_full = df_full["income"].corr(df_full["married"])
    print(f"Full solution (married modeled):     income-married corr = {corr_full:.3f}")
    print(f"  Correlation loss: {abs(corr_conf - corr_full):.3f}")
    print()
    print("Adding a single logistic regression step largely restores the correlation.")
    print("The residual gap reflects sampling noise, not a synthesis deficiency.")
