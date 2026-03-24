"""
10_limitations.py
=================
Chapter 9: Synthetic Data Generation for Federal Statistics

Demonstrates the key limitation of sequential synthesis: a variable that is
not included in the synthesis model loses its correlations with other variables.
Specifically, this script synthesizes income WITHOUT including married in the
model, then shows that the income-married correlation is destroyed.

Why this matters:
    This is the most common source of user complaints about synthetic data:
    "Your synthetic data says income and marital status are unrelated, but we
    know from every other survey that married adults earn more on average."
    The answer is not that the data is wrong — it is that the synthesis was not
    designed to preserve that relationship. Users must understand which
    analyses the synthesis was validated for.

Usage:
    python 10_limitations.py
    (Requires confidential_microdata.csv)

Outputs:
    - Before/after correlation tables printed to stdout
    - correlation_loss.png showing the contrast

Requirements:
    Python 3.9+, numpy, pandas, matplotlib, scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def synthesize_without_married(
    df_conf: pd.DataFrame,
    n_synth: int = 600,
    random_state: int = 2025,
) -> pd.DataFrame:
    """
    Sequential synthesis that does NOT include married in the model.

    married is appended afterward by sampling from its marginal distribution
    independently of income, age, and education. This destroys the
    income-married correlation even though married is present in the dataset.
    """
    rng = np.random.default_rng(random_state)

    # Step 1: age from marginal
    age = np.clip(np.round(rng.normal(df_conf["age"].mean(), df_conf["age"].std(), n_synth)), 18, 80).astype(int)

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

    # Step 4: income | age, educ, region (on log scale)
    le_reg2 = LabelEncoder()
    region_enc_conf = le_reg2.fit_transform(df_conf["region"])
    lr_inc = LinearRegression()
    X_inc = np.column_stack([df_conf["age"].values, df_conf["educ"].values, region_enc_conf])
    log_income = np.log(np.maximum(df_conf["income"].values, 1))
    lr_inc.fit(X_inc, log_income)
    residuals = log_income - lr_inc.predict(X_inc)
    region_enc_synth = le_reg2.transform(region)
    X_synth_inc = np.column_stack([age, educ, region_enc_synth])
    pred_log = lr_inc.predict(X_synth_inc)
    noise = rng.normal(0, residuals.std(), n_synth)
    income = np.clip(np.exp(pred_log + noise), 5000, 300_000).astype(int)

    # Step 5: married sampled INDEPENDENTLY from marginal — no conditioning on income
    # This is the intentional omission that destroys the correlation.
    married_marginal_rate = df_conf["married"].mean()
    married = rng.binomial(1, married_marginal_rate, n_synth)

    return pd.DataFrame({"age": age, "educ": educ, "region": region, "income": income, "married": married})


def print_correlation_comparison(df_conf: pd.DataFrame, df_synth_no_mar: pd.DataFrame) -> None:
    """Print before/after correlation tables for income-married."""
    conf_corr = df_conf[["age", "educ", "income", "married"]].corr()
    synth_corr = df_synth_no_mar[["age", "educ", "income", "married"]].corr()

    print("Correlation with income: confidential vs. synthesis WITHOUT married model")
    print("=" * 68)
    print(f"{'Variable':<12} {'Confidential':>16} {'Synthetic (no mar)':>20} {'Lost':>8}")
    print("-" * 68)
    for col in ["age", "educ", "married"]:
        cv = conf_corr.loc["income", col]
        sv = synth_corr.loc["income", col]
        lost = abs(cv - sv)
        flag = " <-- LOST" if col == "married" else ""
        print(f"{col:<12} {cv:>16.3f} {sv:>20.3f} {lost:>8.3f}{flag}")

    print()
    corr_conf = df_conf["income"].corr(df_conf["married"])
    corr_synth = df_synth_no_mar["income"].corr(df_synth_no_mar["married"])
    print(f"Income-married correlation in confidential data:      {corr_conf:.3f}")
    print(f"Income-married correlation in synthetic (no model):   {corr_synth:.3f}")
    print(f"Correlation lost:                                     {abs(corr_conf - corr_synth):.3f}")
    print()
    print("Explanation:")
    print("  married was appended from its marginal distribution (P=0.52 regardless")
    print("  of income). The synthesizer had no information that higher-income people")
    print("  are more likely to be married. This is not a bug — it is a deliberate")
    print("  omission. Including married in the model (as in 02_sequential_synthesis.py)")
    print("  restores the correlation.")


if __name__ == "__main__":
    df_conf = load_confidential()
    print(f"Loaded confidential microdata: n={len(df_conf)}")
    print()

    df_synth_no_mar = synthesize_without_married(df_conf, n_synth=600, random_state=2025)
    print_correlation_comparison(df_conf, df_synth_no_mar)

    # Simple two-panel plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (df, title) in zip(axes, [
        (df_conf,        "Confidential data"),
        (df_synth_no_mar, "Synthetic (married not modeled)"),
    ]):
        not_married = df[df["married"] == 0]["income"].values
        is_married  = df[df["married"] == 1]["income"].values
        bins = np.linspace(0, 200_000, 40)
        ax.hist(not_married, bins=bins, alpha=0.5, density=True, color="steelblue", label="Not married")
        ax.hist(is_married,  bins=bins, alpha=0.5, density=True, color="tomato",    label="Married")
        ax.set_xlabel("Income ($)")
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle("Income distribution by marital status: confidential vs. synthetic", fontsize=11)
    plt.tight_layout()
    plt.savefig("correlation_loss.png", dpi=150, bbox_inches="tight")
    print("\nSaved figure: correlation_loss.png")
    plt.close()
