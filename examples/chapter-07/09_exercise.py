"""
09_exercise.py
Chapter 7: Imputation Methods for Survey Data

Exercise: imputing hours_wk (hours worked per week).

Introduces 15% MAR missingness into hours_wk (full-time workers miss less),
then provides a full solution comparing three imputation methods:
  - Conditional mean (conditioning on fulltime and region)
  - Hot-deck (clustering on age, educ, fulltime, region)
  - Stochastic regression (predictors: age, educ, region, fulltime)

Chapter 7 activity questions (attempt these before running the solution):
  1. Which method would you recommend for operational use? Justify using the
     decision framework in Chapter 7.
  2. A colleague suggests mean imputation because it is simple and everyone
     understands it. Write a 3-sentence response explaining why this is
     problematic for a published survey estimate.
  3. Suppose the density overlay shows RF imputation systematically
     overestimates hours_wk for part-time workers. What does this suggest
     about the training data? What would you do?

Requires: base_data.csv produced by 01_dataset_and_missingness.py

Usage:
    python 09_exercise.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

INPUT_FILE = "base_data.csv"
RANDOM_SEED = 99   # Different seed than earlier scripts to get a distinct missingness pattern
N_CLASSES = 8
MISS_PROB_FULLTIME = 0.08    # Full-time workers: lower skip probability
MISS_PROB_PARTTIME = 0.22    # Part-time workers: higher skip probability
PREDICTORS_HRS = ["age", "educ", "region", "fulltime"]


# ---------------------------------------------------------------------------
# Introduce missingness into hours_wk
# ---------------------------------------------------------------------------
def introduce_missingness(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Introduce MAR missingness into hours_wk.

    Mechanism: part-time workers are more likely to skip the hours question
    (perhaps due to irregular schedules or social desirability). This is MAR
    because the skip probability depends on fulltime (observed), not on
    hours_wk itself.
    """
    rng = np.random.default_rng(seed)
    miss_prob = np.where(df["fulltime"] == 1, MISS_PROB_FULLTIME, MISS_PROB_PARTTIME)
    hours_missing = rng.random(len(df)) < miss_prob
    df = df.copy()
    df["hours_wk_obs"] = df["hours_wk"].astype(float)
    df.loc[hours_missing, "hours_wk_obs"] = np.nan
    df["hours_missing"] = hours_missing
    return df


# ---------------------------------------------------------------------------
# Imputation helpers
# ---------------------------------------------------------------------------
def impute_conditional_mean(df, target_col, group_cols):
    result = df[target_col].copy()
    group_means = df.groupby(group_cols)[target_col].transform("mean")
    result = result.fillna(group_means)
    result = result.fillna(df[target_col].mean())
    return result


def build_imputation_classes(df, feature_cols, n_classes, seed):
    scaler = StandardScaler()
    complete_mask = df[feature_cols].notna().all(axis=1)
    X_c = scaler.fit_transform(df.loc[complete_mask, feature_cols])
    km = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
    km.fit(X_c)
    X_all = scaler.transform(df[feature_cols].fillna(df[feature_cols].median()))
    return km.predict(X_all), scaler


def hot_deck_impute(df, target_col, class_col, seed):
    rng = np.random.default_rng(seed)
    result = df[target_col].copy()
    for idx in df.index[df[target_col].isna()]:
        cls = df.loc[idx, class_col]
        donors = df.loc[(df[class_col] == cls) & df[target_col].notna(), target_col]
        result[idx] = rng.choice(donors.values) if len(donors) else df[target_col].mean()
    return result


def stochastic_regression_impute(df, target_col, predictor_cols, seed):
    rng = np.random.default_rng(seed)
    complete = df[target_col].notna()
    X_c = pd.get_dummies(df.loc[complete, predictor_cols], drop_first=True)
    y_c = df.loc[complete, target_col]
    model = LinearRegression().fit(X_c, y_c)
    resid_std = (y_c - model.predict(X_c)).std()
    X_all = pd.get_dummies(df[predictor_cols], drop_first=True).reindex(
        columns=X_c.columns, fill_value=0
    )
    preds = model.predict(X_all)
    result = df[target_col].copy()
    n_miss = (~complete).sum()
    result[~complete] = preds[~complete] + rng.normal(0, resid_std, n_miss)
    return result.clip(lower=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_methods(df: pd.DataFrame, methods: dict) -> pd.DataFrame:
    """Compare MAE and variance ratio across methods."""
    true_var = df["hours_wk"].var()
    rows = []
    for name, col in methods.items():
        mae = mean_absolute_error(
            df.loc[df["hours_missing"], "hours_wk"],
            df.loc[df["hours_missing"], col],
        )
        vr = df[col].var() / true_var
        rows.append({"Method": name, "MAE (hours/wk)": mae, "Var ratio": vr})
    return pd.DataFrame(rows)


def density_overlay(df: pd.DataFrame, methods: dict) -> None:
    """Density overlay for fulltime and part-time subgroups."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xs = np.linspace(0, 70, 300)

    for ax, (ft, ft_label) in zip(axes, [(1, "Full-time"), (0, "Part-time")]):
        obs_mask = (df["fulltime"] == ft) & ~df["hours_missing"]
        imp_mask = (df["fulltime"] == ft) & df["hours_missing"]

        if obs_mask.sum() < 5 or imp_mask.sum() < 3:
            ax.set_title(f"{ft_label} (insufficient data)")
            continue

        obs_vals = df.loc[obs_mask, "hours_wk_obs"].dropna()
        kde_obs = gaussian_kde(obs_vals, bw_method=0.4)(xs)
        ax.fill_between(xs, kde_obs, alpha=0.25, color="steelblue")
        ax.plot(xs, kde_obs, color="steelblue", linewidth=2, label="Observed")

        colors = ["darkorange", "seagreen", "mediumpurple"]
        for (name, col), color in zip(methods.items(), colors):
            imp_vals = df.loc[imp_mask, col]
            if len(imp_vals) >= 3:
                kde_imp = gaussian_kde(imp_vals, bw_method=0.4)(xs)
                ax.plot(xs, kde_imp, color=color, linewidth=1.5, linestyle="--", label=name)

        ax.set_xlabel("Hours worked per week")
        ax.set_title(
            f"{ft_label} workers\n"
            f"(n obs={obs_mask.sum()}, n imputed={imp_mask.sum()})"
        )
        ax.legend(fontsize=8)

    fig.suptitle("Density overlay: hours_wk by employment type", fontsize=10)
    plt.tight_layout()
    plt.savefig("09_exercise_density.png", dpi=120, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Chapter 7: Exercise -- hours_wk Imputation ===\n")

    df = pd.read_csv(INPUT_FILE)
    df = introduce_missingness(df, RANDOM_SEED)

    n_miss = df["hours_missing"].sum()
    print(f"hours_wk missingness: {n_miss} records ({n_miss/len(df):.1%})")
    print(f"  Full-time missing:  {df.loc[df.fulltime==1, 'hours_missing'].mean():.1%}")
    print(f"  Part-time missing:  {df.loc[df.fulltime==0, 'hours_missing'].mean():.1%}")
    print(
        "\nMechanism: part-time workers skip more. Once we know fulltime status,\n"
        "the skip probability does not depend on actual hours_wk. This is MAR.\n"
    )

    # --- Solution ---
    print("Running imputation methods ...\n")

    # 1. Conditional mean
    df["hours_cond_imp"] = impute_conditional_mean(
        df, "hours_wk_obs", ["fulltime", "region"]
    )

    # 2. Hot-deck (rebuild clusters without hours_wk, which is now missing)
    region_map = {"Northeast": 0, "Midwest": 1, "South": 2, "West": 3}
    df["region_num"] = df["region"].map(region_map)
    hours_clusters, _ = build_imputation_classes(
        df, ["age", "educ", "fulltime", "region_num"], N_CLASSES, RANDOM_SEED
    )
    df["imp_class_hours"] = hours_clusters
    df["hours_hotdeck"] = hot_deck_impute(df, "hours_wk_obs", "imp_class_hours", RANDOM_SEED)

    # 3. Stochastic regression
    df["hours_reg_stoch"] = stochastic_regression_impute(
        df, "hours_wk_obs", PREDICTORS_HRS, RANDOM_SEED
    )

    methods = {
        "Conditional mean": "hours_cond_imp",
        "Hot-deck": "hours_hotdeck",
        "Stochastic regression": "hours_reg_stoch",
    }

    results = evaluate_methods(df, methods)
    print("Method comparison (hours_wk imputation):")
    print(results.to_string(index=False, float_format="{:.3f}".format))

    print(
        "\nDecision framework applied to hours_wk:\n"
        "  Missing rate: ~15%. Hot-deck or regression are appropriate.\n"
        "  Mechanism: MAR (conditioned on fulltime). Must use fulltime as predictor.\n"
        "  Auditability: hot-deck is fully traceable; regression requires documenting\n"
        "    the model specification in the methodology report.\n"
        "  Published estimates: if standard errors matter, use multiple imputation.\n"
    )

    density_overlay(df, methods)
    print("Figure saved to 09_exercise_density.png")

    print("\n--- Discussion questions ---")
    print(
        "1. Which method would you recommend for operational use and why?\n"
        "   Consider the decision framework: missingness rate, mechanism,\n"
        "   available predictors, and auditability requirements.\n"
    )
    print(
        "2. A colleague suggests mean imputation because it is simple. How would\n"
        "   you respond? (Think: variance collapse, published standard errors.)\n"
    )
    print(
        "3. If the density overlay showed RF imputation overestimating hours for\n"
        "   part-time workers, what would that suggest? What would you investigate?\n"
        "   (Hint: what does the RF training data look like for that subgroup?)\n"
    )
