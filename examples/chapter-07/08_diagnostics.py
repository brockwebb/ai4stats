"""
08_diagnostics.py
Chapter 7: Imputation Methods for Survey Data

Imputation quality diagnostics:
  - plausibility_diagnostics: mean shift, variance ratio, subgroup distribution
  - Density overlays by education group (standard federal statistics diagnostic)
  - Full method comparison table (MAE, variance ratio, auditability score)

The density overlay is the most important visual diagnostic: if the imputed
distribution looks very different from the observed distribution within a
subgroup, something is wrong with the imputation model for that subgroup.

Requires: base_data.csv produced by 01_dataset_and_missingness.py

Usage:
    python 08_diagnostics.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

INPUT_FILE = "base_data.csv"
RANDOM_SEED = 42
PREDICTORS = ["age", "educ", "region", "hours_wk", "fulltime"]
N_CLASSES = 10


# ---------------------------------------------------------------------------
# Imputation helpers (self-contained so this script runs standalone)
# ---------------------------------------------------------------------------
def _impute_mean(series):
    return series.fillna(series.mean())


def _build_clusters(df, feature_cols, n_classes, seed):
    scaler = StandardScaler()
    complete_mask = df[feature_cols].notna().all(axis=1)
    X_c = scaler.fit_transform(df.loc[complete_mask, feature_cols])
    km = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
    km.fit(X_c)
    X_all = scaler.transform(df[feature_cols].fillna(df[feature_cols].median()))
    return km.predict(X_all)


def _hot_deck(df, target_col, class_col, seed):
    rng = np.random.default_rng(seed)
    result = df[target_col].copy()
    for idx in df.index[df[target_col].isna()]:
        cls = df.loc[idx, class_col]
        donors = df.loc[(df[class_col] == cls) & df[target_col].notna(), target_col]
        result[idx] = rng.choice(donors.values) if len(donors) else df[target_col].mean()
    return result


def _stochastic_regression(df, target_col, predictor_cols, seed):
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
    result[~complete] = preds[~complete] + rng.normal(0, resid_std, (~complete).sum())
    return result.clip(lower=0)


def _rf_impute(df, target_col, predictor_cols, seed):
    rng = np.random.default_rng(seed)
    complete = df[target_col].notna()
    X_c = pd.get_dummies(df.loc[complete, predictor_cols], drop_first=True)
    y_c = df.loc[complete, target_col]
    rf = RandomForestRegressor(n_estimators=100, random_state=seed, min_samples_leaf=5)
    rf.fit(X_c, y_c)
    X_all = pd.get_dummies(df[predictor_cols], drop_first=True).reindex(
        columns=X_c.columns, fill_value=0
    )
    all_preds = np.array([t.predict(X_all) for t in rf.estimators_])
    choice = rng.integers(0, len(rf.estimators_), size=len(df))
    stoch_preds = all_preds[choice, np.arange(len(df))]
    result = df[target_col].copy()
    result[~complete] = stoch_preds[~complete]
    return result, rf


# ---------------------------------------------------------------------------
# Plausibility diagnostics
# ---------------------------------------------------------------------------
def plausibility_diagnostics(
    df: pd.DataFrame,
    imputed_col: str,
    original_col: str,
    missing_col: str,
    variable: str,
    group_col: str = None,
) -> pd.DataFrame:
    """
    Check whether imputed values are plausible relative to observed values.

    For each group (if group_col is provided), computes:
      - Mean of observed values
      - Mean of imputed values
      - True mean (if income_true is present)
      - Standard deviation of observed values
      - Standard deviation of imputed values
      - Variance ratio (imputed SD / observed SD); near 1.0 is good

    Parameters
    ----------
    df : pd.DataFrame
    imputed_col : str
        Column containing imputed values for all records.
    original_col : str
        Column with observed values (NaN where missing).
    missing_col : str
        Boolean column, True where imputation was applied.
    variable : str
        Label for the target variable (used in display).
    group_col : str or None
        If provided, compute diagnostics within each group.

    Returns
    -------
    pd.DataFrame with per-group diagnostics.
    """
    if group_col is None:
        groups = [("All", slice(None))]
        group_values = ["All"]
    else:
        group_values = sorted(df[group_col].unique())
        groups = [(g, df[group_col] == g) for g in group_values]

    rows = []
    for label, group_mask in groups:
        obs_mask = group_mask & ~df[missing_col]
        imp_mask = group_mask & df[missing_col]

        obs_vals = df.loc[obs_mask, original_col].dropna()
        imp_vals = df.loc[imp_mask, imputed_col]

        row = {
            group_col or "Group": label,
            "n_obs": obs_mask.sum(),
            "n_imp": imp_mask.sum(),
            "obs_mean": obs_vals.mean() if len(obs_vals) else np.nan,
            "imp_mean": imp_vals.mean() if len(imp_vals) else np.nan,
            "obs_sd": obs_vals.std() if len(obs_vals) else np.nan,
            "imp_sd": imp_vals.std() if len(imp_vals) else np.nan,
        }
        if "income_true" in df.columns:
            row["true_mean"] = df.loc[imp_mask, "income_true"].mean() if imp_mask.sum() else np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    result["var_ratio"] = result["imp_sd"] / result["obs_sd"]
    return result


# ---------------------------------------------------------------------------
# Density overlay diagnostic
# ---------------------------------------------------------------------------
def density_overlay_by_education(df: pd.DataFrame, methods: dict) -> None:
    """
    For each education level, overlay KDE curves for observed vs. imputed
    distributions. This is the standard visual diagnostic for imputation
    evaluation in federal survey programs.

    A good imputation shows imputed density roughly tracking observed density
    within each education group.
    """
    educ_labels = {
        1: "Less than HS", 2: "HS diploma",
        3: "Some college / BA", 4: "Graduate degree",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    xs = np.linspace(0, 150_000, 300)

    for ax_idx, educ_level in enumerate([1, 2, 3, 4]):
        ax = axes[ax_idx // 2][ax_idx % 2]
        obs_mask = (df["educ"] == educ_level) & ~df["missing"]
        imp_mask = (df["educ"] == educ_level) & df["missing"]

        if obs_mask.sum() < 5 or imp_mask.sum() < 3:
            ax.set_title(f"Education: {educ_labels[educ_level]}\n(insufficient data)")
            continue

        obs_vals = df.loc[obs_mask, "income_obs"].dropna()
        kde_obs = gaussian_kde(obs_vals, bw_method=0.3)(xs)
        ax.fill_between(xs, kde_obs, alpha=0.25, color="steelblue")
        ax.plot(xs, kde_obs, color="steelblue", linewidth=2, label="Observed")

        for col, color, label in methods.values():
            imp_vals = df.loc[imp_mask, col]
            if len(imp_vals) >= 3:
                kde_imp = gaussian_kde(imp_vals, bw_method=0.4)(xs)
                ax.plot(xs, kde_imp, color=color, linewidth=1.5, linestyle="--", label=label)

        ax.set_xlabel("Income ($)")
        ax.set_xlim(0, 150_000)
        ax.set_title(
            f"Education: {educ_labels[educ_level]}\n"
            f"(n obs={obs_mask.sum()}, n imputed={imp_mask.sum()})"
        )
        ax.legend(fontsize=7)

    fig.suptitle(
        "Density overlay: observed vs. imputed distributions by education group\n"
        "Good imputation: imputed density should roughly track observed density",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("08_density_overlay.png", dpi=120, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Method comparison table
# ---------------------------------------------------------------------------
def print_comparison_table(df: pd.DataFrame, methods: dict) -> None:
    """
    Print a summary table: MAE, variance ratio, and auditability for each method.
    """
    true_var = df["income_true"].var()
    auditability = {
        "Mean": "High",
        "Conditional mean": "High",
        "Hot-deck": "High",
        "Stochastic regression": "Medium",
        "Random Forest": "Low",
    }
    print(f"\n{'Method':<25} {'MAE ($)':>10} {'Var ratio':>10} {'Auditable':>11}")
    print("-" * 62)
    for name, (col, color, label) in methods.items():
        mae = mean_absolute_error(
            df.loc[df["missing"], "income_true"],
            df.loc[df["missing"], col],
        )
        vr = df[col].var() / true_var
        audit = auditability.get(name, "?")
        print(f"  {name:<23} {mae:>10,.0f} {vr:>10.3f} {audit:>11}")
    print("-" * 62)
    print("Var ratio: imputed-dataset variance / true variance. 1.00 = perfect.")
    print("Auditable = can you explain every imputed value to OMB or an IG?")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Chapter 7: Imputation Diagnostics ===\n")

    df = pd.read_csv(INPUT_FILE)
    df["missing"] = df["missing"].astype(bool)

    # Build all imputed columns
    df["income_mean_imp"] = _impute_mean(df["income_obs"])

    from sklearn.linear_model import LinearRegression as _LR  # noqa (already imported)
    region_map = {"Northeast": 0, "Midwest": 1, "South": 2, "West": 3}
    df["region_num"] = df["region"].map(region_map)
    df["imp_class"] = _build_clusters(
        df, ["age", "educ", "hours_wk", "region_num"], N_CLASSES, RANDOM_SEED
    )
    df["income_hotdeck"] = _hot_deck(df, "income_obs", "imp_class", RANDOM_SEED)
    df["income_reg_stoch"] = _stochastic_regression(
        df, "income_obs", PREDICTORS, RANDOM_SEED
    )
    df["income_rf"], _ = _rf_impute(df, "income_obs", PREDICTORS, RANDOM_SEED)

    methods = {
        "Mean": ("income_mean_imp", "tomato", "Mean"),
        "Hot-deck": ("income_hotdeck", "seagreen", "Hot-deck"),
        "Stochastic regression": ("income_reg_stoch", "darkorange", "Stoch. reg"),
        "Random Forest": ("income_rf", "mediumpurple", "Random Forest"),
    }

    # Plausibility diagnostics
    print("Plausibility diagnostics: hot-deck by education level")
    print("=" * 70)
    diag = plausibility_diagnostics(
        df, "income_hotdeck", "income_obs", "missing", "income", group_col="educ"
    )
    educ_labels = {1: "<HS   ", 2: "HS    ", 3: "College", 4: "Grad  "}
    for _, row in diag.iterrows():
        e = int(row["educ"])
        print(
            f"  Educ {e} ({educ_labels[e]}): "
            f"obs=${row['obs_mean']:,.0f}  imp=${row['imp_mean']:,.0f}  "
            f"true=${row['true_mean']:,.0f}  var_ratio={row['var_ratio']:.2f}"
        )

    print("\nDensity overlay by education group (all methods):")
    density_overlay_by_education(df, methods)
    print("Figure saved to 08_density_overlay.png")

    print("\nMethod comparison table:")
    print_comparison_table(df, methods)
