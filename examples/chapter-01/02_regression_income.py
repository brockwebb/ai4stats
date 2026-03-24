"""
Chapter 1 -- Regression: Predicting Income from Demographic Features

Loads the synthetic ACS-like survey data and fits three regression models
(LinearRegression, Ridge, Lasso) to predict individual income.

What this script demonstrates:
- Train/test split and why the test set must remain unseen during training
- MAE, MSE, and R^2 as regression evaluation metrics
- Coefficient interpretation: each coefficient is change in predicted income
  per one-unit increase in that feature, all else equal
- Diagnostic plots: residual plot (random scatter = good), parity plot
  (proximity to diagonal = good)
- How split size and random seed affect reported metrics
- Ridge (L2) and Lasso (L1) regularization and when to prefer each

Prerequisites: run 01_generate_survey_data.py first.

Usage:
    python 02_regression_income.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = "synthetic_acs_survey.csv"
FEATURES = ["age", "education_years", "hours_per_week", "urban"]
TARGET = "income"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_data(path):
    """Load the synthetic survey CSV and return feature matrix and target."""
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET]
    return df, X, y


def print_metrics(label, y_true, y_pred):
    """Print MAE, MSE, and R^2 for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:<22}  MAE=${mae:>10,.0f}  MSE=${mse:>14,.0f}  R²={r2:.3f}")


def plot_coefficients(model, feature_names, title):
    """Horizontal bar chart of model coefficients."""
    coef_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_}
    ).sort_values("coefficient", ascending=False)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(coef_df["feature"], coef_df["coefficient"])
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Coefficient (dollars per unit increase in feature)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_diagnostics(y_true, y_pred, title_prefix):
    """Residual plot and parity plot side by side."""
    resid = np.asarray(y_true) - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Residual plot: random scatter around zero is ideal.
    axes[0].scatter(y_pred, resid, alpha=0.4, s=15)
    axes[0].axhline(0, color="k", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Predicted income ($)")
    axes[0].set_ylabel("Residual (true - predicted)")
    axes[0].set_title(f"{title_prefix}: residual plot")

    # Parity plot: points should cluster near the diagonal.
    lims = [
        min(np.min(y_true), y_pred.min()),
        max(np.max(y_true), y_pred.max()),
    ]
    axes[1].scatter(y_true, y_pred, alpha=0.4, s=15)
    axes[1].plot(lims, lims, "k--", linewidth=1)
    axes[1].set_xlabel("True income ($)")
    axes[1].set_ylabel("Predicted income ($)")
    axes[1].set_title(f"{title_prefix}: parity plot")

    fig.tight_layout()
    return fig


def split_sensitivity_analysis(X, y, test_sizes, n_seeds=30):
    """
    Fit LinearRegression across multiple split sizes and seeds.

    Returns a DataFrame summarising mean and std of MAE and R^2
    for each test_size.
    """
    rows = []
    for t in test_sizes:
        for s in range(n_seeds):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=t, random_state=s
            )
            m = LinearRegression().fit(X_tr, y_tr)
            yh = m.predict(X_te)
            rows.append({
                "test_size": t,
                "seed": s,
                "MAE": mean_absolute_error(y_te, yh),
                "R2": r2_score(y_te, yh),
            })

    df_splits = pd.DataFrame(rows)
    summary = (
        df_splits.groupby("test_size")
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .reset_index()
    )
    return summary


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Load data and split
    # ------------------------------------------------------------------
    df, X, y = load_data(CSV_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows\n")

    # ------------------------------------------------------------------
    # 2. Fit and evaluate three regression models
    # ------------------------------------------------------------------
    models = {
        "Linear": LinearRegression(),
        "Ridge (alpha=100)": Ridge(alpha=100),
        "Lasso (alpha=50)": Lasso(alpha=50, max_iter=10_000),
    }

    fitted = {}
    print("Model comparison:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print_metrics(name, y_test, y_pred)
        fitted[name] = (model, y_pred)

    # ------------------------------------------------------------------
    # 3. Coefficient table for linear regression
    # ------------------------------------------------------------------
    lin_model, lin_pred = fitted["Linear"]
    coef_df = pd.DataFrame(
        {"feature": FEATURES, "coefficient": lin_model.coef_}
    ).sort_values("coefficient", ascending=False)

    print(f"\nLinear regression intercept: ${lin_model.intercept_:,.0f}")
    print("\nCoefficients (dollars per unit increase):")
    print(coef_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Diagnostic plots for the linear model
    # ------------------------------------------------------------------
    plot_coefficients(lin_model, FEATURES, "Linear regression: income coefficients")
    plt.savefig("fig_coef_linear.png", dpi=120, bbox_inches="tight")

    plot_diagnostics(y_test, lin_pred, "Linear regression")
    plt.savefig("fig_diagnostics_linear.png", dpi=120, bbox_inches="tight")

    # ------------------------------------------------------------------
    # 5. Split sensitivity analysis
    # ------------------------------------------------------------------
    print("\nSplit sensitivity (LinearRegression, 30 seeds each):")
    summary = split_sensitivity_analysis(X, y, test_sizes=[0.10, 0.20, 0.30])
    print(summary.round(1).to_string(index=False))

    plt.show()
    print("\nDone. Figures saved to fig_coef_linear.png and fig_diagnostics_linear.png")
