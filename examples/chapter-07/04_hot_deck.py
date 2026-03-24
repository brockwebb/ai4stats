"""
04_hot_deck.py
Chapter 7: Imputation Methods for Survey Data

Hot-deck imputation: for each missing record, draw a random donor from the
same imputation class (cluster). Key features:
  - Every imputed value is a real observed value (plausibility guarantee)
  - Individual imputed values are traceable to specific donors
  - Preserves the empirical distribution of the observed data
  - Handles categorical variables naturally (no model needed)

Requires: base_data.csv produced by 01_dataset_and_missingness.py

Usage:
    python 04_hot_deck.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

INPUT_FILE = "base_data.csv"

# Number of imputation classes (clusters). Typical Census practice: 8-20.
N_CLASSES = 10
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Imputation class construction
# ---------------------------------------------------------------------------
def build_imputation_classes(
    df: pd.DataFrame,
    feature_cols: list,
    n_classes: int = N_CLASSES,
    random_state: int = RANDOM_SEED,
) -> tuple:
    """
    Cluster respondents on observed features to define hot-deck imputation
    classes. Only complete records (no missing on feature_cols) are used to
    fit the scaler and KMeans model. All records (complete and incomplete)
    are then assigned to the nearest cluster centroid.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset. feature_cols must have no missing values (or be filled before call).
    feature_cols : list of str
        Numeric columns used for clustering.
    n_classes : int
        Number of imputation classes (KMeans k).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    clusters : np.ndarray
        Cluster label for every row in df.
    km_model : KMeans
        Fitted KMeans object.
    scaler : StandardScaler
        Fitted scaler.
    """
    scaler = StandardScaler()
    complete_mask = df[feature_cols].notna().all(axis=1)
    X_complete = scaler.fit_transform(df.loc[complete_mask, feature_cols])

    km = KMeans(n_clusters=n_classes, random_state=random_state, n_init=10)
    km.fit(X_complete)

    # Assign all records; fill any remaining missing on features with median
    X_all = scaler.transform(
        df[feature_cols].fillna(df[feature_cols].median())
    )
    clusters = km.predict(X_all)
    return clusters, km, scaler


# ---------------------------------------------------------------------------
# Hot-deck imputation
# ---------------------------------------------------------------------------
def hot_deck_impute(
    df: pd.DataFrame,
    target_col: str,
    class_col: str,
    random_state: int = RANDOM_SEED,
) -> pd.Series:
    """
    Hot-deck imputation: for each missing record, draw one random donor from
    the same imputation class.

    Donor selection rule: a donor must (a) be in the same imputation class as
    the recipient and (b) have a non-missing value for target_col. If no donor
    exists in a class (all class members are missing), fall back to the
    observed column mean.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset. Must contain class_col and target_col.
    target_col : str
        Column with missing values.
    class_col : str
        Column of integer imputation class labels.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    pd.Series
        Imputed series with donor index stored in the series name attribute.
    """
    rng = np.random.default_rng(random_state)
    result = df[target_col].copy()
    donor_ids = pd.Series(index=df.index, dtype="object")

    missing_idx = df[target_col].isna()
    for idx in df.index[missing_idx]:
        cls = df.loc[idx, class_col]
        donors = df.loc[
            (df[class_col] == cls) & df[target_col].notna(), target_col
        ]
        if len(donors) == 0:
            # Fallback: use column mean when class has no donors
            result[idx] = df[target_col].mean()
            donor_ids[idx] = "fallback_mean"
        else:
            chosen_idx = rng.choice(donors.index)
            result[idx] = donors[chosen_idx]
            donor_ids[idx] = chosen_idx

    result.attrs["donor_ids"] = donor_ids
    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def plot_hot_deck_results(df: pd.DataFrame) -> None:
    """
    Two-panel figure:
      Left  -- imputed distribution vs. observed
      Right -- true vs. imputed scatter (shows realistic spread from donors)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(
        df.loc[~df["missing"], "income_obs"],
        bins=40, alpha=0.5, color="steelblue", density=True, label="Observed",
    )
    ax.hist(
        df.loc[df["missing"], "income_hotdeck"],
        bins=40, alpha=0.5, color="seagreen", density=True, label="Imputed (hot-deck)",
    )
    ax.set_xlabel("Income ($)")
    ax.set_title("Hot-deck: imputed distribution mirrors observed")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_hotdeck"],
        alpha=0.5, s=20, color="seagreen",
    )
    lims = [0, 150_000]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("True income ($)")
    ax.set_ylabel("Imputed income ($)")
    ax.set_title("Hot-deck: realistic spread (each point is a real donor value)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("04_hot_deck.png", dpi=120, bbox_inches="tight")
    plt.show()


def show_cluster_summary(df: pd.DataFrame) -> None:
    """Print imputation class sizes and mean characteristics."""
    summary = df.groupby("imp_class").agg(
        n=("age", "count"),
        n_missing=("missing", "sum"),
        mean_age=("age", "mean"),
        mean_educ=("educ", "mean"),
        mean_hours=("hours_wk", "mean"),
        true_mean_income=("income_true", "mean"),
    ).round(1)
    print("\nImputation class sizes and mean characteristics:")
    print("-" * 65)
    print(summary.to_string())


def demonstrate_traceability(df: pd.DataFrame, donor_ids: pd.Series) -> None:
    """
    Show that each imputed value is traceable to a specific donor record --
    a key auditability property of hot-deck imputation.
    """
    missing_idx = df.index[df["missing"]]
    sample = missing_idx[:5]
    print("\nDonor traceability (first 5 imputed records):")
    print(f"{'Recipient idx':>14} {'Donor idx':>10} {'Imputed income':>15} {'Donor income':>13}")
    print("-" * 58)
    for idx in sample:
        donor = donor_ids[idx]
        if donor == "fallback_mean":
            print(f"{idx:>14} {'(mean fallback)':>10} {df.loc[idx, 'income_hotdeck']:>14,.0f} {'N/A':>13}")
        else:
            print(
                f"{idx:>14} {int(donor):>10} "
                f"${df.loc[idx, 'income_hotdeck']:>13,.0f} "
                f"${df.loc[int(donor), 'income_obs']:>12,.0f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Chapter 7: Hot-Deck Imputation ===\n")

    df = pd.read_csv(INPUT_FILE)
    df["missing"] = df["missing"].astype(bool)

    # Encode region as numeric for clustering
    region_map = {"Northeast": 0, "Midwest": 1, "South": 2, "West": 3}
    df["region_num"] = df["region"].map(region_map)

    feature_cols = ["age", "educ", "hours_wk", "region_num"]

    print(f"Building {N_CLASSES} imputation classes via KMeans ...")
    clusters, km_model, scaler = build_imputation_classes(
        df, feature_cols, n_classes=N_CLASSES
    )
    df["imp_class"] = clusters
    show_cluster_summary(df)

    print("\nRunning hot-deck imputation ...")
    df["income_hotdeck"] = hot_deck_impute(df, "income_obs", "imp_class")
    donor_ids = df["income_hotdeck"].attrs.get("donor_ids", pd.Series(dtype="object"))

    mae_hd = mean_absolute_error(
        df.loc[df["missing"], "income_true"],
        df.loc[df["missing"], "income_hotdeck"],
    )
    print(f"\nHot-deck MAE: ${mae_hd:,.0f}")

    true_var = df["income_true"].var()
    hd_var = df["income_hotdeck"].var()
    print(f"Variance ratio (hot-deck / true): {hd_var / true_var:.3f}")
    print("  (1.00 = perfect preservation; hot-deck should be close)")

    demonstrate_traceability(df, donor_ids)

    plot_hot_deck_results(df)
    print("\nFigure saved to 04_hot_deck.png")

    print("\nKey properties of hot-deck imputation:")
    print("  - Every imputed value is a real observed income from a real respondent")
    print("  - Individual values are traceable to specific donors (auditable)")
    print("  - Preserves the empirical distribution (no impossible values)")
    print("  - Handles categorical variables without modeling assumptions")
    print("  - Limitation: does not use continuous predictor variation within class")
