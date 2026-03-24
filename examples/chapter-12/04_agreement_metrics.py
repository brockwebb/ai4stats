"""
04_agreement_metrics.py -- Chapter 12: Agreement Metrics for LLM Evaluation

Demonstrates how to compute accuracy and Cohen's kappa for LLM coding
evaluation, and how to compare LLM-human agreement against human-human
inter-coder reliability as a baseline.

This script:
- Loads the simulated evaluation dataset (see 03_evaluation_dataset.py)
- Computes overall accuracy and Cohen's kappa
- Produces a per-sector accuracy bar chart
- Simulates a human-human baseline (~90% pairwise, ~0.82 kappa) for comparison
- Prints a results summary table

Standalone: no external data files required. Run with Python 3.9+.
No LLM API calls are made.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from _shared_data import (  # noqa: E402
    get_full_eval_df, SECTORS, SECTOR_CODES, RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Human-human baseline simulation
# Target: ~90% pairwise accuracy, ~0.82 kappa
# Based on published inter-coder reliability for NAICS 2-digit coding
# Published human-human kappa range: 0.8-0.9+ for broad groupings (Elias et al. 2007)
# ---------------------------------------------------------------------------

def simulate_human2(df, seed=RANDOM_SEED + 77):
    """
    Simulate a second human coder with ~90% agreement with coder 1.
    Disagreements follow the same adjacent-sector patterns as LLM errors
    because those cases are genuinely ambiguous for human coders too.
    """
    rng = np.random.default_rng(seed)
    codes2 = []
    for _, row in df.iterrows():
        true_code = row["human_sector"]
        if rng.random() < 0.90:  # Published human-human kappa range: 0.8-0.9+ for broad groupings (Elias et al. 2007)
            codes2.append(true_code)
        else:
            if true_code in ["54", "51", "52"]:
                codes2.append(rng.choice(["54", "51", "52"]))
            elif true_code in ["62", "81"]:
                codes2.append("62" if true_code == "81" else "81")
            elif true_code in ["61", "92"]:
                codes2.append("61" if true_code == "92" else "92")
            else:
                candidates = [c for c in SECTOR_CODES if c != true_code]
                codes2.append(rng.choice(candidates))
    return codes2


if __name__ == "__main__":
    df_eval = get_full_eval_df()

    # Filter out UNCLEAR for standard agreement metrics (handle separately)
    df_coded = df_eval[df_eval["llm_sector"] != "UNCLEAR"].copy()
    n_unclear = (df_eval["llm_sector"] == "UNCLEAR").sum()

    overall_acc = accuracy_score(df_coded["human_sector"], df_coded["llm_sector"])
    kappa = cohen_kappa_score(df_coded["human_sector"], df_coded["llm_sector"])

    print("=" * 60)
    print("OVERALL LLM CODING PERFORMANCE")
    print("=" * 60)
    print(f"Total descriptions:     {len(df_eval)}")
    print(f"Unclear / refused:      {n_unclear} ({n_unclear/len(df_eval):.1%})")
    print(f"Coded descriptions:     {len(df_coded)}")
    print(f"Overall accuracy:       {overall_acc:.3f} ({overall_acc:.1%})")
    print(f"Cohen's kappa:          {kappa:.3f}")
    print()
    print("Kappa interpretation:")
    kappa_bands = [
        (0.81, 1.00, "Almost perfect"),
        (0.61, 0.80, "Substantial"),
        (0.41, 0.60, "Moderate"),
        (0.21, 0.40, "Fair"),
        (0.00, 0.20, "Slight (near chance)"),
    ]
    for lo, hi, label in kappa_bands:
        marker = " <-- our kappa" if lo <= kappa <= hi else ""
        print(f"  {lo:.2f}-{hi:.2f}: {label}{marker}")

    # Per-sector accuracy
    print()
    print("Per-sector accuracy:")
    print(f"  {'Code':<8} {'Sector Name':<44} {'N':>4}  {'Acc':>5}")
    print("  " + "-" * 65)
    sector_results = []
    for code in SECTOR_CODES:
        mask = df_coded["human_sector"] == code
        if mask.sum() == 0:
            continue
        n = mask.sum()
        correct = (df_coded.loc[mask, "llm_sector"] == code).sum()
        acc = correct / n
        sector_results.append({
            "code": code, "name": SECTORS[code],
            "n": n, "correct": correct, "accuracy": acc,
        })
        flag = " *" if acc < 0.80 else ""
        bar = "#" * int(acc * 25)
        print(f"  {code:<8} {SECTORS[code]:<44} {n:>4}  {acc:.2f}  {bar}{flag}")
    print("  (* = below 80% threshold)")

    sector_df = pd.DataFrame(sector_results).sort_values("accuracy")

    # Per-sector bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tomato" if r < 0.80 else "steelblue" for r in sector_df["accuracy"]]
    bars = ax.barh(sector_df["name"], sector_df["accuracy"],
                   color=colors, edgecolor="gray")
    ax.axvline(overall_acc, color="black", linestyle="--", linewidth=1.5,
               label=f"Overall accuracy = {overall_acc:.2f}")
    ax.axvline(0.80, color="gray", linestyle=":", linewidth=1,
               label="80% common operational threshold")
    ax.set_xlabel("Accuracy")
    ax.set_title("LLM industry coding accuracy by NAICS sector (simulated)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.10)
    for bar_obj, row in zip(bars, sector_df.itertuples()):
        ax.text(row.accuracy + 0.01,
                bar_obj.get_y() + bar_obj.get_height() / 2,
                f"{row.accuracy:.0%} (n={row.n})",
                va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),
                             "04_per_sector_accuracy.png"),
                dpi=120, bbox_inches="tight")
    print()
    print("  Chart saved: 04_per_sector_accuracy.png")
    plt.show()

    # Human-human baseline comparison
    human2_codes = simulate_human2(df_eval)
    hh_acc = accuracy_score(df_eval["human_sector"], human2_codes)
    hh_kappa = cohen_kappa_score(df_eval["human_sector"], human2_codes)

    print()
    print("=" * 60)
    print("LLM-HUMAN vs. HUMAN-HUMAN COMPARISON")
    print("=" * 60)
    print(f"  {'Comparison':<30} {'Accuracy':>10}  {'Kappa':>8}")
    print("  " + "-" * 52)
    print(f"  {'Human-Human (two coders)':<30} {hh_acc:>10.3f}  {hh_kappa:>8.3f}")
    print(f"  {'LLM-Human (simulated)':<30} {overall_acc:>10.3f}  {kappa:>8.3f}")
    gap_acc = hh_acc - overall_acc
    gap_kappa = hh_kappa - kappa
    print()
    print(f"  Gap (human-human minus LLM): "
          f"accuracy = {gap_acc:+.3f},  kappa = {gap_kappa:+.3f}")
    print()
    print("  LLM accuracy approaches but does not match human-human agreement.")
    print("  The gap widens considerably at 6-digit NAICS (more specific coding).")
    print("  Human-human ~90% pairwise / ~0.82 kappa represents the practical ceiling for this task.")

    # Comparison chart
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    acc_vals = [hh_acc, overall_acc]
    kap_vals = [hh_kappa, kappa]
    clrs = ["steelblue", "darkorange"]
    lbls = ["Human-Human\n(two trained coders)", "LLM-Human\n(simulated)"]

    bars1 = ax2.bar(x - 0.2, acc_vals, 0.35, label="Accuracy",
                    color=clrs, alpha=0.85)
    bars2 = ax2.bar(x + 0.2, kap_vals, 0.35, label="Cohen's kappa",
                    color=clrs, alpha=0.5, hatch="///")
    ax2.set_xticks(x)
    ax2.set_xticklabels(lbls, fontsize=10)
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Coding agreement: LLM vs. human-human (simulated)")
    ax2.legend()
    ax2.axhline(0.80, color="gray", linestyle=":", linewidth=1)
    for b, v in zip(list(bars1) + list(bars2), acc_vals + kap_vals):
        ax2.text(b.get_x() + b.get_width() / 2,
                 b.get_height() + 0.01,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__),
                             "04_human_llm_comparison.png"),
                dpi=120, bbox_inches="tight")
    print("  Chart saved: 04_human_llm_comparison.png")
    plt.show()
