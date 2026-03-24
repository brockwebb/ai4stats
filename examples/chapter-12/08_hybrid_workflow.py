"""
08_hybrid_workflow.py -- Chapter 12: Hybrid Human-LLM Workflow

Demonstrates confidence-based routing: directing high-confidence LLM outputs
to automatic acceptance and low-confidence outputs to human review. Shows the
accuracy-automation tradeoff curve and analyzes cost implications.

This script:
- Loads the simulated evaluation dataset with confidence scores
- Plots the accuracy-automation tradeoff curve
- Prints threshold analysis table for 90%, 95%, and 98% accuracy targets
- Prints a cost-performance table by model tier
- Provides break-even analysis for LLM vs. human-only coding

Standalone: no external data files required. Run with Python 3.9+.
No LLM API calls are made.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from _shared_data import get_full_eval_df  # noqa: E402

# ---------------------------------------------------------------------------
# Cost-performance table by model tier
# Numbers are illustrative "early 2026 snapshot" -- do NOT treat as current pricing.
# Use these for relative comparisons only; verify with vendors before budgeting.
# ---------------------------------------------------------------------------
# Columns: model_tier, api_cost_per_1k_records, estimated_accuracy,
#          human_review_rate_at_95pct_threshold, effective_cost_per_accepted

COST_TABLE = pd.DataFrame([
    {
        "Model tier": "Large frontier (e.g., GPT-4 class)",
        "API cost / 1K records": "$2.50-$8.00",
        "Est. accuracy (2-digit)": "82-88%",
        "Human review rate (95% target)": "~25-35%",
        "Effective cost / accepted record": "$0.012-$0.040",
    },
    {
        "Model tier": "Mid-size (e.g., GPT-4o-mini class)",
        "API cost / 1K records": "$0.30-$0.80",
        "Est. accuracy (2-digit)": "75-83%",
        "Human review rate (95% target)": "~30-45%",
        "Effective cost / accepted record": "$0.005-$0.015",
    },
    {
        "Model tier": "Small open-source (on-premise, 7-13B)",
        "API cost / 1K records": "~$0.05 (compute only)",
        "Est. accuracy (2-digit)": "65-78%",
        "Human review rate (95% target)": "~40-55%",
        "Effective cost / accepted record": "$0.001-$0.005",
    },
    {
        "Model tier": "Human coder only (no LLM)",
        "API cost / 1K records": "N/A",
        "Est. accuracy (2-digit)": "91-93%",
        "Human review rate (95% target)": "100%",
        "Effective cost / accepted record": "$0.50-$2.00",
    },
])

BREAK_EVEN_NOTES = """
Break-even analysis (illustrative):

  Human coder cost: $0.50-$2.00 per record (training + productivity + QA overhead)
  LLM API cost:     $0.003-$0.008 per record (large frontier model)
  Human review cost after LLM: $0.50-$2.00 per reviewed record

  At 30% human review rate:
    Total cost = API cost + 0.30 * human review cost
               ~ $0.005 + 0.30 * $1.00 = $0.305 per record

  At 100K records:
    Human-only cost:  $100,000 - $200,000
    LLM hybrid cost:  ~$30,500 - $61,000
    Savings:          ~50-70%

  Break-even volume: LLM is cheaper than human-only at virtually any volume
  above ~1,000 records, assuming the required accuracy threshold is achievable.
  The human review cost dominates, not the API cost.

  Latency note: federal production runs are typically batch operations.
  Latency tolerance is hours, not milliseconds. This means:
  - Use batch API endpoints (lower cost)
  - Send all records at once rather than one-by-one
  - Plan for 1-6 hour turnaround per batch, not real-time
  - Throughput matters more than per-call latency for planning
"""


if __name__ == "__main__":
    df_eval = get_full_eval_df()

    # Sweep confidence threshold
    thresholds = np.linspace(0.40, 0.97, 50)
    results = []
    for thresh in thresholds:
        auto_mask = (df_eval["confidence"] >= thresh) & \
                    (df_eval["llm_sector"] != "UNCLEAR")
        auto_df = df_eval[auto_mask]
        human_df = df_eval[~auto_mask]

        auto_rate = auto_mask.mean()
        if len(auto_df) > 0:
            auto_acc = (auto_df["human_sector"] == auto_df["llm_sector"]).mean()
        else:
            auto_acc = 1.0

        results.append({
            "threshold": thresh,
            "auto_rate": auto_rate,
            "auto_accuracy": auto_acc,
            "human_review_rate": 1 - auto_rate,
            "n_auto": len(auto_df),
            "n_human": len(human_df),
        })

    results_df = pd.DataFrame(results)

    # Accuracy-automation tradeoff curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(results_df["threshold"], results_df["auto_rate"] * 100,
            color="steelblue", linewidth=2, label="% auto-coded")
    ax.plot(results_df["threshold"], results_df["human_review_rate"] * 100,
            color="tomato", linewidth=2, label="% sent to human review")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("% of records")
    ax.set_title("Record routing by confidence threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for target in [0.90, 0.95, 0.98]:
        ax.axhline(target * 100, color="gray", linestyle=":", linewidth=0.7)

    ax = axes[1]
    ax.plot(results_df["auto_accuracy"] * 100, results_df["auto_rate"] * 100,
            color="seagreen", linewidth=2, marker="o", markersize=3)
    ax.axvline(95, color="gray", linestyle=":", linewidth=1.2,
               label="95% accuracy target")
    ax.axvline(90, color="lightgray", linestyle=":", linewidth=1.0,
               label="90% accuracy target")
    ax.set_xlabel("Auto-coded accuracy (%)")
    ax.set_ylabel("Auto-coded share (%)")
    ax.set_title("Accuracy-automation tradeoff curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Confidence-Based Routing: Accuracy vs. Automation (simulated)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "08_hybrid_workflow.png"),
        dpi=120, bbox_inches="tight",
    )
    print("Chart saved: 08_hybrid_workflow.png")
    plt.show()

    # Threshold analysis for accuracy targets
    print()
    print("=" * 80)
    print("THRESHOLD ANALYSIS: WHAT CONFIDENCE THRESHOLD ACHIEVES EACH ACCURACY TARGET?")
    print("=" * 80)
    print(f"  {'Target Acc':>12}  {'Threshold':>10}  {'Auto-coded':>11}  "
          f"{'Human Review':>13}  {'N auto':>7}  {'N human':>8}")
    print("  " + "-" * 67)
    for target_acc in [0.90, 0.95, 0.98]:
        achievable = results_df[results_df["auto_accuracy"] >= target_acc]
        if len(achievable) > 0:
            best = achievable.iloc[-1]
            print(
                f"  {target_acc:.0%}        "
                f"  {best['threshold']:>10.2f}"
                f"  {best['auto_rate']:>10.1%}"
                f"  {best['human_review_rate']:>12.1%}"
                f"  {int(best['n_auto']):>7d}"
                f"  {int(best['n_human']):>8d}"
            )
        else:
            print(f"  {target_acc:.0%}         -- not achievable at any threshold in sweep")

    print()
    print("  At N=200 records:")
    print("  These are illustrative numbers. For production planning, run on")
    print("  your full evaluation set (thousands of records per sector).")

    # Cost-performance table
    print()
    print("=" * 80)
    print("COST-PERFORMANCE TABLE BY MODEL TIER (early 2026 illustrative snapshot)")
    print("Verify current pricing with vendors before budgeting.")
    print("=" * 80)
    print(COST_TABLE.to_string(index=False))

    print()
    print(BREAK_EVEN_NOTES)
