"""
09_activity.py -- Chapter 12: In-Class Activity Starter Code

Starter code for the in-class activity. Students complete the TODO items.

Tasks:
1. Identify the two worst-performing sectors and examine the miscoded examples
2. Write a revised prompt for those sectors with targeted few-shot examples
3. Analyze threshold sensitivity: find the threshold that hits 95% accuracy
   while maximizing automation rate

Run this script, examine the partial output, then complete the TODOs.
Standalone: no external data files required. Run with Python 3.9+.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from _shared_data import (  # noqa: E402
    get_full_eval_df, SECTORS, SECTOR_CODES,
)

NAICS_SECTORS = """11 - Agriculture, Forestry, Fishing and Hunting
21 - Mining, Quarrying, Oil and Gas Extraction
22 - Utilities
23 - Construction
31-33 - Manufacturing
42 - Wholesale Trade
44-45 - Retail Trade
48-49 - Transportation and Warehousing
51 - Information
52 - Finance and Insurance
53 - Real Estate and Rental and Leasing
54 - Professional, Scientific, and Technical Services
55 - Management of Companies and Enterprises
56 - Administrative and Support and Waste Management Services
61 - Educational Services
62 - Health Care and Social Assistance
71 - Arts, Entertainment, and Recreation
72 - Accommodation and Food Services
81 - Other Services (except Public Administration)
92 - Public Administration"""


if __name__ == "__main__":
    df_eval = get_full_eval_df()

    # -----------------------------------------------------------------------
    # TASK 1: Find worst-performing sectors
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("TASK 1: WORST-PERFORMING SECTORS")
    print("=" * 65)

    sector_acc = []
    for code in SECTOR_CODES:
        mask = (df_eval["human_sector"] == code) & \
               (df_eval["llm_sector"] != "UNCLEAR")
        sub = df_eval[mask]
        if len(sub) == 0:
            continue
        acc = (sub["llm_sector"] == code).mean()
        sector_acc.append({"code": code, "name": SECTORS[code],
                            "n": len(sub), "accuracy": acc})

    acc_df = pd.DataFrame(sector_acc).sort_values("accuracy").reset_index(drop=True)
    print("\nAll sectors sorted by accuracy (lowest first):")
    print(acc_df[["code", "name", "n", "accuracy"]].to_string(index=False))

    # Show miscoded examples for the two worst sectors
    for rank in range(2):
        worst_code = acc_df.iloc[rank]["code"]
        worst_name = acc_df.iloc[rank]["name"]
        miscoded = df_eval[
            (df_eval["human_sector"] == worst_code) &
            (df_eval["llm_sector"] != worst_code) &
            (df_eval["llm_sector"] != "UNCLEAR")
        ]
        print(f"\nMiscoded examples for {worst_code} ({worst_name}):")
        for _, row in miscoded.head(6).iterrows():
            print(f"  Human: {row['human_sector']:6s} | LLM: {row['llm_sector']:6s} | "
                  f"'{row['description']}'")

    # -----------------------------------------------------------------------
    # TASK 2: Write a revised prompt for the worst-performing sector
    # -----------------------------------------------------------------------
    print()
    print("=" * 65)
    print("TASK 2: REVISED PROMPT FOR WORST-PERFORMING SECTOR")
    print("=" * 65)

    print(f"\nTarget sector: {acc_df.iloc[0]['code']} ({acc_df.iloc[0]['name']})")
    print()
    print("Starter prompt template (zero-shot):")
    print("-" * 65)

    def build_zero_shot_prompt(description):
        """Zero-shot coding prompt. Students: revise this to add few-shot examples."""
        return (
            f"You are an expert industry coder for a federal statistical agency.\n"
            f"Assign the most appropriate NAICS 2-digit sector code to this description.\n"
            f"Respond ONLY with: XX - Sector Name\n\n"
            f"NAICS 2-digit sectors:\n{NAICS_SECTORS}\n\n"
            f'Description: "{description}"\n'
            f"Code: "
        )

    sample_desc = acc_df.iloc[0]["name"]
    # Pick a miscoded example from the worst sector for the prompt demo
    worst_code = acc_df.iloc[0]["code"]
    miscoded_examples = df_eval[
        (df_eval["human_sector"] == worst_code) &
        (df_eval["llm_sector"] != worst_code)
    ].head(1)
    if len(miscoded_examples) > 0:
        demo_desc = miscoded_examples.iloc[0]["description"]
    else:
        demo_desc = f"Example description from {sample_desc}"

    print(build_zero_shot_prompt(demo_desc))
    print("-" * 65)

    # TODO: students add few-shot examples here
    print()
    print("TODO: Add few-shot examples below to help the model distinguish")
    print(f"      {worst_code} ({acc_df.iloc[0]['name']}) from the sectors")
    print("      it most often confuses it with.")
    print()
    print("  few_shot_examples = [")
    print("      # TODO: add 2-3 (description, correct_code) pairs")
    print("      # Choose examples that highlight the distinguishing features")
    print("      # of this sector vs. the sectors it gets confused with")
    print("  ]")
    print()
    print("  After writing the examples, check: does your prompt address the")
    print("  root cause of the errors, or just list more cases of the correct sector?")

    # -----------------------------------------------------------------------
    # TASK 3: Threshold sensitivity analysis
    # -----------------------------------------------------------------------
    print()
    print("=" * 65)
    print("TASK 3: THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 65)

    thresholds = np.linspace(0.40, 0.97, 50)
    threshold_results = []
    for thresh in thresholds:
        auto_mask = (df_eval["confidence"] >= thresh) & \
                    (df_eval["llm_sector"] != "UNCLEAR")
        auto_df = df_eval[auto_mask]
        auto_rate = auto_mask.mean()
        auto_acc = (
            (auto_df["human_sector"] == auto_df["llm_sector"]).mean()
            if len(auto_df) > 0 else 1.0
        )
        threshold_results.append({
            "threshold": thresh,
            "auto_rate": auto_rate,
            "auto_accuracy": auto_acc,
            "n_auto": len(auto_df),
            "n_human_review": (~auto_mask).sum(),
        })

    thresh_df = pd.DataFrame(threshold_results)

    # TODO: find the threshold that achieves 95% accuracy
    # with the highest automation rate
    target_accuracy = 0.95
    print(f"\nFinding confidence threshold for {target_accuracy:.0%} auto-coding accuracy...")
    print()

    achievable = thresh_df[thresh_df["auto_accuracy"] >= target_accuracy]
    if len(achievable) > 0:
        best = achievable.iloc[-1]
        print(f"  Best threshold:    {best['threshold']:.2f}")
        print(f"  Auto-coded:        {best['auto_rate']:.1%} ({int(best['n_auto'])} records)")
        print(f"  Human review:      {1 - best['auto_rate']:.1%} "
              f"({int(best['n_human_review'])} records)")
        print(f"  Accuracy on auto:  {best['auto_accuracy']:.1%}")
    else:
        print(f"  {target_accuracy:.0%} accuracy not achievable in threshold sweep.")
        print("  Consider: more targeted prompts, or lower the accuracy target.")

    # TODO: plot the accuracy-automation curve
    print()
    print("TODO: Uncomment the block below to plot the tradeoff curve")
    print()
    print("  # --- Uncomment to plot ---")
    print("  # fig, ax = plt.subplots(figsize=(8, 5))")
    print("  # ax.plot(thresh_df['auto_accuracy'] * 100,")
    print("  #         thresh_df['auto_rate'] * 100,")
    print("  #         color='seagreen', linewidth=2)")
    print("  # ax.axvline(95, color='gray', linestyle='--', label='95% target')")
    print("  # ax.set_xlabel('Auto-coded accuracy (%)')")
    print("  # ax.set_ylabel('Auto-coded share (%)')")
    print("  # ax.set_title('Accuracy-automation tradeoff')")
    print("  # ax.legend()")
    print("  # plt.tight_layout()")
    print("  # plt.show()")

    # Extension prompt
    print()
    print("=" * 65)
    print("EXTENSION QUESTIONS")
    print("=" * 65)
    print()
    print("  1. How would accuracy change at the 3-digit NAICS subsector level?")
    print("     What additional information in the description would help?")
    print()
    print("  2. If you had to deploy this system with a budget for only 500 human")
    print("     review hours per week, and each review takes 3 minutes on average,")
    print("     how many records per week could the hybrid system process?")
    print()
    print("  3. The model performs poorly on Other Services (81). Look at the")
    print("     miscoded examples. What property of these descriptions makes them")
    print("     hard -- and how would you address that in a revised prompt?")
    print()
    print("  4. The evaluation dataset uses descriptions from the CPS-style surveys.")
    print("     How would performance likely differ for Economic Census descriptions")
    print("     (which are business self-descriptions, not worker self-descriptions)?")
