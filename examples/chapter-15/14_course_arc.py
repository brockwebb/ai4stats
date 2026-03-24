"""
Chapter 15: Course arc table and wheel visualization.
Groups chapters by theme, shows SFV connections per group,
and produces the polar course arc chart.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from difflib import SequenceMatcher
import textwrap

np.random.seed(42)

# Course arc visualization

chapters = [
    {"group": "Foundations", "chapters": "1-4", "theme": "Knowing what you are working with",
     "methods": "Python, Pandas, Census APIs, Record Keys", "sfv_connection": "Data provenance; unit of analysis clarity"},
    {"group": "Core Methods", "chapters": "5-9", "theme": "Knowing what tools are available",
     "methods": "Regression, Cross-Validation, Decision Trees, Neural Nets, Graph Methods",
     "sfv_connection": "Method selection rationale: decisions that must survive session boundaries"},
    {"group": "Advanced Methods", "chapters": "11-12", "theme": "Knowing the limits of the tools",
     "methods": "Dimension Reduction, Imputation", "sfv_connection": "Exclusion decisions; imputation method choices with documented rationale"},
    {"group": "Privacy and Synthetic Data", "chapters": "13-14", "theme": "Knowing what is at stake",
     "methods": "Synthetic Data Generation, Differential Privacy, Disclosure Avoidance",
     "sfv_connection": "Privacy budget as a state variable; T4 (State Supersession Failure) directly threatens privacy guarantees"},
    {"group": "Language Models", "chapters": "10, 11", "theme": "Knowing how the instrument works",
     "methods": "Transformers, LLMs for Survey Operations",
     "sfv_connection": "The LLM is the context window; understanding the instrument is precondition for evaluating its fidelity"},
    {"group": "Governance", "chapters": "14", "theme": "Knowing how to assess fitness for use",
     "methods": "NIST AI RMF, FCSM Quality Standards, TEVV, 10-dimension evaluation rubric",
     "sfv_connection": "SFV as dimension 10 of the rubric; TEVV as the measurement framework for SFV metrics"},
    {"group": "Capstone", "chapters": "15", "theme": "Knowing whether your pipeline maintained integrity",
     "methods": "State Fidelity Validity, T1-T5 taxonomy, countermeasures, reproducibility checklist",
     "sfv_connection": "SFV IS this chapter: the validity framework that makes AI-assisted research defensible"},
]

print("Course arc: from 'how do I use these tools' to 'how do I trust the process'")
print("=" * 70)
for lec in chapters:
    print(f"\n{lec['group']} (Chapters {lec['chapters']})")
    print(f"  Theme:  {lec['theme']}")
    print(f"  Methods: {lec['methods'][:65]}")
    wrapped = textwrap.fill(lec['sfv_connection'], width=63, initial_indent="  SFV connection: ", subsequent_indent="                  ")
    print(wrapped)

# Course arc wheel visualization

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

groups = ["Foundations\n(1-4)", "Core Methods\n(5-9)", "Advanced\n(10-12)",
         "Privacy &\nSynthetics\n(13)", "Language\nModels",
         "Governance\n(14)", "Capstone\n(15)"]
n = len(groups)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
angles += angles[:1]  # close the loop

# Conceptual "completion" of the arc -- each builds on prior
values = [0.5, 0.65, 0.72, 0.82, 0.88, 0.93, 1.0]
values += values[:1]

colors_arc = ["#64B5F6", "#42A5F5", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#4CAF50"]

ax.plot(angles, values, "o-", linewidth=2, color="#1565C0", zorder=3)
ax.fill(angles, values, alpha=0.15, color="#1565C0")

ax.set_xticks(angles[:-1])
ax.set_xticklabels(groups, fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["", "", "", "", "Full\ncoverage"], fontsize=8)

ax.set_title("AI for Official Statistics course arc: cumulative methodological coverage\n"
            "Each section builds on all prior sections",
            fontsize=11, pad=20)

# Annotate the capstone
ax.annotate("SFV integrates\nthe full arc",
           xy=(angles[-2], values[-2]),
           xytext=(angles[-2] - 0.4, values[-2] - 0.25),
           fontsize=9, color="#1B5E20",
           arrowprops=dict(arrowstyle="->", color="#1B5E20"))

plt.tight_layout()
plt.savefig("../assets/diagrams/chapter_15_course_arc.png", dpi=120, bbox_inches="tight")
plt.show()

print("The course arc: each section builds on all prior sections.")
print("SFV, introduced in the capstone, applies to any pipeline that uses")
print("the methods from all prior chapters.")
