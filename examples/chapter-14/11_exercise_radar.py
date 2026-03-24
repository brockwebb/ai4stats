"""
11_exercise_radar.py
Chapter 14: Evaluating AI Systems for Federal Use

Exercise B.1 starter code: student radar chart for AutoCode Pro 2.0.
Students fill in their own scores based on their rubric analysis.

TODO: Replace the placeholder scores below with your analysis.
      Each score should be 0 (missing), 1 (minimum), 2 (adequate),
      or 3 (best practice), with supporting evidence from the one-pager.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
np.random.seed(2025)

DIMENSION_LABELS = [
    "Task fit", "Accuracy", "Reproduc.", "Docs",
    "Failures", "Oversight", "Data gov.", "Fairness",
    "Updates", "SFV",
]

# TODO: Replace these placeholder scores with your own analysis of AutoCode Pro 2.0.
# Score each dimension 0-3 using the rubric in examples/chapter-14/04_evaluation_rubric.py.
# Record your evidence for each score before producing the chart.
EXERCISE_SCORES = [2, 1, 0, 0, 1, 0, 0, 0, 0, 0]  # PLACEHOLDER -- update these

FIGURE_SIZE = (7, 7)
CHART_COLOR = "tomato"
CHART_TITLE = "Exercise B.1: Your AutoCode Pro 2.0 Evaluation\n(Update scores above)"


def radar_chart(scores, title, labels, color="tomato"):
    """
    Create a radar/spider chart for rubric scores.

    Parameters
    ----------
    scores : list of int
        Score 0-3 for each dimension. Length must match labels.
    title : str
        Chart title.
    labels : list of str
        Short axis labels.
    color : str
        Line and fill color.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(scores)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    scores_plot = list(scores) + [scores[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, subplot_kw=dict(polar=True))
    ax.plot(angles, scores_plot, "o-", linewidth=2, color=color)
    ax.fill(angles, scores_plot, alpha=0.2, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 3)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Minimum", "Adequate", "Best\npractice"], size=8)
    ax.set_title(title, size=10, pad=20)
    return fig


def print_score_summary(scores):
    total = sum(scores)
    max_score = 3 * len(scores)
    pct = total / max_score
    print(f"Total score: {total}/{max_score} ({pct:.0%})")
    print()

    for i, (label, score) in enumerate(zip(DIMENSION_LABELS, scores)):
        label_str = {0: "MISSING", 1: "MINIMUM", 2: "ADEQUATE", 3: "BEST PRACTICE"}
        print(f"  {i+1:>2}. {label:<12} {score}/3  [{label_str[score]}]")

    print()
    print("TODO: For each dimension, record:")
    print("  - What evidence from the one-pager supports your score?")
    print("  - What additional information would you need to raise the score?")
    print("  - What is the single most important question to ask the vendor?")


if __name__ == "__main__":
    fig = radar_chart(
        EXERCISE_SCORES,
        title=CHART_TITLE,
        labels=DIMENSION_LABELS,
        color=CHART_COLOR,
    )
    plt.tight_layout()
    plt.show()

    print_score_summary(EXERCISE_SCORES)
