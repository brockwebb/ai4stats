"""
05_radar_chart.py
Chapter 14: Evaluating AI Systems for Federal Use

radar_chart() function and hypothetical vendor scoring visualization.
Scores a fictional AI system on the 10-dimension rubric to illustrate
how gaps show up visually. Includes score summary output.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
np.random.seed(2025)

FIGURE_SIZE = (7, 7)
DIMENSION_LABELS = [
    "Task fit", "Accuracy", "Reproduc.", "Docs",
    "Failures", "Oversight", "Data gov.", "Fairness",
    "Updates", "SFV",
]

# Hypothetical vendor system: strong on accuracy, weak on documentation,
# SFV, fairness, and reproducibility. Illustrates the evaluation gap.
VENDOR_SCORES = [2, 3, 1, 1, 1, 2, 2, 1, 1, 0]

SCORE_LABELS = {0: "Missing", 1: "Minimum", 2: "Adequate", 3: "Best practice"}


def radar_chart(scores, title, labels=None, color="steelblue"):
    """
    Create a radar/spider chart for rubric scores.

    Parameters
    ----------
    scores : list of int
        Score for each dimension, 0 (missing) to 3 (best practice).
    title : str
        Chart title.
    labels : list of str, optional
        Short label for each axis. Defaults to D1, D2, ...
    color : str
        Line and fill color.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if labels is None:
        labels = [f"D{i + 1}" for i in range(len(scores))]

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
    ax.set_yticklabels(
        ["1\n(Minimum)", "2\n(Adequate)", "3\n(Best\npractice)"], size=7
    )
    ax.set_title(title, size=11, pad=20)
    return fig


def print_score_summary(scores, labels):
    total = sum(scores)
    max_score = 3 * len(scores)
    print(f"Total score: {total}/{max_score} ({total / max_score:.0%})")
    print()
    missing = [labels[i] for i, s in enumerate(scores) if s == 0]
    minimum = [labels[i] for i, s in enumerate(scores) if s == 1]
    adequate = [labels[i] for i, s in enumerate(scores) if s == 2]
    best = [labels[i] for i, s in enumerate(scores) if s == 3]
    print(f"Score 0 (missing):       {missing or 'none'}")
    print(f"Score 1 (minimum only):  {minimum or 'none'}")
    print(f"Score 2 (adequate):      {adequate or 'none'}")
    print(f"Score 3 (best practice): {best or 'none'}")
    print()
    print("Conclusion: Not recommended for deployment without addressing")
    print("documentation, failure mode analysis, bias testing,")
    print("update protocol, and SFV requirements.")


if __name__ == "__main__":
    fig = radar_chart(
        VENDOR_SCORES,
        title=(
            "Hypothetical Vendor AI System: Evaluation Profile\n"
            "(0=Missing, 1=Minimum, 2=Adequate, 3=Best practice)"
        ),
        labels=DIMENSION_LABELS,
    )
    plt.tight_layout()
    plt.show()

    print_score_summary(VENDOR_SCORES, DIMENSION_LABELS)
