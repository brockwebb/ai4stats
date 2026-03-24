"""
08_differential_privacy.py
==========================
Chapter 9: Synthetic Data Generation for Federal Statistics

Demonstrates the Laplace mechanism for differential privacy and shows how
epsilon controls the privacy-accuracy tradeoff for aggregate statistics.
Contextualizes this against the 2020 Census Disclosure Avoidance System.

Why this matters:
    Differential privacy provides a formal mathematical guarantee about how
    much information a single person's data contributes to a published
    statistic. The 2020 Census was the first major production use of DP at
    national scale, and it generated significant controversy about small-area
    accuracy. Federal statisticians need to understand what epsilon means
    in practice, not just as a symbol.

The Laplace mechanism:
    To privately release a statistic with sensitivity S (the most any one
    person could change it), add noise drawn from Laplace(0, S/epsilon).
    Lower epsilon = stronger privacy = more noise.

2020 Census context:
    The Census Bureau allocated a total privacy budget of approximately
    17.14 epsilon across all geographic levels. Block-level counts received
    relatively more noise than state totals. Critics argued this made
    redistricting data unreliable for small geographies.

Usage:
    python 08_differential_privacy.py

Outputs:
    - Noise distribution figure for epsilon = 0.1, 1.0, 10.0
    - Accuracy vs. epsilon table printed to stdout
    - dp_laplace_mechanism.png saved to current directory

Requirements:
    Python 3.9+, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def laplace_noise(sensitivity: float, epsilon: float, n: int, seed: int = 2025) -> np.ndarray:
    """
    Draw n samples from Laplace(0, sensitivity/epsilon).

    Parameters
    ----------
    sensitivity : float
        The global sensitivity of the query (max change from one person's data).
    epsilon : float
        Privacy budget. Lower = stronger privacy = more noise.
    n : int
        Number of noise draws.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray of shape (n,)
    """
    rng = np.random.default_rng(seed)
    scale = sensitivity / epsilon
    return rng.laplace(0, scale, n)


def print_accuracy_table(true_value: float, sensitivity: float, n_trials: int = 5000) -> None:
    """
    Show how epsilon affects the accuracy of a privately released count.

    For a given true count, compute the mean absolute error and 95th percentile
    error across many draws of Laplace noise.
    """
    epsilon_values = [0.1, 0.5, 1.0, 5.0, 10.0, 17.14]

    print("Accuracy vs. epsilon tradeoff (Laplace mechanism)")
    print(f"True count: {true_value:,.0f}  |  Sensitivity: {sensitivity}  |  n_trials: {n_trials:,}")
    print("=" * 68)
    print(f"{'Epsilon':>8} {'Noise Scale':>12} {'Mean Abs Error':>16} {'95th pct Error':>16}")
    print("-" * 68)

    for eps in epsilon_values:
        noise = laplace_noise(sensitivity, eps, n_trials, seed=2025)
        mae = np.mean(np.abs(noise))
        p95 = np.percentile(np.abs(noise), 95)
        scale = sensitivity / eps
        note = " <-- 2020 Census total budget" if abs(eps - 17.14) < 0.01 else ""
        print(f"{eps:>8.2f} {scale:>12.2f} {mae:>16,.1f} {p95:>16,.1f}{note}")

    print()
    print("Interpretation:")
    print("  At epsilon=0.1, the noise scale is 10x the sensitivity.")
    print("  A true count of 50,000 might be reported as anywhere in a wide range.")
    print("  At epsilon=10.0, the noise is small relative to the true count.")
    print()
    print("2020 Census DAS context:")
    print("  Total privacy budget: ~17.14 epsilon across all geographic levels.")
    print("  Block-level redistricting data received substantial noise allocation.")
    print("  States challenged the data quality for small-area counts.")
    print("  The Bureau argued this was the first system with formal privacy guarantees.")


def plot_noise_distributions(
    true_value: float,
    sensitivity: float,
    output_path: str = "dp_laplace_mechanism.png",
) -> None:
    """
    Plot noise distributions for three epsilon values on the same axis.

    Shows that smaller epsilon produces a wider distribution — less accurate
    but more private.
    """
    epsilon_values = [0.1, 1.0, 10.0]
    colors = ["tomato", "darkorange", "steelblue"]
    n_draws = 5000

    fig, ax = plt.subplots(figsize=(10, 5))

    for eps, color in zip(epsilon_values, colors):
        noise = laplace_noise(sensitivity, eps, n_draws, seed=2025)
        noisy_counts = true_value + noise
        ax.hist(
            noisy_counts,
            bins=80,
            alpha=0.45,
            density=True,
            color=color,
            label=f"epsilon={eps}  (noise scale={sensitivity/eps:.1f})",
        )

    ax.axvline(
        true_value, color="black", linewidth=2, linestyle="--",
        label=f"True count ({true_value:,.0f})",
    )
    ax.set_xlabel("Differentially private count")
    ax.set_title("Laplace mechanism: smaller epsilon = wider noise = more privacy")
    ax.legend(fontsize=9)
    # Center the x-axis on the true value
    spread = sensitivity / 0.1 * 4
    ax.set_xlim(true_value - spread, true_value + spread)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Example: count of individuals with income > $75,000 in a county
    TRUE_COUNT = 50_000
    SENSITIVITY = 1  # adding or removing one person changes the count by at most 1

    print("Differential Privacy: Laplace Mechanism Demonstration")
    print("=" * 55)
    print()

    print_accuracy_table(TRUE_COUNT, SENSITIVITY)
    print()

    plot_noise_distributions(TRUE_COUNT, SENSITIVITY)

    print()
    print("Key distinction from synthesis:")
    print("  The Laplace mechanism adds noise to PUBLISHED STATISTICS,")
    print("  not to individual records. The 2020 Census DAS injected noise")
    print("  at the histogram cell level, then reconstructed consistent records.")
    print("  This is different from (but related to) sequential regression synthesis.")
