"""
Chapter 5 Example 7: Survey Clustering Effects and Geographic Spillover
=======================================================================

Demonstrates two structural effects that arise when survey data has graph
structure and are invisible when respondents are treated as independent rows.

1. Household response contagion
   Within a household, one member's decision to respond influences others.
   This creates intraclass correlation (ICC) that inflates the variance of
   survey estimates. The design effect (DEFF) quantifies this inflation:

       DEFF = 1 + (mean_cluster_size - 1) x ICC

   A DEFF of 1.4 means standard errors are 1.4x larger than an independent
   sample of the same size, so effective sample size is only 1/1.4 ~ 71%
   of the nominal sample.

2. Geographic spillover
   Shocks (policy changes, economic events, natural disasters) spread across
   county/tract boundaries. Treating geographic units as independent ignores
   spatial autocorrelation and understates uncertainty in area-level estimates.

The GroupKFold implication: when splitting data for model evaluation, groups
must be defined at the household or primary sampling unit (PSU) level, not the
person level. Splitting at the person level leaks within-household correlation
into the test set and produces optimistically biased estimates of model accuracy.

Requirements: numpy, pandas, networkx, matplotlib
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Household response contagion
# ---------------------------------------------------------------------------

def simulate_household_response(n_households=300, seed=42):
    """
    Simulate survey response with within-household correlation.

    Each household has a latent response propensity drawn from Beta(4, 2).
    Individual response decisions are correlated with this household-level
    propensity. This is the statistical model underlying design effects in
    clustered samples.

    Parameters
    ----------
    n_households : int
        Number of households to simulate.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: hh_id, hh_size, responded.
    """
    rng = np.random.default_rng(seed)
    household_sizes = rng.choice(
        [1, 2, 3, 4, 5], size=n_households, p=[0.28, 0.35, 0.18, 0.13, 0.06]
    )
    rows = []
    for hh_id, size in enumerate(household_sizes):
        hh_rr_base = rng.beta(4, 2)
        for _ in range(size):
            responded = int(rng.random() < hh_rr_base * rng.beta(5, 1))
            rows.append({"hh_id": hh_id, "hh_size": size, "responded": responded})
    return pd.DataFrame(rows)


def compute_design_effect(df):
    """
    Compute intraclass correlation (ICC) and design effect (DEFF).

    DEFF = 1 + (mean_cluster_size - 1) * ICC

    Parameters
    ----------
    df : pd.DataFrame
        Output of simulate_household_response(). Must have hh_id and responded.

    Returns
    -------
    dict with keys: overall_rr, n_bar, icc, deff
    """
    overall_rr = df["responded"].mean()
    n_bar      = df.groupby("hh_id").size().mean()
    hh_means   = df.groupby("hh_id")["responded"].mean()
    overall_mean = df["responded"].mean()
    between_var  = ((hh_means - overall_mean) ** 2).mean()
    within_var   = df.groupby("hh_id")["responded"].var().mean()
    icc  = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0
    deff = 1 + (n_bar - 1) * icc
    return {"overall_rr": overall_rr, "n_bar": n_bar, "icc": icc, "deff": deff}


def report_household_effects(df, metrics):
    """
    Print household clustering statistics with interpretation.

    Parameters
    ----------
    df : pd.DataFrame
    metrics : dict
        Output of compute_design_effect().
    """
    print("Household response clustering effects")
    print("=" * 40)
    print(f"  Overall response rate:     {metrics['overall_rr']:.3f}")
    print(f"  Mean household size:       {metrics['n_bar']:.2f}")
    print(f"  Intraclass correlation:    {metrics['icc']:.3f}")
    print(f"  Design effect (DEFF):      {metrics['deff']:.3f}")
    effective_n = len(df) / metrics["deff"]
    print(f"\n  Nominal sample size:       {len(df)}")
    print(f"  Effective sample size:     {effective_n:.0f} (= N / DEFF)")
    print(f"\n  Treating persons as independent underestimates standard errors by")
    print(f"  a factor of sqrt({metrics['deff']:.2f}) = {metrics['deff']**0.5:.2f}.")
    print()

    hh_rr = df.groupby("hh_id")["responded"].mean()
    print("Household response rate distribution:")
    print(hh_rr.describe().round(3).to_string())


# ---------------------------------------------------------------------------
# 2. Geographic spillover
# ---------------------------------------------------------------------------

def simulate_geographic_spillover(grid_size=5, shock_center=(2, 2),
                                   decay=0.3, noise_sd=0.05, seed=42):
    """
    Simulate an economic shock that decays with Manhattan distance from its
    epicenter on a grid_size x grid_size county grid.

    Parameters
    ----------
    grid_size : int
        Side length of the county grid.
    shock_center : tuple
        (row, col) of the epicentered county.
    decay : float
        How much the shock attenuates per unit of distance.
    noise_sd : float
        Standard deviation of measurement noise added to each county.
    seed : int
        Random seed.

    Returns
    -------
    nx.Graph
        Grid graph with 'shock' attribute on each node.
    """
    rng = np.random.default_rng(seed)
    G = nx.grid_2d_graph(grid_size, grid_size)
    for node in G.nodes:
        dist = abs(node[0] - shock_center[0]) + abs(node[1] - shock_center[1])
        raw  = max(0, 1.0 - dist * decay) + rng.normal(0, noise_sd)
        G.nodes[node]["shock"] = max(0, raw)
    return G


def visualize_spillover(G):
    """
    Plot the geographic shock intensity on the county grid.

    Warmer colors (yellow -> red) indicate higher shock values. Edge structure
    shows which counties are neighbors. If counties were independent, the
    spatial gradient in shock intensity would be invisible.

    Parameters
    ----------
    G : nx.Graph
        Output of simulate_geographic_spillover().
    """
    pos        = {node: (node[1], -node[0]) for node in G.nodes}
    shock_vals = [G.nodes[n]["shock"] for n in G.nodes]

    fig, ax = plt.subplots(figsize=(7, 6))
    nx.draw_networkx_nodes(G, pos, node_color=shock_vals,
                           cmap="YlOrRd", node_size=1200, ax=ax, vmin=0, vmax=1)
    nx.draw_networkx_labels(
        G, pos,
        labels={n: f"{G.nodes[n]['shock']:.2f}" for n in G.nodes},
        font_size=7, ax=ax
    )
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="gray")
    ax.set_title(
        "Geographic spillover: economic shock decays with distance\n"
        "Treating counties as independent would miss this spatial gradient"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("geographic_spillover.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: geographic_spillover.png")


if __name__ == "__main__":
    # --- Household effects ---
    df      = simulate_household_response(n_households=300, seed=42)
    metrics = compute_design_effect(df)
    report_household_effects(df, metrics)

    # --- Geographic spillover ---
    print("\nGeographic spillover simulation")
    print("=" * 40)
    G_geo = simulate_geographic_spillover(grid_size=5, shock_center=(2, 2))
    shock_vals = [G_geo.nodes[n]["shock"] for n in G_geo.nodes]
    print(f"  Mean shock across all counties: {np.mean(shock_vals):.3f}")
    center_shock = G_geo.nodes[(2, 2)]["shock"]
    corner_shock = G_geo.nodes[(0, 0)]["shock"]
    print(f"  Epicenter county (2,2) shock:   {center_shock:.3f}")
    print(f"  Corner county (0,0) shock:      {corner_shock:.3f}")
    print(f"  Ratio (center/corner):          {center_shock/corner_shock:.1f}x")
    visualize_spillover(G_geo)
