"""
Chapter 5 Example 1: Household Network Construction and Metrics
===============================================================

Builds a household network from a synthetic Census block, computes basic
graph metrics (degree, connected components, clustering coefficient), and
produces a visualization. Demonstrates why treating survey respondents as
independent rows misses the relational structure that affects data quality
and variance estimation.

Audience: Federal statisticians evaluating linked household datasets.

Requirements: networkx, matplotlib, numpy
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def build_household_network():
    """
    Build a small household network for a synthetic Census block.

    Returns
    -------
    nx.Graph
        Graph with person nodes (attributes: hh_id, age, sex, education,
        income) and edges typed as spouse, parent-child, neighbor, or coworker.
    """
    G = nx.Graph()

    persons = [
        ("H001-P01", {"hh_id": "H001", "age": 43, "sex": "F", "education": 16, "income": 65000}),
        ("H001-P02", {"hh_id": "H001", "age": 41, "sex": "M", "education": 14, "income": 52000}),
        ("H001-P03", {"hh_id": "H001", "age": 16, "sex": "F", "education":  9, "income":     0}),
        ("H002-P01", {"hh_id": "H002", "age": 28, "sex": "M", "education": 18, "income": 85000}),
        ("H002-P02", {"hh_id": "H002", "age": 26, "sex": "F", "education": 16, "income": 71000}),
        ("H003-P01", {"hh_id": "H003", "age": 67, "sex": "M", "education": 12, "income": 38000}),
        ("H003-P02", {"hh_id": "H003", "age": 65, "sex": "F", "education": 12, "income": 24000}),
        ("H004-P01", {"hh_id": "H004", "age": 34, "sex": "F", "education": 14, "income": 44000}),
        ("H004-P02", {"hh_id": "H004", "age":  8, "sex": "M", "education":  0, "income":     0}),
        ("H005-P01", {"hh_id": "H005", "age": 52, "sex": "M", "education": 16, "income": 91000}),
    ]
    G.add_nodes_from(persons)

    within_hh_edges = [
        ("H001-P01", "H001-P02", {"rel": "spouse",       "weight": 3}),
        ("H001-P01", "H001-P03", {"rel": "parent-child", "weight": 3}),
        ("H001-P02", "H001-P03", {"rel": "parent-child", "weight": 3}),
        ("H002-P01", "H002-P02", {"rel": "spouse",       "weight": 3}),
        ("H003-P01", "H003-P02", {"rel": "spouse",       "weight": 3}),
        ("H004-P01", "H004-P02", {"rel": "parent-child", "weight": 3}),
    ]
    neighbor_edges = [
        ("H001-P01", "H003-P01", {"rel": "neighbor", "weight": 1}),
        ("H002-P01", "H004-P01", {"rel": "neighbor", "weight": 1}),
        ("H001-P02", "H005-P01", {"rel": "coworker",  "weight": 2}),
    ]
    G.add_edges_from(within_hh_edges + neighbor_edges)
    return G


def compute_graph_metrics(G):
    """
    Print basic graph metrics and interpret them for data quality purposes.

    Parameters
    ----------
    G : nx.Graph
        Household network built by build_household_network().
    """
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    degrees = dict(G.degree())
    print("\nPerson degree (number of connections):")
    for node, deg in sorted(degrees.items(), key=lambda x: -x[1]):
        print(f"  {node}: degree {deg}")

    isolated = list(nx.isolates(G))
    print(f"\nIsolated nodes (no connections): {isolated if isolated else 'none'}")
    print("  Isolated nodes can indicate genuine one-person households or linkage failures.")

    components = list(nx.connected_components(G))
    print(f"\nConnected components: {len(components)}")
    for i, comp in enumerate(components, 1):
        print(f"  Component {i}: {sorted(comp)}")

    # Subgraph metrics for households with relational edges
    hh_nodes = [n for n in G.nodes if G.nodes[n]["hh_id"] in ["H001", "H002", "H003"]]
    G_sub = G.subgraph(hh_nodes)
    print(f"\nHousehold subgraph (H001-H003):")
    print(f"  Average clustering coefficient: {nx.average_clustering(G_sub):.3f}")
    print(f"  Graph density: {nx.density(G_sub):.3f}")
    print("\nInterpretation:")
    print("  High degree -> person appears in many linked records.")
    print("    Could be a true hub (large family) or a quality issue (duplicate merge).")
    print("  Isolated node -> no linked records found.")
    print("    Could be genuinely isolated or a missed match in linkage.")


def visualize_network(G):
    """
    Produce a color-coded visualization of the household network.

    Nodes are colored by household. Edges are dark for within-household
    relationships and light gray for cross-household ties.

    Parameters
    ----------
    G : nx.Graph
        Household network.
    """
    hh_color_map = {
        "H001": "#4472C4",
        "H002": "#ED7D31",
        "H003": "#70AD47",
        "H004": "#FFC000",
        "H005": "#FF0000",
    }
    node_colors = [hh_color_map[G.nodes[n]["hh_id"]] for n in G.nodes]
    edge_colors = [
        "#333333" if G[u][v].get("rel") in ("spouse", "parent-child") else "#AAAAAA"
        for u, v in G.edges
    ]
    edge_widths = [G[u][v].get("weight", 1) * 0.8 for u, v in G.edges]

    pos = nx.spring_layout(G, seed=7, k=1.8)
    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax, alpha=0.7)
    edge_labels = {(u, v): G[u][v]["rel"] for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)

    legend_patches = [
        Patch(color=c, label=f"Household {hh}") for hh, c in hh_color_map.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    ax.set_title(
        "Household network on a Census block\n"
        "(colors = households, edge weight = relationship strength)"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("household_network.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: household_network.png")


if __name__ == "__main__":
    np.random.seed(42)
    G = build_household_network()
    compute_graph_metrics(G)
    visualize_network(G)
