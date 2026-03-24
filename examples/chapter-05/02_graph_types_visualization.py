"""
Chapter 5 Example 2: Graph Type Illustrations
=============================================

Produces side-by-side visualizations of three graph types common in
federal statistics:
  1. Household network (ACS roster data) - undirected, weighted
  2. Geographic containment hierarchy (State -> County -> Tract) - directed acyclic
  3. Employer-employee bipartite graph (LEHD-style data) - bipartite undirected

Each panel illustrates a different structural pattern that emerges from
standard survey and administrative data when analyzed as a graph.

Requirements: networkx, matplotlib
"""

import networkx as nx
import matplotlib.pyplot as plt


def build_household_type_graph():
    """
    Build a small household graph showing family roles.

    Returns
    -------
    tuple
        (G, pos, node_colors) for use in drawing.
    """
    G = nx.Graph()
    hh_nodes = {
        "P001": {"role": "householder", "age": 45},
        "P002": {"role": "spouse",      "age": 42},
        "P003": {"role": "child",       "age": 17},
        "P004": {"role": "child",       "age": 14},
        "P005": {"role": "parent",      "age": 72},
    }
    for pid, attrs in hh_nodes.items():
        G.add_node(pid, **attrs)
    G.add_edges_from([
        ("P001", "P002", {"rel": "spouse"}),
        ("P001", "P003", {"rel": "parent-child"}),
        ("P001", "P004", {"rel": "parent-child"}),
        ("P002", "P003", {"rel": "parent-child"}),
        ("P002", "P004", {"rel": "parent-child"}),
        ("P001", "P005", {"rel": "parent-child"}),
    ])
    pos = nx.spring_layout(G, seed=0)
    role_colors = {
        "householder": "#4472C4",
        "spouse":      "#ED7D31",
        "child":       "#70AD47",
        "parent":      "#FFC000",
    }
    node_colors = [role_colors[G.nodes[n]["role"]] for n in G.nodes]
    return G, pos, node_colors


def build_geographic_hierarchy_graph():
    """
    Build a directed geographic containment hierarchy (State -> County -> Tract).

    Returns
    -------
    tuple
        (G, pos, node_colors) for use in drawing.
    """
    G = nx.DiGraph()
    G.add_nodes_from([
        ("State",     {"level": 0}),
        ("County A",  {"level": 1}),
        ("County B",  {"level": 1}),
        ("Tract 101", {"level": 2}),
        ("Tract 102", {"level": 2}),
        ("Tract 201", {"level": 2}),
    ])
    G.add_edges_from([
        ("State",    "County A"),
        ("State",    "County B"),
        ("County A", "Tract 101"),
        ("County A", "Tract 102"),
        ("County B", "Tract 201"),
    ])
    pos = {
        "State":     (1,   2),
        "County A":  (0.5, 1),
        "County B":  (1.5, 1),
        "Tract 101": (0,   0),
        "Tract 102": (1,   0),
        "Tract 201": (1.5, 0),
    }
    level_colors = {0: "#4472C4", 1: "#ED7D31", 2: "#70AD47"}
    node_colors = [level_colors[G.nodes[n]["level"]] for n in G.nodes]
    return G, pos, node_colors


def build_employer_employee_bipartite():
    """
    Build an employer-employee bipartite graph (LEHD-style).

    Returns
    -------
    tuple
        (G, pos, node_colors) for use in drawing.
    """
    G = nx.Graph()
    employers = ["Firm A", "Firm B", "Firm C"]
    employees = ["Worker 1", "Worker 2", "Worker 3", "Worker 4", "Worker 5"]
    G.add_nodes_from(employers, bipartite=0)
    G.add_nodes_from(employees, bipartite=1)
    G.add_edges_from([
        ("Firm A", "Worker 1"),
        ("Firm A", "Worker 2"),
        ("Firm B", "Worker 2"),
        ("Firm B", "Worker 3"),
        ("Firm B", "Worker 4"),
        ("Firm C", "Worker 4"),
        ("Firm C", "Worker 5"),
    ])
    pos = nx.bipartite_layout(G, employers)
    node_colors = ["#4472C4" if n in employers else "#70AD47" for n in G.nodes]
    return G, pos, node_colors


def plot_all_three():
    """
    Render all three graph types in a single figure and save to disk.
    """
    G_hh,  pos_hh,  nc_hh  = build_household_type_graph()
    G_geo, pos_geo, nc_geo = build_geographic_hierarchy_graph()
    G_ee,  pos_ee,  nc_ee  = build_employer_employee_bipartite()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    nx.draw(G_hh, pos_hh, ax=axes[0], with_labels=True,
            node_color=nc_hh, node_size=800, font_size=8, edge_color="gray", width=1.5)
    axes[0].set_title("Household network\n(ACS roster data)", fontsize=10)

    nx.draw(G_geo, pos_geo, ax=axes[1], with_labels=True,
            node_color=nc_geo, node_size=900, font_size=8,
            arrows=True, arrowsize=15, edge_color="gray", width=1.5)
    axes[1].set_title("Geographic hierarchy\n(containment graph)", fontsize=10)

    nx.draw(G_ee, pos_ee, ax=axes[2], with_labels=True,
            node_color=nc_ee, node_size=900, font_size=8, edge_color="gray", width=1.5)
    axes[2].set_title("Employer-employee bipartite graph\n(LEHD-style data)", fontsize=10)

    plt.suptitle("Graph representations in federal statistics", fontsize=12)
    plt.tight_layout()
    plt.savefig("graph_types.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: graph_types.png")


if __name__ == "__main__":
    plot_all_three()
