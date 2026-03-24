"""
Chapter 5 Example 6: Linkage Graph and Cluster Analysis
========================================================

Applies the trained logistic regression classifier to all candidate pairs,
accepts pairs above a match probability threshold, and builds a linkage graph.

The linkage graph is a bipartite graph where:
  - Source A record IDs form one partition
  - Source B record IDs form the other partition
  - Accepted match edges connect them

Connected components of this graph are called linkage clusters. Each cluster
should represent one real-world entity. Expected cluster sizes:
  - Size 1:  unlinked singleton (no match found)
  - Size 2:  one A record + one B record (ideal)
  - Size 3+: multi-record cluster; warrants manual review

The threshold controls the precision/recall trade-off: raising it reduces
false positives (spurious links) at the cost of more missed matches.

Requirements: numpy, pandas, networkx, matplotlib, scikit-learn
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
_m3 = import_module("03_synthetic_records")
_m4 = import_module("04_blocking")
_m5 = import_module("05_comparison_scoring")

generate_true_records    = _m3.generate_true_records
make_source_a            = _m3.make_source_a
make_source_b            = _m3.make_source_b
generate_candidate_pairs = _m4.generate_candidate_pairs
compute_comparison_features = _m5.compute_comparison_features
train_linkage_classifier    = _m5.train_linkage_classifier

# Match probability threshold. Pairs with predicted probability >= this value
# are accepted as matches.
MATCH_THRESHOLD = 0.70


def build_linkage_graph(comp_df, clf, scaler, source_a, source_b,
                        threshold=MATCH_THRESHOLD):
    """
    Apply classifier to all candidate pairs and build the linkage graph.

    Parameters
    ----------
    comp_df : pd.DataFrame
        Feature vectors for all candidate pairs (from example 5).
    clf : LogisticRegression
        Fitted classifier.
    scaler : StandardScaler
        Fitted scaler (applied to clf).
    source_a : pd.DataFrame
        Source A records (for node registration).
    source_b : pd.DataFrame
        Source B records (for node registration).
    threshold : float
        Minimum match probability to accept a link.

    Returns
    -------
    tuple
        (G_result nx.Graph, comp_df with match_prob/predicted_match columns)
    """
    feature_cols = ["name_sim", "dob_sim", "addr_sim"]
    X_all = scaler.transform(comp_df[feature_cols].values)
    probs = clf.predict_proba(X_all)[:, 1]
    comp_df = comp_df.copy()
    comp_df["match_prob"]       = probs
    comp_df["predicted_match"]  = (probs >= threshold).astype(int)

    accepted = comp_df[comp_df["predicted_match"] == 1]
    tp = (accepted["is_match"] == 1).sum()
    fp = (accepted["is_match"] == 0).sum()
    print(f"Accepted matches (threshold={threshold:.2f}): {len(accepted)}")
    print(f"  True positives:  {tp}")
    print(f"  False positives: {fp}")
    print(f"  Precision: {tp/(tp+fp):.3f}" if (tp+fp) > 0 else "  Precision: N/A")

    # Build bipartite linkage graph
    G = nx.Graph()
    G.add_nodes_from(source_a["record_id"], source="A")
    G.add_nodes_from(source_b["record_id"], source="B")
    for _, row in accepted.iterrows():
        G.add_edge(row["rec_a"], row["rec_b"], prob=row["match_prob"])

    return G, comp_df


def analyze_clusters(G):
    """
    Compute and print connected component (cluster) size distribution.

    Parameters
    ----------
    G : nx.Graph
        Linkage graph from build_linkage_graph().

    Returns
    -------
    list of frozenset
        All connected components, sorted largest first.
    """
    components = list(nx.connected_components(G))
    sizes      = [len(c) for c in components]
    dist       = Counter(sizes)

    print(f"\nCluster size distribution:")
    print(f"{'Size':>6}  {'Count':>6}  {'Interpretation'}")
    print("-" * 55)
    for sz, count in sorted(dist.items()):
        if sz == 1:
            interp = "singleton (no match found)"
        elif sz == 2:
            interp = "matched pair (expected)"
        else:
            interp = "multi-record cluster -- review recommended"
        print(f"{sz:>6}  {count:>6}  {interp}")

    return sorted(components, key=len, reverse=True)


def visualize_top_clusters(G, components, n_clusters=3):
    """
    Plot the top N largest linkage clusters.

    Blue nodes = Source A (survey), green nodes = Source B (admin).
    Edge labels show the match probability assigned by the classifier.

    Parameters
    ----------
    G : nx.Graph
        Linkage graph.
    components : list of frozenset
        Sorted components (largest first).
    n_clusters : int
        Number of clusters to display.
    """
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5))
    if n_clusters == 1:
        axes = [axes]

    for ax, comp in zip(axes, components[:n_clusters]):
        G_sub   = G.subgraph(comp)
        pos_s   = nx.spring_layout(G_sub, seed=1)
        colors  = [
            "#4472C4" if G.nodes[n].get("source") == "A" else "#70AD47"
            for n in G_sub.nodes
        ]
        edge_labels = {(u, v): f"{d['prob']:.2f}" for u, v, d in G_sub.edges(data=True)}
        nx.draw(G_sub, pos_s, ax=ax, with_labels=True,
                node_color=colors, node_size=700, font_size=7,
                edge_color="gray", width=2)
        nx.draw_networkx_edge_labels(G_sub, pos_s, edge_labels=edge_labels,
                                     font_size=7, ax=ax)
        n_a = sum(1 for n in comp if G.nodes[n].get("source") == "A")
        n_b = sum(1 for n in comp if G.nodes[n].get("source") == "B")
        ax.set_title(f"{n_a} Survey + {n_b} Admin records\nBlue=Survey, Green=Admin")
        ax.axis("off")

    plt.suptitle("Linked entity clusters (connected components)", fontsize=11)
    plt.tight_layout()
    plt.savefig("linkage_clusters.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: linkage_clusters.png")


if __name__ == "__main__":
    true_df  = generate_true_records(n=200, seed=42)
    source_a = make_source_a(true_df)
    source_b = make_source_b(true_df, n_overlap=180, seed=99)

    candidate_pairs = generate_candidate_pairs(source_a, source_b)
    comp_df = compute_comparison_features(source_a, source_b, candidate_pairs)
    clf, scaler, _, _, _ = train_linkage_classifier(comp_df)
    print()

    G, comp_df = build_linkage_graph(comp_df, clf, scaler, source_a, source_b)
    components = analyze_clusters(G)
    visualize_top_clusters(G, components, n_clusters=3)
