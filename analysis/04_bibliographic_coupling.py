"""Script 04: Bibliographic coupling network analysis."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from utils.data_loader import load_and_clean
from utils.reference_parser import build_reference_lists, build_bibliographic_coupling_matrix
from utils.viz_config import savefig, FIG_SQUARE, FIGURE_DIR, ANALYSIS_DIR, CLUSTER_COLORS

def main():
    print("=" * 60)
    print("04 — Bibliographic Coupling Network")
    print("=" * 60)

    df = load_and_clean()
    ref_dict = build_reference_lists(df)

    papers_with_refs = sum(1 for v in ref_dict.values() if len(v) > 0)
    total_refs = sum(len(v) for v in ref_dict.values())
    print(f"Papers with references: {papers_with_refs}/{len(df)}")
    print(f"Total parsed references: {total_refs}")

    # Build coupling matrix
    papers, matrix = build_bibliographic_coupling_matrix(df, ref_dict, min_shared=2)
    print(f"Coupling matrix: {matrix.shape}")
    print(f"Non-zero pairs (shared >= 2 refs): {(matrix > 0).sum() // 2}")

    # Build NetworkX graph
    G = nx.Graph()
    pid_to_info = {row['paper_id']: row for _, row in df.iterrows()}
    for pid in papers:
        info = pid_to_info.get(pid, {})
        G.add_node(pid, year=info.get('Year', 0),
                   title=str(info.get('Title', ''))[:50],
                   cited_by=info.get('Cited by', 0))

    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            if matrix[i, j] > 0:
                G.add_edge(papers[i], papers[j], weight=int(matrix[i, j]))

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Remove isolates for visualization
    isolates = list(nx.isolates(G))
    G_vis = G.copy()
    G_vis.remove_nodes_from(isolates)
    print(f"After removing {len(isolates)} isolates: {G_vis.number_of_nodes()} nodes, {G_vis.number_of_edges()} edges")

    # Louvain community detection
    if G_vis.number_of_nodes() > 0:
        from community import community_louvain
        partition = community_louvain.best_partition(G_vis, random_state=42)
        communities = set(partition.values())
        print(f"Communities detected: {len(communities)}")

        modularity = community_louvain.modularity(partition, G_vis)
        print(f"Modularity: {modularity:.3f}")

        # Visualization
        fig, ax = plt.subplots(figsize=FIG_SQUARE)
        pos = nx.spring_layout(G_vis, k=1.5, iterations=50, seed=42, weight='weight')

        # Draw edges
        edge_weights = [G_vis[u][v]['weight'] for u, v in G_vis.edges()]
        max_w = max(edge_weights) if edge_weights else 1
        nx.draw_networkx_edges(G_vis, pos, ax=ax, alpha=0.2,
                               width=[0.5 + 2 * w / max_w for w in edge_weights],
                               edge_color='#cccccc')

        # Draw nodes colored by community
        for comm_id in sorted(communities):
            nodes = [n for n, c in partition.items() if c == comm_id]
            color = CLUSTER_COLORS[comm_id % len(CLUSTER_COLORS)]
            sizes = [30 + G_vis.nodes[n].get('cited_by', 0) * 3 for n in nodes]
            nx.draw_networkx_nodes(G_vis, pos, nodelist=nodes, ax=ax,
                                   node_color=[color], node_size=sizes,
                                   alpha=0.8, edgecolors='white', linewidths=0.5,
                                   label=f"Community {comm_id + 1}")

        # Label high-degree nodes
        degrees = dict(G_vis.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:8]
        labels = {n: n for n in top_nodes}
        nx.draw_networkx_labels(G_vis, pos, labels, ax=ax, font_size=7)

        ax.set_title('Bibliographic coupling network')
        ax.legend(loc='lower left', fontsize=8)
        ax.axis('off')
        savefig(fig, 'fig_bibcoupling_network.pdf')
    else:
        print("No connected nodes — skipping visualization")
        partition = {}

    # Save graph
    nx.write_graphml(G, os.path.join(ANALYSIS_DIR, 'bibcoupling_graph.graphml'))

    # Save community assignments
    results = {
        'papers_with_refs': papers_with_refs,
        'total_refs': total_refs,
        'n_edges': G.number_of_edges(),
        'n_isolates': len(isolates),
        'n_communities': len(set(partition.values())) if partition else 0,
        'modularity': round(modularity, 3) if partition else None,
        'communities': {},
    }
    for pid, comm in partition.items():
        comm_key = str(comm)
        if comm_key not in results['communities']:
            results['communities'][comm_key] = []
        results['communities'][comm_key].append(pid)

    with open(os.path.join(ANALYSIS_DIR, 'bibcoupling_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDone.")

if __name__ == '__main__':
    main()
