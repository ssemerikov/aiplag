"""Script 05: Co-citation analysis of external references."""
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
from utils.reference_parser import build_reference_lists, count_cited_works, build_cocitation_matrix
from utils.viz_config import savefig, FIG_WIDE, FIG_SQUARE, FIGURE_DIR, ANALYSIS_DIR

def main():
    print("=" * 60)
    print("05 — Co-citation Analysis")
    print("=" * 60)

    df = load_and_clean()
    ref_dict = build_reference_lists(df)

    # Count cited works
    counts = count_cited_works(ref_dict)
    print(f"Unique cited works: {len(counts)}")
    print(f"\nTop 20 most cited works:")
    for ref, cnt in counts.most_common(20):
        # Truncate for display
        display = ref[:80] + "..." if len(ref) > 80 else ref
        print(f"  {cnt:3d}x  {display}")

    # Bar chart of top cited works
    top_n = 25
    top_refs = counts.most_common(top_n)
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = []
    for ref, _ in top_refs:
        # Create readable short label
        parts = ref.split('|')
        if len(parts) >= 2:
            label = f"{parts[0]} ({parts[1]})" if parts[1].isdigit() else parts[0]
        else:
            label = ref[:40]
        labels.append(label[:45])

    y_pos = range(len(top_refs))
    counts_vals = [c for _, c in top_refs]
    ax.barh(y_pos, counts_vals, color='#4e79a7', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Citation count within corpus')
    ax.set_title(f'Top {top_n} most cited works across the corpus')
    savefig(fig, 'fig_cited_works_bar.pdf')

    # Co-citation network
    top_refs_keys, cocit_matrix = build_cocitation_matrix(ref_dict, top_n=30)
    print(f"\nCo-citation matrix: {cocit_matrix.shape}")

    G = nx.Graph()
    for i, ref in enumerate(top_refs_keys):
        parts = ref.split('|')
        label = parts[0][:20] if parts else ref[:20]
        G.add_node(i, label=label, full_ref=ref, count=counts.get(ref, 0))

    for i in range(len(top_refs_keys)):
        for j in range(i + 1, len(top_refs_keys)):
            if cocit_matrix[i, j] > 0:
                G.add_edge(i, j, weight=int(cocit_matrix[i, j]))

    print(f"Co-citation network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Visualize
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    # Remove isolates
    G_vis = G.copy()
    G_vis.remove_nodes_from(list(nx.isolates(G_vis)))

    if G_vis.number_of_nodes() > 2:
        pos = nx.spring_layout(G_vis, k=2.0, iterations=50, seed=42)

        edge_weights = [G_vis[u][v]['weight'] for u, v in G_vis.edges()]
        max_w = max(edge_weights) if edge_weights else 1

        nx.draw_networkx_edges(G_vis, pos, ax=ax, alpha=0.3,
                               width=[0.5 + 2 * w / max_w for w in edge_weights],
                               edge_color='#aaaaaa')

        node_sizes = [100 + G_vis.nodes[n].get('count', 1) * 30 for n in G_vis.nodes()]
        nx.draw_networkx_nodes(G_vis, pos, ax=ax, node_size=node_sizes,
                               node_color='#e15759', alpha=0.7, edgecolors='white')

        labels = {n: G_vis.nodes[n]['label'] for n in G_vis.nodes()}
        nx.draw_networkx_labels(G_vis, pos, labels, ax=ax, font_size=7)

        ax.set_title('Co-citation network of foundational works')
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'Insufficient co-citation links\nfor network visualization',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Co-citation network')

    savefig(fig, 'fig_cocitation_network.pdf')

    # Save results
    results = {
        'unique_cited_works': len(counts),
        'top_cited': [{'ref': ref, 'count': cnt} for ref, cnt in counts.most_common(30)],
        'cocitation_edges': G.number_of_edges(),
    }
    with open(os.path.join(ANALYSIS_DIR, 'cocitation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDone.")

if __name__ == '__main__':
    main()
