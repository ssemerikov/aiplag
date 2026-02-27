"""Script 06: Keyword co-occurrence network and Callon strategic diagram."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from itertools import combinations
from utils.data_loader import load_and_clean
from utils.viz_config import savefig, FIG_SINGLE, FIG_WIDE, FIG_SQUARE, FIGURE_DIR, ANALYSIS_DIR, CLUSTER_COLORS

def main():
    print("=" * 60)
    print("06 — Keyword Co-occurrence Network & Callon Diagram")
    print("=" * 60)

    df = load_and_clean()

    # Merge and normalize keywords
    all_kw_per_paper = df['all_keywords'].tolist()
    kw_counter = Counter(kw for kws in all_kw_per_paper for kw in kws)
    print(f"Total unique keywords: {len(kw_counter)}")

    # Filter: keywords in >= 2 papers
    freq_kws = {kw for kw, cnt in kw_counter.items() if cnt >= 2}
    print(f"Keywords in >= 2 papers: {len(freq_kws)}")

    # Build co-occurrence network
    G = nx.Graph()
    for kw in freq_kws:
        G.add_node(kw, count=kw_counter[kw])

    cooc_counter = Counter()
    for kws in all_kw_per_paper:
        filtered = [kw for kw in kws if kw in freq_kws]
        for a, b in combinations(sorted(set(filtered)), 2):
            cooc_counter[(a, b)] += 1

    for (a, b), w in cooc_counter.items():
        if w >= 1:
            G.add_edge(a, b, weight=w)

    print(f"Keyword network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Centrality metrics
    degree_cent = nx.degree_centrality(G)
    between_cent = nx.betweenness_centrality(G, weight='weight')
    try:
        eigen_cent = nx.eigenvector_centrality(G, max_iter=500, weight='weight')
    except nx.PowerIterationFailedConvergence:
        eigen_cent = {n: 0 for n in G.nodes()}

    # Louvain communities
    from community import community_louvain
    if G.number_of_nodes() > 0:
        partition = community_louvain.best_partition(G, random_state=42)
        n_comm = len(set(partition.values()))
        print(f"Keyword communities: {n_comm}")
    else:
        partition = {}
        n_comm = 0

    # --- Visualizations ---
    # 1. Keyword network
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    G_vis = G.copy()
    G_vis.remove_nodes_from(list(nx.isolates(G_vis)))

    if G_vis.number_of_nodes() > 0:
        pos = nx.spring_layout(G_vis, k=1.2, iterations=80, seed=42, weight='weight')

        # Edges
        edge_weights = [G_vis[u][v]['weight'] for u, v in G_vis.edges()]
        max_w = max(edge_weights) if edge_weights else 1
        nx.draw_networkx_edges(G_vis, pos, ax=ax, alpha=0.2,
                               width=[0.3 + 1.5 * w / max_w for w in edge_weights],
                               edge_color='#cccccc')

        # Nodes by community
        for comm_id in sorted(set(partition.values())):
            nodes = [n for n in G_vis.nodes() if partition.get(n) == comm_id]
            sizes = [50 + kw_counter[n] * 20 for n in nodes]
            color = CLUSTER_COLORS[comm_id % len(CLUSTER_COLORS)]
            nx.draw_networkx_nodes(G_vis, pos, nodelist=nodes, ax=ax,
                                   node_color=[color], node_size=sizes,
                                   alpha=0.75, edgecolors='white', linewidths=0.3)

        # Label top keywords
        top_kw = sorted(G_vis.nodes(), key=lambda n: degree_cent.get(n, 0), reverse=True)[:20]
        labels = {n: n for n in top_kw}
        nx.draw_networkx_labels(G_vis, pos, labels, ax=ax, font_size=7)

        ax.set_title('Keyword co-occurrence network')
        ax.axis('off')

    savefig(fig, 'fig_keyword_network.pdf')

    # 2. Callon strategic diagram
    # For each community: centrality (external links) vs density (internal links)
    if n_comm > 1:
        comm_data = []
        for comm_id in sorted(set(partition.values())):
            nodes = [n for n in G.nodes() if partition.get(n) == comm_id]
            if not nodes:
                continue
            subgraph = G.subgraph(nodes)
            # Density: avg internal edge weight
            internal_edges = subgraph.edges(data=True)
            internal_weights = [d.get('weight', 1) for _, _, d in internal_edges]
            density = np.mean(internal_weights) if internal_weights else 0
            # Centrality: avg external edge weight
            external_weights = []
            for n in nodes:
                for neighbor in G.neighbors(n):
                    if partition.get(neighbor) != comm_id:
                        external_weights.append(G[n][neighbor].get('weight', 1))
            centrality = np.mean(external_weights) if external_weights else 0
            # Label: top keyword by frequency
            top_kw_in_comm = max(nodes, key=lambda n: kw_counter[n])
            comm_data.append({
                'comm_id': comm_id,
                'centrality': centrality,
                'density': density,
                'size': len(nodes),
                'label': top_kw_in_comm,
            })

        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        cdf = pd.DataFrame(comm_data)
        med_x = cdf['centrality'].median()
        med_y = cdf['density'].median()

        for _, row in cdf.iterrows():
            color = CLUSTER_COLORS[int(row['comm_id']) % len(CLUSTER_COLORS)]
            ax.scatter(row['centrality'], row['density'],
                       s=80 + row['size'] * 30, c=[color], alpha=0.7,
                       edgecolors='black', linewidths=0.5)
            ax.annotate(row['label'], (row['centrality'], row['density']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.axhline(med_y, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(med_x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Centrality (external links)')
        ax.set_ylabel('Density (internal cohesion)')
        ax.set_title('Callon strategic diagram of keyword clusters')

        # Quadrant labels
        ax.text(0.95, 0.95, 'Motor\nthemes', transform=ax.transAxes, ha='right', va='top', fontsize=8, color='gray')
        ax.text(0.05, 0.95, 'Niche\nthemes', transform=ax.transAxes, ha='left', va='top', fontsize=8, color='gray')
        ax.text(0.95, 0.05, 'Basic\nthemes', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='gray')
        ax.text(0.05, 0.05, 'Emerging/\ndeclining', transform=ax.transAxes, ha='left', va='bottom', fontsize=8, color='gray')

        savefig(fig, 'fig_callon_diagram.pdf')

    # 3. Word cloud
    try:
        from wordcloud import WordCloud
        wc = WordCloud(width=800, height=400, background_color='white',
                       colormap='Dark2', max_words=60, prefer_horizontal=0.7)
        wc.generate_from_frequencies(kw_counter)
        fig, ax = plt.subplots(figsize=FIG_WIDE)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Keyword frequency word cloud')
        savefig(fig, 'fig_keyword_wordcloud.pdf')
    except ImportError:
        print("wordcloud not installed — skipping word cloud")

    # Save results
    results = {
        'total_unique_keywords': len(kw_counter),
        'keywords_gte_2': len(freq_kws),
        'network_nodes': G.number_of_nodes(),
        'network_edges': G.number_of_edges(),
        'n_communities': n_comm,
        'top_keywords_by_degree': sorted(degree_cent.items(), key=lambda x: -x[1])[:20],
        'top_keywords_by_betweenness': sorted(between_cent.items(), key=lambda x: -x[1])[:20],
    }
    with open(os.path.join(ANALYSIS_DIR, 'keyword_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nDone.")

if __name__ == '__main__':
    main()
