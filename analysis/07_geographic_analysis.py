"""Script 07: Geographic distribution and collaboration analysis."""
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
from utils.viz_config import savefig, FIG_WIDE, FIG_SINGLE, FIG_SQUARE, FIGURE_DIR, ANALYSIS_DIR

def main():
    print("=" * 60)
    print("07 — Geographic & Collaboration Analysis")
    print("=" * 60)

    df = load_and_clean()

    # Country analysis
    country_counter = Counter()
    for countries in df['country_list']:
        for c in countries:
            country_counter[c] += 1

    print(f"Unique countries: {len(country_counter)}")
    print(f"\nTop 15 countries:")
    for country, cnt in country_counter.most_common(15):
        print(f"  {cnt:3d}  {country}")

    # HHI concentration index
    total = sum(country_counter.values())
    shares = [(cnt / total) ** 2 for cnt in country_counter.values()]
    hhi = sum(shares)
    print(f"\nHHI concentration: {hhi:.4f}")

    # 1. Geographic distribution bar chart
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    top_countries = country_counter.most_common(20)
    countries = [c for c, _ in top_countries]
    counts = [n for _, n in top_countries]
    bars = ax.barh(range(len(countries)), counts, color='#4e79a7', alpha=0.8)
    ax.set_yticks(range(len(countries)))
    ax.set_yticklabels(countries, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Number of papers')
    ax.set_title('Geographic distribution of AI plagiarism detection research')
    # Add count labels
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(cnt), va='center', fontsize=8)
    savefig(fig, 'fig_geographic_distribution.pdf')

    # Country-country co-authorship network
    G_country = nx.Graph()
    for _, row in df.iterrows():
        countries = list(set(row['country_list']))
        for c in countries:
            if not G_country.has_node(c):
                G_country.add_node(c, count=0)
            G_country.nodes[c]['count'] += 1
        for c1, c2 in combinations(countries, 2):
            if G_country.has_edge(c1, c2):
                G_country[c1][c2]['weight'] += 1
            else:
                G_country.add_edge(c1, c2, weight=1)

    print(f"\nCountry collaboration network: {G_country.number_of_nodes()} nodes, {G_country.number_of_edges()} edges")

    # 2. Country collaboration network
    fig, ax = plt.subplots(figsize=FIG_SQUARE)
    G_vis = G_country.copy()
    # Keep only countries with >= 2 papers
    small = [n for n in G_vis.nodes() if G_vis.nodes[n].get('count', 0) < 2]
    G_vis.remove_nodes_from(small)
    G_vis.remove_nodes_from(list(nx.isolates(G_vis)))

    if G_vis.number_of_nodes() > 0:
        pos = nx.spring_layout(G_vis, k=2.0, iterations=100, seed=42)
        node_sizes = [100 + G_vis.nodes[n]['count'] * 40 for n in G_vis.nodes()]
        edge_weights = [G_vis[u][v]['weight'] for u, v in G_vis.edges()]
        max_w = max(edge_weights) if edge_weights else 1

        nx.draw_networkx_edges(G_vis, pos, ax=ax, alpha=0.4,
                               width=[0.5 + 2 * w / max_w for w in edge_weights],
                               edge_color='#666666')
        nx.draw_networkx_nodes(G_vis, pos, ax=ax, node_size=node_sizes,
                               node_color='#76b7b2', alpha=0.8,
                               edgecolors='white', linewidths=0.5)
        nx.draw_networkx_labels(G_vis, pos, ax=ax, font_size=8)

        ax.set_title('International collaboration network')
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'No multi-country collaborations found',
                ha='center', va='center', transform=ax.transAxes)

    savefig(fig, 'fig_collaboration_geo.pdf')

    # Institution analysis
    inst_counter = Counter()
    for insts in df['institution_list']:
        for inst in insts:
            inst_counter[inst] += 1

    print(f"\nUnique institutions: {len(inst_counter)}")
    print(f"Top 10 institutions:")
    for inst, cnt in inst_counter.most_common(10):
        print(f"  {cnt:3d}  {inst}")

    # Save results
    results = {
        'unique_countries': len(country_counter),
        'hhi': round(hhi, 4),
        'country_counts': dict(country_counter.most_common()),
        'unique_institutions': len(inst_counter),
        'top_institutions': dict(inst_counter.most_common(20)),
        'collaboration_edges': G_country.number_of_edges(),
    }
    with open(os.path.join(ANALYSIS_DIR, 'geographic_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDone.")

if __name__ == '__main__':
    main()
