"""Script 10: Temporal network evolution analysis."""
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
from utils.viz_config import savefig, FIG_WIDE, FIG_DOUBLE, FIGURE_DIR, ANALYSIS_DIR, YEAR_COLORS

def main():
    print("=" * 60)
    print("10 — Temporal Network Evolution")
    print("=" * 60)

    df = load_and_clean()
    years = sorted(df['Year'].unique())

    # Build year-by-year co-authorship networks
    metrics = {'year': [], 'density': [], 'avg_degree': [],
               'clustering': [], 'n_components': [], 'n_papers': [],
               'n_authors': [], 'n_keywords': []}

    for year in years:
        year_df = df[df['Year'] <= year]  # cumulative
        # Co-authorship
        G_coauth = nx.Graph()
        for _, row in year_df.iterrows():
            aids = row['author_id_list']
            for aid in aids:
                if not G_coauth.has_node(aid):
                    G_coauth.add_node(aid)
            for a, b in combinations(aids, 2):
                if G_coauth.has_edge(a, b):
                    G_coauth[a][b]['weight'] += 1
                else:
                    G_coauth.add_edge(a, b, weight=1)

        n = G_coauth.number_of_nodes()
        metrics['year'].append(year)
        metrics['n_papers'].append(len(year_df))
        metrics['n_authors'].append(n)
        metrics['density'].append(nx.density(G_coauth) if n > 1 else 0)
        metrics['avg_degree'].append(
            np.mean([d for _, d in G_coauth.degree()]) if n > 0 else 0
        )
        metrics['clustering'].append(
            nx.average_clustering(G_coauth) if n > 1 else 0
        )
        metrics['n_components'].append(
            nx.number_connected_components(G_coauth) if n > 0 else 0
        )
        kw_set = set(kw for kws in year_df['all_keywords'] for kw in kws)
        metrics['n_keywords'].append(len(kw_set))

    mdf = pd.DataFrame(metrics)
    print(mdf.to_string(index=False))

    # 1. Temporal network snapshots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, year in enumerate(years):
        ax = axes[idx]
        year_df = df[df['Year'] <= year]
        G = nx.Graph()
        for _, row in year_df.iterrows():
            aids = row['author_id_list']
            for a, b in combinations(aids, 2):
                if G.has_edge(a, b):
                    G[a][b]['weight'] += 1
                else:
                    G.add_edge(a, b, weight=1)

        G.remove_nodes_from(list(nx.isolates(G)))

        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30,
                                   node_color=YEAR_COLORS.get(year, '#999'),
                                   alpha=0.7)
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5)

        ax.set_title(f'{year} (cumulative: {len(year_df)} papers, '
                     f'{G.number_of_nodes()} connected authors)')
        ax.axis('off')

    fig.suptitle('Co-authorship network evolution (2022–2025)', fontsize=13)
    savefig(fig, 'fig_temporal_networks.pdf')

    # 2. Temporal metrics line plots
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    plot_metrics = [
        ('n_papers', 'Cumulative papers'),
        ('n_authors', 'Cumulative authors'),
        ('n_keywords', 'Cumulative keywords'),
        ('density', 'Network density'),
        ('avg_degree', 'Average degree'),
        ('clustering', 'Clustering coefficient'),
    ]

    for ax, (col, title) in zip(axes.flatten(), plot_metrics):
        ax.plot(mdf['year'], mdf[col], 'o-', color='#4e79a7', linewidth=2, markersize=8)
        ax.set_xlabel('Year')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(years)

    fig.suptitle('Temporal evolution of network metrics', fontsize=13)
    savefig(fig, 'fig_temporal_metrics.pdf')

    # 3. Thematic evolution (keyword alluvial-style)
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    # Track top keywords across years
    top_kws_per_year = {}
    for year in years:
        year_df = df[df['Year'] == year]
        kw_cnt = Counter(kw for kws in year_df['all_keywords'] for kw in kws)
        top_kws_per_year[year] = dict(kw_cnt.most_common(10))

    # Get all keywords that appear in top-10 for any year
    all_top_kws = set()
    for kws in top_kws_per_year.values():
        all_top_kws.update(kws.keys())

    # Create matrix
    kw_list = sorted(all_top_kws)
    data_matrix = np.zeros((len(kw_list), len(years)))
    for j, year in enumerate(years):
        for i, kw in enumerate(kw_list):
            data_matrix[i, j] = top_kws_per_year[year].get(kw, 0)

    # Stacked area chart for top keywords
    # Select top 8 by total count
    totals = data_matrix.sum(axis=1)
    top_idx = np.argsort(-totals)[:8]

    bottom = np.zeros(len(years))
    colors = plt.cm.Set2(np.linspace(0, 1, len(top_idx)))
    for i, idx in enumerate(top_idx):
        ax.fill_between(years, bottom, bottom + data_matrix[idx], alpha=0.7,
                         color=colors[i], label=kw_list[idx])
        bottom += data_matrix[idx]

    ax.set_xlabel('Year')
    ax.set_ylabel('Keyword frequency')
    ax.set_title('Thematic evolution of top keywords')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_xticks(years)
    savefig(fig, 'fig_thematic_evolution.pdf')

    # Save results
    with open(os.path.join(ANALYSIS_DIR, 'temporal_results.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    print("\nDone.")

if __name__ == '__main__':
    main()
