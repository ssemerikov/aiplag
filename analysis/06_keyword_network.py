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

# Terms that define the search query itself. They appear in almost every record
# by construction, so they inflate cluster size while contributing little
# discriminating co-occurrence signal. Excluded in the sensitivity variant.
QUERY_SEED_TERMS = {
    'artificial intelligence', 'plagiarism', 'plagiarism detection',
    'ai', 'generative ai', 'artificial intelligence (ai)',
}


def compute_callon(G, partition, kw_counter, label_terms=3):
    """Callon strategic-diagram coordinates using the equivalence index.

    The R1 version used raw co-occurrence weights: density was the mean internal
    edge weight and centrality the mean external edge weight. Both are biased by
    keyword frequency, so a large community built around a ubiquitous term is
    pushed toward the low-density corner regardless of how coherent it is.

    This version uses Callon's normalised equivalence index

        e_ij = c_ij^2 / (c_i * c_j)

    where c_ij is the number of documents in which keywords i and j co-occur and
    c_i the number containing i. Dividing by c_i * c_j removes the frequency
    bias. Following Cobo et al. (2011), density is 100 x the mean internal
    equivalence index and centrality 10 x the sum of external ones.

    Each community is labelled with its ``label_terms`` most frequent keywords
    rather than a single one: a one-term label misreads as a claim about that
    keyword, which is precisely how the R1 figure was misread.
    """
    comm_data = []
    for comm_id in sorted(set(partition.values())):
        nodes = [n for n in G.nodes() if partition.get(n) == comm_id]
        if not nodes:
            continue
        node_set = set(nodes)

        internal_e, external_e = [], []
        for u in nodes:
            c_u = kw_counter[u]
            for v in G.neighbors(u):
                c_uv = G[u][v].get('weight', 1)
                c_v = kw_counter[v]
                e = (c_uv ** 2) / (c_u * c_v) if c_u and c_v else 0.0
                if v in node_set:
                    if u < v:          # count each internal pair once
                        internal_e.append(e)
                else:
                    external_e.append(e)

        density = 100.0 * float(np.mean(internal_e)) if internal_e else 0.0
        centrality = 10.0 * float(np.sum(external_e)) if external_e else 0.0

        top_terms = sorted(nodes, key=lambda n: -kw_counter[n])[:label_terms]
        comm_data.append({
            'comm_id': int(comm_id),
            'centrality': round(centrality, 4),
            'density': round(density, 4),
            'size': len(nodes),
            'label': ', '.join(top_terms),
            'top_terms': top_terms,
            'mean_external_e': round(100.0 * float(np.mean(external_e)), 4) if external_e else 0.0,
        })
    return comm_data


def assign_quadrants(comm_data):
    """Tag each community with its Callon quadrant, split at the medians."""
    if not comm_data:
        return comm_data
    med_c = float(np.median([c['centrality'] for c in comm_data]))
    med_d = float(np.median([c['density'] for c in comm_data]))
    for c in comm_data:
        high_c, high_d = c['centrality'] >= med_c, c['density'] >= med_d
        c['quadrant'] = ('motor' if high_c and high_d else
                         'niche' if high_d else
                         'basic/transversal' if high_c else
                         'emerging/declining')
    return comm_data


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
    if n_comm > 1:
        comm_data = compute_callon(G, partition, kw_counter)

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
                        xytext=(5, 5), textcoords='offset points', fontsize=7)

        ax.axhline(med_y, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(med_x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r'Centrality (10 $\times$ $\Sigma$ external equivalence index)')
        ax.set_ylabel(r'Density (100 $\times$ mean internal equivalence index)')
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

    # Sensitivity check: rebuild the diagram with the query seed terms removed.
    # If a community's position is driven by a ubiquitous search term rather
    # than by its own thematic structure, it moves here.
    callon_no_seed = []
    if n_comm > 1:
        G_ns = G.copy()
        G_ns.remove_nodes_from([n for n in G_ns.nodes() if n in QUERY_SEED_TERMS])
        G_ns.remove_nodes_from(list(nx.isolates(G_ns)))
        if G_ns.number_of_nodes() > 0:
            from community import community_louvain as _cl
            part_ns = _cl.best_partition(G_ns, random_state=42)
            callon_no_seed = assign_quadrants(
                compute_callon(G_ns, part_ns, kw_counter))
            print(f"\nSeed-term-excluded network: {G_ns.number_of_nodes()} nodes, "
                  f"{len(set(part_ns.values()))} communities")

    callon_full = assign_quadrants(comm_data) if n_comm > 1 else []
    if callon_full:
        print("\nCallon communities (equivalence index):")
        for c in callon_full:
            print(f"  [{c['comm_id']}] n={c['size']:>4}  centrality={c['centrality']:>8.2f}  "
                  f"density={c['density']:>6.2f}  {c['quadrant']:<18} {c['label']}")
        for c in callon_full:
            if any(t in QUERY_SEED_TERMS for t in c['top_terms']):
                print(f"  -> seed term(s) present in community {c['comm_id']} "
                      f"({c['quadrant']})")

    # Save results
    results = {
        'total_unique_keywords': len(kw_counter),
        'keywords_gte_2': len(freq_kws),
        'network_nodes': G.number_of_nodes(),
        'network_edges': G.number_of_edges(),
        'n_communities': n_comm,
        'top_keywords_by_degree': sorted(degree_cent.items(), key=lambda x: -x[1])[:20],
        'top_keywords_by_betweenness': sorted(between_cent.items(), key=lambda x: -x[1])[:20],
        'callon_note': ('Points are keyword *communities*, labelled by their most '
                        'frequent member terms -- not individual keywords. Density '
                        'and centrality use Callon\'s equivalence index '
                        'e_ij = c_ij^2/(c_i*c_j), which is normalised for keyword '
                        'frequency.'),
        'callon_communities': callon_full,
        'callon_communities_seed_terms_excluded': callon_no_seed,
        'query_seed_terms_excluded': sorted(QUERY_SEED_TERMS),
    }
    with open(os.path.join(ANALYSIS_DIR, 'keyword_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nDone.")

if __name__ == '__main__':
    main()
