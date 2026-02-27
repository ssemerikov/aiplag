"""Script 09: Build genuine multi-relational knowledge graph."""
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
from utils.data_loader import load_and_clean
from utils.viz_config import savefig, FIG_SQUARE, FIG_WIDE, FIGURE_DIR, ANALYSIS_DIR, NODE_TYPE_COLORS

def main():
    print("=" * 60)
    print("09 — Knowledge Graph Construction")
    print("=" * 60)

    df = load_and_clean()

    # Load topic assignments
    topic_path = os.path.join(ANALYSIS_DIR, 'topic_assignments.csv')
    if os.path.exists(topic_path):
        topics_df = pd.read_csv(topic_path)
        df = df.merge(topics_df, on='paper_id', how='left')
        df['topic'] = df['topic'].fillna(-1).astype(int)
    else:
        df['topic'] = 0

    # Build multi-relational KG
    G = nx.MultiDiGraph()

    # --- Add nodes ---
    # Papers (58)
    for _, row in df.iterrows():
        G.add_node(row['paper_id'], type='Paper', label=str(row['Title'])[:60],
                   year=int(row['Year']), cited_by=int(row['Cited by']))

    # Authors
    author_ids = set()
    author_names = {}
    for _, row in df.iterrows():
        names = row['author_list']
        ids = row['author_id_list']
        for i, aid in enumerate(ids):
            if aid and aid not in author_ids:
                author_ids.add(aid)
                name = names[i] if i < len(names) else aid
                # Clean name
                name = name.split('(')[0].strip().rstrip(',').strip()
                author_names[aid] = name
                G.add_node(f"A_{aid}", type='Author', label=name)

    # Institutions
    inst_counter = Counter(i for insts in df['institution_list'] for i in insts)
    top_insts = {inst for inst, cnt in inst_counter.items() if cnt >= 1}
    for inst in top_insts:
        G.add_node(f"I_{inst[:40]}", type='Institution', label=inst[:40])

    # Countries
    country_counter = Counter(c for cs in df['country_list'] for c in cs)
    for country in country_counter:
        G.add_node(f"C_{country}", type='Country', label=country)

    # Topics
    topic_results_path = os.path.join(ANALYSIS_DIR, 'topic_results.json')
    if os.path.exists(topic_results_path):
        with open(topic_results_path) as f:
            topic_data = json.load(f)
        for t in topic_data.get('bertopic_topics', []):
            tid = t['id']
            if tid >= 0:
                G.add_node(f"T_{tid}", type='Topic', label=t['name'][:40])

    # Keywords (top by frequency)
    kw_counter = Counter(kw for kws in df['all_keywords'] for kw in kws)
    top_kws = {kw for kw, cnt in kw_counter.items() if cnt >= 2}
    for kw in top_kws:
        G.add_node(f"K_{kw}", type='Keyword', label=kw)

    # Technologies and Methods (extracted from keywords)
    tech_keywords = {
        'chatgpt', 'gpt', 'bert', 'transformer', 'deep learning', 'machine learning',
        'neural network', 'nlp', 'natural language processing', 'large language model',
        'llm', 'turnitin', 'generative ai', 'artificial intelligence',
    }
    method_keywords = {
        'text similarity', 'cosine similarity', 'plagiarism detection', 'text mining',
        'sentiment analysis', 'classification', 'clustering', 'semantic analysis',
        'feature extraction', 'topic modeling',
    }
    for kw in kw_counter:
        kw_low = kw.lower()
        for tech in tech_keywords:
            if tech in kw_low:
                G.add_node(f"Tech_{tech}", type='Technology', label=tech)
                break
        for method in method_keywords:
            if method in kw_low:
                G.add_node(f"M_{method}", type='Method', label=method)
                break

    # Venues
    venues = df['Source title'].dropna().unique()
    for venue in venues:
        G.add_node(f"V_{venue[:40]}", type='Venue', label=venue[:40])

    # Years
    for year in df['Year'].unique():
        G.add_node(f"Y_{year}", type='Year', label=str(year))

    print(f"Nodes: {G.number_of_nodes()}")
    node_types = Counter(G.nodes[n].get('type', 'Unknown') for n in G.nodes())
    for ntype, cnt in sorted(node_types.items()):
        print(f"  {ntype}: {cnt}")

    # --- Add edges ---
    for _, row in df.iterrows():
        pid = row['paper_id']

        # AUTHORED_BY
        for aid in row['author_id_list']:
            if f"A_{aid}" in G:
                G.add_edge(pid, f"A_{aid}", relation='AUTHORED_BY')

        # PUBLISHED_IN
        venue = row.get('Source title')
        if pd.notna(venue):
            vnode = f"V_{str(venue)[:40]}"
            if vnode in G:
                G.add_edge(pid, vnode, relation='PUBLISHED_IN')

        # PUBLISHED_YEAR
        G.add_edge(pid, f"Y_{row['Year']}", relation='PUBLISHED_IN_YEAR')

        # HAS_TOPIC
        topic = row.get('topic', -1)
        if topic >= 0 and f"T_{topic}" in G:
            G.add_edge(pid, f"T_{topic}", relation='HAS_TOPIC')

        # USES_KEYWORD
        for kw in row['all_keywords']:
            if f"K_{kw}" in G:
                G.add_edge(pid, f"K_{kw}", relation='USES_KEYWORD')

        # MENTIONS_TECHNOLOGY / USES_METHOD
        for kw in row['all_keywords']:
            kw_low = kw.lower()
            for tech in tech_keywords:
                if tech in kw_low and f"Tech_{tech}" in G:
                    G.add_edge(pid, f"Tech_{tech}", relation='MENTIONS_TECHNOLOGY')
            for method in method_keywords:
                if method in kw_low and f"M_{method}" in G:
                    G.add_edge(pid, f"M_{method}", relation='USES_METHOD')

    # AFFILIATED_WITH (Author -> Institution)
    for _, row in df.iterrows():
        # Parse "Authors with affiliations" for detailed mapping
        aff_str = row.get('Authors with affiliations', '')
        if pd.isna(aff_str):
            continue
        for aid in row['author_id_list']:
            for inst in row['institution_list']:
                inode = f"I_{inst[:40]}"
                if f"A_{aid}" in G and inode in G:
                    G.add_edge(f"A_{aid}", inode, relation='AFFILIATED_WITH')

    # LOCATED_IN (Institution -> Country)
    for _, row in df.iterrows():
        for inst in row['institution_list']:
            inode = f"I_{inst[:40]}"
            for country in row['country_list']:
                cnode = f"C_{country}"
                if inode in G and cnode in G:
                    if not G.has_edge(inode, cnode):
                        G.add_edge(inode, cnode, relation='LOCATED_IN')

    # CO_OCCURS_WITH (keyword-keyword from co-occurrence)
    from itertools import combinations
    for kws in df['all_keywords']:
        filtered = [kw for kw in kws if f"K_{kw}" in G]
        for a, b in combinations(sorted(set(filtered)), 2):
            G.add_edge(f"K_{a}", f"K_{b}", relation='CO_OCCURS_WITH')

    # COLLABORATES_WITH (co-authorship)
    for _, row in df.iterrows():
        aids = row['author_id_list']
        for a, b in combinations(aids, 2):
            if f"A_{a}" in G and f"A_{b}" in G:
                G.add_edge(f"A_{a}", f"A_{b}", relation='COLLABORATES_WITH')

    print(f"Edges: {G.number_of_edges()}")
    edge_types = Counter(d.get('relation', 'unknown') for _, _, d in G.edges(data=True))
    for etype, cnt in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {cnt}")

    # --- Compute graph metrics ---
    G_simple = nx.Graph(G)  # Simplified for metrics
    print(f"\nGraph statistics (simplified):")
    print(f"  Nodes: {G_simple.number_of_nodes()}")
    print(f"  Edges: {G_simple.number_of_edges()}")
    print(f"  Density: {nx.density(G_simple):.4f}")
    components = list(nx.connected_components(G_simple))
    print(f"  Connected components: {len(components)}")
    largest_cc = max(components, key=len)
    print(f"  Largest component: {len(largest_cc)} nodes")

    # Centrality on simplified graph
    degree_cent = nx.degree_centrality(G_simple)
    between_cent = nx.betweenness_centrality(G_simple)

    # Identify peripheral nodes (potential gaps)
    paper_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'Paper']
    paper_centrality = {n: degree_cent.get(n, 0) for n in paper_nodes}
    peripheral_papers = sorted(paper_centrality.items(), key=lambda x: x[1])[:10]
    print(f"\nMost peripheral papers (potential gaps):")
    for pid, cent in peripheral_papers:
        label = G.nodes[pid].get('label', pid)
        print(f"  {cent:.4f}  {pid}: {label}")

    # --- Visualizations ---
    # 1. Full KG (simplified view - top nodes by centrality)
    fig, ax = plt.subplots(figsize=(12, 10))
    top_n = 60
    top_nodes = sorted(degree_cent.items(), key=lambda x: -x[1])[:top_n]
    top_node_ids = [n for n, _ in top_nodes]
    G_sub = G_simple.subgraph(top_node_ids).copy()

    pos = nx.spring_layout(G_sub, k=1.5, iterations=100, seed=42)

    # Draw by type
    for ntype, color in NODE_TYPE_COLORS.items():
        nodes = [n for n in G_sub.nodes() if G.nodes[n].get('type') == ntype]
        if not nodes:
            continue
        sizes = [100 + degree_cent.get(n, 0) * 2000 for n in nodes]
        nx.draw_networkx_nodes(G_sub, pos, nodelist=nodes, ax=ax,
                               node_color=[color], node_size=sizes,
                               alpha=0.75, edgecolors='white', linewidths=0.5,
                               label=ntype)

    nx.draw_networkx_edges(G_sub, pos, ax=ax, alpha=0.1, edge_color='#cccccc', width=0.3)

    # Label top 15 nodes
    top_labels = {n: G.nodes[n].get('label', n)[:20] for n, _ in top_nodes[:15]}
    nx.draw_networkx_labels(G_sub, pos, top_labels, ax=ax, font_size=6)

    ax.set_title(f'Knowledge graph — top {top_n} nodes by centrality')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.axis('off')
    savefig(fig, 'fig_knowledge_graph_full.pdf')

    # 2. Core subgraph (papers + their topics + top keywords)
    fig, ax = plt.subplots(figsize=(10, 8))
    core_nodes = set(paper_nodes)
    for n in paper_nodes:
        for neighbor in G_simple.neighbors(n):
            ntype = G.nodes[neighbor].get('type', '')
            if ntype in ('Topic', 'Keyword', 'Technology', 'Method'):
                core_nodes.add(neighbor)

    # Limit to top keywords
    core_kw_cent = {n: degree_cent.get(n, 0) for n in core_nodes
                    if G.nodes[n].get('type') in ('Keyword', 'Technology', 'Method')}
    top_core_kw = sorted(core_kw_cent.items(), key=lambda x: -x[1])[:20]
    keep_nodes = set(paper_nodes) | {n for n, _ in top_core_kw}
    # Add topics
    keep_nodes |= {n for n in core_nodes if G.nodes[n].get('type') == 'Topic'}

    G_core = G_simple.subgraph(keep_nodes).copy()
    G_core.remove_nodes_from(list(nx.isolates(G_core)))

    if G_core.number_of_nodes() > 0:
        pos = nx.spring_layout(G_core, k=1.0, iterations=80, seed=42)
        for ntype, color in NODE_TYPE_COLORS.items():
            nodes = [n for n in G_core.nodes() if G.nodes[n].get('type') == ntype]
            if not nodes:
                continue
            sizes = [80 if ntype == 'Paper' else 150 for _ in nodes]
            nx.draw_networkx_nodes(G_core, pos, nodelist=nodes, ax=ax,
                                   node_color=[color], node_size=sizes,
                                   alpha=0.7, edgecolors='white', linewidths=0.5,
                                   label=ntype)

        nx.draw_networkx_edges(G_core, pos, ax=ax, alpha=0.1, width=0.3, edge_color='#cccccc')

        # Label non-paper nodes
        labels = {n: G.nodes[n].get('label', n)[:20] for n in G_core.nodes()
                  if G.nodes[n].get('type') != 'Paper'}
        nx.draw_networkx_labels(G_core, pos, labels, ax=ax, font_size=6)

        ax.set_title('Core knowledge graph: papers, topics, and keywords')
        ax.legend(loc='lower left', fontsize=8)
        ax.axis('off')

    savefig(fig, 'fig_knowledge_graph_core.pdf')

    # --- Save ---
    # Save as GraphML (convert MultiDiGraph to simple for GraphML)
    G_export = nx.DiGraph()
    for n, data in G.nodes(data=True):
        G_export.add_node(n, **{k: str(v) for k, v in data.items()})
    for u, v, data in G.edges(data=True):
        if not G_export.has_edge(u, v):
            G_export.add_edge(u, v, **{k: str(v) for k, v in data.items()})

    nx.write_graphml(G_export, os.path.join(ANALYSIS_DIR, 'knowledge_graph.graphml'))

    # Save JSON summary
    kg_json = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'node_types': dict(node_types),
        'edge_types': dict(edge_types),
        'density': round(nx.density(G_simple), 4),
        'n_components': len(components),
        'largest_component_size': len(largest_cc),
        'peripheral_papers': [{'id': pid, 'centrality': round(cent, 4),
                                'label': G.nodes[pid].get('label', '')}
                               for pid, cent in peripheral_papers],
    }
    with open(os.path.join(ANALYSIS_DIR, 'knowledge_graph.json'), 'w') as f:
        json.dump(kg_json, f, indent=2)

    print(f"\nKnowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Done.")

if __name__ == '__main__':
    main()
