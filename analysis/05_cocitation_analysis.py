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
from utils.reference_parser import parse_references, normalize_ref_key
import re
from collections import Counter, defaultdict

REF_BOUNDARY = re.compile(r'(?<=\(\d{4}\));\s*')


def first_author_labels(df):
    """Display labels ("First A. (year)") for Scopus-derived cited-work keys.

    Scopus separates both references and the authors inside a reference with
    semicolons, so the stored key and label of a non-DOI Scopus reference carry
    its *last* listed author. Splitting the References string at reference
    boundaries "(yyyy); " recovers each reference's first author. Keys and all
    counts are unchanged; only the node labels are affected (R4 fix).
    """
    seen = defaultdict(Counter)
    for refs in df.loc[df['source'] == 'Scopus', 'References']:
        if not isinstance(refs, str):
            continue
        for chunk in REF_BOUNDARY.split(refs):
            keys = [k for k in (normalize_ref_key(r) for r in parse_references(chunk)) if k]
            if len(keys) != 1:
                continue
            fa = re.split(r'[;,]', chunk, 1)[0].strip()
            fa = re.sub(r'^(\S+)\s+([A-Z])[a-z]+$', r'\1 \2.', fa)   # "Gehrmann Sebastian" -> "Gehrmann S."
            yr = re.search(r'\((\d{4})\)\s*$', chunk)
            if 0 < len(fa) <= 40 and yr:
                seen[keys[0]][f"{fa} ({yr.group(1)})"] += 1
    return {k: c.most_common(1)[0][0] for k, c in seen.items()}


def main():
    print("=" * 60)
    print("05 — Co-citation Analysis")
    print("=" * 60)

    df = load_and_clean()
    ref_dict = build_reference_lists(df)

    # Human-readable labels for canonical cited-work keys (doi:/ay:), built by 00.
    try:
        with open(os.path.join(ANALYSIS_DIR, 'ref_labels.json')) as f:
            label_map = json.load(f)
    except FileNotFoundError:
        label_map = {}

    fa_labels = first_author_labels(df)

    def lab(key):
        if key in fa_labels:
            return fa_labels[key]
        if label_map.get(key, '').strip():
            return label_map[key]
        if key.startswith('doi:'):
            return key[4:][:40]
        if key.startswith('ay:'):
            p = key[3:].split('|')
            return f"{p[0].title()} ({p[1]})" if len(p) > 1 and p[1].isdigit() else p[0].title()
        return key[:40]

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
    labels = [lab(ref)[:45] for ref, _ in top_refs]

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
        G.add_node(i, label=lab(ref)[:22], full_ref=ref, count=counts.get(ref, 0))

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
        'top_cited': [{'ref': ref, 'label': lab(ref), 'count': cnt} for ref, cnt in counts.most_common(30)],
        'cocitation_edges': G.number_of_edges(),
    }
    with open(os.path.join(ANALYSIS_DIR, 'cocitation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDone.")

if __name__ == '__main__':
    main()
