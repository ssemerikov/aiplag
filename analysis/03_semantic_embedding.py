"""Script 03: Semantic embedding analysis — UMAP projections and similarity."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.data_loader import load_and_clean, get_abstracts_df
from utils.viz_config import savefig, FIG_SINGLE, FIG_WIDE, FIGURE_DIR, ANALYSIS_DIR, YEAR_COLORS

def main():
    print("=" * 60)
    print("03 — Semantic Embedding Analysis")
    print("=" * 60)

    df = load_and_clean()
    abs_df = get_abstracts_df(df)

    # Load or compute embeddings
    emb_path = os.path.join(ANALYSIS_DIR, 'embeddings.npy')
    if os.path.exists(emb_path):
        embeddings = np.load(emb_path)
        print(f"Loaded embeddings: {embeddings.shape}")
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(abs_df['Abstract'].tolist(), show_progress_bar=True)
        np.save(emb_path, embeddings)
        print(f"Computed and saved embeddings: {embeddings.shape}")

    # Load topic assignments if available
    topic_path = os.path.join(ANALYSIS_DIR, 'topic_assignments.csv')
    if os.path.exists(topic_path):
        topics_df = pd.read_csv(topic_path)
        abs_df = abs_df.merge(topics_df, on='paper_id', how='left')
    else:
        abs_df['topic'] = 0

    # --- Cosine similarity matrix ---
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    print(f"Similarity matrix: {sim_matrix.shape}")
    print(f"  Mean pairwise similarity: {sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean():.3f}")
    print(f"  Std: {sim_matrix[np.triu_indices_from(sim_matrix, k=1)].std():.3f}")

    # --- UMAP 2D projection ---
    from umap import UMAP
    umap_2d = UMAP(n_neighbors=8, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
    coords = umap_2d.fit_transform(embeddings)

    # 1. UMAP colored by year
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for year in sorted(abs_df['Year'].unique()):
        mask = abs_df['Year'].values == year
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=YEAR_COLORS.get(year, '#999999'), label=str(year),
                   s=40 + abs_df.loc[abs_df['Year'] == year, 'Cited by'].values * 2,
                   alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Semantic landscape of AI plagiarism detection research')
    ax.legend(title='Year')
    savefig(fig, 'fig_umap_scatter.pdf')

    # 2. UMAP colored by topic
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    topic_vals = abs_df['topic'].values
    unique_topics = sorted(set(topic_vals))
    cmap = plt.cm.Set2
    for i, t in enumerate(unique_topics):
        mask = topic_vals == t
        label = f"Topic {t}" if t >= 0 else "Outlier"
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i / max(len(unique_topics) - 1, 1))], label=label,
                   s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Semantic landscape colored by topic')
    ax.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    savefig(fig, 'fig_umap_by_topic.pdf')

    # 3. Similarity heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    # Sort by year for clearer structure
    sort_idx = abs_df['Year'].values.argsort()
    sorted_sim = sim_matrix[sort_idx][:, sort_idx]
    im = ax.imshow(sorted_sim, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel('Paper index (sorted by year)')
    ax.set_ylabel('Paper index (sorted by year)')
    ax.set_title('Pairwise semantic similarity')
    plt.colorbar(im, ax=ax, label='Cosine similarity')
    savefig(fig, 'fig_similarity_heatmap.pdf')

    print("\nDone — 3 figures saved.")

if __name__ == '__main__':
    main()
