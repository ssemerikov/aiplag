"""Script 02: Topic modeling with BERTopic and LDA on paper abstracts."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.data_loader import load_and_clean, get_abstracts_df
from utils.viz_config import savefig, FIG_WIDE, FIG_SINGLE, FIGURE_DIR, ANALYSIS_DIR, TOPIC_CMAP, get_topic_color

def main():
    print("=" * 60)
    print("02 — Topic Modeling (BERTopic + LDA)")
    print("=" * 60)

    df = load_and_clean()
    abs_df = get_abstracts_df(df)
    abstracts = abs_df['Abstract'].tolist()
    years = abs_df['Year'].tolist()
    print(f"Abstracts for modeling: {len(abstracts)}")

    # --- BERTopic ---
    print("\n--- BERTopic ---")
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

    # Clustering scaled to the corpus size (~780 abstracts): a larger minimum
    # cluster size avoids the dozens of micro-topics that min_cluster_size=3
    # produces, and topics are reduced to an interpretable set for synthesis.
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True)
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=5)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=12,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(abstracts, embeddings)

    topic_info = topic_model.get_topic_info()
    print(f"\nTopics found: {len(topic_info) - 1} (excluding outliers)")
    print(topic_info[['Topic', 'Count', 'Name']].to_string())

    n_topics = len(topic_info) - 1  # exclude -1
    if n_topics < 3:
        print("\nBERTopic underfitting — forcing nr_topics=5")
        topic_model_forced = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            nr_topics=5,
            verbose=True,
        )
        topics, probs = topic_model_forced.fit_transform(abstracts, embeddings)
        topic_model = topic_model_forced
        topic_info = topic_model.get_topic_info()
        print(f"After forcing: {len(topic_info) - 1} topics")
        print(topic_info[['Topic', 'Count', 'Name']].to_string())

    # Save topic assignments back to df
    abs_df = abs_df.copy()
    abs_df['topic'] = topics

    # --- LDA comparison ---
    print("\n--- LDA (gensim) ---")
    from gensim import corpora
    from gensim.models import LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    import re

    def tokenize(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        from gensim.parsing.preprocessing import STOPWORDS
        return [w for w in text.split() if w not in STOPWORDS and len(w) > 2]

    tokenized = [tokenize(a) for a in abstracts]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    lda_coherences = {}
    for k in range(3, 16):
        lda = LdaModel(bow_corpus, num_topics=k, id2word=dictionary, passes=20, random_state=42)
        cm = CoherenceModel(model=lda, texts=tokenized, dictionary=dictionary, coherence='c_v')
        lda_coherences[k] = cm.get_coherence()
        print(f"  LDA k={k}: coherence={lda_coherences[k]:.4f}")

    best_k = max(lda_coherences, key=lda_coherences.get)
    print(f"  Best LDA k={best_k} (coherence={lda_coherences[best_k]:.4f})")

    lda_best = LdaModel(bow_corpus, num_topics=best_k, id2word=dictionary, passes=30, random_state=42)

    # Descriptive cluster labels (used in legends, not as figure titles)
    def short_label(t):
        w = topic_model.get_topic(t)
        terms = ", ".join(x for x, _ in w[:3]) if w else ""
        return f"T{t}: {terms}" if terms else f"Topic {t}"
    topic_labels = {int(t): short_label(int(t)) for t in topic_info['Topic'] if t != -1}
    results_topic_labels = topic_labels  # surfaced in topic_results.json below

    # --- Visualizations ---
    # 1. Topic bar chart (top words per BERTopic topic)
    fig, axes = plt.subplots(1, 1, figsize=FIG_WIDE)
    topics_to_plot = [t for t in topic_info['Topic'] if t != -1]
    topic_words_data = []
    for t in topics_to_plot:
        words = topic_model.get_topic(t)
        if words:
            for w, s in words[:5]:
                topic_words_data.append({'topic': t, 'word': w, 'score': s})

    if topic_words_data:
        twdf = pd.DataFrame(topic_words_data)
        unique_topics = sorted(twdf['topic'].unique())
        n_t = len(unique_topics)
        # Grid layout (~4 columns) so many topics read as 2-3 rows, not one wide strip.
        ncols = 4 if n_t > 6 else max(1, min(n_t, 3))
        nrows = (n_t + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for j in range(n_t, len(axes)):
            axes[j].axis('off')
        for ax, t in zip(axes, unique_topics):
            sub = twdf[twdf['topic'] == t].sort_values('score')
            color = get_topic_color(t, n_t)
            ax.barh(sub['word'], sub['score'], color=color)
            ax.set_xlabel(f"T{t} (score)")  # panel identity as axis label, not title
            ax.tick_params(labelsize=7)
        savefig(fig, 'fig_topics_barchart.pdf')

    # 2. Topics over time
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    topic_year = pd.DataFrame({'topic': topics, 'year': years})
    topic_year = topic_year[topic_year['topic'] != -1]
    ct = topic_year.groupby(['year', 'topic']).size().unstack(fill_value=0)
    ct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of papers')
    # Descriptive cluster names in the legend (reviewer R3.6); no figure title.
    ax.legend([topic_labels.get(int(c), str(c)) for c in ct.columns],
              title='Topic cluster', bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=7, frameon=False)
    savefig(fig, 'fig_topics_over_time.pdf')

    # 3. Topic-year heatmap
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ct_all = pd.DataFrame({'topic': topics, 'year': years})
    ct_all = ct_all[ct_all['topic'] != -1]
    hm = ct_all.groupby(['topic', 'year']).size().unstack(fill_value=0)
    im = ax.imshow(hm.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(hm.shape[1]))
    ax.set_xticklabels(hm.columns)
    ax.set_yticks(range(hm.shape[0]))
    ax.set_yticklabels([f"Topic {t}" for t in hm.index])
    ax.set_xlabel('Year')
    ax.set_ylabel('Topic')
    ax.set_title('Topic-year heatmap')
    plt.colorbar(im, ax=ax, label='Count')
    savefig(fig, 'fig_topics_heatmap.pdf')

    # --- Save results ---
    results = {
        'n_abstracts': len(abstracts),
        'bertopic_n_topics': int(len(topic_info) - 1),
        'bertopic_topics': [],
        'topic_labels': topic_labels,
        'lda_coherences': {str(k): round(v, 4) for k, v in lda_coherences.items()},
        'lda_best_k': best_k,
    }
    for _, row in topic_info.iterrows():
        t = int(row['Topic'])
        words = topic_model.get_topic(t)
        top_words = [w for w, _ in words[:10]] if words else []
        results['bertopic_topics'].append({
            'id': t,
            'count': int(row['Count']),
            'name': row['Name'],
            'top_words': top_words,
        })

    # Save topic assignments
    topic_assign = abs_df[['paper_id', 'topic']].copy()
    topic_assign.to_csv(os.path.join(ANALYSIS_DIR, 'topic_assignments.csv'), index=False)

    results_path = os.path.join(ANALYSIS_DIR, 'topic_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Save embeddings for script 03
    np.save(os.path.join(ANALYSIS_DIR, 'embeddings.npy'), embeddings)
    print(f"Saved: embeddings.npy")

if __name__ == '__main__':
    main()
