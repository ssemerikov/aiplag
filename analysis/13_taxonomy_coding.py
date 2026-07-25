"""Script 13: Taxonomy coding and inter-coder reliability.

Written for revision R2 in response to two reviewer comments:

  R1#2  Cohen's kappa = 0.44 on the methodology dimension is too low to support
        the claim that particular taxonomy cells are sparse. Revisit the
        category definitions with clearer, mutually exclusive operational
        criteria; recode a random sample of 100 papers; report the new kappa;
        and if reliability cannot exceed 0.70, collapse the methodology
        dimension to two categories or drop it.

  R2#6  Of 797 core papers, 16 lack usable abstracts and 116 more are BERTopic
        outliers, so only 665 were explicitly assigned to a topic -- yet the
        taxonomy is said to "scale to the 797-paper corpus". Explain how those
        132 papers were handled.

The R1 reliability figures were computed ad hoc and no script reproduced them.
This module replaces that with an auditable procedure.

Two independent coding routes are compared:

  Coder A (cluster-inherited)  A paper takes the category assigned to its
                               BERTopic cluster by the codebook. Depends on the
                               cluster solution, not on the individual paper.

  Coder B (paper-level rules)  Revised, mutually exclusive decision rules
                               applied to each paper's own title, keywords and
                               abstract. Never consults the cluster.

The routes share no information, so agreement between them is a genuine
reliability estimate rather than a self-consistency check.

Coverage (R2#6). All 797 papers receive a code:
  - 665 inherit their BERTopic cluster (route "cluster");
  - 116 outliers are assigned to their nearest topic centroid in the
    sentence-embedding space, then inherit that cluster (route "centroid");
  -  16 without abstracts are coded from title and keywords alone by the
    paper-level rules (route "title_only"), with no cluster code, so they
    contribute to the taxonomy but not to the reliability comparison.

Outputs: analysis/taxonomy_coding.json, analysis/taxonomy_sample100.csv,
         analysis/taxonomy_codes.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from utils.data_loader import load_and_clean
from utils.viz_config import ANALYSIS_DIR

RANDOM_SEED = 42
SAMPLE_SIZE = 100
KAPPA_THRESHOLD = 0.70          # Reviewer #1's stated bar

# ---------------------------------------------------------------------------
# Codebook: BERTopic cluster -> taxonomy category (manuscript Section 6.1)
# ---------------------------------------------------------------------------
CLUSTER_ORIENTATION = {
    6: 'technical', 3: 'technical', 8: 'technical', 9: 'technical',
    7: 'technical', 5: 'technical', 0: 'technical',
    2: 'pedagogical', 1: 'pedagogical',
    4: 'governance', 10: 'governance',
}
CLUSTER_METHODOLOGY = {
    6: 'computational', 3: 'computational', 8: 'computational',
    9: 'computational', 7: 'computational', 5: 'computational',
    0: 'computational',
    2: 'empirical', 1: 'empirical',
    4: 'conceptual', 10: 'conceptual',
}

# ---------------------------------------------------------------------------
# Revised paper-level decision rules
#
# The R1 methodology definitions were not mutually exclusive: a paper that ran a
# stakeholder survey *and* trained a classifier satisfied two definitions at
# once, and the coders split on which to record. The revision fixes this by
# coding the paper's PRIMARY EVIDENCE TYPE under an explicit, ordered rule:
#
#   computational  the contribution is a computational artefact or a result
#                  produced by running one: a model, classifier, algorithm,
#                  architecture, benchmark or dataset is proposed or evaluated.
#   empirical      evidence comes from data the authors collected about people
#                  or documents -- survey, interview, experiment with students,
#                  content analysis -- and no new computational artefact.
#   conceptual     no new data and no new artefact: argument, review, policy
#                  analysis, commentary, legal or ethical reasoning.
#
# Where signals for more than one category are present, the title decides: a
# title states the primary contribution, so title matches carry triple weight
# and keyword matches double. Only exact ties fall through to the fixed
# precedence computational > empirical > conceptual, which is recorded per
# paper so the frequency of tie-breaking is auditable.
# ---------------------------------------------------------------------------
# Methodology signals describe what the authors DID, not what the paper is
# ABOUT. This distinction is the substance of the revision: the R1 lists mixed
# topic nouns ("detection", "model", "similarity") with method vocabulary, and
# because those nouns appear in nearly every abstract in a corpus assembled by
# searching for them, they pushed almost everything toward "computational".
METHODOLOGY_SIGNALS = {
    'computational': [
        'we propose', 'we present a model', 'proposed model', 'proposed method',
        'proposed approach', 'proposed architecture', 'proposed system',
        'we train', 'trained on', 'training set', 'training data',
        'fine-tun', 'pre-trained', 'hyperparameter', 'ablation',
        'f1-score', 'f1 score', 'precision and recall', 'roc', 'auc',
        'cross-validation', 'held-out', 'test set', 'baseline model',
        'outperform', 'state-of-the-art', 'benchmark dataset',
        'we implement', 'we develop a', 'prototype', 'experimental results',
        'achieves an accuracy', 'accuracy of', 'evaluated on',
    ],
    'empirical': [
        'survey', 'questionnaire', 'respondents', 'participants',
        'interview', 'focus group', 'semi-structured', 'sample of students',
        'undergraduate students', 'university students', 'case study',
        'observational study', 'we surveyed', 'we interviewed',
        'thematic analysis', 'content analysis', 'quasi-experiment',
        'randomly assigned', 'pre-test', 'post-test', 'likert',
        'perceptions of', 'attitudes of', 'self-report', 'cohort',
        'empirical study', 'field study', 'mixed-methods', 'were recruited',
        'completed the', 'responses from', 'descriptive statistics',
    ],
    'conceptual': [
        'we argue', 'this paper argues', 'this article discusses',
        'we discuss', 'commentary', 'position paper',
        'viewpoint', 'editorial', 'narrative review', 'literature review',
        'systematic review', 'scoping review', 'conceptual framework',
        'theoretical framework', 'ethical implications', 'legal implications',
        'policy implications', 'philosophical', 'normative',
        'we reflect', 'reflection on', 'guidelines for', 'call for',
        'this essay', 'critically examines', 'we consider whether',
    ],
}

# Orientation is a topic judgement, so topic vocabulary is appropriate here --
# but terms that are ubiquitous by construction are still removed by the
# document-frequency filter below.
ORIENTATION_SIGNALS = {
    'technical': [
        'detector', 'classifier', 'algorithm', 'neural', 'transformer',
        'embedding', 'nlp', 'natural language processing',
        'machine learning', 'deep learning', 'source code', 'paraphrase',
        'watermark', 'feature extraction', 'adversarial', 'benchmark',
        'text mining', 'semantic similarity', 'stylometry', 'perplexity',
        'tokeniz', 'fine-tun', 'corpus construction',
    ],
    'pedagogical': [
        'student', 'learner', 'teaching', 'assessment', 'curriculum',
        'classroom', 'pedagog', 'education', 'learning outcome', 'exam',
        'coursework', 'assignment', 'instructor', 'faculty', 'academic writing',
        'higher education', 'undergraduate', 'lecturer', 'academic skills',
    ],
    'governance': [
        'policy', 'policies', 'governance', 'regulation', 'ethic',
        'law', 'legal', 'copyright', 'intellectual property', 'guideline',
        'institutional', 'publisher', 'publishing', 'peer review',
        'authorship', 'research integrity', 'misconduct', 'accountability',
        'compliance', 'retraction', 'editorial policy', 'code of conduct',
    ],
}

# A signal appearing in more than this share of abstracts carries no
# discriminating information in a corpus retrieved by searching for it.
UBIQUITY_THRESHOLD = 0.30


def filter_ubiquitous_signals(signal_map, texts, threshold=UBIQUITY_THRESHOLD):
    """Drop signals whose document frequency exceeds `threshold`.

    A term found in most of the corpus cannot discriminate between categories
    within it. This is the same defect that displaced the "artificial
    intelligence" community in the Callon diagram, applied to coding.
    Returns (filtered_map, dropped_report).
    """
    n = len(texts) or 1
    filtered, dropped = {}, {}
    for category, signals in signal_map.items():
        keep = []
        for s in signals:
            df_share = sum(1 for t in texts if s in t) / n
            if df_share > threshold:
                dropped[s] = round(df_share, 4)
            else:
                keep.append(s)
        filtered[category] = keep
    return filtered, dropped

PRECEDENCE = {
    'methodology': ['computational', 'empirical', 'conceptual'],
    'orientation': ['technical', 'pedagogical', 'governance'],
}


def _count_signals(text, signals):
    """Number of distinct signal phrases present in a lowercase text."""
    if not text:
        return 0
    return sum(1 for s in signals if s in text)


def code_paper(title, keywords, abstract, signal_map, dimension):
    """Apply the weighted paper-level rule. Returns (category, tie_broken)."""
    title = (title or '').lower()
    keywords = (keywords or '').lower()
    abstract = (abstract or '').lower()

    scores = {}
    for category, signals in signal_map.items():
        scores[category] = (3 * _count_signals(title, signals)
                            + 2 * _count_signals(keywords, signals)
                            + 1 * _count_signals(abstract, signals))

    best = max(scores.values())
    if best == 0:
        # No signal at all: fall back to precedence, and flag it.
        return PRECEDENCE[dimension][-1], True

    winners = [c for c, s in scores.items() if s == best]
    if len(winners) == 1:
        return winners[0], False
    for category in PRECEDENCE[dimension]:
        if category in winners:
            return category, True
    return winners[0], True


# ---------------------------------------------------------------------------
# Coder C: embedding prototypes
#
# Coders A and B differ in *kind*: A assigns one category to a whole cluster,
# B reads each paper's words. Their disagreement therefore measures how
# methodologically heterogeneous the clusters are -- a real and reportable
# quantity, but not the reliability of paper-level coding.
#
# Coder C supplies a second *paper-level* judgement that is independent of B's
# vocabulary: each category is described in prose, the description is embedded
# with the same all-MiniLM-L6-v2 model used for the corpus, and each paper is
# assigned to its nearest category prototype in that space. B and C share no
# features -- B matches surface strings, C compares sentence semantics -- so
# agreement between them estimates whether the revised definitions can be
# applied consistently to individual papers.
# ---------------------------------------------------------------------------
METHODOLOGY_PROTOTYPES = {
    'computational': (
        'This paper proposes, implements or evaluates a computational artefact. '
        'It introduces a model, classifier, algorithm, architecture, dataset or '
        'benchmark, trains or fine-tunes it on data, and reports quantitative '
        'performance such as accuracy, precision, recall or F1 score.'),
    'empirical': (
        'This paper reports evidence collected by the authors about people or '
        'documents. It uses a survey, questionnaire, interviews, focus groups, '
        'classroom experiment, case study or content analysis, describes its '
        'participants or sample, and reports what they did, said or believed.'),
    'conceptual': (
        'This paper advances an argument without new data and without a new '
        'computational artefact. It reviews literature, analyses policy, law or '
        'ethics, offers a commentary, position or perspective, and develops '
        'concepts, frameworks or recommendations by reasoning.'),
}

ORIENTATION_PROTOTYPES = {
    'technical': (
        'This paper is about detection methods and systems: algorithms, '
        'classifiers, language models, embeddings, stylometry, watermarking, '
        'source-code or paraphrase analysis, and their technical performance.'),
    'pedagogical': (
        'This paper is about teaching and learning: students, assessment '
        'design, curriculum, classroom practice, academic writing instruction, '
        'and how learners and instructors behave and adapt.'),
    'governance': (
        'This paper is about policy, ethics, law and institutions: regulation, '
        'academic integrity policy, copyright and authorship, publishing and '
        'peer review, research misconduct and institutional response.'),
}


def code_by_prototype(texts, prototypes, model):
    """Assign each text to its nearest category prototype (cosine similarity)."""
    labels = list(prototypes)
    proto_vecs = model.encode([prototypes[l] for l in labels],
                              show_progress_bar=False, normalize_embeddings=True)
    doc_vecs = model.encode(texts, show_progress_bar=False, batch_size=64,
                            normalize_embeddings=True)
    sims = np.asarray(doc_vecs) @ np.asarray(proto_vecs).T
    return [labels[i] for i in sims.argmax(axis=1)]


def cohen_kappa(a, b, labels):
    """Cohen's kappa for two label sequences over a fixed label set."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float('nan')
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = float(np.sum(m.sum(axis=0) * m.sum(axis=1))) / (n * n)
    if np.isclose(pe, 1.0):
        return float('nan')
    return (po - pe) / (1 - pe)


def agreement_stats(a, b, labels):
    a, b = list(a), list(b)
    n = len(a)
    raw = sum(1 for x, y in zip(a, b) if x == y) / n if n else float('nan')
    conf = pd.crosstab(pd.Series(a, name='coder_A'), pd.Series(b, name='coder_B'))
    return {
        'n': n,
        'raw_agreement': round(raw, 4),
        'cohen_kappa': round(cohen_kappa(a, b, labels), 4),
        'confusion_matrix': json.loads(conf.to_json(orient='index')),
    }


def resolve_outliers(df, topics, embeddings, abstract_ids):
    """Assign BERTopic outliers (-1) to their nearest topic centroid.

    Answers R2#6 for the 116 outliers: they are not dropped, they are placed in
    the nearest cluster in the same 384-dimensional sentence-embedding space the
    clustering itself used, and the route is recorded per paper.
    """
    pos = {pid: i for i, pid in enumerate(abstract_ids)}
    assigned = {pid: t for pid, t in topics.items() if t != -1}

    centroids = {}
    for topic in sorted(set(assigned.values())):
        rows = [pos[pid] for pid, t in assigned.items() if t == topic and pid in pos]
        if rows:
            centroids[topic] = embeddings[rows].mean(axis=0)

    keys = sorted(centroids)
    mat = np.vstack([centroids[k] for k in keys])
    mat_n = mat / np.linalg.norm(mat, axis=1, keepdims=True)

    resolved = {}
    for pid, t in topics.items():
        if t != -1 or pid not in pos:
            continue
        v = embeddings[pos[pid]]
        v = v / (np.linalg.norm(v) or 1.0)
        resolved[pid] = int(keys[int(np.argmax(mat_n @ v))])
    return resolved


def main():
    print("=" * 60)
    print("13 — Taxonomy coding and inter-coder reliability")
    print("=" * 60)

    rng = np.random.default_rng(RANDOM_SEED)

    df = load_and_clean()
    topics_df = pd.read_csv(os.path.join(ANALYSIS_DIR, 'topic_assignments.csv'))
    topics = dict(zip(topics_df['paper_id'], topics_df['topic']))
    embeddings = np.load(os.path.join(ANALYSIS_DIR, 'embeddings.npy'))
    abstract_ids = list(topics_df['paper_id'])

    print(f"Core corpus: {len(df)} papers; {len(topics)} with abstracts; "
          f"{sum(1 for t in topics.values() if t == -1)} outliers")

    # --- Coverage: resolve outliers, then code every paper ------------------
    resolved = resolve_outliers(df, topics, embeddings, abstract_ids)
    print(f"Outliers reassigned to nearest topic centroid: {len(resolved)}")

    # Strip signals that are ubiquitous in this corpus before coding anything.
    corpus_texts = [
        (str(r.get('Title') or '') + ' ' +
         ' ; '.join(r.get('all_keywords') or []) + ' ' +
         (str(r.get('Abstract') or '') if r.get('has_abstract') else '')).lower()
        for _, r in df.iterrows()
    ]
    method_signals, method_dropped = filter_ubiquitous_signals(
        METHODOLOGY_SIGNALS, corpus_texts)
    orient_signals, orient_dropped = filter_ubiquitous_signals(
        ORIENTATION_SIGNALS, corpus_texts)
    print(f"Ubiquity filter (df > {UBIQUITY_THRESHOLD:.0%}): dropped "
          f"{len(method_dropped)} methodology and {len(orient_dropped)} "
          f"orientation signals")
    for s, v in sorted(orient_dropped.items(), key=lambda x: -x[1]):
        print(f"    orientation signal dropped: '{s}' (df={v:.2f})")
    for s, v in sorted(method_dropped.items(), key=lambda x: -x[1]):
        print(f"    methodology signal dropped: '{s}' (df={v:.2f})")

    records = []
    for _, row in df.iterrows():
        pid = row['paper_id']
        raw_topic = topics.get(pid)

        if raw_topic is None:
            route, cluster = 'title_only', None
        elif raw_topic == -1:
            route, cluster = 'centroid', resolved.get(pid)
        else:
            route, cluster = 'cluster', int(raw_topic)

        title = str(row.get('Title') or '')
        abstract = '' if not row.get('has_abstract') else str(row.get('Abstract') or '')
        keywords = ' ; '.join(row.get('all_keywords') or [])

        b_orient, b_orient_tie = code_paper(
            title, keywords, abstract, orient_signals, 'orientation')
        b_method, b_method_tie = code_paper(
            title, keywords, abstract, method_signals, 'methodology')

        records.append({
            'paper_id': pid,
            'route': route,
            'cluster': cluster,
            'raw_topic': raw_topic,
            'A_orientation': CLUSTER_ORIENTATION.get(cluster) if cluster is not None else None,
            'A_methodology': CLUSTER_METHODOLOGY.get(cluster) if cluster is not None else None,
            'B_orientation': b_orient,
            'B_methodology': b_method,
            'B_orientation_tie_broken': b_orient_tie,
            'B_methodology_tie_broken': b_method_tie,
            'title': title[:200],
        })

    codes = pd.DataFrame(records)

    # --- Coder C: embedding prototypes (second paper-level judgement) -------
    print("\nCoder C: embedding-prototype coding (all-MiniLM-L6-v2)")
    coder_c_ok = True
    try:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer('all-MiniLM-L6-v2')
        doc_texts = [
            (str(r.get('Title') or '') + '. ' +
             ' ; '.join(r.get('all_keywords') or []) + '. ' +
             (str(r.get('Abstract') or '') if r.get('has_abstract') else ''))
            for _, r in df.iterrows()
        ]
        codes['C_orientation'] = code_by_prototype(
            doc_texts, ORIENTATION_PROTOTYPES, st_model)
        codes['C_methodology'] = code_by_prototype(
            doc_texts, METHODOLOGY_PROTOTYPES, st_model)
        print(f"  orientation: {dict(Counter(codes['C_orientation']))}")
        print(f"  methodology: {dict(Counter(codes['C_methodology']))}")
    except Exception as e:
        coder_c_ok = False
        codes['C_orientation'] = None
        codes['C_methodology'] = None
        print(f"  unavailable ({e}); paper-level reliability will be skipped")

    codes.to_csv(os.path.join(ANALYSIS_DIR, 'taxonomy_codes.csv'), index=False)

    coverage = {
        'core_papers': int(len(df)),
        'route_counts': {k: int(v) for k, v in codes['route'].value_counts().items()},
        'coded_on_both_dimensions': int(codes['B_orientation'].notna().sum()),
        'comparable_on_both_routes': int(codes['A_orientation'].notna().sum()),
    }
    print(f"Coverage by route: {coverage['route_counts']}")

    # --- Reliability on a stratified sample of 100 papers -------------------
    comparable = codes[codes['A_orientation'].notna()].copy()
    strata = comparable['cluster'].astype(int)
    sample_idx = []
    for cluster, group in comparable.groupby(strata):
        take = max(1, round(SAMPLE_SIZE * len(group) / len(comparable)))
        take = min(take, len(group))
        sample_idx.extend(rng.choice(group.index.values, size=take, replace=False).tolist())

    # Trim or top up to exactly SAMPLE_SIZE.
    sample_idx = list(dict.fromkeys(sample_idx))
    if len(sample_idx) > SAMPLE_SIZE:
        sample_idx = rng.choice(sample_idx, size=SAMPLE_SIZE, replace=False).tolist()
    elif len(sample_idx) < SAMPLE_SIZE:
        remaining = [i for i in comparable.index if i not in set(sample_idx)]
        extra = rng.choice(remaining, size=SAMPLE_SIZE - len(sample_idx), replace=False)
        sample_idx.extend(extra.tolist())

    sample = comparable.loc[sample_idx].copy()
    sample.to_csv(os.path.join(ANALYSIS_DIR, 'taxonomy_sample100.csv'), index=False)
    print(f"\nReliability sample: {len(sample)} papers, "
          f"{sample['cluster'].nunique()} clusters represented")

    orient_labels = ['technical', 'pedagogical', 'governance']
    method_labels = ['computational', 'empirical', 'conceptual']

    # Two distinct quantities, reported separately because they answer
    # different questions:
    #
    #   A vs B  cluster-level code against paper-level code. Low agreement here
    #           means clusters are methodologically heterogeneous -- a property
    #           of the cluster solution, not of the codebook. This is what the
    #           R1 manuscript reported as if it were coder reliability.
    #   B vs C  two independent paper-level codings. This is the reliability
    #           Reviewer #1 is actually asking about.
    reliability = {
        'A_vs_B_cluster_vs_paper_level': {
            'interpretation': ('Within-cluster heterogeneity, not coder reliability: '
                               'coder A assigns one category per cluster.'),
            'sample_100': {
                'orientation': agreement_stats(
                    sample['A_orientation'], sample['B_orientation'], orient_labels),
                'methodology': agreement_stats(
                    sample['A_methodology'], sample['B_methodology'], method_labels),
            },
            'full_comparable_set': {
                'orientation': agreement_stats(
                    comparable['A_orientation'], comparable['B_orientation'], orient_labels),
                'methodology': agreement_stats(
                    comparable['A_methodology'], comparable['B_methodology'], method_labels),
            },
        },
    }

    if coder_c_ok:
        sample_c = codes.loc[sample.index]
        reliability['B_vs_C_paper_level'] = {
            'interpretation': ('Inter-coder reliability of the revised paper-level '
                               'definitions: lexical rules versus embedding '
                               'prototypes, which share no features.'),
            'sample_100': {
                'orientation': agreement_stats(
                    sample_c['B_orientation'], sample_c['C_orientation'], orient_labels),
                'methodology': agreement_stats(
                    sample_c['B_methodology'], sample_c['C_methodology'], method_labels),
            },
            'full_corpus': {
                'orientation': agreement_stats(
                    codes['B_orientation'], codes['C_orientation'], orient_labels),
                'methodology': agreement_stats(
                    codes['B_methodology'], codes['C_methodology'], method_labels),
            },
        }

    for comparison, block in reliability.items():
        print(f"\n  [{comparison}]")
        for scope, dims in block.items():
            if not isinstance(dims, dict) or scope == 'interpretation':
                continue
            for dim, s in dims.items():
                print(f"    {scope:<22} {dim:<12} n={s['n']:>3}  "
                      f"raw={s['raw_agreement']:.3f}  kappa={s['cohen_kappa']:.3f}")

    # --- Reviewer #1's decision rule ---------------------------------------
    if coder_c_ok:
        method_kappa = reliability['B_vs_C_paper_level']['sample_100']['methodology']['cohen_kappa']
        kappa_basis = 'B_vs_C_paper_level'
    else:
        method_kappa = reliability['A_vs_B_cluster_vs_paper_level']['sample_100']['methodology']['cohen_kappa']
        kappa_basis = 'A_vs_B_cluster_vs_paper_level'
    collapse = bool(method_kappa < KAPPA_THRESHOLD)

    # The reviewer suggests "empirical vs non-empirical" as an example. Both
    # possible binary splits are evaluated, because collapsing a three-category
    # scheme does not automatically raise kappa: merging two categories also
    # raises expected agreement, and with skewed marginals kappa can fall even
    # as raw agreement rises. Reporting only the favourable split would be
    # cherry-picking; reporting both shows which distinction the data support.
    binary = {}
    splits = {
        'empirical_vs_non_empirical': ('empirical', 'non-empirical'),
        'computational_vs_non_computational': ('computational', 'non-computational'),
    }
    left_col, right_col = ('B_methodology', 'C_methodology') if coder_c_ok \
        else ('A_methodology', 'B_methodology')
    sample_basis = codes.loc[sample.index]
    for name, (positive, negative) in splits.items():
        to_binary = lambda s, p=positive, n=negative: np.where(
            np.asarray(s) == p, p, n)
        binary[name] = {
            'sample_100': agreement_stats(
                to_binary(sample_basis[left_col]),
                to_binary(sample_basis[right_col]), [positive, negative]),
            'full_corpus': agreement_stats(
                to_binary(codes[left_col].dropna()),
                to_binary(codes[right_col].dropna()), [positive, negative]),
        }

    print(f"\nMethodology kappa (3 categories, sample of 100): {method_kappa:.3f} "
          f"({'below' if collapse else 'at or above'} the {KAPPA_THRESHOLD} bar)")
    for name, res in binary.items():
        s = res['sample_100']
        print(f"  collapsed [{name}]: raw={s['raw_agreement']:.3f} "
              f"kappa={s['cohen_kappa']:.3f}")

    best_binary = max(binary, key=lambda k: binary[k]['sample_100']['cohen_kappa'])
    best_kappa = binary[best_binary]['sample_100']['cohen_kappa']
    if collapse and best_kappa < KAPPA_THRESHOLD:
        print(f"  -> No binary collapse reaches {KAPPA_THRESHOLD} either "
              f"(best: {best_binary} at {best_kappa:.3f}).")

    # --- Category distributions for the taxonomy figure --------------------
    dist = {
        'orientation_coder_A': {k: int(v) for k, v in codes['A_orientation'].value_counts().items()},
        'orientation_coder_B': {k: int(v) for k, v in codes['B_orientation'].value_counts().items()},
        'methodology_coder_A': {k: int(v) for k, v in codes['A_methodology'].value_counts().items()},
        'methodology_coder_B': {k: int(v) for k, v in codes['B_methodology'].value_counts().items()},
        'tie_broken_orientation': int(codes['B_orientation_tie_broken'].sum()),
        'tie_broken_methodology': int(codes['B_methodology_tie_broken'].sum()),
    }

    out = {
        'random_seed': RANDOM_SEED,
        'sample_size': SAMPLE_SIZE,
        'kappa_threshold': KAPPA_THRESHOLD,
        'coverage': coverage,
        'reliability': reliability,
        'methodology_collapsed_to_binary': collapse,
        'binary_methodology_reliability': binary,
        'best_binary_split': best_binary,
        'kappa_decision_basis': kappa_basis,
        'category_distributions': dist,
        'codebook': {
            'cluster_orientation': {str(k): v for k, v in CLUSTER_ORIENTATION.items()},
            'cluster_methodology': {str(k): v for k, v in CLUSTER_METHODOLOGY.items()},
        },
        'notes': {
            'coder_A': 'Cluster-inherited: paper takes its BERTopic cluster codebook category.',
            'coder_B': 'Paper-level revised rules over title (x3), keywords (x2), abstract (x1).',
            'independence': ('Coder B never consults the cluster assignment, so agreement '
                             'is a reliability estimate, not a self-consistency check.'),
            'outliers': ('116 BERTopic outliers assigned to nearest topic centroid in the '
                         '384-d embedding space; 16 abstract-less papers coded from title '
                         'and keywords by coder B only.'),
        },
    }

    path = os.path.join(ANALYSIS_DIR, 'taxonomy_coding.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {path}")
    print("Done.")


if __name__ == '__main__':
    main()
