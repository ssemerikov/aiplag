"""Script 13: Taxonomy coding, reliability, and cell counts.

Written for revision R2 in response to two reviewer comments:

  R1#2  Cohen's kappa = 0.44 on the methodology dimension is too low to support
        the claim that particular taxonomy cells are sparse. Revisit the
        category definitions with clearer, mutually exclusive operational
        criteria; recode a random sample of 100 papers; report the new kappa;
        and if reliability cannot exceed 0.70, collapse the methodology
        dimension to two categories or drop it.

  R2#6  Of 797 core papers, 16 lack usable abstracts and 116 more are BERTopic
        outliers, so only 665 were explicitly assigned to a topic -- yet the
        taxonomy was said to "scale to the 797-paper corpus".

WHAT CHANGED, AND WHY THE R1 FIGURE WAS MISLEADING
--------------------------------------------------
The R1 procedure assigned categories at the level of the *cluster*: every paper
in a BERTopic cluster inherited that cluster's code. Reliability was then
estimated by comparing those inherited codes against a paper-level lexical
classifier. That comparison does not measure coder reliability. It measures how
methodologically homogeneous the clusters are -- and they are not: the
academic-integrity cluster (T2, n=173) is thematically unified but contains
surveys, position papers and policy analyses in roughly equal measure, so a
single inherited methodology code is invalid for it however reliably coders
agree. This script retains that comparison as a *diagnostic* of within-cluster
heterogeneity, clearly labelled as such, and reports reliability separately.

CODING PROCEDURE
----------------
Coding is now performed per paper, from title, keywords, abstract and document
type, using a written codebook (analysis/coding_task/codebook.md) whose
categories are keyed to the paper's primary evidence type under an ordered
decision rule. All 797 core papers are coded directly, so outliers and
abstract-less papers need no special handling -- which resolves R2#6.

The coders were two independent runs of a large language model (Claude Fable 5)
at two reasoning-effort settings, each given only the codebook and the paper
metadata, blind to each other and to the cluster assignments. A stratified
random sample of 100 papers (seed 42) was double-coded to estimate reliability;
one configuration coded the remaining 697.

INTERPRETING THE COEFFICIENT
----------------------------
Two configurations of one model family share training data and inferential
habits, so they are not independent in the way two human coders are. Their
agreement measures the *determinacy of the codebook* -- whether the definitions
specify a unique answer -- and is an upper bound on reliability, not an estimate
of it. The manuscript states this explicitly rather than reporting the number
bare.

Outputs: analysis/taxonomy_coding.json, analysis/taxonomy_codes.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import glob
import json
from collections import Counter

import numpy as np
import pandas as pd

from utils.data_loader import load_and_clean
from utils.viz_config import ANALYSIS_DIR

CODING_DIR = os.path.join(ANALYSIS_DIR, 'coding_task')
RANDOM_SEED = 42
KAPPA_THRESHOLD = 0.70          # Reviewer #1's stated bar

ORIENT_LABELS = ['technical', 'pedagogical', 'governance']
METHOD_LABELS = ['computational', 'empirical', 'conceptual']

# Codebook mapping used by the *superseded* cluster-level procedure. Retained
# only to quantify within-cluster heterogeneity (the R1 diagnostic).
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


def cohen_kappa(a, b, labels):
    """Cohen's kappa for two label sequences over a fixed label set."""
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float('nan')
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        if x in idx and y in idx:
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
    conf = pd.crosstab(pd.Series(a, name='coder_1'), pd.Series(b, name='coder_2'))
    return {
        'n': n,
        'raw_agreement': round(raw, 4),
        'cohen_kappa': round(cohen_kappa(a, b, labels), 4),
        'confusion_matrix': json.loads(conf.to_json(orient='index')),
    }


def load_codes(pattern):
    """Load coder output files matching a glob into {paper_id: record}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(CODING_DIR, pattern))):
        with open(path) as f:
            for rec in json.load(f):
                out[rec['paper_id']] = rec
    return out


def main():
    print("=" * 60)
    print("13 — Taxonomy coding, reliability and cell counts")
    print("=" * 60)

    df = load_and_clean()
    topics_df = pd.read_csv(os.path.join(ANALYSIS_DIR, 'topic_assignments.csv'))
    topics = dict(zip(topics_df['paper_id'], topics_df['topic']))

    coder_1 = load_codes('codes_coder_A.json')
    coder_2 = load_codes('codes_coder_B.json')
    rest = load_codes('codes_rest_*.json')

    print(f"Double-coded sample: {len(coder_1)} / {len(coder_2)}")
    print(f"Single-coded remainder: {len(rest)}")

    # --- Reliability on the double-coded sample ----------------------------
    shared = sorted(set(coder_1) & set(coder_2))
    reliability = {
        'method': ('Two independent LLM coders (Claude Fable 5, two reasoning-effort '
                   'settings) applying the written codebook, blind to each other.'),
        'caveat': ('Two configurations of one model family are not independent in the '
                   'way two human coders are. This estimates the determinacy of the '
                   'codebook and is an UPPER BOUND on reliability, not an estimate.'),
        'sample_size': len(shared),
        'orientation': agreement_stats(
            [coder_1[p]['orientation'] for p in shared],
            [coder_2[p]['orientation'] for p in shared], ORIENT_LABELS),
        'methodology': agreement_stats(
            [coder_1[p]['methodology'] for p in shared],
            [coder_2[p]['methodology'] for p in shared], METHOD_LABELS),
    }
    for dim in ('orientation', 'methodology'):
        s = reliability[dim]
        print(f"  {dim:<12} n={s['n']:>3}  raw={s['raw_agreement']:.3f}  "
              f"kappa={s['cohen_kappa']:.3f}")

    method_kappa = reliability['methodology']['cohen_kappa']
    decision = ('retain three categories' if method_kappa >= KAPPA_THRESHOLD
                else 'collapse or drop the methodology dimension')
    print(f"  methodology kappa {method_kappa:.3f} vs threshold "
          f"{KAPPA_THRESHOLD} -> {decision}")

    # Disagreements are worth naming individually at this sample size.
    disagreements = [
        {'paper_id': p,
         'coder_1': {k: coder_1[p][k] for k in ('orientation', 'methodology')},
         'coder_2': {k: coder_2[p][k] for k in ('orientation', 'methodology')}}
        for p in shared
        if coder_1[p]['orientation'] != coder_2[p]['orientation']
        or coder_1[p]['methodology'] != coder_2[p]['methodology']
    ]
    print(f"  disagreements: {len(disagreements)}")

    # --- Final per-paper codes --------------------------------------------
    # Sample papers take coder 1's code (the authors adjudicated the sole
    # disagreement in favour of it); the remainder take their single coding.
    final = dict(rest)
    final.update({p: coder_1[p] for p in coder_1})

    rows = []
    for _, row in df.iterrows():
        pid = row['paper_id']
        rec = final.get(pid)
        raw_topic = topics.get(pid)
        rows.append({
            'paper_id': pid,
            'orientation': rec['orientation'] if rec else None,
            'methodology': rec['methodology'] if rec else None,
            'confidence': rec.get('confidence') if rec else None,
            'double_coded': pid in coder_2,
            'cluster': None if raw_topic is None else int(raw_topic),
            'has_abstract': bool(row.get('has_abstract')),
            'title': str(row.get('Title') or '')[:200],
        })
    codes = pd.DataFrame(rows)
    codes.to_csv(os.path.join(ANALYSIS_DIR, 'taxonomy_codes.csv'), index=False)

    coded = codes[codes['orientation'].notna()]
    coverage = {
        'core_papers': int(len(codes)),
        'coded': int(len(coded)),
        'uncoded': int(len(codes) - len(coded)),
        'double_coded': int(codes['double_coded'].sum()),
        'clustering_outliers_coded': int(
            ((codes['cluster'] == -1) & codes['orientation'].notna()).sum()),
        'no_abstract_coded': int(
            ((~codes['has_abstract']) & codes['orientation'].notna()).sum()),
        'note': ('Every paper is coded directly from its own metadata, so '
                 'clustering outliers and abstract-less papers require no '
                 'special handling (reviewer R2#6).'),
    }
    print(f"\nCoverage: {coverage['coded']}/{coverage['core_papers']} coded "
          f"({coverage['clustering_outliers_coded']} outliers, "
          f"{coverage['no_abstract_coded']} without abstracts)")

    # --- Taxonomy cell counts ---------------------------------------------
    cells = pd.crosstab(coded['orientation'], coded['methodology'])
    cells = cells.reindex(index=ORIENT_LABELS, columns=METHOD_LABELS, fill_value=0)
    print("\nTaxonomy cells (rows=orientation, cols=methodology):")
    print(cells.to_string())

    total = int(cells.values.sum())
    cell_share = (cells / total * 100).round(1) if total else cells
    print("\nShare of coded corpus (%):")
    print(cell_share.to_string())

    # --- Diagnostic: cluster-level vs paper-level (why R1's kappa was low) --
    diag = {}
    comparable = codes[codes['cluster'].notna() & (codes['cluster'] != -1)
                       & codes['orientation'].notna()]
    if len(comparable):
        a_or = [CLUSTER_ORIENTATION.get(int(c)) for c in comparable['cluster']]
        a_me = [CLUSTER_METHODOLOGY.get(int(c)) for c in comparable['cluster']]
        diag = {
            'interpretation': ('Cluster-inherited code vs direct paper-level code. '
                               'Low agreement here means clusters are methodologically '
                               'heterogeneous -- it is NOT coder reliability. This is '
                               'what the R1 manuscript reported as kappa=0.71/0.44.'),
            'orientation': agreement_stats(a_or, comparable['orientation'], ORIENT_LABELS),
            'methodology': agreement_stats(a_me, comparable['methodology'], METHOD_LABELS),
        }
        print("\nDiagnostic — cluster-inherited vs paper-level:")
        for dim in ('orientation', 'methodology'):
            s = diag[dim]
            print(f"  {dim:<12} n={s['n']:>3}  raw={s['raw_agreement']:.3f}  "
                  f"kappa={s['cohen_kappa']:.3f}")

        # Which clusters are most methodologically mixed?
        mix = (comparable.groupby('cluster')['methodology']
               .agg(lambda s: round(1 - s.value_counts(normalize=True).max(), 3)))
        diag['within_cluster_methodology_impurity'] = {
            int(k): float(v) for k, v in mix.sort_values(ascending=False).items()}
        print("  most methodologically mixed clusters (impurity): "
              + ", ".join(f"T{k}={v}" for k, v in
                          list(diag['within_cluster_methodology_impurity'].items())[:4]))

    out = {
        'random_seed': RANDOM_SEED,
        'kappa_threshold': KAPPA_THRESHOLD,
        'decision': decision,
        'coders': {
            'coder_1': 'Claude Fable 5, reasoning effort xhigh (also coded the remainder)',
            'coder_2': 'Claude Fable 5, reasoning effort max (reliability sample only)',
            'inputs': 'codebook.md + title, keywords, abstract, document type',
            'blind_to': 'each other, the cluster assignments, and the manuscript',
        },
        'reliability': reliability,
        'disagreements': disagreements,
        'coverage': coverage,
        'taxonomy_cells': json.loads(cells.to_json(orient='index')),
        'taxonomy_cell_share_pct': json.loads(cell_share.to_json(orient='index')),
        'orientation_totals': {k: int(v) for k, v in
                               coded['orientation'].value_counts().items()},
        'methodology_totals': {k: int(v) for k, v in
                               coded['methodology'].value_counts().items()},
        'confidence_distribution': {str(k): int(v) for k, v in
                                    Counter(coded['confidence']).items()},
        'cluster_vs_paper_level_diagnostic': diag,
    }

    path = os.path.join(ANALYSIS_DIR, 'taxonomy_coding.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {path}")
    print("Done.")


if __name__ == '__main__':
    main()
