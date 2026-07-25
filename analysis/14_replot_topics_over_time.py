"""Script 14: Regenerate fig_topics_over_time.pdf from cached assignments.

Reviewer R1#6 asked for the partial 2026 year to be marked in the temporal
figures. The equivalent plotting block in 02_topic_modeling.py has been updated,
but re-running that script would re-fit BERTopic, and the installed version
(0.17.4) differs from the 0.15 used for the reported results. Re-fitting would
silently change the eleven topics and every count that depends on them.

This script therefore rebuilds only the figure, from the archived
topic_assignments.csv and topic_results.json, leaving the topic solution
untouched. It produces the same chart as 02_topic_modeling.py's block, with the
2026 column shaded and annotated.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.data_loader import load_and_clean
from utils.viz_config import savefig, FIG_WIDE, ANALYSIS_DIR

PARTIAL_YEAR = 2026
PARTIAL_YEAR_NOTE = '2026 partial (Jan-May; search 3 Jun 2026)'


def annotate_partial_year(ax, categories):
    """Shade and label the partial year in a categorical bar chart."""
    if PARTIAL_YEAR not in categories:
        return
    pos = list(categories).index(PARTIAL_YEAR)
    ax.axvspan(pos - 0.5, pos + 0.5, color='#bab0ac', alpha=0.22, zorder=0)
    ax.annotate(PARTIAL_YEAR_NOTE,
                xy=(pos, ax.get_ylim()[1]), xytext=(0, 4),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=7, color='#555555')


def main():
    print("14 — Replot topics over time (cached topic solution)")

    df = load_and_clean()
    assignments = pd.read_csv(os.path.join(ANALYSIS_DIR, 'topic_assignments.csv'))
    with open(os.path.join(ANALYSIS_DIR, 'topic_results.json')) as f:
        topic_labels = {int(k): v for k, v in json.load(f)['topic_labels'].items()}

    merged = assignments.merge(df[['paper_id', 'Year']], on='paper_id', how='left')
    merged = merged[merged['topic'] != -1]
    ct = merged.groupby(['Year', 'topic']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of papers')
    ax.legend([topic_labels.get(int(c), str(c)) for c in ct.columns],
              title='Topic cluster', bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=7, frameon=False)
    annotate_partial_year(ax, list(ct.index))
    savefig(fig, 'fig_topics_over_time.pdf')

    print(f"  years: {list(ct.index)}; papers plotted: {int(ct.values.sum())}")
    print("Done.")


if __name__ == '__main__':
    main()
