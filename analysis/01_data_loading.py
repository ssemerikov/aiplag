"""Script 01: Load, validate, and describe the Scopus corpus."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
from utils.data_loader import load_and_clean, summary_stats, ANALYSIS_DIR

def main():
    print("=" * 60)
    print("01 — Data Loading & Validation")
    print("=" * 60)

    df = load_and_clean()  # canonical merged core corpus (built by 00_build_corpus)
    stats = summary_stats(df)

    print(f"\nCorpus: {stats['total_papers']} papers")
    print(f"  With abstracts:        {stats['with_abstracts']}/{stats['total_papers']}")
    print(f"  With references:       {stats['with_references']}/{stats['total_papers']}")
    print(f"  With author keywords:  {stats['with_author_keywords']}/{stats['total_papers']}")
    print(f"  With index keywords:   {stats['with_index_keywords']}/{stats['total_papers']}")
    print(f"\nUnique authors:      {stats['unique_authors']}")
    print(f"Unique countries:    {stats['unique_countries']}")
    print(f"Unique institutions: {stats['unique_institutions']}")
    print(f"\nYear distribution: {stats['year_distribution']}")
    print(f"Doc types: {stats['doc_type_distribution']}")
    print(f"\nCitation stats: {stats['citation_stats']}")

    # Show countries
    all_countries = sorted(set(c for cs in df['country_list'] for c in cs))
    print(f"\nCountries ({len(all_countries)}): {', '.join(all_countries[:20])}...")

    # Show top institutions
    from collections import Counter
    inst_counts = Counter(i for ins in df['institution_list'] for i in ins)
    print(f"\nTop institutions:")
    for inst, cnt in inst_counts.most_common(10):
        print(f"  {cnt:3d}  {inst}")

    # Save stats
    stats_path = os.path.join(ANALYSIS_DIR, 'corpus_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {stats_path}")
    print(f"Saved: {os.path.join(ANALYSIS_DIR, 'corpus_clean.pkl')}")

if __name__ == '__main__':
    main()
