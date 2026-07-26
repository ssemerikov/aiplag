"""Script 11: Keyword burst and emergence detection."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from utils.data_loader import load_and_clean
from utils.viz_config import savefig, FIG_WIDE, FIG_SINGLE, FIGURE_DIR, ANALYSIS_DIR

# The corpus search ran on 3 June 2026, so 2026 counts cover January-May only.
# Plotting a truncated year beside complete ones implies a full-year value and
# understates the true 2026 total; every temporal figure marks it explicitly.
PARTIAL_YEAR = 2026
PARTIAL_YEAR_NOTE = '2026 partial (Jan-May; search 3 Jun 2026)'

def main():
    print("=" * 60)
    print("11 — Keyword Burst & Emergence Detection")
    print("=" * 60)

    df = load_and_clean()
    years = sorted(df['Year'].unique())

    # Count keywords per year
    kw_by_year = {}
    for year in years:
        year_df = df[df['Year'] == year]
        kw_by_year[year] = Counter(kw for kws in year_df['all_keywords'] for kw in kws)

    # Overall keyword frequency
    all_kw = Counter(kw for kws in df['all_keywords'] for kw in kws)
    freq_kws = {kw for kw, cnt in all_kw.items() if cnt >= 2}

    # Compute growth rates
    growth_data = []
    for kw in freq_kws:
        counts = [kw_by_year[y].get(kw, 0) for y in years]
        total = sum(counts)
        first_year = None
        last_year = None
        for i, y in enumerate(years):
            if counts[i] > 0:
                if first_year is None:
                    first_year = y
                last_year = y

        # Growth rate: compare last two periods vs first two
        early = sum(counts[:2])
        late = sum(counts[2:])
        if early > 0:
            growth = (late - early) / early
        elif late > 0:
            growth = float('inf')
        else:
            growth = 0

        growth_data.append({
            'keyword': kw,
            'total': total,
            'counts': counts,
            'first_year': first_year,
            'last_year': last_year,
            'early': early,
            'late': late,
            'growth_rate': growth if growth != float('inf') else 999,
        })

    gdf = pd.DataFrame(growth_data)

    # Emerging keywords: appear only in 2024-2025
    emerging = gdf[(gdf['early'] == 0) & (gdf['late'] > 0)].sort_values('total', ascending=False)
    print(f"\nEmerging keywords (only 2024-2025): {len(emerging)}")
    for _, row in emerging.head(15).iterrows():
        print(f"  {row['total']:3d}  {row['keyword']}")

    # Burst keywords: high growth rate
    bursting = gdf[(gdf['growth_rate'] > 0) & (gdf['growth_rate'] < 999) & (gdf['total'] >= 3)]
    bursting = bursting.sort_values('growth_rate', ascending=False)
    print(f"\nBursting keywords (growing, total >= 3): {len(bursting)}")
    for _, row in bursting.head(15).iterrows():
        print(f"  growth={row['growth_rate']:5.1f}x  total={row['total']:3d}  {row['keyword']}")

    # Declining keywords: present in 2022-2023 but not 2024-2025
    declining = gdf[(gdf['early'] > 0) & (gdf['late'] == 0)].sort_values('total', ascending=False)
    print(f"\nDeclining keywords (only 2022-2023): {len(declining)}")
    for _, row in declining.head(10).iterrows():
        print(f"  {row['total']:3d}  {row['keyword']}")

    # 1. Keyword burst timeline
    fig, ax = plt.subplots(figsize=FIG_WIDE)
    # Show top 15 keywords by total frequency across years
    top_kws = gdf.sort_values('total', ascending=False).head(15)
    y_positions = range(len(top_kws))

    for y_pos, (_, row) in enumerate(top_kws.iterrows()):
        counts = row['counts']
        for j, year in enumerate(years):
            if counts[j] > 0:
                ax.scatter(year, y_pos, s=counts[j] * 60 + 20,
                           c='#4e79a7', alpha=0.7, edgecolors='white', linewidths=0.5)
                ax.text(year, y_pos + 0.3, str(counts[j]),
                        ha='center', va='bottom', fontsize=7, color='#333333')

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(top_kws['keyword'].tolist(), fontsize=9)
    ax.set_xticks(years)
    ax.set_xlabel('Year')
    ax.set_title('Keyword frequency timeline (top 15 keywords)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 2026 is a partial year: the search ran on 3 June 2026, so its counts cover
    # January-May only and are not comparable with the complete years beside
    # them (reviewer R1#6). Shade the column and say so on the axis.
    if PARTIAL_YEAR in years:
        ax.axvspan(PARTIAL_YEAR - 0.45, PARTIAL_YEAR + 0.45,
                   color='#bab0ac', alpha=0.22, zorder=0)
        ax.text(PARTIAL_YEAR, ax.get_ylim()[1], PARTIAL_YEAR_NOTE,
                ha='center', va='bottom', fontsize=7, color='#666666')
    savefig(fig, 'fig_keyword_bursts.pdf')

    # 2. Emerging keywords chart
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    em_top = emerging.head(15)
    if len(em_top) > 0:
        ax.barh(range(len(em_top)), em_top['total'].values,
                color='#59a14f', alpha=0.8)
        ax.set_yticks(range(len(em_top)))
        ax.set_yticklabels(em_top['keyword'].tolist(), fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency (2024–2025)')
        ax.set_title('Emerging keywords (absent in 2022–2023)')
    else:
        ax.text(0.5, 0.5, 'No emerging keywords detected',
                ha='center', va='center', transform=ax.transAxes)
    savefig(fig, 'fig_emerging_keywords.pdf')

    # Save results
    results = {
        'total_keywords_gte2': len(freq_kws),
        'emerging_count': len(emerging),
        'emerging_top': emerging.head(20)[['keyword', 'total']].to_dict('records'),
        'bursting_count': len(bursting),
        'bursting_top': bursting.head(20)[['keyword', 'total', 'growth_rate']].to_dict('records'),
        'declining_count': len(declining),
        'declining_top': declining.head(10)[['keyword', 'total']].to_dict('records'),
    }
    with open(os.path.join(ANALYSIS_DIR, 'keyword_burst_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nDone.")

if __name__ == '__main__':
    main()
