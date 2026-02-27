"""Script 08: Citation analysis and exploratory regression modeling."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.data_loader import load_and_clean
from utils.viz_config import savefig, FIG_SINGLE, FIG_WIDE, FIGURE_DIR, ANALYSIS_DIR

def main():
    print("=" * 60)
    print("08 — Citation Analysis & Modeling")
    print("=" * 60)

    df = load_and_clean()

    # Load topic assignments if available
    topic_path = os.path.join(ANALYSIS_DIR, 'topic_assignments.csv')
    if os.path.exists(topic_path):
        topics_df = pd.read_csv(topic_path)
        df = df.merge(topics_df, on='paper_id', how='left')
        df['topic'] = df['topic'].fillna(-1).astype(int)
    else:
        df['topic'] = 0

    # Descriptive statistics
    print(f"\nCitation distribution:")
    print(df['Cited by'].describe())
    print(f"\nZero citations: {(df['Cited by'] == 0).sum()}/{len(df)}")

    # Citation velocity by year
    year_cite = df.groupby('Year')['Cited by'].agg(['mean', 'median', 'sum', 'count'])
    print(f"\nCitations by year:")
    print(year_cite)

    # 1. Citations by year boxplot
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    years = sorted(df['Year'].unique())
    data_by_year = [df[df['Year'] == y]['Cited by'].values for y in years]
    bp = ax.boxplot(data_by_year, labels=[str(y) for y in years], patch_artist=True)
    from utils.viz_config import YEAR_COLORS
    for patch, year in zip(bp['boxes'], years):
        patch.set_facecolor(YEAR_COLORS.get(year, '#999999'))
        patch.set_alpha(0.7)
    ax.set_xlabel('Publication year')
    ax.set_ylabel('Citation count')
    ax.set_title('Citation distribution by year')
    savefig(fig, 'fig_citations_by_year.pdf')

    # Prepare regression data
    reg_df = df[['Cited by', 'Year', 'author_count', 'is_OA', 'Document Type', 'topic']].copy()
    reg_df['is_OA'] = reg_df['is_OA'].astype(int)
    reg_df['keyword_count'] = df['all_keywords'].apply(len)
    reg_df['years_since_pub'] = 2025 - reg_df['Year']

    # Negative binomial regression
    print("\n--- Negative Binomial Regression ---")
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import glm
        import statsmodels.formula.api as smf

        # Create dummy for doc type
        reg_df['is_article'] = (reg_df['Document Type'] == 'Article').astype(int)

        formula = 'Q("Cited by") ~ years_since_pub + author_count + is_OA + is_article + keyword_count'

        try:
            model = smf.negativebinomial(formula, data=reg_df)
            result = model.fit(disp=False, maxiter=100)
            print(result.summary())

            reg_table = {
                'formula': formula,
                'n': int(result.nobs),
                'aic': round(result.aic, 2),
                'bic': round(result.bic, 2),
                'coefficients': {},
            }
            for param in result.params.index:
                reg_table['coefficients'][str(param)] = {
                    'coef': round(float(result.params[param]), 4),
                    'std_err': round(float(result.bse[param]), 4),
                    'p_value': round(float(result.pvalues[param]), 4),
                }
        except Exception as e:
            print(f"NB regression failed: {e}")
            print("Falling back to Poisson regression...")
            model = smf.poisson(formula, data=reg_df)
            result = model.fit(disp=False, maxiter=100)
            print(result.summary())

            reg_table = {
                'formula': formula,
                'model': 'poisson_fallback',
                'n': int(result.nobs),
                'coefficients': {},
            }
            for param in result.params.index:
                reg_table['coefficients'][str(param)] = {
                    'coef': round(float(result.params[param]), 4),
                    'std_err': round(float(result.bse[param]), 4),
                    'p_value': round(float(result.pvalues[param]), 4),
                }

        # 2. Regression coefficient plot
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        params = [p for p in result.params.index if p != 'Intercept' and p != 'alpha']
        coefs = [float(result.params[p]) for p in params]
        errors = [1.96 * float(result.bse[p]) for p in params]
        pvals = [float(result.pvalues[p]) for p in params]

        y_pos = range(len(params))
        colors = ['#4e79a7' if p < 0.05 else '#bab0ac' for p in pvals]
        ax.barh(y_pos, coefs, xerr=errors, color=colors, alpha=0.8, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Coefficient (95% CI)')
        ax.set_title('Regression: predictors of citation count')
        # Legend
        from matplotlib.patches import Patch
        ax.legend([Patch(color='#4e79a7'), Patch(color='#bab0ac')],
                  ['p < 0.05', 'p >= 0.05'], loc='lower right', fontsize=8)
        savefig(fig, 'fig_citation_regression.pdf')

    except ImportError:
        print("statsmodels not available — skipping regression")
        reg_table = {'error': 'statsmodels not available'}

    # Save results
    results = {
        'citation_stats': {
            'mean': round(float(df['Cited by'].mean()), 2),
            'median': float(df['Cited by'].median()),
            'max': int(df['Cited by'].max()),
            'zero_count': int((df['Cited by'] == 0).sum()),
        },
        'by_year': year_cite.to_dict(),
        'regression': reg_table,
    }
    with open(os.path.join(ANALYSIS_DIR, 'citation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nDone.")

if __name__ == '__main__':
    main()
