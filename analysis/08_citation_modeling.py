"""Script 08: Citation analysis and exploratory regression modeling.

Citation counts are modelled as *associations*, never causally: the available
covariates cannot control for paper quality, author reputation, venue prestige,
or self-citation, all of which plausibly confound the open-access coefficient.

Three models are reported (revision R2):

  M1  NB with ``years_since_pub`` as a covariate  — comparable to the R1 table.
  M2  NB with a log-exposure *offset*             — the preferred specification.
  M3  ZINB                                        — robustness check.

M2 is preferred because a zero in this corpus is a *short-exposure* zero, not a
structural one: a paper indexed in April 2026 had roughly two months to accrue
citations by the 3 June 2026 search date. Modelling exposure explicitly is the
statistically appropriate response to excess zeros of that kind; zero-inflation
posits a latent "never-citable" class, which is hard to justify for indexed
research literature. M3 is nonetheless fitted and reported so readers can judge.

See Thelwall & Wilson (2014) on regression methods for citation data and Wilson
(2015) on why the Vuong test should not be used to test for zero-inflation --
we therefore compare AIC/BIC and predicted-versus-observed zero counts instead.
"""
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

# Corpus search date: 3 June 2026 -> day 154 of the year.
SEARCH_DATE_DECIMAL = 2026 + 153.0 / 365.0
# Fraction of 2026 elapsed at the search date (used for the partial-year midpoint).
FRACTION_2026_OBSERVED = 153.0 / 365.0


def publication_midpoint(year):
    """Decimal-year midpoint of a publication year's observed window.

    Scopus and WoS exports carry only a publication *year*, so exposure is
    approximated by assuming uniform publication within the observed window.
    2022-2025 are complete years (midpoint = year + 0.5); 2026 is truncated at
    the search date, so its midpoint is half of the observed fraction.
    """
    if year >= 2026:
        return year + FRACTION_2026_OBSERVED / 2.0
    return year + 0.5


def nb_predicted_zeros(result, mu):
    """Expected number of zero counts under a fitted NB2 model.

    For NB2, P(Y=0) = (1 / (1 + alpha*mu))^(1/alpha). Comparing the sum of these
    probabilities with the observed zero count is a direct test of whether the
    data actually contain *excess* zeros that a plain NB cannot accommodate.
    """
    alpha = float(result.params.get('alpha', np.nan))
    if not np.isfinite(alpha) or alpha <= 0:
        return float('nan')
    p0 = np.power(1.0 / (1.0 + alpha * mu), 1.0 / alpha)
    return float(np.sum(p0))


def extract_coefficients(result):
    """Coefficient table as a plain dict, with 95% CIs."""
    out = {}
    conf = result.conf_int()
    for param in result.params.index:
        out[str(param)] = {
            'coef': round(float(result.params[param]), 4),
            'std_err': round(float(result.bse[param]), 4),
            'z': round(float(result.tvalues[param]), 4),
            'p_value': round(float(result.pvalues[param]), 4),
            'ci_low': round(float(conf.loc[param, 0]), 4),
            'ci_high': round(float(conf.loc[param, 1]), 4),
        }
    return out

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
    reg_df['venue'] = df['Source title'].fillna('(unknown venue)').astype(str).str.strip()

    # Exposure: years between the assumed publication midpoint and the search
    # date. Replaces the R1 code's `2025 - Year`, which assigned 2026 papers an
    # impossible -1. Because that expression was linear in Year it shifted only
    # the intercept, so the R1 coefficients were unaffected -- but it is wrong.
    reg_df['exposure_years'] = SEARCH_DATE_DECIMAL - reg_df['Year'].apply(publication_midpoint)
    reg_df['exposure_years'] = reg_df['exposure_years'].clip(lower=0.05)
    reg_df['log_exposure'] = np.log(reg_df['exposure_years'])
    # Kept for M1 so the revised table stays comparable with the R1 table.
    reg_df['years_since_pub'] = SEARCH_DATE_DECIMAL - reg_df['Year'] - 0.5

    print("\nExposure by publication year (years to 3 June 2026 search date):")
    print(reg_df.groupby('Year')['exposure_years'].agg(['min', 'max', 'count']).round(3))

    # Open-access provenance check: `is_OA` is derived from the Scopus
    # "Open Access" field, so WoS-only records could in principle be
    # systematically mislabelled. Quantify how much of the corpus that affects.
    oa_provenance = {}
    if 'source' in df.columns:
        crosstab = pd.crosstab(df['source'], df['is_OA'])
        oa_provenance = {
            str(src): {str(k): int(v) for k, v in row.items()}
            for src, row in crosstab.iterrows()
        }
        print("\nOpen-access flag by source database:")
        print(crosstab)

    observed_zeros = int((reg_df['Cited by'] == 0).sum())

    print("\n--- Citation regression models ---")
    models = {}
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

        reg_df['is_article'] = (reg_df['Document Type'] == 'Article').astype(int)

        base_terms = 'author_count + is_OA + is_article + keyword_count'
        f_m1 = f'Q("Cited by") ~ years_since_pub + {base_terms}'
        f_m2 = f'Q("Cited by") ~ {base_terms}'

        # -- M1: NB, years-since-publication as covariate (R1-comparable) ------
        m1 = smf.negativebinomial(f_m1, data=reg_df).fit(disp=False, maxiter=200)
        print("\n[M1] NB with years_since_pub covariate")
        print(m1.summary())
        models['m1_nb_covariate'] = {
            'label': 'NB (years since publication as covariate)',
            'formula': f_m1,
            'n': int(m1.nobs),
            'aic': round(float(m1.aic), 2),
            'bic': round(float(m1.bic), 2),
            'llf': round(float(m1.llf), 2),
            'coefficients': extract_coefficients(m1),
            'predicted_zeros': round(nb_predicted_zeros(m1, m1.predict()), 1),
        }

        # M1 with venue-clustered standard errors: a partial control for venue
        # prestige, which the reviewers correctly note is otherwise unmodelled.
        try:
            m1_cl = smf.negativebinomial(f_m1, data=reg_df).fit(
                disp=False, maxiter=200, cov_type='cluster',
                cov_kwds={'groups': reg_df['venue'].values})
            models['m1_nb_venue_clustered'] = {
                'label': 'NB, standard errors clustered by venue',
                'n_venues': int(reg_df['venue'].nunique()),
                'coefficients': extract_coefficients(m1_cl),
            }
            print(f"\n[M1-clustered] venue-clustered SEs over "
                  f"{reg_df['venue'].nunique()} venues")
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"Venue-clustered SEs unavailable: {e}")

        # -- M2: NB with log-exposure offset (preferred) -----------------------
        m2 = smf.negativebinomial(
            f_m2, data=reg_df, offset=reg_df['log_exposure'].values
        ).fit(disp=False, maxiter=200)
        print("\n[M2] NB with log-exposure offset (preferred specification)")
        print(m2.summary())
        models['m2_nb_offset'] = {
            'label': 'NB with log-exposure offset',
            'formula': f_m2 + ' + offset(log_exposure)',
            'n': int(m2.nobs),
            'aic': round(float(m2.aic), 2),
            'bic': round(float(m2.bic), 2),
            'llf': round(float(m2.llf), 2),
            'coefficients': extract_coefficients(m2),
            'predicted_zeros': round(nb_predicted_zeros(m2, m2.predict()), 1),
        }

        # -- M3: ZINB robustness check ----------------------------------------
        try:
            zinb = ZeroInflatedNegativeBinomialP.from_formula(
                f_m1, data=reg_df, p=2
            ).fit(method='bfgs', maxiter=500, disp=False)
            print("\n[M3] Zero-inflated NB (robustness)")
            print(zinb.summary())
            models['m3_zinb'] = {
                'label': 'Zero-inflated NB (robustness check)',
                'formula': f_m1,
                'n': int(zinb.nobs),
                'aic': round(float(zinb.aic), 2),
                'bic': round(float(zinb.bic), 2),
                'llf': round(float(zinb.llf), 2),
                'coefficients': extract_coefficients(zinb),
                'converged': bool(zinb.mle_retvals.get('converged', False)),
            }
        except Exception as e:
            print(f"ZINB did not fit: {e}")
            models['m3_zinb'] = {'error': str(e)}

        # Preferred model drives the coefficient figure.
        result = m2

        # 2. Regression coefficient plot
        fig, ax = plt.subplots(figsize=FIG_SINGLE)
        params = [p for p in result.params.index if p not in ('Intercept', 'alpha')]
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
        ax.set_title('Associations with citation count (NB, exposure offset)')
        from matplotlib.patches import Patch
        ax.legend([Patch(color='#4e79a7'), Patch(color='#bab0ac')],
                  ['p < 0.05', 'p >= 0.05'], loc='lower right', fontsize=8)
        savefig(fig, 'fig_citation_regression.pdf')

    except ImportError:
        print("statsmodels not available — skipping regression")
        models = {'error': 'statsmodels not available'}

    # Zero-count diagnostic: are there *excess* zeros a plain NB cannot absorb?
    zero_diagnostic = {'observed_zeros': observed_zeros, 'n': int(len(reg_df))}
    for key in ('m1_nb_covariate', 'm2_nb_offset'):
        if key in models and 'predicted_zeros' in models[key]:
            zero_diagnostic[f'{key}_predicted_zeros'] = models[key]['predicted_zeros']
    print(f"\nZero-count diagnostic: observed {observed_zeros}, "
          f"NB-predicted {zero_diagnostic.get('m2_nb_offset_predicted_zeros')}")

    # Save results
    results = {
        'citation_stats': {
            'mean': round(float(df['Cited by'].mean()), 2),
            'median': float(df['Cited by'].median()),
            'max': int(df['Cited by'].max()),
            'zero_count': observed_zeros,
            'zero_share': round(observed_zeros / len(df), 4),
        },
        'by_year': year_cite.to_dict(),
        'exposure': {
            'search_date': '2026-06-03',
            'search_date_decimal': round(SEARCH_DATE_DECIMAL, 4),
            'note': ('Exports carry only a publication year; exposure assumes '
                     'uniform publication within each observed window.'),
            'by_year': {int(y): round(float(v), 3) for y, v in
                        reg_df.groupby('Year')['exposure_years'].first().items()},
        },
        'oa_provenance': oa_provenance,
        'zero_diagnostic': zero_diagnostic,
        'unmodelled_confounders': [
            'paper quality (no proxy for novelty, rigour, or clarity)',
            'author reputation and prior citation record',
            'venue prestige (partially addressed via venue-clustered SEs)',
            'self-citations (not removable: "Cited by" is an aggregate count '
            'with no citing-paper data in the export)',
            'field-specific citation practices',
        ],
        'models': models,
        # Retained for backward compatibility with any consumer of the R1 schema.
        'regression': models.get('m2_nb_offset', {}),
    }
    with open(os.path.join(ANALYSIS_DIR, 'citation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nDone.")

if __name__ == '__main__':
    main()
