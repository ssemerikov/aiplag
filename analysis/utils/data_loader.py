"""Load and clean the corpus.

Historically Scopus-only; now the canonical corpus is the merged, screened
Scopus+WoS *core* corpus produced by ``00_build_corpus.py`` and written to
``corpus_clean.pkl``. ``derive_fields`` is shared so both paths produce an
identical schema for downstream scripts 02-11.
"""
import os
import re
import pickle
import pandas as pd
import numpy as np

try:
    from utils.wos_loader import normalize_country
except ImportError:  # when imported as a top-level module
    from wos_loader import normalize_country

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), '..')
PICKLE_PATH = os.path.join(ANALYSIS_DIR, 'corpus_clean.pkl')


def parse_author_ids(id_str):
    if pd.isna(id_str):
        return []
    return [x.strip() for x in str(id_str).split(';') if x.strip()]


def parse_authors(author_str):
    if pd.isna(author_str):
        return []
    return [x.strip() for x in str(author_str).split(';') if x.strip()]


def parse_keywords(kw_str):
    if pd.isna(kw_str):
        return []
    return [x.strip().lower() for x in str(kw_str).split(';') if x.strip()]


def parse_affiliations(aff_str):
    if pd.isna(aff_str):
        return []
    return [x.strip() for x in str(aff_str).split(';') if x.strip()]


def extract_countries(aff_str):
    if pd.isna(aff_str):
        return []
    parts = str(aff_str).split(';')
    countries = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Country is typically the last comma-separated element
        tokens = [t.strip() for t in part.split(',')]
        if tokens:
            country = tokens[-1].strip()
            # Clean up common artifacts
            country = re.sub(r'\s*\(.*?\)\s*', '', country).strip()
            country = normalize_country(country)
            if country and len(country) > 1 and not country.isdigit():
                countries.append(country)
    return list(set(countries))


def extract_institutions(aff_str):
    if pd.isna(aff_str):
        return []
    parts = str(aff_str).split(';')
    institutions = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        tokens = [t.strip() for t in part.split(',')]
        if len(tokens) >= 2:
            # First element is typically the institution name
            inst = tokens[0].strip()
            if inst and len(inst) > 3:
                institutions.append(inst)
    return list(set(institutions))


def derive_fields(df):
    """Add all derived columns the pipeline needs to a Scopus-shaped frame.

    Shared by the legacy Scopus-only loader and the merged-corpus builder so
    both yield an identical schema.
    """
    df = df.copy()
    df['Cited by'] = pd.to_numeric(df['Cited by'], errors='coerce').fillna(0).astype(int)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df[df['Year'].notna()].copy()
    df['Year'] = df['Year'].astype(int)
    df['has_abstract'] = (df['Abstract'].notna()) & (df['Abstract'] != '[No abstract available]')
    df['has_references'] = df['References'].notna() & (df['References'].astype(str).str.strip() != '')

    # Parse structured fields
    df['author_list'] = df['Author full names'].apply(parse_authors)
    df['author_id_list'] = df['Author(s) ID'].apply(parse_author_ids)
    df['author_keywords'] = df['Author Keywords'].apply(parse_keywords)
    df['index_keywords'] = df['Index Keywords'].apply(parse_keywords)
    df['all_keywords'] = df.apply(
        lambda r: list(set(r['author_keywords'] + r['index_keywords'])), axis=1
    )
    df['country_list'] = df['Affiliations'].apply(extract_countries)
    df['institution_list'] = df['Affiliations'].apply(extract_institutions)
    df['author_count'] = df['author_list'].apply(len)
    df['is_OA'] = df['Open Access'].notna() & (df['Open Access'].astype(str).str.strip() != '')

    df = df.reset_index(drop=True)
    df['paper_id'] = ['P' + str(i + 1).zfill(3) for i in range(len(df))]
    return df


def load_and_clean(force_reload=False):
    """Load the canonical corpus (merged Scopus+WoS core if built).

    ``corpus_clean.pkl`` is written by ``00_build_corpus.py`` and consumed by
    all downstream scripts. The legacy Scopus-only path is kept as a fallback
    when no built corpus exists.
    """
    if os.path.exists(PICKLE_PATH) and not force_reload:
        with open(PICKLE_PATH, 'rb') as f:
            return pickle.load(f)

    # Legacy fallback: build from the original Scopus CSV.
    csv_path = os.path.join(DATA_DIR, 'scopus.csv')
    df = derive_fields(pd.read_csv(csv_path))
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(df, f)
    return df


def get_abstracts_df(df):
    """Return subset with valid abstracts for NLP tasks."""
    return df[df['has_abstract']].copy()


def summary_stats(df):
    stats = {
        'total_papers': int(len(df)),
        'with_abstracts': int(df['has_abstract'].sum()),
        'with_references': int(df['has_references'].sum()),
        'with_author_keywords': int((df['Author Keywords'].notna()).sum()),
        'with_index_keywords': int((df['Index Keywords'].notna()).sum()),
        'unique_authors': int(len(set(aid for ids in df['author_id_list'] for aid in ids))),
        'unique_countries': int(len(set(c for cs in df['country_list'] for c in cs))),
        'unique_institutions': int(len(set(i for ins in df['institution_list'] for i in ins))),
        'year_distribution': {int(k): int(v) for k, v in df['Year'].value_counts().sort_index().items()},
        'doc_type_distribution': {str(k): int(v) for k, v in df['Document Type'].value_counts().items()},
        'citation_stats': {
            'mean': round(float(df['Cited by'].mean()), 2),
            'median': float(df['Cited by'].median()),
            'max': int(df['Cited by'].max()),
            'zero_citations': int((df['Cited by'] == 0).sum()),
        }
    }
    return stats
