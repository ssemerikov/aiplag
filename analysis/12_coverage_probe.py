"""Script 12: Corpus coverage and search-recall probes.

Written for revision R2 in response to two reviewer comments:

  R1#1  Excluding preprint servers (arXiv) and non-English venues is not merely
        a limitation but a potential threat to the validity of the structural
        claims. Either add a preprint server or state explicitly that all
        structural claims are conditional on Scopus+WoS coverage.

  R2#2  The Boolean query requires a general AI term to co-occur with a
        plagiarism term, so evaluations that name a specific tool (Turnitin,
        GPTZero, ...) without saying "artificial intelligence" may be missed.

Rather than answer either comment with prose, this script *measures* the gaps:

  1. Preprint coverage  -- how much relevant preprint literature exists outside
                           the indexed corpus, and how much of it the extended
                           set already contains.
  2. Tool-name recall   -- per detection tool, how many works name it alongside
                           a plagiarism/detection term, and how many of those
                           the extended set already contains.
  3. Language coverage  -- the non-English share of the retrieved records, read
                           from the raw Scopus export (the merged corpus drops
                           the language column).

Preprints are counted through OpenAlex rather than the arXiv API. The arXiv API
returned HTTP 429 on every query even at a 3.5 s interval, which makes it
unusable for a reproducible probe; OpenAlex indexes arXiv as a repository source
and covers other preprint servers besides, so it answers the reviewer's question
on strictly broader evidence. Every stage degrades to a recorded error rather
than aborting the run, so a network failure costs one number, not all of them.

Output: analysis/coverage_probe.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter

import pandas as pd

from utils.data_loader import load_and_clean
from utils.viz_config import ANALYSIS_DIR

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

OPENALEX_API = 'https://api.openalex.org/works'
CONTACT_EMAIL = os.environ.get('OPENALEX_CONTACT_EMAIL', 'anonymous@example.org')
SEARCH_DATE = '2026-06-03'
DATE_FROM = '2022-01-01'

# Detection tools and services a paper might name *instead of* a generic AI
# term -- exactly the recall gap Reviewer #2 describes.
TOOL_NAMES = [
    'Turnitin', 'GPTZero', 'ZeroGPT', 'Originality.ai', 'Copyleaks',
    'iThenticate', 'QuillBot', 'Crossref Similarity Check', 'DetectGPT',
    'Grammarly', 'Compilatio', 'Urkund', 'PlagScan',
]

# Generic AI terms the manuscript's Boolean query requires.
GENERIC_AI_TERMS = [
    'artificial intelligence', 'generative ai', 'large language model',
    'machine learning', 'deep learning', 'chatgpt', 'llm', 'neural network',
    'natural language processing',
]


def normalise_title(title):
    """Lowercase alphanumeric-only title key, for cross-source matching."""
    if not title:
        return ''
    return re.sub(r'[^a-z0-9]+', '', str(title).lower())


def normalise_doi(doi):
    if not doi or (isinstance(doi, float) and pd.isna(doi)):
        return ''
    d = str(doi).strip().lower()
    for prefix in ('https://doi.org/', 'http://doi.org/', 'doi:'):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.strip()


def http_get(url, timeout=45, retries=3, backoff=8):
    """GET with retry-on-429/5xx. Returns bytes, or raises the last error."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': f'aiplag-coverage-probe (mailto:{CONTACT_EMAIL})'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # includes HTTPError 429
            last = e
            wait = backoff * (attempt + 1)
            print(f"    retry {attempt + 1}/{retries} after {type(e).__name__}: waiting {wait}s")
            time.sleep(wait)
    raise last


# ---------------------------------------------------------------------------
# 1. Preprint and tool-recall probes (OpenAlex)
# ---------------------------------------------------------------------------

def openalex_fetch(filt, max_records=1000):
    """Fetch works matching an OpenAlex filter, following the result cursor.

    Returns (records, reported_total). Records carry only the fields the
    overlap analysis needs.
    """
    fields = 'id,doi,title,type,publication_year,primary_location'
    # Must not include a space: urllib rejects URLs containing raw spaces, so
    # spaces have to percent-encode. Colons and commas are OpenAlex filter
    # syntax and must survive intact.
    safe_chars = ':,|'
    cursor, out, total = '*', [], None
    while cursor and len(out) < max_records:
        enc_filter = urllib.parse.quote(filt, safe=safe_chars)
        enc_cursor = urllib.parse.quote(cursor)
        url = (f"{OPENALEX_API}?filter={enc_filter}"
               f"&per-page=200&cursor={enc_cursor}"
               f"&select={fields}&mailto={CONTACT_EMAIL}")
        data = json.loads(http_get(url).decode('utf-8'))
        if total is None:
            total = int(data['meta']['count'])
        results = data.get('results', [])
        if not results:
            break
        for w in results:
            loc = w.get('primary_location') or {}
            src = (loc.get('source') or {}).get('display_name') or ''
            out.append({
                'id': w.get('id'), 'doi': normalise_doi(w.get('doi')),
                'title': w.get('title') or '', 'type': w.get('type'),
                'year': w.get('publication_year'), 'source': src,
            })
        cursor = data['meta'].get('next_cursor')
        time.sleep(0.3)
    return out, (total if total is not None else len(out))


def _overlap(records, core_dois, core_titles, ext_dois, ext_titles):
    """Classify fetched records by whether the study's corpora contain them."""
    in_core = in_ext = 0
    for r in records:
        doi, tkey = r['doi'], normalise_title(r['title'])
        if (doi and doi in core_dois) or (tkey and tkey in core_titles):
            in_core += 1
        if (doi and doi in ext_dois) or (tkey and tkey in ext_titles):
            in_ext += 1
    n = len(records) or 1
    return {
        'fetched': len(records),
        'in_core_corpus': in_core,
        'in_extended_set': in_ext,
        'outside_extended_set': len(records) - in_ext,
        'share_outside_extended': round((len(records) - in_ext) / n, 4),
    }


def probe_preprints(core_dois, core_titles, ext_dois, ext_titles, max_records=1000):
    """Measure the preprint literature sitting outside the indexed corpus.

    Reviewer R1#1 asks specifically about arXiv. The arXiv API rate-limits too
    aggressively for a reproducible probe (it returned HTTP 429 on every query
    even at a 3.5 s interval), so preprints are counted through OpenAlex, which
    indexes arXiv as a repository source and additionally covers other preprint
    servers. That is a superset of the reviewer's request, and it is stable.
    """
    concept = ('title_and_abstract.search:("AI-generated text" OR '
               '"plagiarism detection" OR "machine-generated text" OR '
               '"AI detection") AND (detection OR detector OR plagiarism)')
    window = f'from_publication_date:{DATE_FROM},to_publication_date:{SEARCH_DATE}'

    out = {'concept_query': concept, 'date_window': f'{DATE_FROM} to {SEARCH_DATE}'}
    try:
        all_recs, all_total = openalex_fetch(f'{concept},{window}', max_records)
        out['all_types_total'] = all_total
    except Exception as e:
        out['all_types_error'] = str(e)
        all_recs = []

    try:
        pre_recs, pre_total = openalex_fetch(f'{concept},{window},type:preprint', max_records)
        out['preprint_total'] = pre_total
        out['preprint_overlap'] = _overlap(pre_recs, core_dois, core_titles,
                                           ext_dois, ext_titles)
        arxiv = [r for r in pre_recs if 'arxiv' in (r['source'] or '').lower()]
        out['arxiv_hosted_in_sample'] = len(arxiv)
        out['arxiv_overlap'] = _overlap(arxiv, core_dois, core_titles,
                                        ext_dois, ext_titles)
        by_year = Counter(r['year'] for r in pre_recs if r['year'])
        out['preprints_by_year'] = {int(k): int(v) for k, v in sorted(by_year.items())}
        print(f"  preprints matching the concept pair: {pre_total} "
              f"({out['preprint_overlap']['share_outside_extended']:.1%} outside "
              f"the extended set)")
    except Exception as e:
        out['preprint_error'] = str(e)
        print(f"  preprint probe FAILED: {e}")

    if 'preprint_total' in out and 'all_types_total' in out and out['all_types_total']:
        out['preprint_share_of_matching_literature'] = round(
            out['preprint_total'] / out['all_types_total'], 4)

    out['note'] = ('Preprints are counted via OpenAlex rather than the arXiv API, '
                   'which rate-limited every request. The figure bounds how much '
                   'relevant literature the Scopus+WoS corpus cannot see.')
    return out


# ---------------------------------------------------------------------------
# 2. Tool-name recall probe (OpenAlex)
# ---------------------------------------------------------------------------

def probe_tool_recall(core_dois, core_titles, ext_dois, ext_titles, cap=400):
    """Measure the recall gap Reviewer #2 describes, per tool.

    For each tool, fetch the OpenAlex works that name it alongside a
    plagiarism/detection term in the 2022-2026 window, then check how many the
    study's extended set already contains. Works outside the extended set are
    the concrete shortfall; reporting them per tool makes the gap attributable
    rather than a lump sum.
    """
    results, errors, per_tool = {}, [], {}
    window = f'from_publication_date:{DATE_FROM},to_publication_date:{SEARCH_DATE}'

    for tool in TOOL_NAMES:
        filt = (f'title_and_abstract.search:"{tool}" AND (plagiarism OR detection),'
                f'{window}')
        try:
            recs, total = openalex_fetch(filt, cap)
            ov = _overlap(recs, core_dois, core_titles, ext_dois, ext_titles)
            per_tool[tool] = {'openalex_total': total, **ov}
            print(f"    {tool:<26} {total:>6} works, "
                  f"{ov['in_extended_set']:>4} already in extended set")
        except Exception as e:
            errors.append({'tool': tool, 'error': str(e)})
            print(f"    {tool:<26} FAILED: {e}")
        time.sleep(0.4)

    results['per_tool'] = per_tool
    if per_tool:
        results['totals'] = {
            'openalex_total': sum(v['openalex_total'] for v in per_tool.values()),
            'fetched': sum(v['fetched'] for v in per_tool.values()),
            'in_extended_set': sum(v['in_extended_set'] for v in per_tool.values()),
            'outside_extended_set': sum(v['outside_extended_set'] for v in per_tool.values()),
        }
        results['totals']['note'] = ('Sums double-count works naming more than one '
                                     'tool, so they are an upper bound.')
    results['fetch_cap_per_tool'] = cap
    results['tool_names_probed'] = TOOL_NAMES
    results['generic_ai_terms'] = GENERIC_AI_TERMS
    results['errors'] = errors
    return results


def probe_corpus_tool_mentions(df):
    """How many core-corpus papers name a tool, and would the query have caught
    them without the generic AI term? Answerable offline, from our own data."""
    text = (df['Title'].fillna('') + ' ' + df['Abstract'].fillna('') + ' ' +
            df['Author Keywords'].fillna('') + ' ' + df['Index Keywords'].fillna(''))
    text = text.str.lower()

    ai_pattern = '|'.join(re.escape(t) for t in GENERIC_AI_TERMS)
    has_ai = text.str.contains(ai_pattern, regex=True, na=False)

    per_tool, tool_any = {}, pd.Series(False, index=text.index)
    for tool in TOOL_NAMES:
        hit = text.str.contains(re.escape(tool.lower()), regex=True, na=False)
        per_tool[tool] = int(hit.sum())
        tool_any |= hit

    tool_without_ai = int((tool_any & ~has_ai).sum())
    return {
        'core_papers': int(len(df)),
        'papers_naming_a_tool': int(tool_any.sum()),
        'papers_naming_tool_without_generic_ai_term': tool_without_ai,
        'share_tool_named_lacking_ai_term': (
            round(tool_without_ai / int(tool_any.sum()), 4) if tool_any.sum() else None),
        'per_tool_counts_in_core': per_tool,
        'note': ('Papers in the core corpus that name a tool but no generic AI '
                 'term entered via the seed corpus or via a plagiarism term, '
                 'which bounds the recall loss the reviewer describes.'),
    }


# ---------------------------------------------------------------------------
# 3. Language coverage
# ---------------------------------------------------------------------------

def probe_language():
    """Non-English share of the raw Scopus export.

    `Language of Original Document` is dropped when the Scopus and WoS frames
    are merged, so it has to be read from the raw export.
    """
    path = os.path.join(DATA_DIR, 'scopus_2026jun.csv')
    if not os.path.exists(path):
        return {'error': f'raw export not found: {path}'}

    df = pd.read_csv(path, usecols=lambda c: c in ('Language of Original Document', 'Year'),
                     low_memory=False)
    col = 'Language of Original Document'
    if col not in df.columns:
        return {'error': 'language column absent from export'}

    langs = df[col].fillna('(unspecified)').astype(str).str.split(';').str[0].str.strip()
    counts = Counter(langs)
    total = int(len(langs))
    english = int(sum(v for k, v in counts.items() if k.lower() == 'english'))
    return {
        'records_in_raw_scopus_export': total,
        'english': english,
        'non_english': total - english,
        'non_english_share': round((total - english) / total, 4) if total else None,
        'top_languages': dict(counts.most_common(12)),
        'note': ('Scopus indexes non-English work but under-represents venues '
                 'published wholly in other languages (e.g. CNKI). This share is '
                 'a lower bound on the language gap, not an estimate of it.'),
    }


def main():
    print("=" * 60)
    print("12 — Coverage and recall probes")
    print("=" * 60)

    df = load_and_clean()
    corpus_dois = {normalise_doi(d) for d in df['DOI'].fillna('') if normalise_doi(d)}
    corpus_titles = {normalise_title(t) for t in df['Title'].fillna('') if t}
    print(f"Core corpus: {len(df)} papers, {len(corpus_dois)} DOIs")

    # The extended set is the right comparison for a coverage question: it is
    # everything the search retrieved and screened, not just the focused core.
    ext_dois, ext_titles = set(corpus_dois), set(corpus_titles)
    ext_path = os.path.join(ANALYSIS_DIR, 'corpus_extended.pkl')
    if os.path.exists(ext_path):
        # Trusted local artefact written by 00_build_corpus.py in this repo;
        # the same pickle cache the rest of the pipeline already loads.
        import pickle
        with open(ext_path, 'rb') as f:
            ext = pickle.load(f)
        ext_dois |= {normalise_doi(d) for d in ext['DOI'].fillna('') if normalise_doi(d)}
        ext_titles |= {normalise_title(t) for t in ext['Title'].fillna('') if t}
        print(f"Extended set: {len(ext)} records, {len(ext_dois)} DOIs")

    out = {
        'search_date': SEARCH_DATE,
        'core_corpus_size': int(len(df)),
        'core_corpus_dois': len(corpus_dois),
        'extended_set_dois': len(ext_dois),
    }

    print("\n[1/4] Corpus-internal tool-name check (offline)")
    out['corpus_tool_mentions'] = probe_corpus_tool_mentions(df)
    print(f"  {out['corpus_tool_mentions']['papers_naming_a_tool']} core papers name a tool; "
          f"{out['corpus_tool_mentions']['papers_naming_tool_without_generic_ai_term']} "
          f"without any generic AI term")

    print("\n[2/4] Language coverage (raw Scopus export)")
    out['language_coverage'] = probe_language()
    lc = out['language_coverage']
    if 'non_english_share' in lc:
        print(f"  non-English: {lc['non_english']}/{lc['records_in_raw_scopus_export']} "
              f"({lc['non_english_share']:.2%})")

    print("\n[3/4] OpenAlex tool-name recall probe")
    try:
        out['tool_recall_openalex'] = probe_tool_recall(
            corpus_dois, corpus_titles, ext_dois, ext_titles)
    except Exception as e:
        out['tool_recall_openalex'] = {'error': str(e)}
        print(f"  FAILED: {e}")

    print("\n[4/4] Preprint coverage probe (OpenAlex)")
    try:
        out['preprint_coverage'] = probe_preprints(
            corpus_dois, corpus_titles, ext_dois, ext_titles)
    except Exception as e:
        out['preprint_coverage'] = {'error': str(e)}
        print(f"  FAILED: {e}")

    path = os.path.join(ANALYSIS_DIR, 'coverage_probe.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
