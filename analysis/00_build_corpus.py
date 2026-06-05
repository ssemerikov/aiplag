"""Script 00: Build the merged, screened, two-tier corpus (PRISMA).

Identification: refreshed Scopus (data/scopus_2026jun.csv) + WoS Core Collection
(data/wos_*.txt), both retrieved 2026-06-03 with the identical Boolean string.

Funnel:
  Identification -> cross-database de-duplication -> date filter (2022-2026)
  -> relevance screen (AI-term AND plagiarism/integrity-term) = EXTENDED set
  -> eligibility (topical focus: keyword OR semantic) = CORE corpus.

Outputs: corpus_core.pkl, corpus_extended.pkl, corpus_clean.pkl (== core, the
canonical path for scripts 02-11), prisma_counts.json, ref_labels.json,
core_corpus.csv / extended_corpus.csv (for the public repo).
"""
import sys, os, re, json, pickle, difflib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from utils.data_loader import derive_fields, parse_keywords, DATA_DIR, ANALYSIS_DIR, PICKLE_PATH
from utils.wos_loader import load_wos
from utils.reference_parser import scopus_refs_to_keys, wos_cr_to_keys

# --- Screening vocabulary (mirrors the executed Boolean query) ----------------
AI_TERMS = [
    'artificial intelligence', 'machine learning', 'deep learning', 'chatgpt',
    'large language model', 'generative ai', 'generative artificial intelligence',
    ' llm', 'llms', ' gpt', 'gpt-', 'neural network', 'natural language processing',
    'transformer', 'ai-generated', 'ai generated', 'ai-assisted',
]
PLAG_TERMS = [
    'plagiarism', 'academic integrity', 'academic misconduct', 'text similarity',
    'ai-generated text detection', 'ai text detection', 'generated text detection',
    'contract cheating', 'authorship', 'originality', 'text-matching',
    'academic dishonesty',
]
# Document types excluded from the core analytic corpus.
EXCLUDE_DOCTYPES = ('erratum', 'correction', 'retracted', 'retraction')
YEAR_MIN, YEAR_MAX = 2022, 2026

# Focused-core vocabulary: signals that a paper is *about* AI-assisted
# plagiarism / AI-generated-text detection, not merely AI-in-education. A paper
# enters the CORE when one of these matches its title or author keywords.
PLAG_FOCUS = ['plagiar', 'text similarity', 'text-matching', 'authorship verif',
              'authorship attribut']
AITEXT_FOCUS = ['ai-generated text', 'ai generated text', 'machine-generated text',
                'machine generated text', 'llm-generated', 'llm generated',
                'generated text detection', 'ai text detect', 'ai detector',
                'ai detection', 'gpt detect', 'chatgpt detect', 'detecting ai',
                'detect ai', 'detection of ai', 'fake text', 'synthetic text detection',
                'deepfake text']
AI_CUES = ['ai-generated', 'generative ai', 'llm', 'chatgpt', 'gpt',
           'machine-generated', 'large language model']


def _has(text, terms):
    return any(t in text for t in terms)


def focus(title, kw):
    """True if title/keywords signal a plagiarism/AI-text-detection focus."""
    t = (str(title or '') + ' ' + str(kw or '')).lower()
    if any(p in t for p in PLAG_FOCUS):
        return True
    if any(p in t for p in AITEXT_FOCUS):
        return True
    if 'detect' in t and any(c in t for c in AI_CUES):
        return True
    return False


def norm_title(t):
    t = re.sub(r'[^a-z0-9 ]', ' ', str(t).lower())
    return re.sub(r'\s+', ' ', t).strip()


def norm_doi(d):
    d = str(d or '').strip().lower()
    m = re.search(r'10\.\d{4,9}/\S+', d)
    return m.group(0).rstrip('.') if m else ''


# --- Load both databases into one canonical column set -----------------------
CANON = ['Title', 'Year', 'Abstract', 'Author full names', 'Author(s) ID',
         'Author Keywords', 'Index Keywords', 'Affiliations', 'Open Access',
         'Document Type', 'Cited by', 'DOI', 'References', 'Source title',
         'cr_list', 'source']


def load_scopus():
    df = pd.read_csv(os.path.join(DATA_DIR, 'scopus_2026jun.csv'), low_memory=False)
    df['source'] = 'Scopus'
    df['cr_list'] = None
    for c in CANON:
        if c not in df.columns:
            df[c] = '' if c != 'cr_list' else None
    return df[CANON].copy()


def load_wos_df():
    df = load_wos()
    for c in CANON:
        if c not in df.columns:
            df[c] = '' if c != 'cr_list' else None
    return df[CANON].copy()


def dedup_key(row):
    d = norm_doi(row['DOI'])
    return 'doi:' + d if d else 'ttl:' + norm_title(row['Title'])[:120]


def merge_near_titles(groups):
    """Merge DOI-less title groups whose normalized titles are near-identical."""
    title_keys = [k for k in groups if k.startswith('ttl:')]
    title_keys.sort()
    merged = {}
    canonical = {}
    by_prefix = {}
    for k in title_keys:
        by_prefix.setdefault(k[4:16], []).append(k)
    for _, ks in by_prefix.items():
        reps = []
        for k in ks:
            t = k[4:]
            hit = next((r for r in reps
                        if difflib.SequenceMatcher(None, t, r[4:]).ratio() >= 0.93), None)
            canonical[k] = hit if hit else k
            if not hit:
                reps.append(k)
    for k in groups:
        merged.setdefault(canonical.get(k, k), []).extend(groups[k])
    return merged


def pick_representative(rows):
    """Choose one record per duplicate group; prefer one WITH references,
    then Scopus (richer structured metadata); carry max citation count."""
    def score(r):
        has_ref = bool(str(r['References']).strip()) or bool(r['cr_list'])
        return (has_ref, r['source'] == 'Scopus', len(str(r['Abstract'])))
    rep = max(rows, key=score).copy()
    rep['Cited by'] = max(int(pd.to_numeric(r['Cited by'], errors='coerce') or 0) for r in rows)
    rep['in_scopus'] = any(r['source'] == 'Scopus' for r in rows)
    rep['in_wos'] = any(r['source'] == 'WoS' for r in rows)
    return rep


def main():
    print("=" * 64)
    print("00 — Build merged, screened, two-tier corpus (PRISMA)")
    print("=" * 64)
    counts = {}

    scop = load_scopus()
    wos = load_wos_df()
    counts['identified_scopus'] = len(scop)
    counts['identified_wos'] = len(wos)
    allrec = pd.concat([scop, wos], ignore_index=True)
    counts['identified_total'] = len(allrec)
    print(f"Identified: Scopus {len(scop)} + WoS {len(wos)} = {len(allrec)}")

    # De-duplicate (cross-database) by DOI, then near-identical titles
    groups = {}
    for _, row in allrec.iterrows():
        groups.setdefault(dedup_key(row), []).append(row)
    groups = merge_near_titles(groups)
    reps = [pick_representative(rows) for rows in groups.values()]
    dedup = pd.DataFrame(reps).reset_index(drop=True)
    counts['after_dedup'] = len(dedup)
    counts['cross_db_overlap'] = int((dedup['in_scopus'] & dedup['in_wos']).sum())
    counts['duplicates_removed'] = counts['identified_total'] - counts['after_dedup']
    print(f"After de-duplication: {len(dedup)} "
          f"(removed {counts['duplicates_removed']}; "
          f"Scopus∩WoS overlap {counts['cross_db_overlap']})")

    # Date filter 2022-2026
    dedup['Year'] = pd.to_numeric(dedup['Year'], errors='coerce')
    dated = dedup[(dedup['Year'] >= YEAR_MIN) & (dedup['Year'] <= YEAR_MAX)].copy()
    counts['after_date_filter'] = len(dated)
    counts['excluded_out_of_range'] = len(dedup) - len(dated)
    print(f"After date filter {YEAR_MIN}-{YEAR_MAX}: {len(dated)} "
          f"(excluded {counts['excluded_out_of_range']})")

    # Relevance screen: AI-term AND plagiarism/integrity-term -> EXTENDED
    def screen_text(r):
        return ' '.join(str(r[c]) for c in
                        ['Title', 'Abstract', 'Author Keywords', 'Index Keywords']).lower()
    dated['_txt'] = dated.apply(screen_text, axis=1)
    dated['_ai'] = dated['_txt'].apply(lambda t: _has(t, AI_TERMS))
    dated['_plag'] = dated['_txt'].apply(lambda t: _has(t, PLAG_TERMS))
    dt = dated['Document Type'].astype(str).str.lower()
    dated['_okdoc'] = ~dt.apply(lambda x: any(e in x for e in EXCLUDE_DOCTYPES))
    extended = dated[dated['_ai'] & dated['_plag'] & dated['_okdoc']].copy().reset_index(drop=True)
    counts['excluded_not_relevant'] = int((~(dated['_ai'] & dated['_plag'])).sum())
    counts['excluded_doctype'] = int((dated['_ai'] & dated['_plag'] & ~dated['_okdoc']).sum())
    counts['extended_set'] = len(extended)
    print(f"EXTENDED set (AI ∧ plagiarism, valid doctype): {len(extended)}")

    # Continuity anchor: mark records matching the original hand-validated 58
    orig = pd.read_csv(os.path.join(DATA_DIR, 'scopus_2025aug_58.csv'), low_memory=False)
    orig_doi = set(norm_doi(d) for d in orig.get('DOI', []) if norm_doi(d))
    orig_ttl = set(norm_title(t)[:120] for t in orig.get('Title', []))
    extended['_is_orig'] = extended.apply(
        lambda r: (norm_doi(r['DOI']) in orig_doi and norm_doi(r['DOI']) != '')
        or norm_title(r['Title'])[:120] in orig_ttl, axis=1)
    counts['original58_retained_in_extended'] = int(extended['_is_orig'].sum())

    # Recover original-58 papers missing from the new broad pull (continuity).
    orig_canon = orig.copy()
    orig_canon['source'] = 'Scopus'
    orig_canon['cr_list'] = None
    for c in CANON:
        if c not in orig_canon.columns:
            orig_canon[c] = '' if c != 'cr_list' else None
    orig_canon = orig_canon[CANON].copy()
    orig_canon['in_scopus'] = True
    orig_canon['in_wos'] = False
    ext_keys = set(extended.apply(dedup_key, axis=1))
    miss_mask = ~orig_canon.apply(dedup_key, axis=1).isin(ext_keys)
    missing = orig_canon[miss_mask].copy()
    for col, val in [('_is_orig', True), ('_ai', True), ('_plag', True),
                     ('_okdoc', True), ('_txt', '')]:
        missing[col] = val
    counts['original58_recovered'] = int(len(missing))
    extended = pd.concat([extended, missing], ignore_index=True)
    counts['extended_set'] = len(extended)
    print(f"Recovered {len(missing)} original papers absent from the new pull; "
          f"extended now {len(extended)}")

    # Eligibility -> CORE: detection/plagiarism topical focus (title OR keywords),
    # always retaining the hand-validated original corpus.
    is_focus = extended.apply(lambda r: focus(r['Title'], r['Author Keywords']), axis=1)
    core_mask = is_focus | extended['_is_orig']
    core = extended[core_mask].copy().reset_index(drop=True)
    counts['core_corpus'] = len(core)
    counts['core_via_focus'] = int(is_focus.sum())
    counts['original58_in_core'] = int(core['_is_orig'].sum())
    counts['original58_total'] = len(orig)
    print(f"CORE corpus (detection-focused ∪ orig-58): {len(core)} "
          f"(focus {int(is_focus.sum())}; orig-58 in core {int(core['_is_orig'].sum())}/{len(orig)})")

    # Compute per-paper canonical cited-work keys + a global label map
    label_map = {}

    def ref_keys_for(row):
        if row['source'] == 'WoS' and row['cr_list']:
            keys, labels = wos_cr_to_keys(row['cr_list'])
        else:
            keys, labels = scopus_refs_to_keys(row['References'])
        label_map.update(labels)
        return keys

    for frame in (core, extended):
        frame['ref_keys'] = frame.apply(ref_keys_for, axis=1)

    # Derive analysis fields + assign paper_id (shared schema)
    core_d = derive_fields(core)
    ext_d = derive_fields(extended)

    # Year distributions for reporting
    counts['core_year_distribution'] = {int(k): int(v) for k, v in core_d['Year'].value_counts().sort_index().items()}
    counts['extended_year_distribution'] = {int(k): int(v) for k, v in ext_d['Year'].value_counts().sort_index().items()}
    counts['core_with_abstract'] = int(core_d['has_abstract'].sum())
    counts['core_with_references'] = int(core_d['has_references'].sum())

    # Context statistics on the broad EXTENDED set (for the coverage paragraph)
    from collections import Counter
    ext_countries = Counter(c for cs in ext_d['country_list'] for c in cs)
    core_countries = Counter(c for cs in core_d['country_list'] for c in cs)
    counts['extended_total'] = int(len(ext_d))
    counts['extended_unique_countries'] = len(ext_countries)
    counts['extended_top_countries'] = {k: int(v) for k, v in ext_countries.most_common(12)}
    counts['extended_unique_authors'] = int(len(set(a for ids in ext_d['author_id_list'] for a in ids)))
    counts['extended_doctypes'] = {str(k): int(v) for k, v in ext_d['Document Type'].astype(str).value_counts().head(8).items()}
    counts['core_unique_countries'] = len(core_countries)
    counts['core_top_countries'] = {k: int(v) for k, v in core_countries.most_common(12)}
    counts['core_unique_authors'] = int(len(set(a for ids in core_d['author_id_list'] for a in ids)))
    counts['core_unique_institutions'] = int(len(set(i for ins in core_d['institution_list'] for i in ins)))

    # Persist
    with open(os.path.join(ANALYSIS_DIR, 'corpus_core.pkl'), 'wb') as f:
        pickle.dump(core_d, f)
    with open(os.path.join(ANALYSIS_DIR, 'corpus_extended.pkl'), 'wb') as f:
        pickle.dump(ext_d, f)
    with open(PICKLE_PATH, 'wb') as f:           # canonical path for 02-11
        pickle.dump(core_d, f)
    with open(os.path.join(ANALYSIS_DIR, 'ref_labels.json'), 'w') as f:
        json.dump(label_map, f)
    with open(os.path.join(ANALYSIS_DIR, 'prisma_counts.json'), 'w') as f:
        json.dump(counts, f, indent=2)
    keep_cols = ['paper_id', 'Title', 'Year', 'DOI', 'source', 'in_scopus', 'in_wos',
                 'Cited by', 'Document Type', 'Source title']
    core_d[[c for c in keep_cols if c in core_d.columns]].to_csv(
        os.path.join(ANALYSIS_DIR, 'core_corpus.csv'), index=False)
    ext_d[[c for c in keep_cols if c in ext_d.columns]].to_csv(
        os.path.join(ANALYSIS_DIR, 'extended_corpus.csv'), index=False)

    print("\nPRISMA funnel:")
    for k, v in counts.items():
        if not isinstance(v, dict):
            print(f"  {k:38s} {v}")
    print(f"\nSaved corpus_core.pkl ({len(core_d)}), corpus_extended.pkl ({len(ext_d)}), "
          f"prisma_counts.json")


if __name__ == '__main__':
    main()
