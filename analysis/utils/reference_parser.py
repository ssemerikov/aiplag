"""Parse Scopus AND Web of Science reference strings and build
reference-based networks.

Cited references arrive in two formats: Scopus ("Author A.A., Title, (Year)…")
and WoS CR lines ("Surname AB, YYYY, JOURNAL, Vvol, Ppage, DOI 10.x"). Both are
reduced to a single canonical key (DOI-first, author|year fallback) so that
bibliographic coupling and co-citation match the SAME cited work across the two
databases.
"""
import re
import pandas as pd
from collections import Counter

# DOI anywhere in a reference string (handles bracketed WoS "DOI [10.x, 10.y]").
DOI_RE = re.compile(r'10\.\d{4,9}/[^\s,;\]\[]+', re.I)
ARXIV_RE = re.compile(r'arxiv[:\s]*(\d{4}\.\d{4,5})', re.I)


def extract_doi(text):
    """Return a normalized DOI (or arXiv id) found in a reference string."""
    if not text:
        return None
    m = DOI_RE.search(str(text))
    if m:
        return m.group(0).lower().rstrip('.').rstrip(')').rstrip(']')
    a = ARXIV_RE.search(str(text))
    if a:
        return 'arxiv:' + a.group(1)
    return None


def parse_wos_cr_line(line):
    """Parse one WoS CR line into a structured reference dict."""
    raw = str(line).strip()
    ref = {'raw': raw, 'source': 'wos'}
    parts = [p.strip() for p in raw.split(',')]
    if parts and parts[0]:
        ref['first_author'] = parts[0].split()[0].strip().lower()
    if len(parts) > 1 and re.fullmatch(r'(19|20)\d{2}', parts[1]):
        ref['year'] = int(parts[1])
    else:
        ym = re.search(r'\b(19|20)\d{2}\b', raw)
        if ym:
            ref['year'] = int(ym.group(0))
    if len(parts) > 2:
        ref['source_token'] = parts[2]
    doi = extract_doi(raw)
    if doi:
        ref['doi'] = doi
    return ref


def canonical_cited_key(ref):
    """Single matching key for a cited work, consistent across DBs.

    DOI (or arXiv id) when available; otherwise first-author surname + year
    (+ a short source/title token). Returns ``None`` when neither a DOI nor a
    surname+year is available — surname-only keys are dropped because common
    surnames (Wang, Liu, Zhang) would otherwise collapse hundreds of distinct
    works into one spurious node, inflating coupling and co-citation.
    """
    doi = ref.get('doi')
    if doi:
        return 'doi:' + doi
    fa = re.sub(r'[^a-z]', '', (ref.get('first_author') or '').lower())
    yr = ref.get('year')
    if not (fa and yr):
        return None
    parts = [fa, str(yr)]
    tok = ref.get('source_token') or ref.get('title_fragment')
    if tok:
        tok = re.sub(r'[^a-z0-9\s]', '', str(tok).lower())
        if tok.split():
            parts.append(' '.join(tok.split()[:3]))
    return 'ay:' + '|'.join(parts)


def ref_label(ref):
    """Human-readable label for a cited work (for network node labels)."""
    a = (ref.get('first_author') or '').title()
    y = ref.get('year', '')
    return f"{a} ({y})" if a or y else str(ref.get('raw', ''))[:40]


def wos_cr_to_keys(cr_list):
    """Map a WoS cr_list to (keys, key->label) for one paper (drops low-specificity refs)."""
    keys, labels = [], {}
    for line in cr_list or []:
        ref = parse_wos_cr_line(line)
        k = canonical_cited_key(ref)
        if not k:
            continue
        keys.append(k)
        labels.setdefault(k, ref_label(ref))
    return keys, labels


def scopus_refs_to_keys(ref_str):
    """Map a Scopus 'References' string to (keys, key->label) for one paper (drops low-specificity refs)."""
    keys, labels = [], {}
    for ref in parse_references(ref_str):
        k = normalize_ref_key(ref)
        if not k:
            continue
        keys.append(k)
        labels.setdefault(k, ref_label(ref))
    return keys, labels


def parse_references(ref_str):
    """Parse semicolon-separated Scopus reference strings into structured dicts."""
    if pd.isna(ref_str) or not str(ref_str).strip():
        return []

    refs = []
    # Scopus separates references by semicolons, but titles/content may also have semicolons.
    # References typically start with an author name pattern.
    raw_parts = str(ref_str).split(';')

    current_ref = ""
    parsed_refs = []

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        # Heuristic: new reference starts with what looks like an author name
        # (word followed by initial or comma) and current_ref is non-empty
        if current_ref and re.match(r'^[A-Z][a-z]+\s+[A-Z]', part):
            parsed_refs.append(current_ref.strip())
            current_ref = part
        else:
            if current_ref:
                current_ref += "; " + part
            else:
                current_ref = part

    if current_ref.strip():
        parsed_refs.append(current_ref.strip())

    for raw in parsed_refs:
        ref = _parse_single_reference(raw)
        if ref:
            refs.append(ref)

    return refs


def _parse_single_reference(raw):
    """Extract structured info from a single reference string."""
    ref = {'raw': raw, 'source': 'scopus'}

    # Extract DOI (strongest cross-database matching key)
    doi = extract_doi(raw)
    if doi:
        ref['doi'] = doi

    # Extract year
    year_match = re.search(r'\((\d{4})\)', raw)
    if year_match:
        ref['year'] = int(year_match.group(1))

    # Extract first author (before first comma or period)
    author_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z]\.?(?:\s*[A-Z]\.?)*)?)', raw)
    if author_match:
        ref['first_author'] = author_match.group(1).strip()

    # Try to extract title (text before journal/volume info)
    # Common patterns: after authors, before volume/page info
    parts = raw.split(',')
    if len(parts) >= 2:
        # Title is often the second or third element
        for p in parts[1:4]:
            p = p.strip()
            if len(p) > 20 and not re.match(r'^\d+$', p) and 'pp.' not in p:
                ref['title_fragment'] = p[:100]
                break

    return ref


def normalize_ref_key(ref):
    """Create a normalized key for matching references across papers.

    DOI-first (so a Scopus and a WoS citation to the same work collapse to one
    node), falling back to author|year|title.
    """
    return canonical_cited_key(ref)


def build_reference_lists(df):
    """Return dict: paper_id -> list of canonical cited-work keys.

    Uses the precomputed ``ref_keys`` column when present (the merged corpus
    stores per-paper keys built per source); otherwise parses the Scopus
    ``References`` string on the fly.
    """
    if 'ref_keys' in df.columns:
        return {row['paper_id']: list(row['ref_keys']) for _, row in df.iterrows()}
    ref_dict = {}
    for _, row in df.iterrows():
        pid = row['paper_id']
        refs = parse_references(row.get('References'))
        ref_dict[pid] = [k for k in (normalize_ref_key(r) for r in refs) if k]
    return ref_dict


def count_cited_works(ref_dict):
    """Count how often each external work is cited across the corpus."""
    all_refs = []
    for pid, refs in ref_dict.items():
        all_refs.extend(refs)
    return Counter(all_refs)


def build_bibliographic_coupling_matrix(df, ref_dict, min_shared=2):
    """Build paper x paper bibliographic coupling matrix.

    Edge weight = number of shared references between two papers.
    """
    import numpy as np

    papers = list(ref_dict.keys())
    n = len(papers)
    matrix = np.zeros((n, n), dtype=int)

    ref_sets = {pid: set(refs) for pid, refs in ref_dict.items()}

    for i in range(n):
        for j in range(i + 1, n):
            shared = len(ref_sets[papers[i]] & ref_sets[papers[j]])
            if shared >= min_shared:
                matrix[i, j] = shared
                matrix[j, i] = shared

    return papers, matrix


def build_cocitation_matrix(ref_dict, top_n=30):
    """Build co-citation matrix for the top-N most cited external works."""
    # Count all references
    counts = count_cited_works(ref_dict)
    top_refs = [ref for ref, _ in counts.most_common(top_n)]

    import numpy as np

    n = len(top_refs)
    ref_to_idx = {ref: i for i, ref in enumerate(top_refs)}
    matrix = np.zeros((n, n), dtype=int)

    for pid, refs in ref_dict.items():
        ref_set = set(refs)
        present = [r for r in top_refs if r in ref_set]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                idx_i = ref_to_idx[present[i]]
                idx_j = ref_to_idx[present[j]]
                matrix[idx_i, idx_j] += 1
                matrix[idx_j, idx_i] += 1

    return top_refs, matrix
