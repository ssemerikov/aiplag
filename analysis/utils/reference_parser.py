"""Parse Scopus reference strings and build reference-based networks."""
import re
import pandas as pd
from collections import Counter


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
    ref = {'raw': raw}

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
    """Create a normalized key for fuzzy matching references across papers."""
    parts = []
    if 'first_author' in ref:
        # Normalize author: lowercase, strip periods
        author = ref['first_author'].lower().replace('.', '').strip()
        parts.append(author)
    if 'year' in ref:
        parts.append(str(ref['year']))
    if 'title_fragment' in ref:
        # First few words of title, lowercased
        title = ref['title_fragment'].lower()
        title = re.sub(r'[^a-z0-9\s]', '', title)
        words = title.split()[:5]
        parts.append(' '.join(words))
    return '|'.join(parts) if parts else ref.get('raw', '')[:80].lower()


def build_reference_lists(df):
    """Parse references for all papers, return dict: paper_id -> list of ref keys."""
    ref_dict = {}
    for _, row in df.iterrows():
        pid = row['paper_id']
        refs = parse_references(row.get('References'))
        ref_dict[pid] = [normalize_ref_key(r) for r in refs]
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
