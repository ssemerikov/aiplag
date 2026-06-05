"""Load and normalize Web of Science (WoS) field-tagged exports.

WoS "Full Record and Cited References" plain-text exports use 2-character
field tags; multi-line fields use 3-space continuation lines. Records run from
``PT`` to ``ER``; the file is wrapped by ``FN``/``VR`` (header) and ``EF`` (end).

This module parses those files into the SAME Scopus-shaped column schema that
``data_loader`` consumes, so the merged corpus can flow through the existing
pipeline (scripts 02-11) unchanged.
"""
import os
import re
import glob
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Tags whose lines are a LIST of items (one per line), not a wrapped paragraph.
_LIST_TAGS = {'AU', 'AF', 'CR', 'C1'}

# Country-name normalization so WoS ("Peoples R China", "USA", "England") and
# Scopus ("China", "United States", "United Kingdom") agree in the merged corpus.
COUNTRY_NORM = {
    'peoples r china': 'China', 'china': 'China', 'taiwan': 'Taiwan',
    'usa': 'United States', 'united states': 'United States',
    'england': 'United Kingdom', 'scotland': 'United Kingdom',
    'wales': 'United Kingdom', 'north ireland': 'United Kingdom',
    'u arab emirates': 'United Arab Emirates', 'uae': 'United Arab Emirates',
    'united arab emirates': 'United Arab Emirates',
    'south korea': 'South Korea', 'korea': 'South Korea',
    'republic of korea': 'South Korea', 'korea rep': 'South Korea',
    'russia': 'Russian Federation', 'russian federation': 'Russian Federation',
    'viet nam': 'Vietnam', 'vietnam': 'Vietnam',
    'czech republic': 'Czech Republic', 'czechia': 'Czech Republic',
    'saudi arabia': 'Saudi Arabia', 'hong kong': 'Hong Kong',
    'turkiye': 'Turkey', 'turkey': 'Turkey',
}


def normalize_country(name):
    if not name:
        return name
    key = re.sub(r'\.$', '', str(name).strip()).strip().lower()
    return COUNTRY_NORM.get(key, str(name).strip().rstrip('.').strip())


def _flush(records, cur):
    if cur:
        records.append(cur)


def parse_wos_file(path):
    """Parse one WoS field-tagged file into a list of {tag: value|list} dicts."""
    records = []
    cur = {}
    last_tag = None
    with open(path, encoding='utf-8-sig', errors='replace') as fh:
        for raw in fh:
            line = raw.rstrip('\n')
            if not line.strip():
                continue
            if line.startswith('FN ') or line.startswith('VR '):
                continue
            if line.strip() == 'EF':
                break
            tag = line[:2]
            is_cont = line.startswith('   ')  # 3-space continuation
            if (not is_cont) and re.match(r'^[A-Z0-9]{2} ', line):
                content = line[3:]
                if tag == 'PT':           # start of a new record
                    _flush(records, cur)
                    cur = {}
                if tag == 'ER':           # end of record
                    _flush(records, cur)
                    cur = {}
                    last_tag = None
                    continue
                if tag in _LIST_TAGS:
                    cur[tag] = [content]
                else:
                    cur[tag] = content
                last_tag = tag
            else:  # continuation line
                content = line.strip()
                if last_tag is None:
                    continue
                if last_tag in _LIST_TAGS:
                    cur.setdefault(last_tag, []).append(content)
                else:
                    cur[last_tag] = cur.get(last_tag, '') + ' ' + content
    _flush(records, cur)
    return records


def _norm_author_id(full_name):
    """Pseudo author-ID from a WoS full name ('Su, Zhao' -> 'su, zhao')."""
    return re.sub(r'\s+', ' ', str(full_name).strip().lower())


def _strip_author_bracket(c1_entry):
    """Drop the leading '[Author, A; ...]' prefix from a WoS C1 affiliation."""
    return re.sub(r'^\[[^\]]*\]\s*', '', c1_entry).strip()


def _record_to_row(rec):
    af = rec.get('AF') or rec.get('AU') or []
    de = rec.get('DE', '')
    idk = rec.get('ID', '')
    c1 = rec.get('C1', [])
    affs = [_strip_author_bracket(x) for x in c1] if c1 else (
        [s.strip() for s in rec.get('C3', '').split(';') if s.strip()])
    cr = rec.get('CR', [])
    try:
        year = int(re.search(r'(\d{4})', rec.get('PY', '')).group(1))
    except (AttributeError, ValueError):
        year = None
    tc = rec.get('TC') or rec.get('Z9') or '0'
    try:
        cited = int(re.search(r'\d+', tc).group(0))
    except (AttributeError, ValueError):
        cited = 0
    return {
        'Title': rec.get('TI', ''),
        'Year': year,
        'Abstract': rec.get('AB', '') or '[No abstract available]',
        'Author full names': '; '.join(af),
        'Author(s) ID': '; '.join(_norm_author_id(a) for a in af),
        'Author Keywords': de,
        'Index Keywords': idk,
        'Affiliations': '; '.join(affs),
        'Open Access': rec.get('OA', ''),
        'Document Type': rec.get('DT', ''),
        'Cited by': cited,
        'DOI': (rec.get('DI', '') or '').strip(),
        'References': '; '.join(cr),
        'cr_list': cr,
        'Source title': rec.get('SO', ''),
        'wos_ut': rec.get('UT', ''),
        'source': 'WoS',
    }


def load_wos(paths=None):
    """Load one or more WoS files into a Scopus-shaped DataFrame."""
    if paths is None:
        paths = sorted(glob.glob(os.path.join(DATA_DIR, 'wos_*.txt')))
    rows = []
    for p in paths:
        for rec in parse_wos_file(p):
            if 'TI' in rec:  # skip stray/empty records
                rows.append(_record_to_row(rec))
    df = pd.DataFrame(rows)
    return df


if __name__ == '__main__':
    df = load_wos()
    print(f"WoS records parsed: {len(df)}")
    print(f"  with abstract: {(df['Abstract'] != '[No abstract available]').sum()}")
    print(f"  with DOI:      {(df['DOI'] != '').sum()}")
    print(f"  with refs(CR): {(df['References'] != '').sum()}")
    print(f"  year range:    {df['Year'].min()}-{df['Year'].max()}")
    print(df[['Title', 'Year', 'Cited by', 'DOI']].head(3).to_string())
