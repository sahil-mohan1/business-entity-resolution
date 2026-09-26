"""
normalize.py
------------
Text normalization and abbreviation expansion for entity matching.

Two APIs are provided:
  - Scalar API  : normalize_name(str) -> str   (used in features.py, rare calls)
  - Vectorized API: normalize_name_s(Series) -> Series  (used in blocking.py, fast)

The vectorized API uses pd.Series.str.replace() which runs regex on the full
column at the C level — ~100x faster than a Python for-loop over rows.
"""

import re
import unicodedata
import pandas as pd

# ---------------------------------------------------------------------------
# Abbreviation expansion tables
# ---------------------------------------------------------------------------

# Business-entity suffixes -> canonical form
ENTITY_SUFFIX_MAP = {
    r'\binc\b': 'incorporated',
    r'\bcorp\b': 'corporation',
    r'\bltd\b': 'limited',
    r'\bco\b': 'company',
    r'\bllc\b': 'limited liability company',
    r'\bllp\b': 'limited liability partnership',
    r'\blp\b': 'limited partnership',
    r'\bpvt\b': 'private',
    r'\bpte\b': 'private',
    r'\bplc\b': 'public limited company',
    r'\bnv\b': 'naamloze vennootschap',
    r'\bbv\b': 'besloten vennootschap',
}

# Address abbreviations -> canonical form
ADDRESS_ABBREV_MAP = {
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bct\b': 'court',
    r'\bpl\b': 'place',
    r'\bpkwy\b': 'parkway',
    r'\bhwy\b': 'highway',
    r'\bste\b': 'suite',
    r'\bapt\b': 'apartment',
    r'\bfl\b': 'floor',
    r'\bbldg\b': 'building',
    r'\bn\b': 'north',
    r'\bs\b': 'south',
    r'\be\b': 'east',
    r'\bw\b': 'west',
    r'\bne\b': 'northeast',
    r'\bnw\b': 'northwest',
    r'\bse\b': 'southeast',
    r'\bsw\b': 'southwest',
}

# Common word abbreviations in business names
WORD_ABBREV_MAP = {
    r'&': 'and',
    r'\bintl\b': 'international',
    r'\bnatl\b': 'national',
    r'\bmfg\b': 'manufacturing',
    r'\bmgmt\b': 'management',
    r'\bsvcs\b': 'services',
    r'\bsvc\b': 'service',
    r'\btech\b': 'technology',
    r'\bsys\b': 'systems',
    r'\bsol\b': 'solutions',
    r'\bassoc\b': 'associates',
    r'\bengg\b': 'engineering',
    r'\bgrp\b': 'group',
    r'\binst\b': 'institute',
    r'\bhosp\b': 'hospital',
    r'\bmed\b': 'medical',
    r'\bpharma\b': 'pharmaceutical',
    r'\bfin\b': 'financial',
    r'\bfdn\b': 'foundation',
}

_PUNCT_RE = re.compile(r'[^a-z0-9\s]')
_SPACE_RE = re.compile(r'\s+')

# Pre-compiled scalar pattern lists (kept for scalar API / features.py)
def _build_pattern_list(mapping):
    return [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in mapping.items()]

_ENTITY_PATTERNS  = _build_pattern_list(ENTITY_SUFFIX_MAP)
_ADDRESS_PATTERNS = _build_pattern_list(ADDRESS_ABBREV_MAP)
_WORD_PATTERNS    = _build_pattern_list(WORD_ABBREV_MAP)


# ---------------------------------------------------------------------------
# ASCII transliteration (must remain per-element; no vectorized equivalent)
# ---------------------------------------------------------------------------

def _ascii_list(arr):
    """
    NFKD-transliterate an iterable of strings to ASCII.
    Returns a list. Faster than .apply() because it avoids Series overhead.
    """
    out = []
    for text in arr:
        if not isinstance(text, str) or not text:
            out.append('')
            continue
        try:
            out.append(
                unicodedata.normalize('NFKD', text)
                           .encode('ascii', errors='ignore')
                           .decode('ascii')
            )
        except Exception:
            out.append('')
    return out


def to_ascii(text):
    """Scalar: transliterate a single string to ASCII."""
    if not isinstance(text, str) or not text.strip():
        return ''
    try:
        return (unicodedata.normalize('NFKD', text)
                           .encode('ascii', errors='ignore')
                           .decode('ascii'))
    except Exception:
        return ''


# ---------------------------------------------------------------------------
# Vectorized Series API  (used by blocking.py — fast path)
# ---------------------------------------------------------------------------

def _apply_abbrev_map_series(s: pd.Series, mapping: dict) -> pd.Series:
    """Apply all regex replacements from mapping to a pd.Series using str.replace (C-level)."""
    for pat, repl in mapping.items():
        s = s.str.replace(pat, repl, regex=True, case=False)
    return s


def normalize_name_s(names: pd.Series) -> pd.Series:
    """
    Vectorized normalization of a Series of business names.
    Pipeline: ASCII transliterate -> lower -> expand abbrevs -> strip punct -> collapse spaces.
    ~100x faster than calling normalize_name() row-by-row.
    """
    # ASCII transliteration (unavoidably per-element, but done in one tight Python loop)
    s = pd.Series(_ascii_list(names), dtype='object', index=names.index)
    s = s.str.lower().fillna('')
    # Abbreviation expansion via vectorized str.replace
    s = _apply_abbrev_map_series(s, WORD_ABBREV_MAP)
    s = _apply_abbrev_map_series(s, ENTITY_SUFFIX_MAP)
    # Strip non-alphanumeric, collapse spaces
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s


def normalize_address_s(addresses: pd.Series) -> pd.Series:
    """
    Vectorized normalization of a Series of addresses.
    Pipeline: ASCII transliterate -> lower -> expand abbrevs -> strip punct -> collapse spaces.
    """
    s = pd.Series(_ascii_list(addresses), dtype='object', index=addresses.index)
    s = s.str.lower().fillna('')
    s = _apply_abbrev_map_series(s, ADDRESS_ABBREV_MAP)
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s


def normalize_country_s(countries: pd.Series) -> pd.Series:
    """Vectorized: uppercase + strip."""
    return countries.fillna('').astype(str).str.strip().str.upper()


def first_token_s(names_normalized: pd.Series) -> pd.Series:
    """
    Vectorized first-significant-token extraction from already-normalized names.
    Pass the output of normalize_name_s() directly.
    Tokens of length <= 1 are skipped by extracting the first word of length >= 2.
    """
    # Extract first token of length >= 2 using regex
    return names_normalized.str.extract(r'\b([a-z0-9]{2,})', expand=False).fillna('')


_NUMERIC_RE = re.compile(r'\b\d{3,10}\b')

def numeric_tokens_list(addresses: pd.Series) -> list:
    """
    Extract lists of 3-10 digit numeric tokens from each address.
    Returns a Python list of lists (cannot be stored in a Series column natively).
    Uses a tight Python loop — unavoidable for variable-length outputs, but
    done in ONE pass over the column.
    """
    return [_NUMERIC_RE.findall(a) if isinstance(a, str) else [] for a in addresses]


# ---------------------------------------------------------------------------
# Scalar API (kept for features.py compatibility)
# ---------------------------------------------------------------------------

def normalize_name(name, expand_abbrevs=True):
    """Scalar: canonical form of a business name."""
    text = to_ascii(name).lower()
    if expand_abbrevs:
        for pattern, repl in _WORD_PATTERNS:
            text = pattern.sub(repl, text)
        for pattern, repl in _ENTITY_PATTERNS:
            text = pattern.sub(repl, text)
    text = _PUNCT_RE.sub(' ', text)
    return _SPACE_RE.sub(' ', text).strip()


def normalize_address(address, expand_abbrevs=True):
    """Scalar: canonical form of an address string."""
    text = to_ascii(address).lower()
    if expand_abbrevs:
        for pattern, repl in _ADDRESS_PATTERNS:
            text = pattern.sub(repl, text)
    text = _PUNCT_RE.sub(' ', text)
    return _SPACE_RE.sub(' ', text).strip()


def first_name_token(name):
    """Scalar: first significant token of a normalized name."""
    tokens = normalize_name(name, expand_abbrevs=False).split()
    for tok in tokens:
        if len(tok) > 1:
            return tok
    return ''


def extract_numeric_tokens(text):
    """Scalar: extract 3-10 digit numeric tokens from address text."""
    if not isinstance(text, str):
        return []
    return _NUMERIC_RE.findall(text)


def normalize_country(country):
    """Scalar: uppercase stripped country string."""
    if not isinstance(country, str):
        return ''
    return country.strip().upper()


def combined_search_text(name, address, country, expand_abbrevs=True):
    """Scalar: concatenate normalized name + address + country."""
    parts = [
        normalize_name(name, expand_abbrevs=expand_abbrevs),
        normalize_address(address, expand_abbrevs=expand_abbrevs),
        normalize_country(country).lower(),
    ]
    return ' '.join(p for p in parts if p)
