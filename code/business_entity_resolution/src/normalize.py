"""
normalize.py
------------
Text normalization and abbreviation expansion for entity matching.

Used by blocking.py and features.py to produce a clean, canonical representation
of business names and addresses before any similarity computation.
"""

import re
import unicodedata

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
    r'\b&\b': 'and',
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


def _build_pattern_list(mapping):
    return [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in mapping.items()]


_ENTITY_PATTERNS = _build_pattern_list(ENTITY_SUFFIX_MAP)
_ADDRESS_PATTERNS = _build_pattern_list(ADDRESS_ABBREV_MAP)
_WORD_PATTERNS = _build_pattern_list(WORD_ABBREV_MAP)

_PUNCT_RE = re.compile(r'[^a-z0-9\s]')
_SPACE_RE = re.compile(r'\s+')


def to_ascii(text):
    """Transliterate Unicode to closest ASCII; non-transliterable chars dropped."""
    if not isinstance(text, str) or not text.strip():
        return ''
    try:
        normalized = unicodedata.normalize('NFKD', text)
        return normalized.encode('ascii', errors='ignore').decode('ascii')
    except Exception:
        return ''


def normalize_name(name, expand_abbrevs=True):
    """Canonical form of a business name (ASCII -> lower -> expand -> clean)."""
    text = to_ascii(name).lower()
    if expand_abbrevs:
        for pattern, repl in _WORD_PATTERNS:
            text = pattern.sub(repl, text)
        for pattern, repl in _ENTITY_PATTERNS:
            text = pattern.sub(repl, text)
    text = _PUNCT_RE.sub(' ', text)
    return _SPACE_RE.sub(' ', text).strip()


def normalize_address(address, expand_abbrevs=True):
    """Canonical form of an address string."""
    text = to_ascii(address).lower()
    if expand_abbrevs:
        for pattern, repl in _ADDRESS_PATTERNS:
            text = pattern.sub(repl, text)
    text = _PUNCT_RE.sub(' ', text)
    return _SPACE_RE.sub(' ', text).strip()


def first_name_token(name):
    """Return first significant token of a normalized name (cheap blocking key)."""
    tokens = normalize_name(name, expand_abbrevs=False).split()
    for tok in tokens:
        if len(tok) > 1:
            return tok
    return ''


def extract_numeric_tokens(text):
    """Extract 3-10 digit numeric tokens (ZIP/PIN, house numbers) from address text."""
    if not isinstance(text, str):
        return []
    return re.findall(r'\b\d{3,10}\b', text)


def normalize_country(country):
    """Return uppercase stripped country string."""
    if not isinstance(country, str):
        return ''
    return country.strip().upper()


def combined_search_text(name, address, country, expand_abbrevs=True):
    """Concatenate normalized name + address + country for TF-IDF/BM25 input."""
    parts = [
        normalize_name(name, expand_abbrevs=expand_abbrevs),
        normalize_address(address, expand_abbrevs=expand_abbrevs),
        normalize_country(country).lower(),
    ]
    return ' '.join(p for p in parts if p)
