"""
blocking.py
-----------
Multi-strategy candidate generation (blocking) for entity resolution.

Strategies (applied per country block):
  1. TF-IDF cosine similarity on expanded name + address + country text
  2. Character n-gram TF-IDF on raw (un-expanded) text  (catches transliterations)
  3. First-name-token exact-match index                  (fast, high-recall)
  4. Numeric-token index (ZIP/PIN/house#)                (precise address anchor)

Results from all strategies are UNION-ed per S1 entity, then truncated to top_k
by TF-IDF score (strategy 1) to keep downstream feature extraction manageable.

All heavy vectoriser work is done per-country to stay within memory limits.
"""

import gc
import os
import re

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

from normalize import (
    # Vectorized Series API (fast path for bulk loading)
    normalize_name_s,
    normalize_address_s,
    normalize_country_s,
    first_token_s,
    numeric_tokens_list,
    # Scalar API (kept for compatibility / features.py)
    normalize_name,
    normalize_address,
    normalize_country,
    combined_search_text,
    first_name_token,
    extract_numeric_tokens,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _top_k_sparse_row(row, top_k, threshold):
    """Return (indices, scores) for top-k entries above threshold in a sparse row."""
    if row.nnz == 0:
        return [], []
    data, indices = row.data, row.indices
    mask = data >= threshold
    data, indices = data[mask], indices[mask]
    if len(data) == 0:
        return [], []
    if len(data) > top_k:
        part = np.argpartition(data, -top_k)[-top_k:]
        order = part[np.argsort(-data[part])]
    else:
        order = np.argsort(-data)
    return indices[order].tolist(), data[order].tolist()


def _build_word_vectorizer(corpus, sample_size=250_000):
    """Fit a word (1,2)-gram TF-IDF vectorizer on a sample of corpus."""
    sample = corpus if len(corpus) <= sample_size else corpus.sample(n=sample_size, random_state=42)
    vec = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.6,
        max_features=100_000,
        dtype=np.float32,
        sublinear_tf=True,
    )
    vec.fit(sample)
    return vec


def _build_char_vectorizer(corpus, sample_size=250_000):
    """Fit a character (3,5)-gram TF-IDF vectorizer (transliteration-robust)."""
    sample = corpus if len(corpus) <= sample_size else corpus.sample(n=sample_size, random_state=42)
    vec = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.7,
        max_features=150_000,
        dtype=np.float32,
        sublinear_tf=True,
    )
    vec.fit(sample)
    return vec


def _tfidf_candidates(vec, X_cand_T, X_s1, cand_ids, top_k, threshold, batch_size):
    """
    Batched dot-product retrieval.

    Returns
    -------
    dict[int, set[str]]
        Mapping from S1 row-index -> set of candidate IDs.
    """
    n_queries = X_s1.shape[0]
    result = defaultdict(set)
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        sim = X_s1[start:end].dot(X_cand_T).tocsr()
        for i in range(sim.shape[0]):
            matched_idx, _ = _top_k_sparse_row(sim[i], top_k, threshold)
            for idx in matched_idx:
                result[start + i].add(cand_ids[idx])
    return result


def _tfidf_candidates_subtok(
    X_s1,
    X_cand,
    s1_ids,
    cand_ids,
    s1_first_tokens,
    cand_first_tokens,
    top_k,
    threshold,
    batch_size,
    max_bucket_cands=50_000,
    rng=None,
):
    """
    Sub-blocked TF-IDF using pre-computed sparse matrices.

    Key optimisation
    ----------------
    vec.transform() is called ONCE per country (before this function) — not once
    per first-token bucket. Here we only do fast CSR row-slicing per bucket:
      X_q   = X_s1[s1_idxs]        # O(nnz in slice) — near zero overhead
      X_c_T = X_cand[cand_idxs].T  # same
    Then a small dot product on the sliced matrices.

    Parameters
    ----------
    X_s1   : CSR sparse (n_s1   × vocab)
    X_cand : CSR sparse (n_cand × vocab)
    max_bucket_cands : cap oversized generic-token buckets.

    Returns
    -------
    dict[int, set[str]]
        Mapping from S1 row-index -> set of matched candidate IDs.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Build inverted index: first_token -> candidate row-indices
    cand_tok_index = defaultdict(list)
    for j, tok in enumerate(cand_first_tokens):
        cand_tok_index[tok if tok else '_EMPTY_'].append(j)

    # Group S1 by first_token
    s1_tok_groups = defaultdict(list)
    for i, tok in enumerate(s1_first_tokens):
        s1_tok_groups[tok if tok else '_EMPTY_'].append(i)

    result = defaultdict(set)

    for tok, s1_idxs in s1_tok_groups.items():
        cand_idxs = list(cand_tok_index.get(tok, []))
        if tok != '_EMPTY_':
            cand_idxs += cand_tok_index.get('_EMPTY_', [])
        if not cand_idxs:
            continue

        # Cap oversized buckets (e.g. 'national', 'india', 'new')
        if len(cand_idxs) > max_bucket_cands:
            cand_idxs = rng.choice(cand_idxs, size=max_bucket_cands, replace=False).tolist()

        # Slice pre-computed matrices — no transform() call here
        X_q   = X_s1[s1_idxs]           # (n_s1_tok  × vocab)  — CSR slice, fast
        X_c_T = X_cand[cand_idxs].T     # (vocab × n_cand_tok) — CSR slice + T

        sub_cand_ids = cand_ids[cand_idxs]

        for start in range(0, len(s1_idxs), batch_size):
            end = min(start + batch_size, len(s1_idxs))
            sim = X_q[start:end].dot(X_c_T).tocsr()
            for i in range(sim.shape[0]):
                matched_idx, _ = _top_k_sparse_row(sim[i], top_k, threshold)
                for idx in matched_idx:
                    result[s1_idxs[start + i]].add(sub_cand_ids[idx])

    return result




def _first_token_candidates(s1_rows, cand_rows):
    """
    Exact first-name-token match between S1 and candidates.

    Returns
    -------
    dict[int, set[str]]
        Mapping from S1 row-index -> set of candidate IDs.
    """
    # Build inverted index: token -> list of cand ids
    token_index = defaultdict(list)
    for cand_idx, row in enumerate(cand_rows):
        tok = row['first_token']
        if tok:
            token_index[tok].append(row['id'])

    result = defaultdict(set)
    for s1_idx, row in enumerate(s1_rows):
        tok = row['first_token']
        if tok and tok in token_index:
            for cid in token_index[tok]:
                result[s1_idx].add(cid)
    return result


def _numeric_token_candidates(s1_rows, cand_rows, min_shared=1):
    """
    Candidates sharing at least `min_shared` numeric address token(s).

    Returns
    -------
    dict[int, set[str]]
        Mapping from S1 row-index -> set of candidate IDs.
    """
    # Build inverted index: token -> list of cand ids
    num_index = defaultdict(list)
    for row in cand_rows:
        for tok in row['num_tokens']:
            num_index[tok].append(row['id'])

    result = defaultdict(set)
    for s1_idx, row in enumerate(s1_rows):
        seen = set()
        for tok in row['num_tokens']:
            if tok in num_index:
                for cid in num_index[tok]:
                    seen.add(cid)
        for cid in seen:
            result[s1_idx].add(cid)
    return result


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def load_source_df(path, source_label=''):
    """
    Read a TSV source file and return a lightweight DataFrame with:
      id, country, name, address, expanded_text, raw_text, first_token, num_tokens

    Performance optimizations vs. the original:
      - Single vectorized pass using pd.Series.str.replace() (C-level regex)
      - Parquet cache: subsequent runs load in ~0.2s instead of 2+ minutes
      - numeric_tokens stored as pipe-separated string to allow parquet round-trip
    """
    # --- Pickle cache ---
    cache_path = path + '.blocking_cache.pkl'
    src_mtime = os.path.getmtime(path)
    if os.path.exists(cache_path):
        cache_mtime = os.path.getmtime(cache_path)
        if cache_mtime >= src_mtime:
            print(f"  [cache hit] Loading {os.path.basename(path)} [{source_label}] from cache...")
            import pickle
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

    print(f"  Loading {os.path.basename(path)} [{source_label}]...")
    df = pd.read_csv(path, sep='\t', low_memory=False)

    id_col     = 'entity_id'         if 'entity_id'         in df.columns else df.columns[0]
    name_col   = 'business_name'     if 'business_name'     in df.columns else None
    addr_col   = 'business_address'  if 'business_address'  in df.columns else None
    country_col= 'country'           if 'country'           in df.columns else None

    ids       = df[id_col].astype(str)
    names     = df[name_col].fillna('').astype(str)     if name_col    else pd.Series([''] * len(df), dtype=str)
    addresses = df[addr_col].fillna('').astype(str)     if addr_col    else pd.Series([''] * len(df), dtype=str)
    countries = df[country_col].fillna('').astype(str)  if country_col else pd.Series(['US'] * len(df), dtype=str)

    del df
    gc.collect()

    # --- Single vectorized normalization pass ---
    print(f"    Normalizing names...")
    norm_names     = normalize_name_s(names)
    print(f"    Normalizing addresses...")
    norm_addresses = normalize_address_s(addresses)
    norm_countries_s = normalize_country_s(countries)

    # expanded_text = norm_name + norm_address + lower(country)  (abbrevs already expanded)
    expanded_texts = (
        norm_names + ' ' + norm_addresses + ' ' + norm_countries_s.str.lower()
    ).str.strip()

    # raw_text: ASCII-only lower without abbrev expansion (char n-gram TF-IDF input)
    # Reuse norm_names/addresses without abbrev expansion by stripping only punct
    # We do a light pass: lower + strip-punct on original bytes (ASCII transliteration already done)
    import unicodedata as _ud
    def _raw_ascii(s):
        """Fast ASCII lower without abbrev expansion."""
        out = []
        for t in s:
            if not isinstance(t, str) or not t:
                out.append('')
                continue
            try:
                out.append(_ud.normalize('NFKD', t).encode('ascii', errors='ignore').decode('ascii').lower())
            except Exception:
                out.append('')
        return out

    print(f"    Building raw texts...")
    raw_name_list = _raw_ascii(names)
    raw_addr_list = _raw_ascii(addresses)
    raw_texts = pd.Series(
        [f"{n} {a} {c.lower()}".strip()
         for n, a, c in zip(raw_name_list, raw_addr_list, norm_countries_s)],
        dtype=str
    )
    raw_texts = raw_texts.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    raw_texts = raw_texts.str.replace(r'\s+', ' ', regex=True).str.strip()

    print(f"    Extracting first tokens and numeric tokens...")
    first_tokens   = first_token_s(norm_names)
    num_toks       = numeric_tokens_list(addresses)   # list of lists

    result = pd.DataFrame({
        'id':            ids.values,
        'country':       norm_countries_s.values,
        'name':          names.values,
        'address':       addresses.values,
        'expanded_text': expanded_texts.values,
        'raw_text':      raw_texts.values,
        'first_token':   first_tokens.values,
        'num_tokens':    num_toks,                    # list of lists — stored separately in cache
    })

    # --- Write pickle cache ---
    try:
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    Cached to {os.path.basename(cache_path)}")
    except Exception as e:
        print(f"    Warning: could not write cache: {e}")

    gc.collect()
    return result


def generate_candidates(
    s1_path,
    s2_path,
    s3_path,
    output_path='output/candidate_pairs.tsv',
    top_k=50,
    tfidf_threshold=0.05,
    char_threshold=0.05,
    batch_size=500,
    use_char_ngram=False,
    use_first_token=True,
    use_numeric_token=True,
    max_bucket_cands=50_000,
    max_s1=0,
):
    """
    Multi-strategy blocking: generate up to top_k candidate pairs per S1 entity.

    Strategy union order (all within same country, sub-blocked by first name token):
      1. Word TF-IDF (expanded text) — 2 transforms per country, then matrix slicing
      2. Char n-gram TF-IDF (raw text) — off by default; adds cost for marginal gain
         on training data (enable with use_char_ngram=True for final test run)
      3. First-name-token exact index    [if use_first_token]
      4. Numeric-address-token index     [if use_numeric_token]

    TF-IDF sub-blocking
    -------------------
    transform() is called ONCE per country for S1 and ONCE for candidates.
    Per-bucket work is only CSR row-slicing (near zero overhead), then a small
    dot product on the sliced sub-matrices.

    Parameters
    ----------
    max_s1 : int
        If > 0, randomly sample this many S1 entities before blocking.
        Useful for fast recall measurement on training data (e.g. --max-s1 20000).

    Outputs
    -------
    TSV file at output_path with columns: source1_entity_id, candidate_entity_ids
    """

    print("=== Stage 3: Multi-Strategy Blocking ===")
    print("Loading source files...")
    df_s1 = load_source_df(s1_path, 'S1')
    df_s2 = load_source_df(s2_path, 'S2')
    df_s3 = load_source_df(s3_path, 'S3')

    print("Combining S2 + S3 candidates...")
    df_cand = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()

    print(f"S1 entities: {len(df_s1):,}  |  Candidates (S2+S3): {len(df_cand):,}")

    # --- Optional S1 sampling (for fast recall measurement on training data) ---
    if max_s1 and max_s1 < len(df_s1):
        df_s1 = df_s1.sample(n=max_s1, random_state=42).reset_index(drop=True)
        print(f"  Sampled {len(df_s1):,} S1 entities (--max-s1 {max_s1})")

    # Fit vectorisers on combined candidate pool (sample for speed)
    print("\nFitting word (1,2)-gram TF-IDF on expanded text...")
    word_vec = _build_word_vectorizer(df_cand['expanded_text'])
    print(f"  Word vocab size: {len(word_vec.vocabulary_):,}")

    char_vec = None
    if use_char_ngram:
        print("Fitting char (3,5)-gram TF-IDF on raw text...")
        char_vec = _build_char_vectorizer(df_cand['raw_text'])
        print(f"  Char vocab size: {len(char_vec.vocabulary_):,}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"\nStreaming results to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")

        countries = sorted(c for c in df_s1['country'].unique() if c)
        if not countries:
            countries = ['ALL']

        for country in countries:
            if country != 'ALL':
                s1_mask = df_s1['country'] == country
                cand_mask = df_cand['country'] == country
            else:
                s1_mask = pd.Series([True] * len(df_s1))
                cand_mask = pd.Series([True] * len(df_cand))

            sub_s1 = df_s1[s1_mask].reset_index(drop=True)
            sub_cand = df_cand[cand_mask].reset_index(drop=True)

            n_q = len(sub_s1)
            n_c = len(sub_cand)

            if n_q == 0 or n_c == 0:
                print(f"  Skipping country '{country}' (no records in one side)")
                continue

            print(f"\n--- Country '{country}': {n_q:,} S1 queries x {n_c:,} candidates ---")
            print(f"  Unique first tokens in S1: {sub_s1['first_token'].nunique():,}")

            cand_ids   = sub_cand['id'].values
            s1_ids     = sub_s1['id'].values
            rng        = np.random.default_rng(42)

            s1_ftoks   = sub_s1['first_token'].values
            cand_ftoks = sub_cand['first_token'].values

            # Strategy 1: Word TF-IDF
            # transform() called ONCE here; _tfidf_candidates_subtok only slices
            print("  [1/4] Word TF-IDF: transforming...")
            X_s1_word   = word_vec.transform(sub_s1['expanded_text']).tocsr()
            X_cand_word = word_vec.transform(sub_cand['expanded_text']).tocsr()
            print(f"        {X_s1_word.shape[0]:,} x {X_cand_word.shape[0]:,} sparse (sub-blocking by first token)...")
            word_hits = _tfidf_candidates_subtok(
                X_s1_word, X_cand_word, s1_ids, cand_ids,
                s1_ftoks, cand_ftoks,
                top_k, tfidf_threshold, batch_size, max_bucket_cands, rng
            )
            del X_s1_word, X_cand_word
            gc.collect()

            # Strategy 2: Char n-gram TF-IDF (off by default on training scale)
            char_hits = defaultdict(set)
            if use_char_ngram and char_vec is not None:
                print("  [2/4] Char TF-IDF: transforming...")
                X_s1_char   = char_vec.transform(sub_s1['raw_text']).tocsr()
                X_cand_char = char_vec.transform(sub_cand['raw_text']).tocsr()
                char_hits = _tfidf_candidates_subtok(
                    X_s1_char, X_cand_char, s1_ids, cand_ids,
                    s1_ftoks, cand_ftoks,
                    top_k, char_threshold, batch_size, max_bucket_cands, rng
                )
                del X_s1_char, X_cand_char
                gc.collect()

            # Strategy 3: First-name-token
            token_hits = defaultdict(set)
            if use_first_token:
                print("  [3/4] First-name-token index lookup...")
                s1_rows = sub_s1[['id', 'first_token']].to_dict('records')
                cand_rows = sub_cand[['id', 'first_token']].to_dict('records')
                token_hits = _first_token_candidates(s1_rows, cand_rows)

            # Strategy 4: Numeric address tokens
            num_hits = defaultdict(set)
            if use_numeric_token:
                print("  [4/4] Numeric address-token index lookup...")
                s1_rows_num = sub_s1[['id', 'num_tokens']].to_dict('records')
                cand_rows_num = sub_cand[['id', 'num_tokens']].to_dict('records')
                num_hits = _numeric_token_candidates(s1_rows_num, cand_rows_num)

            # Union all strategies and write output
            print("  Merging strategies and writing output...")
            lines = []
            for i, s1_id in enumerate(sub_s1['id'].values):
                all_candidates = (
                    word_hits.get(i, set())
                    | char_hits.get(i, set())
                    | token_hits.get(i, set())
                    | num_hits.get(i, set())
                )
                # Truncate to top_k (preserves word-TF-IDF ranked order for the word_hits subset)
                candidate_list = list(all_candidates)[:top_k]
                lines.append(f"{s1_id}\t{','.join(str(x) for x in candidate_list)}\n")

            out_f.writelines(lines)
            print(f"  Written {len(lines):,} S1 rows for country '{country}'")

            del sub_s1, sub_cand, word_hits, char_hits, token_hits, num_hits
            gc.collect()

    del df_s1, df_cand
    gc.collect()
    print(f"\nBlocking complete! Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Stage 3: Multi-strategy blocking')
    parser.add_argument('--s1',             default='dataset/test/test_source1.tsv')
    parser.add_argument('--s2',             default='dataset/test/test_source2.tsv')
    parser.add_argument('--s3',             default='dataset/test/test_source3.tsv')
    parser.add_argument('--output',         default='output/candidate_pairs.tsv')
    parser.add_argument('--top-k',          type=int,   default=50)
    parser.add_argument('--tfidf-threshold', type=float, default=0.05)
    parser.add_argument('--char-threshold', type=float, default=0.05)
    parser.add_argument('--batch-size',        type=int,   default=500)
    parser.add_argument('--max-bucket-cands',  type=int,   default=50000)
    parser.add_argument('--no-char-ngram',     action='store_true')
    parser.add_argument('--no-first-token',    action='store_true')
    parser.add_argument('--no-numeric',        action='store_true')
    parser.add_argument('--char-ngram',        action='store_true',
                        help='Enable char n-gram TF-IDF (off by default; use for final test run)')
    parser.add_argument('--max-s1',            type=int,   default=0,
                        help='Sample this many S1 entities (0=all). Use for fast recall measurement.')
    args = parser.parse_args()

    generate_candidates(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        output_path=args.output,
        top_k=args.top_k,
        tfidf_threshold=args.tfidf_threshold,
        char_threshold=args.char_threshold,
        batch_size=args.batch_size,
        use_char_ngram=args.char_ngram,
        use_first_token=not args.no_first_token,
        use_numeric_token=not args.no_numeric,
        max_bucket_cands=args.max_bucket_cands,
        max_s1=args.max_s1,
    )
