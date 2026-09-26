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
    """
    print(f"  Loading {os.path.basename(path)} [{source_label}]...")
    df = pd.read_csv(path, sep='\t')

    id_col = 'entity_id' if 'entity_id' in df.columns else df.columns[0]
    name_col = 'business_name' if 'business_name' in df.columns else None
    addr_col = 'business_address' if 'business_address' in df.columns else None
    country_col = 'country' if 'country' in df.columns else None

    ids = df[id_col].astype(str)
    names = df[name_col].fillna('').astype(str) if name_col else pd.Series([''] * len(df))
    addresses = df[addr_col].fillna('').astype(str) if addr_col else pd.Series([''] * len(df))
    countries = df[country_col].fillna('').astype(str) if country_col else pd.Series(['US'] * len(df))

    expanded_texts = [
        combined_search_text(n, a, c, expand_abbrevs=True)
        for n, a, c in zip(names, addresses, countries)
    ]
    raw_texts = [
        combined_search_text(n, a, c, expand_abbrevs=False)
        for n, a, c in zip(names, addresses, countries)
    ]
    first_tokens = [first_name_token(n) for n in names]
    num_tokens_list = [extract_numeric_tokens(a) for a in addresses]
    norm_countries = [normalize_country(c) for c in countries]

    result = pd.DataFrame({
        'id': ids.values,
        'country': norm_countries,
        'name': names.values,
        'address': addresses.values,
        'expanded_text': expanded_texts,
        'raw_text': raw_texts,
        'first_token': first_tokens,
        'num_tokens': num_tokens_list,
    })

    del df
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
    batch_size=50,
    use_char_ngram=True,
    use_first_token=True,
    use_numeric_token=True,
):
    """
    Multi-strategy blocking: generate up to top_k candidate pairs per S1 entity.

    Strategy union order (all within same country):
      1. Word TF-IDF (expanded text)
      2. Char n-gram TF-IDF (raw text)  [if use_char_ngram]
      3. First-name-token exact index    [if use_first_token]
      4. Numeric-address-token index     [if use_numeric_token]

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

            cand_ids = sub_cand['id'].values

            # Strategy 1: Word TF-IDF
            print("  [1/4] Word TF-IDF scoring...")
            X_cand_word_T = word_vec.transform(sub_cand['expanded_text']).tocsr().T
            X_s1_word = word_vec.transform(sub_s1['expanded_text']).tocsr()
            word_hits = _tfidf_candidates(
                word_vec, X_cand_word_T, X_s1_word, cand_ids,
                top_k, tfidf_threshold, batch_size
            )
            del X_cand_word_T, X_s1_word
            gc.collect()

            # Strategy 2: Char n-gram TF-IDF (union)
            char_hits = defaultdict(set)
            if use_char_ngram and char_vec is not None:
                print("  [2/4] Char n-gram TF-IDF scoring...")
                X_cand_char_T = char_vec.transform(sub_cand['raw_text']).tocsr().T
                X_s1_char = char_vec.transform(sub_s1['raw_text']).tocsr()
                char_hits = _tfidf_candidates(
                    char_vec, X_cand_char_T, X_s1_char, cand_ids,
                    top_k, char_threshold, batch_size
                )
                del X_cand_char_T, X_s1_char
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
    parser.add_argument('--batch-size',     type=int,   default=50)
    parser.add_argument('--no-char-ngram',  action='store_true')
    parser.add_argument('--no-first-token', action='store_true')
    parser.add_argument('--no-numeric',     action='store_true')
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
        use_char_ngram=not args.no_char_ngram,
        use_first_token=not args.no_first_token,
        use_numeric_token=not args.no_numeric,
    )
