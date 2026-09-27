"""
quick_recall.py
---------------
Fast blocking recall estimation: O(minutes) instead of O(hours).

The Problem
-----------
Running vec.transform() on 4.1M India candidates takes 10-20+ minutes.
Even with --max-s1, the candidate transform is always O(n_candidates).

The Fix
-------
Use a SAMPLED candidate pool:
  1. Sample n_distractor random candidates per country (default 50K)
  2. Inject ALL true GT-matched records into the pool (so recall is measurable)
  3. Run blocking against this small pool (~50K + GT matches)
  4. Measure recall: if blocking finds the GT match in top-k, it is recalled

This gives a statistically valid recall estimate because:
  - True positives are always present in the pool
  - 50K distractors provide realistic top-k competition
  - Transform on 50K rows: ~1-2s instead of 10+ minutes

Usage
-----
python src/quick_recall.py \
    --s1  dataset/train/train_source1.tsv \
    --s2  dataset/train/train_source2.tsv \
    --s3  dataset/train/train_source3.tsv \
    --gt  dataset/train/train_ground_truth.tsv \
    [--top-k 50] [--threshold 0.05] [--n-distractors 50000] [--max-s1 0]
"""

import os, sys, gc, argparse
import numpy as np
import pandas as pd
from collections import defaultdict

_src = os.path.dirname(os.path.abspath(__file__))
if _src not in sys.path:
    sys.path.insert(0, _src)

from normalize import (
    normalize_name_s, normalize_address_s, normalize_country_s,
    first_token_s, numeric_tokens_list, normalize_name, first_name_token,
    extract_numeric_tokens
)
from blocking import (
    load_source_df, _build_word_vectorizer, _tfidf_candidates_subtok,
    _top_k_sparse_row
)


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def load_gt(gt_path):
    """Returns {s1_id -> set of matched S2/S3 ids}."""
    df = pd.read_csv(gt_path, sep='\t', low_memory=False)
    id_col    = 'source1_entity_id' if 'source1_entity_id' in df.columns else df.columns[0]
    match_col = 'matched_entity_ids' if 'matched_entity_ids' in df.columns else df.columns[1]
    ids_arr    = df[id_col].astype(str).str.strip().values
    splits_ser = df[match_col].astype(str).str.strip().str.split(',')
    gt = {}
    for s1_id, parts in zip(ids_arr, splits_ser):
        gt[s1_id] = {x.strip() for x in parts if x.strip() and x.strip().lower() != 'nan'}
    return gt


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------

def estimate_recall(
    s1_path, s2_path, s3_path, gt_path,
    top_k=50,
    tfidf_threshold=0.05,
    n_distractors=50_000,
    max_s1=0,
    seed=42,
):
    """
    Estimate blocking recall using a sampled candidate pool.

    Parameters
    ----------
    n_distractors : int
        Random non-GT candidates per country used as competition in the pool.
        50K gives realistic top-k competition; transforms in ~1-2s.
    max_s1 : int
        Evaluate only this many S1 entities (0 = all in GT).
    """
    rng = np.random.default_rng(seed)

    print("Loading source files (using pickle cache if available)...")
    df_s1 = load_source_df(s1_path, 'S1')
    df_s2 = load_source_df(s2_path, 'S2')
    df_s3 = load_source_df(s3_path, 'S3')

    print("Combining S2+S3 candidates...")
    df_cand = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()

    # Index candidates by ID for fast GT lookup
    cand_id_to_idx = {cid: i for i, cid in enumerate(df_cand['id'].values)}

    print(f"Loading GT from {gt_path}...")
    gt_full = load_gt(gt_path)
    print(f"  {len(gt_full):,} S1 entities in GT")

    # Optional S1 sampling
    s1_ids_eval = list(gt_full.keys())
    if max_s1 and max_s1 < len(s1_ids_eval):
        s1_ids_eval = list(rng.choice(s1_ids_eval, size=max_s1, replace=False))
        print(f"  Sampled {len(s1_ids_eval):,} S1 entities for evaluation")

    # Build s1_id -> row index in df_s1
    s1_id_to_idx = {sid: i for i, sid in enumerate(df_s1['id'].values)}

    # Collect valid (s1_id, gt_match_id) pairs
    eval_pairs = []
    for s1_id in s1_ids_eval:
        matches = gt_full.get(s1_id, set())
        s1_idx = s1_id_to_idx.get(s1_id)
        if s1_idx is None:
            continue
        for mx_id in matches:
            cand_idx = cand_id_to_idx.get(mx_id)
            if cand_idx is not None:
                eval_pairs.append((s1_id, s1_idx, mx_id, cand_idx))

    print(f"  {len(eval_pairs):,} evaluable GT pairs (s1 in S1 + match in S2/S3)")

    # ---------------------------------------------------------------------------
    # Per-country evaluation
    # ---------------------------------------------------------------------------
    total_gt    = 0
    recalled    = 0
    missed_ex   = []

    countries = sorted(c for c in df_s1['country'].unique() if c)
    if not countries:
        countries = ['ALL']

    for country in countries:
        if country != 'ALL':
            s1_mask   = df_s1['country'] == country
            cand_mask = df_cand['country'] == country
        else:
            s1_mask   = pd.Series([True] * len(df_s1))
            cand_mask = pd.Series([True] * len(df_cand))

        sub_s1   = df_s1[s1_mask].reset_index(drop=True)
        sub_cand = df_cand[cand_mask].reset_index(drop=True)

        # Filter eval_pairs to this country
        country_pairs = [
            (s1_id, s1_idx, mx_id, cand_idx)
            for s1_id, s1_idx, mx_id, cand_idx in eval_pairs
            if df_s1.at[s1_idx, 'country'] == country
        ]
        if not country_pairs:
            continue

        n_q = len(sub_s1)
        n_c = len(sub_cand)
        print(f"\n--- Country '{country}': {n_q:,} S1, {n_c:,} candidates, "
              f"{len(country_pairs):,} GT pairs to evaluate ---")

        # ----- Build sampled candidate pool -----
        # Always include the GT-matched records + random distractors
        gt_cand_global_idxs = [cand_idx for _, _, _, cand_idx in country_pairs]
        gt_cand_global_set  = set(gt_cand_global_idxs)

        # All candidate indices within this country (need local→global mapping)
        cand_local_idxs = np.where(cand_mask.values)[0]  # global indices

        # Sample distractors (exclude GT matches to get cleaner pool)
        non_gt_local = [i for i in cand_local_idxs if i not in gt_cand_global_set]
        n_dist = min(n_distractors, len(non_gt_local))
        distractor_idxs = rng.choice(non_gt_local, size=n_dist, replace=False).tolist()

        pool_global_idxs = np.array(sorted(set(distractor_idxs) | gt_cand_global_set), dtype=np.int64)
        pool_df = df_cand.iloc[pool_global_idxs].reset_index(drop=True)

        # Map global cand idx -> pool row index
        global_to_pool = {g: p for p, g in enumerate(pool_global_idxs)}

        pool_size = len(pool_df)
        print(f"  Pool: {pool_size:,} candidates ({n_dist:,} distractors + {len(gt_cand_global_set):,} GT matches)")

        # ----- Build word TF-IDF on pool -----
        print("  Fitting word TF-IDF on pool...")
        word_vec = _build_word_vectorizer(pool_df['expanded_text'], sample_size=min(250_000, pool_size))

        # S1 entities that appear in eval_pairs for this country
        eval_s1_global_idxs = list({s1_idx for _, s1_idx, _, _ in country_pairs})
        eval_s1_df = df_s1.iloc[eval_s1_global_idxs].reset_index(drop=True)
        s1_global_to_local = {g: l for l, g in enumerate(eval_s1_global_idxs)}

        print(f"  Transforming {len(eval_s1_df):,} S1 + {pool_size:,} pool records...")
        X_s1_word   = word_vec.transform(eval_s1_df['expanded_text']).tocsr()
        X_pool_word = word_vec.transform(pool_df['expanded_text']).tocsr()

        pool_ids     = pool_df['id'].values
        s1_local_ids = eval_s1_df['id'].values
        pool_ftoks   = pool_df['first_token'].values
        s1_ftoks     = eval_s1_df['first_token'].values
        pool_cand_ids_arr = pool_df['id'].values

        # Run sub-blocked TF-IDF on the small pool
        word_hits = _tfidf_candidates_subtok(
            X_s1_word, X_pool_word,
            s1_local_ids, pool_cand_ids_arr,
            s1_ftoks, pool_ftoks,
            top_k, tfidf_threshold, batch_size=500,
            max_bucket_cands=50_000, rng=rng,
        )
        # word_hits: {s1_local_idx -> set of matched candidate IDs}

        # Also run first-token exact match on the pool
        token_hits = defaultdict(set)
        pool_tok_index = defaultdict(list)
        for j, tok in enumerate(pool_ftoks):
            if tok:
                pool_tok_index[tok].append(pool_cand_ids_arr[j])
        for i, tok in enumerate(s1_ftoks):
            if tok and tok in pool_tok_index:
                token_hits[i].update(pool_tok_index[tok])

        # Also run numeric-token match on the pool
        num_hits = defaultdict(set)
        pool_num_toks = pool_df['num_tokens'].tolist()
        s1_num_toks   = eval_s1_df['num_tokens'].tolist()
        pool_num_index = defaultdict(list)
        for j, toks in enumerate(pool_num_toks):
            for t in toks:
                pool_num_index[t].append(pool_cand_ids_arr[j])
        for i, toks in enumerate(s1_num_toks):
            for t in toks:
                if t in pool_num_index:
                    num_hits[i].update(pool_num_index[t])

        # ----- Score each GT pair -----
        for s1_id, s1_global_idx, mx_id, cand_global_idx in country_pairs:
            total_gt += 1
            s1_local = s1_global_to_local.get(s1_global_idx)
            if s1_local is None:
                continue
            # Union of all strategy hits for this S1
            all_hits = (
                word_hits.get(s1_local, set())
                | token_hits.get(s1_local, set())
                | num_hits.get(s1_local, set())
            )
            if mx_id in all_hits:
                recalled += 1
            else:
                missed_ex.append((s1_id, mx_id))

        del X_s1_word, X_pool_word, pool_df, eval_s1_df
        gc.collect()

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    recall = (recalled / total_gt * 100) if total_gt > 0 else 0.0
    print("\n" + "=" * 60)
    print("BLOCKING RECALL ESTIMATE (sampled candidate pool)")
    print("=" * 60)
    print(f"  GT pairs evaluated         : {total_gt:,}")
    print(f"  Recalled by blocking       : {recalled:,}")
    print(f"  Missed                     : {total_gt - recalled:,}")
    print(f"  Recall                     : {recall:.2f}%")
    print(f"  Distractor pool size       : {n_distractors:,} per country")
    print("=" * 60)

    if missed_ex:
        print(f"\nSample of {min(20, len(missed_ex))} missed pairs:")
        for s1_id, mx_id in missed_ex[:20]:
            print(f"  S1={s1_id}  missed={mx_id}")

    target = 90.0
    if recall >= target:
        print(f"\n\u2705 Recall {recall:.2f}% >= {target}% target")
    else:
        print(f"\n\u26a0\ufe0f  Recall {recall:.2f}% < {target}% target")
        print("  Suggestions: increase --top-k, lower --threshold, or add more strategies")

    return recall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fast blocking recall estimation via sampled candidate pool')
    parser.add_argument('--s1',             default='dataset/train/train_source1.tsv')
    parser.add_argument('--s2',             default='dataset/train/train_source2.tsv')
    parser.add_argument('--s3',             default='dataset/train/train_source3.tsv')
    parser.add_argument('--gt',             default='dataset/train/train_ground_truth.tsv')
    parser.add_argument('--top-k',          type=int,   default=50)
    parser.add_argument('--threshold',      type=float, default=0.05)
    parser.add_argument('--n-distractors',  type=int,   default=50_000,
                        help='Random candidate distractors per country (default 50000)')
    parser.add_argument('--max-s1',         type=int,   default=0,
                        help='Evaluate only this many S1 entities from GT (0=all)')
    parser.add_argument('--seed',           type=int,   default=42)
    args = parser.parse_args()

    estimate_recall(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        gt_path=args.gt,
        top_k=args.top_k,
        tfidf_threshold=args.threshold,
        n_distractors=args.n_distractors,
        max_s1=args.max_s1,
        seed=args.seed,
    )
