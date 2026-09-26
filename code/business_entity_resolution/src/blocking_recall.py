"""
blocking_recall.py
------------------
Measure blocking recall on the training ground-truth split.

Usage
-----
python src/blocking_recall.py \
    --s1  dataset/train/train_source1.tsv \
    --s2  dataset/train/train_source2.tsv \
    --s3  dataset/train/train_source3.tsv \
    --gt  dataset/train/train_ground_truth.tsv \
    --candidates output/train_candidate_pairs.tsv \
    [--generate]   # also run blocking first if candidates file is absent

Outputs
-------
Prints per-country blocking recall and overall recall.
Blocking recall = fraction of true positive pairs that appear in candidate set.
"""

import os
import sys
import argparse
import pandas as pd

# ---------------------------------------------------------------------------
# GT parsing (handles comma-sep S2+S3 IDs in matched_entity_ids)
# ---------------------------------------------------------------------------

def load_ground_truth(gt_path):
    """
    Returns a dict: {s1_id -> set of matched S2/S3 ids}.
    GT column format: matched_entity_ids = "S2-123,S2-456,S3-789"
    """
    df = pd.read_csv(gt_path, sep='\t', low_memory=False)

    id_col    = 'source1_entity_id' if 'source1_entity_id' in df.columns else df.columns[0]
    match_col = 'matched_entity_ids' if 'matched_entity_ids' in df.columns else df.columns[1]

    # Vectorized split at C level, then one tight zip loop to build the dict
    ids_arr    = df[id_col].astype(str).str.strip().values
    splits_ser = df[match_col].astype(str).str.strip().str.split(',')

    return {
        s1_id: {x.strip() for x in parts if x.strip()} if not (isinstance(parts, list) and parts == ['nan']) else set()
        for s1_id, parts in zip(ids_arr, splits_ser)
        if parts is not None
    }


def load_candidates(cand_path):
    """Returns dict: {s1_id -> set of candidate ids}."""
    df = pd.read_csv(cand_path, sep='\t', low_memory=False)
    id_col   = 'source1_entity_id'    if 'source1_entity_id'    in df.columns else df.columns[0]
    cand_col = 'candidate_entity_ids' if 'candidate_entity_ids' in df.columns else df.columns[1]

    # Vectorized split at C level, then one tight zip loop to build the dict
    ids_arr    = df[id_col].astype(str).str.strip().values
    splits_ser = df[cand_col].astype(str).str.strip().str.split(',')

    return {
        s1_id: {x.strip() for x in parts if x.strip()} if not (isinstance(parts, list) and parts == ['nan']) else set()
        for s1_id, parts in zip(ids_arr, splits_ser)
        if parts is not None
    }


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def compute_blocking_recall(gt, candidates, gt_path_for_country=None):
    """
    Compute blocking recall:
      recall = |{(s1,sx) in GT and in candidates}| / |{(s1,sx) in GT}|

    Returns (total_true_pairs, recovered_pairs, recall_pct)
    """
    total_true = 0
    recovered = 0

    missing = []

    for s1_id, true_matches in gt.items():
        if not true_matches:
            continue
        cand_set = candidates.get(s1_id, set())
        for sx_id in true_matches:
            total_true += 1
            if sx_id in cand_set:
                recovered += 1
            else:
                missing.append((s1_id, sx_id))

    recall = (recovered / total_true * 100) if total_true > 0 else 0.0
    return total_true, recovered, recall, missing


def main():
    parser = argparse.ArgumentParser(description='Measure blocking recall on train GT')
    parser.add_argument('--s1',         default='dataset/train/train_source1.tsv')
    parser.add_argument('--s2',         default='dataset/train/train_source2.tsv')
    parser.add_argument('--s3',         default='dataset/train/train_source3.tsv')
    parser.add_argument('--gt',         default='dataset/train/train_ground_truth.tsv')
    parser.add_argument('--candidates', default='output/train_candidate_pairs.tsv')
    parser.add_argument('--generate',   action='store_true',
                        help='Run blocking.generate_candidates() first if candidate file missing')
    parser.add_argument('--top-k',      type=int, default=50)
    parser.add_argument('--threshold',  type=float, default=0.05)
    args = parser.parse_args()

    if args.generate or not os.path.exists(args.candidates):
        print("Generating candidate pairs on training data...")
        # Import must be relative to src/
        src_dir = os.path.dirname(os.path.abspath(__file__))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from blocking import generate_candidates
        generate_candidates(
            s1_path=args.s1,
            s2_path=args.s2,
            s3_path=args.s3,
            output_path=args.candidates,
            top_k=args.top_k,
            tfidf_threshold=args.threshold,
            char_threshold=args.threshold,
        )

    print(f"\nLoading ground truth from {args.gt}...")
    gt = load_ground_truth(args.gt)
    print(f"  {len(gt):,} S1 entities in GT")

    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"  {len(candidates):,} S1 entities have candidates")

    total, recovered, recall, missing = compute_blocking_recall(gt, candidates)

    print("\n" + "=" * 60)
    print(f"BLOCKING RECALL REPORT")
    print("=" * 60)
    print(f"  True positive pairs in GT : {total:,}")
    print(f"  Recovered by blocking     : {recovered:,}")
    print(f"  Missed by blocking        : {total - recovered:,}")
    print(f"  Blocking recall           : {recall:.2f}%")
    print("=" * 60)

    if missing:
        print(f"\nSample of {min(20, len(missing))} missed pairs:")
        for s1_id, sx_id in missing[:20]:
            print(f"  S1={s1_id}  missed={sx_id}")

    target = 90.0
    if recall >= target:
        print(f"\n✅ Recall {recall:.2f}% meets target of {target}%")
    else:
        print(f"\n⚠️  Recall {recall:.2f}% is below target of {target}%. Consider increasing --top-k or tuning thresholds.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
