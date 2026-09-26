"""
evaluate.py — Local F₀.₅ scorer for the Business Entity Resolution pipeline.

Holds out a validation split from the training ground truth, runs blocking +
feature extraction + model training on the remaining training slice, then
scores predictions against the held-out slice.

Usage (from student_resource/ with venv active):
    python code/business_entity_resolution/src/evaluate.py
    python code/business_entity_resolution/src/evaluate.py --val-ratio 0.2 --threshold 0.5
"""

import argparse
import os
import random
import tempfile

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# F₀.₅ implementation (matches the leaderboard formula exactly)
# ---------------------------------------------------------------------------

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F_beta. Returns 0.0 if both precision and recall are 0."""
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(predicted: set, ground_truth: set) -> float:
    """Compute per-entity F₀.₅.

    Special cases per the problem statement:
    - If ground_truth is empty and predicted is empty → 1.0 (correct singleton)
    - If ground_truth is empty and predicted is non-empty → 0.0 (false merge on singleton)
    """
    if not ground_truth and not predicted:
        return 1.0
    if not ground_truth and predicted:
        return 0.0
    if not predicted:
        # Recall = 0, so F₀.₅ = 0
        return 0.0

    tp = len(predicted & ground_truth)
    precision = tp / len(predicted)
    recall = tp / len(ground_truth)
    return f_beta(precision, recall)


def compute_macro_f05(
    predictions: dict[str, list],
    ground_truth: dict[str, set],
) -> tuple[float, dict]:
    """Compute macro-average F₀.₅ across all Source-1 entities.

    Args:
        predictions:  {s1_id: [matched_id, ...]}
        ground_truth: {s1_id: {matched_id, ...}}  — all S1 entities, including singletons

    Returns:
        (macro_f05, per_entity_scores_dict)
    """
    per_entity: dict[str, float] = {}
    for s1_id, true_matches in ground_truth.items():
        pred_matches = set(predictions.get(s1_id, []))
        per_entity[s1_id] = score_entity(pred_matches, true_matches)

    macro = float(np.mean(list(per_entity.values()))) if per_entity else 0.0
    return macro, per_entity


# ---------------------------------------------------------------------------
# Validation split helpers
# ---------------------------------------------------------------------------

def split_ground_truth(
    gt_path: str,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly split ground truth rows into train and validation DataFrames."""
    df = pd.read_csv(gt_path, sep='\t')
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def gt_df_to_dict(df: pd.DataFrame) -> dict[str, set]:
    """Convert a ground truth DataFrame to {s1_id: set(matched_ids)}."""
    result: dict[str, set] = {}
    for _, row in df.iterrows():
        s1_id = str(row['source1_entity_id'])
        matches_str = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ''
        result[s1_id] = {m.strip() for m in matches_str.split(',') if m.strip()}
    return result


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def run_evaluation(
    train_dir: str = "dataset/train",
    val_ratio: float = 0.2,
    threshold: float = 0.5,
    neg_ratio: int = 5,
    random_seed: int = 42,
):
    """
    1. Split train_ground_truth.tsv into train_slice / val_slice.
    2. Train a model on the train_slice.
    3. Generate candidate pairs using the training source files.
    4. Run inference with the trained model.
    5. Score predictions against val_slice using macro F₀.₅.
    """
    # Import here to avoid circular imports if this module is imported elsewhere
    from train_and_match import train_model, extract_features_from_candidates, load_data_dict
    import pickle

    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    print(f"Splitting ground truth (val_ratio={val_ratio}, seed={random_seed})...")
    train_gt, val_gt = split_ground_truth(gt_path, val_ratio=val_ratio, random_seed=random_seed)

    val_ground_truth = gt_df_to_dict(val_gt)
    print(f"  Train slice: {len(train_gt)} entities | Val slice: {len(val_gt)} entities")

    # Write the train slice to a temp file for train_model() to consume
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_gt_path = os.path.join(tmpdir, "train_ground_truth.tsv")
        train_gt.to_csv(tmp_gt_path, sep='\t', index=False)

        # Copy source file paths (they're shared between train/val splits)
        tmp_train_dir = train_dir  # source TSVs are the same; only GT is split

        model_path = os.path.join(tmpdir, "eval_model.pkl")

        print("\n--- Training on train slice ---")
        model = train_model(
            train_dir=train_dir,
            model_path=model_path,
            neg_ratio=neg_ratio,
            _gt_override=tmp_gt_path,  # pass split GT instead of full GT
        )

        # Build candidate pairs for all val S1 entities from training sources
        # We use the full source files since they contain all entities
        print("\n--- Generating candidates for validation S1 entities ---")
        from generate_Candidates import generate_Candidates

        candidate_path = os.path.join(tmpdir, "val_candidates.tsv")

        # Filter S1 to only val entities for efficiency
        df_s1_full = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep='\t')
        val_s1_ids = set(val_gt['source1_entity_id'].astype(str))
        df_s1_val = df_s1_full[df_s1_full['entity_id'].astype(str).isin(val_s1_ids)]

        val_s1_path = os.path.join(tmpdir, "val_source1.tsv")
        df_s1_val.to_csv(val_s1_path, sep='\t', index=False)

        generate_Candidates(
            s1_path=val_s1_path,
            s2_path=os.path.join(train_dir, "train_source2.tsv"),
            s3_path=os.path.join(train_dir, "train_source3.tsv"),
            output_path=candidate_path,
        )

        # Feature extraction + scoring
        print("\n--- Scoring candidate pairs ---")

        # Temporarily point source_prefix to train files
        s1_dict   = load_data_dict(os.path.join(train_dir, "train_source1.tsv"))
        s2_dict   = load_data_dict(os.path.join(train_dir, "train_source2.tsv"))
        s3_dict   = load_data_dict(os.path.join(train_dir, "train_source3.tsv"))
        cand_dict = {**s2_dict, **s3_dict}

        from train_and_match import compute_pairwise_features, FEATURE_COLUMNS

        df_candidates = pd.read_csv(candidate_path, sep='\t')
        rows, pair_metadata = [], []
        for _, row in df_candidates.iterrows():
            s1_id    = str(row['source1_entity_id'])
            cand_str = str(row['candidate_entity_ids']) if pd.notna(row['candidate_entity_ids']) else ''
            if not cand_str.strip():
                continue
            s1_data = s1_dict.get(s1_id, {})
            for cand_id in (c.strip() for c in cand_str.split(',') if c.strip()):
                if cand_id in cand_dict:
                    rows.append(compute_pairwise_features(s1_data, cand_dict[cand_id]))
                    pair_metadata.append((s1_id, cand_id))

        X_val = pd.DataFrame(rows, columns=FEATURE_COLUMNS) if rows else pd.DataFrame(columns=FEATURE_COLUMNS)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        predictions: dict[str, list] = {}
        if not X_val.empty:
            proba = model.predict_proba(X_val)[:, 1]
            for (s1_id, cand_id), score in zip(pair_metadata, proba):
                if score >= threshold:
                    predictions.setdefault(s1_id, []).append(cand_id)

    # Score
    print("\n--- Results ---")
    macro_f05, per_entity = compute_macro_f05(predictions, val_ground_truth)

    scores = list(per_entity.values())
    print(f"  Macro F₀.₅         : {macro_f05:.4f}")
    print(f"  Entities scored     : {len(scores)}")
    print(f"  Perfect (1.0)       : {sum(1 for s in scores if s == 1.0)} ({100*sum(1 for s in scores if s==1.0)/len(scores):.1f}%)")
    print(f"  Zero (0.0)          : {sum(1 for s in scores if s == 0.0)} ({100*sum(1 for s in scores if s==0.0)/len(scores):.1f}%)")
    print(f"  Threshold used      : {threshold}")
    print(f"  Val ratio           : {val_ratio}")

    return macro_f05, per_entity


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local F₀.₅ evaluation on a validation split")
    parser.add_argument("--train-dir",  default="dataset/train")
    parser.add_argument("--val-ratio",  type=float, default=0.2,
                        help="Fraction of training ground truth to hold out for validation")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Match probability threshold (same as run_pipeline --threshold)")
    parser.add_argument("--neg-ratio",  type=int,   default=5)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    run_evaluation(
        train_dir=args.train_dir,
        val_ratio=args.val_ratio,
        threshold=args.threshold,
        neg_ratio=args.neg_ratio,
        random_seed=args.seed,
    )
