import os
import random
import pickle
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import xgboost as xgb


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_pairwise_features(df1: dict, df2: dict) -> dict:
    """Computes string similarity and field match features between two entities.

    The TSV schema has exactly three content fields:
        business_name, business_address, country
    (entity_id is the key and is NOT used as a feature).
    """
    # --- Issue 2 fix ---
    # Previous code used 'name', 'address', 'city', 'phone' — none of which exist
    # in the TSV.  The actual column names are business_name, business_address, country.
    name1  = str(df1.get('business_name',    '') or '')
    name2  = str(df2.get('business_name',    '') or '')
    addr1  = str(df1.get('business_address', '') or '')
    addr2  = str(df2.get('business_address', '') or '')
    cty1   = str(df1.get('country',          '') or '')
    cty2   = str(df2.get('country',          '') or '')

    features = {
        # Name similarities
        'name_ratio':           fuzz.ratio(name1, name2)           / 100.0,
        'name_partial_ratio':   fuzz.partial_ratio(name1, name2)   / 100.0,
        'name_token_sort_ratio': fuzz.token_sort_ratio(name1, name2) / 100.0,
        'name_token_set_ratio': fuzz.token_set_ratio(name1, name2)  / 100.0,

        # Address similarities
        'addr_ratio':           fuzz.ratio(addr1, addr2)            / 100.0,
        'addr_token_sort_ratio': fuzz.token_sort_ratio(addr1, addr2) / 100.0,

        # Country exact match (replaces non-existent city/phone fields)
        'country_exact_match':  1.0 if cty1.lower() == cty2.lower() and cty1 != '' else 0.0,
    }
    return features


FEATURE_COLUMNS = [
    'name_ratio', 'name_partial_ratio', 'name_token_sort_ratio', 'name_token_set_ratio',
    'addr_ratio', 'addr_token_sort_ratio', 'country_exact_match',
]


def load_data_dict(tsv_path: str) -> dict:
    """Loads entity TSV into a dictionary keyed by entity_id for fast lookup."""
    df = pd.read_csv(tsv_path, sep='\t')
    id_col = 'entity_id' if 'entity_id' in df.columns else df.columns[0]
    return df.set_index(id_col).to_dict(orient='index')


# ---------------------------------------------------------------------------
# Issue 1 fix — actual training on ground truth
# ---------------------------------------------------------------------------

def _generate_training_pairs(
    gt_path: str,
    cand_dict: dict,
    neg_ratio: int = 5,
    random_seed: int = 42,
) -> list:
    """Build (s1_id, cand_id, label) tuples from the ground-truth file.

    Positives: every (S1, S2/S3) pair listed in train_ground_truth.tsv.
    Negatives: randomly sampled non-matching (S1, S2/S3) pairs at neg_ratio:1.
    """
    random.seed(random_seed)
    df_gt = pd.read_csv(gt_path, sep='\t')

    positive_pairs = []
    for _, row in df_gt.iterrows():
        s1_id = str(row['source1_entity_id'])
        matches_str = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ''
        for mid in (m.strip() for m in matches_str.split(',') if m.strip()):
            if mid in cand_dict:
                positive_pairs.append((s1_id, mid, 1))

    print(f"  Positive pairs: {len(positive_pairs)}")

    # Hard negatives sampled within the full candidate pool
    all_cand_ids = list(cand_dict.keys())
    seen_pairs   = {(s1, c) for s1, c, _ in positive_pairs}
    negative_pairs = []

    for s1_id, _, _ in positive_pairs:
        sampled = 0
        attempts = 0
        while sampled < neg_ratio and attempts < neg_ratio * 20:
            neg_cand = random.choice(all_cand_ids)
            if (s1_id, neg_cand) not in seen_pairs:
                negative_pairs.append((s1_id, neg_cand, 0))
                seen_pairs.add((s1_id, neg_cand))
                sampled += 1
            attempts += 1

    print(f"  Negative pairs: {len(negative_pairs)}")
    return positive_pairs + negative_pairs


def _extract_features_from_pairs(
    pairs: list,
    s1_dict: dict,
    cand_dict: dict,
) -> tuple:
    """Extract pairwise features for a list of (s1_id, cand_id, label) tuples."""
    rows, labels, meta = [], [], []
    for s1_id, cand_id, label in pairs:
        feats = compute_pairwise_features(
            s1_dict.get(s1_id, {}),
            cand_dict.get(cand_id, {}),
        )
        rows.append(feats)
        labels.append(label)
        meta.append((s1_id, cand_id))
    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return X, np.array(labels), meta


def train_model(
    train_dir: str = "dataset/train",
    model_path: str = "output/er_model.pkl",
    neg_ratio: int = 5,
    _gt_override: str = None,  # Internal: used by evaluate.py to pass a split GT file
) -> xgb.XGBClassifier:
    """Train XGBoost on ground-truth pairs and save the fitted model to disk.

    Must be called before run_pipeline().

    Args:
        train_dir:   Directory containing train_source{1,2,3}.tsv and
                     train_ground_truth.tsv.
        model_path:  Where to save the pickled model.
        neg_ratio:   Number of negative samples per positive pair.

    Returns:
        The fitted XGBClassifier.
    """
    print("=== Training Phase ===")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = _gt_override if _gt_override else os.path.join(train_dir, "train_ground_truth.tsv")

    print("Loading training entity data...")
    s1_dict   = load_data_dict(s1_path)
    s2_dict   = load_data_dict(s2_path)
    s3_dict   = load_data_dict(s3_path)
    cand_dict = {**s2_dict, **s3_dict}
    print(f"  S1: {len(s1_dict):,}  S2: {len(s2_dict):,}  S3: {len(s3_dict):,} entities")

    print("Generating labeled pairs from ground truth...")
    pairs = _generate_training_pairs(gt_path, cand_dict, neg_ratio=neg_ratio)

    print("Extracting pairwise features...")
    X, y, _ = _extract_features_from_pairs(pairs, s1_dict, cand_dict)
    print(f"  Feature matrix: {X.shape}  |  positives: {int(y.sum())}  negatives: {int((y == 0).sum())}")

    # Compensate for class imbalance — XGBoost scale_pos_weight = neg / pos
    pos_count = int(y.sum())
    neg_count = int((y == 0).sum())
    spw = neg_count / pos_count if pos_count > 0 else 1.0

    print(f"Training XGBoost (scale_pos_weight={spw:.2f})...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved → '{model_path}'")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def extract_features_from_candidates(
    candidate_tsv: str,
    dataset_dir: str,
    source_prefix: str = "test",
) -> tuple:
    """Parses candidates.tsv and builds a feature matrix for inference."""
    print("Loading entity datasets...")
    s1_dict   = load_data_dict(os.path.join(dataset_dir, f"{source_prefix}_source1.tsv"))
    s2_dict   = load_data_dict(os.path.join(dataset_dir, f"{source_prefix}_source2.tsv"))
    s3_dict   = load_data_dict(os.path.join(dataset_dir, f"{source_prefix}_source3.tsv"))
    cand_dict = {**s2_dict, **s3_dict}

    df_candidates = pd.read_csv(candidate_tsv, sep='\t')

    rows, pair_metadata = [], []

    print("Extracting similarity features for candidate pairs...")
    for _, row in df_candidates.iterrows():
        s1_id    = str(row['source1_entity_id'])
        cand_str = str(row['candidate_entity_ids']) if pd.notna(row['candidate_entity_ids']) else ''

        if not cand_str.strip():
            continue

        cand_ids = [c.strip() for c in cand_str.split(',') if c.strip()]
        s1_data  = s1_dict.get(s1_id, {})

        for cand_id in cand_ids:
            if cand_id in cand_dict:
                feats = compute_pairwise_features(s1_data, cand_dict[cand_id])
                rows.append(feats)
                pair_metadata.append((s1_id, cand_id))

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS) if rows else pd.DataFrame(columns=FEATURE_COLUMNS)
    return X, pair_metadata, df_candidates


def run_pipeline(
    candidate_tsv: str = "output/candidate_pairs.tsv",
    test_dir: str = "dataset/test",
    output_tsv: str = "output/matching_results.tsv",
    model_path: str = "output/er_model.pkl",
    probability_threshold: float = 0.5,  # Calibrate on a validation split; higher → more precision
):
    """Run inference using the trained model.

    Loads the model saved by train_model(), scores every candidate pair, and
    writes matching_results.tsv with exactly one row per Source-1 entity.

    Raises FileNotFoundError if the model has not been trained yet.
    """
    # --- Issue 1 fix ---
    # Load the model that was fitted on real training data instead of using
    # hardcoded weight vectors that never touch the ground truth.
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Call train_model() first to fit and save the model."
        )

    print(f"Loading trained model from '{model_path}'...")
    with open(model_path, 'rb') as f:
        model: xgb.XGBClassifier = pickle.load(f)

    # Step 1: Feature extraction for test candidate pairs
    X_test, pair_metadata, df_candidates = extract_features_from_candidates(
        candidate_tsv, test_dir
    )

    if X_test.empty:
        print("No candidate pairs to evaluate.")
    else:
        # Step 2: Score with trained model (predict_proba, column 1 = match probability)
        print("Scoring candidate pairs with trained model...")
        proba = model.predict_proba(X_test)[:, 1]

        # Step 3: Filter by threshold (tuned for F_0.5 precision-heavy weighting)
        predictions: dict[str, list] = {}
        for (s1_id, cand_id), score in zip(pair_metadata, proba):
            if score >= probability_threshold:
                predictions.setdefault(s1_id, []).append(cand_id)

    # Step 4: Build matching_results.tsv — every Source-1 entity must have exactly one row
    results = []
    seen_s1: set = set()
    for _, row in df_candidates.iterrows():
        s1_id = str(row['source1_entity_id'])
        if s1_id in seen_s1:
            continue
        seen_s1.add(s1_id)

        matched_list = predictions.get(s1_id, []) if not X_test.empty else []

        # Deduplicate while preserving insertion order
        seen_ids: set = set()
        deduped = [x for x in matched_list if not (x in seen_ids or seen_ids.add(x))]

        results.append({
            'source1_entity_id': s1_id,
            'matched_entity_ids': ','.join(deduped),
        })

    os.makedirs(os.path.dirname(output_tsv) or '.', exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_tsv, sep='\t', index=False)
    print(f"Matching complete. Saved {len(df_out)} predictions to '{output_tsv}'.")


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Entity Resolution — train on ground truth then run inference."
    )
    parser.add_argument(
        "--mode", choices=["train", "predict", "all"], default="all",
        help=(
            "train  : fit model on training data and save it.\n"
            "predict: load saved model and score test candidates.\n"
            "all    : train then predict (default)."
        ),
    )
    parser.add_argument("--train-dir",     default="dataset/train",
                        help="Directory with train_source*.tsv and train_ground_truth.tsv")
    parser.add_argument("--test-dir",      default="dataset/test",
                        help="Directory with test_source*.tsv")
    parser.add_argument("--candidate-tsv", default="output/candidate_pairs.tsv",
                        help="Candidate pairs produced by generate_Candidates.py")
    parser.add_argument("--output-tsv",    default="output/matching_results.tsv")
    parser.add_argument("--model-path",    default="output/er_model.pkl")
    parser.add_argument("--threshold",     type=float, default=0.5,
                        help="Match probability threshold (higher = more precision, fewer matches)")
    parser.add_argument("--neg-ratio",     type=int, default=5,
                        help="Negative samples per positive pair during training")
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        train_model(
            train_dir=args.train_dir,
            model_path=args.model_path,
            neg_ratio=args.neg_ratio,
        )

    if args.mode in ("predict", "all"):
        run_pipeline(
            candidate_tsv=args.candidate_tsv,
            test_dir=args.test_dir,
            output_tsv=args.output_tsv,
            model_path=args.model_path,
            probability_threshold=args.threshold,
        )