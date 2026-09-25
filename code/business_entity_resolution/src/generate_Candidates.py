import os
import re
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# NearestNeighbors imported for optional future use
# from sklearn.neighbors import NearestNeighbors


def clean_text(text: str) -> str:
    """Standardize and clean text fields for entity matching."""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove common business suffixes
    text = re.sub(r'\b(inc|incorporated|llc|corp|corporation|ltd|limited|co|company)\b', '', text)
    # Collapse multiple spaces
    cleaned = ' '.join(text.split())
    return cleaned


def combine_features(df: pd.DataFrame) -> pd.Series:
    """Concatenate business_name, business_address, and country into a single search string.

    We explicitly select only the content columns (not entity_id) so that the S1-/S2-/S3-
    ID prefixes do not contaminate the TF-IDF character n-gram space.
    """
    content_cols = [c for c in ['business_name', 'business_address', 'country'] if c in df.columns]
    if content_cols:
        return df[content_cols].fillna('').astype(str).agg(' '.join, axis=1)
    # Fallback: use all columns (original behavior) if expected columns are missing
    return df.astype(str).agg(' '.join, axis=1)


def get_top_k_sparse(sparse_row, top_k, threshold):
    """Utility to extract top-k indices and scores directly from a sparse matrix row."""
    if sparse_row.nnz == 0:
        return [], []

    data = sparse_row.data
    indices = sparse_row.indices

    # Filter by threshold first
    mask = data >= threshold
    data = data[mask]
    indices = indices[mask]

    if len(data) == 0:
        return [], []

    # Get top-k indices
    if len(data) > top_k:
        top_k_partition = np.argpartition(data, -top_k)[-top_k:]
        sorted_order = top_k_partition[np.argsort(-data[top_k_partition])]
    else:
        sorted_order = np.argsort(-data)

    return indices[sorted_order], data[sorted_order]


def generate_Candidates(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_path: str = "output/candidate_pairs.tsv",
    top_k: int = 15,
    similarity_threshold: float = 0.10,
    batch_size: int = 50  # Reduced batch size to stay safely within memory bounds
):
    print("Loading datasets...")
    df_s1 = pd.read_csv(s1_path, sep='\t')
    df_s2 = pd.read_csv(s2_path, sep='\t')
    df_s3 = pd.read_csv(s3_path, sep='\t')

    id_col = 'entity_id' if 'entity_id' in df_s1.columns else df_s1.columns[0]
    df_candidates = pd.concat([df_s2, df_s3], ignore_index=True)

    print("Cleaning and building feature strings...")
    s1_text = combine_features(df_s1)
    candidates_text = combine_features(df_candidates)
    candidate_ids = df_candidates[id_col].values

    print("Vectorizing with Character n-grams TF-IDF...")
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True
    )

    all_text = pd.concat([s1_text, candidates_text], ignore_index=True)
    vectorizer.fit(all_text)

    X_s1 = vectorizer.transform(s1_text).tocsr()
    X_cand = vectorizer.transform(candidates_text).tocsr().T  # Transpose for dot product

    print(f"Finding top {top_k} candidates per Source-1 entity...")
    n_s1 = X_s1.shape[0]
    results = []

    for start_idx in range(0, n_s1, batch_size):
        end_idx = min(start_idx + batch_size, n_s1)
        batch_s1 = X_s1[start_idx:end_idx]

        # Dot product remains strictly in CSR sparse format
        sim_sparse = batch_s1.dot(X_cand).tocsr()

        for i in range(sim_sparse.shape[0]):
            s1_id = df_s1[id_col].iloc[start_idx + i]
            row = sim_sparse[i]

            matched_indices, _ = get_top_k_sparse(row, top_k, similarity_threshold)
            matched_cand_ids = candidate_ids[matched_indices].tolist()

            # Output column names must match what train_and_match.py reads and what
            # validate_submission.py checks: source1_entity_id / candidate_entity_ids.
            # IDs are written as a comma-separated string (not a Python list repr).
            results.append({
                'source1_entity_id': s1_id,
                'candidate_entity_ids': ','.join(str(x) for x in matched_cand_ids),
            })

        if (start_idx // batch_size) % 100 == 0:
            print(f"  Processed {end_idx}/{n_s1} Source-1 records...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving results to {output_path}...")
    pd.DataFrame(results).to_csv(output_path, sep='\t', index=False)
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Blocking / candidate generation stage.")
    parser.add_argument("--s1",         default="dataset/test/test_source1.tsv")
    parser.add_argument("--s2",         default="dataset/test/test_source2.tsv")
    parser.add_argument("--s3",         default="dataset/test/test_source3.tsv")
    parser.add_argument("--output",     default="output/candidate_pairs.tsv")
    parser.add_argument("--top-k",      type=int,   default=15)
    parser.add_argument("--threshold",  type=float, default=0.10)
    parser.add_argument("--batch-size", type=int,   default=50)
    args = parser.parse_args()

    generate_Candidates(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        output_path=args.output,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        batch_size=args.batch_size,
    )