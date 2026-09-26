import os
import re
import gc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


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
    ID prefixes do not contaminate the TF-IDF space.
    """
    content_cols = [c for c in ['business_name', 'business_address', 'country'] if c in df.columns]
    if content_cols:
        return df[content_cols].fillna('').astype(str).agg(' '.join, axis=1)
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


def load_processed_df(path: str) -> pd.DataFrame:
    """Reads TSV and retains only id, country, and combined text to conserve memory."""
    print(f"  Reading {os.path.basename(path)}...")
    df = pd.read_csv(path, sep='\t')
    id_col = 'entity_id' if 'entity_id' in df.columns else df.columns[0]
    country = (
        df['country'].fillna('').astype(str).str.strip().str.upper()
        if 'country' in df.columns
        else pd.Series(['US'] * len(df))
    )
    text = combine_features(df)
    res = pd.DataFrame({
        'id': df[id_col].astype(str),
        'country': country,
        'clean_text': text
    })
    del df
    gc.collect()
    return res


def generate_Candidates(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_path: str = "output/candidate_pairs.tsv",
    top_k: int = 15,
    similarity_threshold: float = 0.10,
    batch_size: int = 50
):
    print("Loading datasets with strict memory management...")
    df_s1 = load_processed_df(s1_path)
    df_s2 = load_processed_df(s2_path)
    df_s3 = load_processed_df(s3_path)

    print("Combining candidate records...")
    df_candidates = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()

    print("Configuring Word (1, 2) n-gram TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.6,
        max_features=100000,
        dtype=np.float32,
        sublinear_tf=True
    )

    # Fit vectorizer on a representative sample (250,000 rows)
    sample_size = min(250000, len(df_candidates))
    print(f"Fitting vocabulary on a representative sample of {sample_size:,} records (takes ~5s)...")
    sample_corpus = df_candidates['clean_text'].sample(n=sample_size, random_state=42)
    vectorizer.fit(sample_corpus)
    del sample_corpus
    gc.collect()
    print(f"Vocabulary successfully built: {len(vectorizer.vocabulary_):,} features")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Opening output file {output_path} for streaming writes...")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")

        # Process country by country to keep matrix sizes within memory bounds
        countries = [c for c in df_s1['country'].unique() if c]
        if not countries:
            countries = ['ALL']

        for c in countries:
            if c != 'ALL':
                s1_mask = (df_s1['country'] == c).values
                cand_mask = (df_candidates['country'] == c).values
                block_name = f"Country '{c}'"
            else:
                s1_mask = np.ones(len(df_s1), dtype=bool)
                cand_mask = np.ones(len(df_candidates), dtype=bool)
                block_name = "All Records"

            sub_s1_text = df_s1.loc[s1_mask, 'clean_text']
            sub_s1_ids = df_s1.loc[s1_mask, 'id'].values

            sub_cand_text = df_candidates.loc[cand_mask, 'clean_text']
            sub_cand_ids = df_candidates.loc[cand_mask, 'id'].values

            n_queries = len(sub_s1_ids)
            n_cands = len(sub_cand_ids)

            if n_queries == 0 or n_cands == 0:
                continue

            print(f"\nProcessing {block_name}: {n_queries:,} S1 queries against {n_cands:,} candidates...")

            print(f"  Transforming {n_cands:,} candidates into sparse matrix...")
            X_cand = vectorizer.transform(sub_cand_text).tocsr().T  # Transposed for dot product
            del sub_cand_text
            gc.collect()

            print(f"  Transforming {n_queries:,} queries into sparse matrix...")
            X_s1 = vectorizer.transform(sub_s1_text).tocsr()
            del sub_s1_text
            gc.collect()

            print(f"  Searching top-{top_k} candidates across {n_queries:,} queries in batches of {batch_size}...")

            for start_idx in range(0, n_queries, batch_size):
                end_idx = min(start_idx + batch_size, n_queries)
                batch_s1 = X_s1[start_idx:end_idx]

                # Dot product with batch_size=50 creates an intermediate matrix <100MB
                sim_sparse = batch_s1.dot(X_cand).tocsr()

                lines = []
                for i in range(sim_sparse.shape[0]):
                    s1_id = sub_s1_ids[start_idx + i]
                    row = sim_sparse[i]

                    matched_indices, _ = get_top_k_sparse(row, top_k, similarity_threshold)
                    matched_cand_ids = sub_cand_ids[matched_indices].tolist()

                    lines.append(f"{s1_id}\t{','.join(str(x) for x in matched_cand_ids)}\n")

                out_f.writelines(lines)

                if (start_idx // batch_size) % 200 == 0 or end_idx == n_queries:
                    print(f"  Processed {end_idx:,}/{n_queries:,} queries in {block_name}...")

            del X_cand, X_s1, sub_s1_ids, sub_cand_ids
            gc.collect()

    del df_s1, df_candidates
    gc.collect()
    print(f"\nCandidate generation complete! Results saved to {output_path}")


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