# Business Entity Resolution — Developer Guide

> **Challenge**: Match records from Source 2 & Source 3 to deduplicated Source 1 entities using noisy business name, address, and country fields.  
> **Metric**: F₀.₅ (precision-heavy — false merges are penalized 2× more than missed links).

---

## Repository Layout

```
student_resource/
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── generate_Candidates.py   # Stage 3 — blocking / candidate generation
│   │   ├── train_and_match.py       # Stage 5 — training + inference
│   │   └── evaluate.py              # Stage 2 — local F₀.₅ scorer (validation split)
│   ├── run_all.py                   # Top-level orchestration script
│   └── requirements.txt             # Pinned Python dependencies
├── dataset/                         # ⚠️  NOT in git — obtain from shared Drive
│   ├── train/  (train_source{1,2,3}.tsv, train_ground_truth.tsv)
│   └── test/   (test_source{1,2,3}.tsv)
├── output/                          # Generated output files (gitignored model pickle)
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
├── utils/
│   └── validate_submission.py       # Provided validator — stdlib only, no deps
├── problem_statement.md
└── Documentation_template.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd student_resource
```

### 2. Place the dataset

The `dataset/` folder is excluded from git (large files). Download from the shared Google Drive link and place it so the structure matches:

```
student_resource/
└── dataset/
    ├── train/
    │   ├── train_source1.tsv
    │   ├── train_source2.tsv
    │   ├── train_source3.tsv
    │   └── train_ground_truth.tsv
    └── test/
        ├── test_source1.tsv
        ├── test_source2.tsv
        └── test_source3.tsv
```

### 3. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## Running the Pipeline

All commands are run from the `student_resource/` directory with the venv active.

### Option A — Run everything at once

```bash
python code/business_entity_resolution/run_all.py
```

### Option B — Run stages individually

#### Step 1: Generate candidate pairs (blocking)

```bash
python code/business_entity_resolution/src/generate_Candidates.py
```

Output: `output/candidate_pairs.tsv`

#### Step 2: Train the matching model

```bash
python code/business_entity_resolution/src/train_and_match.py --mode train
```

Output: `output/er_model.pkl`

#### Step 3: Run inference

```bash
python code/business_entity_resolution/src/train_and_match.py --mode predict --threshold 0.5
```

Output: `output/matching_results.tsv`

#### Step 4: Validate output before submitting

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Should print `PASS` (exit 0). Fix any reported issues before uploading to the Portal.

---

## Evaluating on a Local Validation Split

To measure F₀.₅ without submitting, hold out part of the training data:

```bash
python code/business_entity_resolution/src/evaluate.py
```

This splits `train_ground_truth.tsv` 80/20, runs blocking + training on the 80% split, scores predictions against the 20% held-out split, and prints per-entity and macro-average F₀.₅.

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable — only merge completed, validated stages |
| `stage-2-infra` | Infrastructure (current) |
| `stage-3-blocking` | Improved candidate generation |
| `stage-4-features` | Richer pairwise features |
| `stage-5-training` | Classifier training & threshold tuning |
| `stage-7-embeddings` | Embedding-based re-ranking |

Open a PR to `main` when your stage is complete and `evaluate.py` shows improvement.

---

## Key Constraints (from problem statement)

- **No external data**: No APIs, geocoding, government databases, or internet lookups — immediate disqualification
- **Model license**: MIT or Apache 2.0 only
- **Model size**: ≤ 8 billion parameters
- **Every test S1 entity** must appear in `matching_results.tsv` (even singletons with empty matches)
- **Matched IDs** must only be S2-/S3- IDs that exist in the test set

---

## Submission Checklist

- [ ] `output/matching_results.tsv` and `output/candidate_pairs.tsv` generated
- [ ] `validate_submission.py` exits 0 (`PASS`)
- [ ] Upload `matching_results.tsv` to the Portal leaderboard
- [ ] For final submission: zip `output/`, `code/`, `Documentation_template.md` as `<teamname>_submission.zip`
