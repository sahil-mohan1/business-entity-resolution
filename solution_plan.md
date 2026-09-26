# Business Entity Resolution Challenge — Solution Plan

## Task Summary
Match records from **Source 2** and **Source 3** to the deduplicated reference **Source 1** using noisy business name + address + country fields.  
Metric: **F₀.₅ (precision-heavy)** — false merges penalized 2× more than missed links.

---

## ⚠️ Critical Issues Status

### ✅ Issue 1 — ~~`train_and_match.py` is a Fake ML Model~~ **FIXED**
Added `train_model()` that reads `train_ground_truth.tsv`, generates positive + hard-negative pairs, extracts features, and fits `XGBClassifier` with `scale_pos_weight`.  
`run_pipeline()` now loads the pickled model and calls `predict_proba()` instead of hardcoded weight dot-product.

---

### ✅ Issue 2 — ~~Wrong Column Names in Feature Extraction~~ **FIXED**
`compute_pairwise_features()` now correctly uses `business_name`, `business_address`, `country`.  
Non-existent `city` / `phone` fields removed; `country_exact_match` feature added.

---

### ✅ Issue 3 — ~~`candidate_pairs.tsv` Output Column is Wrong~~ **FIXED**
`generate_Candidates.py` now writes `source1_entity_id` / `candidate_entity_ids` as comma-separated strings (not Python list repr). Matches what `train_and_match.py` and `validate_submission.py` expect.

---

### 🟡 Issue 4 — No Validation Split / Offline Scoring
There is no code to hold out a validation set from training data and compute an F₀.₅ score locally. Without this, you are flying blind on what `probability_threshold` to pick.  
**Fix**: Add an `evaluate.py` script that splits training data and scores F₀.₅.

---

### ✅ Issue 5 — ~~Duplicate `import` Statements~~ **FIXED**
Duplicate `numpy` / `pandas` / `TfidfVectorizer` imports removed from `generate_Candidates.py`.  
Also fixed: `entity_id` column now excluded from TF-IDF text to avoid S1-/S2-/S3- prefix contamination.

---

### 🟡 Issue 6 — Data Scale vs. Memory
- **Training data**: Source 1 ~200 MB, Source 2 ~489 MB, Source 3 ~504 MB  
- **Ground truth**: ~127 MB  
- The batch TF-IDF approach in `generate_Candidates.py` is correctly batched, but training label extraction (negative sampling) needs to be done carefully at scale.  
**Fix**: Use chunked reading or Dask/sampling strategy during training pair construction.

---

## Solution Stages

### Stage 1 — Data Exploration & EDA ✅ *complete*
**Goal**: Understand data distributions, noise patterns, and baseline statistics.

- [x] Load small samples of all 6 TSV files
- [x] Analyze ground truth: distribution of match counts (0, 1, 2+ matches per S1 entity), singleton ratio
- [x] Profile noise: name length variance, address token overlap between matched pairs, country distribution (US / India)
- [x] Compute naive baseline: if every S1 maps to nothing → what is the F₀.₅ score?
- [x] Identify character encoding issues and any rows with missing fields
- **Output**: ✅ `eda_stage1.py` + `eda_summary.md`; key findings documented

> **Key EDA Findings:**
> - **GT format**: `matched_entity_ids` is a comma-separated string of S2 **and** S3 IDs (e.g. `S2-123,S2-456,S3-789`). Existing GT parsing in `train_and_match.py` is broken — must fix.
> - **No singletons** in train GT — every S1 entity has ≥1 match; mean 3.67 matches (1.67 S2 + 1.79 S3).
> - **Country = perfect blocking key** — 100% of matched pairs share country. Only US and India.
> - **~16.5% of true matched pairs have Name Jaccard < 0.2** — these are Hindi/Tamil script names in S2/S3 vs ASCII in S1. TF-IDF will fail for them.
> - **S2 15% non-ASCII, S3 12% non-ASCII, S1 0% non-ASCII** — cross-script matching is a key challenge.
> - **Most matches (57%) have Name Jaccard > 0.5** — TF-IDF signal is strong for the majority.

---

### Stage 2 — Fix Existing Code & Infrastructure ✅ *complete*
**Goal**: Make the existing codebase actually runnable end-to-end.

- [x] **Fix Issue 1**: `train_model()` added — fits XGBoost on ground truth labels
- [x] **Fix Issue 2**: Column names corrected in `compute_pairwise_features()`
- [x] **Fix Issue 3**: Output column names standardized in `generate_Candidates.py`
- [x] **Fix Issue 5**: Duplicate imports removed; `entity_id` excluded from TF-IDF text
- [x] Git repository initialized in `student_resource/`; `.gitignore` excludes `dataset/` and `venv/`; initial commit on `main`
- [x] **Fix Issue 4**: `evaluate.py` added — 80/20 GT split, exact leaderboard F₀.₅ formula, singleton-aware
- [x] `requirements.txt` added — pinned from venv; placed in `code/business_entity_resolution/`
- [x] `run_all.py` skeleton added — one command chains blocking → train → predict → validate with `--mode`, `--threshold`, `--top-k` flags
- [x] `generate_Candidates.py` CLI added — `--s1/--s2/--s3/--output/--top-k/--threshold/--batch-size` so `run_all.py` can invoke it
- [x] `README.md` replaced — developer guide with setup steps, data placement, run commands, branch strategy, submission checklist
- [x] All changes committed on `stage-2-infra` branch (ready to push when GitHub repo is created)
- **Output**: ✅ Working baseline pipeline runs end-to-end; teammates can clone, install deps, and run `run_all.py`
- [x] `stage-3-blocking` branch created from `main`; Stage 3 code (`normalize.py`, `blocking.py`, `blocking_recall.py`) added and committed

---

### Stage 3 — Improved Blocking / Candidate Generation 🔄 *in progress (branch: stage-3-blocking)*
**Goal**: Maximize recall ceiling (every true match must be a candidate).

Current approach: character n-gram TF-IDF + cosine similarity, top-15 per S1 entity.

Improvements:
- [x] **`normalize.py`**: ASCII transliteration, entity suffix expansion (Inc→incorporated, Pvt→private…), address abbreviation expansion (St→street, Rd→road…), word abbreviation expansion (&→and, tech→technology…), `first_name_token()`, `extract_numeric_tokens()`
- [x] **Multi-strategy blocking** (`blocking.py`): 4 strategies unioned per country:
  - [x] **Strategy 1** — Word (1,2)-gram TF-IDF on abbreviation-expanded name+address+country
  - [x] **Strategy 2** — Char (3,5)-gram TF-IDF on raw text (catches transliterations / Hindi/Tamil script similarity)
  - [x] **Strategy 3** — First-name-token exact-match inverted index
  - [x] **Strategy 4** — Numeric address-token index (ZIP/PIN/house numbers)
- [x] **Country-aware filtering**: blocking done per country block (100% of GT pairs share country)
- [x] **`generate_Candidates.py`** updated to delegate to `blocking.py`; default `top_k` raised 15→**50**, threshold lowered 0.10→**0.05**
- [x] **`blocking_recall.py`**: measures blocking recall against training GT; targets ≥ 90%
- [ ] **Run `blocking_recall.py`** on training data and record recall %, tune if below 90%
- [ ] Target: ≥ **top-50** candidates per S1, recall ceiling > 90%
- **Output**: `candidate_pairs.tsv` with high recall

---

### Stage 4 — Feature Engineering
**Goal**: Rich pairwise similarity features for the classifier.

Fix column names (Issue 2) and add:

**Name features**
- [ ] `fuzz.ratio`, `fuzz.partial_ratio`, `fuzz.token_sort_ratio`, `fuzz.token_set_ratio`
- [ ] Jaro-Winkler similarity
- [ ] Normalized character n-gram Jaccard
- [ ] Abbreviation-expanded exact match

**Address features**
- [ ] Token overlap Jaccard (bag-of-words)
- [ ] `fuzz.token_set_ratio` on address
- [ ] Shared numeric token match (house/building number)
- [ ] City/state exact match (after parsing address tokens)
- [ ] PIN/ZIP code exact match

**Cross-field / meta features**
- [ ] Country exact match flag
- [ ] Name-in-address overlap (some records embed business name in address)
- [ ] Length ratio for name / address fields

**Output**: Feature matrix `X` with ~20 features per candidate pair

---

### Stage 5 — Classifier Training
**Goal**: Train a precision-optimized binary classifier on labeled pairs.

- [ ] **Fix Issue 6**: `train_model()` loads all three source files (~1.2 GB) into memory as dicts simultaneously — add chunked reading or reservoir-sample negatives to stay within memory bounds
- [ ] **Label generation**: For each S1 entity in training set, generate positive pairs from `train_ground_truth.tsv` and sample negative pairs from non-matching candidates (hard negatives preferred)
- [ ] **Negative sampling ratio**: ~10:1 negatives per positive (adjust based on class balance)
- [ ] **Model**: XGBoost (already imported) — tune `scale_pos_weight`, `max_depth`, `learning_rate`
- [ ] **Threshold calibration**: Sweep `probability_threshold` on validation set using `evaluate.py`, maximize F₀.₅
- [ ] **Alternatives to explore**: LightGBM, Random Forest, simple logistic regression as baseline
- [ ] Save trained model to disk (`er_model.pkl`) for reproducible inference
- **Output**: Trained model + optimal threshold

---

### Stage 6 — Inference & Output Generation
**Goal**: Generate valid, submission-ready output files.

- [ ] Run blocking on **test** sources → `candidate_pairs.tsv`
- [ ] Extract features for all candidate pairs
- [ ] Score with trained model; apply calibrated threshold
- [ ] Build `matching_results.tsv` — every test S1 entity must have a row (singletons get empty `matched_entity_ids`)
- [ ] Run `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
- [ ] Verify exit code 0 (PASS) before submitting
- **Output**: `output/matching_results.tsv` + `output/candidate_pairs.tsv`

---

### Stage 7 — Iteration & Optimization
**Goal**: Improve F₀.₅ score iteratively.

- [ ] Analyze error cases: which S1 entities are hardest to match? Are they singletons or multi-match?
- [ ] False positives analysis: what features are most predictive of false merges?
- [ ] Try **embedding-based matching**: sentence-transformers (`all-MiniLM-L6-v2`, 80M params — within 8B limit) for semantic name similarity
- [ ] Ensemble: combine TF-IDF blocking + embedding re-ranking
- [ ] Experiment with different negative sampling strategies
- [ ] Track experiments with a simple CSV log

---

### Stage 8 — Packaging & Submission
**Goal**: Produce the complete submission zip.

- [ ] Final `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- [ ] Clean `code/business_entity_resolution/src/` with all source files
- [ ] `code/business_entity_resolution/README.md` with end-to-end reproduction steps
- [ ] **Finalize `requirements.txt`** — re-pin all versions after Stage 7 additions (e.g. sentence-transformers, lightgbm if used)
- [ ] **Finalize `run_all.py`** — complete end-to-end script wiring all stages; anyone should be able to run it on fresh data and reproduce both output files
- [ ] Fill in `Documentation_template.md`
- [ ] Create `<teamname>_submission.zip` per required structure
- [ ] Upload `matching_results.tsv` to Portal leaderboard

---

## File Structure (Target)

```
student_resource/
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── normalize.py          # text cleaning + abbreviation expansion
│   │   ├── blocking.py           # multi-strategy candidate generation
│   │   ├── features.py           # pairwise feature extraction
│   │   ├── train.py              # label generation + model training
│   │   ├── predict.py            # inference + output generation
│   │   └── evaluate.py           # local F₀.₅ scoring
│   ├── README.md
│   └── requirements.txt
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
└── Documentation_template.md    # filled in
```

---

## Priority Order

1. ~~🔴 Fix Issues 1, 2, 3~~ ✅ Done
2. ~~🔴 Fix Issue 5 (duplicate imports)~~ ✅ Done
3. ~~🔧 Initialize git & commit~~ ✅ Done
4. ~~🛠️ Stage 2: `evaluate.py`, `requirements.txt`, `run_all.py`, developer README~~ ✅ Done
5. ~~**Stage 1**: EDA on training data sample (distributions, noise, singleton ratio, baseline F₀.₅)~~ ✅ Done
6. **Next → Stage 3**: Improve blocking recall — **fix GT parsing bug first** (matched_entity_ids is comma-sep S2+S3)
7. 🤖 Stage 4–5: Richer features + real model training + threshold calibration via `evaluate.py`
8. 📤 Stage 6: Generate valid test outputs and run `validate_submission.py`
9. 🔄 Stage 7: Iterate — error analysis, embeddings, ensemble
10. 📦 Stage 8: Package zip, finalize `requirements.txt`, `run_all.py`, `Documentation_template.md`
