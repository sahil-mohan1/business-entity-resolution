"""
run_all.py — End-to-end pipeline orchestration for Business Entity Resolution.

Chains: blocking → training → inference → validation

Usage (from student_resource/ with venv active):
    python code/business_entity_resolution/run_all.py
    python code/business_entity_resolution/run_all.py --skip-blocking
    python code/business_entity_resolution/run_all.py --mode train-only
    python code/business_entity_resolution/run_all.py --threshold 0.6 --neg-ratio 10
"""

import argparse
import os
import subprocess
import sys

# Resolve paths relative to student_resource/ regardless of where this is called from
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # student_resource/
SRC_DIR     = os.path.join(REPO_ROOT, "code", "business_entity_resolution", "src")
UTILS_DIR   = os.path.join(REPO_ROOT, "utils")

TRAIN_DIR   = os.path.join(REPO_ROOT, "dataset", "train")
TEST_DIR    = os.path.join(REPO_ROOT, "dataset", "test")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "output")

CANDIDATE_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCHING_TSV  = os.path.join(OUTPUT_DIR, "matching_results.tsv")
MODEL_PATH    = os.path.join(OUTPUT_DIR, "er_model.pkl")


def run(cmd: list[str], step: str):
    """Run a subprocess command, printing a clear step header. Exits on failure."""
    print(f"\n{'='*60}")
    print(f"  STEP: {step}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"\n❌  Step '{step}' failed (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)
    print(f"✅  {step} complete.")


def main():
    parser = argparse.ArgumentParser(description="Run the full entity resolution pipeline.")
    parser.add_argument(
        "--mode",
        choices=["all", "train-only", "predict-only"],
        default="all",
        help=(
            "all          : blocking + train + predict + validate (default)\n"
            "train-only   : blocking + train (no inference)\n"
            "predict-only : inference + validate (model must already exist)"
        ),
    )
    parser.add_argument("--skip-blocking", action="store_true",
                        help="Skip blocking step (reuse existing candidate_pairs.tsv)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Match probability threshold for inference (default: 0.5)")
    parser.add_argument("--neg-ratio", type=int, default=5,
                        help="Negative samples per positive pair during training (default: 5)")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Top-k candidates per S1 entity from blocking (default: 15)")
    args = parser.parse_args()

    python = sys.executable  # Use the same Python that launched this script

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Blocking — generate candidate pairs
    # ------------------------------------------------------------------
    if not args.skip_blocking and args.mode in ("all", "train-only"):
        run(
            [python, os.path.join(SRC_DIR, "generate_Candidates.py"),
             "--top-k", str(args.top_k)],
            step="Blocking / Candidate Generation (test set)",
        )

    # ------------------------------------------------------------------
    # Step 2: Train model on training data
    # ------------------------------------------------------------------
    if args.mode in ("all", "train-only"):
        run(
            [python, os.path.join(SRC_DIR, "train_and_match.py"),
             "--mode", "train",
             "--train-dir", TRAIN_DIR,
             "--model-path", MODEL_PATH,
             "--neg-ratio", str(args.neg_ratio)],
            step="Model Training (on ground truth)",
        )

    # ------------------------------------------------------------------
    # Step 3: Inference — score test candidate pairs
    # ------------------------------------------------------------------
    if args.mode in ("all", "predict-only"):
        run(
            [python, os.path.join(SRC_DIR, "train_and_match.py"),
             "--mode", "predict",
             "--candidate-tsv", CANDIDATE_TSV,
             "--test-dir", TEST_DIR,
             "--output-tsv", MATCHING_TSV,
             "--model-path", MODEL_PATH,
             "--threshold", str(args.threshold)],
            step="Inference / Matching",
        )

    # ------------------------------------------------------------------
    # Step 4: Validate output files
    # ------------------------------------------------------------------
    if args.mode in ("all", "predict-only"):
        run(
            [python, os.path.join(UTILS_DIR, "validate_submission.py"),
             "--matching", MATCHING_TSV,
             "--candidate", CANDIDATE_TSV,
             "--test-dir", TEST_DIR],
            step="Output Validation",
        )

    print("\n🎉  Pipeline finished successfully.")
    print(f"   Matching results : {MATCHING_TSV}")
    print(f"   Candidate pairs  : {CANDIDATE_TSV}")
    print("\nNext: upload matching_results.tsv to the Portal leaderboard.")


if __name__ == "__main__":
    main()
