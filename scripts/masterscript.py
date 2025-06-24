#!/usr/bin/env python3
"""
Master script to run the entire hallucination detection pipeline:
1. Generate prompts (easy, auto-generated, hard)
2. Generate model responses
3. Detect hallucinations (rule-based or ML-based)
4. Compute metrics
5. Graph patterns

This version will continue through all steps even if some files are missing or empty.
"""
import sys
import subprocess
import os
import logging

# Use the same Python interpreter
PYTHON = sys.executable

# Configuration
AUTO_PROMPTS_CONFIG = "config/prompts_config.yaml"
PROMPT_FILES = [
    "data/raw/prompts_easy.csv",
    "data/raw/prompts_auto-generated.csv",
    "data/raw/prompts_hard.csv",
]
RAW_RESPONSES = [
    "data/processed/responses_easy_raw.csv",
    "data/processed/responses_auto-generated_raw.csv",
    "data/processed/responses_hard_raw.csv",
]
LABELED_RESPONSES = [
    "data/processed/responses_easy_labeled.csv",
    "data/processed/responses_auto-generated_labeled.csv",
    "data/processed/responses_hard_labeled.csv",
]
METRICS_FILES = [
    "data/metrics/responses_easy_metrics.json",
    "data/metrics/responses_auto-generated_metrics.json",
    "data/metrics/responses_hard_metrics.json",
]
GRAPH_IMAGES = [
    "data/graphs/responses_easy_patterns.png",
    "data/graphs/responses_auto-generated_patterns.png",
    "data/graphs/responses_hard_patterns.png",
]

# Optional ML model paths
ML_MODEL = None
VECTORIZER = None
# To enable ML-based detection, uncomment and set paths below:
# ML_MODEL = "models/hallu_clf.joblib"
# VECTORIZER = "models/tfidf.joblib"


def run_step(description, command):
    logging.info(f"Running: {description}")
    try:
        subprocess.run(command, check=True)
        logging.info(f"{description} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        logging.warning(f"Error during {description}: {e}. Continuing to next step.")


def ensure_dirs():
    # Create all necessary output directories
    for path in RAW_RESPONSES + LABELED_RESPONSES + METRICS_FILES + GRAPH_IMAGES:
        dirpath = os.path.dirname(path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)


def main():
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    # Ensure output directories exist
    ensure_dirs()

    # Step 1: Generate auto prompts
    run_step(
        "Generate auto prompts",
        [PYTHON, "scripts/make_prompts.py",
         "--config", AUTO_PROMPTS_CONFIG,
         "--out-csv", PROMPT_FILES[1]]
    )

    # Step 2: Generate raw responses
    response_cmd = [PYTHON, "scripts/generate_responses.py", "--model", "gpt2"]
    response_cmd += ["--prompt-files"] + PROMPT_FILES
    response_cmd += ["--response-files"] + RAW_RESPONSES
    run_step("Generate model responses", response_cmd)

    # Step 3: Detect hallucinations
    for raw, labeled in zip(RAW_RESPONSES, LABELED_RESPONSES):
        if not os.path.isfile(raw):
            logging.warning(f"Raw response file missing: {raw}. Skipping detection for this file.")
            continue
        if os.path.getsize(raw) == 0:
            logging.warning(f"Raw response file empty: {raw}. Creating empty labeled file.")
            open(labeled, 'w').close()
            continue
        cmd = [PYTHON, "scripts/detect.py",  # call detect script directly
               "--input", raw,
               "--output", labeled]
        if ML_MODEL and VECTORIZER:
            cmd += ["--model", ML_MODEL, "--vectorizer", VECTORIZER]
        run_step(f"Detect hallucinations for {os.path.basename(raw)}", cmd)

    # Step 4: Compute metrics
    for labeled, metrics in zip(LABELED_RESPONSES, METRICS_FILES):
        if not os.path.isfile(labeled):
            logging.warning(f"Labeled file missing: {labeled}. Skipping metrics computation.")
            continue
        if os.path.getsize(labeled) == 0:
            logging.warning(f"Labeled file empty: {labeled}. Creating empty metrics file.")
            with open(metrics, 'w') as f:
                f.write('{}')
            continue
        run_step(
            f"Compute metrics for {os.path.basename(labeled)}",
            [PYTHON, "scripts/evaluate.py",  # call evaluate directly
             "--input", labeled,
             "--output", metrics]
        )

    # Step 5: Graph patterns
    for labeled, metrics, image in zip(LABELED_RESPONSES, METRICS_FILES, GRAPH_IMAGES):
        if not os.path.isfile(labeled) or not os.path.isfile(metrics):
            logging.warning(f"Missing data for graphing ({labeled}, {metrics}). Skipping graph generation.")
            continue
        run_step(
            f"Graph patterns for {os.path.basename(labeled)}",
            [PYTHON, "scripts/graph_patterns.py",  # call graph script directly
             "--input", labeled,
             "--metrics", metrics,
             "--output", image]
        )

    logging.info("Full pipeline finished (with skips for missing/empty files).")


if __name__ == "__main__":
    main()
