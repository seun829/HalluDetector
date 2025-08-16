#!/usr/bin/env python3
"""
Script to compute hallucination metrics from labeled response files.
"""
import argparse
import logging
import json
import pandas as pd

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def compute_metrics(labels):
    total = len(labels)
    if total == 0:
        return {"hallucination_rate": None}
    hallu_count = sum(1 for label in labels if label == True or label == "True")
    return {"hallucination_rate": hallu_count / total}

def process_files(response_files, metric_files):
    for response_file, metric_file in zip(response_files, metric_files):
        logging.info(f"Processing {response_file}...")
        try:
            df = pd.read_csv(response_file)
        except Exception as e:
            logging.error(f"Failed to read {response_file}: {e}")
            continue

        if "hallucinated" not in df.columns:
            logging.error(f"Missing 'hallucinated' column in {response_file}. Skipping.")
            continue

        metrics = compute_metrics(df["hallucinated"].tolist())
        with open(metric_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Metrics saved to {metric_file}.")

def main():
    parser = argparse.ArgumentParser(description="Compute hallucination metrics for response files.")
    parser.add_argument("--response-files", nargs='+', required=True, help="List of input response CSV files.")
    parser.add_argument("--metric-files", nargs='+', required=True, help="List of output JSON metric files.")
    args = parser.parse_args()

    if len(args.response_files) != len(args.metric_files):
        logging.error("Number of response files and metric files must match.")
        return

    process_files(args.response_files, args.metric_files)

if __name__ == "__main__":
    main()
