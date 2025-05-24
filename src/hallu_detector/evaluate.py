# src/hallu_detector/evaluate.py
"""
Script to compute hallucination metrics from labeled data.
"""
import argparse
import logging
import json
import pandas as pd

def compute_metrics(labels):
    """
    Given a list of labels like ['correct', 'hallucinated', 'correct'],
    return a dictionary with hallucination rate.
    """
    total = len(labels)
    if total == 0:
        return {"hallucination_rate": None}
    
    hallu_count = sum(1 for label in labels if label == "hallucinated")
    return {"hallucination_rate": hallu_count / total}

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hallucination detection performance"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input CSV with 'label' column"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON metrics file"
    )
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Loading labeled data from {args.input}...")
    df = pd.read_csv(args.input)

    if 'label' not in df.columns:
        logging.error("Missing 'label' column in input CSV.")
        return

    logging.info("Computing metrics...")
    metrics = compute_metrics(df['label'].tolist())
    logging.info(f"Metrics: {metrics}")

    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved metrics to {args.output}")


if __name__ == '__main__':
    main()


