import argparse
import json
import logging
import pandas as pd

"""
Script to compute hallucination detection metrics from labeled responses.
"""

def compute_metrics(input_csv: str) -> dict:
    """
    Reads a CSV with a 'label' column ('hallucinated' or 'correct') and computes:
    - total responses
    - number hallucinated
    - number correct
    - hallucination rate
    """
    df = pd.read_csv(input_csv)
    total = len(df)
    hallucinated = int((df['label'] == 'hallucinated').sum())
    correct = total - hallucinated
    rate = hallucinated / total if total > 0 else 0.0

    return {
        'total_responses': total,
        'hallucinated': hallucinated,
        'correct': correct,
        'hallucination_rate': rate
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute and save hallucination detection metrics."
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help="Path to labeled CSV with a 'label' column."
    )
    parser.add_argument(
        '--output', '-o', required=True,
        help="Path to output JSON file for metrics."
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    metrics = compute_metrics(args.input)

    # Write metrics to JSON
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Saved metrics to {args.output}")
    # Also print to stdout
    print(json.dumps(metrics, indent=4))


if __name__ == '__main__':
    main()
