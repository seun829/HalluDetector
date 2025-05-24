"""
Script to detect hallucinations in model responses.
"""
import argparse
import logging
import pandas as pd
from tqdm import tqdm

def is_hallucinated(answer, correct_answer):
    """
    Check if the model's answer matches the correct answer (roughly).
    Ignores case and extra punctuation.
    """
    if not answer and correct_answer:
        return True  # empty answer = hallucination

    if not correct_answer:
        return False  # no ground truth = can't call it hallucinated

    answer = answer.strip().lower()
    correct_answer = correct_answer.strip().lower()

    return correct_answer not in answer

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO
    )


def main():
    parser = argparse.ArgumentParser(
        description="Detect hallucinations in model responses"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input CSV with columns id,prompt,model_response,correct_answer"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output CSV path for labeled results"
    )
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Loading responses from {args.input}...")
    df = pd.read_csv(args.input)

    if 'correct_answer' not in df.columns:
        logging.error("Missing 'correct_answer' column in input CSV.")
        return

    logging.info("Detecting hallucinations...")
    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting"):
        label = 'hallucinated' if is_hallucinated(
            row['model_response'], row['correct_answer']
        ) else 'correct'
        labels.append(label)

    df['label'] = labels
    df.to_csv(args.output, index=False)
    logging.info(f"Saved {len(df)} labeled rows to {args.output}")


if __name__ == '__main__':
    main()


