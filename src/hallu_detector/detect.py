import argparse
import logging
import pandas as pd
import string
from tqdm import tqdm

"""
Script to detect hallucinations in model responses.
"""

def normalize_text(text):
    """
    Normalize text for comparison: lowercase, remove punctuation, collapse whitespace.
    """
    if text is None:
        return ""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse multiple spaces
    text = ' '.join(text.split())
    return text


def is_hallucinated(answer, correct_answer):
    """
    Determine if the model's answer is a hallucination compared to the correct answer.
    Returns True if hallucinated (i.e., the answer does not match the correct answer).
    """
    ans = normalize_text(answer)
    corr = normalize_text(correct_answer)

    # If there is no ground truth, assume not hallucinated
    if not corr:
        return False
    # Empty answer when a correct answer exists => hallucination
    if not ans:
        return True
    # If the correct answer is contained in the answer, or vice versa, consider it correct
    if corr in ans or ans in corr:
        return False
    # Otherwise, mark as hallucinated
    return True


def main():
    parser = argparse.ArgumentParser(description="Label model responses as hallucinated or correct.")
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV with at least 'model_response' and 'correct_answer' columns."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output CSV where labeled responses will be saved."
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    df = pd.read_csv(args.input)
    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Labeling responses"):
        resp = row.get('model_response', '')
        truth = row.get('correct_answer', '')
        label = 'hallucinated' if is_hallucinated(resp, truth) else 'correct'
        labels.append(label)

    df['label'] = labels
    df.to_csv(args.output, index=False)
    logging.info(f"Saved {len(df)} labeled rows to {args.output}")


if __name__ == '__main__':
    main()
