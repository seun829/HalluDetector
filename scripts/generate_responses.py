"""
Read prompts CSV, generate model responses (HF or OpenAI) and save to CSV.
"""
import argparse
import os
import logging
import pandas as pd
from tqdm import tqdm
from hallu_detector.generate import simple_generate_hf, simple_generate_openai

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO
    )


def batch_generate(
    df: pd.DataFrame, model: str, use_openai: bool
) -> pd.DataFrame:
    prompt_list = list(df[['id', 'prompt', 'correct_answer']].itertuples(index=False, name=None))
    if use_openai:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logging.error("OPENAI_API_KEY not set. Exiting.")
            raise EnvironmentError("Missing OpenAI API key.")
        gen_fn = simple_generate_openai
    else:
        gen_fn = simple_generate_hf

    results = []
    for pid, prompt, correct_answer in tqdm(prompt_list, desc="Generating responses"):
        try:
            out = gen_fn([(pid, prompt)], model_name=model)[0]
            is_correct = correct_answer.lower() in out[2].lower()  # Simple correctness check
            results.append((pid, prompt, out[2], correct_answer, is_correct))
        except Exception as e:
            logging.warning(f"Failed to generate for prompt {pid}: {e}")
            results.append((pid, prompt, "", correct_answer, False))
    return pd.DataFrame(results, columns=["id", "prompt", "model_response", "correct_answer", "is_correct"])


def main():
    parser = argparse.ArgumentParser(
        description="Generate model responses from prompts"
    )
    parser.add_argument("in_csv", help="Input prompts CSV (id,prompt)")
    parser.add_argument("out_csv", help="Output responses CSV")
    parser.add_argument(
        "--model", default="gpt2", help="Model name (HF or OpenAI)"
    )
    parser.add_argument(
        "--use-openai", action='store_true', help="Toggle OpenAI API"
    )
    args = parser.parse_args()

    setup_logging()
    logging.info("Loading prompts...")
    df = pd.read_csv(args.in_csv)

    logging.info(f"Generating with model={args.model}, openai={args.use_openai}")
    out_df = batch_generate(df, args.model, args.use_openai)
    out_df.to_csv(args.out_csv, index=False)
    logging.info(f"Saved {len(out_df)} responses to {args.out_csv}")


if __name__ == '__main__':
    main()