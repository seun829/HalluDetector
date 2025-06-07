"""
Read multiple prompts CSV files, generate model responses (via HuggingFace or OpenAI), and save to corresponding CSV files.
"""
import sys
import os
import argparse
import logging
import pandas as pd
from tqdm import tqdm

# Add the `src` folder to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from hallu_detector.generate import simple_generate_hf, simple_generate_openai

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO
    )

def batch_generate(df: pd.DataFrame, model: str, use_openai: bool) -> pd.DataFrame:
    # Build a list of tuples: (id, prompt, correct_answer)
    prompt_list = list(df[['id', 'prompt', 'correct_answer']].itertuples(index=False, name=None))
    
    if use_openai:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logging.error("OPENAI_API_KEY not set. Exiting.")
            raise EnvironmentError("Missing OpenAI API key.")
        gen_fn = simple_generate_openai
    else:
        gen_fn = simple_generate_hf

    # Batch call: send all prompts at once
    try:
        responses = gen_fn(prompt_list, model_name=model)
    except Exception as e:
        logging.error(f"Error during batch generation: {e}")
        raise

    results = []
    # Assume that each response is a tuple (id, prompt, answer)
    for (pid, prompt, correct_answer), resp in zip(prompt_list, responses):
        # resp[2] is assumed to be the generated answer
        results.append((pid, prompt, resp[2], correct_answer))
    return pd.DataFrame(results, columns=["id", "prompt", "model_response", "correct_answer"])

def process_files(prompt_files, response_files, model, use_openai):
    """
    Process each prompt file and write the corresponding responses.
    """
    for prompt_file, response_file in zip(prompt_files, response_files):
        logging.info(f"Processing {prompt_file}...")
        if not os.path.exists(prompt_file):
            logging.error(f"File {prompt_file} does not exist. Skipping.")
            continue

        # Check file is not empty
        if os.stat(prompt_file).st_size == 0:
            logging.error(f"File {prompt_file} is empty. Skipping.")
            continue

        try:
            df = pd.read_csv(prompt_file)
        except pd.errors.EmptyDataError:
            logging.error(f"File {prompt_file} contains no data. Skipping.")
            continue

        if df.empty:
            logging.warning(f"{prompt_file} is empty after reading. Skipping.")
            continue

        df_responses = batch_generate(df, model=model, use_openai=use_openai)
        df_responses.to_csv(response_file, index=False)
        logging.info(f"Responses saved to {response_file}.")

def main():
    parser = argparse.ArgumentParser(description="Generate model responses from multiple prompt files.")
    parser.add_argument("--prompt-files", nargs='+', required=True, help="List of input prompt CSV files.")
    parser.add_argument("--response-files", nargs='+', required=True, help="List of output response CSV files.")
    parser.add_argument("--model", default="gpt2", help="Model name (HF or OpenAI).")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI API for generation.")
    args = parser.parse_args()

    setup_logging()
    if len(args.prompt_files) != len(args.response_files):
        logging.error("Number of prompt files and response files must match.")
        return

    process_files(args.prompt_files, args.response_files, model=args.model, use_openai=args.use_openai)

if __name__ == "__main__":
    main()