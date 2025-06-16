#!/usr/bin/env python
"""
Read multiple prompt CSVs, wrap each question in clear instruction prompts,
generate a single, concise response, clean it to extract only the actual answer line,
and merge with the correct answer.
"""
import sys
import os
import argparse
import logging
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from tqdm import tqdm

# add src/ to path so we can import our package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from hallu_detector.generate import simple_generate_hf, simple_generate_openai


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO
    )


def clean_response(resp: str, prompt: str) -> str:
    """
    Extract only the model's answer by:
    - Splitting into lines
    - Skipping instruction/prompt echo and 'Question:' labels
    - Stripping 'Answer:' prefix and returning its content
    - Returning the first valid content line otherwise
    """
    lines = str(resp).splitlines()
    for line in lines:
        text = line.strip()
        if not text:
            continue
        # skip instruction lines
        if text.startswith("You are an expert quiz assistant"):
            continue
        # skip the question echo
        if prompt and text.startswith(prompt):
            continue
        # skip question label
        if text.lower().startswith("question:"):
            continue
        # handle answer label
        if text.lower().startswith("answer:"):
            # strip 'Answer:' prefix
            ans = text[len(text.split(":",1)[0])+1:].strip()
            if ans:
                return ans
            else:
                continue
        # first valid line is the answer
        return text
    return ""


def process_files(prompt_files, response_files, model, use_openai):
    any_processed = False
    for pth_in, pth_out in zip(prompt_files, response_files):
        logging.info(f"Reading prompts from {pth_in}")
        try:
            df_prompts = pd.read_csv(pth_in)
        except (EmptyDataError, ParserError) as e:
            logging.warning(f"Skipping {pth_in}: cannot parse CSV ({e})")
            continue
        if df_prompts.empty:
            logging.warning(f"Skipping {pth_in}: file is empty.")
            continue
        required = {'id', 'prompt', 'correct_answer'}
        if not required.issubset(df_prompts.columns):
            missing = required - set(df_prompts.columns)
            logging.error(f"Skipping {pth_in}: missing columns {missing}.")
            continue

        any_processed = True
        # Build instruction-wrapped prompts (id, wrapped_prompt, placeholder)
        wrapped = []
        for idx, prompt, _ in df_prompts[['id','prompt','correct_answer']].itertuples(index=False, name=None):
            instruction = (
                "You are an expert quiz assistant. "
                "Answer the following question with a single concise statement. "
                "Do not repeat the question or add any commentary.\n"
            )
            full = f"{instruction}Question: {prompt}\nAnswer:"
            wrapped.append((idx, full, None))

        backend = 'OpenAI' if use_openai else 'HF'
        logging.info(f"Generating {len(wrapped)} responses with {backend}::{model}")

        # Call the generation function
        if use_openai:
            results = simple_generate_openai(wrapped, model_name=model)
        else:
            results = simple_generate_hf(wrapped, model_name=model)

        # Build output DataFrame
        df_out = pd.DataFrame(results, columns=['id','wrapped_prompt','model_response_raw'])
        # Clean each response to extract only the answer
        df_out['model_response'] = df_out.apply(
            lambda row: clean_response(row['model_response_raw'], df_prompts.loc[df_prompts['id']==row['id'], 'prompt'].values[0]),
            axis=1
        )
        # Restore original prompt and merge correct_answer
        df_out = df_out.merge(df_prompts[['id','prompt','correct_answer']], on='id', how='left')

        # Write out only the cleaned answer
        os.makedirs(os.path.dirname(pth_out), exist_ok=True)
        df_out[['id','prompt','model_response','correct_answer']].to_csv(pth_out, index=False, encoding='utf-8')
        logging.info(f"Wrote {len(df_out)} responses to {pth_out}")

    if not any_processed:
        logging.error("No prompt files were processed.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate single-line, concise model responses for prompt CSVs"
    )
    parser.add_argument(
        "--prompt-files", "-i", nargs='+', required=True,
        help="Input prompt CSV(s) (must have id,prompt,correct_answer)"
    )
    parser.add_argument(
        "--response-files", "-o", nargs='+', required=True,
        help="Output response CSV(s)"
    )
    parser.add_argument(
        "--model", "-m", default="gpt-3.5-turbo",
        help="Model name for HF or OpenAI"
    )
    parser.add_argument(
        "--use-openai", action="store_true",
        help="Use OpenAI API instead of HuggingFace"
    )
    args = parser.parse_args()

    setup_logging()
    if len(args.prompt_files) != len(args.response_files):
        logging.error("Number of prompt-files and response-files must match.")
        sys.exit(1)

    process_files(
        prompt_files=args.prompt_files,
        response_files=args.response_files,
        model=args.model,
        use_openai=args.use_openai
    )

if __name__ == '__main__':
    main()
