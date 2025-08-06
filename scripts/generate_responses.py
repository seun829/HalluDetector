#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import pandas as pd
import re

# — Fix import path for hallu_detector —
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SRC_DIR      = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.hallu_detector.generate import simple_generate_hf, simple_generate_openai
from src.hallu_detector.detect  import is_hallucinated

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)

def clean_response(raw: str, prompt: str) -> str:
    """
    Extract the first non-empty line after 'Answer:' (case-insensitive).
    Fallback to the first non-empty line if no 'Answer:' marker.
    Then strip any echoed prompt text.
    """
    text = raw or ""
    match = re.search(r'Answer:\s*(.*)', text, flags=re.IGNORECASE|re.DOTALL)
    if match:
        after = match.group(1).strip()
        lines = [ln.strip() for ln in after.splitlines() if ln.strip()]
        text = lines[0] if lines else after
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = lines[0] if lines else text.strip()
    if prompt and text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text

def process_files(prompt_files, response_files, model_name, use_openai):
    for pth_in, pth_out in zip(prompt_files, response_files):
        logging.info(f"Reading prompts from {pth_in}")
        df_prompts = pd.read_csv(pth_in)
        if df_prompts.empty:
            logging.warning(f"Skipping empty file: {pth_in}")
            continue

        wrapped = []
        for idx, prompt, corr in df_prompts[['id','prompt','correct_answer']].itertuples(index=False):
            instruction = (
                "You are an expert quiz assistant. "
                "Answer the following question with a single concise statement. "
                "Do not repeat the question or add any commentary.\n"
            )
            full = f"{instruction}Question: {prompt}\nAnswer:"
            wrapped.append((idx, full, None))

        backend = "OpenAI" if use_openai else "HF"
        logging.info(f"Generating {len(wrapped)} responses via {backend}::{model_name}")

        if use_openai:
            results = simple_generate_openai(wrapped, model_name=model_name)
        else:
            results = simple_generate_hf(wrapped, model_name=model_name)

        df_out = pd.DataFrame(results, columns=['id','wrapped_prompt','model_response_raw'])
        df_out = df_out.merge(
            df_prompts[['id','prompt','correct_answer']],
            on='id', how='left'
        )

        # Clean raw outputs
        df_out['model_response'] = df_out.apply(
            lambda r: clean_response(r['model_response_raw'], r['prompt']),
            axis=1
        )

        # Label hallucinations
        df_out['hallucinated'] = df_out.apply(
            lambda r: is_hallucinated(r['model_response'], r['correct_answer']),
            axis=1
        )

        # --- <== NEW: duplicate for evaluate.py ---
        df_out['label'] = df_out['hallucinated']

        os.makedirs(os.path.dirname(pth_out), exist_ok=True)
        df_out[[
            'id','prompt','model_response','correct_answer',
            'hallucinated','label'
        ]].to_csv(pth_out, index=False, encoding='utf-8')
        logging.info(f"Wrote {len(df_out)} rows to {pth_out}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate model responses, clean them, and label hallucinations."
    )
    parser.add_argument(
        "--prompt-files", "-i", nargs="+", required=True,
        help="Input CSV(s) with columns: id,prompt,correct_answer"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--response-files", "-o", nargs="+",
        help="Exact output CSV path(s), matching the count of prompt-files"
    )
    group.add_argument(
        "--output-dir",
        help="Directory to write labeled response CSVs (filenames derived)"
    )
    parser.add_argument(
        "--model", "-m", required=True,
        help="Model name for HF or OpenAI (e.g., gpt-3.5-turbo, gpt-4)"
    )
    parser.add_argument(
        "--use-openai", action="store_true",
        help="Force use of OpenAI API instead of HuggingFace"
    )
    args = parser.parse_args()

    openai_models = {"gpt-3.5-turbo","gpt-4","gpt-4-32k"}
    use_openai    = args.use_openai or (args.model in openai_models)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        response_files = [
            os.path.join(
                args.output_dir,
                os.path.basename(pf).replace("prompts","responses_labeled")
            )
            for pf in args.prompt_files
        ]
    else:
        response_files = args.response_files

    process_files(
        prompt_files=args.prompt_files,
        response_files=response_files,
        model_name=args.model,
        use_openai=use_openai
    )

if __name__ == "__main__":
    main()
