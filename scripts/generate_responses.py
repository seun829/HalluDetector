#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import pandas as pd
import re
import csv

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
    # Hard fail on mismatched counts rather than letting zip() silently drop items
    if len(prompt_files) != len(response_files):
        raise ValueError("prompt_files and response_files must have the same length.")

    for pth_in, pth_out in zip(prompt_files, response_files):
        logging.info(f"Reading prompts from {pth_in}")

        # Existence + non-empty check, with clear logs
        if not os.path.exists(pth_in):
            logging.error(f"Input file not found: {pth_in}")
            continue
        if os.path.getsize(pth_in) == 0:
            logging.warning(f"Skipping empty file: {pth_in}")
            continue

        # Optionally skip legacy file named 'prompts.csv' so only auto-generated file is used
        basename = os.path.basename(pth_in).lower()
        if basename == "prompts.csv":
            logging.info(f"Skipping legacy prompts file: {pth_in}")
            continue

        # Try to sniff delimiter to handle CSV/TSV/; or | without new helpers
        sep = ","
        try:
            with open(pth_in, 'r', encoding='utf-8', errors='replace') as f:
                sample = f.read(4096)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
                sep = dialect.delimiter
            except csv.Error:
                sep = ","
        except Exception as e:
            logging.warning(f"Failed to peek {pth_in} for delimiter ({e}); defaulting to comma.")

        # Robust read with clear error handling
        try:
            df_prompts = pd.read_csv(pth_in, sep=sep)
        except pd.errors.EmptyDataError:
            logging.warning(f"Skipping file with no columns/data: {pth_in}")
            continue
        except Exception as e:
            logging.error(f"Failed to read {pth_in}: {e}")
            continue

        if df_prompts is None or df_prompts.shape[1] == 0:
            logging.warning(f"Skipping file that parsed to zero columns: {pth_in}")
            continue
        if df_prompts.empty:
            logging.warning(f"Skipping empty dataframe from: {pth_in}")
            continue

        # Validate required columns
        required_cols = {"id", "prompt", "correct_answer"}
        missing = required_cols - set(df_prompts.columns)
        if missing:
            logging.error(f"{pth_in} is missing required column(s): {sorted(missing)}")
            continue

        # Normalize and filter rows
        df_prompts["prompt"] = df_prompts["prompt"].fillna("").astype(str)
        df_prompts["correct_answer"] = df_prompts["correct_answer"].fillna("").astype(str)

        n_before = len(df_prompts)
        df_prompts = df_prompts[df_prompts["prompt"].str.strip() != ""].copy()
        if df_prompts.empty:
            logging.warning(f"Skipping {pth_in}: all prompts are empty.")
            continue
        if len(df_prompts) < n_before:
            logging.info(f"Filtered {n_before - len(df_prompts)} empty-prompt rows in {pth_in}")

        # Warn on duplicate ids (can cause many-to-many merges)
        dups = df_prompts["id"].duplicated(keep=False)
        if dups.any():
            bad = df_prompts.loc[dups, "id"].unique().tolist()
            logging.warning(f"{pth_in} contains duplicate ids: {bad}")

        # Wrap prompts
        wrapped = []
        for idx, prompt, corr in df_prompts[['id','prompt','correct_answer']].itertuples(index=False):
            instruction = (
                "You are an expert quiz assistant. "
                "Answer the following question with a single concise statement. "
                "If your answer is a number, please write it as a numeral (e.g 3 instead of three)."
                "Do not repeat the question or add any commentary.\n"
            )
            full = f"{instruction}Question: {prompt}\nAnswer:"
            wrapped.append((idx, full, None))

        if not wrapped:
            logging.warning(f"No prompts to generate for {pth_in}")
            continue

        backend = "OpenAI" if use_openai else "HF"
        logging.info(f"Generating {len(wrapped)} responses via {backend}::{model_name}")

        try:
            if use_openai:
                results = simple_generate_openai(wrapped, model_name=model_name)
            else:
                results = simple_generate_hf(wrapped, model_name=model_name)
        except Exception as e:
            logging.error(f"Generation failed for {pth_in}: {e}")
            continue

        if not results:
            logging.error(f"No results generated for {pth_in}")
            continue

        df_out = pd.DataFrame(results, columns=['id','wrapped_prompt','model_response_raw'])

        # Merge back on id; validate to catch accidental one-to-many joins
        try:
            df_out = df_out.merge(
                df_prompts[["question_type", "template", "id", "prompt", "correct_answer"]],
                on='id', how='left', validate='many_to_one'
            )
        except Exception as e:
            logging.error(f"Merge failed for {pth_in}: {e}")
            continue

        # Clean raw outputs
        df_out['model_response'] = df_out.apply(
            lambda r: clean_response(r['model_response_raw'], r['prompt']),
            axis=1
        )

        # Label hallucinations (inputs sanitized above)
        df_out['hallucinated'] = df_out.apply(
            lambda r: is_hallucinated((r['model_response'] or "").strip(),
                                      (r['correct_answer'] or "").strip()),
            axis=1
        )

        # Write outputs
        try:
            os.makedirs(os.path.dirname(pth_out), exist_ok=True)
            df_out[[ 'id','prompt','model_response','correct_answer', 'hallucinated']].to_csv(pth_out, index=False, encoding='utf-8')
            logging.info(f"Wrote {len(df_out)} rows to {pth_out}")
        except Exception as e:
            logging.error(f"Failed to write {pth_out}: {e}")
            continue

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
        help="Model name for HF or OpenAI (e.g., gpt-4o, gpt-4-mini, roberta-large-mnli)"
    )
    parser.add_argument(
        "--use-openai", action="store_true",
        help="Force use of OpenAI API instead of HuggingFace"
    )
    args = parser.parse_args()

    # Remap legacy model identifier "gpt-4-32k" to "gpt-4o"
    if args.model.lower() == "gpt-4-32k":
        logging.info("Replacing model 'gpt-4-32k' with 'gpt-4o'")
        args.model = "gpt-4o"

    # Instead of using a fixed set, allow any model name starting with "gpt-"
    use_openai = args.use_openai or args.model.lower().startswith("gpt-")
    logging.info(f"Using OpenAI backend: {use_openai} (model: {args.model})")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        response_files = [
            os.path.join(
                args.output_dir,
                os.path.basename(pf).replace("prompts", "responses_labeled")
            )
            for pf in args.prompt_files
        ]
        logging.info(f"Outputs will be written under: {args.output_dir}")
    else:
        response_files = args.response_files or []
        if len(args.prompt_files) != len(response_files):
            parser.error("When using --response-files, the count must match --prompt-files.")

    process_files(
        prompt_files=args.prompt_files,
        response_files=response_files,
        model_name=args.model,
        use_openai=use_openai
    )

if __name__ == "__main__":
    main()