#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import pandas as pd

# — Fix import path for hallu_detector —
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# Prefer src directory (if using src/ layout), then project root
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from hallu_detector.generate import simple_generate_hf, simple_generate_openai

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)


def clean_response(resp: str, prompt: str) -> str:
    lines = str(resp).splitlines()
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text.lower().startswith("question:") or text.lower().startswith("answer:"):
            text = text.split(":", 1)[1].strip()
        if text:
            return text
    return ""


def process_files(prompt_files, response_files, model, use_openai):
    for pth_in, pth_out in zip(prompt_files, response_files):
        logging.info(f"Reading prompts from {pth_in}")
        # Ensure the prompt file exists
        if not os.path.exists(pth_in):
            logging.error(f"Prompt file not found: {pth_in}")
            continue
        # Attempt to read the CSV, catching common errors
        try:
            df_prompts = pd.read_csv(pth_in)
        except pd.errors.EmptyDataError:
            logging.warning(f"No data in prompt file: {pth_in}")
            continue
        except Exception as e:
            logging.error(f"Failed to parse CSV {pth_in}: {e}")
            continue

        if df_prompts.empty:
            logging.warning(f"Skipping {pth_in}: empty DataFrame")
            continue

        # wrap prompts
        wrapped = []
        for idx, prompt, _ in df_prompts[["id", "prompt", "correct_answer"]].itertuples(index=False):
            instruction = (
                "You are an expert quiz assistant. "
                "Answer the following question with a single concise statement. "
                "Do not repeat the question or add any commentary.\n"
            )
            full = f"{instruction}Question: {prompt}\nAnswer:"
            wrapped.append((idx, full, None))

        backend = "OpenAI" if use_openai else "HF"
        logging.info(f"Generating {len(wrapped)} responses with {backend}::{model}")

        # delegate to the appropriate backend
        if use_openai:
            results = simple_generate_openai(wrapped, model_name=model)
        else:
            results = simple_generate_hf(wrapped, model_name=model)

        df_out = pd.DataFrame(results, columns=["id", "wrapped_prompt", "model_response_raw"])
        # label hallucination by comparing model_response to correct_answer
        df_out = df_out.merge(
            df_prompts[["id", "correct_answer"]], on="id", how="left"
        )
        from hallu_detector.detect import is_hallucinated
        df_out['hallucinated'] = df_out.apply(
            lambda row: is_hallucinated(row['model_response'], row['correct_answer']), axis=1
        )
        # now merge prompt and other fields
        df_out = df_out.merge(
            df_prompts[["id", "prompt"]], on="id", how="left"
        )
        df_out["model_response"] = df_out.apply(
            lambda row: clean_response(
                row["model_response_raw"],
                df_prompts.loc[df_prompts["id"] == row["id"], "prompt"].values[0]
            ),
            axis=1
        )
        df_out = df_out.merge(
            df_prompts[["id", "prompt", "correct_answer"]],
            on="id", how="left"
        )

        os.makedirs(os.path.dirname(pth_out), exist_ok=True)
        df_out[["id", "prompt", "model_response", "correct_answer"]] \
            .to_csv(pth_out, index=False, encoding="utf-8")
        logging.info(f"Wrote {len(df_out)} responses to {pth_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate concise model responses for prompt CSVs"
    )
    parser.add_argument(
        "--prompt-files", "-i", nargs="+", required=True,
        help="Input prompt CSV(s) (id,prompt,correct_answer)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--response-files", "-o", nargs="+",
        help="Explicit output CSV path(s), matching count of prompt-files"
    )
    group.add_argument(
        "--output-dir",
        help="Directory to write labeled response CSVs (filenames derived)"
    )
    parser.add_argument(
        "--model", "-m", default="gpt-3.5-turbo",
        help="Model name for HuggingFace or OpenAI (e.g., gpt-3.5-turbo, gpt-4)"
    )
    parser.add_argument(
        "--use-openai", action="store_true",
        help="Force use of OpenAI API (otherwise auto-detected from model name)"
    )
    args = parser.parse_args()

    # Auto-detect OpenAI usage if model is a known GPT model or --use-openai flag is set
    openai_models = {"gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"}
    use_openai = args.use_openai or args.model in openai_models

    # Build response_files list
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        response_files = [
            os.path.join(
                args.output_dir,
                os.path.basename(pf).replace("prompts", "responses_labeled")
            )
            for pf in args.prompt_files
        ]
    else:
        response_files = args.response_files

    process_files(
        prompt_files=args.prompt_files,
        response_files=response_files,
        model=args.model,
        use_openai=use_openai
    )


if __name__ == "__main__":
    main()
