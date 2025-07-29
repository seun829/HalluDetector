#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import pandas as pd

# allow importing your generate logic
sys.path.append(os.path.abspath(os.path.join(__file__, "../src")))
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
        df_prompts = pd.read_csv(pth_in)
        if df_prompts.empty:
            logging.warning(f"Skipping {pth_in}: empty")
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

        if use_openai:
            results = simple_generate_openai(wrapped, model_name=model)
        else:
            results = simple_generate_hf(wrapped, model_name=model)

        df_out = pd.DataFrame(results, columns=["id", "wrapped_prompt", "model_response_raw"])
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
    p = argparse.ArgumentParser(
        description="Generate concise model responses for prompt CSVs"
    )
    p.add_argument(
        "--prompt-files", "-i", nargs="+", required=True,
        help="Input prompt CSV(s) (id,prompt,correct_answer)"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--response-files", "-o", nargs="+",
        help="Explicit output CSV path(s), matching count of prompt-files"
    )
    group.add_argument(
        "--output-dir",
        help="Directory to write labeled response CSVs (filenames derived)"
    )
    p.add_argument(
        "--model", "-m", default="gpt-3.5-turbo",
        help="Model name for HuggingFace or OpenAI"
    )
    p.add_argument(
        "--use-openai", action="store_true",
        help="Use OpenAI API instead of HF"
    )
    args = p.parse_args()

    # build response_files list
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
        use_openai=args.use_openai
    )


if __name__ == "__main__":
    main()
