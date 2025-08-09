# scripts/make_prompts.py
#!/usr/bin/env python3
import argparse
import os
import glob
import logging
import re
import sys
import yaml
import pandas as pd
import wikipedia
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)

# OpenAI setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY not set. Exiting...")
    sys.exit(1)
client = OpenAI(api_key=OPENAI_API_KEY)


def load_config(config_path: str | None) -> dict:
    """
    Load YAML config or fallback to defaults if none provided.
    """
    default = {
        "seeds": ["France", "the internet", "democracy"],
        "questions_per_seed": 3
    }
    if not config_path or not os.path.exists(config_path):
        logging.info(f"No config at '{config_path}'. Using defaults.")
        return default
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return {**default, **cfg}


def generate_questions_for_seed(seed: str, count: int) -> list[str]:
    """
    Use ChatGPT to generate `count` tricky questions about `seed`.
    """
    system_msg = (
        f"You are a creative prompt generator."
        f" Generate exactly {count} unique, tricky, weirdly worded questions that ask about '{seed}'."
        f" Each question should be concise and on a new line"
        f" Do not add any commentary or additional text."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_msg}],
            temperature=0.8,
            max_tokens=4000
        )
        content = resp.choices[0].message.content.strip()
        questions = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip numbering
            q = re.sub(r"^\d+[\.\)]\s*", "", line)
            questions.append(q)
        return questions
    except Exception as e:
        logging.error(f"Failed to generate questions for '{seed}': {e}")
        return []


def determine_correct_answer(question: str, seed: str) -> str:
    """
    Query Wikipedia for a concise answer, or return "Unknown".
    """
    try:
        results = wikipedia.search(question) or wikipedia.search(seed)
        if not results:
            return "Unknown"
        page = wikipedia.page(results[0])
        ans = page.summary.split('. ')[0]
        return ans + ('.' if not ans.endswith('.') else '')
    except Exception:
        return "Unknown"


def build_prompts(seeds: list[str], per_seed: int) -> pd.DataFrame:
    """
    Auto-generate prompts and answers for each seed.
    """
    rows = []
    pid = 1
    for seed in seeds:
        qs = generate_questions_for_seed(seed, per_seed)
        for q in qs:
            ans = determine_correct_answer(q, seed)
            rows.append({"id": pid, "prompt": q, "correct_answer": ans})
            pid += 1
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate hallucination-detection prompts.")
    parser.add_argument(
        "--input-dir", help="Folder containing prompt CSVs (simulation_data)"
    )
    parser.add_argument(
        "--config", help="YAML config with 'seeds' and 'questions_per_seed'"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write combined prompts CSV"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'prompts.csv')

    # 1️⃣ If simulation_data is provided, concatenate those CSVs
    if args.input_dir:
        files = glob.glob(os.path.join(args.input_dir, '*.csv'))
        if not files:
            logging.error(f"No CSV files found in {args.input_dir}")
            sys.exit(1)
        df_list = []
        for f in files:
            try:
                df_list.append(pd.read_csv(f))
            except Exception as e:
                logging.warning(f"Skipping {f}: {e}")
        if not df_list:
            logging.error("No valid CSVs to concatenate.")
            sys.exit(1)
        df_all = pd.concat(df_list, ignore_index=True)
        df_all.to_csv(out_path, index=False)
        logging.info(f"✅ Combined {len(df_all)} prompts into {out_path}")
        sys.exit(0)

    # 2️⃣ Fallback: auto-generate from OpenAI + Wikipedia
    cfg = load_config(args.config)
    df_auto = build_prompts(cfg['seeds'], cfg['questions_per_seed'])
    df_auto.to_csv(out_path, index=False)
    logging.info(f"✅ Wrote {len(df_auto)} auto-generated prompts to {out_path}")


if __name__ == '__main__':
    main()