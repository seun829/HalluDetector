#!/usr/bin/env python3
import argparse
import os
import logging
import re
import yaml
import pandas as pd
import wikipedia
from openai import OpenAI

# — Configure logging —
logging.basicConfig(level=logging.INFO)

# — OpenAI setup —
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY not set. Exiting...")
    exit(1)
client = OpenAI(api_key=OPENAI_API_KEY)


def load_config(config_path: str | None) -> dict:
    """
    Load YAML config or fallback to defaults if no path given or not found.
    """
    default = {
        "seeds": ["France", "the internet", "democracy"],
        "questions_per_seed": 3
    }
    if not config_path or not os.path.exists(config_path):
        logging.info(f"No config at '{config_path}'. Using default seeds/questions_per_seed.")
        return default
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return {**default, **cfg}


def generate_questions_for_seed(seed: str, count: int) -> list[str]:
    """
    Ask OpenAI to generate `count` tricky questions about `seed`.
    """
    system_msg = (
        f"You are a creative prompt generator. "
        f"Generate exactly {count} unique, tricky, weirdly worded questions that ask about '{seed}'."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg}],
            temperature=0.8,
            max_tokens=150
        )
        content = resp.choices[0].message.content.strip()
        questions = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            # Remove leading numbering like "1. " or "2)"
            q = re.sub(r"^\d+[\.\)]\s*", "", line)
            questions.append(q)
        return questions
    except Exception as e:
        logging.error(f"Failed to generate questions for '{seed}': {e}")
        return []


def determine_correct_answer(question: str, seed: str) -> str:
    """
    Use Wikipedia to find a concise answer, fallback to "Unknown".
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
    Build DataFrame of auto-generated questions and their answers.
    """
    rows = []
    pid = 1
    for seed in seeds:
        qs = generate_questions_for_seed(seed, per_seed)
        for q in qs:
            ans = determine_correct_answer(q, seed)
            rows.append({
                "id": pid,
                "prompt": q,
                "correct_answer": ans
            })
            pid += 1
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate hallucination-detection prompts."
    )
    parser.add_argument(
        "--config",
        help="(Optional) YAML config with 'seeds' and 'questions_per_seed'."
    )
    parser.add_argument(
        "--out-csv",
        default="data/raw/prompts_auto-generated.csv",
        help="Fallback CSV path if --output-dir not provided."
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write the CSV into (uses same filename as --out-csv)."
    )
    args = parser.parse_args()

    # Load config (optional)
    cfg = load_config(args.config)
    df = build_prompts(cfg["seeds"], cfg["questions_per_seed"])

    # Determine final output path
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        base = os.path.basename(args.out_csv)
        out_path = os.path.join(args.output_dir, base)
    else:
        out_path = args.out_csv

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    logging.info(f"✅ Wrote {len(df)} prompts to {out_path}")


if __name__ == "__main__":
    main()
