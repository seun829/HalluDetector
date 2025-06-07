import argparse
import pandas as pd
import yaml
import logging
import os

# External dependencies
import wikipedia  # pip install wikipedia
import openai     # pip install openai

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY not set. Exiting...")
    exit(1)
else:
    openai.api_key = OPENAI_API_KEY


def load_config(config_path: str) -> dict:
    """
    Load YAML config or default: seeds list and number of questions per seed.
    """
    default = {"seeds": ["France", "the internet", "democracy"], "questions_per_seed": 3}
    if not os.path.exists(config_path):
        logging.warning(f"Config file {config_path} not found. Using default config.")
        return default
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return {**default, **cfg}


def generate_questions_for_seed(seed: str, count: int) -> list:
    """
    Ask OpenAI to generate `count` tricky, weirdly worded questions about `seed`.
    Expects a numbered list in the response.
    """
    system_msg = ("You are a creative prompt generator. "
                  "Generate exactly {n} unique, tricky, weirdly worded questions that ask about '{seed}'."
                  .format(n=count, seed=seed))
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg}],
            temperature=0.8,
            max_tokens=150
        )
        content = resp.choices[0].message.content.strip()
        # Split by lines and strip numbering
        questions = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            # remove leading numbering e.g. '1.', '1)'
            q = line.lstrip('0123456789. )')
            questions.append(q)
        return questions
    except Exception as e:
        logging.error(f"Failed to generate questions for '{seed}': {e}")
        return []


def determine_correct_answer(question: str, seed: str) -> str:
    """
    Use Wikipedia to find a concise answer to `question`, fallback to Unknown.
    """
    try:
        # search by question, fallback to seed
        results = wikipedia.search(question) or wikipedia.search(seed)
        if not results:
            return "Unknown"
        page = wikipedia.page(results[0])
        # take first sentence
        ans = page.summary.split('. ')[0]
        return ans + ('.' if not ans.endswith('.') else '')
    except Exception:
        return "Unknown"


def build_prompts(seeds: list, per_seed: int) -> pd.DataFrame:
    """
    Build DataFrame of auto-generated questions and their answers.
    """
    rows = []
    pid = 1
    for seed in seeds:
        questions = generate_questions_for_seed(seed, per_seed)
        for q in questions:
            ans = determine_correct_answer(q, seed)
            rows.append({"id": pid, "prompt": q, "correct_answer": ans})
            pid += 1
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 50 hallucination-detection prompts without templates; questions are invented by the model."
    )
    parser.add_argument("--config", required=True, help="YAML with 'seeds' and 'questions_per_seed'.")
    parser.add_argument("--out-csv", default="data/raw/prompts_auto-generated.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    config = load_config(args.config)
    df = build_prompts(config['seeds'], config['questions_per_seed'])
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    logging.info(f"Wrote {len(df)} prompts to {args.out_csv}")


if __name__ == '__main__':
    main()
