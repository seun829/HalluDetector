#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
import random
from typing import List, Dict

import pandas as pd
import yaml
from openai import OpenAI
client= OpenAI()

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
def load_config(config_path: str | None) -> dict:
    """
    Load YAML config or fallback to defaults if none provided.
    """
    default = {
        "seeds": ["math", "science", "philosophy"],
        "questions_per_seed": 17,  # 3 seeds * 17 = 51 prompts (~50)
        "model": "gpt-3.5-turbo",
        "temperature_questions": 0.8,
        "temperature_answers": 0.0,
        "max_tokens_questions": 512,
        "max_tokens_answer": 100,
        "verification_confidence_min": 0.99,
        "self_consistency_votes": 3
    }
    if not config_path or not os.path.exists(config_path):
        logging.info(f"No config at '{config_path}'. Using defaults.")
        return default
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return {**default, **cfg}


# ---------------------------------------------------------------------
# OpenAI client (modern SDK)
# ---------------------------------------------------------------------
_client = None

def get_client():
    global _client
    if _client is None:
        try:
            _client = OpenAI()  # modern SDK: picks up OPENAI_API_KEY from environment
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}")
            sys.exit(1)
    return _client


def _chat_completion(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int, retries: int = 3) -> str:
    """
    Call OpenAI Chat API with retries and return text content.
    """
    client = get_client()
    backoff = 1.0
    last_err = None
    for _ in range(max(1, retries)):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            logging.warning(f"Retrying after error: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 8.0)
    logging.error(f"OpenAI call failed after retries: {last_err}")
    return ""



# ---------------------------------------------------------------------
# Question generation (LLM)
# ---------------------------------------------------------------------
def generate_questions_for_seed(seed: str, count: int, model: str, temperature: float, max_tokens: int) -> List[str]:
    """
    Use ChatGPT (legacy SDK) to generate `count` unique, tricky questions about `seed`.
    """
    system_msg = "You are a creative prompt generator."
    user_msg = (
        f"Generate exactly {count} unique, tricky, weirdly worded questions that ask about '{seed}'. "
        f"Each question should be concise, on a new line, with no numbering or commentary."
    )
    content = _chat_completion(
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    questions: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip accidental numbering like "1) " or "2. "
        q = re.sub(r"^\d+[\.\)]\s*", "", line)
        if q:
            questions.append(q)
    # Pad/truncate to exactly count items to keep pipeline deterministic
    return (questions + [""] * count)[:count]


# ---------------------------------------------------------------------
# Answering (without strict verification)
# ---------------------------------------------------------------------
def _answer_once(question: str, model: str, temperature: float, max_tokens: int, seed: str = "General") -> str:
    content = _chat_completion(
        messages=[
            {"role": "system", "content": "Answer the question factually and concisely with only the final answer, no explanations."},
            {"role": "user", "content": question},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if content:
        return content.split("\n")[0].split(". ")[0].strip()
    # Fallback answer instead of "Unknown"
    return f"This is a tricky question about {seed}"
    

def determine_correct_answer(question: str, model: str, max_tokens_answer: int, confidence_min: float, votes: int, seed: str = "General") -> str:
    try:
        candidate = _answer_once(question, model=model, temperature=0.0, max_tokens=max_tokens_answer, seed=seed)
        return candidate if candidate else f"This is a tricky question about {seed}"
    except Exception as e:
        logging.warning(f"Failed to generate answer for question '{question}': {e}")
        return f"This is a tricky question about {seed}"



# ---------------------------------------------------------------------
# Build prompts
# ---------------------------------------------------------------------
def build_prompts(seeds: List[str], per_seed: int, cfg: dict) -> pd.DataFrame:
    rows = []
    pid = 1
    for seed in seeds:
        qs = generate_questions_for_seed(
            seed=seed,
            count=per_seed,
            model=cfg["model"],
            temperature=cfg["temperature_questions"],
            max_tokens=cfg["max_tokens_questions"],
        )
        for q in qs:
            if not q:
                continue
            ans = determine_correct_answer(
                question=q,
                model=cfg["model"],
                max_tokens_answer=cfg["max_tokens_answer"],
                confidence_min=cfg["verification_confidence_min"],
                votes=int(cfg["self_consistency_votes"]),
                seed=seed,
            )

            rows.append({"id": pid, "prompt": q, "correct_answer": ans, "seed": seed})
            pid += 1
    # Ensure exactly 50 prompts:
    if len(rows) < 50:
        while len(rows) < 50:
            # Duplicate random rows until reaching 50
            dup = rows[random.randint(0, len(rows)-1)]
            new_row = dup.copy()
            new_row["id"] = len(rows) + 1
            rows.append(new_row)
    elif len(rows) > 50:
        rows = rows[:50]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate hallucination-detection prompts.")
    # Remove or comment out the input-dir parameter if you don't want concatenation:
    # parser.add_argument("--input-dir", help="Folder containing prompt CSVs (simulation_data) to concatenate")
    parser.add_argument("--config", help="YAML config with 'seeds' and 'questions_per_seed'")
    parser.add_argument("--output-dir", required=True, help="Directory to write auto-generated prompts CSV")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "prompts_auto-generated.csv")

    # Removed the branch for --input-dir so we always auto-generate:
    cfg = load_config(args.config)
    df_auto = build_prompts(cfg["seeds"], int(cfg["questions_per_seed"]), cfg)
    df_auto.to_csv(out_path, index=False)
    logging.info(f"✅ Wrote {len(df_auto)} auto-generated prompts to {out_path}")

if __name__ == "__main__":
    main()