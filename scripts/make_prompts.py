# scripts/make_prompts.py
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
from typing import List, Dict

import pandas as pd
import yaml

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
        "seeds": ["France", "the internet", "democracy"],
        "questions_per_seed": 3,
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
# OpenAI (legacy SDK) helpers — Option A
# ---------------------------------------------------------------------
def ensure_openai():
    """
    Import legacy OpenAI SDK lazily and ensure key is present.
    (Option A: openai==0.28.1 style, no client object.)
    """
    import importlib

    try:
        openai = importlib.import_module("openai")
    except ImportError:
        logging.error("The 'openai' package is not installed. Run: pip install 'openai==0.28.1'")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not set. Exiting...")
        sys.exit(1)

    openai.api_key = api_key
    return openai


def _chat_completion(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int, retries: int = 3) -> str:
    """
    Call openai.ChatCompletion.create with basic retries and return text content.
    """
    openai = ensure_openai()
    backoff = 1.0
    last_err = None
    for _ in range(max(1, retries)):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # legacy SDK returns message as dict-like
            return resp.choices[0].message["content"].strip()
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 8.0)
    logging.error(f"OpenAI call failed after retries: {last_err}")
    return ""


# ---------------------------------------------------------------------
# Question generation (LLM)
# ---------------------------------------------------------------------
def generate_questions_for_seed(seed: str, count: int, model: str, temperature: float, max_tokens: int) -> List[str]:
    """
    Use ChatGPT (legacy SDK) to generate `count` tricky questions about `seed`.
    """
    system_msg = "You are a creative prompt generator."
    user_msg = (
        f"Generate exactly {count} unique, tricky, weirdly worded questions that ask about '{seed}'. "
        f"Each question should be concise and on a new line. "
        f"Do not add any commentary or numbering; just the questions."
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
# Answering + strict verification (no external KB)
# ---------------------------------------------------------------------
_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _normalize(s: str) -> str:
    """
    Lightweight normalization to compare answers.
    """
    s = s.strip().lower()
    # Remove surrounding quotes and trailing punctuation
    s = s.strip(' "\'')
    s = re.sub(r"[^\w\s\.-]", " ", s)       # keep letters, digits, space, dot, dash
    s = re.sub(r"\s+", " ", s).strip()
    # Remove trailing period if present
    s = s[:-1] if s.endswith(".") else s
    return s


def _answer_once(question: str, model: str, temperature: float, max_tokens: int) -> str:
    content = _chat_completion(
        messages=[
            {"role": "system", "content": "Answer the question factually and concisely with only the final answer, no explanations."},
            {"role": "user", "content": question},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # Heuristic: reject verbose or uncertain text
    bad_markers = ["i cannot", "i'm unable", "as an ai", "not sure", "uncertain", "depends", "it depends", "approximately"]
    if any(b in content.lower() for b in bad_markers):
        return ""
    # reject responses that contain multiple sentences unless they're short facts
    if content.count("\n") > 0 or content.strip().count(". ") > 1:
        # try to take the first sentence
        content = content.split("\n")[0].split(". ")[0]
    return content.strip()


def _verify_answer(question: str, proposed_answer: str, model: str, confidence_min: float) -> tuple[bool, str]:
    """
    Ask the model to verify in strict JSON. Accept only if verdict == YES and confidence >= threshold.
    Returns (ok, normalized_answer).
    """
    verify_prompt = (
        "You are a strict verifier. Given a question and a proposed answer, check if it is factually correct.\n"
        "If and only if you are certain (no guesses), output JSON with exactly these keys:\n"
        '{"verdict":"YES|NO","normalized_answer":"<short canonical answer>","confidence":<float 0..1>}.\n'
        "Do not include any other text."
    )
    user = f"Question: {question}\nProposed answer: {proposed_answer}\nRespond with JSON only."
    content = _chat_completion(
        messages=[{"role": "system", "content": verify_prompt}, {"role": "user", "content": user}],
        model=model,
        temperature=0.0,
        max_tokens=80,
    )
    m = _JSON_PATTERN.search(content)
    if not m:
        return False, ""
    try:
        data = json.loads(m.group(0))
    except Exception:
        return False, ""

    verdict = str(data.get("verdict", "")).strip().upper()
    normalized_answer = str(data.get("normalized_answer", "")).strip()
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    if verdict != "YES":
        return False, ""
    if confidence < confidence_min:
        return False, ""
    if not normalized_answer:
        return False, ""
    return True, normalized_answer


def determine_correct_answer(question: str, model: str, max_tokens_answer: int, confidence_min: float, votes: int) -> str:
    """
    Bulletproof approach:
      1) Get a concise answer (temp=0).
      2) Verify with strict JSON; require confidence >= threshold.
      3) Self-consistency: answer the question multiple times (temp=0) and require
         all normalized answers to match the verifier's normalized answer.
      4) Otherwise, return 'Unknown'.
    """
    candidate = _answer_once(question, model=model, temperature=0.0, max_tokens=max_tokens_answer)
    if not candidate:
        return "Unknown"

    ok, normalized = _verify_answer(question, candidate, model=model, confidence_min=confidence_min)
    if not ok:
        return "Unknown"

    # Self-consistency votes
    normalized_target = _normalize(normalized)
    for _ in range(max(1, votes - 1)):
        a = _answer_once(question, model=model, temperature=0.0, max_tokens=max_tokens_answer)
        if not a:
            return "Unknown"
        if _normalize(a) != normalized_target:
            return "Unknown"

    # Passed all checks
    return normalized


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
            )
            rows.append({"id": pid, "prompt": q, "correct_answer": ans, "seed": seed})
            pid += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate hallucination-detection prompts.")
    parser.add_argument("--input-dir", help="Folder containing prompt CSVs (simulation_data) to concatenate")
    parser.add_argument("--config", help="YAML config with 'seeds' and 'questions_per_seed'")
    parser.add_argument("--output-dir", required=True, help="Directory to write combined prompts CSV")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "prompts.csv")

    # 1) Concatenate provided CSVs (pipeline-compatible)
    if args.input_dir:
        files = glob.glob(os.path.join(args.input_dir, "*.csv"))
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

    # 2) Auto-generate from LLM only (no external KB/Wikipedia), with strict verification
    cfg = load_config(args.config)
    df_auto = build_prompts(cfg["seeds"], int(cfg["questions_per_seed"]), cfg)
    df_auto.to_csv(out_path, index=False)
    logging.info(f"✅ Wrote {len(df_auto)} auto-generated prompts to {out_path}")


if __name__ == "__main__":
    main()
