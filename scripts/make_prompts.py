#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_prompts.py — API-first generator (CSV only)

Writes ONE file: <output-dir>/prompts_auto-generated.csv
Columns (in order): template,id,prompt,correct_answer

ENV:
  OPENAI_API_KEY must be set.

CLI:
  --config <yaml>            (optional; overrides defaults)
  --output-dir <dir>         (required)
  --exact-total 50           (optional; default 50)
  --seeds ...                (optional)
  --questions-per-seed N     (optional)
  --random-seed 1337         (optional)
"""

import argparse, logging, os, random, re, time, json, csv
from typing import List
import pandas as pd

# Optional YAML config
try:
    import yaml
except Exception:
    yaml = None

# OpenAI (modern SDK)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_CFG = {
    "seeds": ["mathematics","physics","chemistry","philosophy","logic",
              "astronomy","music","law","networking","science"],
    "questions_per_seed": 5,   # 10 * 5 = 50
    "model": "gpt-4o",
    "temperature_questions": 0.7,
    "temperature_answers": 0.0,
    "max_tokens_questions": 400,
    "max_tokens_answer": 64,
    "self_consistency_votes": 3,
    "random_seed": 1337,
}

# -------------------- helpers --------------------
def load_config(path: str | None) -> dict:
    if path and os.path.exists(path):
        if yaml is None:
            raise RuntimeError("pyyaml not installed; cannot parse YAML config.")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg = {**DEFAULT_CFG, **cfg}
    else:
        logging.info("No config at '%s'. Using defaults.", path)
        cfg = DEFAULT_CFG.copy()
    return cfg

_client = None
def _client_ok() -> bool:
    key = os.getenv("OPENAI_API_KEY")
    return bool(key and key.strip())

def get_client():
    global _client
    if _client is not None:
        return _client
    if OpenAI is None:
        raise RuntimeError("openai package not available. Install `openai` >= 1.0 and set OPENAI_API_KEY.")
    if not _client_ok():
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    _client = OpenAI()
    return _client

def _chat(messages, model, temperature, max_tokens, retries=4, backoff=1.0) -> str:
    c = get_client()
    last = None
    for _ in range(retries):
        try:
            resp = c.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last = e
            logging.warning("OpenAI error (%s). Retrying...", e)
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 8.0)
    raise RuntimeError(f"OpenAI call failed after {retries} retries: {last}")

def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())

def _strip_numbering(s: str) -> str:
    return re.sub(r"^\s*\d+[\.\)]\s*", "", s).strip()

def _atomic_write_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    # enforce column order & quoting
    ordered = df[["template","id","prompt","correct_answer"]]
    ordered.to_csv(
        tmp,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,      # more robust for downstream consumers
        lineterminator="\n",
        escapechar="\\",
    )
    if ordered.shape[0] == 0:
        raise RuntimeError("Refusing to write empty prompts CSV.")
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)

# ---------- Robust JSON-array extraction from model output ----------
_CODE_FENCE_RE = re.compile(r"```(?:json|javascript|js|python)?\s*(.*?)```", re.S | re.I)

def _unfence(s: str) -> str:
    """Return the first fenced block contents if present; else original string."""
    m = _CODE_FENCE_RE.search(s)
    return m.group(1) if m else s

def _extract_json_array_text(s: str) -> str | None:
    """Return substring spanning the first balanced JSON array; else None."""
    start = s.find('[')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def _parse_questions_text(txt: str) -> List[str]:
    """Best-effort to obtain a list[str] of questions from possibly noisy model text."""
    txt = _unfence(txt)

    # Try strict JSON
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return [_norm(x) for x in data if isinstance(x, str) and x.strip()]
    except Exception:
        pass

    # Try extracting the first JSON array region
    arr_txt = _extract_json_array_text(txt)
    if arr_txt:
        try:
            data = json.loads(arr_txt)
            if isinstance(data, list):
                return [_norm(x) for x in data if isinstance(x, str) and x.strip()]
        except Exception:
            pass

    # Fallback: line-based, ignore fence/bracket noise and trailing commas
    out = []
    for line in txt.splitlines():
        line = _strip_numbering(_norm(line.rstrip(',')))
        if not line:
            continue
        if line in ('```', '```json', '```javascript', '```js', '[', ']', ','):
            continue
        if re.fullmatch(r'^[\[\]{}(),]+$', line):
            continue
        out.append(line)
    return out

# -------------------- API logic --------------------
def generate_questions_for_seed_api(seed: str, count: int, cfg: dict) -> List[str]:
    """
    Request a JSON array of strings for robust parsing. Up to 3 attempts to reach `count`.
    """
    sys_msg = "You are a careful prompt generator. Always follow output format exactly."
    def _ask(k: int) -> List[str]:
        user_msg = (
            "Generate exactly {k} unique, tricky, one-sentence questions ABOUT the topic: '{seed}'. "
            "Do NOT include answers. Do NOT number them. "
            "Return a JSON array ONLY (no code fences, no markdown, no extra text), exactly like "
            "[\"q1\", \"q2\", ...]. "
            "Each question must be answerable with a short, unambiguous phrase."
        ).format(k=k, seed=seed)
        txt = _chat(
            messages=[{"role":"system","content":sys_msg},{"role":"user","content":user_msg}],
            model=cfg["model"],
            temperature=cfg["temperature_questions"],
            max_tokens=cfg["max_tokens_questions"],
        )
        return _parse_questions_text(txt)

    seen, out = set(), []
    attempts = 0
    need = count
    while len(out) < count and attempts < 3:
        batch = _ask(need)
        for q in batch:
            q = q.strip()
            if q and q not in seen:
                out.append(q); seen.add(q)
                if len(out) == count:
                    break
        need = count - len(out)
        attempts += 1

    if len(out) != count:
        raise RuntimeError(f"Model returned {len(out)} questions (expected {count}) for seed '{seed}'.")
    return out

def answer_question_api(question: str, cfg: dict) -> str:
    """
    Short factual answer via self-consistency voting; returns lowercase string without trailing punctuation.
    """
    sys_msg = (
        "Answer the user's question with ONLY the final answer as a short phrase/number/name. "
        "It must be super concise. "
        "No punctuation at the end, no units unless necessary, no explanation."
    )
    votes = max(1, int(cfg.get("self_consistency_votes", 3)))
    answers: List[str] = []
    for _ in range(votes):
        txt = _chat(
            messages=[{"role":"system","content":sys_msg},{"role":"user","content":question}],
            model=cfg["model"],
            temperature=cfg["temperature_answers"],
            max_tokens=cfg["max_tokens_answer"],
        )
        if txt:
            first = _norm(txt.splitlines()[0])
            first = re.sub(r"[.\s]+$", "", first)
            if first:
                answers.append(first.lower())
    if not answers:
        raise RuntimeError("Model failed to answer.")
    from collections import Counter
    return Counter(answers).most_common(1)[0][0]

# -------------------- build & main --------------------
def build_prompts(seeds: List[str], per_seed: int, cfg: dict, exact_total: int | None) -> pd.DataFrame:
    rows = []
    pid = 1
    for seed in seeds:
        qs = generate_questions_for_seed_api(seed, per_seed, cfg)
        for q in qs:
            ans = answer_question_api(q, cfg)
            rows.append({"template": seed, "id": pid, "prompt": q, "correct_answer": ans})
            pid += 1

    df = pd.DataFrame(rows, columns=["template","id","prompt","correct_answer"])

    # Exact total handling (default 50)
    if isinstance(exact_total, int) and exact_total > 0:
        if len(df) < exact_total:
            need = exact_total - len(df)
            extra = df.sample(n=need, replace=True, random_state=42).copy()
            # reassign sequential ids
            extra["id"] = range(len(df) + 1, len(df) + need + 1)
            df = pd.concat([df, extra], ignore_index=True)
        elif len(df) > exact_total:
            df = df.sample(n=exact_total, random_state=42).reset_index(drop=True)
            df["id"] = range(1, exact_total + 1)

    # Final sanity
    if df.empty or df["prompt"].astype(str).str.strip().eq("").all():
        raise RuntimeError("Generated an empty prompt set; aborting.")
    if df["correct_answer"].astype(str).str.strip().eq("").any():
        raise RuntimeError("At least one correct_answer is empty; aborting.")
    # enforce column order explicitly
    df = df[["template","id","prompt","correct_answer"]]
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate prompts + correct answers via OpenAI (CSV only).")
    parser.add_argument("--config", help="YAML config path (optional)")
    parser.add_argument("--output-dir", required=True, help="Directory to write prompts_auto-generated.csv")
    parser.add_argument("--exact-total", type=int, default=50, help="Total prompts to write (default 50)")
    parser.add_argument("--seeds", nargs="*", help="Override seeds")
    parser.add_argument("--questions-per-seed", type=int, help="Override per-seed count")
    parser.add_argument("--random-seed", type=int, help="Override RNG seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seeds: cfg["seeds"] = args.seeds
    if args.questions_per_seed is not None: cfg["questions_per_seed"] = int(args.questions_per_seed)
    if args.random_seed is not None: cfg["random_seed"] = int(args.random_seed)

    random.seed(int(cfg.get("random_seed", 1337)))

    if not _client_ok():
        raise SystemExit("ERROR: OPENAI_API_KEY is not set. Export it and retry.")

    logging.info("Using OpenAI backend: True (model: %s)", cfg["model"])
    df = build_prompts(cfg["seeds"], int(cfg["questions_per_seed"]), cfg, args.exact_total)

    out_csv = os.path.join(args.output_dir, "prompts_auto-generated.csv")
    _atomic_write_csv(df, out_csv)

    #update
    logging.info("Wrote %d prompts to %s", len(df), out_csv)

if __name__ == "__main__":
    main()
