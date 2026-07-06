#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import logging
import pandas as pd
import re
import csv
from typing import List

# — Fix import path for hallu_detector —
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SRC_DIR      = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.hallu_detector.generate import simple_generate_hf, simple_generate_openai
from src.hallu_detector.detect  import detect_details, Thresholds, preflight_strict


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)

# -------------------- helpers --------------------
HEADER_ALIASES = {
    "prompt": {"prompt", "text", "question"},
    "correct_answer": {"correct_answer", "answer", "gold", "ground_truth", "label"},
    "id": {"id", "index", "qid"},
    "template": {"template", "domain", "topic", "seed"},
}

def _find_col(df: pd.DataFrame, target: str) -> str | None:
    want = HEADER_ALIASES[target]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in want:
        if cand in lower_map:
            return lower_map[cand]
    return None

def _sniff_delimiter(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            return dialect.delimiter
        except csv.Error:
            return ","
    except Exception:
        return ","

def _read_csv_robust(path: str) -> pd.DataFrame | None:
    sep = _sniff_delimiter(path)
    try:
        return pd.read_csv(path, sep=sep, keep_default_na=False)
    except Exception:
        pass
    try:
        return pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip", keep_default_na=False)
    except Exception as e:
        logging.error(f"Failed to read {path}: {e}")
        return None

def _clean_response(raw: str, prompt: str) -> str:
    """
    Extract first concise answer. Prefer text after 'Answer:' (case-insensitive),
    else first non-empty line. Strip echoed prompt and trailing punctuation.
    """
    text = raw or ""
    m = re.search(r'Answer:\s*(.*)', text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        after = m.group(1).strip()
        lines = [ln.strip() for ln in after.splitlines() if ln.strip()]
        text = lines[0] if lines else after
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = lines[0] if lines else text.strip()
    if prompt and text.startswith(prompt):
        text = text[len(prompt):].strip()
    text = re.sub(r"[.\s]+$", "", text)
    return text

def _wrap_prompt(base_q: str) -> str:
    instruction = (
        "You are an expert quiz assistant. "
        "Answer the following question with a single concise statement. "
        "Don't include any words if they are not necessary"
        "If your answer is a number, write it as a numeral (e.g., 3). "
        "Do not repeat the question or add commentary.\n"
    )
    return f"{instruction}Question: {base_q}\nAnswer:"

def _derive_template_from_filename(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]  # e.g., prompts_riddle
    if "prompts_" in stem:
        return stem.split("prompts_", 1)[1].strip() or stem
    return stem

def _normalize_schema(df: pd.DataFrame, src_path: str) -> pd.DataFrame | None:
    col_prompt = _find_col(df, "prompt")
    col_answer = _find_col(df, "correct_answer")
    col_id     = _find_col(df, "id")
    col_tpl    = _find_col(df, "template")

    if col_prompt is None:
        logging.error(f"{src_path} is missing a prompt/text/question column.")
        return None
    if col_answer is None:
        logging.error(f"{src_path} is missing a correct_answer/answer/gold column.")
        return None

    out = pd.DataFrame({
        "prompt": df[col_prompt].astype(str),
        "correct_answer": df[col_answer].astype(str),
    })

    if col_id is not None:
        out["id"] = pd.to_numeric(df[col_id], errors="coerce").fillna(0).astype(int)
    else:
        out["id"] = range(1, len(out) + 1)

    if col_tpl is not None:
        out["template"] = df[col_tpl].astype(str)
    else:
        out["template"] = _derive_template_from_filename(src_path)

    n_before = len(out)
    out = out[out["prompt"].str.strip() != ""].copy()
    if out.empty:
        logging.warning(f"All prompts empty after normalization: {src_path}")
        return None
    if len(out) < n_before:
        logging.info(f"Filtered {n_before - len(out)} empty prompts in {src_path}")

    if out["id"].duplicated(keep=False).any():
        logging.warning(f"{src_path} contains duplicate ids; reassigning sequential ids.")
        out = out.reset_index(drop=True)
        out["id"] = range(1, len(out) + 1)

    return out[["template", "id", "prompt", "correct_answer"]]

# -------------------- core --------------------
def process_files(prompt_files: List[str], response_files: List[str], model_name: str, use_openai: bool):
    if len(prompt_files) != len(response_files):
        raise ValueError("prompt_files and response_files must have the same length.")

    for pth_in, pth_out in zip(prompt_files, response_files):
        logging.info(f"Reading prompts from {pth_in}")

        if not os.path.exists(pth_in):
            logging.error(f"Input file not found: {pth_in}")
            continue
        if os.path.getsize(pth_in) == 0:
            logging.warning(f"Skipping empty file: {pth_in}")
            continue

        df_raw = _read_csv_robust(pth_in)
        if df_raw is None or df_raw.shape[1] == 0:
            logging.error(f"Skipping unreadable file: {pth_in}")
            continue

        df_prompts = _normalize_schema(df_raw, pth_in)
        if df_prompts is None or df_prompts.empty:
            logging.error(f"Skipping {pth_in}: cannot normalize schema.")
            continue

        wrapped = [(int(row["id"]), _wrap_prompt(row["prompt"]), None) for _, row in df_prompts.iterrows()]

        backend = "OpenAI" if use_openai else "HF"
        logging.info(f"Generating {len(wrapped)} responses via {backend}::{model_name}")

        try:
            if use_openai:
                gen = simple_generate_openai(wrapped, model_name=model_name)
            else:
                gen = simple_generate_hf(wrapped, model_name=model_name)
        except Exception as e:
            logging.error(f"Generation failed for {pth_in}: {e}")
            continue

        if not gen:
            logging.error(f"No results generated for {pth_in}")
            continue

        df_gen = pd.DataFrame(gen, columns=["id", "wrapped_prompt", "model_response_raw"])

        try:
            df = df_prompts.merge(df_gen[["id", "model_response_raw"]], on="id", how="left", validate="one_to_one")
        except Exception as e:
            logging.error(f"Merge failed for {pth_in}: {e}")
            continue

        df["model_response"] = df.apply(lambda r: _clean_response(r["model_response_raw"], r["prompt"]), axis=1)

        # Detect hallucinations + baselines in one pass (avoid double model calls).
        th = Thresholds()  # default thresholds; keep stable across runs for comparability
        hallu = []
        b_exact = []
        b_embed = []
        b_embed_score = []
        b_embed_method = []
        reason = []

        for _, r in df.iterrows():
            ans = (r.get("model_response") or "").strip()
            corr = (r.get("correct_answer") or "").strip()
            details = detect_details(ans, corr, th)
            if ans == "":
                hallu.append(False)
                logging.info("Model Abstained from Answering, counted as non-hallucination")
            else:
                hallu.append(bool(details.get("hallucinated", False)))
            b_exact.append(bool(details.get("baseline_exact", False)))
            b_embed.append(bool(details.get("baseline_embed", False)))
            b_embed_score.append(details.get("baseline_embed_score", None))
            b_embed_method.append(details.get("baseline_embed_method", "none"))
            reason.append(details.get("reason", "none"))


        df["hallucinated"] = hallu
        df["baseline_exact"] = b_exact
        df["baseline_embed"] = b_embed
        df["baseline_embed_score"] = b_embed_score
        df["baseline_embed_method"] = b_embed_method
        df["reasoning"] = reason

        # Write with model_response included
        out_df = df[[
            "template",
            "id",
            "prompt",
            "model_response",
            "correct_answer",
            "hallucinated",
            "baseline_exact",
            "baseline_embed",
            "baseline_embed_score",
            "baseline_embed_method",
            "reasoning"
        ]].copy()

        try:
            os.makedirs(os.path.dirname(pth_out), exist_ok=True)
            out_df.to_csv(pth_out, index=False, encoding="utf-8")
            logging.info(f"Wrote {len(out_df)} rows to {pth_out}")
        except Exception as e:
            logging.error(f"Failed to write {pth_out}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(
        description="Generate model responses, label hallucinations, and write outputs with model_response."
    )
    parser.add_argument("--prompt-files", "-i", nargs="+", required=True,
                        help="Input CSV(s) (headers may be prompt/text/question and correct_answer/answer/gold)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--response-files", "-o", nargs="+",
                       help="Exact output CSV path(s), matching the count of prompt-files")
    group.add_argument("--output-dir",
                       help="Directory to write labeled response CSVs (filenames derived)")
    parser.add_argument("--model", "-m", required=True,
                        help="Model name, e.g., gpt-4o, gpt-4-mini, or a HF model id")
    parser.add_argument("--use-openai", action="store_true",
                        help="Force OpenAI API; otherwise auto when model starts with 'gpt-'")
    args = parser.parse_args()

    preflight_strict(Thresholds())

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
