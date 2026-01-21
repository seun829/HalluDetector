#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""make_audit_set.py — create a human-audit labeling file.

Credibility utility:
- Sample N rows from labeled response CSVs
- Stratify to (roughly) half detector-positive and half detector-negative
- Output a CSV with empty columns for humans to fill in

Usage (PowerShell examples):
  python scripts/make_audit_set.py --input-root output --n 450 --output audit/audit_set.csv

  python scripts/make_audit_set.py --response-files "output/<RUN_ID>/processed/responses_labeled_*.csv" --n 300 --output audit/audit_set.csv

Output columns:
  source_file,template,id,prompt,correct_answer,model_response,detector_hallucinated,human_hallucinated,human_notes
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, List, Optional, Sequence

import pandas as pd


def _as_bool(x: Any) -> bool:
    if x is True:
        return True
    if x is False or x is None:
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y"}


def _discover_response_files(input_root: Path, pattern: str) -> List[str]:
    """Recursively discover matching CSVs under input_root."""
    files = sorted(str(p) for p in input_root.rglob(pattern) if p.is_file())
    return files


def _load_frames(
    response_files: Sequence[str],
    *,
    n_total: int,
    seed: int,
    stop_when_pool_reaches: bool = True,
) -> pd.DataFrame:
    """
    Load labeled response CSV(s) into one dataframe suitable for stratified sampling.

    If stop_when_pool_reaches is True, we stop early once the pool is comfortably large
    for drawing the requested sample.
    """
    rng = random.Random(int(seed))

    # We want ~half pos / half neg; keep a pool somewhat larger than needed
    # to reduce the chance we stop early with a lopsided pool.
    pos_target = n_total // 2
    neg_target = n_total - pos_target
    # "Comfortable" pool sizes before we stop reading more files:
    pos_pool_goal = max(pos_target * 2, pos_target + 25)
    neg_pool_goal = max(neg_target * 2, neg_target + 25)

    frames: List[pd.DataFrame] = []
    pos_count = 0
    neg_count = 0

    # Iterate files in a deterministic order; you can shuffle if you prefer.
    for rf in response_files:
        df = pd.read_csv(rf)

        required = {"prompt", "correct_answer", "model_response", "hallucinated"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"{rf} missing required columns: {sorted(missing)}")

        keep_cols = [
            c for c in ["template", "id", "prompt", "correct_answer", "model_response", "hallucinated"]
            if c in df.columns
        ]
        d2 = df[keep_cols].copy()
        d2.insert(0, "source_file", os.path.basename(rf))

        d2["detector_hallucinated"] = d2["hallucinated"].apply(_as_bool)

        # Update running pool counts
        pos_count += int((d2["detector_hallucinated"] == True).sum())  # noqa: E712
        neg_count += int((d2["detector_hallucinated"] == False).sum())  # noqa: E712

        frames.append(d2)

        if stop_when_pool_reaches and pos_count >= pos_pool_goal and neg_count >= neg_pool_goal:
            # We have a healthy pool for both strata; stop early.
            break

    all_df = pd.concat(frames, ignore_index=True)
    if all_df.empty:
        raise SystemExit("No rows loaded from the provided response files.")

    # Shuffle rows to reduce any ordering artifacts before sampling
    all_df = all_df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    return all_df


def main():
    p = argparse.ArgumentParser(description="Create a stratified human-audit CSV from labeled response files.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--response-files", nargs="+", help="Input labeled response CSV(s) (can include globs)")
    src.add_argument("--input-root", help="Scan this directory recursively for labeled response CSVs")

    p.add_argument("--pattern", default="responses_labeled_*.csv",
                   help="Filename pattern used with --input-root (default: responses_labeled_*.csv)")

    p.add_argument("--output", required=True, help="Output audit CSV path")
    p.add_argument("--n", type=int, default=300, help="Total samples to include (default: 300)")
    p.add_argument("--seed", type=int, default=1337, help="Random seed")

    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing audit file: {out_path}")

    n_total = int(args.n)
    seed = int(args.seed)
    rng = random.Random(seed)

    if args.input_root:
        input_root = Path(args.input_root)
        if not input_root.exists() or not input_root.is_dir():
            raise SystemExit(f"--input-root must be an existing directory: {input_root}")
        response_files = _discover_response_files(input_root, args.pattern)
        if not response_files:
            raise SystemExit(f"No files matching pattern '{args.pattern}' found under: {input_root}")
    else:
        # PowerShell note: wildcard expansion may not happen depending on quoting.
        # We accept whatever the shell passed; pandas will error if files don't exist.
        response_files = list(args.response_files or [])
        if not response_files:
            raise SystemExit("No --response-files provided.")

    all_df = _load_frames(response_files, n_total=n_total, seed=seed, stop_when_pool_reaches=True)

    pos = all_df[all_df["detector_hallucinated"] == True].copy()  # noqa: E712
    neg = all_df[all_df["detector_hallucinated"] == False].copy()  # noqa: E712

    n_pos_target = n_total // 2
    n_neg_target = n_total - n_pos_target

    n_pos = min(len(pos), n_pos_target)
    n_neg = min(len(neg), n_neg_target)

    # If one stratum is short, fill with the other
    remaining = n_total - (n_pos + n_neg)
    if remaining > 0:
        if len(pos) - n_pos > len(neg) - n_neg:
            n_pos = min(len(pos), n_pos + remaining)
        else:
            n_neg = min(len(neg), n_neg + remaining)

    pos_idx = rng.sample(list(pos.index), k=n_pos) if n_pos > 0 else []
    neg_idx = rng.sample(list(neg.index), k=n_neg) if n_neg > 0 else []

    sample_df = pd.concat([pos.loc[pos_idx], neg.loc[neg_idx]], ignore_index=True)
    sample_df = sample_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Human labeling columns
    sample_df["human_hallucinated"] = ""  # to be filled: True/False
    sample_df["human_notes"] = ""         # optional

    out_cols = [
        "source_file",
        "template" if "template" in sample_df.columns else None,
        "id" if "id" in sample_df.columns else None,
        "prompt",
        "correct_answer",
        "model_response",
        "detector_hallucinated",
        "human_hallucinated",
        "human_notes",
    ]
    out_cols = [c for c in out_cols if c is not None]

    sample_df[out_cols].to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote audit set with {len(sample_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
