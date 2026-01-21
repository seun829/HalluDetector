#!/usr/bin/env python3
"""Compute hallucination metrics from labeled response files.

Backwards compatible with the original script, but adds optional bootstrap
confidence intervals for arXiv/IEEE-ready reporting.

Output JSON always includes:
- hallucination_rate

When --bootstrap is provided (>0), also includes:
- hallucinations
- n
- hallucination_rate_ci95
- bootstrap_samples
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from typing import Any, Dict, Iterable, List

import pandas as pd

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def _as_bool(x: Any) -> bool:
    if x is True:
        return True
    if x is False or x is None:
        return False
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y"}


def compute_metrics(labels: Iterable[Any], bootstrap_n: int = 0, seed: int = 0) -> Dict[str, Any]:
    bools: List[bool] = [_as_bool(v) for v in labels]
    total = len(bools)
    if total == 0:
        return {"hallucination_rate": None}

    hallu_count = sum(1 for v in bools if v)
    rate = hallu_count / total

    out: Dict[str, Any] = {
        "hallucination_rate": rate,
    }

    if bootstrap_n and bootstrap_n > 0:
        rng = random.Random(int(seed))
        rates: List[float] = []
        for _ in range(int(bootstrap_n)):
            # sample with replacement
            s_count = 0
            for _j in range(total):
                if bools[rng.randrange(total)]:
                    s_count += 1
            rates.append(s_count / total)

        rates.sort()
        lo_idx = max(0, min(len(rates) - 1, int(0.025 * len(rates))))
        hi_idx = max(0, min(len(rates) - 1, int(0.975 * len(rates)) - 1))
        out.update(
            {
                "hallucinations": int(hallu_count),
                "n": int(total),
                "hallucination_rate_ci95": [rates[lo_idx], rates[hi_idx]],
                "bootstrap_samples": int(bootstrap_n),
                "bootstrap_seed": int(seed),
            }
        )

    return out


def process_files(response_files: List[str], metric_files: List[str], bootstrap_n: int = 0, seed: int = 0):
    for response_file, metric_file in zip(response_files, metric_files):
        logging.info("Processing %s...", response_file)
        try:
            df = pd.read_csv(response_file)
        except Exception as e:
            logging.error("Failed to read %s: %s", response_file, e)
            continue

        if "hallucinated" not in df.columns:
            logging.error("Missing 'hallucinated' column in %s. Skipping.", response_file)
            continue

        metrics = compute_metrics(df["hallucinated"].tolist(), bootstrap_n=bootstrap_n, seed=seed)
        with open(metric_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logging.info("Metrics saved to %s.", metric_file)


def main():
    parser = argparse.ArgumentParser(description="Compute hallucination metrics for response files.")
    parser.add_argument("--response-files", nargs="+", required=True, help="List of input response CSV files.")
    parser.add_argument("--metric-files", nargs="+", required=True, help="List of output JSON metric files.")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap samples for 95%% CI (0 disables; recommended: 1000).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrapping.")
    args = parser.parse_args()

    if len(args.response_files) != len(args.metric_files):
        logging.error("Number of response files and metric files must match.")
        return

    process_files(args.response_files, args.metric_files, bootstrap_n=args.bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
