"""
Generate templated prompts into a CSV for the hallucination detection pipeline.
"""
import argparse
import pandas as pd
from itertools import product
import yaml
import logging

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_prompts(templates: list[str], seeds: list[str]) -> pd.DataFrame:
    rows = []
    pid = 1
    for tpl, txt in product(templates, seeds):
        prompt = tpl.format(txt)
        rows.append({
            "id": pid,
            "template": tpl,
            "seed": txt,
            "prompt": prompt,
        })
        pid += 1
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 100 automated prompts ranging in difficulty for ai hallucination tests"
    )
    parser.add_argument(
        "--config", required=True,
        help="YAML config with 'templates' and 'seeds' lists"
    )
    parser.add_argument(
        "--out-csv", default="data/processed/automated_prompts.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)

    df = build_prompts(
        templates=config.get('templates', []),
        seeds=config.get('seeds', [])
    )
    df.to_csv(args.out_csv, index=False)
    logging.info(f"Wrote {len(df)} prompts to {args.out_csv}")


if __name__ == '__main__':
    main()
