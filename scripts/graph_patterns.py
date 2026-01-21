#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_patterns.py

Visualize hallucination patterns and generate a simple webpage.

Conventions (paper-stable):
- hallucination_rate = mean(hallucinated) within a group
- input CSVs must contain: prompt, model_response, correct_answer, hallucinated
- extra columns (e.g., baselines) are ignored safely
"""

from __future__ import annotations

import argparse
import os
import logging
import re
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.errors import EmptyDataError, ParserError
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Robust bool conversion for hallucinated column."""
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def load_data(filepaths):
    dfs = []
    for fp in filepaths:
        if not os.path.exists(fp):
            logging.warning(f"Skipping missing file: {fp}")
            continue
        try:
            df = pd.read_csv(fp)
        except (EmptyDataError, ParserError) as e:
            logging.warning(f"Skipping malformed: {fp} ({e})")
            continue
        if df.empty:
            logging.warning(f"Skipping empty: {fp}")
            continue

        required = {"prompt", "model_response", "correct_answer", "hallucinated"}
        if not required.issubset(df.columns):
            logging.warning(f"Skipping {fp}: missing columns {required - set(df.columns)}")
            continue

        df = df.copy()
        df["source"] = os.path.basename(fp).replace("_labeled.csv", "")
        df["prompt_length"] = df["prompt"].astype(str).apply(len)
        df["response_length"] = df["model_response"].astype(str).apply(len)
        df["hallucinated"] = _to_bool_series(df["hallucinated"])

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add paper-friendly metrics."""
    df = df.copy()
    # hallucination_rate is computed in group-level aggregations,
    # but we keep is_correct for tables/heatmap convenience.
    df["is_correct"] = (~df["hallucinated"]).astype(int)
    return df


def analyze_by_template(df):
    """Return dataframe: template, hallucination_rate"""
    if "template" not in df.columns:
        return None
    g = df.groupby("template", dropna=False)["hallucinated"].mean().reset_index()
    g.rename(columns={"hallucinated": "hallucination_rate"}, inplace=True)
    return g.sort_values("hallucination_rate", ascending=False)


def analyze_keywords_tfidf(df, top_n=10):
    """
    Find top TF-IDF keywords among hallucinated prompts, then compute hallucination_rate
    for prompts containing each keyword.
    """
    corpus = df["prompt"].astype(str)
    vect = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
        max_features=top_n * 5,
    )
    X = vect.fit_transform(corpus)
    features = vect.get_feature_names_out()

    hallu_mask = df["hallucinated"].to_numpy(dtype=bool)
    if hallu_mask.sum() == 0:
        return pd.DataFrame(columns=["keyword", "hallucination_rate"])

    hallu_avg = X[hallu_mask].mean(axis=0).A1
    pairs = sorted(zip(features, hallu_avg), key=lambda x: x[1], reverse=True)
    kws = [w for w, _ in pairs[:top_n]]

    rows = []
    for kw in kws:
        mask = df["prompt"].astype(str).str.contains(rf"\b{re.escape(kw)}\b", case=False, na=False)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "keyword": kw,
                "hallucination_rate": float(df.loc[mask, "hallucinated"].mean()),
            }
        )
    return pd.DataFrame(rows)


def visualize_results(df, x, y, title, xlabel, ylabel, output):
    if df is None or df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x, y=y, data=df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.savefig(output)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Visualize hallucination patterns and generate webpage.")
    p.add_argument("--input-dir", "-i", required=True, help="Directory with labeled response CSVs")
    p.add_argument("--output-dir", "-o", required=True, help="Directory to write PNG graphs, processed data, and webpage")
    p.add_argument("--top-keywords", type=int, default=10, help="Number of top keywords to extract")
    args = p.parse_args()

    # gather CSVs
    files = sorted(
        [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".csv")
        ]
    )

    df = load_data(files)
    if df.empty:
        logging.error("No valid data; exiting.")
        return

    df = compute_metrics(df)

    # Save processed data (questions, AI responses, correctness)
    processed_dir = os.path.join(args.output_dir, "user", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    proc_csv = os.path.join(processed_dir, "processed_data.csv")
    proc_html = os.path.join(processed_dir, "processed_data.html")

    df[["prompt", "model_response", "is_correct", "hallucinated"]].to_csv(proc_csv, index=False)
    df[["prompt", "model_response", "is_correct", "hallucinated"]].to_html(
        proc_html, index=False, classes="table table-striped"
    )
    logging.info(f"Saved processed CSV to {proc_csv} and HTML to {proc_html}")

    # Generate visualizations FIRST (so webpage can show them reliably)
    out_template = os.path.join(args.output_dir, "hallucinations_by_template.png")
    out_keywords = os.path.join(args.output_dir, "hallucinations_by_keywords.png")
    out_heatmap = os.path.join(args.output_dir, "feature_correlation_heatmap.png")

    ta = analyze_by_template(df)
    if ta is not None and not ta.empty:
        visualize_results(
            ta,
            x="template",
            y="hallucination_rate",
            title="Hallucination Rate by Template",
            xlabel="Template",
            ylabel="Hallucination Rate",
            output=out_template,
        )

    kwa = analyze_keywords_tfidf(df, top_n=args.top_keywords)
    if not kwa.empty:
        visualize_results(
            kwa,
            x="keyword",
            y="hallucination_rate",
            title="Hallucination Rate by Top TF-IDF Keywords",
            xlabel="Keyword",
            ylabel="Hallucination Rate",
            output=out_keywords,
        )

    # Heatmap: keep it interpretable and robust
    actual_feats = ["prompt_length", "response_length"] + [c for c in df.columns if c.startswith("contains_")]
    actual_feats = [c for c in actual_feats if c in df.columns]

    if actual_feats:
        corr = df[actual_feats + ["hallucinated"]].corr(numeric_only=True)
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation of Features and Hallucination")
        plt.tight_layout()
        plt.savefig(out_heatmap)
        plt.close()
    else:
        logging.info("No interpretable features found for heatmap; skipping.")

    # Generate index.html (after graphs)
    index_fp = os.path.join(args.output_dir, "index.html")
    with open(index_fp, "w", encoding="utf-8") as f:
        f.write(
            "<html><head><title>Hallucination Analysis</title>"
            "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'>"
            "</head><body class='m-4'>\n"
        )
        f.write("<h1>Processed Data</h1>\n")
        rel_path = os.path.relpath(proc_html, args.output_dir)
        f.write(f"<iframe src='{rel_path}' width='100%' height='400'></iframe>\n")

        graphs = [
            "hallucinations_by_template.png",
            "hallucinations_by_keywords.png",
            "feature_correlation_heatmap.png",
        ]
        for g in graphs:
            gfp = os.path.join(args.output_dir, g)
            if os.path.exists(gfp):
                f.write(f"<h2>{g.replace('_', ' ').replace('.png', '').title()}</h2>\n")
                f.write(f"<img src='{g}' class='img-fluid mb-4'/><br/>\n")

        f.write("</body></html>\n")

    logging.info(f"Generated webpage at {index_fp}")
    logging.info(f"Graphs and webpage written to {args.output_dir}")


if __name__ == "__main__":
    main()
