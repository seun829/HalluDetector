#!/usr/bin/env python3
import argparse
import os
import logging
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from pandas.errors import EmptyDataError, ParserError
from sklearn.feature_extraction.text import TfidfVectorizer

# configure logging
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


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
        required = {"prompt", "model_response", "correct_answer"}
        if not required.issubset(df.columns):
            logging.warning(f"Skipping {fp}: missing columns {required - set(df.columns)}")
            continue
        df["source"] = os.path.basename(fp).replace("_labeled.csv", "")
        df["prompt_length"]   = df["prompt"].astype(str).apply(len)
        df["response_length"] = df["model_response"].astype(str).apply(len)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def compute_correctness(df):
    df["is_correct"] = df.apply(
        lambda r: int(
            str(r["model_response"]).strip().lower() ==
            str(r["correct_answer"]).strip().lower()
        ),
        axis=1
    )
    return df


def analyze_by_question_type(df):
    if "question_type" not in df.columns:
        return None
    g = df.groupby("question_type")["is_correct"].mean().reset_index()
    g.rename(columns={"is_correct":"accuracy"}, inplace=True)
    return g


def analyze_by_template(df):
    if "template" not in df.columns:
        return None
    g = df.groupby("template")["is_correct"].mean().reset_index()
    g.rename(columns={"is_correct":"accuracy"}, inplace=True)
    return g


def analyze_keywords_tfidf(df, top_n=10):
    corpus = df["prompt"].astype(str)
    vect = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
        max_features=top_n * 5
    )
    X = vect.fit_transform(corpus)
    features = vect.get_feature_names_out()
    is_hallu = df["is_correct"] == 0
    hallu_avg = X[is_hallu].mean(axis=0).A1
    pairs = list(zip(features, hallu_avg))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_n]
    kws = [w for w,_ in top]

    rows = []
    for kw in kws:
        mask = df["prompt"].str.contains(rf"\b{re.escape(kw)}\b", case=False, na=False)
        rows.append({
            "keyword": kw,
            "accuracy": df.loc[mask, "is_correct"].mean()
        })
    return pd.DataFrame(rows)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.fc(x)


def extract_features(df):
    feat_cols = [
        "prompt_length", "response_length"
    ] + [c for c in df.columns if c.startswith("contains_")]
    sub = df[feat_cols].fillna(0).astype(float)
    t = torch.tensor(sub.values, dtype=torch.float32)
    model = FeatureExtractor(t.shape[1])
    with torch.no_grad():
        feats = model(t).numpy()
    for i in range(feats.shape[1]):
        df[f"extracted_feature_{i}"] = feats[:, i]
    return df


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
    p = argparse.ArgumentParser(description="Visualize hallucination patterns.")
    p.add_argument(
        "--input-dir", "-i", required=True,
        help="Directory with labeled response CSVs"
    )
    p.add_argument(
        "--output-dir", "-o", required=True,
        help="Directory to write PNG graphs"
    )
    p.add_argument(
        "--top-keywords", type=int, default=10,
        help="Number of top keywords to extract"
    )
    args = p.parse_args()

    # gather CSVs
    files = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(".csv")
    ])
    df = load_data(files)
    if df.empty:
        logging.error("No valid data; exiting.")
        return

    df = compute_correctness(df)

    # by question type
    qta = analyze_by_question_type(df)
    if qta is not None:
        visualize_results(
            qta, "question_type", "accuracy",
            "Accuracy by Question Type", "Question Type", "Accuracy",
            output=os.path.join(args.output_dir, "accuracy_by_question_type.png")
        )

    # by template
    ta = analyze_by_template(df)
    if ta is not None:
        visualize_results(
            ta, "template", "accuracy",
            "Accuracy by Template", "Template", "Accuracy",
            output=os.path.join(args.output_dir, "accuracy_by_template.png")
        )

    # keyword TF-IDF
    kwa = analyze_keywords_tfidf(df, top_n=args.top_keywords)
    if not kwa.empty:
        visualize_results(
            kwa, "keyword", "accuracy",
            "Accuracy by Top TF-IDF Keywords", "Keyword", "Accuracy",
            output=os.path.join(args.output_dir, "accuracy_by_keywords.png")
        )

    # feature correlation heatmap
    df_feat = extract_features(df)
    feats = [c for c in df_feat.columns if c.startswith("extracted_feature_")]
    corr = df_feat[feats + ["is_correct"]].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation of Extracted Features and Accuracy")
    out = os.path.join(args.output_dir, "feature_correlation_heatmap.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

    logging.info(f"✅ Graphs written to {args.output_dir}")


if __name__ == "__main__":
    main()
