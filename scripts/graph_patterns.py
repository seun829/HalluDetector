#!/usr/bin/env python
"""
Analyze hallucination patterns in GPT/Llama responses and visualize results.
Uses PyTorch for feature extraction and TF-IDF for robust keyword extraction and creative graphing.
Automatically computes an 'is_correct' flag from model_response vs. correct_answer.
"""
import os
import argparse
import logging
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless-friendly backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from pandas.errors import EmptyDataError, ParserError
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(filepaths):
    dfs = []
    for filepath in filepaths:
        if not os.path.exists(filepath):
            logging.warning(f"File {filepath} does not exist. Skipping.")
            continue
        try:
            df = pd.read_csv(filepath)
        except (EmptyDataError, ParserError) as e:
            logging.warning(f"File {filepath} is malformed or empty: {e}. Skipping.")
            continue
        if df.empty:
            logging.warning(f"File {filepath} is empty. Skipping.")
            continue
        required = {'prompt', 'model_response', 'correct_answer'}
        missing = required - set(df.columns)
        if missing:
            logging.warning(f"File {filepath} missing required columns {missing}. Skipping.")
            continue
        df['source'] = os.path.basename(filepath).replace('_labeled.csv', '')
        df['prompt_length'] = df['prompt'].astype(str).apply(len)
        df['response_length'] = df['model_response'].astype(str).apply(len)
        dfs.append(df)
    if not dfs:
        logging.error("No valid input files loaded.")
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def compute_correctness(df):
    df['is_correct'] = df.apply(
        lambda r: int(
            str(r['model_response']).strip().lower() ==
            str(r['correct_answer']).strip().lower()
        ),
        axis=1
    )
    return df


def analyze_by_question_type(df):
    if 'question_type' not in df.columns:
        logging.info("No 'question_type' column found. Skipping question_type analysis.")
        return None
    grouped = df.groupby('question_type')['is_correct'].mean().reset_index()
    grouped.rename(columns={'is_correct': 'accuracy'}, inplace=True)
    return grouped


def analyze_by_template(df):
    if 'template' not in df.columns:
        logging.info("No 'template' column found. Skipping template analysis.")
        return None
    grouped = df.groupby('template')['is_correct'].mean().reset_index()
    grouped.rename(columns={'is_correct': 'accuracy'}, inplace=True)
    return grouped


def analyze_keywords_tfidf(df, top_n=10):
    """
    Extract top N keywords from prompts using TF-IDF over all responses,
    and compute accuracy per keyword.
    """
    corpus = df['prompt'].astype(str)
    vectorizer = TfidfVectorizer(
        stop_words='english',
        token_pattern=r"\b[a-zA-Z]{2,}\b",
        max_features=top_n*5
    )
    X = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names_out()
    # Sum TF-IDF scores over hallucinated vs. correct prompts
    is_hallu = df['is_correct'] == 0
    # Compute average tf-idf for each term in hallucinated responses
    hallu_avg = X[is_hallu].mean(axis=0).A1
    # Pair and sort by descending avg
    pairs = list(zip(features, hallu_avg))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_n]
    keywords = [w for w, _ in top]
    # Compute accuracy for each keyword
    results = []
    for kw in keywords:
        mask = df['prompt'].str.contains(rf"\b{re.escape(kw)}\b", case=False, na=False)
        acc = df.loc[mask, 'is_correct'].mean()
        results.append({'keyword': kw, 'accuracy': acc})
    return pd.DataFrame(results)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.fc(x)


def extract_features(df):
    feature_cols = ['prompt_length', 'response_length'] + [c for c in df.columns if c.startswith('contains_')]
    subdf = df[feature_cols].fillna(0).astype(float)
    tensor = torch.tensor(subdf.values, dtype=torch.float32)
    model = FeatureExtractor(tensor.shape[1])
    with torch.no_grad():
        feats = model(tensor).numpy()
    for i in range(feats.shape[1]):
        df[f'extracted_feature_{i}'] = feats[:, i]
    return df


def visualize_results(df, x_col, y_col, title, xlabel, ylabel, output=None):
    if df is None or df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x_col, y=y_col, data=df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if output:
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        plt.savefig(output)
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize hallucination patterns.")
    parser.add_argument(
        "--response-files", "-i", nargs='+',
        default=[
            "data/processed/responses_easy_labeled.csv",
            "data/processed/responses_auto-generated_labeled.csv",
            "data/processed/responses_hard_labeled.csv"
        ],
        help="Labeled response CSV files."
    )
    parser.add_argument(
        "--top-keywords", type=int, default=10,
        help="Number of top keywords to extract with TF-IDF."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save visualizations. If omitted, shows plots interactively."
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    df = load_data(args.response_files)
    if df.empty:
        return
    df = compute_correctness(df)

    # Question type and template analyses
    qta = analyze_by_question_type(df)
    if qta is not None:
        logging.info("Accuracy by question type:\n%s", qta)
        visualize_results(
            qta, 'question_type', 'accuracy',
            'Accuracy by Question Type', 'Question Type', 'Accuracy',
            output=os.path.join(args.output_dir or '', 'accuracy_by_question_type.png') if args.output_dir else None
        )
    ta = analyze_by_template(df)
    if ta is not None:
        logging.info("Accuracy by template:\n%s", ta)
        visualize_results(
            ta, 'template', 'accuracy',
            'Accuracy by Template', 'Template', 'Accuracy',
            output=os.path.join(args.output_dir or '', 'accuracy_by_template.png') if args.output_dir else None
        )

    # TF-IDF based keyword analysis
    kwa = analyze_keywords_tfidf(df, top_n=args.top_keywords)
    if not kwa.empty:
        logging.info("Keyword accuracy:\n%s", kwa)
        visualize_results(
            kwa, 'keyword', 'accuracy',
            'Accuracy by Top TF-IDF Keywords', 'Keyword', 'Accuracy',
            output=os.path.join(args.output_dir or '', 'accuracy_by_keywords.png') if args.output_dir else None
        )

    # Feature extraction and correlation heatmap
    df_feat = extract_features(df)
    feats = [c for c in df_feat.columns if c.startswith('extracted_feature_')]
    corr = df_feat[feats + ['is_correct']].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation of Extracted Features and Accuracy')
    if args.output_dir:
        path = os.path.join(args.output_dir, 'feature_correlation_heatmap.png')
        os.makedirs(args.output_dir, exist_ok=True)
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    main()
