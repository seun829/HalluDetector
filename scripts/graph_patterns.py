#!/usr/bin/env python
"""
Analyze hallucination patterns in GPT/Llama responses and visualize results.
Uses PyTorch for feature extraction and creative graphing.
Automatically computes an 'is_correct' flag from model_response vs. correct_answer.
"""
import os
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from pandas.errors import EmptyDataError, ParserError


def load_data(filepaths):
    dfs = []
    for filepath in filepaths:
        if not os.path.exists(filepath):
            logging.warning(f"File {filepath} does not exist. Skipping.")
            continue
        try:
            df = pd.read_csv(filepath)
        except (EmptyDataError, ParserError) as e:
            logging.warning(f"File {filepath} has no columns to parse or is malformed: {e}. Skipping.")
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


def analyze_by_keywords(df, keywords):
    if 'is_correct' not in df.columns:
        logging.info("No 'is_correct' column found. Skipping keyword analysis.")
        return pd.DataFrame()
    results = []
    for kw in keywords:
        col = f'contains_{kw}'
        df[col] = df['prompt'].astype(str).str.contains(kw, case=False, na=False)
        acc = df.loc[df[col], 'is_correct'].mean()
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
    """
    Use PyTorch to extract features from the data.
    """
    # Define feature columns and ensure numeric types
    feature_cols = ['prompt_length', 'response_length'] + [c for c in df.columns if c.startswith('contains_')]
    subdf = df[feature_cols].fillna(0)
    # Convert all to float (bools -> 0.0/1.0, ints -> floats)
    subdf = subdf.astype(float)
    data = subdf.values

    # Create tensor and pass through extractor
    tensor = torch.tensor(data, dtype=torch.float32)
    model = FeatureExtractor(tensor.shape[1])
    with torch.no_grad():
        feats = model(tensor).numpy()

    # Add extracted features back to DataFrame
    for i in range(feats.shape[1]):
        df[f'extracted_feature_{i}'] = feats[:, i]
    return df


def visualize_results(df, x_col, y_col, title, xlabel, ylabel):
    if df is None or df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x_col, y=y_col, data=df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
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
        help="Labeled response CSV files. If omitted, defaults are used."
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    df = load_data(args.response_files)
    if df.empty:
        return
    df = compute_correctness(df)

    qta = analyze_by_question_type(df)
    if qta is not None:
        print(qta)
        visualize_results(
            qta, 'question_type', 'accuracy',
            'Accuracy by Question Type', 'Question Type', 'Accuracy'
        )

    ta = analyze_by_template(df)
    if ta is not None:
        print(ta)
        visualize_results(
            ta, 'template', 'accuracy',
            'Accuracy by Template', 'Template', 'Accuracy'
        )

    keywords = ['capital', 'who', 'what', 'when', 'why', 'how']
    kwa = analyze_by_keywords(df, keywords)
    if not kwa.empty:
        print(kwa)
        visualize_results(
            kwa, 'keyword', 'accuracy',
            'Accuracy by Keywords', 'Keyword', 'Accuracy'
        )

    df = extract_features(df)
    feats = [c for c in df.columns if c.startswith('extracted_feature_')]
    corr = df[feats + ['is_correct']].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation of Extracted Features and Accuracy')
    plt.show()

if __name__ == '__main__':
    main()
