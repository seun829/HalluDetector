#!/usr/bin/env python3
import os, glob, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# — Determine project root & data directory —
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data', 'processed')

# — Gather all labeled CSVs —
csv_paths = glob.glob(os.path.join(DATA_DIR, 'responses_*_labeled.csv'))
dfs = [pd.read_csv(p) for p in csv_paths if os.path.getsize(p) > 0]
df = pd.concat(dfs, ignore_index=True)

# — Create binary label: hallucination if response ≠ correct_answer —
df['label'] = (
    df['model_response'].astype(str).str.strip()
    != df['correct_answer'].astype(str).str.strip()
).astype(int)

# — Vectorize prompts & train —
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['prompt'])
y = df['label']

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# — Save model + vectorizer —
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'vectorizer': vectorizer, 'clf': clf}, f)

print(f'✅ Training complete. Model saved to {MODEL_PATH}')
