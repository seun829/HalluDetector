#!/usr/bin/env python3
import os
import sys
import argparse
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict

# — Argument parsing —
parser = argparse.ArgumentParser(
    description="Fine-tune BERT to detect AI hallucination from prompts, with optimal hyperparameters"
)
parser.add_argument(
    "--output-dir", type=str, default="./results",
    help="Directory for checkpoints and logs"
)
parser.add_argument(
    "--model-dir", type=str, default="./bert_model",
    help="Directory to save fine-tuned model and tokenizer"
)
parser.add_argument(
    "--epochs", type=int, default=5,
    help="Number of training epochs (default: 5)"
)
parser.add_argument(
    "--batch-size", type=int, default=16,
    help="Batch size per device (default: 16)"
)
parser.add_argument(
    "--learning-rate", type=float, default=2e-5,
    help="Learning rate for optimizer (default: 2e-5)"
)
args = parser.parse_args()

# — Load HaluEval datasets —
QA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
GEN_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/general_data.json"

# QA: all hallucinations
qa_df = pd.DataFrame(requests.get(QA_URL).json())
qa_df = qa_df.rename(columns={'question': 'prompt'})
qa_df['label'] = 1

# General: mixed labels
gen_df = pd.DataFrame(requests.get(GEN_URL).json())
gen_df = gen_df.rename(columns={'user_query': 'prompt'})
gen_df['label'] = gen_df['hallucination_label'].map({'Yes': 1, 'No': 0})

# Combine, shuffle, and log
df = pd.concat([qa_df[['prompt','label']], gen_df[['prompt','label']]], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total examples: {len(df)}")
print("Label distribution:\n", df['label'].value_counts())

# Prompt-only text
df['text'] = df['prompt'].astype(str).str.strip()

# Stratified train-test split
train_df, test_df = train_test_split(
    df[['text','label']], test_size=0.2, random_state=42, stratify=df['label']
)
print(f"Training examples: {len(train_df)}, Test examples: {len(test_df)}")

# Create HuggingFace dataset
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df.reset_index(drop=True)),
    'test':  Dataset.from_pandas(test_df.reset_index(drop=True))
})

# Tokenization
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
def tokenize(batch): return tokenizer(batch['text'], truncation=True, padding=True)
tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.remove_columns(['text'])

# Model initialization
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Training arguments with early stopping
training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir=os.path.join(args.output_dir, 'logs'),
    logging_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

# Trainer with early stopping callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train and evaluate
trainer.train()
metrics = trainer.evaluate()
print("Final metrics:")
for k,v in metrics.items(): print(f"  {k}: {v:.4f}")

# Save model & tokenizer
model.save_pretrained(args.model_dir)
tokenizer.save_pretrained(args.model_dir)
print(f"Model saved to {args.model_dir}")
