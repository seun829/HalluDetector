

from sentence_transformers import SentenceTransformer
from transformers import pipeline

print("Downloading embedding model (MiniLM)...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Downloading NLI model (RoBERTa MNLI)...")
pipeline("text-classification", model="roberta-large-mnli")

print("Done! Models downloaded.")