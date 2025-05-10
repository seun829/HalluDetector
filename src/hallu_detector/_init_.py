"""
hallu_detector: A package for generating AI responses and detecting hallucinations.
"""

from .generate import generate_hf, generate_openai
from .detect import is_hallucinated
from .evaluate import compute_metrics

__all__ = [
    "generate_hf",
    "generate_openai",
    "is_hallucinated",
    "compute_metrics",
]

