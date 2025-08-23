# src/hallu_detector/detect.py
# -*- coding: utf-8 -*-

"""
Hallucination detection (research-grade, robust)

This module exposes:
- detect_details(answer, correct_answer, th=Thresholds()) -> dict
    Returns a rich dictionary with scores (semantic similarity, bidirectional NLI,
    numeric mismatch) and a rationale string for analysis/papers.

- is_hallucinated(answer, correct_answer, sim_lo=..., entail_lo=..., contr_hi=...) -> bool
    Compact boolean for app usage; uses the same rules as detect_details.

Design notes
- Semantic similarity: SentenceTransformers (all-MiniLM-L6-v2) cosine similarity.
- Bidirectional NLI: roberta-large-mnli via HuggingFace `pipeline("text-classification")`,
  called with (premise, hypothesis) and robust parsing of outputs across transformers versions.
- Numeric guard: detects mismatched numeric values with relative tolerance.
- Thresholds are collected in a dataclass for easy tuning on a dev set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ---------------------------------------------------------------------
# Lazy singletons for models
# ---------------------------------------------------------------------
_ST = None        # SentenceTransformer encoder
_NLI = None       # MNLI text-classification pipeline


def _ensure_models():
    """
    Lazily load the embedding model and NLI pipeline.
    Works on CPU by default; uses GPU if available.
    """
    global _ST, _NLI
    if _ST is None:
        _ST = SentenceTransformer("all-MiniLM-L6-v2")
    if _NLI is None:
        device = 0 if torch.cuda.is_available() else -1
        # Do NOT set deprecated return_all_scores here; we'll request top_k=None at call.
        _NLI = pipeline(
            task="text-classification",
            model="roberta-large-mnli",
            device=device,
        )


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

def _numbers(text: str) -> list[float]:
    return [float(x) for x in _NUM_RE.findall(text or "")]

def _close(a: float, b: float, rel: float = 0.01, abs_: float = 1e-6) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))

def _semantic_similarity(a: str, b: str) -> float:
    """
    Cosine similarity between normalized sentence embeddings.
    """
    _ensure_models()
    try:
        e1 = _ST.encode(a or "", convert_to_tensor=True, normalize_embeddings=True)
        e2 = _ST.encode(b or "", convert_to_tensor=True, normalize_embeddings=True)
        return float(util.cos_sim(e1, e2).item())
    except Exception:
        # If something goes wrong, return a neutral mid-low similarity
        return 0.5

def _map_mnli_label(label: str) -> str:
    """
    Normalize various possible label strings to standard MNLI classes.
    """
    u = (label or "").upper()
    if u in ("ENTAILMENT", "ENTAIL", "SUPPORTS"):
        return "ENTAILMENT"
    if u in ("CONTRADICTION", "CONTRA", "REFUTES"):
        return "CONTRADICTION"
    if u == "NEUTRAL":
        return "NEUTRAL"
    if u.startswith("LABEL_"):
        # roberta-large-mnli: LABEL_0=CONTRADICTION, LABEL_1=NEUTRAL, LABEL_2=ENTAILMENT
        try:
            idx = int(u.split("_")[-1])
        except Exception:
            return "NEUTRAL"
        return ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"][max(0, min(2, idx))]
    return "NEUTRAL"

def _nli_probs(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Robustly compute MNLI probabilities for (premise -> hypothesis).
    Returns dict with keys: 'entail', 'neutral', 'contradict'.
    Compatible with multiple transformers versions/output shapes.
    """
    _ensure_models()
    inputs = {"text": (premise or ""), "text_pair": (hypothesis or "")}

    # Try modern call first (top_k=None + padding/truncation to avoid tensor shape errors)
    try:
        out = _NLI(inputs, top_k=None, padding=True, truncation=True, max_length=256)
    except TypeError:
        # Older transformers: no top_k in __call__
        try:
            out = _NLI(inputs, padding=True, truncation=True, max_length=256)
        except TypeError:
            # Oldest signatures: plain call
            out = _NLI(inputs)

    # Normalize to a flat list of dicts {label, score}
    if isinstance(out, dict):
        arr = [out]
    elif isinstance(out, list):
        arr = out[0] if out and isinstance(out[0], list) else out
    else:
        arr = []

    probs = {"ENTAILMENT": 0.0, "NEUTRAL": 0.0, "CONTRADICTION": 0.0}
    for item in arr:
        if not isinstance(item, dict):
            continue
        lab = _map_mnli_label(item.get("label", ""))
        try:
            sc = float(item.get("score", 0.0))
        except Exception:
            sc = 0.0
        if lab in probs:
            # take the maximum score per label in case of duplicates
            probs[lab] = max(probs[lab], sc)

    return {
        "entail": probs["ENTAILMENT"],
        "neutral": probs["NEUTRAL"],
        "contradict": probs["CONTRADICTION"],
    }

def _numeric_mismatch(a: str, c: str, rel_tol: float = 0.02) -> bool:
    """
    Return True if numbers detected in the two strings do not match within tolerance.
    Detects classic hallucinations where the model paraphrases but changes numbers/dates.
    """
    na, nc = _numbers(a), _numbers(c)
    if not na and not nc:
        return False
    if len(na) != len(nc):
        return True
    for x, y in zip(sorted(na), sorted(nc)):
        if not _close(x, y, rel=rel_tol):
            return True
    return False


# ---------------------------------------------------------------------
# Thresholds & core logic
# ---------------------------------------------------------------------
@dataclass
class Thresholds:
    # Tune these on a dev set for your paper; these are safe defaults.
    sim_lo: float = 0.78       # below → likely hallucination (if entailment also low)
    sim_hi: float = 0.90       # above + good entailment → safe
    entail_lo: float = 0.60    # min(A→C, C→A) must exceed this to be safe
    contr_hi: float = 0.50     # max contradiction across directions to flag
    abstain_band: float = 0.05 # near-threshold region (for research reporting)
    numeric_strict: bool = True


def detect_details(answer: str, correct_answer: str, th: Thresholds = Thresholds()) -> Dict[str, Any]:
    """
    Return a rich dictionary describing whether the model answer is hallucinated
    relative to the provided correct_answer, with scores and rationale.
    """
    ans = (answer or "").strip()
    corr = (correct_answer or "").strip()

    # Trivial cases
    if not corr:
        return {"hallucinated": False, "reason": "no_correct_answer_provided"}
    if not ans:
        return {"hallucinated": True, "reason": "empty_answer"}

    # Quick containment shortcut
    if corr.lower() in ans.lower() or ans.lower() in corr.lower():
        numeric_bad = _numeric_mismatch(ans, corr) if th.numeric_strict else False
        return {
            "hallucinated": bool(numeric_bad),
            "reason": "substring_match_numeric_mismatch" if numeric_bad else "substring_match",
            "scores": {"substring": 1.0, "numeric_mismatch": float(numeric_bad)},
        }

    # Scores
    sim = _semantic_similarity(ans, corr)
    nli_ac = _nli_probs(ans, corr)   # answer -> correct
    nli_ca = _nli_probs(corr, ans)   # correct -> answer
    entail_min = min(nli_ac["entail"], nli_ca["entail"])
    contr_max  = max(nli_ac["contradict"], nli_ca["contradict"])
    numeric_bad = _numeric_mismatch(ans, corr) if th.numeric_strict else False

    # Rule ensemble (transparent & tunable)
    decision = None
    rationale = []

    if contr_max >= th.contr_hi:
        decision = True; rationale.append("high_contradiction")
    if numeric_bad:
        decision = True; rationale.append("numeric_mismatch")

    if decision is None:
        if sim >= th.sim_hi and entail_min >= th.entail_lo:
            decision = False; rationale.append("high_sim_and_entail")
        elif sim < th.sim_lo and entail_min < th.entail_lo:
            decision = True; rationale.append("low_sim_and_entail")

    if decision is None:
        # Ambiguous region: prefer to abstain in research, but return bool here
        if abs(entail_min - th.entail_lo) <= th.abstain_band or abs(sim - th.sim_lo) <= th.abstain_band:
            rationale.append("abstain_band")
        decision = entail_min < th.entail_lo

    return {
        "hallucinated": bool(decision),
        "scores": {
            "sim": sim,
            "entail_min": entail_min,
            "contr_max": contr_max,
            "nli_ans_to_corr": nli_ac,
            "nli_corr_to_ans": nli_ca,
            "numeric_mismatch": float(numeric_bad),
        },
        "reason": "+".join(rationale) if rationale else ("rule_default_true" if decision else "rule_default_false"),
    }


def is_hallucinated(answer: str, correct_answer: str, sim_lo: float = 0.78, entail_lo: float = 0.60,
                    contr_hi: float = 0.50) -> bool:
    """
    Compact boolean wrapper for app usage.
    For research/plots, prefer detect_details(...).
    """
    th = Thresholds(sim_lo=sim_lo, entail_lo=entail_lo, contr_hi=contr_hi)
    return bool(detect_details(answer, correct_answer, th)["hallucinated"])
