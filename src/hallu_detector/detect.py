# src/hallu_detector/detect.py
# -*- coding: utf-8 -*-

"""
Hallucination detection (research-grade, robust)

Public API (unchanged):
- detect_details(answer, correct_answer, th=Thresholds()) -> dict
- is_hallucinated(answer, correct_answer, sim_lo=..., entail_lo=..., contr_hi=...) -> bool

Design highlights
- Text normalization: Unicode NFKC, whitespace/punct cleanup.
- Semantic similarity: SentenceTransformers (all-MiniLM-L6-v2) cosine similarity, with lexical Jaccard.
- Bidirectional NLI: roberta-large-mnli; robust output parsing across transformers versions.
- Numeric guard: unit-aware parsing (%, currency symbol presence), year exactness, float tolerance.
- Negation guard: XOR of negation presence flags contradiction risk.
- Thresholds in a dataclass for reproducible tuning; rationale lists per-rule contributions.

This file is self-contained and fails "softly": if any model is unavailable,
it returns conservative defaults rather than raising, so experiments/logging can proceed.
"""

from __future__ import annotations

import math
import re
import string
import threading
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

# --- Optional heavy deps (graceful degradation) -------------------------------
try:
    import torch
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer, util = None, None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

# ---------------------------------------------------------------------
# Lazy singletons for models (thread-safe)
# ---------------------------------------------------------------------
_ST = None        # SentenceTransformer encoder
_NLI = None       # MNLI text-classification pipeline
_LOCK = threading.Lock()


def _ensure_models():
    """
    Lazily load the embedding model and NLI pipeline.
    Works on CPU by default; uses GPU if available.
    Soft-fails if libraries are missing.
    """
    global _ST, _NLI
    if _ST is not None and _NLI is not None:
        return
    with _LOCK:
        if _ST is None and SentenceTransformer is not None:
            try:
                _ST = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                _ST = None
        if _NLI is None and pipeline is not None:
            try:
                device = 0 if (torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else -1
                _NLI = pipeline(
                    task="text-classification",
                    model="roberta-large-mnli",
                    device=device,
                )
            except Exception:
                _NLI = None


# ---------------------------------------------------------------------
# Normalization & utilities
# ---------------------------------------------------------------------
try:
    import unicodedata
except Exception:
    unicodedata = None

_PUNCT_TABLE = str.maketrans({c: " " for c in (set(string.punctuation) - set("%$-"))})
# Keep %, $, and '-' (for negatives and hyphenated tokens)

def _nfkc(s: str) -> str:
    if unicodedata is None:
        return s
    try:
        return unicodedata.normalize("NFKC", s)
    except Exception:
        return s

def _normalize_text(s: str) -> str:
    s = _nfkc((s or "").strip())
    # collapse weird whitespace
    s = " ".join(s.split())
    # harmonize quotes/dashes then light de-punct except %, $, -
    s = s.translate(_PUNCT_TABLE)
    s = " ".join(s.split()).lower()
    return s

def _tokenize(s: str) -> List[str]:
    return [t for t in _normalize_text(s).split() if t]

def _jaccard(a: str, b: str) -> float:
    A, B = set(_tokenize(a)), set(_tokenize(b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

# --- Numbers & units ----------------------------------------------------------
_NUM_RE = re.compile(r"(?P<sign>-)?(?P<num>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_PERCENT_RE = re.compile(r"%")
_CURRENCY_RE = re.compile(r"[$€£¥]")

def _numbers(text: str) -> list[float]:
    return [float(m.group(0)) for m in _NUM_RE.finditer(text or "")]

def _numbers_with_meta(text: str):
    """
    Extract numbers plus nearby unit hints: percent/currency.
    Returns list of dicts: {'value': float, 'is_percent': bool, 'is_currency': bool}
    """
    out = []
    t = text or ""
    for m in _NUM_RE.finditer(t):
        start, end = m.span()
        window = t[max(0, start-1):min(len(t), end+1)]
        is_percent = bool(_PERCENT_RE.search(window))
        # scan a slightly larger window for currency symbol
        win2 = t[max(0, start-2):min(len(t), end+2)]
        is_curr = bool(_CURRENCY_RE.search(win2))
        out.append({"value": float(m.group(0)), "is_percent": is_percent, "is_currency": is_curr})
    return out

def _is_year(x: float) -> bool:
    # Treat 1000–2100 as likely years; adjust as needed
    return float(x).is_integer() and 1000 <= int(x) <= 2100

def _close(a: float, b: float, rel: float = 0.01, abs_: float = 1e-9) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))

def _numeric_mismatch(a: str, c: str, rel_tol: float = 0.02) -> Dict[str, Any]:
    """
    Strong numeric guard. Returns a dict with details, including a boolean 'mismatch'.
    Rules:
      - If both sides have no numbers -> no mismatch.
      - If counts differ -> mismatch.
      - Year tokens must match exactly (0 tolerance).
      - Percent-marked numbers must compare against percent-marked.
      - Currency-marked numbers must compare against currency-marked.
      - Floats use relative tolerance rel_tol (default 2%).
    """
    A, C = _numbers_with_meta(a), _numbers_with_meta(c)
    if not A and not C:
        return {"mismatch": False, "reason": "no_numbers"}

    if len(A) != len(C):
        return {"mismatch": True, "reason": "count_mismatch", "counts": (len(A), len(C))}

    A_sorted = sorted(A, key=lambda d: d["value"])
    C_sorted = sorted(C, key=lambda d: d["value"])

    for i, (x, y) in enumerate(zip(A_sorted, C_sorted)):
        vx, vy = x["value"], y["value"]
        # unit coherence
        if x["is_percent"] != y["is_percent"]:
            return {"mismatch": True, "reason": f"percent_unit_mismatch@{i}", "pair": (vx, vy)}
        if x["is_currency"] != y["is_currency"]:
            return {"mismatch": True, "reason": f"currency_unit_mismatch@{i}", "pair": (vx, vy)}
        # year strictness
        if _is_year(vx) or _is_year(vy):
            if int(round(vx)) != int(round(vy)):
                return {"mismatch": True, "reason": f"year_mismatch@{i}", "pair": (vx, vy)}
            else:
                continue
        # tolerant floats/ints
        if not _close(vx, vy, rel=rel_tol):
            return {"mismatch": True, "reason": f"value_mismatch@{i}", "pair": (vx, vy), "rel_tol": rel_tol}

    return {"mismatch": False, "reason": "values_match"}

# --- Negation heuristic -------------------------------------------------------
_NEG_TOKENS = {"no", "not", "never", "none", "nothing", "nobody", "nowhere", "neither", "nor", "without"}

def _has_negation(s: str) -> bool:
    toks = set(_tokenize(s))
    if any(t in toks for t in _NEG_TOKENS):
        return True
    # basic "n't" detection
    return "n't" in _normalize_text(s)

def _negation_mismatch(a: str, b: str) -> bool:
    return _has_negation(a) ^ _has_negation(b)

# --- Semantic similarity ------------------------------------------------------
def _semantic_similarity(a: str, b: str) -> float:
    """
    Cosine similarity between normalized sentence embeddings.
    Soft-fails to lexical Jaccard if ST is unavailable.
    """
    _ensure_models()
    if _ST is None or util is None:
        return _jaccard(a, b)  # graceful degradation
    try:
        e1 = _ST.encode(_normalize_text(a), convert_to_tensor=True, normalize_embeddings=True)
        e2 = _ST.encode(_normalize_text(b), convert_to_tensor=True, normalize_embeddings=True)
        return float(util.cos_sim(e1, e2).item())
    except Exception:
        return _jaccard(a, b)

# --- NLI helpers --------------------------------------------------------------
def _map_mnli_label(label: str) -> str:
    u = (label or "").upper()
    if u in ("ENTAILMENT", "ENTAIL", "SUPPORTS"):
        return "ENTAILMENT"
    if u in ("CONTRADICTION", "CONTRA", "REFUTES"):
        return "CONTRADICTION"
    if u == "NEUTRAL":
        return "NEUTRAL"
    if u.startswith("LABEL_"):
        try:
            idx = int(u.split("_")[-1])
        except Exception:
            return "NEUTRAL"
        return ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"][max(0, min(2, idx))]
    return "NEUTRAL"

def _softmax(xs):
    m = max(xs) if xs else 0.0
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

def _nli_probs(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Robustly compute MNLI probabilities for (premise -> hypothesis).
    Returns dict with keys: 'entail', 'neutral', 'contradict'.
    Compatible with multiple transformers versions/output shapes.
    Soft-fails to neutral if pipeline is unavailable.
    """
    _ensure_models()
    if _NLI is None:
        return {"entail": 0.0, "neutral": 1.0, "contradict": 0.0}

    inputs = {"text": (premise or ""), "text_pair": (hypothesis or "")}

    # Try modern call first (top_k=None + padding/truncation to avoid tensor shape errors)
    try:
        out = _NLI(inputs, top_k=None, padding=True, truncation=True, max_length=256)
    except TypeError:
        try:
            out = _NLI(inputs, padding=True, truncation=True, max_length=256)
        except TypeError:
            out = _NLI(inputs)
    except Exception:
        return {"entail": 0.0, "neutral": 1.0, "contradict": 0.0}

    # Normalize to a flat list of dicts {label, score} or logits
    if isinstance(out, dict):
        arr = [out]
    elif isinstance(out, list):
        arr = out[0] if out and isinstance(out[0], list) else out
    else:
        arr = []

    probs = {"ENTAILMENT": 0.0, "NEUTRAL": 0.0, "CONTRADICTION": 0.0}
    bucket = {"ENTAILMENT": [], "NEUTRAL": [], "CONTRADICTION": []}

    for item in arr:
        if not isinstance(item, dict):
            continue
        lab = _map_mnli_label(item.get("label", ""))
        # Some versions surface "score" as post-softmax; others surface "logits"
        if "logits" in item and isinstance(item["logits"], (list, tuple)):
            # logits ordering is typically [CONTRADICTION, NEUTRAL, ENTAILMENT]
            logits = item["logits"]
            if len(logits) == 3:
                c, n, e = _softmax(logits)
                bucket["CONTRADICTION"].append(c)
                bucket["NEUTRAL"].append(n)
                bucket["ENTAILMENT"].append(e)
                continue
        try:
            sc = float(item.get("score", 0.0))
        except Exception:
            sc = 0.0
        if lab in bucket:
            bucket[lab].append(sc)

    # Aggregate by max (robust to duplicate label entries)
    for k in probs:
        probs[k] = max(bucket[k]) if bucket[k] else probs[k]

    return {
        "entail": probs["ENTAILMENT"],
        "neutral": probs["NEUTRAL"],
        "contradict": probs["CONTRADICTION"],
    }


# ---------------------------------------------------------------------
# Thresholds & core logic
# ---------------------------------------------------------------------
@dataclass
class Thresholds:
    # Tune these on a dev set; defaults are conservative but effective.
    sim_lo: float = 0.78        # below → likely hallucination (if entailment also low)
    sim_hi: float = 0.90        # above + good entailment → safe
    entail_lo: float = 0.60     # min(A→C, C→A) must exceed this to be safe
    contr_hi: float = 0.50      # max contradiction across directions to flag
    abstain_band: float = 0.05  # near-threshold region (logged in rationale)
    numeric_rel_tol: float = 0.02
    numeric_strict: bool = True
    use_negation_guard: bool = True
    jaccard_lo: float = 0.20    # lexical floor; very low overlap suggests drift

def detect_details(answer: str, correct_answer: str, th: Thresholds = Thresholds()) -> Dict[str, Any]:
    """
    Return a rich dictionary describing whether the model answer is hallucinated
    relative to the provided correct_answer, with scores, features and rationale.
    """
    ans_raw = (answer or "").strip()
    corr_raw = (correct_answer or "").strip()

    if not corr_raw:
        return {"hallucinated": False, "reason": "no_correct_answer_provided"}
    if not ans_raw:
        return {"hallucinated": True, "reason": "empty_answer"}

    # Quick normalized forms + lexical signals
    ans = _normalize_text(ans_raw)
    corr = _normalize_text(corr_raw)

    # Substring shortcut (case-insensitive, normalized)
    if corr in ans or ans in corr:
        num = _numeric_mismatch(ans_raw, corr_raw, rel_tol=th.numeric_rel_tol) if th.numeric_strict else {"mismatch": False}
        return {
            "hallucinated": bool(num.get("mismatch", False)),
            "reason": "substring_match_numeric_mismatch" if num.get("mismatch", False) else "substring_match",
            "scores": {
                "substring": 1.0,
                "numeric": num,
            },
            "features": {"ans": ans, "corr": corr},
        }

    # Scores & features
    sim = _semantic_similarity(ans_raw, corr_raw)
    jac = _jaccard(ans_raw, corr_raw)
    nli_ac = _nli_probs(ans_raw, corr_raw)   # answer -> correct
    nli_ca = _nli_probs(corr_raw, ans_raw)   # correct -> answer
    entail_min = min(nli_ac["entail"], nli_ca["entail"])
    contr_max  = max(nli_ac["contradict"], nli_ca["contradict"])
    num = _numeric_mismatch(ans_raw, corr_raw, rel_tol=th.numeric_rel_tol) if th.numeric_strict else {"mismatch": False, "reason": "numeric_disabled"}
    neg_mis = _negation_mismatch(ans_raw, corr_raw) if th.use_negation_guard else False

    # Rule ensemble (transparent & tunable)
    rationale = []
    decision: bool | None = None

    # Hard red flags
    if contr_max >= th.contr_hi:
        decision = True; rationale.append("high_contradiction")
    if th.numeric_strict and num.get("mismatch", False):
        decision = True; rationale.append(f"numeric_mismatch:{num.get('reason','')}")

    if th.use_negation_guard and neg_mis:
        # Don't immediately decide; mark as strong evidence unless entailment is high
        rationale.append("negation_mismatch")

    # Clear greens
    if decision is None:
        if sim >= th.sim_hi and entail_min >= th.entail_lo and contr_max < th.contr_hi and not num.get("mismatch", False):
            decision = False; rationale.append("high_sim_and_entail")

    # Clear reds
    if decision is None:
        if (sim < th.sim_lo and entail_min < th.entail_lo) or (jac < th.jaccard_lo and entail_min < th.entail_lo):
            decision = True; rationale.append("low_sim_or_jaccard_and_low_entail")

    # Ambiguous region: prefer to abstain in research, but return bool for product.
    if decision is None:
        near_entail = abs(entail_min - th.entail_lo) <= th.abstain_band
        near_sim = abs(sim - th.sim_lo) <= th.abstain_band
        if near_entail or near_sim:
            rationale.append("abstain_band")
        # Bias towards safety: treat as hallucinated if entailment below threshold
        decision = entail_min < th.entail_lo or (neg_mis and entail_min < (th.entail_lo + 0.05))

    return {
        "hallucinated": bool(decision),
        "scores": {
            "sim": sim,
            "lexical_jaccard": jac,
            "entail_min": entail_min,
            "contr_max": contr_max,
            "nli_ans_to_corr": nli_ac,
            "nli_corr_to_ans": nli_ca,
            "numeric": num,
            "negation_mismatch": float(bool(neg_mis)),
        },
        "thresholds": asdict(th),
        "features": {
            "answer_norm": ans,
            "correct_norm": corr,
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
