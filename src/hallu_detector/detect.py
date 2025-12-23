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
# NOTE: Soft fallbacks are disabled. Missing deps/models now raise at runtime.
_TORCH_IMPORT_ERROR = None
_ST_IMPORT_ERROR = None
_TR_IMPORT_ERROR = None

try:
    import torch
except Exception as e:
    torch = None
    _TORCH_IMPORT_ERROR = e

try:
    from sentence_transformers import SentenceTransformer, util
except Exception as e:
    SentenceTransformer, util = None, None
    _ST_IMPORT_ERROR = e

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception as e:
    AutoTokenizer, AutoModelForSequenceClassification = None, None
    _TR_IMPORT_ERROR = e

# ---------------------------------------------------------------------
# Lazy singletons for models (thread-safe)
# ---------------------------------------------------------------------
_ST = None              # SentenceTransformer encoder
_NLI_MODEL = None       # MNLI model
_NLI_TOK = None         # MNLI tokenizer
_NLI_LABEL2ID = None    # label mapping
_DEVICE = None          # torch.device
_LOCK = threading.Lock()


def _ensure_models():
    """
    Lazily load the embedding model and NLI pipeline.
    Works on CPU by default; uses GPU if available.
    Soft-fails if libraries are missing.
    """
    global _ST, _NLI_MODEL, _NLI_TOK, _NLI_LABEL2ID, _DEVICE
    if _ST is not None and _NLI_MODEL is not None and _NLI_TOK is not None:
        return
    with _LOCK:
        if _ST is not None and _NLI_MODEL is not None and _NLI_TOK is not None:
            return

        # Fail fast if deps are missing (no soft fallback).
        if torch is None:
            raise RuntimeError(f"Missing dependency: torch. Import error: {_TORCH_IMPORT_ERROR!r}")
        if SentenceTransformer is None or util is None:
            raise RuntimeError(f"Missing dependency: sentence-transformers. Import error: {_ST_IMPORT_ERROR!r}")
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError(f"Missing dependency: transformers. Import error: {_TR_IMPORT_ERROR!r}")

        _DEVICE = torch.device("cuda:0" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu")

        if _ST is None:
            try:
                _ST = SentenceTransformer("all-MiniLM-L6-v2", device=str(_DEVICE))
            except Exception as e:
                raise RuntimeError(f"Failed to load SentenceTransformer(all-MiniLM-L6-v2): {e!r}") from e

        if _NLI_MODEL is None or _NLI_TOK is None:
            try:
                name = "roberta-large-mnli"
                _NLI_TOK = AutoTokenizer.from_pretrained(name)
                _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(name)
                _NLI_MODEL.to(_DEVICE)
                _NLI_MODEL.eval()
                _NLI_LABEL2ID = getattr(_NLI_MODEL.config, "label2id", None) or {
                    "CONTRADICTION": 0,
                    "NEUTRAL": 1,
                    "ENTAILMENT": 2,
                }
            except Exception as e:
                raise RuntimeError(f"Failed to load MNLI model(roberta-large-mnli): {e!r}") from e

        # Warmup: catches device/cache issues early and avoids first-call latency spikes.
        try:
            _ = _ST.encode("warmup", convert_to_tensor=True, normalize_embeddings=True)
            toks = _NLI_TOK("warmup", "warmup", return_tensors="pt", truncation=True, max_length=256)
            toks = {k: v.to(_DEVICE) for k, v in toks.items()}
            with torch.no_grad():
                _ = _NLI_MODEL(**toks).logits
        except Exception as e:
            raise RuntimeError(f"Model warmup failed: {e!r}") from e

        import logging
        logging.info(
            "Hallucination detector models loaded. device=%s ST=%s NLI=%s",
            _DEVICE,
            "all-MiniLM-L6-v2",
            "roberta-large-mnli",
        )


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
    try:
        e1 = _ST.encode(_normalize_text(a), convert_to_tensor=True, normalize_embeddings=True)
        e2 = _ST.encode(_normalize_text(b), convert_to_tensor=True, normalize_embeddings=True)
        return float(util.cos_sim(e1, e2).item())
    except Exception as e:
        raise RuntimeError(f"Embedding similarity failed: {e!r}") from e

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
    try:
        toks = _NLI_TOK(premise or "", hypothesis or "", return_tensors="pt",
                        truncation=True, max_length=256)
        toks = {k: v.to(_DEVICE) for k, v in toks.items()}
        with torch.no_grad():
            logits = _NLI_MODEL(**toks).logits.squeeze(0)  # [3]
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        lid = _NLI_LABEL2ID or {"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2}
        c = probs[int(lid.get("CONTRADICTION", 0))]
        n = probs[int(lid.get("NEUTRAL", 1))]
        e = probs[int(lid.get("ENTAILMENT", 2))]

        return {"entail": float(e), "neutral": float(n), "contradict": float(c)}
    except Exception as e:
        raise RuntimeError(f"NLI inference failed: {e!r}") from e


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
    # Hardened: only allow when one side is short (to avoid "contains token" false-safes).
    corr_toks = _tokenize(corr_raw)
    ans_toks = _tokenize(ans_raw)
    ans_pad = f" {ans} "
    corr_pad = f" {corr} "

    short_corr = len(corr_toks) <= 4
    short_ans = len(ans_toks) <= 4

    if (short_corr and corr_pad in ans_pad) or (short_ans and ans_pad in corr_pad):
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
