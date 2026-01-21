# ==================================================================================================
# detect.py
# ==================================================================================================
"""
Hallucination detection core.

This file is designed to be used in a research-grade evaluation pipeline.

Key goals
---------
1) No silent fallbacks for enabled features.
   If embeddings or NLI are enabled but unavailable, we raise a clear error.
2) Formatting-robust checks.
   Many gold answers are short; models often return the right fact with extra context
   (or slightly different formatting). We normalize and use multiple signals.
3) Hallucination policy alignment.
   We follow the repo's definition:

       hallucinated = True  iff the answer contains at least one fact that is false,
       fabricated, or unsupported by known reality (relative to the provided reference).

   With only (model_response, correct_answer) available, we use a conservative proxy:
   - DO NOT mark hallucination merely for incompleteness or reduced specificity.
   - DO mark hallucination when there is *positive evidence* of inconsistency, such as:
       * numeric mismatch (when a numeric gold is expected AND the response provides a conflicting number)
       * explicit negation of the gold answer
       * strong NLI contradiction (when available)
       * "no such / not available" claims that contradict a concrete gold
       * mutually-exclusive numeric alternatives ("X or Y") where an alternative differs from the gold
       * atomic entity/code mismatch (e.g., gold is a short code/name and response is a different short code/name)

Public API expected by the repo
-------------------------------
- Thresholds
- preflight_strict
- detect_details
- is_hallucinated
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import re
import unicodedata

__all__ = [
    "Thresholds",
    "preflight_strict",
    "detect_details",
    "is_hallucinated",
]


# ----------------------------
# Configuration / thresholds
# ----------------------------

@dataclass(frozen=True)
class Thresholds:
    # Dependency policy
    strict_dependencies: bool = True

    # Main signals
    allow_token_subsequence: bool = True
    enable_embeddings: bool = True
    enable_nli: bool = True

    # Token-subsequence gating (avoid pathological matches)
    subseq_min_gold_chars: int = 4
    subseq_max_answer_chars: int = 5000

    # Numeric tolerance
    # - integers (years/ids) are treated as exact-match by default via heuristics
    numeric_rel_tol: float = 5e-3   # 0.5%
    numeric_abs_tol: float = 5e-2   # 0.05

    # Embedding baseline
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_min_similarity: float = 0.72

    # Embedding gating (reduce short-text false positives)
    embed_require_shared_content_tokens: bool = True
    embed_min_shared_content_tokens: int = 1
    embed_disable_for_very_short_gold_tokens: int = 1  # if gold has <= this many tokens, don't treat embedding as strong support

    # NLI baseline
    nli_model_name: str = "roberta-large-mnli"
    nli_min_entailment: float = 0.65
    nli_max_contradiction: float = 0.50
    nli_min_tokens: int = 3

    # NLI contradiction veto (hallucination evidence)
    nli_contradiction_veto: float = 0.75
    nli_entailment_veto_floor: float = 0.30

    # Lexical baseline
    jaccard_min: float = 0.40

    # Weak-signal diagnostics (not used to force hallucination)
    weak_signals_required: int = 2

    # Disjunction / alternatives heuristic
    numeric_alt_markers: Tuple[str, ...] = ("or", "either")

    # Atomic entity/code mismatch heuristic
    atomic_gold_max_tokens: int = 2
    atomic_resp_max_tokens: int = 5
    atomic_gold_max_chars: int = 48

    # Unavailability claim cues (assert a world-fact like "no such ID exists")
    unavailability_cues: Tuple[str, ...] = (
        "no specific",
        "not available",
        "not publicly available",
        "no publicly available",
        "does not exist",
        "doesn't exist",
        "no such",
        "cannot be found",
        "can't be found",
        "not listed",
        "unlisted",
        "unavailable",
        "there is no",
        "there's no",
        "none exists",
    )


# ----------------------------
# Strict preflight
# ----------------------------

def preflight_strict(thresholds: Optional[Thresholds] = None) -> None:
    """
    Validate that required deps are importable for enabled features.

    Called by scripts/generate_responses.py as:
        preflight_strict(Thresholds())

    So this function MUST accept an optional Thresholds parameter.
    """
    th = thresholds or Thresholds()

    if th.enable_embeddings:
        _require_sentence_transformers()

    if th.enable_nli:
        _require_transformers_torch()


def _require_sentence_transformers() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Embeddings enabled, but 'sentence-transformers' is not available. "
            "Install it (and a backend like torch) or disable enable_embeddings."
        ) from e


def _require_transformers_torch() -> None:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "NLI enabled, but 'transformers' and/or 'torch' are not available. "
            "Install them or disable enable_nli."
        ) from e


# ----------------------------
# Normalization helpers
# ----------------------------

_WS_RE = re.compile(r"\s+")
# Keep -, ., /, ^, +, :, = for scientific notation, identifiers, codes (IPv6, key=value, etc.).
_PUNCT_RE = re.compile(r"[^\w\s\-\./\^\+:=]", re.UNICODE)

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "that", "this",
    "as", "at", "from", "into", "within", "over", "under", "between", "during",
    "does", "do", "did", "what", "which", "who", "whom", "when", "where", "why", "how",
}

_SUP_DIGITS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "−": "-",  # include unicode minus variants
})

def _strip_accents(s: str) -> str:
    # Normalize to NFKD to split accents into combining marks
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _normalize_math_unicode(s: str) -> str:
    """
    Normalize common unicode math typography that impacts numeric parsing:
    - superscript digits and signs
    - unicode minus variants
    - multiplication sign
    """
    if not s:
        return s
    # superscripts -> ascii
    s = s.translate(_SUP_DIGITS)
    # multiplication sign -> x
    s = s.replace("×", "x")
    # normalize some minus variants (already covered above, but keep safe)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    return s


# Detect patterns like "10⁻²¹" (after translate becomes "10-21", so we must interpret it as exponent)
_SUP_EXP_RE = re.compile(r"\b10(?P<exp>[+-]?\d{1,4})\b")

def _rewrite_implicit_superscript_powers(text: str) -> str:
    """
    Convert implicit superscript power patterns into explicit caret form.

    Examples:
      "10⁻²¹" -> "10^-21"
    After _normalize_math_unicode, that text becomes "10-21" (ambiguous), so we use a guard:
    - Only rewrite if original contained superscript characters OR we detect "10" followed immediately
      by digits with a leading sign in contexts that are likely exponents.

    Implementation strategy:
    - We perform a conservative rewrite on sequences that *originated* from superscripts by checking
      for any superscript chars in the original string before translation.
    """
    # This function expects to be called only when superscripts were present.
    # We rewrite "10-21" or "10+21" into "10^-21"/"10^+21" when it looks like exponent form.
    # To avoid rewriting normal "10-21" ranges, we only rewrite when there's no surrounding word chars.
    def repl(m: re.Match) -> str:
        exp = m.group("exp")
        return f"10^{exp}"
    return _SUP_EXP_RE.sub(repl, text)


def canonicalize(text: str, th: Optional[Thresholds] = None) -> str:
    """
    Make a string comparable across common formatting differences:
    - unicode normalization (NFKC)
    - lowercasing
    - accent stripping
    - normalize math unicode (superscripts, ×, unicode minus)
    - whitespace collapsing
    - remove most punctuation (keep -, ., /, ^, +, :, =)
    """
    if text is None:
        return ""

    t = str(text)
    t = unicodedata.normalize("NFKC", t)
    t = _normalize_math_unicode(t)
    t = t.strip().lower()
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    t = _strip_accents(t)
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def tokenize(text_can: str) -> List[str]:
    if not text_can:
        return []
    return [tok for tok in text_can.split(" ") if tok]


def _is_numeric_token(tok: str) -> bool:
    if not tok:
        return False
    try:
        float(tok)
        return True
    except Exception:
        return False


def content_tokens(tokens: List[str]) -> List[str]:
    """
    Tokens that carry semantic content for overlap gating:
    - remove stopwords
    - remove pure numeric tokens (numeric is handled separately)
    """
    out: List[str] = []
    for t in tokens:
        if not t or t in _STOPWORDS:
            continue
        if _is_numeric_token(t):
            continue
        if len(t) < 2:
            continue
        out.append(t)
    return out


def shared_content_token_count(resp_can: str, gold_can: str) -> int:
    r = set(content_tokens(tokenize(resp_can)))
    g = set(content_tokens(tokenize(gold_can)))
    return len(r & g)


# ----------------------------
# Lexical similarity
# ----------------------------

def jaccard_similarity(a_can: str, b_can: str) -> float:
    a_toks = set(tokenize(a_can))
    b_toks = set(tokenize(b_can))
    if not a_toks and not b_toks:
        return 1.0
    if not a_toks or not b_toks:
        return 0.0
    inter = len(a_toks & b_toks)
    union = len(a_toks | b_toks)
    return inter / union if union else 0.0


def token_subsequence_match(resp_tokens: List[str], gold_tokens: List[str]) -> bool:
    """Return True if gold tokens appear in resp tokens in order (not necessarily contiguous)."""
    if not gold_tokens:
        return False
    if not resp_tokens:
        return False
    j = 0
    for tok in resp_tokens:
        if tok == gold_tokens[j]:
            j += 1
            if j == len(gold_tokens):
                return True
    return False


def normalized_substring_match(resp_can: str, gold_can: str) -> bool:
    """Strong match for short codes/entities where tokenization can be brittle."""
    if not resp_can or not gold_can:
        return False
    return gold_can in resp_can


# ----------------------------
# Numeric / coordinate parsing
# ----------------------------

# 1e-21
_SCI_E_RE = re.compile(r"\b([+-]?\d+(?:\.\d+)?)\s*e\s*([+-]?\d+)\b")
# 10^-21, 10^(-21)
_SCI_10_RE = re.compile(r"\b10\s*\^\s*\(?\s*([+-]?\d+)\s*\)?\b")
# 1.2 x 10^-3, 1.2×10^-3, 1.2*10^-3
_SCI_MUL10_RE = re.compile(r"\b([+-]?\d+(?:\.\d+)?)\s*(?:x|\*|×)\s*10\s*\^\s*\(?\s*([+-]?\d+)\s*\)?\b")

_NUM_RE = re.compile(r"(?<!\w)([+-]?\d+(?:\.\d+)?)(?!\w)")

# Coordinates: "90 S", "90°S", "90 south", "90 degrees south"
_COORD_RE = re.compile(
    r"\b([+-]?\d+(?:\.\d+)?)\s*(?:deg(?:rees?)?|\°)?\s*(n|s|e|w|north|south|east|west)\b"
)


def _maybe_parse_scientific(text: str) -> Optional[float]:
    """
    Parse common scientific notation variants:
    - 1e-21
    - 10^-21, 10^(-21)
    - 1.2x10^-3, 1.2×10^-3
    - implicit-superscript powers rewritten to caret form (handled upstream)
    """
    if not text:
        return None

    t = canonicalize(text)

    m = _SCI_E_RE.search(t)
    if m:
        base = float(m.group(1))
        exp = int(m.group(2))
        return base * (10.0 ** exp)

    m = _SCI_MUL10_RE.search(t)
    if m:
        base = float(m.group(1))
        exp = int(m.group(2))
        return base * (10.0 ** exp)

    m = _SCI_10_RE.search(t)
    if m:
        exp = int(m.group(1))
        return 10.0 ** exp

    return None


def _parse_coord(text: str) -> Optional[float]:
    """
    Parse coordinates like:
      - "90 S" -> -90
      - "90 south" -> -90
      - "-90 degrees" (no direction) is NOT a coord parse; it's numeric.
    """
    if not text:
        return None
    t = canonicalize(text)
    m = _COORD_RE.search(t)
    if not m:
        return None
    val = float(m.group(1))
    dirc = m.group(2)
    if dirc in ("s", "south", "w", "west"):
        return -abs(val)
    return abs(val)


def _is_integerish(x: float, eps: float = 1e-9) -> bool:
    return abs(x - round(x)) <= eps


def _looks_like_numeric_answer(text_raw: str) -> bool:
    """
    Decide if a gold answer should be treated as numeric (so numeric matching applies).
    Reject code-like patterns (e.g., IPv6 ::1, IDs with letters, tokens with ':', '=').
    NOTE: coordinates are handled separately (before this check).
    """
    if text_raw is None:
        return False

    # normalize math unicode, but keep original structure
    raw = _normalize_math_unicode(str(text_raw))
    t = canonicalize(raw)
    if not t:
        return False
    if not any(ch.isdigit() for ch in t):
        return False

    # Allow digits, whitespace, . - + ^ / parentheses, and 'e' for sci notation.
    allowed_letters = set("e")
    for ch in t:
        if ch.isdigit() or ch.isspace():
            continue
        if ch in {".", "-", "+", "^", "(", ")", "/"}:
            continue
        if ch.isalpha() and ch in allowed_letters:
            continue
        # everything else (notably ":" and "=") => not numeric for our purposes
        return False
    return True


def _extract_numbers(text: str, th: Optional["Thresholds"] = None) -> List[float]:
    """
    Extract all parseable numbers from a string, preferring scientific notation as a single number.

    Robustness features:
    - supports "10^-21" and "1e-21"
    - supports unicode superscripts like "10⁻²¹" by rewriting to caret form before parsing
    """
    if text is None:
        return []

    raw = str(text)
    # If superscripts appear, normalize and rewrite implicit powers
    if any(ch in raw for ch in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"):
        raw2 = _normalize_math_unicode(raw)
        raw2 = _rewrite_implicit_superscript_powers(raw2)
    else:
        raw2 = _normalize_math_unicode(raw)

    t = canonicalize(raw2, th or Thresholds())
    if not t:
        return []

    nums: List[float] = []

    sci = _maybe_parse_scientific(raw2)
    if sci is not None:
        nums.append(float(sci))

    for m in _NUM_RE.finditer(t):
        try:
            nums.append(float(m.group(1)))
        except Exception:
            continue

    return nums


def _numeric_match(resp_raw: str, gold_raw: str, th: Thresholds) -> Tuple[bool, Dict[str, Any]]:
    """
    Determine if numeric answers match:
    - treat coordinate formats (90 S vs -90) as equivalent
    - allow tolerance for non-integer answers
    - treat large integer-looking answers (years/ids) as exact-match only
    - only apply numeric logic when the gold looks numeric OR coordinate-like
    """
    info: Dict[str, Any] = {"used": False}

    if resp_raw is None or gold_raw is None:
        return False, info

    # Coordinate handling first (works even if gold contains words like "south")
    g_coord = _parse_coord(gold_raw)
    if g_coord is not None:
        r_coord = _parse_coord(resp_raw)
        if r_coord is not None:
            info.update({"used": True, "mode": "coord", "gold": g_coord, "resp": r_coord})
            return math.isclose(r_coord, g_coord, rel_tol=0.0, abs_tol=0.0), info

        r_nums = _extract_numbers(resp_raw, th)
        if r_nums:
            r_val = float(r_nums[0])
            info.update({"used": True, "mode": "coord_vs_num", "gold": g_coord, "resp": r_val})
            return math.isclose(r_val, g_coord, rel_tol=0.0, abs_tol=0.0), info

        # No numeric provided: not a mismatch here (incompleteness != falsity)
        info.update({"used": True, "mode": "coord_missing_resp_number", "gold": g_coord})
        return False, info

    # Not a coord; numeric only if gold looks numeric
    if not _looks_like_numeric_answer(gold_raw):
        return False, info

    g_nums = _extract_numbers(gold_raw, th)
    a_nums = _extract_numbers(resp_raw, th)
    if not g_nums or not a_nums:
        return False, info

    g = float(g_nums[0])
    a = float(a_nums[0])
    info.update({"used": True, "mode": "numeric", "gold": g, "resp": a})

    # Strict for big integer-ish (years/IDs)
    if _is_integerish(g) and _is_integerish(a):
        g_int = int(round(g))
        a_int = int(round(a))
        if abs(g_int) >= 100 or abs(a_int) >= 100:
            info["integer_strict"] = True
            return g_int == a_int, info

    ok = math.isclose(a, g, rel_tol=th.numeric_rel_tol, abs_tol=th.numeric_abs_tol)
    info["rel_tol"] = th.numeric_rel_tol
    info["abs_tol"] = th.numeric_abs_tol
    return ok, info


def _numeric_conflict_if_present(resp_raw: str, gold_raw: str, th: Thresholds) -> Tuple[bool, Dict[str, Any]]:
    """
    Hallucination evidence for numeric/coord gold:
    If gold is numeric/coord and the response contains numeric values, require that at least one
    of the response numbers matches the gold (within strict/tolerant regime).
    If response contains *no* numbers, do NOT call it hallucination (may be incomplete).
    """
    info: Dict[str, Any] = {"used": False}

    if resp_raw is None or gold_raw is None:
        return False, info

    # coordinates treated as numeric targets too
    g_coord = _parse_coord(gold_raw)
    if g_coord is not None:
        r_nums = _extract_numbers(resp_raw, th)
        if not r_nums:
            info.update({"used": True, "mode": "coord_no_resp_numbers", "conflict": False, "gold": g_coord})
            return False, info
        any_match = any(math.isclose(float(x), g_coord, rel_tol=0.0, abs_tol=0.0) for x in r_nums)
        info.update({"used": True, "mode": "coord_numbers_present", "gold": g_coord, "resp_numbers": r_nums, "any_match": any_match})
        return (not any_match), info

    if not _looks_like_numeric_answer(gold_raw):
        return False, info

    g_nums = _extract_numbers(gold_raw, th)
    r_nums = _extract_numbers(resp_raw, th)
    if not g_nums:
        return False, info

    if not r_nums:
        info.update({"used": True, "mode": "no_response_numbers", "conflict": False})
        return False, info  # absence != falsity

    g = float(g_nums[0])

    def matches(x: float) -> bool:
        if _is_integerish(g) and _is_integerish(x):
            g_int = int(round(g))
            x_int = int(round(x))
            if abs(g_int) >= 100 or abs(x_int) >= 100:
                return g_int == x_int
        return math.isclose(x, g, rel_tol=th.numeric_rel_tol, abs_tol=th.numeric_abs_tol)

    any_match = any(matches(float(x)) for x in r_nums)
    info.update({"used": True, "mode": "numbers_present", "gold": g, "resp_numbers": r_nums, "any_match": any_match})
    return (not any_match), info  # conflict if none match


def _numeric_alternative_conflict(resp_raw: str, gold_raw: str, th: Thresholds) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect a common hallucination pattern for numeric/coord answers:
        "The answer is X or Y" / "either X or Y"
    where at least one alternative differs from the gold numeric value.
    """
    info: Dict[str, Any] = {"used": False}

    if resp_raw is None or gold_raw is None:
        return False, info

    # only apply if gold is numeric or coord
    g_coord = _parse_coord(gold_raw)
    gold_is_numeric = g_coord is not None or _looks_like_numeric_answer(gold_raw)
    if not gold_is_numeric:
        return False, info

    resp_can = canonicalize(resp_raw, th)
    if not resp_can:
        return False, info

    if not any(f" {m} " in f" {resp_can} " for m in th.numeric_alt_markers):
        return False, info

    # gold target
    if g_coord is not None:
        g = float(g_coord)
    else:
        g_nums = _extract_numbers(gold_raw, th)
        if not g_nums:
            return False, info
        g = float(g_nums[0])

    r_nums = _extract_numbers(resp_raw, th)
    if len(r_nums) < 2:
        return False, info

    differing: List[float] = []
    for x in r_nums:
        xf = float(x)
        if _is_integerish(g) and _is_integerish(xf) and (abs(int(round(g))) >= 100 or abs(int(round(xf))) >= 100):
            if int(round(xf)) != int(round(g)):
                differing.append(xf)
        else:
            if not math.isclose(xf, g, rel_tol=th.numeric_rel_tol, abs_tol=th.numeric_abs_tol):
                differing.append(xf)

    if differing:
        info.update({"used": True, "mode": "numeric_alternatives", "gold": g, "diff_alternatives": differing})
        return True, info

    return False, info


# ----------------------------
# Embeddings
# ----------------------------

_EMBEDDER = None


def _get_embedder(th: Thresholds):
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    _require_sentence_transformers()
    from sentence_transformers import SentenceTransformer  # type: ignore
    _EMBEDDER = SentenceTransformer(th.embed_model_name)
    return _EMBEDDER


def _cosine(u: List[float], v: List[float]) -> float:
    # minimal cosine without numpy dependency
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu <= 0.0 or nv <= 0.0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


def embedding_similarity(resp_can: str, gold_can: str, th: Thresholds) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (cos_sim, info). Raises if dependencies are missing (strict policy).
    """
    embedder = _get_embedder(th)

    vecs = embedder.encode([resp_can, gold_can], normalize_embeddings=False)
    try:
        u = vecs[0].tolist()
        v = vecs[1].tolist()
    except Exception:
        u = list(vecs[0])
        v = list(vecs[1])

    score = float(_cosine(u, v))
    info = {"used": True, "model": th.embed_model_name}
    return score, info


def _embedding_support(resp_can: str, gold_can: str, embed_score: Optional[float], th: Thresholds) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide whether to treat embedding similarity as evidence of support.
    Adds robust gating for short strings and low-overlap matches.
    """
    info: Dict[str, Any] = {"used": False}

    if embed_score is None:
        return False, info

    g_tokens = tokenize(gold_can)
    if len(g_tokens) <= th.embed_disable_for_very_short_gold_tokens:
        info.update({"used": True, "gated_off": "gold_too_short", "gold_tokens": len(g_tokens)})
        return False, info

    if th.embed_require_shared_content_tokens:
        shared = shared_content_token_count(resp_can, gold_can)
        info["shared_content_tokens"] = shared
        if shared < th.embed_min_shared_content_tokens:
            info.update({"used": True, "gated_off": "insufficient_shared_content"})
            return False, info

    ok = bool(embed_score >= th.embed_min_similarity)
    info.update({"used": True, "passed": ok, "threshold": th.embed_min_similarity})
    return ok, info


# ----------------------------
# NLI entailment / contradiction
# ----------------------------

_NLI_TOKENIZER = None
_NLI_MODEL = None
_NLI_LABELS = None


def _get_nli(th: Thresholds):
    global _NLI_TOKENIZER, _NLI_MODEL, _NLI_LABELS
    if _NLI_TOKENIZER is not None and _NLI_MODEL is not None and _NLI_LABELS is not None:
        return _NLI_TOKENIZER, _NLI_MODEL, _NLI_LABELS

    _require_transformers_torch()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(th.nli_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(th.nli_model_name)

    id2label = getattr(model.config, "id2label", None) or {}
    labels = {int(k): str(v).lower() for k, v in id2label.items()} if id2label else {}

    _NLI_TOKENIZER, _NLI_MODEL, _NLI_LABELS = tok, model, labels
    return tok, model, labels


def nli_entailment(premise_can: str, hypothesis_can: str, th: Thresholds) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
    """
    Compute MNLI probs for: premise=premise_can, hypothesis=hypothesis_can
    Returns (probs_dict or None, info).
    probs_dict keys: entailment/neutral/contradiction
    """
    info: Dict[str, Any] = {"used": False}

    if len(tokenize(premise_can)) < th.nli_min_tokens or len(tokenize(hypothesis_can)) < th.nli_min_tokens:
        info.update({"used": False, "skipped": "too_short"})
        return None, info

    tok, model, labels = _get_nli(th)
    import torch  # type: ignore

    inputs = tok(premise_can, hypothesis_can, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().tolist()

    out = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
    if labels:
        for i, p in enumerate(probs):
            lab = labels.get(i, "")
            if "entail" in lab:
                out["entailment"] = float(p)
            elif "contrad" in lab:
                out["contradiction"] = float(p)
            elif "neutral" in lab:
                out["neutral"] = float(p)
    else:
        # common MNLI head: contradiction, neutral, entailment
        if len(probs) == 3:
            out["contradiction"] = float(probs[0])
            out["neutral"] = float(probs[1])
            out["entailment"] = float(probs[2])

    info.update({"used": True, "model": th.nli_model_name})
    return out, info


# ----------------------------
# Contradiction / unavailability heuristics
# ----------------------------

def _contradiction_heuristic(resp_can: str, gold_can: str) -> bool:
    """Catch explicit negation targeting the gold answer."""
    if not resp_can or not gold_can:
        return False

    g = re.escape(gold_can)

    # "not <gold>" / "isn't <gold>" / "no <gold>"
    if re.search(rf"\b(not|isn't|isnt|no)\s+{g}\b", resp_can):
        return True

    # "<gold> is not"
    if re.search(rf"\b{g}\s+(is|was|are|were)\s+(not|no)\b", resp_can):
        return True

    return False


def _unavailability_claim(resp_can: str, th: Thresholds) -> bool:
    """Detect assertions that the answer does not exist / is unavailable."""
    if not resp_can:
        return False
    r = f" {resp_can} "
    return any(f" {cue} " in r for cue in th.unavailability_cues)


def _is_atomic_gold_answer(gold_raw: str, gold_can: str, th: Thresholds) -> bool:
    """
    Decide whether the gold answer is "atomic" (entity/code) rather than a definition phrase.

    We ONLY use atomic mismatch as hallucination evidence, to avoid false positives on
    short definition-style golds like "packets from circulating indefinitely".
    """
    if not gold_can:
        return False
    if len(gold_can) > th.atomic_gold_max_chars:
        return False

    g_toks = tokenize(gold_can)
    if len(g_toks) > th.atomic_gold_max_tokens:
        return False

    # If the raw gold has no spaces, it's very likely an atomic code/formula (e.g., "::1", "VES", "C6H8O6")
    if gold_raw and (" " not in gold_raw.strip()):
        return True

    # Otherwise, require that (after stopword removal) we still have all tokens (i.e., not a phrase)
    g_content = content_tokens(g_toks)
    if len(g_content) != len(g_toks):
        return False

    return True


# ----------------------------
# Main detector
# ----------------------------

def detect_details(
    model_response: str,
    correct_answer: str,
    thresholds: Optional[Thresholds] = None,
) -> Dict[str, Any]:
    """
    Compute hallucination decision + rich diagnostics.

    Output keys include:
      - hallucinated (bool)
      - reason (str)
      - baseline_exact (bool)
      - baseline_embed (bool)
      - baseline_embed_score (float)
      - baseline_embed_method (str)
      - plus intermediate signal metadata
    """
    th = thresholds or Thresholds()

    resp_raw = "" if model_response is None else str(model_response)
    gold_raw = "" if correct_answer is None else str(correct_answer)

    resp = canonicalize(resp_raw, th)
    gold = canonicalize(gold_raw, th)

    resp_tokens = tokenize(resp)
    gold_tokens = tokenize(gold)

    details: Dict[str, Any] = {
        "model_response": resp_raw,
        "correct_answer": gold_raw,
        "normalized_response": resp,
        # Backwards-compat: older code used "normalized_answer" for the gold.
        "normalized_answer": gold,
        "normalized_correct_answer": gold,
    }

    # Empty response: for this repo's binary output, mark hallucinated.
    if not resp:
        details.update(
            {
                "hallucinated": True,
                "accepted": False,
                "reason": "empty_response",
                "baseline_exact": False,
                "baseline_embed": False,
                "baseline_embed_score": None,
                "baseline_embed_method": "none",
            }
        )
        return details

    # 1) Exact normalized match
    exact = resp == gold
    details["baseline_exact"] = exact

    # 2) Lexical overlap
    jac = jaccard_similarity(resp, gold)
    details["jaccard"] = jac

    # 3) Token subsequence containment
    subseq_ok = False
    if (
        th.allow_token_subsequence
        and len(gold) >= th.subseq_min_gold_chars
        and len(resp) <= th.subseq_max_answer_chars
        and gold_tokens
    ):
        subseq_ok = token_subsequence_match(resp_tokens, gold_tokens)
    details["token_subsequence"] = subseq_ok

    # 3b) Normalized substring
    substr_ok = normalized_substring_match(resp, gold)
    details["normalized_substring"] = substr_ok

    # 4) Numeric equivalence + numeric conflict evidence
    num_ok, num_info = _numeric_match(resp_raw, gold_raw, th)
    details["numeric"] = num_info
    details["numeric_match"] = bool(num_ok)

    num_conflict, num_conflict_info = _numeric_conflict_if_present(resp_raw, gold_raw, th)
    details["numeric_conflict_if_present"] = {"flag": bool(num_conflict), **num_conflict_info}

    num_alt_conflict, num_alt_info = _numeric_alternative_conflict(resp_raw, gold_raw, th)
    details["numeric_alternative_conflict"] = {"flag": bool(num_alt_conflict), **num_alt_info}

    # 5) Embedding similarity baseline
    embed_score: Optional[float] = None
    embed_info: Dict[str, Any] = {"used": False}
    if th.enable_embeddings:
        try:
            embed_score, embed_info = embedding_similarity(resp, gold, th)
        except Exception as e:
            if th.strict_dependencies:
                raise
            embed_info = {"used": False, "error": str(e)}
            embed_score = None
    details["embedding"] = embed_info

    # Baseline column behavior (unchanged contract):
    # - If embeddings used, baseline_embed_score=cosine; else jaccard.
    if embed_info.get("used") and embed_score is not None:
        details["baseline_embed_method"] = "embedding"
        details["baseline_embed_score"] = float(embed_score)
        baseline_embed = bool(embed_score >= th.embed_min_similarity)
    else:
        details["baseline_embed_method"] = "jaccard"
        details["baseline_embed_score"] = float(jac)
        baseline_embed = bool(jac >= th.jaccard_min)
    details["baseline_embed"] = baseline_embed

    # Embedding as gated support (diagnostics)
    embed_support_ok, embed_support_info = _embedding_support(resp, gold, embed_score, th)
    details["embedding_support"] = embed_support_info
    details["embedding_match"] = bool(embed_support_ok)

    # 6) Bidirectional NLI (contradiction evidence + entailment diagnostics)
    nli_f_probs: Optional[Dict[str, float]] = None
    nli_f_info: Dict[str, Any] = {"used": False}
    nli_r_probs: Optional[Dict[str, float]] = None
    nli_r_info: Dict[str, Any] = {"used": False}

    if th.enable_nli:
        try:
            nli_f_probs, nli_f_info = nli_entailment(resp, gold, th)  # answer => gold
            nli_r_probs, nli_r_info = nli_entailment(gold, resp, th)  # gold => answer
        except Exception as e:
            if th.strict_dependencies:
                raise
            nli_f_info = {"used": False, "error": str(e)}
            nli_r_info = {"used": False, "error": str(e)}
            nli_f_probs = None
            nli_r_probs = None

    details["nli"] = {"probs": nli_f_probs, **nli_f_info}  # backwards-compat
    details["nli_reverse"] = {"probs": nli_r_probs, **nli_r_info}

    def _get(p: Optional[Dict[str, float]], k: str) -> Optional[float]:
        return float(p[k]) if p and k in p else None

    f_ent = _get(nli_f_probs, "entailment")
    f_con = _get(nli_f_probs, "contradiction")
    r_ent = _get(nli_r_probs, "entailment")
    r_con = _get(nli_r_probs, "contradiction")

    details["nli_entailment"] = f_ent
    details["nli_contradiction"] = f_con
    details["nli_entailment_reverse"] = r_ent
    details["nli_contradiction_reverse"] = r_con

    ents = [x for x in [f_ent, r_ent] if x is not None]
    cons = [x for x in [f_con, r_con] if x is not None]

    entail_min = min(ents) if ents else None
    contr_max = max(cons) if cons else None

    details["nli_entail_min"] = entail_min
    details["nli_contradiction_max"] = contr_max

    nli_contradiction_veto = bool(
        entail_min is not None
        and contr_max is not None
        and contr_max >= th.nli_contradiction_veto
        and entail_min <= th.nli_entailment_veto_floor
    )
    details["nli_contradiction_veto"] = nli_contradiction_veto

    # 7) Negation/contradiction heuristics
    contradiction_flag = _contradiction_heuristic(resp, gold)
    details["contradiction_heuristic"] = contradiction_flag

    unavailability_flag = _unavailability_claim(resp, th)
    details["unavailability_claim"] = unavailability_flag

    # ----------------------------
    # Hallucination decision (positive-evidence only)
    # ----------------------------

    # A) Explicit contradiction cue
    if contradiction_flag:
        details["hallucinated"] = True
        details["accepted"] = False
        details["reason"] = "contradiction_cue"
        return details

    # B) Strong NLI contradiction (bidirectional)
    if nli_contradiction_veto:
        details["hallucinated"] = True
        details["accepted"] = False
        details["reason"] = "nli_contradiction_veto"
        return details

    # C) Numeric-based contradictions
    if num_alt_conflict:
        details["hallucinated"] = True
        details["accepted"] = False
        details["reason"] = "numeric_alternative_conflict"
        return details

    if num_conflict:
        details["hallucinated"] = True
        details["accepted"] = False
        details["reason"] = "numeric_conflict"
        return details

    # D) Unavailability/nonexistence claim contradicting a concrete gold answer
    if unavailability_flag and not substr_ok and not exact and not num_ok:
        details["hallucinated"] = True
        details["accepted"] = False
        details["reason"] = "unavailability_claim_conflicts_with_gold"
        return details

    # E) Atomic entity/code mismatch (safe; avoids definition false positives like TTL)
    if _is_atomic_gold_answer(gold_raw, gold, th):
        r_toks = tokenize(resp)
        if len(r_toks) <= th.atomic_resp_max_tokens and not exact and not substr_ok and not num_ok:
            shared = set(content_tokens(tokenize(gold))) & set(content_tokens(r_toks))
            if not shared:
                details["hallucinated"] = True
                details["accepted"] = False
                details["reason"] = "atomic_entity_mismatch"
                return details

    # Otherwise: no positive evidence of falsity/contradiction => NOT hallucinated.
    details["hallucinated"] = False
    details["accepted"] = True

    # Best-effort reason for transparency
    if exact:
        details["reason"] = "exact_match"
    elif num_ok:
        details["reason"] = "numeric_match"
    elif substr_ok:
        details["reason"] = "substring_match"
    elif subseq_ok:
        details["reason"] = "token_subsequence_match"
    elif embed_support_ok:
        details["reason"] = "embedding_support"
    else:
        details["reason"] = "no_contradiction_evidence"

    # Weak signal counts for diagnostics (not used to force hallucination)
    weak_passes = 0
    if subseq_ok:
        weak_passes += 1
    if jac >= th.jaccard_min:
        weak_passes += 1
    if baseline_embed:
        weak_passes += 1
    details["weak_passes"] = weak_passes
    details["weak_required"] = th.weak_signals_required

    return details


def is_hallucinated(
    model_response: str,
    correct_answer: str,
    thresholds: Optional[Thresholds] = None,
) -> bool:
    """Convenience wrapper used by package __init__.py."""
    return bool(detect_details(model_response, correct_answer, thresholds).get("hallucinated"))
