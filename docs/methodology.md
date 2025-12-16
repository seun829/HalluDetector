# Methodology: Hallucination Detection

## Overview

This system detects **hallucinations in model-generated answers** by comparing a model response against a known correct reference answer.  
A hallucination is defined here as an answer that is **factually inconsistent, contradictory, numerically incorrect, or semantically unsupported** by the reference.

The detector is **reference-based**: it does not assess truth in isolation, but rather **consistency with a provided ground-truth answer**.

The approach combines:
- text normalization,
- semantic similarity,
- bidirectional natural language inference (NLI),
- numeric consistency checks,
- negation heuristics,
- and a transparent, rule-based decision layer.

The system is designed to be **robust and interpretable**, returning both a binary decision and detailed diagnostic scores.

---

## High-Level Pipeline

Given:
- `answer`: the model-generated response
- `correct_answer`: the ground-truth reference

The detector proceeds as follows:

1. Text normalization
2. Shortcut checks (substring containment)
3. Semantic similarity estimation
4. Bidirectional entailment and contradiction analysis
5. Numeric consistency verification
6. Negation mismatch detection
7. Rule-based decision aggregation

---

## 1. Text Normalization

Both `answer` and `correct_answer` are normalized to reduce superficial differences:

- Unicode normalization (NFKC)
- Lowercasing
- Whitespace collapse
- Light punctuation removal (retaining `%`, currency symbols, and `-`)

This ensures that differences in formatting, casing, or typography do not affect downstream comparisons.

---

## 2. Substring Shortcut

If one normalized text is a substring of the other, the detector assumes **strong semantic alignment** and skips expensive model-based checks.

However, even in this case, **numeric consistency is still enforced**:
- If numbers disagree, the answer is flagged as hallucinated.
- Otherwise, the answer is considered non-hallucinated.

This optimization improves performance while preserving correctness.

---

## 3. Semantic Similarity

### Embedding-Based Similarity (Primary)

Semantic similarity is computed using cosine similarity between sentence embeddings produced by:

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`

This captures paraphrasing and semantic equivalence beyond surface wording.

### Lexical Jaccard Similarity (Auxiliary / Fallback)

Token-level Jaccard similarity is also computed as a lexical signal.  
If embeddings are unavailable, Jaccard similarity is used as a fallback.

---

## 4. Bidirectional Natural Language Inference (NLI)

Logical consistency is assessed using **bidirectional NLI** with:

- **Model**: `roberta-large-mnli`

Two inferences are computed:

1. Does `answer` entail `correct_answer`?
2. Does `correct_answer` entail `answer`?

Each inference produces probabilities for:
- Entailment
- Neutral
- Contradiction

From these, the system derives:
- `entail_min`: the minimum entailment probability across both directions
- `contr_max`: the maximum contradiction probability across both directions

### Rationale

Bidirectional inference detects:
- Overly specific answers (answer adds unsupported details)
- Missing constraints
- Explicit contradictions

This is critical for distinguishing:
> “incomplete but correct”  
from  
> “confidently wrong”

---

## 5. Numeric Consistency Guard

Numbers are treated as **high-risk hallucination triggers**.

### Extraction

All numeric values are extracted along with metadata:
- Percent markers (`%`)
- Currency symbols (`$€£¥`)
- Position context

### Matching Rules

- If one side contains numbers and the other does not → mismatch
- If the number of numeric tokens differs → mismatch
- Percent numbers must align with percent numbers
- Currency numbers must align with currency numbers
- **Years (1000–2100)** must match exactly
- Other numbers must match within a configurable relative tolerance (default: 2%)

If numeric strictness is enabled and a mismatch is detected, the answer is immediately flagged as hallucinated.

---

## 6. Negation Mismatch Heuristic

The system checks for negation tokens (e.g., *not, no, never, without*).

If one text contains negation and the other does not, this is marked as a **strong contradiction signal**.

Negation alone does not force a hallucination decision, but it biases the final outcome when entailment is weak.

---

## 7. Rule-Based Decision Logic

All signals are combined using a transparent rule set.

### Immediate Hallucination (Hard Red Flags)

An answer is marked hallucinated if:
- Maximum contradiction probability exceeds a threshold
- Numeric mismatch is detected (when numeric strictness is enabled)

### Clear Non-Hallucination (Green Case)

An answer is marked non-hallucinated if:
- Semantic similarity is high
- Bidirectional entailment is strong
- Contradiction probability is low
- No numeric mismatch exists

### Clear Hallucination (Low Support)

An answer is marked hallucinated if:
- Semantic similarity is low **and**
- Entailment is low  
(or lexical overlap is extremely low with low entailment)

### Ambiguous Region

When scores lie near thresholds:
- The system records an `"abstain_band"` rationale
- A conservative bias is applied: low entailment defaults to hallucinated

This ensures safer behavior in uncertain cases.

---

## 8. Outputs

### `detect_details(...)`

Returns a structured dictionary containing:
- Final hallucination decision
- Semantic similarity scores
- NLI entailment and contradiction probabilities
- Numeric analysis details
- Negation flags
- Thresholds used
- Human-readable rationale string

### `is_hallucinated(...)`

A lightweight wrapper that returns only the boolean decision.

---

## Model Availability and Fallback Behavior

If required models are unavailable:
- Embedding similarity falls back to lexical Jaccard similarity
- NLI defaults to neutral-only outputs

**Important**: Thresholds are tuned for full model availability.  
Running in fallback mode without adjusting thresholds may significantly increase false positives.

---

## Intended Use and Limitations

### Intended Use
- Evaluation of model answers against gold references
- Research on hallucination detection
- Diagnostic analysis of factual consistency

### Limitations
- Reference-based only (cannot detect hallucinations without a correct answer)
- Sensitive to numeric precision depending on tolerance settings
- Not designed for open-ended truth verification

---

## Summary

This hallucination detector combines semantic similarity, logical inference, and strict numeric validation to determine whether a model-generated answer is **supported by a reference answer**.

Its design prioritizes:
- Interpretability
- Robustness
- Research reproducibility

Correct installation of the embedding and NLI models is essential for reliable performance.

