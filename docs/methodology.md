# Methodology: Hallucination Detection

## Overview

This system detects **hallucinations in model-generated answers** by comparing a model response against a known correct reference answer.  
A hallucination is defined here as an answer that **contains at least one fact that is false, fabricated, or unsupported by known reality**, *relative to the provided reference*.

The detector is **reference-based**: it does not attempt open-world truth verification. Instead, it evaluates whether the model response is **inconsistent with** (or directly contradicts) the reference answer.

Crucially, under this definition:
- **Incompleteness is not hallucination** (e.g., leaving out a qualifier is not automatically false).
- **Paraphrases are not hallucination** (e.g., using equivalent wording is acceptable).
- Hallucination requires **positive evidence of inconsistency** (e.g., wrong number, contradiction, explicit negation, mutually exclusive alternatives).

The approach combines:
- robust text normalization,
- safe shortcut matching for short entities/codes,
- numeric and coordinate consistency checks (including scientific notation),
- semantic similarity (embeddings + lexical) with guardrails,
- bidirectional natural language inference (NLI) for contradiction evidence,
- explicit negation and “nonexistence/unavailability” heuristics,
- and a transparent, rule-based decision layer producing rich diagnostics.

The system is designed to be **robust, conservative, and interpretable**, returning both a binary decision and detailed signal traces suitable for research publication.

---

## High-Level Pipeline

Given:
- `answer`: the model-generated response  
- `correct_answer`: the ground-truth reference  

The detector proceeds as follows:

1. Text normalization  
2. Safe shortcut matching (substring / subsequence)  
3. Numeric and coordinate consistency analysis  
4. Semantic similarity estimation (embeddings + lexical)  
5. Bidirectional NLI entailment/contradiction analysis  
6. Negation + unavailability heuristics  
7. Rule-based aggregation with **contradiction vetoes**  

---

## 1. Text Normalization

Both `answer` and `correct_answer` are normalized to reduce superficial differences while preserving scientific and code-relevant structure:

- Unicode normalization (NFKC)
- Lowercasing
- Accent stripping
- Standardization of mathematical typography:
  - Unicode minus variants → `-`
  - Multiplication sign `×` → `x`
  - Superscript digits/signs (e.g., `10⁻²¹`) normalized for scientific parsing
- Whitespace collapse
- Light punctuation removal  
  (retaining `- . / ^ + : =` for scientific notation, identifiers, and codes)

This ensures that formatting and typography do not distort downstream comparisons.

---

## 2. Safe Shortcut Matching

### 2.1 Normalized Substring Match

If the normalized `correct_answer` appears as a substring of the normalized `answer`, the detector treats this as **strong evidence of alignment**, especially for:

- short entity names,
- codes and identifiers (e.g., `::1`, `VES`),
- chemical formulas.

**Numeric and coordinate consistency checks are still enforced** even if substring matching succeeds.

### 2.2 Token Subsequence Match

If all tokens of `correct_answer` appear in order within `answer` (not necessarily contiguously), this is recorded as a **supporting signal**.  
It is not treated as a hallucination filter by itself.

---

## 3. Numeric and Coordinate Consistency Guard

Numbers are treated as **high-risk hallucination triggers**, but only when the task expects numeric content.

### 3.1 Scope: When Numeric Checks Apply

Numeric matching is applied when the gold answer is numeric or coordinate-like, including:

- integers and decimals,
- scientific notation (`1e-21`, `10^-21`, `1.2×10^-3`, `10⁻²¹`),
- coordinates (`90 S`, `90 south`, `-90 degrees`).

Code-like strings (e.g., IPv6 `::1`) are explicitly excluded.

### 3.2 Matching Rules

- Coordinates normalize direction:
  - `90 S` ≡ `-90`
  - `90 N` ≡ `+90`
- Integer-like values with ≥ 3 digits (e.g., years, IDs) must match **exactly**.
- Other values must match within configurable tolerance:
  - default: ±0.5% relative OR ±0.05 absolute.

### 3.3 “Conflict If Present” Principle

A key policy for hallucination alignment:

- If the response **contains numbers** and none match the gold target → **hallucination** (positive evidence of wrong fact).
- If the response contains **no numbers**, the detector does **not** mark hallucination (could be incomplete or verbal).

### 3.4 Numeric Alternatives Conflict

The detector flags answers of the form:

> “X or Y”

when the gold is numeric/coordinate and **at least one alternative differs** from the gold.  
This captures a common hallucination pattern: hedging with mutually exclusive values.

---

## 4. Semantic Similarity

### 4.1 Embedding-Based Similarity (Primary)

Semantic similarity is computed using cosine similarity between sentence embeddings from:

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`

This captures paraphrases and equivalent meaning beyond surface text.

### 4.2 Guardrails Against Short-Text False Positives

Embedding similarity is **gated** to prevent accepting unrelated text due to short answers:

- If the gold answer is extremely short, embeddings are not treated as strong evidence.
- A minimum number of **shared content tokens** (non-stopword, non-numeric) is required.

### 4.3 Lexical Jaccard Similarity (Baseline / Fallback)

Token-level Jaccard similarity is computed for interpretability and as a fallback baseline if embeddings are unavailable.

---

## 5. Bidirectional Natural Language Inference (NLI)

Logical consistency is assessed using bidirectional NLI with:

- **Model**: `roberta-large-mnli`

Two directions are evaluated:

1. `answer` ⇒ `correct_answer`  
2. `correct_answer` ⇒ `answer`

Each yields probabilities for:
- entailment,
- neutral,
- contradiction.

The detector summarizes these as:
- `entail_min`: minimum entailment across both directions
- `contr_max`: maximum contradiction across both directions

### Role in Hallucination Detection

NLI is used primarily to detect **positive inconsistency evidence**:

- A strong contradiction (high `contr_max` with low `entail_min`) triggers a **hallucination veto**.

Importantly:
- Low entailment alone is **not** treated as hallucination (it may reflect incompleteness).

---

## 6. Negation and Unavailability Heuristics

### 6.1 Explicit Negation Targeting the Gold

The detector flags clear contradiction patterns like:

- “not \<gold>”
- “\<gold> is not …”

This is treated as **direct inconsistency evidence**.

### 6.2 Nonexistence / Unavailability Claims

The detector flags assertions such as:

- “not publicly available”
- “does not exist”
- “no such …”

If such a claim appears and the gold is concrete (and not matched otherwise), it is treated as **hallucination**, since the model asserted a false world-fact relative to the reference.

---

## 7. Rule-Based Decision Logic

The system uses a transparent rule set aligned to the definition:  
**hallucination requires positive evidence of inconsistency.**

### 7.1 Immediate Hallucination (Hard Red Flags)

An answer is marked hallucinated if any occurs:

- numeric/coordinate mismatch **when numbers are present**
- numeric alternatives conflict (“X or Y” with wrong alternative)
- explicit negation of the gold
- strong NLI contradiction veto
- unavailability/nonexistence claim that conflicts with a concrete gold
- atomic entity/code mismatch (short gold code/name vs different short response)

### 7.2 Otherwise: Non-Hallucinated

If none of the contradiction red flags fire, the answer is marked **non-hallucinated**, even if:
- it is incomplete,
- it omits qualifiers,
- it is less specific than the gold,
- it uses a paraphrase.

This enforces the intended distinction between:
- **incompleteness** (not hallucination), and
- **fabricated/false claims** (hallucination).

---

## 8. Outputs

### `detect_details(...)`

Returns a structured dictionary containing:
- final hallucination decision (`hallucinated`)
- acceptance flag (`accepted`)
- rationale string (`reason`)
- exact/subsequence/substring indicators
- numeric analysis details (including coordinate/scientific notation parsing)
- embedding similarity and gating diagnostics
- bidirectional NLI probabilities and veto indicators
- negation and unavailability flags

### `is_hallucinated(...)`

A lightweight wrapper returning only the boolean hallucination decision.

---

## Model Availability and Dependency Policy

If required models are unavailable:
- the system raises an error in **strict mode**
- no silent fallback occurs for enabled features

This prevents hidden failure modes and improves reproducibility.

---

## Intended Use and Limitations

### Intended Use
- evaluation of model answers against gold references
- hallucination detection research (reference-based)
- diagnostics of factual consistency and contradiction patterns

### Limitations
- reference-based only (cannot assess truth without gold)
- cannot perfectly detect “extra unsupported facts” without external evidence;
  it instead focuses on **detectable inconsistency** (numbers, contradictions, negation, nonexistence claims)
- NLI may be imperfect on very short texts; guardrails and multiple signals mitigate this

---

## Summary

This hallucination detector identifies hallucinations as **positive evidence of inconsistency with a reference answer**, not mere incompleteness.

It combines:
- robust normalization (including scientific notation and coordinates),
- numeric contradiction detection,
- bidirectional NLI contradiction vetoes,
- and conservative rule-based aggregation

to provide an interpretable, research-grade hallucination signal.
