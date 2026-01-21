"""src.hallu_detector.generate

Generation helpers used by the web app and CLI scripts.

Public API (kept stable):
  - simple_generate_hf(prompt_list, model_name="gpt2")
  - simple_generate_openai(prompt_list, model_name)

Both functions accept a prompt_list of tuples:
  (id, prompt, correct_answer)

This module is intentionally conservative:
  - Defaults preserve the previous behavior unless you pass explicit decoding args.
  - GPT-5 models use the Responses API; other models use Chat Completions.
  - Avoids passing unsupported parameters (temperature/top_p) to GPT-5 variants
    that reject them.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


PromptItem = Tuple[int, str, Any]
GenItem = Tuple[int, str, str]


def simple_generate_hf(
    prompt_list: Sequence[PromptItem],
    model_name: str = "gpt2",
    *,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> List[GenItem]:
    """Generate text using a HuggingFace causal LM.

    Backwards compatible:
      - If max_new_tokens/temperature/top_p are not provided, matches the old
        behavior (max_new_tokens=50, greedy decoding).

    If temperature > 0, sampling is enabled; otherwise greedy decoding is used.
    """
    max_new = 50 if max_new_tokens is None else int(max_new_tokens)
    temp = temperature
    tp = top_p

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Some tokenizers (e.g., GPT-2) do not define pad_token_id.
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    do_sample = bool(temp is not None and float(temp) > 0)

    results: List[GenItem] = []
    for pid, prompt, _ in prompt_list:
        inputs = tokenizer(prompt, return_tensors="pt")
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new,
            "pad_token_id": pad_token_id,
        }
        if do_sample:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": float(temp),
            })
            if tp is not None:
                gen_kwargs["top_p"] = float(tp)
        else:
            gen_kwargs["do_sample"] = False

        outputs = model.generate(**inputs, **gen_kwargs)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        results.append((int(pid), prompt, answer))

    return results


def _supports_sampling_params_for_gpt5(model_name: str) -> bool:
    """Return True if we should send temperature/top_p to this GPT-5 model.

    In practice, GPT-5.1 and GPT-5.2 support temperature/top_p when reasoning
    effort is set to none. Other GPT-5 variants may reject these parameters.
    """
    m = (model_name or "").strip().lower()
    return m.startswith(("gpt-5.2", "gpt-5.1"))


def simple_generate_openai(
    prompt_list: Sequence[PromptItem],
    model_name: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> List[GenItem]:
    """Generate text using an OpenAI model.

    Backwards compatible:
      - If temperature/top_p/max_output_tokens are not provided, this function
        behaves like the original: different defaults depending on model family.

    Notes:
      - GPT-5 models use the Responses API.
      - Non-GPT-5 models use Chat Completions.
      - We avoid passing unsupported parameters to GPT-5 variants.
    """


    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    m = (model_name or "").strip()
    m_lower = m.lower()

    use_responses = m_lower.startswith("gpt-5")
    results: List[GenItem] = []

    print("Generating from " + m)

    for pid, prompt, _ in prompt_list:
        if use_responses:
            # Responses API expects an 'input' message array.
            input_msgs: List[Dict[str, str]] = []
            if system_prompt:
                input_msgs.append({"role": "system", "content": system_prompt})
            input_msgs.append({"role": "user", "content": prompt})

            req: Dict[str, Any] = {
                "model": m,
                "input": input_msgs,
                "max_output_tokens": int(max_output_tokens or 512),
            }
            # Only include sampling params for GPT-5.1/5.2, and only when
            # explicitly disabling reasoning effort.
            if _supports_sampling_params_for_gpt5(m):
                req["reasoning"] = {"effort": "none"}
                if temperature is not None:
                    req["temperature"] = float(temperature)
                if top_p is not None:
                    req["top_p"] = float(top_p)

            r = client.responses.create(**req)
            answer = (getattr(r, "output_text", "") or "").strip()
            results.append((int(pid), prompt, answer))
            continue

        # Chat Completions for non-GPT-5
        # Preserve the old defaults unless explicitly overridden.
        if m_lower.startswith("gpt-4"):
            default_max = 2000 if m_lower == "gpt-4" else 4000
            default_temp = 0.8
        else:
            default_max = 512
            default_temp = 1.0

        req2: Dict[str, Any] = {
            "model": m,
            "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else [])
            + [{"role": "user", "content": prompt}],
            "max_tokens": int(max_output_tokens or default_max),
            "temperature": float(default_temp if temperature is None else temperature),
        }
        if top_p is not None:
            req2["top_p"] = float(top_p)

        resp = client.chat.completions.create(**req2)
        answer = (resp.choices[0].message.content or "").strip()
        results.append((int(pid), prompt, answer))

    return results
