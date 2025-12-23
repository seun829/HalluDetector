import os
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM


# No normalization needed—model_name comes from a validated dropdown

def simple_generate_hf(prompt_list, model_name="gpt2"):
    """
    Generate text using a HuggingFace model (e.g., GPT-2).
    Each item in prompt_list should be a tuple: (id, prompt, correct_answer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    results = []
    for pid, prompt, _ in prompt_list:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        results.append((pid, prompt, answer))
    return results

def simple_generate_openai(prompt_list, model_name: str):
    """
    Generate text using an OpenAI GPT model.
    Expects OPENAI_API_KEY to be set in the environment.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    m = (model_name or "").strip()

    # Decide endpoint: GPT-5 family works best with Responses API
    use_responses = m.startswith("gpt-5")
    results = []

    for pid, prompt, _ in prompt_list:
        if use_responses:
            # GPT-5 models: use Responses API and max_output_tokens
            req = {
                "model": m,
                "input": [{"role": "user", "content": prompt}],
                "max_output_tokens": 512,  # your prompts want short answers; bump if needed
            }

            # Temperature compatibility:
            # - GPT-5.2 supports temperature only when reasoning effort is "none"
            # - Older GPT-5 (gpt-5, gpt-5-mini, gpt-5-nano) will error if you include temperature
            #   per OpenAI docs. :contentReference[oaicite:4]{index=4}
            if m.startswith(("gpt-5.2", "gpt-5.1")):
                req["reasoning"] = {"effort": "none"}
                req["temperature"] = 0.7

            r = client.responses.create(**req)
            answer = (getattr(r, "output_text", "") or "").strip()
            results.append((pid, prompt, answer))
            continue

        # Non-GPT-5: keep Chat Completions
        if m.startswith("gpt-4"):
            max_tok = 2000 if m == "gpt-4" else 4000
            temperature = 0.8
        else:
            max_tok = 512
            temperature = 1.0

        response = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tok,
        )
        answer = response.choices[0].message.content.strip()
        results.append((pid, prompt, answer))

    return results
