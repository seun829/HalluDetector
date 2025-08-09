import os
import openai
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


def simple_generate_openai(prompt_list, model_name="gpt-3.5-turbo"):
    """
    Generate text using an OpenAI GPT model.
    Expects OPENAI_API_KEY to be set in the environment.
    Model name is taken directly from a dropdown, so it should be valid (e.g., 'gpt-3.5-turbo' or 'gpt-4').
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = model_name

    # Adjust parameters for GPT-4 vs GPT-3.5
    if model.startswith("gpt-4"):
        # GPT-4 base context ~8k, -32k variant
        max_tok = 2000 if model == "gpt-4" else 4000
        temperature = 0.8
    else:
        # Default for GPT-3.5-turbo
        max_tok = 512
        temperature = 1.0

    results = []
    for pid, prompt, _ in prompt_list:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tok
        )
        answer = response.choices[0].message.content.strip()
        results.append((pid, prompt, answer))
    return results
