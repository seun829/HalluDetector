import os
import openai
import os
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append((pid, prompt, answer))
    return results

def simple_generate_openai(prompt_list, model_name="gpt-3.5-turbo"):
    """
    Generate text using an OpenAI GPT model (e.g., GPT-3.5-turbo).
    Expects OPENAI_API_KEY to be set in the environment.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    results = []
    for pid, prompt, _ in prompt_list:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=60
        )
        answer = response.choices[0].message.content
        results.append((pid, prompt, answer))
    return results