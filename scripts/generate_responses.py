import pandas as pd
import os
from hallu_detector.generate import simple_generate_hf, simple_generate_openai

# SETTINGS
USE_OPENAI = False  # Set to True if using OpenAI API
MODEL_NAME = "gpt2"  # Or "gpt-3.5-turbo" if using OpenAI
INPUT_FILE = "data/raw/prompts_easy.csv"
INPUT_FILE2 = "data/raw/prompts_hard.csv"
OUTPUT_FILE = "data/raw/responses_raw_easy.csv"
OUTPUT_FILE2 = "data/raw/responses_raw_hard.csv"


# Load prompts from CSV
df = pd.read_csv(INPUT_FILE)
df2 = pd.read_csv(INPUT_FILE2)
prompt_list = list(df[['id', 'prompt']].itertuples(index=False, name=None))
prompt_list2 = list(df2[['id', 'prompt']].itertuples(index=False, name=None))

# Generate responses
if USE_OPENAI:
    os.environ["OPENAI_API_KEY"] = "your_api_key_here"  # or export it beforehand
    responses = simple_generate_openai(prompt_list, model_name=MODEL_NAME)
    responses2 = simple_generate_openai(prompt_list2, model_name = MODEL_NAME)
else:
    responses = simple_generate_hf(prompt_list, model_name=MODEL_NAME)
    responses2 = simple_generate_hf(prompt_list2, model_name=MODEL_NAME)

# Save responses to new CSV
df_out = pd.DataFrame(responses, columns=["id", "prompt", "model_response"])
df_out2 = pd.DataFrame(responses2, columns=["id", "prompt", "model_response"])

df_out.to_csv(OUTPUT_FILE, index=False)
df_out.to_csv(OUTPUT_FILE2, index = False)

print(f"Saved {len(df_out)} responses to {OUTPUT_FILE}")
print(f"Saved {len(df_out)} responses to {OUTPUT_FILE2}")
