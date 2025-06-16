"""
Master script to run the entire hallucination detection pipeline for multiple files,
including auto-generating prompts.
"""
import subprocess
import os

def run_step(description, command):
    print(f"Running: {description}")
    try:
        subprocess.run(command, check=True)
        print(f"{description} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}")
        raise

def main():
    # Step 1: Auto-generate prompts using make_prompts.py
    # The config file may not exist; make_prompts.py will then use a default config.
    auto_prompts_config = "config/prompts_config.yaml"  # This file need not exist.
    auto_prompts_out = "data/raw/prompts_auto-generated.csv"
    run_step("Generate auto prompts", 
             ["python", "scripts/make_prompts.py", 
              "--config", auto_prompts_config, 
              "--out-csv", auto_prompts_out])
    
    # Define input prompt files
    prompt_files = [
        "data/raw/prompts_easy.csv",
        auto_prompts_out,  # now using the auto-generated prompts from make_prompts.py
        "data/raw/prompts_hard.csv"
    ]
    # Define corresponding response files
    response_files = [
        "data/processed/responses_easy_labeled.csv",
        "data/processed/responses_auto-generated_labeled.csv",
        "data/processed/responses_hard_labeled.csv"
    ]

    # Validate prompt files exist (for auto-generated prompt the file is created above)
    for file in prompt_files:
        if not os.path.exists(file):
            print(f"Error: File {file} does not exist.")
            return
        if os.stat(file).st_size == 0:
            print(f"Error: File {file} is empty.")
            return

    # Define pipeline steps
    steps = [
        ("Generate responses", ["python", "scripts/generate_responses.py", 
                                  "--prompt-files"] + prompt_files +
                                  ["--response-files"] + response_files +
                                  ["--model", "gpt2"]),
        ("Analyze patterns", ["python", "scripts/graph_patterns.py"])
    ]

    # Execute each step sequentially
    for description, command in steps:
        run_step(description, command)

if __name__ == "__main__":
    main()