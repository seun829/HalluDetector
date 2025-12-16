### ai-hallucination-detection 
 AI Hallucination Detection

## Overview:
This project explores how often and why AI language models hallucinate—i.e., generate incorrect, misleading, or fabricated information—especially in response to complex or rare factual prompts. The repo provides a Flask web app, CLI utilities, a rule-based detector (semantic similarity + NLI), visualization scripts, and a prompt-only BERT baseline. Understanding and detecting hallucinations is critical to ensure AI systems remain trustworthy in high-risk fields.

---

## Objectives:
- Measure hallucination rates across **pre-made** and **auto-generated** question sets that are made to trick the AI with logic.  
- Analyze patterns in hallucination behavior (topics, keywords, lengths).  
- Explore preliminary methods to **predict** hallucinations from output and prompt characteristics.

---

## Methods:
- Prompt a language model with sets of logic-based factual and riddle questions.  
- Collect and fact-check responses; label with a rule-based detector (semantic similarity + NLI).  
- Analyze hallucination frequency, severity, and topic sensitivity; visualize keyword/feature correlations.  
- (Optional) Train a prompt-only BERT classifier to predict hallucination risk from prompts alone.

---

## Repository Structure
```
ai-hallucination-detection-main/
├── README.md
├── app.py                        # Flask app: serves UI + /detect, /predict, /run_pipeline
├── requirements.txt              # Core Python deps
├── docs/
│   └── methodology.md            # (empty placeholder)
├── scripts/
│   ├── generate_responses.py     # Generate model outputs + label with rule-based detector
│   ├── graph_patterns.py         # Build graphs + simple results page
│   └── make_prompts.py           # Combine CSVs or auto-generate via OpenAI + Wikipedia
├── simulation_data/
│   └── raw/
│       ├── prompts_factual.csv            # example prompts (truncated)
│       └── prompts_riddle.csv            # example prompts (truncated)
├── src/
│   └── hallu_detector/
│       ├── __init__.py
│       ├── detect.py              # similarity (all-MiniLM-L6-v2) + NLI (roberta-large-mnli)
│       ├── evaluate.py            # compute hallucination metrics from labeled CSVs
│       └── generate.py            # HF/OpenAI helpers (OpenAI uses old ChatCompletion API)    
├── static/
│   ├── index.html                 # frontend UI (3 panels)
│   ├── script.js                  # front-end logic
│   └── style.css                  # styling
├── tests/
│   └── test_detect.py             # unit tests for is_hallucinated
└── output/
    └── <run_id>/
        ├── raw/                   # copied prompts
        ├── processed/             # responses_labeled_*.csv
        └── graphs/                # accuracy_by_keywords.png, feature_correlation_heatmap.png, index.html
```

> Some files include literal `...` where code is abridged, but the interfaces are present (e.g., `is_hallucinated`, `compute_metrics`, CLI stubs).

---

## Installation
**Python 3.10+** recommended.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Additional packages used by the code but not listed in requirements.txt:
pip install flask datasets
```

### Optional setup
- **OpenAI API** (for generation/auto-prompts):  
  Set `OPENAI_API_KEY` in your environment.
  ```bash
  export OPENAI_API_KEY=sk-...       # PowerShell: $env:OPENAI_API_KEY="sk-..."
  ```
- **GPU** (recommended): `torch`/`sentence_transformers` will use CUDA if available.

---

## Data Formats

### Prompts (CSV)
Required columns:
```csv
question_type, template, id,prompt,correct_answer
1,"How many months have 28 days?",12
2,"Which weighs more, a pound of feathers or a pound of gold?",They weigh the same
```

### Labeled Responses (CSV)
Produced by `generate_responses.py`:
```csv
question_type, template, id, prompt, correct_answer, model_response, hallucinated[,label]
```
- `hallucinated` is `"True"`/`"False"` from the rule-based detector.
- Some outputs also include `label` mirroring `hallucinated` for training convenience.

---

## Quickstart

### Run the Web App
```bash
python app.py
# open http://localhost:5000
```

**UI panels**
1. **Detect with Ground Truth** → calls `/detect`  
2. **Prompt-only Predict** → calls `/predict` (needs a trained `./bert_model`)  
3. **Run Entire Pipeline** → calls `/run_pipeline` to copy prompts → generate → analyze → evaluate

**Endpoints**
- `POST /detect`  
  Body: `{"answer":"Paris","correct":"Paris"}` → `{"hallucinated": false}`
- `POST /predict` (prompt-only BERT)  
  Body: `{"prompt":"..."}` → `{"hallucination_probability": 0.xx, "will_hallucinate": false}`  
  *(Requires a model saved in `./bert_model`.)*
- `POST /run_pipeline`  
  Body: `{"model":"gpt2"}` `{"model":"gpt-3.5-turbo"}`  `{"model":"gpt-4"}`   `{"model":"gpt-4o"}`  
  Returns `{ stage, run_id, logs, graphs, metrics, processed_csvs }` and writes files under `static/output/<run_id>/...`.

---

## CLI Usage

### 1) Make prompts
Concatenate CSVs in `simulation_data/raw`:
```bash
python scripts/make_prompts.py   --input-dir simulation_data/raw   --output-dir output/<RUN_ID>/raw
# writes: output/<RUN_ID>/raw/prompts.csv
```

Auto-generate via OpenAI + Wikipedia (requires `OPENAI_API_KEY`):
```bash
python scripts/make_prompts.py   --config configs/prompts.yaml   --output-dir output/<RUN_ID>/raw
```

### 2) Generate responses (+ label)
```bash
# Hugging Face (default: gpt2)
python scripts/generate_responses.py   --prompt-files output/<RUN_ID>/raw/prompts_easy.csv output/<RUN_ID>/raw/prompts_hard.csv   --output-dir  output/<RUN_ID>/processed   --model gpt2

# OpenAI
python scripts/generate_responses.py   --prompt-files output/<RUN_ID>/raw/prompts.csv   --output-dir  output/<RUN_ID>/processed   --model gpt-4o --use-openai
```

### 3) Visualize patterns
```bash
python scripts/graph_patterns.py   --input-dir  output/<RUN_ID>/processed   --output-dir output/<RUN_ID>/graphs   --top-keywords 10
```
Creates:
- `accuracy_by_keywords.png`
- `feature_correlation_heatmap.png`
- `graphs/index.html` (embeds `graphs/user/processed/processed_data.html`)

### 4) Compute metrics
```bash
python src/hallu_detector/evaluate.py   --response-files output/<RUN_ID>/processed/responses_labeled_easy.csv                    output/<RUN_ID>/processed/responses_labeled_hard.csv   --metric-files   output/<RUN_ID>/metrics_easy.json                    output/<RUN_ID>/metrics_hard.json
```
Each JSON: `{"hallucination_rate": <float|null>}`

---

## Core Components

### Rule-based detector (`src/hallu_detector/detect.py`)
- **Sentence-BERT** (`all-MiniLM-L6-v2`) for semantic similarity  
- **RoBERTa-MNLI** for textual entailment  
- Public API:
  ```python
  is_hallucinated(answer: str, correct_answer: str,
                  sem_thresh: float = 0.75,
                  nli_thresh: float = 0.80) -> bool
  ```

### Generation helpers (`src/hallu_detector/generate.py`)
- `simple_generate_hf(prompt_list, model_name="gpt2")` (Hugging Face)  
- `simple_generate_openai(prompt_list, model=None, ...)` (OpenAI; uses the **old** `openai.ChatCompletion.create`)

### Metrics (`src/hallu_detector/evaluate.py`)
- `compute_metrics(labels)` → hallucination rate  
- CLI function `process_files(...)` reads CSVs and writes JSON metrics.

---

## Train the Prompt-only BERT
`train_model.py` fine-tunes a sequence classifier (BERT) and saves to `./bert_model` (for `/predict`).

Example:
```bash
python train_model.py   --epochs 5   --batch-size 16   --learning-rate 2e-5   --model-dir ./bert_model   --output-dir ./results
```
> Script references HaluEval and the `datasets` library; install `datasets` and ensure your data/flags match the script.

---

## Testing
```bash
python -m unittest tests/test_detect.py
```
Covers baseline behaviors of `is_hallucinated`.

---

## Known Gaps / Caveats
- Several files contain abridged sections (`...`) that you must fill to run end-to-end:
  - `app.py`, `src/hallu_detector/detect.py`, `src/hallu_detector/evaluate.py`, `src/hallu_detector/generate.py`
  - `scripts/generate_responses.py`, `scripts/graph_patterns.py`, `scripts/make_prompts.py`
  - `static/index.html` / `static/script.js` minor polish

---

## Future Work:
- Automate hallucination detection using calibrated uncertainty or output features (self-consistency, log-prob stats, abstention).  
- Test additional models and compare results.  
- Apply/adapt techniques in high-risk fields (medical, legal).  
- Harden the detector and complete the abridged sections.

---

## Important Note
In order for the hallucination detection to work at it's intended function you must already have the SentenceTransformers models "all-MiniLM-L6-v2" and "roberta-large-mnli" downloaded on your computer.

---

## Citation:
If you use or extend this work, please cite the repository and this README.
