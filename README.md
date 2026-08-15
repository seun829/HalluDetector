# HalluDetector

HalluDetector is a reference-based research toolkit for measuring explicit contradictions in language-model answers. Given a model response and a known correct answer, it combines deterministic checks, sentence embeddings, and natural-language inference (NLI) to look for positive evidence that the response conflicts with the reference.

The repository includes the detector, OpenAI and Hugging Face generation helpers, an end-to-end Flask pipeline, graph and metric scripts, a human-audit workflow, current experiment artifacts, and archived datasets from earlier experiments.

> HalluDetector is not an open-world fact checker. It cannot verify a claim without a supplied reference answer, and a non-hallucinated label means only that the detector found no supported contradiction under its current rules.

## What it does

- Normalizes text, scientific notation, coordinates, short codes, and numeric answers.
- Detects numeric conflicts, explicit negation, nonexistence claims, atomic answer mismatches, and strong NLI contradictions.
- Uses `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity and `roberta-large-mnli` for bidirectional NLI.
- Generates answers through OpenAI models or local Hugging Face causal language models.
- Produces labeled CSVs, summary metrics, keyword/template charts, a correlation heatmap, and browsable HTML output.
- Builds balanced samples for human review of detector decisions.

The decision policy is intentionally conservative: incompleteness alone is not considered a hallucination. See [docs/methodology.md](docs/methodology.md) for the full rules, thresholds, and limitations.

## Repository structure

```text
HalluDetector/
|-- app.py                         # Flask API and end-to-end pipeline runner
|-- requirements.txt              # Python dependencies
|-- audit/
|   `-- audit_set.csv              # Human-audit dataset
|-- docs/
|   |-- methodology.md             # Detector design and decision policy
|   `-- paper.tex                  # Research paper source
|-- scripts/
|   |-- generate_responses.py      # Generate, normalize, and label model answers
|   |-- graph_patterns.py          # Create charts, processed tables, and HTML
|   |-- make_audit_set.py          # Sample labeled outputs for human review
|   `-- make_prompts.py            # Generate prompt/reference pairs with OpenAI
|-- simulation_data/
|   `-- raw/
|       `-- hard_prompts3.csv      # Active 400-question benchmark
|-- src/
|   `-- hallu_detector/
|       |-- __init__.py            # Public package exports
|       |-- detect.py              # Reference-based contradiction detector
|       |-- download.py            # Optional model pre-download helper
|       |-- evaluate.py            # Rates and bootstrap confidence intervals
|       `-- generate.py            # OpenAI and Hugging Face generation backends
|-- static/
|   |-- index.html                 # Browser interface
|   |-- script.js                  # Front-end behavior
|   `-- style.css                  # Front-end styles
|-- tests/
|   `-- test_detect.py             # Detector unit tests
|-- output/                        # Current experiment runs
|-- old_prompts/                   # Archived factual, riddle, and hard prompts
`-- output_old/                    # Archived legacy experiment runs
```

Each experiment in `output/` follows this layout:

```text
output/<run-directory>/
|-- raw/                            # Prompt CSVs used for the run
|-- processed/                      # Responses and detector labels
|-- metrics/                        # Hallucination-rate JSON files
`-- graphs/
    |-- hallucinations_by_template.png
    |-- hallucinations_by_keywords.png
    |-- feature_correlation_heatmap.png
    |-- index.html
    `-- user/processed/             # Combined CSV and HTML tables
```

The checked-in runs use model/timestamp names such as `gpt4o_20260703_212121_9c56c41f`. New Flask runs use a timestamp and short UUID. `output/` contains the newer `hard_prompts3` experiments, while `output_old/` preserves earlier runs across the factual, riddle, auto-generated, and previous hard-prompt sets.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone <repository-url>
cd HalluDetector
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The detector loads its two Hugging Face models on first use. They can be downloaded ahead of time with:

```bash
python src/hallu_detector/download.py
```

Model downloads require an internet connection and enough local storage. Detection is strict by default: if an enabled embedding or NLI dependency is unavailable, the detector raises an error instead of silently switching methods.

### OpenAI setup

An OpenAI API key is required only for OpenAI generation and automatic prompt creation. API use may incur charges.

```bash
# macOS/Linux
export OPENAI_API_KEY="your-key"

# Windows PowerShell
$env:OPENAI_API_KEY="your-key"
```

## Quick start

### Detect one response

Use the package directly from the repository root:

```python
from src.hallu_detector.detect import detect_details, is_hallucinated

flag = is_hallucinated("London", "Paris")
details = detect_details("London", "Paris")

print(flag)               # True
print(details["reason"])  # Decision rationale
```

`detect_details` returns the final decision plus normalization, numeric, embedding, NLI, and heuristic diagnostics. `is_hallucinated` returns only the boolean decision.

### Run the web server

```bash
python app.py
```

Open <http://127.0.0.1:5000>. The server uses `HOST`, `PORT`, `FLASK_DEBUG`, and `APP_LOGLEVEL` environment variables when set.

Implemented routes:

- `GET /` serves the browser interface.
- `GET /health` returns a liveness check.
- `POST /detect` accepts `{"answer": "London", "correct": "Paris"}`.
- `POST /run_pipeline` accepts `{"model": "gpt-4o"}` and runs the active CSVs from `simulation_data/raw/` through generation, detection, graphs, and metrics.

Pipeline runs are written to `output/<timestamp>_<run-id>/`. The endpoint returns per-stage logs and a `generated` object listing the raw CSVs, processed CSVs, graphs, and metric files.

## Command-line workflow

The Flask pipeline uses the checked-in prompt files. The same stages can be run independently for more control.

### 1. Generate prompt/reference pairs (optional)

`make_prompts.py` uses the modern OpenAI Python client and writes exactly one file named `prompts_auto-generated.csv`.

```bash
python scripts/make_prompts.py \
  --output-dir output/manual/raw \
  --exact-total 50
```

Optional flags include `--config`, `--seeds`, `--questions-per-seed`, and `--random-seed`. A YAML config can override the defaults in `scripts/make_prompts.py`, including the model, topics, temperatures, token limits, and self-consistency vote count.

### 2. Generate and label responses

OpenAI models are selected automatically when the model name starts with `gpt-`:

```bash
python scripts/generate_responses.py \
  --prompt-files simulation_data/raw/hard_prompts3.csv \
  --output-dir output/manual/processed \
  --model gpt-4o
```

For a Hugging Face causal language model, provide its model ID. Models whose IDs start with `gpt-` require `--use-openai` behavior, so choose a different Hugging Face ID when using the local backend.

```bash
python scripts/generate_responses.py \
  --prompt-files simulation_data/raw/hard_prompts3.csv \
  --output-dir output/manual/processed \
  --model gpt2
```

Instead of `--output-dir`, use `--response-files` to provide one exact output path for each input file. The script accepts common input aliases such as `text` or `question` for `prompt`, and `answer` or `gold` for `correct_answer`.

### 3. Create graphs and HTML

```bash
python scripts/graph_patterns.py \
  --input-dir output/manual/processed \
  --output-dir output/manual/graphs \
  --top-keywords 10
```

The script combines compatible labeled CSVs, then creates template and TF-IDF keyword summaries, a feature correlation heatmap, processed CSV/HTML tables, and `graphs/index.html`.

### 4. Compute metrics

Create `output/manual/metrics/`, then run:

```bash
python src/hallu_detector/evaluate.py \
  --response-files output/manual/processed/hard_responses_labeled3.csv \
  --metric-files output/manual/metrics/hard_responses_labeled3_metrics.json \
  --bootstrap 1000 \
  --seed 1337
```

Without `--bootstrap`, the JSON contains `hallucination_rate`. With bootstrapping enabled, it also contains the sample size, hallucination count, 95% confidence interval, bootstrap sample count, and seed.

### 5. Build a human-audit sample

The audit builder samples approximately equal numbers of detector-positive and detector-negative rows. It refuses to overwrite an existing audit file.

```bash
python scripts/make_audit_set.py \
  --input-root output \
  --pattern "*responses_labeled*.csv" \
  --n 300 \
  --seed 1337 \
  --output audit/audit_sample.csv
```

Reviewers fill in `human_hallucinated` and, optionally, `human_notes`.

## Data formats

### Prompt CSV

The canonical schema is:

```csv
template,id,prompt,correct_answer
geography,1,What is the capital of France?,Paris
```

`prompt` and `correct_answer` are required. `id` is generated when absent, and `template` is derived from the filename when absent.

### Labeled response CSV

`generate_responses.py` writes:

```text
template
id
prompt
model_response
correct_answer
hallucinated
baseline_exact
baseline_embed
baseline_embed_score
baseline_embed_method
reasoning
```

Empty generated answers are treated as abstentions and recorded as non-hallucinations by the batch-generation script.

### Audit CSV

`make_audit_set.py` writes:

```text
source_file
template
id
prompt
correct_answer
model_response
detector_hallucinated
human_hallucinated
human_notes
```

## Testing

```bash
python -m unittest tests/test_detect.py
```

The detector tests load the embedding and NLI models, so the first run may be slower while model files are downloaded and initialized.

## Current limitations

- Detection is reference-based and focuses on explicit inconsistency. It does not perform retrieval, source verification, or general factuality checking.
- A response can be incomplete or unsupported without triggering a contradiction label. Conversely, cautious source-limit language can resemble a contradiction cue on non-atomic prompts.
- The browser's prompt-only prediction panel references a removed `/predict` backend and is not currently functional.
- Pipeline artifacts are created correctly under the repository-level `output/` directory, but the current front end still expects an older response shape and static output path, so generated graphs and CSVs may not render in the browser. They remain available on disk and through the endpoint's JSON response.
- Local Hugging Face generation and NLI can be slow without a GPU. OpenAI runs require network access, a valid model available to the account, and sufficient API quota.

For the project's experimental design and a fuller analysis of detector scope, see [docs/paper.tex](docs/paper.tex).

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

## Citation

If you use or extend HalluDetector, cite this repository and the accompanying paper in `docs/paper.tex`.
