#!/usr/bin/env python3
import os
import sys
import uuid
import subprocess
import shutil
from flask import Flask, request, jsonify, send_from_directory
from transformers import pipeline

# — Paths —
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR      = os.path.join(PROJECT_ROOT, 'src')
SCRIPTS_DIR  = os.path.join(PROJECT_ROOT, 'scripts')
STATIC_DIR   = os.path.join(PROJECT_ROOT, 'static')

# allow imports from src/
sys.path.insert(0, SRC_DIR)

app = Flask(__name__)

# — Serve front-end assets —
@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


def run_script(script_path, args=None, cwd=None):
    """Run a Python script and capture output."""
    cmd = ['python', script_path] + (args or [])
    res = subprocess.run(
        cmd,
        cwd=(cwd or PROJECT_ROOT),
        capture_output=True,
        text=True
    )
    return {'stdout': res.stdout, 'stderr': res.stderr, 'returncode': res.returncode}


# 1) Rule-based detect
@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json() or {}
    ans  = data.get('answer', '')
    corr = data.get('correct', '')
    from hallu_detector.detect import is_hallucinated
    hall = is_hallucinated(ans, corr)
    return jsonify({'hallucinated': hall})


# Load BERT classifier for /predict
bert_model_dir = os.path.join(PROJECT_ROOT, 'bert_model')
try:
    bert_classifier = pipeline(
        'text-classification',
        model=bert_model_dir,
        tokenizer=bert_model_dir,
        return_all_scores=False
    )
except Exception:
    bert_classifier = None


# 2) Prompt-only BERT predict
@app.route('/predict', methods=['POST'])
def predict():
    if bert_classifier is None:
        return jsonify({'error': 'BERT model not available.'}), 500
    data   = request.get_json() or {}
    prompt = data.get('prompt', '')
    result = bert_classifier(prompt)[0]
    will_hallucinate = (result['label'] == 'LABEL_1')
    return jsonify({
        'hallucination_probability': result['score'],
        'will_hallucinate': will_hallucinate
    })


@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    data       = request.get_json() or {}
    model_name = data.get('model')
    if not model_name:
        return jsonify({'error': 'No model specified.'}), 400

    run_id    = uuid.uuid4().hex
    run_raw   = os.path.join('output', run_id, 'raw')
    run_proc  = os.path.join('output', run_id, 'processed')
    run_graph = os.path.join('output', run_id, 'graphs')
    metrics_dir = os.path.join('output', run_id, 'metrics')
    for d in (run_raw, run_proc, run_graph, metrics_dir):
        os.makedirs(d, exist_ok=True)

    logs = {}

    # --- Step 1: make_prompts ---
    auto_prompts_config = os.path.join(PROJECT_ROOT, 'config', 'prompts_config.yaml')
    logs['make_prompts'] = run_script(
        os.path.join(SCRIPTS_DIR, 'make_prompts.py'),
        ['--config', auto_prompts_config, '--output-dir', run_raw]
    )
    if logs['make_prompts']['returncode'] != 0:
        return jsonify({'stage': 'make_prompts', 'logs': logs}), 500

    expected_auto = os.path.join(run_raw, 'prompts_auto-generated.csv')
    if not os.path.exists(expected_auto):
        logs['make_prompts']['stderr'] += "Expected prompts_auto-generated.csv not found.\n"
        logs['make_prompts']['returncode'] = 1
        return jsonify({'stage': 'make_prompts', 'logs': logs}), 500

    # --- Step 2: prompts_copied ---
    logs['prompts_copied'] = {'stdout': '', 'stderr': '', 'returncode': 0}
    raw_dir = os.path.join(PROJECT_ROOT, 'simulation_data', 'raw')
    if os.path.isdir(raw_dir):
        for fname in os.listdir(raw_dir):
            if fname.endswith('.csv'):
                try:
                    shutil.copyfile(
                        os.path.join(raw_dir, fname),
                        os.path.join(run_raw, fname)
                    )
                    logs['prompts_copied']['stdout'] += f"Copied {fname}\n"
                except Exception as e:
                    logs['prompts_copied']['stderr'] += f"{e}\n"
                    logs['prompts_copied']['returncode'] = 1
    else:
        logs['prompts_copied']['stdout'] = "No additional prompts directory found.\n"
    if logs['prompts_copied']['returncode'] != 0:
        return jsonify({'stage': 'prompts_copied', 'logs': logs}), 500

    # --- Step 3: generate_responses ---
    prompt_files = [os.path.join(run_raw, f) for f in os.listdir(run_raw) if f.endswith('.csv')]
    if not prompt_files:
        return jsonify({'stage': 'generate_responses', 'error': 'No prompt CSVs found.'}), 500
    logs['generate_responses'] = run_script(
        os.path.join(SCRIPTS_DIR, 'generate_responses.py'),
        ['--prompt-files', *prompt_files,
         '--output-dir', run_proc,
         '--model', model_name]
    )
    if logs['generate_responses']['returncode'] != 0:
        return jsonify({'stage': 'generate_responses', 'logs': logs}), 500

    # --- Step 4: analyze_patterns ---
    logs['analyze_patterns'] = run_script(
        os.path.join(SCRIPTS_DIR, 'graph_patterns.py'),
        ['--input-dir', run_proc, '--output-dir', run_graph]
    )
    if logs['analyze_patterns']['returncode'] != 0:
        return jsonify({'stage': 'analyze_patterns', 'logs': logs}), 500
    if not any(os.scandir(run_graph)):
        logs['analyze_patterns']['stderr'] += "No graphs produced.\n"
        logs['analyze_patterns']['returncode'] = 1
        return jsonify({'stage': 'analyze_patterns', 'logs': logs}), 500

    # --- Step 5: evaluate_metrics ---
    response_csvs = [os.path.abspath(os.path.join(run_proc, f))
                    for f in os.listdir(run_proc)
                    if f.lower().endswith('.csv')]
    if not response_csvs:
        return jsonify({'stage': 'evaluate_metrics', 'error': 'No response CSV files found.'}), 500
    metric_files = [os.path.join(metrics_dir, os.path.splitext(os.path.basename(r))[0] + '_metrics.json')
                    for r in response_csvs]
    logs['evaluate_metrics'] = run_script(
        os.path.join(SRC_DIR, 'hallu_detector', 'evaluate.py'),
        ['--response-files', *response_csvs,
         '--metric-files', *metric_files]
    )
    if logs['evaluate_metrics']['returncode'] != 0:
        return jsonify({'stage': 'evaluate_metrics', 'logs': logs}), 500

    return jsonify({
        'stage': 'complete',
        'run_id': run_id,
        'logs': logs,
        'graphs': sorted(os.listdir(run_graph)),
        'metrics': sorted(os.listdir(metrics_dir)),
        'processed_csvs': sorted(f for f in os.listdir(run_proc) if f.endswith('.csv'))
    })


if __name__ == '__main__':
    app.run(debug=True)