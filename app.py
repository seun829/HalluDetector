#!/usr/bin/env python3
import os
import uuid
import subprocess
import shutil
from flask import Flask, request, jsonify, send_from_directory
from transformers import pipeline

# — Paths —
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_DIR  = os.path.join(PROJECT_ROOT, 'scripts')
STATIC_DIR   = os.path.join(PROJECT_ROOT, 'static')

# — Helper to run external Python scripts —
def run_script(script_name, args=None, cwd=None):
    cmd = ['python', os.path.join(SCRIPTS_DIR, script_name)] + (args or [])
    res = subprocess.run(cmd, cwd=(cwd or PROJECT_ROOT), capture_output=True, text=True)
    return {'stdout': res.stdout, 'stderr': res.stderr, 'returncode': res.returncode}

# — Load fine-tuned BERT classifier —
bert_model_dir = os.path.join(PROJECT_ROOT, 'bert_model')
try:
    bert_classifier = pipeline(
        'text-classification', model=bert_model_dir,
        tokenizer=bert_model_dir, return_all_scores=False
    )
except Exception:
    bert_classifier = None

# — Flask setup —
app = Flask(__name__)

# — Serve front-end assets —
@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

# — 1) Rule-based detect —
@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json() or {}
    ans  = data.get('answer', '')
    corr = data.get('correct', '')
    from hallu_detector.detect import is_hallucinated
    hall = is_hallucinated(ans, corr)
    return jsonify({'hallucinated': hall})

# — 2) Prompt-only BERT predict —
@app.route('/predict', methods=['POST'])
def predict():
    if bert_classifier is None:
        return jsonify({'error': 'BERT model not found.'}), 500
    data = request.get_json() or {}
    prompt = data.get('prompt', '')
    result = bert_classifier(prompt)[0]
    hall = (result['label'] == 'LABEL_1')
    return jsonify({'hallucination_probability': result['score'], 'will_hallucinate': hall})

# — 3) Full pipeline —
@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    data       = request.get_json() or {}
    model_name = data.get('model')
    if not model_name:
        return jsonify({'error': 'No model specified.'}), 400

    # Use existing prompts from simulation_data/raw
    raw_prompt_dir = os.path.join(PROJECT_ROOT, 'simulation_data', 'raw')

    run_id    = uuid.uuid4().hex
    run_raw   = os.path.join(STATIC_DIR, 'output', run_id, 'raw')
    run_proc  = os.path.join(STATIC_DIR, 'output', run_id, 'processed')
    run_graph = os.path.join(STATIC_DIR, 'output', run_id, 'graphs')
    for d in (run_raw, run_proc, run_graph):
        os.makedirs(d, exist_ok=True)

    logs = {}
    # 3a) Copy existing prompts
    logs['prompts_copied'] = {'stdout': '', 'stderr': '', 'returncode': 0}
    for fname in os.listdir(raw_prompt_dir):
        if fname.endswith('.csv'):
            src = os.path.join(raw_prompt_dir, fname)
            dst = os.path.join(run_raw, fname)
            try:
                shutil.copyfile(src, dst)
                logs['prompts_copied']['stdout'] += f"Copied {fname}\n"
            except Exception as e:
                logs['prompts_copied']['stderr'] += f"Error copying {fname}: {e}\n"
                logs['prompts_copied']['returncode'] = 1
    if logs['prompts_copied']['returncode'] != 0:
        return jsonify({'stage': 'prompts_copied', 'logs': logs}), 500

    # 3b) Generate responses
    prompt_files = [os.path.join(run_raw, f) for f in os.listdir(run_raw) if f.endswith('.csv')]
    logs['generate_responses'] = run_script(
        'generate_responses.py',
        ['--prompt-files', *prompt_files, '--output-dir', run_proc, '--model', model_name]
    )
    if logs['generate_responses']['returncode'] != 0:
        return jsonify({'stage': 'generate_responses', 'logs': logs}), 500

    # 3c) Analyze patterns
    logs['analyze_patterns'] = run_script(
        'graph_patterns.py',
        ['--input-dir', run_proc, '--output-dir', run_graph]
    )
    if logs['analyze_patterns']['returncode'] != 0:
        return jsonify({'stage': 'analyze_patterns', 'logs': logs}), 500

    graphs = sorted(os.listdir(run_graph))

    # 3d) Evaluate metrics
    response_csvs = [os.path.join(run_proc, f) for f in os.listdir(run_proc) if f.endswith('.csv')]
    metrics_dir   = os.path.join(STATIC_DIR, 'output', run_id, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    metric_files = [
        os.path.join(metrics_dir, os.path.splitext(os.path.basename(r))[0] + '_metrics.json')
        for r in response_csvs
    ]
    logs['evaluate_metrics'] = run_script(
        'evaluate.py',
        ['--response-files', *response_csvs, '--metric-files', *metric_files]
    )
    if logs['evaluate_metrics']['returncode'] != 0:
        return jsonify({'stage': 'evaluate_metrics', 'logs': logs}), 500

    metrics = sorted(os.listdir(metrics_dir))

    return jsonify({
        'stage':    'complete',
        'run_id':   run_id,
        'logs':     logs,
        'graphs':   graphs,
        'metrics':  metrics
    })

if __name__ == '__main__':
    app.run(debug=True)
