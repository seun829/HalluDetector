#!/usr/bin/env python3
import os
import sys
import uuid
import subprocess
import pickle
from flask import Flask, request, jsonify, send_from_directory

# — Paths —
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPTS_DIR  = os.path.join(PROJECT_ROOT, 'scripts')
STATIC_DIR   = os.path.join(PROJECT_ROOT, 'static')
MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'model.pkl')

# — Make sure we can import detect.is_hallucinated() —
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from hallu_detector.detect import is_hallucinated

# — Load ML predictor if available —
vectorizer = clf = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    vectorizer, clf = data['vectorizer'], data['clf']

# — Flask setup; serve static/ as your UI —
app = Flask(__name__,
            static_folder=STATIC_DIR,
            static_url_path='')

def run_script(script_name, args=None, cwd=None):
    """Run a script under PROJECT_ROOT/scripts, capture output."""
    cmd = ['python', os.path.join(SCRIPTS_DIR, script_name)] + (args or [])
    res = subprocess.run(cmd,
                         cwd=(cwd or PROJECT_ROOT),
                         capture_output=True,
                         text=True)
    return {
        'stdout':    res.stdout,
        'stderr':    res.stderr,
        'returncode': res.returncode
    }

@app.route('/')
def index():
    return app.send_static_file('index.html')

# — 1) Rule-based detect —
@app.route('/detect', methods=['POST'])
def detect():
    js   = request.get_json() or {}
    ans  = js.get('answer','')
    corr = js.get('correct','')
    hall = is_hallucinated(ans, corr)
    return jsonify({'hallucinated': hall})

# — 2) ML-based predict —
@app.route('/predict', methods=['POST'])
def predict():
    if vectorizer is None or clf is None:
        return jsonify({'error':'Model not trained. Run train_model.py first.'}), 500
    prompt = (request.get_json() or {}).get('prompt','')
    X      = vectorizer.transform([prompt])
    prob   = float(clf.predict_proba(X)[0][1])
    pred   = bool(clf.predict(X)[0])
    return jsonify({
        'hallucination_probability': prob,
        'will_hallucinate': pred
    })

# — 3) Run make_prompts & generate_responses & analyze_patterns in isolation —
@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    run_id = uuid.uuid4().hex
    # create per-run subdirs: raw prompts, processed responses, graphs
    run_raw   = os.path.join(STATIC_DIR, 'output', run_id, 'raw')
    run_proc  = os.path.join(STATIC_DIR, 'output', run_id, 'processed')
    run_graph = os.path.join(STATIC_DIR, 'output', run_id, 'graphs')
    for d in (run_raw, run_proc, run_graph):
        os.makedirs(d, exist_ok=True)

    logs = {}

    # 3a) make_prompts → writes to run_raw
    logs['make_prompts'] = run_script(
        'make_prompts.py',
        ['--output-dir', run_raw]
    )
    if logs['make_prompts']['returncode'] != 0:
        return jsonify({'stage':'make_prompts','logs':logs}), 500

    # 3b) generate_responses → read from run_raw, write to run_proc
    logs['generate_responses'] = run_script(
        'generate_responses.py',
        [
          '--prompt-files', *[
            os.path.join(run_raw, fname)
            for fname in os.listdir(run_raw)
            if fname.endswith('.csv')
          ],
          '--output-dir', run_proc,
          '--model', 'gpt2'
        ]
    )
    if logs['generate_responses']['returncode'] != 0:
        return jsonify({'stage':'generate_responses','logs':logs}), 500

    # 3c) analyze_patterns → reads run_proc & writes graphs into run_graph
    logs['analyze_patterns'] = run_script(
        'graph_patterns.py',
        ['--input-dir', run_proc, '--output-dir', run_graph]
    )
    if logs['analyze_patterns']['returncode'] != 0:
        return jsonify({'stage':'analyze_patterns','logs':logs}), 500

    # 3d) list the graph files for the UI
    graphs = sorted(os.listdir(run_graph))

    return jsonify({
      'stage': 'complete',
      'run_id': run_id,
      'logs': logs,
      'graphs': graphs
    })

# — serve per-run graphs —
@app.route('/output/<run_id>/graphs/<fname>')
def serve_graph(run_id, fname):
    dir_path = os.path.join(STATIC_DIR, 'output', run_id, 'graphs')
    return send_from_directory(dir_path, fname)

if __name__ == '__main__':
    app.run(debug=True)
