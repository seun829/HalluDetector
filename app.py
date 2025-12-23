#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
app.py — Robust pipeline server for hallucination detection experiments.

Endpoints
- GET  /                    -> serves static/index.html
- GET  /<path>              -> serves assets from ./static
- GET  /health              -> liveness check
- POST /detect              -> rule-based detection (answer vs. correct)
- POST /predict             -> prompt-only BERT classifier (local model folder)
- POST /run_pipeline        -> end-to-end run (prompts -> responses -> graphs -> metrics)

Design goals
- Zero-silent-fail: every stage validates its outputs and returns structured logs.
- Compatible with scripts/make_prompts.py that writes prompts_auto-generated.csv with 'prompt' (and alias 'text').
- Uses sys.executable to invoke child Python scripts to avoid venv mismatches.
"""

from __future__ import annotations

import os
import sys
import uuid
import json
import shutil
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# ------------------------------------------------------------------------------
# Paths & bootstrap
# ------------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
SCRIPTS_DIR  = os.path.join(PROJECT_ROOT, "scripts")
STATIC_DIR   = os.path.join(PROJECT_ROOT, "static")
CONFIG_DIR   = os.path.join(PROJECT_ROOT, "config")

# Ensure local src/ is importable
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("APP_LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s: %(message)s",
)

# ------------------------------------------------------------------------------
# Flask
# ------------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def run_script(script_path: str, args: Optional[List[str]] = None, cwd: Optional[str] = None,
               timeout: Optional[int] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """
    Run a Python script with the same interpreter, capture stdout/stderr, and return code.
    """
    cmd = [sys.executable, script_path] + (args or [])
    try:
        res = subprocess.run(
            cmd,
            cwd=(cwd or PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **(env or {})},
        )
        return {
            "cmd": " ".join(cmd),
            "stdout": res.stdout,
            "stderr": res.stderr,
            "returncode": res.returncode,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "cmd": " ".join(cmd),
            "stdout": e.stdout or "",
            "stderr": f"TIMED OUT after {timeout}s\n" + (e.stderr or ""),
            "returncode": 124,
        }
    except Exception as e:
        return {
            "cmd": " ".join(cmd),
            "stdout": "",
            "stderr": f"FAILED to execute: {e}",
            "returncode": 127,
        }


def _validate_prompts_csv(path: str) -> Tuple[bool, str]:
    """
    Ensure the prompts CSV exists, is non-empty, readable, and has a prompt/text column with at least one non-empty value.
    """
    if not os.path.exists(path):
        return False, "prompts_auto-generated.csv not found."
    if os.path.getsize(path) == 0:
        return False, "prompts_auto-generated.csv is empty (0 bytes)."
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, f"Failed to read prompts CSV: {e}"
    col = "prompt" if "prompt" in df.columns else ("text" if "text" in df.columns else None)
    if col is None:
        return False, "prompts CSV missing required 'prompt' (or 'text') column."
    # Non-empty rows?
    non_empty = df[col].dropna().astype(str).str.strip()
    if not (non_empty != "").any():
        return False, "prompts CSV has no non-empty prompts."
    return True, "ok"


def _list_files(d: str, exts: Optional[Tuple[str, ...]] = None) -> List[str]:
    if not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            if not exts or name.lower().endswith(exts):
                out.append(name)
    return sorted(out)


# ------------------------------------------------------------------------------
# Static / Health
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename: str):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/health")
def health():
    return jsonify({"ok": True, "time": _now()})


# ------------------------------------------------------------------------------
# /detect (rule-based)
# ------------------------------------------------------------------------------
@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(silent=True) or {}
    ans  = str(data.get("answer", "")).strip()
    corr = str(data.get("correct", "")).strip()

    # Lazy import to avoid hard crash if module is missing during early development
    try:
        from hallu_detector.detect import is_hallucinated  # type: ignore
    except Exception as e:
        return jsonify({"error": f"Rule detector unavailable: {e}"}), 500

    try:
        hall = bool(is_hallucinated(ans, corr))
    except Exception as e:
        return jsonify({"error": f"Detection failed: {e}"}), 500

    return jsonify({"hallucinated": hall})


# ------------------------------------------------------------------------------
# /predict (prompt-only BERT classifier)
# ------------------------------------------------------------------------------
# Try to lazily load a local text-classification pipeline from ./bert_model
_BERT = None
_BERT_ERR = None


def _load_bert():
    global _BERT, _BERT_ERR
    if _BERT is not None or _BERT_ERR is not None:
        return
    model_dir = os.getenv("BERT_MODEL_DIR", os.path.join(PROJECT_ROOT, "bert_model"))
    try:
        from transformers import pipeline  # lazy import
        _BERT = pipeline(
            "text-classification",
            model=model_dir,
            tokenizer=model_dir,
            return_all_scores=False,
        )
    except Exception as e:
        _BERT = None
        _BERT_ERR = f"Failed to load BERT classifier from {model_dir}: {e}"
        logging.warning(_BERT_ERR)


@app.route("/predict", methods=["POST"])
def predict():
    _load_bert()
    if _BERT is None:
        return jsonify({"error": _BERT_ERR or "BERT model not available."}), 500

    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    try:
        result = _BERT(prompt)[0]  # {'label': 'LABEL_0/1', 'score': float}
        will_hallucinate = (result.get("label") == "LABEL_1")
        return jsonify(
            {
                "hallucination_probability": float(result.get("score", 0.0)),
                "will_hallucinate": bool(will_hallucinate),
                "raw": result,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


# ------------------------------------------------------------------------------
# /run_pipeline (end-to-end orchestration)
# ------------------------------------------------------------------------------
@app.route("/run_pipeline", methods=["POST"])
def run_pipeline():
    data = request.get_json(silent=True) or {}
    model_name = data.get("model")
    if not model_name:
        return jsonify({"error": "No model specified. Provide {'model': '<name>'} in JSON body."}), 400

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_root = os.path.join(PROJECT_ROOT, "output", run_id)
    run_raw = os.path.join(run_root, "raw")
    run_proc = os.path.join(run_root, "processed")
    run_graph = os.path.join(run_root, "graphs")
    metrics_dir = os.path.join(run_root, "metrics")

    for d in (run_raw, run_proc, run_graph, metrics_dir):
        os.makedirs(d, exist_ok=True)

    logs: Dict[str, Dict[str, object]] = {}

    # ------------------- Step 1: make_prompts -------------------
    prompts_config = os.path.join(CONFIG_DIR, "prompts_config.yaml")
    make_prompts_py = os.path.join(SCRIPTS_DIR, "make_prompts.py")
    make_args: List[str] = ["--output-dir", run_raw]
    if os.path.exists(prompts_config):
        make_args = ["--config", prompts_config, "--output-dir", run_raw]

    logs["make_prompts"] = run_script(make_prompts_py, make_args, cwd=PROJECT_ROOT, timeout=600)
    expected_auto = os.path.join(run_raw, "prompts_auto-generated.csv")
    ok, msg = _validate_prompts_csv(expected_auto)
    if logs["make_prompts"]["returncode"] != 0 or not ok:
        logs["make_prompts"]["stderr"] = (logs["make_prompts"].get("stderr") or "") + f"\nValidation: {msg}\n"
        logs["make_prompts"]["returncode"] = logs["make_prompts"].get("returncode", 1) or 1
        return jsonify({"stage": "make_prompts", "run_id": run_id, "logs": logs}), 500

    # ------------------- Step 2: prompts_copied (optional) -------------------
    logs["prompts_copied"] = {"cmd": "", "stdout": "", "stderr": "", "returncode": 0}
    raw_dir = os.path.join(PROJECT_ROOT, "simulation_data", "raw")
    if os.path.isdir(raw_dir):
        for fname in os.listdir(raw_dir):
            if fname.lower().endswith(".csv"):
                src = os.path.join(raw_dir, fname)
                dst = os.path.join(run_raw, fname)
                try:
                    shutil.copyfile(src, dst)
                    logs["prompts_copied"]["stdout"] += f"Copied {fname}\n"
                except Exception as e:
                    logs["prompts_copied"]["stderr"] += f"{e}\n"
                    logs["prompts_copied"]["returncode"] = 1
    else:
        logs["prompts_copied"]["stdout"] = "No simulation_data/raw directory found.\n"
    if logs["prompts_copied"]["returncode"] != 0:
        return jsonify({"stage": "prompts_copied", "run_id": run_id, "logs": logs}), 500

    # ------------------- Step 3: generate_responses -------------------
    prompt_files = [os.path.join(run_raw, f) for f in os.listdir(run_raw) if f.lower().endswith(".csv")]
    if not prompt_files:
        return jsonify({"stage": "generate_responses", "run_id": run_id, "error": "No prompt CSVs found."}), 500

    gen_py = os.path.join(SCRIPTS_DIR, "generate_responses.py")
    gen_args = ["--prompt-files", *prompt_files, "--output-dir", run_proc, "--model", model_name]
    logs["generate_responses"] = run_script(gen_py, gen_args, cwd=PROJECT_ROOT, timeout=3600)
    if logs["generate_responses"]["returncode"] != 0:
        return jsonify({"stage": "generate_responses", "run_id": run_id, "logs": logs}), 500

    # ------------------- Step 4: analyze_patterns (graphs) -------------------
    graph_py = os.path.join(SCRIPTS_DIR, "graph_patterns.py")
    graph_args = ["--input-dir", run_proc, "--output-dir", run_graph]
    logs["analyze_patterns"] = run_script(graph_py, graph_args, cwd=PROJECT_ROOT, timeout=900)
    # Validate at least one image-like artifact exists
    graph_files = _list_files(run_graph, exts=(".png", ".jpg", ".jpeg", ".svg", ".pdf"))
    if logs["analyze_patterns"]["returncode"] != 0 or len(graph_files) == 0:
        logs["analyze_patterns"]["stderr"] = (logs["analyze_patterns"].get("stderr") or "") + "No graphs produced.\n"
        logs["analyze_patterns"]["returncode"] = logs["analyze_patterns"].get("returncode", 1) or 1
        return jsonify({"stage": "analyze_patterns", "run_id": run_id, "logs": logs}), 500

    # ------------------- Step 5: evaluate_metrics -------------------
    resp_csvs = [os.path.join(run_proc, f) for f in os.listdir(run_proc) if f.lower().endswith(".csv")]
    if not resp_csvs:
        return jsonify({"stage": "evaluate_metrics", "run_id": run_id, "error": "No response CSV files found."}), 500

    eval_py = os.path.join(SRC_DIR, "hallu_detector", "evaluate.py")
    metric_files = [os.path.join(metrics_dir, os.path.splitext(os.path.basename(r))[0] + "_metrics.json")
                    for r in resp_csvs]
    eval_args = ["--response-files", *resp_csvs, "--metric-files", *metric_files]
    logs["evaluate_metrics"] = run_script(eval_py, eval_args, cwd=PROJECT_ROOT, timeout=1200)
    if logs["evaluate_metrics"]["returncode"] != 0:
        return jsonify({"stage": "evaluate_metrics", "run_id": run_id, "logs": logs}), 500

    # Success payload
    return jsonify({
        "stage": "complete",
        "run_id": run_id,
        "run_root": run_root,
        "generated": {
            "graphs": graph_files,
            "metrics": _list_files(metrics_dir, exts=(".json",)),
            "processed_csvs": _list_files(run_proc, exts=(".csv",)),
            "raw_csvs": _list_files(run_raw, exts=(".csv",)),
        },
        "logs": logs,
        "time": _now(),
    })


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
