// static/script.js
async function postJSON(url, data) {
  const res = await fetch(url, {
    method: 'POST',
    headers:{ 'Content-Type':'application/json' },
    body: JSON.stringify(data)
  });
  if (!res.ok) {
    const msg = await res.text().catch(()=>String(res.status));
    throw new Error(`HTTP ${res.status}: ${msg}`);
  }
  return res.json();
}

const PIPELINE_STEPS = [
  'make_prompts',
  'prompts_copied',
  'generate_responses',
  'analyze_patterns',
  'evaluate_metrics'
];

const STEP_LABELS = {
  make_prompts: 'Create Auto-Generated Prompts',
  prompts_copied: 'Copied Prompts in the Appropriate Directory',
  generate_responses: 'Generate Responses from the Appropriate Model',
  analyze_patterns: 'Analyze Patterns',
  evaluate_metrics: 'Evaluate Metrics'
};

// 1) Detect with GT
document.getElementById('detect-btn').onclick = async () => {
  const out = document.getElementById('detect-result');
  out.textContent = 'Checking…';
  try {
    const ans  = document.getElementById('detect-response').value;
    const corr = document.getElementById('detect-correct').value;
    const payload = await postJSON('/detect',{answer:ans,correct:corr});
    const hallucinated = Boolean(payload && payload.hallucinated);
    out.textContent = hallucinated ? '🔴 Hallucinated' : '🟢 Not hallucinated';
  } catch (e) {
    console.error(e);
    out.textContent = '❌ Error';
  }
};

// 2) Predict BERT
document.getElementById('predict-btn').onclick = async () => {
  const out = document.getElementById('predict-result');
  out.textContent = 'Predicting…';
  try {
    const prompt = document.getElementById('predict-prompt').value;
    const payload = await postJSON('/predict',{prompt});
    const p = Number(payload && payload.hallucination_probability) || 0;
    const will = Boolean(payload && payload.will_hallucinate);
    out.textContent = `Prob: ${(p*100).toFixed(1)}% — ` + (will ? '🔴 Likely' : '🟢 Unlikely');
  } catch (e) {
    console.error(e);
    out.textContent = '❌ Error';
  }
};

// 3) Full Pipeline
document.getElementById('run-pipeline-btn').onclick = async () => {
  const stepsDiv  = document.getElementById('pipeline-steps');
  const logsDiv   = document.getElementById('pipeline-logs');
  const graphsDiv = document.getElementById('pipeline-graphs');
  const procDiv   = document.getElementById('pipeline-processed');
  const btn       = document.getElementById('run-pipeline-btn');

  stepsDiv.innerHTML  = '';
  logsDiv.textContent = '';
  graphsDiv.innerHTML = '';
  procDiv.innerHTML   = '';
  btn.disabled        = true;
  btn.textContent     = 'Running…';

  // build step list (always show labels)
  for (const step of PIPELINE_STEPS) {
    const li = document.createElement('li');
    li.id = `step-${step}`;
    const label = STEP_LABELS[step] || step;
    li.textContent = `⚪ ${label}`;
    li.classList.add('step-pending');
    stepsDiv.appendChild(li);
  }


  try {
    const model = document.getElementById('pipeline-model').value;
    const res   = await postJSON('/run_pipeline', { model });

    // ensure expected shapes
    const logs   = (res && res.logs) || {};
    const graphs = Array.isArray(res && res.graphs) ? res.graphs : [];
    const proc   = Array.isArray(res && res.processed_csvs) ? res.processed_csvs : [];
    const runId  = (res && res.run_id) || '';

    // update steps & logs

    for (const step of PIPELINE_STEPS) {
      const lg = logs[step] || {};
      const ok = (lg && lg.returncode === 0);

      const li = document.getElementById(`step-${step}`);
      if (li) {
        const label = STEP_LABELS[step] || step;
        li.textContent = `${ok ? '🟢' : '🔴'} ${label}`;
        li.classList.remove('step-pending', 'step-success', 'step-fail');
        li.classList.add(ok ? 'step-success' : 'step-fail');
      }

      const stdOut = (lg && lg.stdout) ? String(lg.stdout) : '';
      const stdErr = (lg && lg.stderr) ? String(lg.stderr) : '';
      logsDiv.textContent += `--- ${STEP_LABELS[step] || step} (${step}) ---\n${stdOut}${stdErr}\n\n`;

      if (!ok) break;
    }

    // only attempt these if the pipeline completed
    if (res && res.stage === 'complete') {
      // graphs (guarded)
      for (const fn of graphs) {
        if (!fn) continue;
        const img = document.createElement('img');
        img.alt = fn;
        img.loading = 'lazy';
        img.src = `/output/${runId}/graphs/${encodeURIComponent(fn)}`;
        graphsDiv.appendChild(img);
      }

      // processed CSVs → inline tables (guarded)
      if (proc.length > 0) {
        const h3 = document.createElement('h3');
        h3.textContent = 'Labeled Responses';
        procDiv.appendChild(h3);

        for (const fn of proc) {
          try {
            const url = `/output/${runId}/processed/${encodeURIComponent(fn)}`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`Fetch ${fn} failed`);
            const txt = (await resp.text()).trim();
            if (!txt) continue;

            // naive CSV split; OK for our simple files (no quoted commas)
            const lines = txt.split(/\r?\n/);
            if (lines.length === 0) continue;

            const headers = lines[0].split(',');
            const rows = lines.slice(1);

            const table = document.createElement('table');
            table.classList.add('csv-table');

            const thead = table.createTHead();
            const thr = thead.insertRow();
            for (const h of headers) {
              const th = document.createElement('th');
              th.textContent = h;
              thr.appendChild(th);
            }

            const tb = table.createTBody();
            for (const r of rows) {
              if (!r) continue;
              const tr = tb.insertRow();
              for (const c of r.split(',')) {
                const td = tr.insertCell();
                td.textContent = c;
              }
            }

            const caption = document.createElement('div');
            caption.classList.add('csv-caption');
            caption.textContent = fn;
            procDiv.appendChild(caption);
            procDiv.appendChild(table);
          } catch (e) {
            const errP = document.createElement('p');
            errP.textContent = `Could not render ${fn}: ${e.message || e}`;
            procDiv.appendChild(errP);
          }
        }
      }
    }
  } catch (err) {
    console.error("Pipeline error:", err);
    const logsDiv = document.getElementById('pipeline-logs');
    logsDiv.textContent = `Pipeline error: ${err && err.message ? err.message : String(err)}`;
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Run Entire Pipeline';
  }
};
