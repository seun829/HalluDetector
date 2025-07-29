async function postJSON(url, data) {
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(data)
  });
  return res.json();
}

// 1) Detect
document.getElementById('detect-btn').onclick = async () => {
  const resp = document.getElementById('detect-response').value;
  const corr = document.getElementById('detect-correct').value;
  const out  = document.getElementById('detect-result');
  out.textContent = 'Checking…';
  try {
    const { hallucinated } = await postJSON('/detect', { answer: resp, correct: corr });
    out.textContent = hallucinated ? '🔴 Hallucinated' : '🟢 Not hallucinated';
  } catch {
    out.textContent = '❌ Error – check console';
  }
};

// 2) Predict
document.getElementById('predict-btn').onclick = async () => {
  const prompt = document.getElementById('predict-prompt').value;
  const out    = document.getElementById('predict-result');
  out.textContent = 'Predicting…';
  try {
    const { hallucination_probability, will_hallucinate } =
            await postJSON('/predict', { prompt });
    out.textContent = `Prob: ${(hallucination_probability*100).toFixed(1)}% — ` +
                      (will_hallucinate ? '🔴 Likely' : '🟢 Unlikely');
  } catch {
    out.textContent = '❌ Error – check console';
  }
};

// 3) Full pipeline
document.getElementById('run-pipeline-btn').onclick = async () => {
  const logsDiv   = document.getElementById('pipeline-logs');
  const graphsDiv = document.getElementById('pipeline-graphs');
  const btn       = document.getElementById('run-pipeline-btn');
  logsDiv.textContent = '';
  graphsDiv.innerHTML = '';
  btn.disabled = true;
  btn.textContent = 'Running…';

  try {
    const res = await postJSON('/run_pipeline', {});
    // show logs
    for (let stage in res.logs) {
      const lg = res.logs[stage];
      logsDiv.textContent += `--- ${stage} ---\n${lg.stdout}${lg.stderr}\n\n`;
      if (lg.returncode !== 0) {
        logsDiv.textContent += `⛔ Pipeline failed at ${stage}\n`;
        btn.disabled = false;
        btn.textContent = 'Run Entire Pipeline';
        return;
      }
    }
    // render graphs
    if (res.stage === 'complete') {
      res.graphs.forEach(fname => {
        const img = document.createElement('img');
        img.src = `/output/${res.run_id}/graphs/${fname}`;
        img.alt = fname;
        graphsDiv.appendChild(img);
      });
    }
  } catch (err) {
    logsDiv.textContent = '❌ Error running pipeline – see console';
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Entire Pipeline';
  }
};
