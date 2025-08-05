// static/script.js

/**
 * Helper to POST JSON and return the parsed response.
 */
async function postJSON(url, data) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return res.json();
}

// 1) Detect with Ground Truth
document.getElementById('detect-btn').onclick = async () => {
  const responseText = document.getElementById('detect-response').value;
  const correctText  = document.getElementById('detect-correct').value;
  const outDiv       = document.getElementById('detect-result');

  outDiv.textContent = 'Checking…';
  try {
    const { hallucinated } = await postJSON('/detect', {
      answer: responseText,
      correct: correctText
    });
    outDiv.textContent = hallucinated
      ? '🔴 Hallucinated'
      : '🟢 Not hallucinated';
  } catch (err) {
    console.error(err);
    outDiv.textContent = '❌ Error – check console';
  }
};

// 2) Predict from Prompt
document.getElementById('predict-btn').onclick = async () => {
  const prompt = document.getElementById('predict-prompt').value;
  const outDiv = document.getElementById('predict-result');

  outDiv.textContent = 'Predicting…';
  try {
    const { hallucination_probability, will_hallucinate } =
      await postJSON('/predict', { prompt });

    outDiv.textContent = `Prob: ${(hallucination_probability * 100).toFixed(1)}% — ` +
      (will_hallucinate ? '🔴 Likely' : '🟢 Unlikely');
  } catch (err) {
    console.error(err);
    outDiv.textContent = '❌ Error – check console';
  }
};

// 3) Full Pipeline
document.getElementById('run-pipeline-btn').onclick = async () => {
  const logsDiv   = document.getElementById('pipeline-logs');
  const graphsDiv = document.getElementById('pipeline-graphs');
  const btn       = document.getElementById('run-pipeline-btn');

  // Reset UI
  logsDiv.textContent   = '';
  graphsDiv.innerHTML   = '';
  btn.disabled          = true;
  btn.textContent       = 'Running…';

  try {
    const model = document.getElementById('pipeline-model').value;
    const res   = await postJSON('/run_pipeline', { model });

    // Show stage logs
    for (const stage in res.logs) {
      const lg = res.logs[stage];
      logsDiv.textContent += `--- ${stage} ---\n${lg.stdout}${lg.stderr}\n\n`;

      if (lg.returncode !== 0) {
        logsDiv.textContent += `⛔ Pipeline failed at ${stage}\n`;
        btn.disabled    = false;
        btn.textContent = 'Run Entire Pipeline';
        return;
      }
    }

    // Render graphs on success
    if (res.stage === 'complete') {
      res.graphs.forEach(fname => {
        const img = document.createElement('img');
        img.src = `/output/${res.run_id}/graphs/${fname}`;
        img.alt = fname;
        graphsDiv.appendChild(img);
      });
    }

  } catch (err) {
    console.error(err);
    logsDiv.textContent = '❌ Error running pipeline – see console';
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Run Entire Pipeline';
  }
};
