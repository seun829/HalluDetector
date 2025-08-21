// static/script.js

async function postJSON(url, data) {
  const res = await fetch(url, {
    method: 'POST',
    headers:{ 'Content-Type':'application/json' },
    body: JSON.stringify(data)
  });
  return res.json();
}

const PIPELINE_STEPS = [
  'make_prompts',
  'prompts_copied',
  'generate_responses',
  'analyze_patterns',
  'evaluate_metrics'
];

// 1) Detect with GT
document.getElementById('detect-btn').onclick = async () => {
  const out = document.getElementById('detect-result');
  out.textContent = 'Checking…';
  try {
    const ans  = document.getElementById('detect-response').value;
    const corr = document.getElementById('detect-correct').value;
    const { hallucinated } = await postJSON('/detect',{answer:ans,correct:corr});
    out.textContent = hallucinated ? '🔴 Hallucinated' : '🟢 Not hallucinated';
  } catch {
    out.textContent = '❌ Error';
  }
};

// 2) Predict BERT
document.getElementById('predict-btn').onclick = async () => {
  const out = document.getElementById('predict-result');
  out.textContent = 'Predicting…';
  try {
    const prompt = document.getElementById('predict-prompt').value;
    const { hallucination_probability, will_hallucinate } =
      await postJSON('/predict',{prompt});
    out.textContent = `Prob: ${(hallucination_probability*100).toFixed(1)}% — `
      + (will_hallucinate?'🔴 Likely':'🟢 Unlikely');
  } catch {
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

  PIPELINE_STEPS.forEach(step => {
    const li = document.createElement('li');
    li.id          = `step-${step}`;
    li.textContent = `⚪ ${step}`;
    li.classList.add('step-pending');
    stepsDiv.appendChild(li);
  });

  try {
    const model = document.getElementById('pipeline-model').value;
    const res   = await postJSON('/run_pipeline', { model });

    // update steps & logs
    for (const step of PIPELINE_STEPS) {
      const lg = res.logs?.[step] || {};
      const ok = lg.returncode===0;
      const li = document.getElementById(`step-${step}`);
      if (li) {
        li.textContent = `${ok?'✅':'❌'} ${step}`;
        li.classList.replace('step-pending', ok?'step-success':'step-fail');
      }
      logsDiv.textContent += `--- ${step} ---\n${lg.stdout||''}${lg.stderr||''}\n\n`;
      if (!ok) break;
    }

    if (res.stage==='complete') {
      // graphs
      res.graphs.forEach(fn=>{
        const img=document.createElement('img');
        img.src=`/output/${res.run_id}/graphs/${fn}`;
        graphsDiv.appendChild(img);
      });

      // processed CSVs → inline tables
      if (res.processed_csvs?.length) {
        const h3=document.createElement('h3');
        h3.textContent='Labeled Responses';
        procDiv.appendChild(h3);

        for (const fn of res.processed_csvs) {
          const url = `/output/${res.run_id}/processed/${fn}`;
          const txt = await (await fetch(url)).text();
          const [hdr,...rows] = txt.trim().split('\n');
          const headers = hdr.split(',');
          const table = document.createElement('table');
          table.classList.add('csv-table');
          const thead = table.createTHead();
          const thr = thead.insertRow();
          headers.forEach(h=>{
            const th= document.createElement('th');
            th.textContent=h;
            thr.appendChild(th);
          });
          const tb = table.createTBody();
          rows.forEach(r=>{
            const tr= tb.insertRow();
            r.split(',').forEach(c=>{
              const td=tr.insertCell();
              td.textContent=c;
            });
          });
          procDiv.appendChild(table);
        }
      }
    }
  } catch (err) {
    console.error("Pipeline error:", err);
    logsDiv.textContent = `❌ Pipeline error: ${err.message || err}`;
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Run Entire Pipeline';
  }
};
