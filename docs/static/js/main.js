// ---------------------------------------------------------------------------
// Quantitative results (from the paper, Tables 1 and 2)
// ---------------------------------------------------------------------------
const SPHERES = ['Diffuse', 'Matte', 'Mirror'];

// Table 1: global illumination from a single image (all metrics: lower is better)
const GLOBAL_TABLE = {
  metrics: [
    { name: 'si-RMSE', higherBetter: false },
    { name: 'Angular Error', higherBetter: false },
    { name: 'Normalized RMSE', higherBetter: false },
  ],
  groups: [
    {
      name: 'Laval Indoor',
      rows: [
        ['StyleLight',     [0.064, 0.152, 0.265, 3.874, 4.606, 10.974, 0.232, 0.411, 0.520]],
        ['DiffusionLight', [0.075, 0.163, 0.282, 3.415, 4.157, 6.483, 0.246, 0.391, 0.457]],
        ['LuxDiT',         [0.056, 0.140, 0.274, 2.759, 3.875, 6.099, 0.207, 0.388, 0.546]],
        ['Ours',           [0.046, 0.120, 0.252, 2.700, 3.773, 6.257, 0.180, 0.319, 0.411]],
      ],
    },
    {
      name: 'Laval Outdoor',
      rows: [
        ['StyleLight',     [0.131, 0.286, 0.456, 3.524, 4.864, 7.609, 0.293, 0.377, 0.475]],
        ['DiffusionLight', [0.097, 0.252, 0.431, 2.838, 4.237, 6.835, 0.185, 0.320, 0.434]],
        ['LuxDiT',         [0.132, 0.281, 0.456, 2.794, 4.149, 6.622, 0.237, 0.351, 0.460]],
        ['Ours',           [0.087, 0.221, 0.403, 2.940, 3.898, 6.309, 0.162, 0.297, 0.423]],
      ],
    },
  ],
};

// Table 2: spatially-varying lighting estimation
const SV_TABLE = {
  metrics: [
    { name: 'RMSE', higherBetter: false },
    { name: 'si-RMSE', higherBetter: false },
    { name: 'Angular Error', higherBetter: false },
    { name: 'SSIM', higherBetter: true },
  ],
  groups: [
    {
      name: 'Infinigen (single-view)',
      rows: [
        ['DiffusionLight', [0.157, 0.228, 0.460, 0.066, 0.180, 0.428, 2.935, 4.053, 5.394, 0.947, 0.679, 0.471]],
        ['4DLighting',     [0.164, 0.207, 0.430, 0.043, 0.134, 0.381, 1.457, 2.418, 3.595, 0.972, 0.754, 0.533]],
        ['Ours',           [0.110, 0.149, 0.369, 0.042, 0.108, 0.329, 2.762, 3.382, 4.594, 0.973, 0.832, 0.679]],
      ],
    },
    {
      name: 'In-the-wild (single-view)',
      rows: [
        ['DiffusionLight', [0.172, 0.236, 0.351, 0.081, 0.193, 0.310, 1.592, 2.681, 3.856, 0.923, 0.654, 0.447]],
        ['4DLighting',     [0.141, 0.190, 0.289, 0.048, 0.132, 0.238, 1.186, 1.904, 2.865, 0.969, 0.755, 0.519]],
        ['Ours',           [0.153, 0.176, 0.276, 0.047, 0.114, 0.225, 1.345, 2.379, 3.563, 0.964, 0.797, 0.616]],
      ],
    },
    {
      name: 'Laval Indoor SV (single-view)',
      rows: [
        ['DiffusionLight', [0.262, 0.289, 0.365, 0.072, 0.169, 0.253, 2.882, 3.604, 4.316, 0.931, 0.728, 0.572]],
        ['4DLighting',     [0.169, 0.247, 0.322, 0.085, 0.180, 0.251, 3.099, 3.753, 4.459, 0.915, 0.698, 0.577]],
        ['Ours',           [0.183, 0.216, 0.298, 0.069, 0.154, 0.248, 2.572, 3.493, 4.167, 0.939, 0.769, 0.599]],
      ],
    },
    {
      name: 'Infinigen (multi-view)',
      rows: [
        ['4DLighting',     [0.213, 0.231, 0.443, 0.037, 0.124, 0.375, 1.596, 2.397, 3.568, 0.978, 0.762, 0.511]],
        ['Ours',           [0.080, 0.095, 0.253, 0.024, 0.058, 0.210, 1.925, 2.416, 3.340, 0.988, 0.942, 0.849]],
      ],
    },
    {
      name: 'In-the-wild (multi-view)',
      rows: [
        ['4DLighting',     [0.182, 0.204, 0.291, 0.038, 0.111, 0.210, 1.168, 1.784, 2.728, 0.972, 0.773, 0.505]],
        ['Ours',           [0.104, 0.118, 0.172, 0.027, 0.065, 0.143, 1.522, 1.898, 3.003, 0.984, 0.907, 0.788]],
      ],
    },
  ],
};

function decimals(v) {
  const s = String(v);
  const i = s.indexOf('.');
  return i < 0 ? 0 : s.length - i - 1;
}

function renderTable(tableId, spec) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const nSph = SPHERES.length;

  let head = '<thead><tr><th class="left" rowspan="2">Dataset</th><th class="left" rowspan="2">Method</th>';
  spec.metrics.forEach((m) => {
    head += `<th class="sep" colspan="${nSph}">${m.name} ${m.higherBetter ? '&uarr;' : '&darr;'}</th>`;
  });
  head += '</tr><tr>';
  spec.metrics.forEach(() => {
    SPHERES.forEach((s, j) => { head += `<th${j === 0 ? ' class="sep"' : ''}>${s}</th>`; });
  });
  head += '</tr></thead>';

  let body = '<tbody>';
  spec.groups.forEach((g) => {
    const nCols = g.rows[0][1].length;
    // Format each value with the precision used in the paper (at least 3 decimals)
    const fmt = (v, c) => {
      const d = Math.max(3, ...g.rows.map((r) => decimals(r[1][c])));
      return v.toFixed(d);
    };
    const best = [];
    for (let c = 0; c < nCols; c++) {
      const hb = spec.metrics[Math.floor(c / nSph)].higherBetter;
      const vals = g.rows.map((r) => r[1][c]);
      best.push(hb ? Math.max(...vals) : Math.min(...vals));
    }
    g.rows.forEach((r, ri) => {
      const [method, vals] = r;
      const cls = [ri === 0 ? 'group-start' : '', method === 'Ours' ? 'ours' : ''].join(' ').trim();
      body += `<tr${cls ? ` class="${cls}"` : ''}>`;
      if (ri === 0) body += `<td class="group" rowspan="${g.rows.length}">${g.name}</td>`;
      body += `<td class="left">${method === 'Ours' ? '<b>Ours</b>' : method}</td>`;
      vals.forEach((v, c) => {
        const classes = [];
        if (c % nSph === 0) classes.push('sep');
        if (v === best[c]) classes.push('best');
        body += `<td${classes.length ? ` class="${classes.join(' ')}"` : ''}>${fmt(v, c)}</td>`;
      });
      body += '</tr>';
    });
  });
  body += '</tbody>';

  table.innerHTML = head + body;
}

// ---------------------------------------------------------------------------
// Lightbox
// ---------------------------------------------------------------------------
function initLightbox() {
  const box = document.getElementById('lightbox');
  const img = document.getElementById('lightbox-img');
  const close = () => { box.classList.remove('open'); box.setAttribute('aria-hidden', 'true'); img.src = ''; };

  document.querySelectorAll('img.zoomable').forEach((el) => {
    el.addEventListener('click', () => {
      img.src = el.src;
      img.alt = el.alt;
      box.classList.add('open');
      box.setAttribute('aria-hidden', 'false');
    });
  });
  box.addEventListener('click', close);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
}

// ---------------------------------------------------------------------------
// BibTeX copy
// ---------------------------------------------------------------------------
function initCopy() {
  const btn = document.getElementById('copy-bibtex');
  const text = document.getElementById('bibtex-text');
  if (!btn || !text) return;
  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(text.innerText);
      btn.textContent = 'Copied!';
    } catch (e) {
      const range = document.createRange();
      range.selectNodeContents(text);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      btn.textContent = 'Press Ctrl+C';
    }
    setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  renderTable('table-global', GLOBAL_TABLE);
  renderTable('table-sv', SV_TABLE);
  initLightbox();
  initCopy();
});
