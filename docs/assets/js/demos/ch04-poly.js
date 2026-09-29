// Chapter 04 demo: polynomial degree slider -> watch overfitting.
// Pure JS least squares (Householder QR) + Plotly. Data are simulated
// dose-response points (seeded, so every reader sees the same data).
(function () {
  function init() {
  const el = document.getElementById("ch04-poly-demo");
  if (!el || typeof Plotly === "undefined") return;

  // ---------- seeded random numbers ----------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const rand = mulberry32(42);
  const randn = () => {
    const u = Math.max(rand(), 1e-12), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const curve = (x) => (100 * x) / (2 + x);

  const nTrain = 25, nTest = 120, noise = 8;
  const xTr = Array.from({ length: nTrain }, () => 0.5 + 9.5 * rand()).sort((a, b) => a - b);
  const yTr = xTr.map((x) => curve(x) + noise * randn());
  const xMin = xTr[0], xMax = xTr[nTrain - 1];
  const xTe = Array.from({ length: nTest }, () => xMin + (xMax - xMin) * rand());
  const yTe = xTe.map((x) => curve(x) + noise * randn());
  const scale = (x) => (2 * (x - xMin)) / (xMax - xMin) - 1; // map to [-1, 1]

  // ---------- least squares via Householder QR ----------
  function fitPoly(xs, ys, d) {
    const m = xs.length, n = d + 1;
    const A = xs.map((x) => { const s = scale(x); return Array.from({ length: n }, (_, j) => Math.pow(s, j)); });
    const b = ys.slice();
    for (let k = 0; k < n; k++) {
      let norm = 0;
      for (let i = k; i < m; i++) norm += A[i][k] * A[i][k];
      norm = Math.sqrt(norm);
      if (norm === 0) continue;
      const alpha = A[k][k] > 0 ? -norm : norm;
      const v = new Array(m).fill(0);
      v[k] = A[k][k] - alpha;
      for (let i = k + 1; i < m; i++) v[i] = A[i][k];
      let vv = 0;
      for (let i = k; i < m; i++) vv += v[i] * v[i];
      if (vv === 0) continue;
      for (let j = k; j < n; j++) {
        let dot = 0;
        for (let i = k; i < m; i++) dot += v[i] * A[i][j];
        const f = (2 * dot) / vv;
        for (let i = k; i < m; i++) A[i][j] -= f * v[i];
      }
      let dotb = 0;
      for (let i = k; i < m; i++) dotb += v[i] * b[i];
      const fb = (2 * dotb) / vv;
      for (let i = k; i < m; i++) b[i] -= fb * v[i];
    }
    const coef = new Array(n).fill(0);
    for (let k = n - 1; k >= 0; k--) {
      let s = b[k];
      for (let j = k + 1; j < n; j++) s -= A[k][j] * coef[j];
      coef[k] = s / A[k][k];
    }
    return coef;
  }
  const predict = (coef, x) => { const s = scale(x); let y = 0; for (let j = coef.length - 1; j >= 0; j--) y = y * s + coef[j]; return y; };
  const mse = (coef, xs, ys) => xs.reduce((acc, x, i) => acc + (predict(coef, x) - ys[i]) ** 2, 0) / xs.length;

  const degrees = Array.from({ length: 15 }, (_, i) => i + 1);
  const fits = degrees.map((d) => fitPoly(xTr, yTr, d));
  const trainErr = fits.map((c) => mse(c, xTr, yTr));
  const testErr = fits.map((c) => mse(c, xTe, yTe));
  const grid = Array.from({ length: 300 }, (_, i) => xMin + ((xMax - xMin) * i) / 299);

  // ---------- DOM ----------
  el.innerHTML = '<div id="ch04-poly-fit"></div><div id="ch04-poly-err"></div>';
  const slider = document.getElementById("ch04-poly-degree");
  const label = document.getElementById("ch04-poly-degree-val");
  const readout = document.getElementById("ch04-poly-readout");

  function colors() {
    const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
    return { fg: dark ? "#e0e0e0" : "#263238", grid: dark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.08)",
      pts: dark ? "#eceff1" : "#212121", test: dark ? "rgba(176,190,197,0.45)" : "rgba(96,125,139,0.4)" };
  }
  function baseLayout(title, xt, yt) {
    const c = colors();
    return { title: { text: title, font: { size: 14 } }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: c.fg }, margin: { l: 55, r: 15, t: 40, b: 45 }, height: 300,
      xaxis: { title: xt, gridcolor: c.grid, zeroline: false }, yaxis: { title: yt, gridcolor: c.grid, zeroline: false },
      legend: { orientation: "h", y: -0.25 } };
  }

  function draw() {
    const d = Number(slider.value), i = d - 1, c = colors();
    const over = d >= 10;
    label.textContent = d;
    const fitLayout = baseLayout(`${d} 次方多項式`, "劑量（任意單位）", "反應");
    fitLayout.yaxis.range = [-20, 120];
    Plotly.react("ch04-poly-fit", [
      { x: xTe, y: yTe, mode: "markers", name: "測試集", marker: { size: 5, color: c.test } },
      { x: xTr, y: yTr, mode: "markers", name: "訓練集", marker: { size: 7, color: c.pts } },
      { x: grid, y: grid.map((x) => predict(fits[i], x)), mode: "lines", name: "模型",
        line: { width: 3, color: over ? "#F4511E" : "#00897B" } },
    ], fitLayout, { displayModeBar: false, responsive: true });

    const errLayout = baseLayout("訓練誤差 vs 測試誤差", "多項式次方", "MSE（對數尺度）");
    errLayout.yaxis.type = "log";
    errLayout.xaxis.dtick = 1;
    Plotly.react("ch04-poly-err", [
      { x: degrees, y: trainErr, mode: "lines+markers", name: "訓練 MSE", line: { color: "#00897B" } },
      { x: degrees, y: testErr, mode: "lines+markers", name: "測試 MSE", line: { color: "#F4511E" } },
      { x: [d], y: [testErr[i]], mode: "markers", name: "目前選擇", marker: { size: 14, symbol: "circle-open", color: c.fg, line: { width: 2 } } },
    ], errLayout, { displayModeBar: false, responsive: true });

    readout.textContent = `訓練 MSE ${trainErr[i].toFixed(1)}，測試 MSE ${testErr[i].toFixed(1)}` +
      (over ? "：訓練誤差很小，測試誤差卻變大，這就是過擬合。" : "");
  }

  slider.addEventListener("input", draw);
  draw();

  // Re-draw when the reader toggles light / dark mode.
  if (window.__ch04PolyObserver) window.__ch04PolyObserver.disconnect();
  window.__ch04PolyObserver = new MutationObserver(() => { if (document.getElementById("ch04-poly-fit")) draw(); });
  window.__ch04PolyObserver.observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
