// Chapter 04 demo: logistic regression threshold slider.
// Simulated 1-D "biomarker" for benign vs malignant; the logistic curve is
// fitted in the browser by gradient descent, then the threshold slider
// updates the confusion matrix, sensitivity and specificity.
(function () {
  function init() {
  const el = document.getElementById("ch04-sigmoid-demo");
  if (!el || typeof Plotly === "undefined") return;

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const rand = mulberry32(7);
  const randn = () => {
    const u = Math.max(rand(), 1e-12), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  // 70 benign (label 0), 40 malignant (label 1); biomarker in arbitrary units.
  const xs = [], ys = [];
  for (let i = 0; i < 70; i++) { xs.push(3 + 1.1 * randn()); ys.push(0); }
  for (let i = 0; i < 40; i++) { xs.push(5.5 + 1.1 * randn()); ys.push(1); }

  // Fit p = sigmoid(b0 + b1 * x) with plain gradient descent on log loss.
  const sig = (z) => 1 / (1 + Math.exp(-z));
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  let b0 = 0, b1 = 0;
  for (let it = 0; it < 3000; it++) {
    let g0 = 0, g1 = 0;
    for (let i = 0; i < xs.length; i++) {
      const e = sig(b0 + b1 * (xs[i] - mean)) - ys[i];
      g0 += e; g1 += e * (xs[i] - mean);
    }
    b0 -= (0.5 * g0) / xs.length; b1 -= (0.5 * g1) / xs.length;
  }
  const prob = (x) => sig(b0 + b1 * (x - mean));
  const cutoffX = (t) => mean + (Math.log(t / (1 - t)) - b0) / b1; // x where p = t

  el.innerHTML = '<div id="ch04-sigmoid-plot"></div><div id="ch04-sigmoid-table"></div>';
  const slider = document.getElementById("ch04-sigmoid-th");
  const label = document.getElementById("ch04-sigmoid-th-val");
  const xMin = Math.min(...xs) - 0.5, xMax = Math.max(...xs) + 0.5;
  const grid = Array.from({ length: 200 }, (_, i) => xMin + ((xMax - xMin) * i) / 199);
  const jitter = xs.map(() => (rand() - 0.5) * 0.06);

  function colors() {
    const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
    return { fg: dark ? "#e0e0e0" : "#263238", grid: dark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.08)",
      border: dark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.2)" };
  }

  function draw() {
    const t = Number(slider.value), c = colors();
    label.textContent = t.toFixed(2);
    let tp = 0, fp = 0, tn = 0, fn = 0;
    xs.forEach((x, i) => {
      const pos = prob(x) >= t;
      if (pos && ys[i] === 1) tp++; else if (pos) fp++; else if (ys[i] === 1) fn++; else tn++;
    });
    const cx = cutoffX(t);
    const pts = (lab) => {
      const idx = xs.map((_, i) => i).filter((i) => ys[i] === lab);
      return { x: idx.map((i) => xs[i]), y: idx.map((i) => lab + jitter[i]), mode: "markers",
        name: lab === 1 ? "實際惡性" : "實際良性",
        marker: { size: 8, symbol: lab === 1 ? "x" : "circle", color: lab === 1 ? "#F4511E" : "#00897B", opacity: 0.8 } };
    };
    Plotly.react("ch04-sigmoid-plot", [
      pts(0), pts(1),
      { x: grid, y: grid.map(prob), mode: "lines", name: "邏輯迴歸預測機率", line: { width: 3, color: "#607D8B" } },
      { x: [xMin, xMax], y: [t, t], mode: "lines", name: "閾值", line: { dash: "dash", color: c.fg, width: 1.5 } },
    ], {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: c.fg }, height: 330,
      margin: { l: 55, r: 15, t: 20, b: 95 }, legend: { orientation: "h", x: 0, xanchor: "left", y: -0.32, yanchor: "top" },
      xaxis: { title: "模擬生物標記（任意單位）", gridcolor: c.grid, zeroline: false, range: [xMin, xMax] },
      yaxis: { title: "惡性機率", gridcolor: c.grid, range: [-0.1, 1.1] },
      shapes: [{ type: "rect", x0: cx, x1: xMax, y0: -0.1, y1: 1.1, fillcolor: "rgba(244,81,30,0.08)", line: { width: 0 } }],
    }, { displayModeBar: false, responsive: true });

    const sens = tp / (tp + fn), spec = tn / (tn + fp);
    const cell = `style="border:1px solid ${c.border};padding:.3rem .6rem;text-align:center"`;
    document.getElementById("ch04-sigmoid-table").innerHTML =
      `<table style="border-collapse:collapse;margin:.5rem auto;font-size:.75rem">` +
      `<tr><td ${cell}></td><td ${cell}><b>預測良性</b></td><td ${cell}><b>預測惡性</b></td></tr>` +
      `<tr><td ${cell}><b>實際良性</b></td><td ${cell}>TN = ${tn}</td><td ${cell}>FP = ${fp}（誤報）</td></tr>` +
      `<tr><td ${cell}><b>實際惡性</b></td><td ${cell}>FN = ${fn}（漏診）</td><td ${cell}>TP = ${tp}</td></tr>` +
      `</table><p style="text-align:center;margin:.2rem 0;font-size:.75rem">敏感度 ${(100 * sens).toFixed(1)}%　特異度 ${(100 * spec).toFixed(1)}%</p>`;
  }

  slider.addEventListener("input", draw);
  draw();

  if (window.__ch04SigObserver) window.__ch04SigObserver.disconnect();
  window.__ch04SigObserver = new MutationObserver(() => { if (document.getElementById("ch04-sigmoid-plot")) draw(); });
  window.__ch04SigObserver.observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
