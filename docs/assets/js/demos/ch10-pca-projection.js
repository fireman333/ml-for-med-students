// Chapter 10 demo: project 2D data onto an axis; find the direction of maximum variance (PC1).
function ch10Init() {
  const el = document.getElementById("ch10-demo");
  if (!el || !window.Plotly) return;
  const slider = document.getElementById("ch10-angle");
  const angleLabel = document.getElementById("ch10-angle-val");
  const readout = document.getElementById("ch10-readout");
  const btn = document.getElementById("ch10-snap");

  // Seeded RNG (mulberry32) + Box-Muller -> reproducible correlated data
  let seed = 42;
  const rand = () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const randn = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const n = 80, r = 0.8;
  const xs = [], ys = [];
  for (let i = 0; i < n; i++) {
    const a = randn(), b = randn();
    xs.push(a);
    ys.push(r * a + Math.sqrt(1 - r * r) * b);
  }
  const mean = (v) => v.reduce((s, x) => s + x, 0) / v.length;
  const mx = mean(xs), my = mean(ys);
  const cx = xs.map((x) => x - mx), cy = ys.map((y) => y - my);
  const sxx = mean(cx.map((x) => x * x)), syy = mean(cy.map((y) => y * y));
  const sxy = mean(cx.map((x, i) => x * cy[i]));
  const total = sxx + syy;
  // PC1 angle of the 2x2 covariance matrix
  const pc1Deg = ((0.5 * Math.atan2(2 * sxy, sxx - syy)) * 180) / Math.PI;
  const pc1 = Math.round((pc1Deg + 180) % 180);

  const colors = () => {
    const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
    return {
      fg: dark ? "#e0e0e0" : "#333333",
      grid: dark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.08)",
      point: dark ? "#4DB6AC" : "#00897B",
      proj: "#F4511E",
      seg: dark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.2)",
    };
  };

  function render() {
    const deg = Number(slider.value);
    const th = (deg * Math.PI) / 180;
    const ux = Math.cos(th), uy = Math.sin(th);
    const scores = cx.map((x, i) => x * ux + cy[i] * uy);
    const px = scores.map((s) => s * ux), py = scores.map((s) => s * uy);
    const varProj = mean(scores.map((s) => s * s));
    const c = colors();
    const segX = [], segY = [];
    cx.forEach((x, i) => { segX.push(x, px[i], null); segY.push(cy[i], py[i], null); });
    const L = 4;
    const traces = [
      { x: segX, y: segY, mode: "lines", line: { color: c.seg, width: 1 }, hoverinfo: "skip", showlegend: false },
      { x: [-L * ux, L * ux], y: [-L * uy, L * uy], mode: "lines", name: "投影軸",
        line: { color: c.proj, width: 2, dash: "dash" } },
      { x: cx, y: cy, mode: "markers", name: "原始資料點", marker: { color: c.point, size: 7, opacity: 0.75 } },
      { x: px, y: py, mode: "markers", name: "投影後的位置", marker: { color: c.proj, size: 6 } },
    ];
    const layout = {
      margin: { l: 50, r: 10, t: 10, b: 45 },
      height: 420,
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: c.fg, size: 12 },
      xaxis: { title: "身高（標準化）", range: [-3.5, 3.5], zeroline: false, gridcolor: c.grid },
      yaxis: { title: "體重（標準化）", range: [-3.5, 3.5], zeroline: false, gridcolor: c.grid,
               scaleanchor: "x", scaleratio: 1 },
      legend: { orientation: "h", y: -0.18 },
      showlegend: true,
    };
    Plotly.react(el, traces, layout, { displayModeBar: false, responsive: true });
    angleLabel.textContent = deg + "°";
    const pct = ((varProj / total) * 100).toFixed(1);
    const note = Math.abs(deg - pc1) <= 1 || Math.abs(deg - pc1) >= 179
      ? "（這就是第一主成分的方向）" : "";
    readout.textContent = `投影後的變異數 = ${varProj.toFixed(2)}，保留了總變異的 ${pct}%${note}`;
  }

  slider.addEventListener("input", render);
  btn.addEventListener("click", () => { slider.value = pc1; render(); });
  new MutationObserver(render).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  render();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", ch10Init);
else ch10Init();
