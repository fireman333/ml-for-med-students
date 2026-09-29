// Chapter 06 interactive demo: SVM decision boundary vs. kernel / C / gamma.
// Data (precomputed grids) comes from ch06-svm-data.js (scripts/demo_ch06.py).
(function () {

  function isDark() {
    return document.body.getAttribute("data-md-color-scheme") === "slate";
  }

  function fmt(v) {
    return (v * 100).toFixed(1) + "%";
  }

  function render(state) {
    const D = window.CH06_SVM_DATA;
    const el = document.getElementById("ch06-demo");
    if (!el || !D || typeof Plotly === "undefined") return;
    const dark = isDark();
    const fg = dark ? "#e0e0e0" : "#263238";
    const key = state.kernel === "linear" ? "linear_" + state.ci : "rbf_" + state.ci + "_" + state.gi;
    const m = D.models[key];
    const n = D.x.length;
    const z = [];
    for (let r = 0; r < n; r++) {
      z.push(m.z.slice(r * n, (r + 1) * n).map((v) => v / D.scale));
    }
    const pts = state.showTest ? D.test : D.train;
    const svSet = new Set(m.sv);
    const line = (lv, color, dash, width) => ({
      type: "contour", x: D.x, y: D.y, z: z, showscale: false, hoverinfo: "skip",
      contours: { coloring: "lines", start: lv, end: lv, size: 1 },
      colorscale: [[0, color], [1, color]], line: { width: width, dash: dash },
    });
    const traces = [
      {
        type: "heatmap", x: D.x, y: D.y, z: z, zmin: -2, zmax: 2, showscale: false, hoverinfo: "skip",
        zsmooth: "best",
        colorscale: [[0, "rgba(0,137,123,0.45)"], [0.5, "rgba(128,128,128,0.02)"], [1, "rgba(244,81,30,0.45)"]],
      },
      line(-1, "#90A4AE", "dash", 1.5),
      line(1, "#90A4AE", "dash", 1.5),
      line(0, fg, "solid", 2.5),
    ];
    [0, 1].forEach((cls) => {
      const xs = [], ys = [];
      pts.y.forEach((lab, i) => { if (lab === cls) { xs.push(pts.x1[i]); ys.push(pts.x2[i]); } });
      traces.push({
        type: "scatter", mode: "markers", x: xs, y: ys, name: cls === 0 ? "類別 A" : "類別 B",
        marker: { color: cls === 0 ? "#00897B" : "#F4511E", size: 8, symbol: cls === 0 ? "circle" : "square",
          line: { color: dark ? "#111" : "#fff", width: 1 } },
        hoverinfo: "skip",
      });
    });
    if (state.showSV && !state.showTest) {
      const xs = [], ys = [];
      D.train.y.forEach((_, i) => { if (svSet.has(i)) { xs.push(D.train.x1[i]); ys.push(D.train.x2[i]); } });
      traces.push({
        type: "scatter", mode: "markers", x: xs, y: ys, name: "支援向量", hoverinfo: "skip",
        marker: { size: 15, color: "rgba(0,0,0,0)", line: { color: fg, width: 1.5 } },
      });
    }
    const layout = {
      margin: { l: 50, r: 10, t: 10, b: 40 }, height: 420,
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: fg },
      xaxis: { range: [D.x[0], D.x[n - 1]], zeroline: false, showgrid: false, title: "特徵 1" },
      yaxis: { range: [D.y[0], D.y[n - 1]], zeroline: false, showgrid: false, title: "特徵 2", scaleanchor: "x" },
      legend: { orientation: "h", y: -0.12 },
    };
    Plotly.react(el, traces, layout, { displayModeBar: false, responsive: true });
    const info = document.getElementById("ch06-info");
    if (info) {
      info.textContent = "訓練準確率 " + fmt(m.train) + "　測試準確率 " + fmt(m.test) +
        "　支援向量 " + m.sv.length + " / " + D.train.y.length + " 個";
    }
  }

  function init() {
    const el = document.getElementById("ch06-demo");
    if (!el) return;
    const D = window.CH06_SVM_DATA;
    if (!D) { el.textContent = "互動資料載入失敗，請重新整理頁面。"; return; }
    const $ = (id) => document.getElementById(id);
    const state = { kernel: "rbf", ci: 1, gi: 2, showSV: true, showTest: false };

    function sync() {
      state.kernel = $("ch06-kernel").value;
      state.ci = +$("ch06-c").value;
      state.gi = +$("ch06-gamma").value;
      state.showSV = $("ch06-sv").checked;
      state.showTest = $("ch06-test").checked;
      $("ch06-c-val").textContent = D.C[state.ci];
      $("ch06-gamma-val").textContent = state.kernel === "linear" ? "（線性核不使用）" : D.gamma[state.gi];
      $("ch06-gamma").disabled = state.kernel === "linear";
      render(state);
    }

    ["ch06-kernel", "ch06-c", "ch06-gamma", "ch06-sv", "ch06-test"].forEach((id) => {
      const c = $(id);
      if (c) c.addEventListener("input", sync);
    });
    sync();

    // Re-render when the user toggles light/dark mode.
    if (window.__ch06Obs) window.__ch06Obs.disconnect();
    const obs = new MutationObserver(() => render(state));
    window.__ch06Obs = obs;
    obs.observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  }

  // navigation.instant is off, so each page loads fully: start on DOMContentLoaded (or now if already parsed).
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
