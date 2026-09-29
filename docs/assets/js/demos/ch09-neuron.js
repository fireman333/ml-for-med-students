// Chapter 09: single-neuron (sigmoid) demo. Sliders for weight w and bias b.
function initCh09Neuron() {
  const el = document.getElementById("ch09-demo");
  if (!el || typeof Plotly === "undefined") return;
  const wIn = document.getElementById("ch09-w");
  const bIn = document.getElementById("ch09-b");
  const wOut = document.getElementById("ch09-w-val");
  const bOut = document.getElementById("ch09-b-val");
  const info = document.getElementById("ch09-info");

  const xs = [];
  for (let x = -4; x <= 4.0001; x += 0.05) xs.push(+x.toFixed(2));
  const sig = (z) => 1 / (1 + Math.exp(-z));

  function colors() {
    const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
    return { fg: dark ? "#e0e0e0" : "#333333", grid: dark ? "#555555" : "#dddddd" };
  }

  function draw() {
    const w = parseFloat(wIn.value);
    const b = parseFloat(bIn.value);
    wOut.textContent = w.toFixed(1);
    bOut.textContent = b.toFixed(1);
    const ys = xs.map((x) => sig(w * x + b));
    const c = colors();
    const traces = [
      { x: xs, y: ys, mode: "lines", line: { color: "#00897B", width: 3 }, name: "輸出 = sigmoid(w·x + b)" },
      { x: [-4, 4], y: [0.5, 0.5], mode: "lines", line: { color: "#607D8B", dash: "dash", width: 1 }, name: "閾值 0.5" },
    ];
    let msg;
    if (Math.abs(w) < 1e-9) {
      msg = `w = 0：不論輸入多少，輸出都固定在 ${sig(b).toFixed(2)}，這個神經元對 x 完全「不敏感」。`;
    } else {
      const x0 = -b / w;
      if (x0 >= -4 && x0 <= 4) {
        traces.push({ x: [x0], y: [0.5], mode: "markers", marker: { color: "#F4511E", size: 10 }, name: "判斷分界點" });
      }
      msg = `分界點在 x = ${x0.toFixed(2)}：x ${w > 0 ? "大於" : "小於"}這個值時輸出超過 0.5。` +
        `|w| 越大，曲線越陡，判斷越「果斷」；b 則把分界點左右平移。`;
    }
    info.textContent = msg;
    Plotly.react(el, traces, {
      height: 320,
      margin: { l: 50, r: 20, t: 20, b: 45 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: c.fg },
      xaxis: { title: "輸入 x（例如標準化後的腫瘤半徑）", range: [-4, 4], gridcolor: c.grid, zeroline: false },
      yaxis: { title: "神經元輸出", range: [-0.05, 1.05], gridcolor: c.grid, zeroline: false },
      legend: { orientation: "h", y: -0.3 },
      showlegend: false,
    }, { displayModeBar: false, responsive: true });
  }

  wIn.addEventListener("input", draw);
  bIn.addEventListener("input", draw);
  draw();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initCh09Neuron);
else initCh09Neuron();
