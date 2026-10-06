// Chapter 15: softmax temperature demo for next-token prediction.
// A toy language model has produced raw scores (logits) for the next word after
// "The patient presents with chest ..."; the slider changes the sampling temperature.
function initCh15Softmax() {
  const el = document.getElementById("ch15-softmax-demo");
  if (!el || typeof Plotly === "undefined") return;
  const tIn = document.getElementById("ch15-temp");
  const tOut = document.getElementById("ch15-temp-val");
  const info = document.getElementById("ch15-softmax-info");

  // Hand-picked toy logits (not from a real model).
  const words = ["pain", "tightness", "discomfort", "wall", "x-ray", "banana"];
  const logits = [4.0, 2.6, 2.3, 1.2, 0.6, -2.0];

  function softmax(z, T) {
    const s = z.map((v) => v / T);
    const m = Math.max(...s);
    const e = s.map((v) => Math.exp(v - m));
    const tot = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / tot);
  }

  function colors() {
    const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
    return { fg: dark ? "#e0e0e0" : "#333333", grid: dark ? "#555555" : "#dddddd" };
  }

  function draw() {
    const T = parseFloat(tIn.value);
    tOut.textContent = T.toFixed(1);
    const p = softmax(logits, T);
    const c = colors();
    const barColors = words.map((w, i) => (i === 0 ? "#00897B" : w === "banana" ? "#F4511E" : "#607D8B"));
    Plotly.react(el, [{
      type: "bar", x: words, y: p, marker: { color: barColors },
      text: p.map((v) => (v * 100).toFixed(1) + "%"), textposition: "outside",
      hovertemplate: "%{x}: %{y:.3f}<extra></extra>",
    }], {
      height: 320,
      margin: { l: 50, r: 20, t: 20, b: 45 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: c.fg },
      xaxis: { title: "下一個字的候選", gridcolor: c.grid },
      yaxis: { title: "機率", range: [0, 1.1], gridcolor: c.grid, zeroline: false },
      showlegend: false,
    }, { displayModeBar: false, responsive: true });

    const top = (p[0] * 100).toFixed(1);
    const odd = (p[5] * 100).toFixed(2);
    let msg;
    if (T < 0.6) msg = `溫度 ${T.toFixed(1)}：分布很尖，幾乎每次都選「pain」（${top}%），回答穩定但單調。`;
    else if (T <= 1.2) msg = `溫度 ${T.toFixed(1)}：接近模型原本的分布，「pain」${top}%，其他合理的字也有機會被選到。`;
    else msg = `溫度 ${T.toFixed(1)}：分布被壓平，連不合理的「banana」都有 ${odd}% 機會被抽到——溫度越高，回答越多樣，也越容易離題。`;
    info.textContent = msg + " 注意：這些機率只代表「模型覺得哪個字接下去最順」，不代表內容正確的機率。";
  }

  tIn.addEventListener("input", draw);
  draw();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initCh15Softmax);
else initCh15Softmax();
