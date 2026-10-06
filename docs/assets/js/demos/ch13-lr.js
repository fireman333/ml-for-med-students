// Chapter 13: learning-rate demo. Gradient descent on a 1-D bowl, loss(w) = w^2.
// Slider sets the learning rate; left plot shows the path on the bowl, right plot the loss per step.
function initCh13Lr() {
  const el = document.getElementById("ch13-demo");
  const curve = document.getElementById("ch13-curve");
  if (!el || !curve || typeof Plotly === "undefined") return;
  const lrIn = document.getElementById("ch13-lr");
  const lrOut = document.getElementById("ch13-lr-val");
  const info = document.getElementById("ch13-info");
  const STEPS = 20;
  const W0 = 4;

  // slider is on a log scale: value v -> lr = 10^v
  const lrOf = (v) => Math.pow(10, v);

  function colors() {
    const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
    return { fg: dark ? "#e0e0e0" : "#333333", grid: dark ? "#555555" : "#dddddd" };
  }

  function layout(c, xTitle, yTitle, extra) {
    return Object.assign({
      height: 300,
      margin: { l: 55, r: 15, t: 15, b: 45 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: c.fg },
      showlegend: false,
      xaxis: { title: xTitle, gridcolor: c.grid, zeroline: false },
      yaxis: { title: yTitle, gridcolor: c.grid, zeroline: false },
    }, extra || {});
  }

  function draw() {
    const lr = lrOf(parseFloat(lrIn.value));
    lrOut.textContent = lr < 0.1 ? lr.toFixed(3) : lr.toFixed(2);
    // gradient of w^2 is 2w  ->  w_new = w - lr * 2w = (1 - 2 lr) w
    const ws = [W0];
    for (let i = 0; i < STEPS; i++) ws.push(ws[ws.length - 1] * (1 - 2 * lr));
    const losses = ws.map((w) => w * w);
    const c = colors();

    const bx = [];
    for (let x = -6; x <= 6.0001; x += 0.1) bx.push(+x.toFixed(2));
    const shown = ws.map((w) => Math.max(-6, Math.min(6, w)));
    Plotly.react(el, [
      { x: bx, y: bx.map((x) => x * x), mode: "lines", line: { color: "#607D8B", width: 2 } },
      { x: shown, y: shown.map((w) => w * w), mode: "lines+markers",
        line: { color: "#F4511E", width: 1.5 }, marker: { color: "#F4511E", size: 7 } },
    ], layout(c, "權重 w", "損失 = w²", { yaxis: { title: "損失 = w²", range: [0, 37], gridcolor: c.grid } }),
    { displayModeBar: false, responsive: true });

    Plotly.react(curve, [
      { x: losses.map((_, i) => i), y: losses.map((v) => Math.max(v, 1e-6)), mode: "lines+markers",
        line: { color: "#00897B", width: 2.5 }, marker: { size: 5 } },
    ], layout(c, "更新次數", "損失（對數刻度）", { yaxis: { title: "損失（對數刻度）", type: "log", gridcolor: c.grid } }),
    { displayModeBar: false, responsive: true });

    let msg;
    if (lr < 0.1) msg = "學習率太小：每一步都往對的方向走，但步伐很小，20 步後離谷底還很遠。";
    else if (lr < 0.5) msg = "學習率適中：損失穩定下降，十幾步內就接近谷底。";
    else if (lr === 0.5 || Math.abs(lr - 0.5) < 0.02) msg = "剛好一步跳到谷底——真實的損失地形不會這麼單純，不要期待這種好運。";
    else if (Math.abs(lr - 1) < 0.03) msg = "學習率約等於 1：每一步剛好跳到谷底對面同樣高的位置，永遠在兩點間來回、損失不再下降。";
    else if (lr < 1) msg = "學習率偏大：每一步都跨過谷底、在兩側來回彈跳，仍會收斂但走得歪七扭八。";
    else msg = "學習率太大：每次都跳得比上次更遠，損失越來越高，訓練「爆掉」（實務上會看到 loss 變成 nan）。";
    info.textContent = msg;
  }

  lrIn.addEventListener("input", draw);
  draw();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initCh13Lr);
else initCh13Lr();
