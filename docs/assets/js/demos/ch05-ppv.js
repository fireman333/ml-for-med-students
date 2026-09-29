// Chapter 05 demo: prevalence / sensitivity / specificity -> PPV, NPV, LR calculator.
// Pure function `ch05Calc` is exported for node-based testing.
(function () {
  "use strict";

  function ch05Calc(prev, sens, spec, n) {
    n = n || 10000;
    var diseased = prev * n;
    var healthy = n - diseased;
    var tp = sens * diseased;
    var fn = diseased - tp;
    var tn = spec * healthy;
    var fp = healthy - tn;
    return {
      tp: tp, fn: fn, fp: fp, tn: tn,
      ppv: tp + fp > 0 ? tp / (tp + fp) : NaN,
      npv: tn + fn > 0 ? tn / (tn + fn) : NaN,
      lrPos: spec < 1 ? sens / (1 - spec) : Infinity,
      lrNeg: spec > 0 ? (1 - sens) / spec : Infinity
    };
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { ch05Calc: ch05Calc };
  }
  if (typeof document === "undefined") return;

  function pct(x, d) { return (x * 100).toFixed(d === undefined ? 1 : d) + "%"; }
  function cnt(x) { return Math.round(x).toLocaleString("en-US"); }

  function init() {
    var el = document.getElementById("ch05-demo");
    if (!el) return;

    var sPrev = document.getElementById("ch05-prev");
    var sSens = document.getElementById("ch05-sens");
    var sSpec = document.getElementById("ch05-spec");
    var out = document.getElementById("ch05-out");
    var plotDiv = document.getElementById("ch05-plot");
    if (!sPrev || !sSens || !sSpec || !out) return;

    // Resolve the theme text colour to rgb(a); Plotly cannot parse hsla(...deg...).
    function fg() {
      var probe = document.createElement("span");
      probe.style.color = "var(--md-default-fg-color)";
      el.appendChild(probe);
      var v = getComputedStyle(probe).color;
      el.removeChild(probe);
      return v || "#333";
    }

    function update() {
      var prev = Math.pow(10, parseFloat(sPrev.value));      // log10 scale slider
      var sens = parseFloat(sSens.value) / 100;
      var spec = parseFloat(sSpec.value) / 100;
      document.getElementById("ch05-prev-v").textContent = pct(prev, prev < 0.01 ? 2 : 1);
      document.getElementById("ch05-sens-v").textContent = pct(sens, 0);
      document.getElementById("ch05-spec-v").textContent = pct(spec, 0);

      var r = ch05Calc(prev, sens, spec, 10000);
      out.innerHTML =
        '<table style="margin:.4rem auto;font-size:.75rem">' +
        "<tr><th>每 10,000 人</th><th>檢驗陽性</th><th>檢驗陰性</th></tr>" +
        "<tr><td>有病（" + cnt(r.tp + r.fn) + "）</td><td>真陽性 " + cnt(r.tp) +
        "</td><td>偽陰性 " + cnt(r.fn) + "</td></tr>" +
        "<tr><td>沒病（" + cnt(r.fp + r.tn) + "）</td><td>偽陽性 " + cnt(r.fp) +
        "</td><td>真陰性 " + cnt(r.tn) + "</td></tr></table>" +
        '<p style="text-align:center;font-size:.85rem;margin:.3rem 0">' +
        "<strong>PPV = " + pct(r.ppv) + "</strong>　NPV = " + pct(r.npv, 2) +
        "　LR+ = " + (isFinite(r.lrPos) ? r.lrPos.toFixed(1) : "∞") +
        "　LR− = " + (isFinite(r.lrNeg) ? r.lrNeg.toFixed(2) : "∞") + "</p>";

      if (plotDiv && typeof Plotly !== "undefined") {
        var xs = [], ys = [];
        for (var i = 0; i <= 200; i++) {
          var lp = -3 + i * (Math.log10(0.5) + 3) / 200;
          var p = Math.pow(10, lp);
          xs.push(p * 100);
          ys.push(ch05Calc(p, sens, spec).ppv * 100);
        }
        var c = fg();
        Plotly.react(plotDiv, [
          { x: xs, y: ys, mode: "lines", line: { color: "#00897B", width: 3 }, name: "PPV",
            hovertemplate: "盛行率 %{x:.2f}%<br>PPV %{y:.1f}%<extra></extra>" },
          { x: [prev * 100], y: [r.ppv * 100], mode: "markers",
            marker: { color: "#F4511E", size: 12 }, name: "目前設定", hoverinfo: "skip" }
        ], {
          height: 300, margin: { l: 55, r: 15, t: 10, b: 45 }, showlegend: false,
          paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: c, size: 12 },
          xaxis: { type: "log", title: "盛行率（%，對數刻度）", gridcolor: "rgba(128,128,128,0.25)",
                   tickvals: [0.1, 0.3, 1, 3, 10, 30, 50], ticktext: ["0.1", "0.3", "1", "3", "10", "30", "50"] },
          yaxis: { range: [0, 100], title: "PPV（%）", gridcolor: "rgba(128,128,128,0.25)" }
        }, { displayModeBar: false, responsive: true });
      }
    }

    [sPrev, sSens, sSpec].forEach(function (s) { s.addEventListener("input", update); });
    update();

    // Re-render when the light/dark palette toggles.
    if (window.__ch05Observer) window.__ch05Observer.disconnect();
    window.__ch05Observer = new MutationObserver(update);
    window.__ch05Observer.observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
