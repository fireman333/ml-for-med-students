// Chapter 15: interactive self-attention viewer.
// Data (window.CH15_ATTENTION) comes from ch15-attention-data.js, exported by scripts/figs_ch15.py
// from the chapter's trained mini Transformer. Hover or tap a word to see where it "looks".
function initCh15Attention() {
  const el = document.getElementById("ch15-attn-demo");
  const data = window.CH15_ATTENTION;
  if (!el || !data) return;
  const info = document.getElementById("ch15-attn-info");
  const headSel = document.getElementById("ch15-attn-head");
  const n = data.tokens.length;

  function weightsFor(q) {
    const h = headSel.value;
    if (h === "avg") {
      return data.tokens.map((_, k) => data.heads.reduce((s, m) => s + m[q][k], 0) / data.heads.length);
    }
    return data.heads[parseInt(h, 10)][q];
  }

  el.innerHTML = "";
  el.style.lineHeight = "2.4";
  el.style.fontSize = "0.95rem";
  const spans = data.tokens.map((tok, k) => {
    const s = document.createElement("span");
    s.textContent = tok;
    s.tabIndex = 0;
    s.style.cssText = "display:inline-block;padding:0 .35em;margin:0 .12em;border-radius:4px;" +
      "cursor:pointer;border:1px solid transparent;transition:background-color .15s;";
    s.addEventListener("mouseenter", () => select(k));
    s.addEventListener("focus", () => select(k));
    s.addEventListener("click", () => select(k));
    el.appendChild(s);
    return s;
  });

  let current = 0;
  function select(q) {
    current = q;
    const w = weightsFor(q);
    const maxW = Math.max(...w);
    spans.forEach((s, k) => {
      const a = Math.min(1, w[k] / maxW) * 0.85;
      s.style.backgroundColor = `rgba(0, 137, 123, ${a.toFixed(3)})`;
      s.style.color = a > 0.5 ? "#ffffff" : "";
      s.style.borderColor = k === q ? "#F4511E" : "transparent";
      s.title = `${(w[k] * 100).toFixed(1)}%`;
    });
    const order = w.map((v, k) => [v, k]).sort((x, y) => y[0] - x[0]).slice(0, 3);
    const top = order.map(([v, k]) => `${data.tokens[k]}（${(v * 100).toFixed(1)}%）`).join("、");
    const total = w.reduce((s, v) => s + v, 0);
    info.textContent = `「${data.tokens[q]}」看得最多的三個字：${top}。` +
      `全部權重加總 = ${total.toFixed(2)}；若平均分配，每個字約 ${(100 / n).toFixed(1)}%。` +
      `（真實診斷：${data.label}；模型預測：${data.pred}）`;
  }

  headSel.addEventListener("change", () => select(current));
  select(0);
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initCh15Attention);
else initCh15Attention();
