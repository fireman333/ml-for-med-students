// Chapter 14 demo: a 1D convolution filter sliding along a (synthetic) ECG strip.
// The ECG is generated from Gaussian bumps (P, Q, R, S, T); it is NOT patient data.
(function () {
  "use strict";
  const FS = 100;            // samples per second of the toy signal
  const SECONDS = 5;

  function gauss(t, mu, sd, amp) { return amp * Math.exp(-0.5 * ((t - mu) / sd) ** 2); }

  // beat list: [R-peak time (s), type]
  const BEATS = [[0.45, "N"], [1.25, "N"], [1.8, "V"], [2.85, "N"], [3.65, "N"], [4.45, "N"]];

  function makeEcg() {
    const n = FS * SECONDS, x = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      const t = i / FS;
      let v = 0.04 * Math.sin(2 * Math.PI * 0.3 * t);           // slow baseline wander
      for (const [r, type] of BEATS) {
        if (type === "N") {
          v += gauss(t, r - 0.2, 0.025, 0.15) + gauss(t, r - 0.03, 0.01, -0.12) + gauss(t, r, 0.012, 1.0)
             + gauss(t, r + 0.03, 0.012, -0.25) + gauss(t, r + 0.25, 0.05, 0.3);
        } else {                                                  // PVC: no P wave, wide QRS, opposite T
          v += gauss(t, r, 0.04, 1.25) + gauss(t, r + 0.09, 0.03, -0.35) + gauss(t, r + 0.28, 0.07, -0.45);
        }
      }
      x[i] = v;
    }
    return x;
  }

  function normalise(k) {          // zero mean, unit length -> outputs are comparable between filters
    const m = k.reduce((a, b) => a + b, 0) / k.length;
    const z = k.map((v) => v - m);
    const norm = Math.sqrt(z.reduce((a, b) => a + b * b, 0)) || 1;
    return z.map((v) => v / norm);
  }
  function template(sd, len) {     // a bump of width sd (s), sampled over len points
    const h = (len - 1) / 2;
    return normalise(Array.from({ length: len }, (_, i) => gauss((i - h) / FS, 0, sd, 1)));
  }

  const FILTERS = {
    narrow: { label: "窄 QRS 偵測器（長 9 點）", w: template(0.012, 9) },
    wide: { label: "寬 QRS 偵測器（長 17 點）", w: template(0.04, 17) },
    slope: { label: "上升斜率偵測器（長 5 點）", w: normalise([-2, -1, 0, 1, 2]) },
    smooth: { label: "移動平均（長 7 點，不歸零）", w: Array(7).fill(1 / 7) },
  };

  // Read the page's own theme colours (CSS variables), so the canvas follows light/dark schemes.
  function colors(el) {
    const cs = getComputedStyle(el);
    const fg = cs.color || "#263238";
    const grid = cs.getPropertyValue("--md-default-fg-color--lightest").trim() || "rgba(128,128,128,.35)";
    return { fg: fg, grid: grid, sig: "#00897B", acc: "#F4511E", out: "#3f7fd0", win: "rgba(244,81,30,.18)" };
  }
  const fmt = (v) => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3);

  function init() {
    const root = document.getElementById("ch14-demo");
    if (!root) return;
    const ecg = makeEcg();
    const sel = document.getElementById("ch14-filter");
    const pos = document.getElementById("ch14-pos");
    const relu = document.getElementById("ch14-relu");
    const play = document.getElementById("ch14-play");
    const info = document.getElementById("ch14-info");
    for (const [key, f] of Object.entries(FILTERS)) {
      const o = document.createElement("option"); o.value = key; o.textContent = f.label; sel.appendChild(o);
    }

    const canvas = document.createElement("canvas");
    canvas.style.width = "100%"; canvas.style.display = "block";
    root.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    let timer = null;

    function conv(w) {
      const k = w.length, out = [];
      for (let i = 0; i + k <= ecg.length; i++) {
        let s = 0;
        for (let j = 0; j < k; j++) s += w[j] * ecg[i + j];
        out.push(relu.checked ? Math.max(0, s) : s);
      }
      return out;
    }

    function draw() {
      const c = colors(root);
      const W = root.clientWidth || 600, narrow = W < 520, H = narrow ? 370 : 330, dpr = window.devicePixelRatio || 1;
      canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.height = H + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.font = "12px sans-serif";
      const f = FILTERS[sel.value], w = f.w, k = w.length, out = conv(w);
      pos.max = String(out.length - 1);
      const p = Math.min(+pos.value, out.length - 1);
      const L = 40, R = W - 10, xs = (i) => L + (R - L) * i / (ecg.length - 1);

      // panel 1: ECG with the filter window
      const y1 = (v) => 20 + (110 - 20) * (1 - (v + 0.6) / 2.0);
      ctx.fillStyle = c.win; ctx.fillRect(xs(p), 14, xs(p + k - 1) - xs(p) + 1, 100);
      ctx.strokeStyle = c.sig; ctx.lineWidth = 1.5; ctx.beginPath();
      ecg.forEach((v, i) => (i ? ctx.lineTo(xs(i), y1(v)) : ctx.moveTo(xs(i), y1(v)))); ctx.stroke();
      ctx.fillStyle = c.fg;
      ctx.fillText("合成心電圖（示意，非病人資料）", L, 12);
      for (const [r, type] of BEATS) {
        ctx.fillStyle = type === "V" ? c.acc : c.fg;
        ctx.fillText(type === "V" ? "PVC" : "N", xs(r * FS) - 6, 128);
      }

      // panel 2: filter weights (bar chart)
      const bx = L, bw = Math.min(160, (R - L) * 0.35), by = 150, bh = 50;
      const wmax = Math.max(...w.map(Math.abs));
      ctx.fillStyle = c.fg; ctx.fillText("濾鏡權重（要學的數字）", bx, by - 6);
      ctx.strokeStyle = c.grid; ctx.beginPath(); ctx.moveTo(bx, by + bh / 2); ctx.lineTo(bx + bw, by + bh / 2); ctx.stroke();
      w.forEach((v, j) => {
        ctx.fillStyle = v >= 0 ? c.sig : c.acc;
        const hh = (v / wmax) * bh / 2;
        ctx.fillRect(bx + j * bw / k + 1, by + bh / 2 - Math.max(hh, 0), bw / k - 2, Math.abs(hh));
      });
      let dot = 0;
      for (let j = 0; j < k; j++) dot += w[j] * ecg[p + j];
      const shown = relu.checked ? Math.max(0, dot) : dot;
      ctx.fillStyle = c.fg;
      const tx = narrow ? bx : bx + bw + 20, ty = narrow ? by + bh + 16 : by + 12;
      ctx.fillText(`目前位置：第 ${p}–${p + k - 1} 點（${(p / FS).toFixed(2)}–${((p + k - 1) / FS).toFixed(2)} 秒）`, tx, ty);
      ctx.fillText(`對應相乘再加總 = ${fmt(dot)}${relu.checked ? `，ReLU 後 = ${fmt(shown)}` : ""}`, tx, ty + 20);

      // panel 3: output (feature map), drawn up to the current position
      const lo = Math.min(0, ...out), hi = Math.max(...out), span = hi - lo || 1;
      const top3 = narrow ? 268 : 228;
      const y3 = (v) => top3 + 90 * (1 - (v - lo) / span);
      ctx.strokeStyle = c.grid; ctx.beginPath(); ctx.moveTo(L, y3(0)); ctx.lineTo(R, y3(0)); ctx.stroke();
      ctx.strokeStyle = c.out; ctx.lineWidth = 1.8; ctx.beginPath();
      for (let i = 0; i <= p; i++) {
        const x = xs(i + (k - 1) / 2);
        i ? ctx.lineTo(x, y3(out[i])) : ctx.moveTo(x, y3(out[i]));
      }
      ctx.stroke();
      ctx.fillStyle = c.acc; ctx.beginPath(); ctx.arc(xs(p + (k - 1) / 2), y3(out[p]), 4, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = c.fg; ctx.fillText("輸出（特徵圖）：每個位置一個數字", L, top3 - 6);

      if (p === out.length - 1) {
        const best = out.indexOf(hi);
        const t = (best + (k - 1) / 2) / FS;
        const near = BEATS.reduce((a, b) => (Math.abs(b[0] - t) < Math.abs(a[0] - t) ? b : a));
        info.textContent = `全部算完：最大輸出 ${fmt(hi)} 出現在 ${t.toFixed(2)} 秒，最接近的心跳是 ${near[1] === "V" ? "PVC" : "正常心跳"}。`;
      } else {
        info.textContent = "拖動滑桿或按「播放」，看濾鏡沿時間軸一點一點滑過去。";
      }
    }

    function stop() { clearInterval(timer); timer = null; play.textContent = "播放"; }
    play.addEventListener("click", () => {
      if (timer) { stop(); return; }
      if (+pos.value >= +pos.max) pos.value = "0";
      play.textContent = "暫停";
      timer = setInterval(() => {
        pos.value = String(Math.min(+pos.value + 3, +pos.max));
        draw();
        if (+pos.value >= +pos.max) stop();
      }, 30);
    });
    sel.addEventListener("change", () => { stop(); draw(); });
    pos.addEventListener("input", draw);
    relu.addEventListener("change", draw);
    window.addEventListener("resize", draw);
    new MutationObserver(draw).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme", "data-md-color-primary"] });
    sel.value = "wide";
    pos.value = "0";
    draw();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
