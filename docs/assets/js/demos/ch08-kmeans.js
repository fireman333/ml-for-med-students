/* Chapter 08: step-by-step K-Means animation (vanilla JS + canvas, init on DOMContentLoaded).
 * Core algorithm lives in the pure object `KM` so it can be tested with node:
 *   node -e "const KM=require('./docs/assets/js/demos/ch08-kmeans.js'); ..."
 */
(function () {
  "use strict";

  // ---------- pure core (no DOM) ----------
  var KM = {};

  // Small seeded PRNG (mulberry32) so a given seed always gives the same data.
  KM.makeRng = function (seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  function gauss(rng) {
    var u = 1 - rng(), v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function clamp01(x) { return Math.min(0.98, Math.max(0.02, x)); }

  // Well-separated round blobs in the unit square.
  KM.genBlobs = function (rng, n, nBlobs) {
    var centers = [], pts = [], tries = 0;
    while (centers.length < nBlobs && tries < 500) {
      tries++;
      var c = [0.15 + 0.7 * rng(), 0.15 + 0.7 * rng()];
      var ok = centers.every(function (o) { return Math.hypot(o[0] - c[0], o[1] - c[1]) > 0.28; });
      if (ok) centers.push(c);
    }
    for (var i = 0; i < n; i++) {
      var b = centers[i % centers.length];
      pts.push([clamp01(b[0] + 0.06 * gauss(rng)), clamp01(b[1] + 0.06 * gauss(rng))]);
    }
    return pts;
  };

  // Uniform cloud: no real cluster structure at all.
  KM.genUniform = function (rng, n) {
    var pts = [];
    for (var i = 0; i < n; i++) pts.push([0.05 + 0.9 * rng(), 0.05 + 0.9 * rng()]);
    return pts;
  };

  // Two interleaving half moons (non-spherical clusters).
  KM.genMoons = function (rng, n) {
    var pts = [];
    for (var i = 0; i < n; i++) {
      var t = Math.PI * rng(), x, y;
      if (i % 2 === 0) { x = Math.cos(t); y = Math.sin(t); }
      else { x = 1 - Math.cos(t); y = 0.5 - Math.sin(t); }
      pts.push([clamp01(0.2 + 0.25 * (x + 1) + 0.02 * gauss(rng)),
                clamp01(0.3 + 0.3 * (y + 0.5) + 0.02 * gauss(rng))]);
    }
    return pts;
  };

  KM.dist2 = function (a, b) {
    var dx = a[0] - b[0], dy = a[1] - b[1];
    return dx * dx + dy * dy;
  };

  // Step 1: choose k distinct data points as the initial centers.
  KM.initCenters = function (points, k, rng) {
    var idx = points.map(function (_, i) { return i; });
    for (var i = idx.length - 1; i > 0; i--) {      // Fisher-Yates shuffle
      var j = Math.floor(rng() * (i + 1)), tmp = idx[i];
      idx[i] = idx[j]; idx[j] = tmp;
    }
    return idx.slice(0, Math.min(k, points.length)).map(function (i) { return points[i].slice(); });
  };

  // Step 2: every point joins its nearest center.
  KM.assign = function (points, centers) {
    return points.map(function (p) {
      var best = 0, bestD = Infinity;
      for (var j = 0; j < centers.length; j++) {
        var d = KM.dist2(p, centers[j]);
        if (d < bestD) { bestD = d; best = j; }
      }
      return best;
    });
  };

  // Step 3: every center moves to the mean of its members (empty cluster keeps its place).
  KM.update = function (points, labels, centers) {
    var sums = centers.map(function () { return [0, 0, 0]; });
    points.forEach(function (p, i) {
      var s = sums[labels[i]];
      s[0] += p[0]; s[1] += p[1]; s[2] += 1;
    });
    return sums.map(function (s, j) {
      return s[2] > 0 ? [s[0] / s[2], s[1] / s[2]] : centers[j].slice();
    });
  };

  // Within-cluster sum of squares (what sklearn calls inertia_).
  KM.inertia = function (points, labels, centers) {
    var total = 0;
    points.forEach(function (p, i) { total += KM.dist2(p, centers[labels[i]]); });
    return total;
  };

  KM.sameLabels = function (a, b) {
    if (!a || !b || a.length !== b.length) return false;
    for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  };

  // Step 4: repeat assign/update until the assignment stops changing.
  KM.run = function (points, k, rng, maxIter) {
    var centers = KM.initCenters(points, k, rng), labels = null, it = 0;
    maxIter = maxIter || 100;
    while (it < maxIter) {
      var newLabels = KM.assign(points, centers);
      it++;
      if (KM.sameLabels(labels, newLabels)) { labels = newLabels; break; }
      labels = newLabels;
      centers = KM.update(points, labels, centers);
    }
    return { labels: labels, centers: centers, iterations: it,
             inertia: KM.inertia(points, labels, centers) };
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = KM;
  }

  // ---------- browser UI ----------
  if (typeof window === "undefined" || typeof document === "undefined") return;

  var COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"];
  var W = 640, H = 400, N = 180;

  function cssVar(name, fallback) {
    var v = getComputedStyle(document.body).getPropertyValue(name).trim();
    return v || fallback;
  }

  function setup() {
    var root = document.getElementById("ch08-demo");
    if (!root || root.dataset.ready === "1") return;
    root.dataset.ready = "1";
    var box = root.parentElement;

    root.innerHTML =
      '<canvas width="' + W + '" height="' + H + '" aria-label="K-Means 逐步動畫" ' +
      'style="width:100%;max-width:' + W + 'px;cursor:crosshair;border-radius:4px"></canvas>' +
      '<p class="ch08-status" style="font-size:.75rem;margin:.4rem 0 0;min-height:2.4em"></p>';
    var canvas = root.querySelector("canvas");
    var statusEl = root.querySelector(".ch08-status");
    var ctx = canvas.getContext("2d");
    var dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    var controls = box.querySelector(".controls");
    controls.innerHTML =
      '<label>資料 <select data-k="data">' +
      '<option value="blobs">分明的團塊</option>' +
      '<option value="uniform">均勻散布（沒有真的群）</option>' +
      '<option value="moons">兩個半月</option></select></label>' +
      '<label>K = <span data-k="kval">3</span> <input data-k="k" type="range" min="1" max="6" value="3"></label>' +
      '<button class="md-button md-button--primary" data-k="step" type="button">下一步</button>' +
      '<button class="md-button" data-k="play" type="button">自動播放</button>' +
      '<button class="md-button" data-k="reinit" type="button">重新放中心</button>' +
      '<button class="md-button" data-k="newdata" type="button">換一批資料</button>';
    function q(key) { return controls.querySelector('[data-k="' + key + '"]'); }
    controls.querySelectorAll(".md-button").forEach(function (b) {
      b.style.padding = ".3em .9em"; b.style.fontSize = ".75rem";
    });

    var seed = 42, rng = KM.makeRng(seed);
    var state = { points: [], centers: [], labels: null, phase: "empty", iter: 0, converged: false };
    var timer = null;

    function makeData() {
      var kind = q("data").value, r = KM.makeRng(seed);
      if (kind === "uniform") state.points = KM.genUniform(r, N);
      else if (kind === "moons") state.points = KM.genMoons(r, N);
      else state.points = KM.genBlobs(r, N, 3 + (seed % 2));
    }

    function reset() {
      stop();
      state.centers = []; state.labels = null; state.iter = 0;
      state.converged = false; state.phase = "empty";
      draw();
    }

    function step() {
      var k = +q("k").value;
      if (state.converged) return;
      if (state.phase === "empty") {
        state.centers = KM.initCenters(state.points, k, rng);
        state.phase = "init";
      } else if (state.phase === "init" || state.phase === "updated") {
        var nl = KM.assign(state.points, state.centers);
        if (state.phase === "updated" && KM.sameLabels(nl, state.labels)) {
          state.converged = true; stop();
        }
        state.labels = nl; state.iter++;
        state.phase = "assigned";
      } else if (state.phase === "assigned") {
        state.centers = KM.update(state.points, state.labels, state.centers);
        state.phase = "updated";
      }
      draw();
    }

    function stop() {
      if (timer) { clearInterval(timer); timer = null; }
      q("play").textContent = "自動播放";
    }

    function toX(x) { return 20 + x * (W - 40); }
    function toY(y) { return H - 20 - y * (H - 40); }

    function draw() {
      var bg = cssVar("--md-default-bg-color", "#ffffff");
      var fg = cssVar("--md-default-fg-color", "#222222");
      var muted = cssVar("--md-default-fg-color--light", "#888888");
      ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);

      // lines from each point to its center during the assignment step
      if (state.labels && state.phase === "assigned") {
        ctx.globalAlpha = 0.18; ctx.lineWidth = 1;
        state.points.forEach(function (p, i) {
          var c = state.centers[state.labels[i]];
          ctx.strokeStyle = COLORS[state.labels[i] % COLORS.length];
          ctx.beginPath(); ctx.moveTo(toX(p[0]), toY(p[1])); ctx.lineTo(toX(c[0]), toY(c[1])); ctx.stroke();
        });
        ctx.globalAlpha = 1;
      }
      state.points.forEach(function (p, i) {
        ctx.fillStyle = state.labels ? COLORS[state.labels[i] % COLORS.length] : muted;
        ctx.globalAlpha = state.labels ? 0.85 : 0.6;
        ctx.beginPath(); ctx.arc(toX(p[0]), toY(p[1]), 4, 0, 2 * Math.PI); ctx.fill();
      });
      ctx.globalAlpha = 1;
      state.centers.forEach(function (c, j) {
        var x = toX(c[0]), y = toY(c[1]), s = 9;
        ctx.lineWidth = 6; ctx.strokeStyle = fg;
        ctx.beginPath(); ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s);
        ctx.moveTo(x + s, y - s); ctx.lineTo(x - s, y + s); ctx.stroke();
        ctx.lineWidth = 3; ctx.strokeStyle = COLORS[j % COLORS.length];
        ctx.beginPath(); ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s);
        ctx.moveTo(x + s, y - s); ctx.lineTo(x - s, y + s); ctx.stroke();
      });

      var msg;
      if (state.phase === "empty") msg = "按「下一步」隨機放下 K 個群中心（X）。也可以點畫布新增資料點。";
      else if (state.phase === "init") msg = "步驟 1：已隨機選 " + state.centers.length + " 個點當群中心。下一步：每個點找最近的中心。";
      else if (state.converged) msg = "已收斂（第 " + state.iter + " 輪分配結果和上一輪相同）。群內平方和 = " +
        KM.inertia(state.points, state.labels, state.centers).toFixed(3) + "。試試「重新放中心」，結果一樣嗎？";
      else if (state.phase === "assigned") msg = "第 " + state.iter + " 輪・步驟 2（分配）：每個點歸給最近的中心。群內平方和 = " +
        KM.inertia(state.points, state.labels, state.centers).toFixed(3) + "。下一步：中心搬到成員平均。";
      else msg = "第 " + state.iter + " 輪・步驟 3（更新）：中心已搬到成員的平均位置。下一步：重新分配。";
      statusEl.textContent = msg;
    }

    q("step").addEventListener("click", function () { stop(); step(); });
    q("play").addEventListener("click", function () {
      if (timer) { stop(); return; }
      if (state.converged) reset();
      q("play").textContent = "暫停";
      step();
      timer = setInterval(function () {
        if (!document.body.contains(canvas)) { stop(); return; }
        step();
      }, 700);
    });
    q("reinit").addEventListener("click", reset);
    q("newdata").addEventListener("click", function () { seed++; makeData(); reset(); });
    q("data").addEventListener("change", function () { makeData(); reset(); });
    q("k").addEventListener("input", function () { q("kval").textContent = q("k").value; reset(); });
    canvas.addEventListener("click", function (e) {
      var rect = canvas.getBoundingClientRect();
      var px = (e.clientX - rect.left) * (W / rect.width), py = (e.clientY - rect.top) * (H / rect.height);
      var x = (px - 20) / (W - 40), y = (H - 20 - py) / (H - 40);
      if (x < 0 || x > 1 || y < 0 || y > 1) return;
      state.points.push([x, y]);
      if (state.labels) { state.converged = false; state.labels = KM.assign(state.points, state.centers); state.phase = "assigned"; }
      draw();
    });

    // redraw when the reader toggles light/dark mode
    new MutationObserver(draw).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });

    makeData();
    reset();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setup);
  } else {
    setup();
  }
})();
