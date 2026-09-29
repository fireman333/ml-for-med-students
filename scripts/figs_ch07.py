"""Generate figures (and the interactive-demo data) for Chapter 07: decision tree & random forest.

Run from ml-site/:  .venv/bin/python scripts/figs_ch07.py
Outputs:
  docs/assets/img/ch07/*.png
  docs/assets/js/demos/ch07-depth.js   (Plotly depth-slider demo with embedded precomputed grids)
"""
import io
import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.tree import DecisionTreeClassifier, plot_tree

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "assets" / "img" / "ch07"
JS = ROOT / "docs" / "assets" / "js" / "demos" / "ch07-depth.js"
IMG.mkdir(parents=True, exist_ok=True)

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
SEED = 42

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
ZH = {
    "age": "年齡", "sex": "性別(男=1)", "cp": "胸痛型態", "trestbps": "靜止血壓",
    "chol": "膽固醇", "fbs": "空腹血糖>120", "restecg": "靜止心電圖",
    "thalach": "最大心跳", "exang": "運動誘發心絞痛", "oldpeak": "運動ST下降",
    "slope": "ST斜率", "ca": "透視顯影血管數", "thal": "鉈掃描結果",
    "random_id": "隨機編號(雜訊)",
}


def load_cleveland():
    with urllib.request.urlopen(URL, timeout=60) as r:
        raw = r.read().decode()
    df = pd.read_csv(io.StringIO(raw), names=COLS, na_values="?")
    df = df.dropna().reset_index(drop=True)
    y = (df["num"] > 0).astype(int)
    X = df.drop(columns="num")
    return X, y


def save(fig, name):
    fig.savefig(IMG / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", IMG / name)


def fig_impurity():
    p = np.linspace(0.001, 0.999, 300)
    gini = 1 - p**2 - (1 - p) ** 2
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(p, gini, color=TEAL, lw=2.5, label="Gini 不純度（最大 0.5）")
    ax.plot(p, ent, color=ORANGE, lw=2.5, label="Entropy 熵（最大 1）")
    ax.axvline(0.5, color=GREY, ls="--", lw=1)
    ax.annotate("一半有病、一半沒病\n＝最混亂", xy=(0.5, 0.5), xytext=(0.58, 0.18),
                arrowprops=dict(arrowstyle="->", color=GREY), fontsize=10, color=GREY)
    ax.annotate("全部同一類\n＝最純", xy=(0.02, 0.03), xytext=(0.06, 0.55),
                arrowprops=dict(arrowstyle="->", color=GREY), fontsize=10, color=GREY)
    ax.set_xlabel("節點裡「有病」的比例 p")
    ax.set_ylabel("不純度")
    ax.set_title("兩種不純度指標：愈混雜愈高，愈單純愈低")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "impurity_curves.png")


def fig_tree(Xtr, ytr):
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Xtr, ytr)
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    plot_tree(tree, feature_names=[ZH[c] for c in Xtr.columns],
              class_names=["無狹窄", "有狹窄"], filled=True, rounded=True,
              impurity=True, proportion=False, fontsize=8, ax=ax)
    ax.set_title("深度 3 的決策樹（Cleveland 心臟病資料，訓練集 207 人）", fontsize=12)
    save(fig, "tree_depth3.png")


def fig_depth(X, y):
    depths = list(range(1, 13))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    tr_m, te_m, te_s = [], [], []
    for d in depths:
        r = cross_validate(DecisionTreeClassifier(max_depth=d, random_state=SEED), X, y,
                           cv=cv, return_train_score=True)
        tr_m.append(r["train_score"].mean())
        te_m.append(r["test_score"].mean())
        te_s.append(r["test_score"].std())
    tr_m, te_m, te_s = map(np.array, (tr_m, te_m, te_s))
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(depths, tr_m, "o-", color=GREY, lw=2, label="訓練集準確率")
    ax.plot(depths, te_m, "o-", color=TEAL, lw=2.5, label="驗證集準確率（5 折 × 5 次平均）")
    ax.fill_between(depths, te_m - te_s, te_m + te_s, color=TEAL, alpha=0.15)
    best = depths[int(np.argmax(te_m))]
    ax.axvline(best, color=ORANGE, ls="--", lw=1.2)
    ax.text(best + 0.2, 0.62, f"驗證集最好：深度 {best}", color=ORANGE, fontsize=10)
    ax.text(9.2, 0.97, "背下訓練集", color=GREY, fontsize=10)
    ax.set_xlabel("樹的最大深度 max_depth")
    ax.set_ylabel("準確率")
    ax.set_ylim(0.6, 1.03)
    ax.set_xticks(depths)
    ax.set_title("樹愈深，訓練分數一路上升，驗證分數卻不跟著升")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "depth_overfit.png")
    print("depth curve (val mean):", dict(zip(depths, te_m.round(3))))


def fig_forest(Xtr, ytr):
    ns = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
    oob = []
    for n in ns:
        rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=SEED, n_jobs=-1)
        import warnings
        with warnings.catch_warnings():
            # very small forests leave some samples without OOB votes; expected here
            warnings.simplefilter("ignore", UserWarning)
            rf.fit(Xtr, ytr)
        oob.append(rf.oob_score_)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(ns, oob, "o-", color=TEAL, lw=2.5, label="隨機森林的 OOB 準確率")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("樹的數量 n_estimators（對數刻度）")
    ax.set_ylabel("OOB 準確率")
    ax.set_title("樹多到一定數量後，分數就趨於平穩（不會因樹多而過度配適）")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "forest_oob.png")
    print("oob:", dict(zip(ns, np.round(oob, 3))))


def importance_cv(X, y):
    """Average impurity & permutation importance over 5 CV folds, with one pure-noise column added."""
    rng = np.random.default_rng(SEED)
    X2 = X.copy()
    X2["random_id"] = rng.integers(0, 1000, len(X2))
    imps, perms = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X2, y):
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
        rf.fit(X2.iloc[tr], y.iloc[tr])
        imps.append(rf.feature_importances_)
        pi = permutation_importance(rf, X2.iloc[te], y.iloc[te], n_repeats=10,
                                    random_state=SEED, n_jobs=-1)
        perms.append(pi.importances_mean)
    imp = pd.Series(np.mean(imps, axis=0), index=X2.columns)
    perm = pd.Series(np.mean(perms, axis=0), index=X2.columns)
    perm_sd = pd.Series(np.std(perms, axis=0), index=X2.columns)
    return imp, perm, perm_sd


def fig_importance(X, y):
    imp, perm, perm_sd = importance_cv(X, y)
    rank_imp = list(imp.sort_values(ascending=False).index).index("random_id") + 1
    rank_perm = list(perm.sort_values(ascending=False).index).index("random_id") + 1
    order = imp.sort_values().index
    colors = [ORANGE if c == "random_id" else TEAL for c in order]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    axes[0].barh([ZH[c] for c in order], imp[order], color=colors)
    axes[0].set_title("不純度重要性（feature_importances_）", fontsize=11)
    axes[0].set_xlabel("平均不純度下降")
    axes[1].barh([ZH[c] for c in order], perm[order], xerr=perm_sd[order], color=colors,
                 error_kw=dict(ecolor=GREY, lw=0.8, capsize=2))
    axes[1].set_title("排列重要性（在驗證折上打亂）", fontsize=11)
    axes[1].set_xlabel("打亂後準確率下降（誤差線＝各折標準差）")
    for ax in axes:
        ax.axvline(0, color=GREY, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"加入一欄純雜訊「隨機編號」：不純度重要性排第 {rank_imp}（共 14 欄），"
                 f"排列重要性約為 0（排第 {rank_perm}）", fontsize=12)
    fig.tight_layout()
    save(fig, "importance_compare.png")
    print("random_id impurity", round(imp["random_id"], 3), "rank", rank_imp,
          "| permutation", round(perm["random_id"], 4), "rank", rank_perm)
    print("permutation:", perm.sort_values(ascending=False).round(3).to_dict())


def write_demo(X, y):
    feats = ["thalach", "oldpeak"]
    X2 = X[feats]
    Xtr, Xte, ytr, yte = train_test_split(X2, y, test_size=0.3, stratify=y, random_state=SEED)
    gx = np.linspace(70, 205, 70)
    gy = np.linspace(-0.3, 6.5, 70)
    xx, yy = np.meshgrid(gx, gy)
    grid = pd.DataFrame({"thalach": xx.ravel(), "oldpeak": yy.ravel()})
    depths = []
    for d in range(1, 13):
        t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(Xtr, ytr)
        z = t.predict_proba(grid)[:, 1]
        depths.append({
            "depth": d,
            "leaves": int(t.get_n_leaves()),
            "train": round(float(t.score(Xtr, ytr)), 3),
            "test": round(float(t.score(Xte, yte)), 3),
            # store P(disease) rounded to 1 decimal to keep file small
            "z": "".join(str(min(9, int(v * 10))) for v in z),
        })
    data = {
        "gx": [round(float(v), 2) for v in gx],
        "gy": [round(float(v), 3) for v in gy],
        "train": {"x": Xtr["thalach"].tolist(), "y": Xtr["oldpeak"].tolist(), "c": ytr.tolist()},
        "test": {"x": Xte["thalach"].tolist(), "y": Xte["oldpeak"].tolist(), "c": yte.tolist()},
        "depths": depths,
    }
    js = JS_TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    JS.parent.mkdir(parents=True, exist_ok=True)
    JS.write_text(js, encoding="utf-8")
    print("saved", JS, f"({len(js) / 1024:.0f} KB)")
    print("demo acc:", [(d["depth"], d["train"], d["test"]) for d in depths])


JS_TEMPLATE = r"""// Chapter 07 interactive demo: decision-tree depth slider.
// Generated by scripts/figs_ch07.py - do not edit by hand (data are precomputed in Python).
(function () {
  const DATA = __DATA__;
  function draw() {
    const el = document.getElementById("ch07-demo");
    if (!el || typeof Plotly === "undefined") return;
    const slider = document.getElementById("ch07-depth");
    const label = document.getElementById("ch07-depth-val");
    const info = document.getElementById("ch07-info");
    if (!slider) return;
    const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
    const fg = dark ? "#e0e0e0" : "#263238";
    const gridc = dark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.08)";
    const teal = "#00897B", orange = "#F4511E";

    function zMatrix(s) {
      const nx = DATA.gx.length, ny = DATA.gy.length, z = [];
      for (let j = 0; j < ny; j++) {
        const row = [];
        for (let i = 0; i < nx; i++) row.push(Number(s[j * nx + i]) / 10 + 0.05);
        z.push(row);
      }
      return z;
    }
    function pts(set, cls, name, symbol) {
      const xs = [], ys = [];
      set.c.forEach((c, k) => { if (c === cls) { xs.push(set.x[k]); ys.push(set.y[k]); } });
      return {
        type: "scatter", mode: "markers", x: xs, y: ys, name: name,
        marker: { symbol: symbol, size: symbol === "x" ? 8 : 7, color: cls ? orange : teal,
                  line: { width: symbol === "x" ? 0 : 1, color: dark ? "#111" : "#fff" } },
        hovertemplate: "最大心跳 %{x}<br>ST 下降 %{y}<extra>" + name + "</extra>"
      };
    }
    function render() {
      const d = DATA.depths[Number(slider.value) - 1];
      label.textContent = d.depth;
      info.innerHTML = "葉節點 " + d.leaves + " 個 ・ 訓練集準確率 <b>" +
        (d.train * 100).toFixed(0) + "%</b> ・ 測試集準確率 <b>" + (d.test * 100).toFixed(0) + "%</b>";
      const traces = [
        { type: "heatmap", x: DATA.gx, y: DATA.gy, z: zMatrix(d.z), zmin: 0, zmax: 1,
          colorscale: [[0, "rgba(0,137,123,0.35)"], [0.5, "rgba(200,200,200,0.15)"], [1, "rgba(244,81,30,0.35)"]],
          showscale: false, hoverinfo: "skip" },
        pts(DATA.train, 0, "訓練：無狹窄", "circle"),
        pts(DATA.train, 1, "訓練：有狹窄", "circle"),
        pts(DATA.test, 0, "測試：無狹窄", "x"),
        pts(DATA.test, 1, "測試：有狹窄", "x")
      ];
      const layout = {
        margin: { l: 55, r: 10, t: 10, b: 45 }, height: 400,
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: fg, size: 12 },
        xaxis: { title: "運動時最大心跳（thalach）", gridcolor: gridc, range: [70, 205] },
        yaxis: { title: "運動 ST 段下降（oldpeak）", gridcolor: gridc, range: [-0.3, 6.5] },
        legend: { orientation: "h", y: -0.22 }
      };
      Plotly.react(el, traces, layout, { displayModeBar: false, responsive: true });
    }
    slider.oninput = render;
    render();
  }
  // redraw when the reader toggles light/dark mode
  let observing = false;
  function watchScheme() {
    if (observing) return;
    observing = true;
    new MutationObserver(draw).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  }
  function init() { draw(); watchScheme(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
"""


def main():
    X, y = load_cleveland()
    print("data", X.shape, "positive rate", round(y.mean(), 3))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    fig_impurity()
    fig_tree(Xtr, ytr)
    fig_depth(X, y)
    fig_forest(Xtr, ytr)
    fig_importance(X, y)
    write_demo(X, y)


if __name__ == "__main__":
    main()
