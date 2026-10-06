"""Figures for chapter 14 (ECG, 1D-CNN, data leakage).

Run from ml-site/:  .venv/bin/python scripts/figs_ch14.py
Outputs PNGs to docs/assets/img/ch14/.

No data is stored in the repo or on the site. The ECG figure re-runs the
notebook's own download / reading / beat-cutting cells (their source is read
from docs/notebooks/ch14_ecg.ipynb and executed inside docs/notebooks/, so the
official PhysioNet zip lands in the git-ignored docs/notebooks/data/ and is
downloaded only if missing). The class counts are asserted to equal the ones
printed in the executed notebook. The results figure reads every number from the
RESULTS_JSON line saved in the executed notebook; nothing is hand-typed.
"""
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "img" / "ch14"
OUT.mkdir(parents=True, exist_ok=True)
NB_PATH = ROOT / "docs" / "notebooks" / "ch14_ecg.ipynb"
NB = json.loads(NB_PATH.read_text())
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
STDOUT = ["".join("".join(o.get("text", "")) for o in c.get("outputs", []) if o.get("output_type") == "stream")
          for c in NB["cells"] if c["cell_type"] == "code"]


def cell(marker):
    hits = [c for c in CODE if marker in c]
    assert len(hits) == 1, marker
    return hits[0]


def cell_stdout(marker):
    hits = [o for c, o in zip(CODE, STDOUT) if marker in c]
    assert len(hits) == 1, marker
    return hits[0]


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def load_beats():
    """Execute the notebook's own data cells (same code, same parameters)."""
    import hashlib, time, urllib.request, zipfile  # noqa: E401,F401  (used by the exec'd cells)
    import pandas as pd
    from scipy.signal import butter, filtfilt  # noqa: F401
    ns = {"os": os, "time": time, "zipfile": zipfile, "hashlib": hashlib, "urllib": urllib,
          "np": np, "pd": pd, "butter": butter, "filtfilt": filtfilt}
    here = os.getcwd()
    os.chdir(NB_PATH.parent)
    try:
        exec(cell("def plan_a_zip"), ns)
        exec(cell("def read_header"), ns)
        exec(cell("AAMI = {"), ns)
    finally:
        os.chdir(here)
    # class counts must match the executed notebook's printed table
    table = cell_stdout("AAMI = {")
    for k, c in enumerate(ns["CLASSES"]):
        printed = int(re.search(rf"^{c}\s+(\d+)", table, re.M).group(1))
        assert printed == int((ns["y"] == k).sum()), (c, printed)
    return ns


def fig_ecg(ns):
    fs, sig_raw, samp, sym = ns["records"]["208"]
    b, a = ns["butter"](2, [0.5, 40], btype="band", fs=fs)
    sig = ns["filtfilt"](b, a, sig_raw)
    start, length = int(18.0 * fs), int(6 * fs)
    t = np.arange(start, start + length) / fs
    m = (samp >= start) & (samp < start + length) & (sym != "?")

    fig = plt.figure(figsize=(10.5, 6.2))
    ax = fig.add_axes([0.06, 0.58, 0.92, 0.36])
    ax.plot(t, sig[start:start + length], color=TEAL, lw=1)
    ymax = sig[start:start + length].max()
    for s, lab in zip(samp[m], sym[m]):
        ax.plot(s / fs, sig[s], "o", ms=4, color=ORANGE if lab != "N" else GREY)
        ax.text(s / fs, ymax + 0.25, lab, ha="center", fontsize=10, color=ORANGE if lab != "N" else GREY)
    # shade the cutting window of one V beat
    v = [s for s, lab in zip(samp[m], sym[m]) if lab == "V"][0]
    pre, post = ns["PRE"] / fs, ns["POST"] / fs
    ax.add_patch(Rectangle((v / fs - pre, ax.get_ylim()[0]), pre + post, ymax - ax.get_ylim()[0] + 0.15,
                           color=ORANGE, alpha=0.12, lw=0))
    ax.text(v / fs - pre, ymax + 0.75, "橘色區＝以 R 峰為中心切出的一個心跳（R 峰前 0.25 秒、後 0.40 秒）",
            fontsize=9, color=ORANGE)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(top=ymax + 1.2)
    ax.set_xlabel("時間（秒）")
    ax.set_ylabel("mV（已濾波）")
    ax.set_title("MIT-BIH record 208，MLII 導程，360 Hz：點＝標註的 R 峰，字母＝心跳類別", fontsize=11)

    X, y = ns["X"][:, :, 0], ns["y"]
    t_ms = (np.arange(X.shape[1]) * ns["STEP"] - ns["PRE"]) / 360 * 1000
    titles = ["N 正常", "S 上心室異位", "V 心室異位", "F 融合"]
    rng = np.random.default_rng(42)
    for k in range(4):
        axk = fig.add_axes([0.06 + k * 0.235, 0.07, 0.2, 0.36])
        beats = X[y == k]
        idx = rng.choice(len(beats), 30, replace=False)
        axk.plot(t_ms, beats[idx].T, color="0.82", lw=0.5)
        axk.plot(t_ms, beats.mean(axis=0), color=ORANGE if k else TEAL, lw=2)
        axk.set_title(f"{titles[k]}（n = {len(beats):,}）", fontsize=10)
        axk.set_xlabel("距 R 峰（毫秒）", fontsize=9)
        axk.set_ylim(-4, 6)
        if k == 0:
            axk.set_ylabel("標準化振幅", fontsize=9)
        else:
            axk.set_yticklabels([])
    fig.text(0.06, 0.475, "下排：各 AAMI 類別的心跳（灰：隨機 30 個；粗線：平均波形）", fontsize=10)
    save(fig, "ecg_beats.png")


def fig_split_schematic():
    """Toy picture: 6 people x 8 beats, random-beat split vs split by person."""
    rng = np.random.default_rng(42)
    colors = plt.get_cmap("tab10").colors
    n_p, n_b = 6, 8
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    rand_test = rng.random((n_p, n_b)) < 1 / 3
    person_test = np.zeros((n_p, n_b), bool)
    person_test[[1, 4]] = True
    for ax, test, title in [(axes[0], rand_test, "切法 A：把心跳打散後隨機切"),
                            (axes[1], person_test, "切法 B：依受試者切")]:
        for p in range(n_p):
            for j in range(n_b):
                x = j + (n_b + 1.5 if test[p, j] else 0)
                ax.add_patch(Rectangle((x, n_p - 1 - p), 0.85, 0.75, color=colors[p], alpha=0.9))
        ax.text(n_b / 2, n_p + 0.2, "訓練集", ha="center", fontsize=11)
        ax.text(n_b + 1.5 + n_b / 2, n_p + 0.2, "測試集", ha="center", fontsize=11)
        ax.plot([n_b + 0.75] * 2, [-0.2, n_p + 0.6], color=GREY, ls="--", lw=1)
        ax.set_xlim(-0.5, 2 * n_b + 2)
        ax.set_ylim(-1.0, n_p + 0.9)
        ax.axis("off")
        ax.set_title(title, fontsize=12)
    axes[0].text(n_b + 0.75, -0.75, "同一個人（同一個顏色）兩邊都有 → 考題見過", ha="center", fontsize=9, color=ORANGE)
    axes[1].text(n_b + 0.75, -0.75, "測試集的人完全沒在訓練出現 → 像真的新病人", ha="center", fontsize=9, color=TEAL)
    fig.text(0.5, -0.02, "示意圖：每一列是一位受試者，每一格是他的一個心跳（實際每人約 1,500–4,100 個）",
             ha="center", fontsize=9, color=GREY)
    save(fig, "split_schematic.png")


def fig_results():
    out = cell_stdout("RESULTS_JSON")
    rows = json.loads(re.search(r"RESULTS_JSON (\[.*\])", out).group(1))
    splits = ["A: random beats", "B: by subject"]
    for s in splits:
        assert sorted(r["fold"] for r in rows if r["split"] == s) == [1, 2, 3]
    metrics = [("acc", "準確率"), ("macro_f1", "macro-F1"), ("sens_N", "敏感度 N"),
               ("sens_S", "敏感度 S"), ("sens_V", "敏感度 V"), ("sens_F", "敏感度 F")]
    labels = {"A: random beats": "切法 A：隨機切心跳", "B: by subject": "切法 B：依受試者切"}
    colors = {"A: random beats": GREY, "B: by subject": TEAL}

    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.36
    for i, s in enumerate(splits):
        for k, (m, _) in enumerate(metrics):
            vals = np.array([r[m] for r in rows if r["split"] == s])
            x = k + (i - 0.5) * w
            ax.bar(x, vals.mean(), w * 0.92, color=colors[s], alpha=0.85, label=labels[s] if k == 0 else None)
            ax.plot([x, x], [vals.min(), vals.max()], color="black", lw=1)
            ax.plot(np.full(3, x), vals, "o", ms=3.5, color="black")
            ax.text(x, vals.max() + 0.03, f"{vals.mean():.2f}", ha="center", fontsize=8)
    base = np.mean([r["baseline_acc"] for r in rows])
    ax.plot([-0.45, 0.45], [base, base], color=ORANGE, ls="--", lw=1.5,
            label=f"「全部猜 N」的準確率（多數類基準，約 {base:.2f}）")
    ax.set_xticks(range(len(metrics)), [m[1] for m in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("測試集分數")
    ax.set_title("同一個 1D-CNN、只換切法（MIT-BIH，3 折；長條＝平均，黑點＝各折，直線＝範圍）", fontsize=11)
    ax.legend(loc="upper center", fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "split_results.png")


if __name__ == "__main__":
    fig_split_schematic()
    fig_results()
    fig_ecg(load_beats())
