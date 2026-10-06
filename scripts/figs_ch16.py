"""Figures for chapter 16 (autoencoders and generative models).

Run from ml-site/:
    .venv-colab/bin/python scripts/figs_ch16.py     # needs keras + tensorflow

Outputs PNGs to docs/assets/img/ch16/ and the histogram data used by the
interactive threshold demo to docs/assets/js/demos/ch16-threshold-data.js.

No data is stored in the repo or on the site. The script executes every code
cell of the executed notebook docs/notebooks/ch16_generative.ipynb (same code,
same seeds) inside docs/notebooks/, so the official PhysioNet zip lands in the
git-ignored docs/notebooks/data/ and is downloaded only if missing. It then
asserts that the re-trained numbers (AUC, threshold, sensitivity, specificity,
split sizes, histogram counts, leaky-split comparison) equal the RESULTS_JSON
line saved in the notebook; if they differ (other hardware / versions), it stops
instead of drawing figures that disagree with the chapter text.
"""
import contextlib
import io
import json
import os
import re
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"   # must be set before importing keras

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
CLASS_COLORS = [GREY, "#1f77b4", ORANGE, "#9467bd"]
CLASS_NAMES = ["N 正常類", "S 上心室異位", "V 心室異位", "F 融合"]
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "img" / "ch16"
OUT.mkdir(parents=True, exist_ok=True)
DEMO_DATA = ROOT / "docs" / "assets" / "js" / "demos" / "ch16-threshold-data.js"
NB_PATH = ROOT / "docs" / "notebooks" / "ch16_generative.ipynb"
NB = json.loads(NB_PATH.read_text())
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
STDOUT = ["".join("".join(o.get("text", "")) for o in c.get("outputs", []) if o.get("output_type") == "stream")
          for c in NB["cells"] if c["cell_type"] == "code"]


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def arrow(ax, p, q, color=GREY, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))


def notebook_results():
    hits = [o for o in STDOUT if "RESULTS_JSON" in o]
    assert len(hits) == 1
    return json.loads(re.search(r"RESULTS_JSON (\{.*\})", hits[0]).group(1))


def run_notebook():
    """Execute every notebook code cell (plots suppressed) and return the namespace."""
    ns = {}
    show = plt.show
    plt.show = lambda *a, **k: plt.close("all")
    here = os.getcwd()
    os.chdir(NB_PATH.parent)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            for src in CODE:
                exec(src, ns)
    finally:
        os.chdir(here)
        plt.show = show
    out = buf.getvalue()
    ns["_stdout"] = out
    return ns, json.loads(re.search(r"RESULTS_JSON (\{.*\})", out).group(1))


def check_consistency(saved, rerun):
    """Every number the chapter quotes must equal the executed notebook's saved output."""
    keys = ["subjects", "beats", "params", "ae", "pca", "pca_var", "random", "tradeoff",
            "fp_top_subject", "spec_without_top", "fp_total", "fp_top_n", "pca_fp_total", "pca_fp_top_n",
            "train_mse", "val_mse", "comp_main", "comp_random", "sweep", "robust", "hist_bins", "hist_counts"]
    assert set(keys) == set(saved), set(saved) ^ set(keys)          # every saved number is checked
    for key in keys:
        assert saved[key] == rerun[key], f"re-run differs from saved notebook output: {key}"
    # derived quantities add up
    b = saved["beats"]
    assert sum(sum(v) for v in saved["hist_counts"].values()) == b["test"]
    assert b["test"] - sum(saved["hist_counts"]["N"]) == b["test_abnormal"]
    s = saved["subjects"]
    assert s["train"] + s["val"] + s["test"] == 43
    assert sum(saved["comp_main"]) == b["test"] and saved["comp_main"][0] == sum(saved["hist_counts"]["N"])
    assert saved["fp_top_n"] <= saved["fp_total"] and saved["pca_fp_top_n"] <= saved["pca_fp_total"]
    assert len(saved["robust"]) == 9 and {(r["fold"], r["seed"]) for r in saved["robust"]} == \
        {(f, sd) for f in (1, 2, 3) for sd in (42, 1, 2)}
    main = [r for r in saved["robust"] if r["fold"] == 1 and r["seed"] == 42][0]
    assert main["AUC"] == saved["ae"]["auc"] and main["sens"] == round(saved["ae"]["sensitivity"], 4)
    print("consistency check passed: re-run == saved notebook RESULTS_JSON")


# ---------------------------------------------------------------- 1. autoencoder schematic
def fig_ae_schematic(ns):
    te, y, X2 = ns["te"], ns["y"], ns["X2"]
    i = int(np.where(y[te] == 0)[0][0])
    beat = X2[te[i]]
    z = ns["encoder"].predict(beat[None], verbose=0)[0]
    rec = ns["ae"].predict(beat[None], verbose=0)[0]
    t = np.linspace(0, 1, len(beat))

    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.6, 3.6)
    ax.axis("off")

    def mini_beat(x0, sig, color, title):
        s = (sig - sig.min()) / (sig.max() - sig.min())
        ax.plot(x0 + t * 1.7, 0.8 + s * 1.6, color=color, lw=1.6)
        ax.text(x0 + 0.85, 2.75, title, ha="center", fontsize=10)

    mini_beat(0.1, beat, GREY, "輸入：一個心搏\n117 個數字")
    ax.add_patch(Polygon([[2.3, 0.4], [4.0, 1.25], [4.0, 2.15], [2.3, 3.0]], color=TEAL, alpha=0.25))
    ax.text(3.15, 1.7, "編碼器\n117→64→32→8", ha="center", va="center", fontsize=10)
    zz = (z - z.min()) / (z.max() - z.min() + 1e-9)
    for k, v in enumerate(zz):
        ax.add_patch(FancyBboxPatch((4.25 + k * 0.17, 1.25), 0.12, 0.1 + 0.8 * v, boxstyle="round,pad=0.01",
                                    color=ORANGE, lw=0))
    ax.text(4.9, 2.75, "瓶頸：潛在向量\n只有 8 個數字", ha="center", fontsize=10)
    ax.text(4.9, 0.95, "（實際值，示意高度）", ha="center", fontsize=8, color=GREY)
    ax.add_patch(Polygon([[5.8, 1.25], [7.5, 0.4], [7.5, 3.0], [5.8, 2.15]], color=TEAL, alpha=0.25))
    ax.text(6.65, 1.7, "解碼器\n8→32→64→117", ha="center", va="center", fontsize=10)
    mini_beat(7.75, rec, ORANGE, "輸出：還原的心搏\n117 個數字")
    s_in = (beat - beat.min()) / (beat.max() - beat.min())
    ax.plot(9.5 + t * 1.4, 0.8 + s_in * 1.6, color=GREY, lw=1)
    ax.text(10.2, 2.75, "比一比", ha="center", fontsize=10)
    ax.plot(9.5 + t * 1.4, 0.8 + (rec - beat.min()) / (beat.max() - beat.min()) * 1.6, color=ORANGE, lw=1, ls="--")
    ax.text(5.5, -0.35, "訓練目標：輸出 ≈ 輸入（重建誤差越小越好），不需要任何標籤；"
            "只拿正常心搏訓練，它就只學會「還原正常心搏」",
            ha="center", fontsize=10, color=TEAL)
    save(fig, "ae_schematic.png")


# ---------------------------------------------------------------- 2. reconstruction examples
def fig_recon(ns, res):
    te, y, X2, groups = ns["te"], ns["y"], ns["X2"], ns["groups"]
    err = ns["err_te_ae"]
    thr = res["ae"]["threshold"]
    t_ms = (np.arange(X2.shape[1]) * ns["STEP"] - ns["PRE"]) / 360 * 1000
    rng = np.random.default_rng(42)
    picks = []
    for k in range(4):
        idx = np.where((y[te] == k) & (groups[te] != res["fp_top_subject"]))[0]
        picks.append((k, int(rng.choice(idx)), CLASS_NAMES[k]))
    idx117 = np.where((y[te] == 0) & (groups[te] == res["fp_top_subject"]))[0]
    picks.append((0, int(rng.choice(idx117)), f"N 正常類（受試者 {res['fp_top_subject']}）"))
    fig, axes = plt.subplots(1, 5, figsize=(10.5, 2.9), sharey=True)
    for ax, (k, i, name) in zip(axes, picks):
        rec = ns["ae"].predict(X2[te[i]][None], verbose=0)[0]
        ax.plot(t_ms, X2[te[i]], color=GREY, lw=2, label="原始")
        ax.plot(t_ms, rec, color=ORANGE, lw=1.5, ls="--", label="還原")
        flag = "判異常" if err[i] > thr else "判正常"
        ax.set_title(f"{name}\n重建誤差 {err[i]:.3f} → {flag}", fontsize=10,
                     color=ORANGE if err[i] > thr else TEAL)
        ax.set_xlabel("距 R 峰（毫秒）", fontsize=9)
    axes[0].set_ylabel("標準化振幅")
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(f"測試組心搏與自編碼器的還原（閾值 {thr:.3f}）", fontsize=11, y=1.12)
    save(fig, "recon_examples.png")


# ---------------------------------------------------------------- 3. error histogram
def fig_hist(res):
    bins = np.array(res["hist_bins"])
    c = res["hist_counts"]
    n = np.array(c["N"])
    a = np.array(c["S"]) + np.array(c["V"]) + np.array(c["F"])
    centers = (bins[:-1] + bins[1:]) / 2
    w = bins[1] - bins[0]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(centers, n / n.sum() / w, w, color=GREY, alpha=0.6, label=f"N 正常類（{n.sum():,} 個）")
    ax.bar(centers, a / a.sum() / w, w, color=ORANGE, alpha=0.6, label=f"S／V／F 異常（{a.sum():,} 個）")
    ymax = 1.15 * max((n / n.sum() / w).max(), (a / a.sum() / w).max())
    ax.set_ylim(0, ymax * 1.12)
    lines = []
    for row, ls in zip(res["tradeoff"], [":", "-.", "--", (0, (1, 3))]):
        q = int(round(row["validation percentile"] * 100))
        x = np.log10(row["threshold"])
        ax.axvline(x, color="black", ls=ls, lw=1.8 if q == 95 else 1)
        ax.text(x, ymax * 1.06, f"{q}", ha="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        lines.append(f"第 {q} 百分位：敏感度 {row['sensitivity']:.2f}、特異度 {row['specificity']:.2f}")
    ax.text(0.99, 0.97, "閾值＝驗證組正常心搏誤差的\n" + "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8"))
    i117 = int(np.argmax(np.where(centers > -0.6, n, 0)))
    ax.annotate(f"多為受試者 {res['fp_top_subject']} 的正常心搏", xy=(centers[i117], n[i117] / n.sum() / w),
                xytext=(centers[i117] + 0.05, ymax * 0.45), fontsize=8.5, color=GREY,
                arrowprops=dict(arrowstyle="-|>", color=GREY))
    ax.set_xticks([-3, -2, -1, 0, 1], ["0.001", "0.01", "0.1", "1", "10"])
    ax.set_xlim(-2.6, 1.3)
    ax.set_xlabel("重建誤差（對數刻度）")
    ax.set_ylabel("密度（各組面積為 1）")
    ax.set_title(f"測試組 15 位受試者的重建誤差；直線＝依驗證組正常心搏的百分位數定出的閾值（AUC {res['ae']['auc']:.2f}）",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "error_hist.png")


# ---------------------------------------------------------------- 4. latent space
def fig_latent(ns):
    te, y, z2 = ns["te"], ns["y"], ns["z2"]
    fig, ax = plt.subplots(figsize=(7, 5.4))
    for k in range(4):
        idx = np.where(y[te] == k)[0]
        idx = idx[:: max(1, len(idx) // 1500)]
        ax.scatter(z2[idx, 0], z2[idx, 1], s=5, alpha=0.5, color=CLASS_COLORS[k],
                   label=f"{CLASS_NAMES[k]}（n = {int((y[te] == k).sum()):,}）")
    ax.set_xlabel("潛在空間主成分 1")
    ax.set_ylabel("潛在空間主成分 2")
    ax.set_title("測試組心搏的 8 維潛在向量，用 PCA 投影到 2 維\n（每類最多畫約 1,500 點；模型訓練時沒看過任何異常心搏）",
                 fontsize=10)
    ax.legend(markerscale=4, fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "latent_2d.png")


# ---------------------------------------------------------------- 5. generative family (concept)
def fig_family():
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2))
    for ax in axes.flat:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")

    def box(ax, x, y, w, h, text, color=TEAL, alpha=0.18, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", color=color, alpha=alpha, lw=0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    a = axes[0, 0]
    a.set_title("自編碼器（AE）：壓縮再還原", fontsize=11)
    box(a, 0.2, 1.5, 1.6, 1, "心搏")
    box(a, 2.4, 1.5, 1.8, 1, "編碼器")
    box(a, 4.8, 1.6, 1.0, 0.8, "z\n（一個點）", ORANGE, 0.25)
    box(a, 6.4, 1.5, 1.8, 1, "解碼器")
    box(a, 8.6, 1.5, 1.3, 1, "還原")
    for x0, x1 in [(1.85, 2.35), (4.25, 4.75), (5.85, 6.35), (8.25, 8.55)]:
        arrow(a, (x0, 2), (x1, 2))
    a.text(5, 0.6, "用途：壓縮、去雜訊、異常偵測（本章實作）\n潛在空間不保證「連續」，隨便取一點解碼常常不像心搏",
           ha="center", fontsize=9, color=GREY)

    a = axes[0, 1]
    a.set_title("變分自編碼器（VAE）：瓶頸變成一團機率分布", fontsize=11)
    box(a, 0.2, 1.5, 1.6, 1, "心搏")
    box(a, 2.4, 1.5, 1.8, 1, "編碼器")
    box(a, 4.6, 1.5, 1.4, 1, "平均 μ\n標準差 σ", ORANGE, 0.25)
    box(a, 6.4, 1.5, 1.8, 1, "解碼器")
    box(a, 8.6, 1.5, 1.3, 1, "還原")
    for x0, x1 in [(1.85, 2.35), (4.25, 4.55), (6.05, 6.35), (8.25, 8.55)]:
        arrow(a, (x0, 2), (x1, 2))
    a.text(5.3, 3.0, "從 N(μ, σ²) 抽一個 z", ha="center", fontsize=9, color=ORANGE)
    a.text(5, 0.6, "訓練目標＝還原得像 ＋ 讓這團分布接近標準常態\n→ 從標準常態抽 z 交給解碼器，就能「生成」新樣本",
           ha="center", fontsize=9, color=GREY)

    a = axes[1, 0]
    a.set_title("生成對抗網路（GAN）：偽造者對上鑑定師", fontsize=11)
    box(a, 0.2, 2.4, 1.6, 0.9, "隨機雜訊")
    box(a, 2.3, 2.4, 2.0, 0.9, "生成器\n（偽造者）", ORANGE, 0.25)
    box(a, 4.9, 2.4, 1.6, 0.9, "假樣本")
    box(a, 4.9, 0.9, 1.6, 0.9, "真樣本")
    box(a, 7.0, 1.6, 2.0, 1.0, "鑑別器\n（鑑定師）\n真？假？")
    arrow(a, (1.85, 2.85), (2.25, 2.85))
    arrow(a, (4.35, 2.85), (4.85, 2.85))
    arrow(a, (6.55, 2.8), (6.95, 2.3))
    arrow(a, (6.55, 1.35), (6.95, 1.9))
    a.annotate("", xy=(3.3, 3.4), xytext=(8.0, 2.7),
               arrowprops=dict(arrowstyle="-|>", color=ORANGE, ls="--", connectionstyle="arc3,rad=0.3"))
    a.text(5.2, 0.25, "鑑定結果回饋給偽造者：兩邊一起進步，直到假樣本難以分辨", ha="center", fontsize=9, color=GREY)

    a = axes[1, 1]
    a.set_title("擴散模型：學會一步一步「去雜訊」", fontsize=11)
    xs = np.linspace(0.4, 8.6, 5)
    labels = ["真樣本", "", "", "", "純雜訊"]
    for k, x in enumerate(xs):
        box(a, x, 2.3, 1.1, 0.9, labels[k] or f"加雜訊\n第 {k} 步", TEAL, 0.10 + 0.08 * k)
    for x in xs[:-1]:
        arrow(a, (x + 1.15, 2.95), (x + 2.0, 2.95), GREY)
        arrow(a, (x + 2.0, 2.45), (x + 1.15, 2.45), ORANGE)
    a.text(5, 1.6, "灰色箭頭（往右）：固定的加雜訊過程，不需要學", ha="center", fontsize=9, color=GREY)
    a.text(5, 1.1, "橘色箭頭（往左）：神經網路學「這一步的雜訊長什麼樣」並把它扣掉", ha="center", fontsize=9, color=ORANGE)
    a.text(5, 0.5, "生成時從純雜訊出發，反覆去雜訊幾十到上千步", ha="center", fontsize=9, color=GREY)
    fig.suptitle("四種常見的生成式架構（概念示意，非實際模型）", fontsize=12)
    plt.tight_layout()
    save(fig, "generative_family.png")


# ---------------------------------------------------------------- 6. diffusion forward process on a real beat
def fig_diffusion(ns):
    """Forward noising q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I) on the mean training N beat."""
    X2, fit_n = ns["X2"], ns["fit_n"]
    x0 = X2[fit_n].mean(axis=0)
    x0 = x0 / x0.std()
    T = 1000
    betas = np.linspace(1e-4, 0.02, T)          # linear schedule of Ho et al. 2020
    abar = np.cumprod(1 - betas)
    rng = np.random.default_rng(42)
    eps = rng.standard_normal(len(x0))           # one fixed noise draw for all panels
    t_ms = (np.arange(len(x0)) * ns["STEP"] - ns["PRE"]) / 360 * 1000
    steps = [0, 50, 150, 300, 1000]
    fig, axes = plt.subplots(1, len(steps), figsize=(10.5, 2.5), sharey=True)
    for ax, s in zip(axes, steps):
        xt = x0 if s == 0 else np.sqrt(abar[s - 1]) * x0 + np.sqrt(1 - abar[s - 1]) * eps
        ax.plot(t_ms, xt, color=TEAL if s == 0 else GREY, lw=1.4)
        ax.set_title("第 0 步：原始平均心搏" if s == 0 else f"第 {s} 步", fontsize=10)
        ax.set_xticks([])
    axes[0].set_ylabel("振幅")
    fig.text(0.5, -0.06, "→ 往右：照固定公式加雜訊（不需要學）　　← 往左：擴散模型要學會的，是從雜訊一步步還原成心搏",
             ha="center", fontsize=10, color=ORANGE)
    fig.suptitle("擴散模型的「加雜訊」過程：以訓練組正常心搏的平均波形示範（線性排程，共 1,000 步）",
                 fontsize=11, y=1.06)
    save(fig, "diffusion_noise.png")


def write_demo_data(res):
    data = {"bins": res["hist_bins"], "counts": res["hist_counts"],
            "threshold95": res["ae"]["threshold"], "auc": res["ae"]["auc"]}
    DEMO_DATA.write_text(
        "// Auto-generated by scripts/figs_ch16.py from the RESULTS_JSON line of\n"
        "// docs/notebooks/ch16_generative.ipynb: histogram of log10(reconstruction error)\n"
        "// of the 15 test subjects' beats (MIT-BIH, ODC-By), per AAMI class.\n"
        "window.CH16_THRESHOLD_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print("saved", DEMO_DATA)


if __name__ == "__main__":
    saved = notebook_results()
    fig_family()
    ns, rerun = run_notebook()
    check_consistency(saved, rerun)
    write_demo_data(saved)
    fig_ae_schematic(ns)
    fig_recon(ns, saved)
    fig_hist(saved)
    fig_latent(ns)
    fig_diffusion(ns)
