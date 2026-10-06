"""Figures (and demo data) for chapter 12 (CNN).

Run from ml-site/:  .venv/bin/python scripts/figs_ch12.py
Outputs PNGs to docs/assets/img/ch12/ and the X-ray pixel data used by the
interactive demo to docs/assets/js/demos/ch12-conv-data.js.

Needs only numpy/matplotlib (no Keras). PneumoniaMNIST (CC BY 4.0, MedMNIST v2)
is downloaded once from Zenodo into the system temp directory.
"""
import json
import tempfile
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "img" / "ch12"
OUT.mkdir(parents=True, exist_ok=True)
DEMO_DATA = ROOT / "docs" / "assets" / "js" / "demos" / "ch12-conv-data.js"
RS = 42

NPZ_URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
NPZ_PATH = Path(tempfile.gettempdir()) / "mlsite_pneumoniamnist.npz"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def conv2d(img, k, stride=1, padding=0):
    if padding:
        img = np.pad(img, padding)
    n = (img.shape[0] - k.shape[0]) // stride + 1
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = np.sum(img[i * stride:i * stride + 3, j * stride:j * stride + 3] * k)
    return out


def max_pool(img, size=2):
    h, w = img.shape[0] // size, img.shape[1] // size
    return img[:h * size, :w * size].reshape(h, size, w, size).max(axis=(1, 3))


def load_pneumonia():
    if not NPZ_PATH.exists():
        print("downloading PneumoniaMNIST (~4 MB) ...")
        urllib.request.urlretrieve(NPZ_URL, NPZ_PATH)
    return np.load(NPZ_PATH)


SOBEL_V = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
SOBEL_H = SOBEL_V.T.copy()
BLUR = np.ones((3, 3)) / 9


# ---------------------------------------------------------------- 1. sliding window
def fig_conv_sliding():
    img = np.array([
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0],
    ], dtype=float)
    k = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)
    out = conv2d(img, k)
    pos = (1, 1)  # top-left corner of the highlighted 3x3 window

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), gridspec_kw={"width_ratios": [6, 3, 4]})

    def grid(ax, arr, title, cmap, vmin, vmax, fmt="{:.0f}"):
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        for (i, j), v in np.ndenumerate(arr):
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=12,
                    color="white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.6 else "black")
        ax.set_xticks(np.arange(-.5, arr.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, arr.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_title(title, fontsize=12)

    grid(axes[0], img, "輸入影像 6×6（0 = 暗、1 = 亮）", "Greys", -0.3, 1.6)
    axes[0].add_patch(Rectangle((pos[1] - .5, pos[0] - .5), 3, 3, fill=False, ec=ORANGE, lw=3.5))
    grid(axes[1], k, "濾鏡 3×3\n（垂直邊緣偵測）", "RdBu_r", -1.6, 1.6)
    grid(axes[2], out, "特徵圖 4×4", "RdBu_r", -3.5, 3.5)
    axes[2].add_patch(Rectangle((pos[1] - .5, pos[0] - .5), 1, 1, fill=False, ec=ORANGE, lw=3.5))

    patch = img[pos[0]:pos[0] + 3, pos[1]:pos[1] + 3]
    terms = " + ".join(f"{p:.0f}×({w:.0f})" if w < 0 else f"{p:.0f}×{w:.0f}"
                       for p, w in zip(patch.ravel(), k.ravel()))
    fig.text(0.5, -0.04, f"橘框 9 格逐一相乘再加總：{terms} = {out[pos]:.0f}",
             ha="center", fontsize=10, color=ORANGE)
    fig.text(0.5, -0.11, "濾鏡往右滑一格、再算一次，直到走遍整張圖：6×6 輸入 → 4×4 特徵圖。"
             "「左暗右亮」的邊界輸出大正值（紅）。", ha="center", fontsize=10, color=GREY)
    fig.canvas.draw()  # fix axes positions before placing the arrows between panels
    for left_ax, right_ax in [(axes[0], axes[1]), (axes[1], axes[2])]:
        a = left_ax.get_position().x1 + 0.008
        b = right_ax.get_position().x0 - 0.008
        y = (left_ax.get_position().y0 + left_ax.get_position().y1) / 2
        fig.add_artist(FancyArrowPatch((a, y), (b, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=18, color=GREY, lw=1.5))
    save(fig, "conv_sliding.png")


# ---------------------------------------------------------------- 2. X-ray through filters
def fig_xray_filters(pneu):
    xray = pneu["train_images"][0].astype(float) / 255.0
    v = conv2d(xray, SOBEL_V)
    h = conv2d(xray, SOBEL_H)
    b = conv2d(xray, BLUR)
    pooled = max_pool(np.maximum(v, 0))
    panels = [
        (xray, "原始胸部 X 光\n28×28", "gray", None),
        (v, "垂直邊緣濾鏡\n26×26", "RdBu_r", np.abs(v).max()),
        (h, "水平邊緣濾鏡\n26×26", "RdBu_r", np.abs(h).max()),
        (b, "模糊濾鏡\n26×26", "gray", None),
        (pooled, "垂直邊緣 → ReLU →\n2×2 最大池化 13×13", "magma", None),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.3))
    for ax, (arr, title, cmap, lim) in zip(axes, panels):
        if lim is None:
            ax.imshow(arr, cmap=cmap)
        else:
            ax.imshow(arr, cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.text(0.5, 0.06, "資料：PneumoniaMNIST（MedMNIST v2，CC BY 4.0）訓練集第 1 張；"
             "紅＝濾鏡反應為正、藍＝為負。濾鏡是人工設定的示範，CNN 會自己學出濾鏡權重。",
             ha="center", fontsize=9, color=GREY)
    save(fig, "xray_filters.png")


# ---------------------------------------------------------------- 3. parameter counts
def fig_params():
    dense_layers = [("第 1 層 Dense(128)", 784 * 128 + 128), ("第 2 層 Dense(64)", 128 * 64 + 64),
                    ("輸出 Dense(1)", 64 + 1)]
    cnn_layers = [("Conv2D(16, 3×3)", 16 * (9 * 1 + 1)), ("Conv2D(32, 3×3)", 32 * (9 * 16 + 1)),
                  ("輸出 Dense(1)", 5 * 5 * 32 + 1)]
    dense_total = sum(v for _, v in dense_layers)
    cnn_total = sum(v for _, v in cnn_layers)
    assert dense_total == 108_801 and cnn_total == 5_601   # matches model.summary() in the notebook

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    colors_d = [TEAL, "#4DB6AC", "#B2DFDB"]
    colors_c = [ORANGE, "#FF8A65", "#FFCCBC"]
    for row, (layers, cols) in enumerate([(dense_layers, colors_d), (cnn_layers, colors_c)]):
        left = 0
        for (name, n), c in zip(layers, cols):
            ax.barh(row, n, left=left, color=c, edgecolor="white")
            left += n
    ax.set_yticks([0, 1], [f"全連接網路（第 9 章）\n共 {dense_total:,} 個參數",
                           f"小型 CNN（本章）\n共 {cnn_total:,} 個參數"])
    ax.invert_yaxis()
    ax.set_xlabel("參數數量")
    ax.set_title("28×28 胸部 X 光：兩個模型的參數量", fontsize=12)
    ax.text(dense_layers[0][1] / 2, 0, f"第 1 層 {dense_layers[0][1]:,}", ha="center", va="center",
            color="white", fontsize=10)
    ax.text(cnn_total + 1500, 1, "卷積層 160 + 4,640\n輸出層 801", va="center", fontsize=9, color=ORANGE)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    sizes = np.array([28, 64, 128, 224])
    dense_first = sizes ** 2 * 128 + 128
    conv_first = np.full_like(sizes, 16 * 10)
    ax.plot(sizes, dense_first, "o-", color=TEAL, lw=2, label="Dense(128) 第一層")
    ax.plot(sizes, conv_first, "s-", color=ORANGE, lw=2, label="Conv2D(16, 3×3) 第一層")
    for s, d in zip(sizes, dense_first):
        ax.annotate(f"{d:,}", (s, d), textcoords="offset points",
                    xytext=(6, -12) if s == 28 else (0, 8), ha="left" if s == 28 else "center",
                    fontsize=8, color=TEAL)
    ax.annotate("永遠 160（與影像大小無關）", (128, 160), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color=ORANGE)
    ax.set_yscale("log")
    ax.set_ylim(50, 3e7)
    ax.set_xticks(sizes, [f"{s}×{s}" for s in sizes])
    ax.set_xlabel("灰階影像大小（像素）")
    ax.set_ylabel("第一層參數數量（對數尺度）")
    ax.set_title("影像越大，全連接層的參數爆炸", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "params_dense_vs_cnn.png")


# ---------------------------------------------------------------- 4. demo data
def write_demo_data(pneu):
    imgs, labels = pneu["test_images"], pneu["test_labels"].ravel()
    normal_idx = int(np.where(labels == 0)[0][0])
    pneu_idx = int(np.where(labels == 1)[0][0])
    data = {
        "source": "PneumoniaMNIST test set (MedMNIST v2, CC BY 4.0; Yang et al. Sci Data 2023)",
        "images": [
            {"label": f"正常（測試集第 {normal_idx + 1} 張）", "pixels": imgs[normal_idx].ravel().tolist()},
            {"label": f"肺炎（測試集第 {pneu_idx + 1} 張）", "pixels": imgs[pneu_idx].ravel().tolist()},
        ],
    }
    DEMO_DATA.write_text(
        "// Auto-generated by scripts/figs_ch12.py -- do not edit by hand.\n"
        "// 28x28 grayscale chest X-rays (0-255) from PneumoniaMNIST, CC BY 4.0.\n"
        f"window.CH12_XRAYS = {json.dumps(data, ensure_ascii=False, separators=(',', ':'))};\n",
        encoding="utf-8")
    print("saved", DEMO_DATA)


if __name__ == "__main__":
    np.random.seed(RS)
    fig_conv_sliding()
    fig_params()
    pneu = load_pneumonia()
    fig_xray_filters(pneu)
    write_demo_data(pneu)
