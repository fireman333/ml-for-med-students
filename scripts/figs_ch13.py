"""Figures for chapter 13 (training tricks, transfer learning, Grad-CAM).

Run from ml-site/:  .venv/bin/python scripts/figs_ch13.py
Outputs PNGs to docs/assets/img/ch13/.

Figures 1-3 need only numpy/scipy/matplotlib (BreastMNIST npz is downloaded once
to notebooks/_data_cache/). Figures 4-5 are built from the *executed* notebook
docs/notebooks/ch13_transfer-learning.ipynb (numbers and Grad-CAM images are read
from its saved outputs, so the site never hand-types a result). Re-execute the
notebook with the ml-colab kernel first if you change it.
"""
import base64
import io
import json
import re
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import ndimage

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "img" / "ch13"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "notebooks" / "_data_cache"
NB = ROOT / "docs" / "notebooks" / "ch13_transfer-learning.ipynb"
NPZ_URL = "https://zenodo.org/records/10519652/files/breastmnist_128.npz?download=1"
RS = 42


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def load_breastmnist():
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "breastmnist_128.npz"
    if not path.exists():
        print("downloading BreastMNIST 128 (~11 MB) ...")
        urllib.request.urlretrieve(NPZ_URL, path)
    return np.load(path)


# ---------------------------------------------------------------- 1. augmentation collage
def fig_augment():
    d = load_breastmnist()
    labels = d["train_labels"].ravel()
    img = d["train_images"][np.where(labels == 0)[0][2]].astype(float)  # a malignant example

    def zoom(im, f):
        z = ndimage.zoom(im, f, order=1)
        s = (z.shape[0] - im.shape[0]) // 2
        return z[s:s + im.shape[0], s:s + im.shape[1]]

    panels = [
        ("原圖", img, True),
        ("左右翻轉", img[:, ::-1], True),
        ("旋轉 15°", ndimage.rotate(img, 15, reshape=False, mode="reflect"), True),
        ("放大 1.2 倍", zoom(img, 1.2), True),
        ("亮度＋20%", np.clip(img * 1.2, 0, 255), True),
        ("上下翻轉（不合理）", img[::-1, :], False),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(10.5, 2.6))
    for ax, (title, im, ok) in zip(axes, panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11, color=TEAL if ok else ORANGE)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(TEAL if ok else ORANGE); sp.set_linewidth(2.5)
    fig.suptitle("同一張乳房超音波（BreastMNIST，惡性）的資料擴增：上下翻轉會把深層組織翻到探頭端", fontsize=12)
    plt.tight_layout()
    save(fig, "augment.png")


# ---------------------------------------------------------------- 2. transfer-learning schematic
def fig_transfer():
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.6))
    rows = [
        ("從零訓練", ["train"] * 6, "所有層從隨機權重開始學，需要大量資料"),
        ("特徵擷取", ["frozen"] * 6, "骨幹全部凍結，只訓練新的分類頭"),
        ("微調", ["frozen"] * 4 + ["train"] * 2, "凍結前段，解凍最後幾層，用很小的學習率一起調"),
    ]
    for ax, (name, kinds, note) in zip(axes, rows):
        ax.set_xlim(0, 10); ax.set_ylim(0, 1.3); ax.axis("off")
        ax.text(0.0, 0.62, name, fontsize=13, va="center")
        for i, k in enumerate(kinds):
            x = 1.5 + i * 0.95
            color = "#CFD8DC" if k == "frozen" else "#B2DFDB"
            edge = GREY if k == "frozen" else TEAL
            ax.add_patch(FancyBboxPatch((x, 0.3), 0.75, 0.65, boxstyle="round,pad=0.02", fc=color, ec=edge, lw=1.5))
            ax.text(x + 0.375, 0.62, "凍結" if k == "frozen" else "訓練", ha="center", va="center", fontsize=9,
                    color=GREY if k == "frozen" else TEAL)
            if i < len(kinds) - 1:
                ax.add_patch(FancyArrowPatch((x + 0.75, 0.62), (x + 0.95, 0.62), arrowstyle="-|>",
                                             mutation_scale=8, color=GREY))
        hx = 1.5 + 6 * 0.95 + 0.1
        ax.add_patch(FancyArrowPatch((hx - 0.3, 0.62), (hx, 0.62), arrowstyle="-|>", mutation_scale=8, color=GREY))
        ax.add_patch(FancyBboxPatch((hx, 0.3), 1.1, 0.65, boxstyle="round,pad=0.02", fc="#FFCCBC", ec=ORANGE, lw=1.5))
        ax.text(hx + 0.55, 0.62, "新分類頭", ha="center", va="center", fontsize=9, color=ORANGE)
        ax.text(1.5, 0.08, note, fontsize=10, color="#333333")
    axes[0].text(1.5 + 2.6, 1.15, "骨幹（卷積層，可來自 ImageNet 預訓練）", ha="center", fontsize=10, color=GREY)
    plt.tight_layout()
    save(fig, "transfer_modes.png")


# ---------------------------------------------------------------- 3. optimizers on an elongated bowl
def fig_optimizers():
    def grad(w):
        return np.array([0.1 * w[0], 10 * w[1]])

    def loss(w):
        return 0.5 * (0.1 * w[0] ** 2 + 10 * w[1] ** 2)

    def run(method, lr, steps=60):
        w, v, m, s = np.array([-8.0, 1.5]), np.zeros(2), np.zeros(2), np.zeros(2)
        path = [w.copy()]
        for t in range(1, steps + 1):
            g = grad(w)
            if method == "sgd":
                w = w - lr * g
            elif method == "momentum":
                v = 0.9 * v + g
                w = w - lr * v
            else:
                m = 0.9 * m + 0.1 * g
                s = 0.999 * s + 0.001 * g ** 2
                w = w - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(s / (1 - 0.999 ** t)) + 1e-8)
            path.append(w.copy())
        return np.array(path)

    runs = [("SGD，學習率 0.01（太小）", "sgd", 0.01, GREY),
            ("SGD，學習率 0.19（陡峭方向來回彈）", "sgd", 0.19, ORANGE),
            ("SGD＋動量，學習率 0.01", "momentum", 0.01, "#1E88E5"),
            ("Adam，學習率 0.3", "adam", 0.3, TEAL)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
    xx, yy = np.meshgrid(np.linspace(-9, 2, 200), np.linspace(-2, 2, 200))
    ax1.contour(xx, yy, 0.5 * (0.1 * xx ** 2 + 10 * yy ** 2), levels=np.logspace(-2, 1.5, 12), colors="#B0BEC5",
                linewidths=0.8)
    for name, mth, lr, c in runs:
        p = run(mth, lr)
        ax1.plot(p[:, 0], np.clip(p[:, 1], -2, 2), "-o", ms=2.5, lw=1.2, color=c, label=name)
        ax2.semilogy([loss(w) for w in p], color=c, lw=2)
    ax1.plot(0, 0, "*", color="black", ms=12)
    ax1.set_xlabel("參數 1（平坦方向）"); ax1.set_ylabel("參數 2（陡峭方向）")
    ax1.set_title("狹長山谷上的四種走法（星號為最低點）")
    ax1.legend(fontsize=8.5, loc="lower right")
    ax2.set_xlabel("更新次數"); ax2.set_ylabel("損失（對數刻度）")
    ax2.set_title("損失隨更新次數的變化")
    plt.tight_layout()
    save(fig, "optimizers.png")


# ---------------------------------------------------------------- notebook helpers
def nb_cells():
    if not NB.exists():
        raise SystemExit(f"{NB} not found; build and execute the notebook first.")
    return json.loads(NB.read_text(encoding="utf-8"))["cells"]


def nb_stream_text():
    return "\n".join("".join(o.get("text", "")) for c in nb_cells() if c["cell_type"] == "code"
                     for o in c.get("outputs", []) if o.get("output_type") == "stream")


def nb_images(marker):
    """PNG outputs of the code cell whose source contains `marker`."""
    for c in nb_cells():
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return [plt.imread(io.BytesIO(base64.b64decode(o["data"]["image/png"])))
                    for o in c.get("outputs", []) if "data" in o and "image/png" in o["data"]]
    return []


# ---------------------------------------------------------------- 4. results with CI (read from notebook)
def fig_results():
    text = nb_stream_text()
    pat = re.compile(r"^(A|B|C1|C2): .*?AUC (\d\.\d+) \((\d\.\d+)-(\d\.\d+)\)\s+sens (\d\.\d+)\s+spec (\d\.\d+)", re.M)
    rows = {m.group(1): tuple(float(m.group(i)) for i in range(2, 7)) for m in pat.finditer(text)}
    assert set(rows) == {"A", "B", "C1", "C2"}, f"could not parse all results from notebook: {rows}"
    names = {"A": "A 從零訓練小 CNN", "B": "B 小 CNN＋擴增＋class_weight",
             "C1": "C1 凍結 MobileNetV2＋邏輯迴歸", "C2": "C2 微調 MobileNetV2 最後一個區塊"}
    keys = ["A", "B", "C1", "C2"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, k in enumerate(keys):
        auc, lo, hi, sens, spec = rows[k]
        c = GREY if k in ("A", "B") else TEAL
        ax.errorbar(auc, i, xerr=[[auc - lo], [hi - auc]], fmt="o", color=c, capsize=5, ms=7, lw=2)
        ax.text(hi + 0.01, i, f"AUC {auc:.2f}（{lo:.2f}–{hi:.2f}）  敏感度 {sens:.2f}／特異度 {spec:.2f}",
                va="center", fontsize=9)
    ax.set_yticks(range(len(keys)), [names[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0.7, 1.25)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("測試集 AUC（95% bootstrap 信賴區間；n = 156，惡性 42）")
    ax.set_title("BreastMNIST 128：四種做法的測試集表現（信賴區間大幅重疊）")
    plt.tight_layout()
    save(fig, "results.png")


# ---------------------------------------------------------------- 5. Grad-CAM (images from notebook)
def fig_gradcam():
    trained = nb_images('show_cams((lower, upper, head), picks')
    rand = nb_images('show_cams(parts_r, picks')
    assert trained and rand, "Grad-CAM images not found in the executed notebook"
    m = re.search(r"correlation between trained and random-weight heatmaps: \[([^\]]+)\]", nb_stream_text())
    corr = " ".join(m.group(1).split()) if m else "?"
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4))
    for ax, im, title in [(axes[0], trained[0], "上：微調後的 MobileNetV2（四張測試影像，最右一張是模型判錯的病例）"),
                          (axes[1], rand[0], f"下：同架構、隨機權重（未訓練）——與上排熱圖的相關係數：{corr}")]:
        ax.imshow(im); ax.axis("off"); ax.set_title(title, fontsize=11, loc="left")
    plt.tight_layout()
    save(fig, "gradcam.png")


if __name__ == "__main__":
    np.random.seed(RS)
    fig_augment()
    fig_transfer()
    fig_optimizers()
    fig_results()
    fig_gradcam()
