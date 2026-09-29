"""Figures for chapter 09 (ANN basics).

Run from ml-site/:  .venv/bin/python scripts/figs_ch09.py
Outputs PNGs to docs/assets/img/ch09/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "img" / "ch09"
OUT.mkdir(parents=True, exist_ok=True)
RS = 42


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def arrow(ax, p, q, color=GREY, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))


# ---------------------------------------------------------------- 1. neuron
def fig_neuron():
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    inputs = [("$x_1$ 腫瘤半徑", 4.0), ("$x_2$ 表面紋理", 2.5), ("$x_3$ 凹點數", 1.0)]
    weights = ["$w_1$", "$w_2$", "$w_3$"]
    for (lab, y), w in zip(inputs, weights):
        ax.add_patch(Circle((1.0, y), 0.32, color=TEAL, alpha=0.15, ec=TEAL))
        ax.text(0.55, y, lab, ha="right", va="center", fontsize=10)
        arrow(ax, (1.35, y), (4.05, 2.5 + (y - 2.5) * 0.15), color=TEAL)
        ax.text(2.6, 2.5 + (y - 2.5) * 0.62, w, color=TEAL, fontsize=12, ha="center", va="bottom")
    ax.add_patch(Circle((4.6, 2.5), 0.55, color="white", ec=GREY, lw=1.8))
    ax.text(4.6, 2.5, r"$\Sigma$", ha="center", va="center", fontsize=20)
    ax.text(4.6, 1.55, "加權總和\n$z=w_1x_1+w_2x_2+w_3x_3+b$", ha="center", va="top", fontsize=9)
    ax.text(4.6, 3.35, "偏差 $b$", ha="center", fontsize=10, color=GREY)
    arrow(ax, (5.15, 2.5), (6.25, 2.5))
    ax.add_patch(FancyBboxPatch((6.3, 1.95), 1.5, 1.1, boxstyle="round,pad=0.05", fc="white", ec=ORANGE, lw=1.8))
    zz = np.linspace(-4, 4, 50)
    ax.plot(6.45 + (zz + 4) / 8 * 1.2, 2.1 + 0.8 / (1 + np.exp(-2 * zz)), color=ORANGE, lw=2)
    ax.text(7.05, 1.55, "活化函數\n$a=f(z)$", ha="center", va="top", fontsize=9)
    arrow(ax, (7.85, 2.5), (8.9, 2.5))
    ax.text(9.0, 2.5, "輸出\n惡性機率", va="center", fontsize=10)
    # biology analogy row
    bio = [(1.0, "樹突：接收訊號"), (2.6, "突觸強度 ≈ 權重"), (4.6, "細胞體：空間／時間加總"), (7.05, "軸丘：超過閾值才放電")]
    for x, t in bio:
        ax.text(x, 4.75, t, ha="center", fontsize=9, color=GREY)
    ax.text(5.0, 0.15, "上排灰字是類比，不是等價：生物神經元是全有全無的動作電位，人工神經元輸出的是連續數值",
            ha="center", fontsize=8.5, color=GREY)
    save(fig, "neuron.png")


# ---------------------------------------------------------- 2. activations
def fig_activations():
    z = np.linspace(-5, 5, 400)
    funcs = [
        ("階梯函數（感知器）", (z >= 0).astype(float), GREY),
        ("Sigmoid：壓到 0～1", 1 / (1 + np.exp(-z)), TEAL),
        ("tanh：壓到 −1～1", np.tanh(z), "#5E35B1"),
        ("ReLU：負的歸零、正的照傳", np.maximum(0, z), ORANGE),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.9), sharex=True)
    for ax, (title, y, c) in zip(axes, funcs):
        ax.axhline(0, color="#CFD8DC", lw=0.8)
        ax.axvline(0, color="#CFD8DC", lw=0.8)
        ax.plot(z, y, color=c, lw=2.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("輸入 $z$（加權總和）")
        ax.set_ylim(-1.3, 5.2 if "ReLU" in title else 1.3)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("輸出 $f(z)$")
    fig.tight_layout()
    save(fig, "activations.png")


# ------------------------------------------------------------- 3. XOR boundary
def fig_xor():
    rng = np.random.default_rng(RS)
    centers = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    X = np.vstack([c + rng.normal(0, 0.13, size=(60, 2)) for c in centers])
    y = np.repeat(labels, 60)
    models = [
        ("邏輯迴歸（單一神經元）", LogisticRegression()),
        ("神經網路（1 個隱藏層、8 個神經元）",
         MLPClassifier(hidden_layer_sizes=(8,), activation="relu", max_iter=3000, random_state=RS)),
    ]
    xx, yy = np.meshgrid(np.linspace(-0.6, 1.6, 300), np.linspace(-0.6, 1.6, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (title, m) in zip(axes, models):
        m.fit(X, y)
        p = m.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, p, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
        ax.contour(xx, yy, p, levels=[0.5], colors="k", linewidths=1.5)
        ax.scatter(*X[y == 0].T, s=14, color=TEAL, label="類別 0")
        ax.scatter(*X[y == 1].T, s=14, color=ORANGE, marker="^", label="類別 1")
        ax.set_title(f"{title}\n訓練準確率 {m.score(X, y):.0%}", fontsize=11)
        ax.set_xlabel("特徵 1")
        ax.set_aspect("equal")
    axes[0].set_ylabel("特徵 2")
    axes[0].legend(loc="upper center", fontsize=9, ncol=2)
    fig.tight_layout()
    save(fig, "xor_boundary.png")


# ------------------------------------------------------ 4. learning curves
def epoch_curves(alpha, epochs=300):
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.5, stratify=y, random_state=RS)
    # use only 80 training cases so that a big network can memorise them
    X_tr, y_tr = X_tr[:80], y_tr[:80]
    sc = StandardScaler().fit(X_tr)
    X_tr, X_va = sc.transform(X_tr), sc.transform(X_va)
    m = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=alpha, learning_rate_init=1e-3,
                      random_state=RS)
    tr, va = [], []
    for _ in range(epochs):
        m.partial_fit(X_tr, y_tr, classes=[0, 1])
        tr.append(log_loss(y_tr, m.predict_proba(X_tr), labels=[0, 1]))
        va.append(log_loss(y_va, m.predict_proba(X_va), labels=[0, 1]))
    return np.array(tr), np.array(va)


def fig_learning_curve():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, alpha, title in [(axes[0], 1e-5, "幾乎不正則化（alpha=0.00001）"),
                             (axes[1], 3.0, "加上 L2 正則化（alpha=3）")]:
        tr, va = epoch_curves(alpha)
        ep = np.arange(1, len(tr) + 1)
        ax.plot(ep, tr, color=TEAL, lw=2, label="訓練集損失")
        ax.plot(ep, va, color=ORANGE, lw=2, label="驗證集損失")
        best = int(np.argmin(va))
        ax.axvline(best + 1, color=GREY, ls="--", lw=1)
        ax.text(best + 4, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0.5,
                f"驗證損失最低\n（第 {best + 1} 個 epoch）", fontsize=9, color=GREY, va="top")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch（把訓練資料看完幾遍）")
        ax.grid(alpha=0.25)
        print(f"alpha={alpha}: final train {tr[-1]:.4f}, val {va[-1]:.4f}, best val {va.min():.4f} @ {best + 1}")
    axes[0].set_ylabel("對數損失（log loss，越低越好）")
    axes[0].set_ylim(0, 0.8)
    axes[0].legend()
    fig.suptitle("WDBC：只用 80 位病人訓練一個大網路（256×256）", fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, "learning_curve.png")


if __name__ == "__main__":
    fig_neuron()
    fig_activations()
    fig_xor()
    fig_learning_curve()
