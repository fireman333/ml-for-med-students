"""Generate static figures for Chapter 06 (SVM).

Run from ml-site/:  .venv/bin/python scripts/figs_ch06.py
Outputs: docs/assets/img/ch06/*.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, make_blobs, make_circles, make_moons
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "img" / "ch06"
OUT.mkdir(parents=True, exist_ok=True)
RS = 42


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def scatter(ax, X, y, s=36):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=TEAL, s=s, edgecolor="white", linewidth=0.6, label="良性（示意）")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=ORANGE, s=s, edgecolor="white", linewidth=0.6, marker="s", label="惡性（示意）")


def draw_boundary(ax, clf, X, levels=(-1, 0, 1), fill=False, pad=0.5, n=300):
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    if fill:
        ax.contourf(xx, yy, zz > 0, levels=[-0.5, 0.5, 1.5], colors=[TEAL, ORANGE], alpha=0.12)
    styles = {-1: "--", 0: "-", 1: "--"}
    for lv in levels:
        ax.contour(xx, yy, zz, levels=[lv], colors=[GREY if lv else "black"], linestyles=styles[lv], linewidths=1.2 if lv else 1.8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])


# 1. Many separating lines vs. maximum margin -------------------------------
def fig_many_lines():
    X, y = make_blobs(n_samples=40, centers=[[-1.6, -1.2], [1.6, 1.2]], cluster_std=0.65, random_state=RS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    ax = axes[0]
    scatter(ax, X, y)
    xs = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 50)
    for slope, icpt, col in [(-0.25, 0.05, "#8E24AA"), (-2.8, 0.2, "#3949AB"), (-0.9, -0.3, "#6D4C41")]:
        ax.plot(xs, slope * xs + icpt, color=col, lw=1.6)
    ax.scatter([0.35], [0.9], marker="*", s=260, c="gold", edgecolor="black", zorder=5, label="新病人")
    ax.set_title("三條線都能把訓練資料完全分開\n但對新病人（星號）的判斷不一樣")
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.legend(loc="lower right", fontsize=8)
    clean(ax)

    ax = axes[1]
    clf = SVC(kernel="linear", C=1000).fit(X, y)
    draw_boundary(ax, clf, X)
    scatter(ax, X, y)
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=190, facecolors="none", edgecolors="black", linewidths=1.4, label="支援向量")
    ax.scatter([0.35], [0.9], marker="*", s=260, c="gold", edgecolor="black", zorder=5)
    ax.set_title("SVM 選「走道最寬」的那一條\n只有圈起來的點（支援向量）決定位置")
    ax.legend(loc="lower right", fontsize=8)
    clean(ax)
    fig.tight_layout()
    save(fig, "margin_intuition.png")


# 2. Soft margin: effect of C ----------------------------------------------
def fig_soft_margin():
    X, y = make_blobs(n_samples=80, centers=[[-1.0, -0.8], [1.0, 0.8]], cluster_std=1.05, random_state=RS)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for ax, C in zip(axes, [0.01, 1, 100]):
        clf = SVC(kernel="linear", C=C).fit(X, y)
        draw_boundary(ax, clf, X, fill=True)
        scatter(ax, X, y, s=28)
        sv = clf.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=110, facecolors="none", edgecolors="black", linewidths=0.9)
        acc = clf.score(X, y)
        ax.set_title(f"C = {C}\n支援向量 {len(sv)} 個・訓練準確率 {acc:.2f}")
        clean(ax)
    axes[0].set_xlabel("C 小：走道寬、容忍越界（較平滑）")
    axes[2].set_xlabel("C 大：走道窄、盡量不犯錯（較敏感）")
    fig.suptitle("軟間隔：C 控制「對越界的處罰有多重」（實線＝決策邊界，虛線＝走道邊緣，圈＝支援向量）", fontsize=11)
    fig.tight_layout()
    save(fig, "soft_margin_C.png")


# 3. Kernel trick: lift 1D data to 2D ---------------------------------------
def fig_kernel_lift():
    rng = np.random.default_rng(RS)
    x_in = rng.uniform(-1.2, 1.2, 14)
    x_out = np.concatenate([rng.uniform(-3, -1.8, 7), rng.uniform(1.8, 3, 7)])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.scatter(x_in, np.zeros_like(x_in), c=TEAL, s=50, edgecolor="white", label="正常範圍")
    ax.scatter(x_out, np.zeros_like(x_out), c=ORANGE, s=50, marker="s", edgecolor="white", label="過高或過低")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_yticks([])
    ax.set_xlabel("某項檢驗值（標準化後）")
    ax.set_title("一條軸上：兩端都異常，中間正常\n找不到「一個切點」能分開")
    ax.legend(loc="upper center", fontsize=8)
    ax.set_ylim(-1, 1)

    ax = axes[1]
    ax.scatter(x_in, x_in**2, c=TEAL, s=50, edgecolor="white")
    ax.scatter(x_out, x_out**2, c=ORANGE, s=50, marker="s", edgecolor="white")
    ax.axhline(2.4, color="black", lw=1.6, label="一條水平線就分開了")
    ax.set_xlabel("原本的檢驗值 x")
    ax.set_ylabel("新增的特徵 x²（偏離程度）")
    ax.set_title("加一個維度 x² 把資料「抬」起來\n在新空間裡變成直線可分")
    ax.legend(loc="upper center", fontsize=8)
    fig.tight_layout()
    save(fig, "kernel_lift.png")


# 4. Kernels compared on circles -------------------------------------------
def fig_kernels():
    X, y = make_circles(n_samples=160, factor=0.4, noise=0.12, random_state=RS)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for ax, (name, clf) in zip(axes, [
        ("線性（linear）", SVC(kernel="linear", C=1)),
        ("多項式（poly, degree=2）", SVC(kernel="poly", degree=2, C=1, coef0=1)),
        ("RBF（徑向基底）", SVC(kernel="rbf", C=1, gamma="scale")),
    ]):
        clf.fit(X, y)
        draw_boundary(ax, clf, X, levels=(0,), fill=True, pad=0.3)
        scatter(ax, X, y, s=24)
        ax.set_title(f"{name}\n訓練準確率 {clf.score(X, y):.2f}")
        clean(ax)
    fig.suptitle("同一份「內圈／外圈」資料，換核函數的效果", fontsize=11)
    fig.tight_layout()
    save(fig, "kernels_compare.png")


# 5. RBF gamma: under- vs over-fitting --------------------------------------
def fig_gamma():
    X, y = make_moons(n_samples=150, noise=0.3, random_state=RS)
    Xt, yt = make_moons(n_samples=400, noise=0.3, random_state=RS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for ax, g in zip(axes, [0.1, 1, 50]):
        clf = SVC(kernel="rbf", C=1, gamma=g).fit(X, y)
        draw_boundary(ax, clf, X, levels=(0,), fill=True, pad=0.3)
        scatter(ax, X, y, s=24)
        ax.set_title(f"gamma = {g}\n訓練 {clf.score(X, y):.2f}／新資料 {clf.score(Xt, yt):.2f}")
        clean(ax)
    axes[0].set_xlabel("太小：邊界太僵硬（配適不足）")
    axes[2].set_xlabel("太大：繞著每個點畫圈（過度配適）")
    fig.suptitle("RBF 核的 gamma：每個訓練點的「影響半徑」（gamma 愈大，半徑愈小）", fontsize=11)
    fig.tight_layout()
    save(fig, "rbf_gamma.png")


# 6. Scaling matters on WDBC -----------------------------------------------
def fig_scaling():
    X, y = load_breast_cancer(return_X_y=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    raw = cross_val_score(SVC(), X, y, cv=cv)
    sc = cross_val_score(make_pipeline(StandardScaler(), SVC()), X, y, cv=cv)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    names = load_breast_cancer().feature_names
    idx = [list(names).index(n) for n in ["mean smoothness", "mean radius", "mean area", "worst area"]]
    rng_ = X[:, idx].max(axis=0) - X[:, idx].min(axis=0)
    ax.barh([names[i] for i in idx], rng_, color=GREY)
    ax.set_xscale("log")
    ax.set_xlabel("該特徵的全距（最大－最小，對數刻度）")
    ax.set_title("WDBC 各特徵的尺度差了上萬倍")
    ax = axes[1]
    bars = ax.bar(["未標準化", "標準化後"], [raw.mean(), sc.mean()], color=[GREY, TEAL], yerr=[raw.std(), sc.std()], capsize=6)
    for b, v in zip(bars, [raw.mean(), sc.mean()]):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.06, f"{v:.3f}", ha="center", color="white", fontsize=11, weight="bold")
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("5 折交叉驗證準確率")
    ax.set_title("同一個 RBF SVM，只差有沒有標準化")
    fig.tight_layout()
    save(fig, "scaling_wdbc.png")
    print(f"  WDBC CV acc raw={raw.mean():.3f} scaled={sc.mean():.3f}")


if __name__ == "__main__":
    fig_many_lines()
    fig_soft_margin()
    fig_kernel_lift()
    fig_kernels()
    fig_gamma()
    fig_scaling()
