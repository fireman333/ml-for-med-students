"""Generate figures for Chapter 01 (intro).

Run from ml-site/:  .venv/bin/python scripts/figs_ch01.py
Outputs: docs/assets/img/ch01/*.png
"""
from pathlib import Path
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "img" / "ch01"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def box(ax, x, y, w, h, text, color, fontsize=13, text_color="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=color, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2))


# ---------------------------------------------------------------- fig 1
def fig_traditional_vs_ml():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("傳統程式：人寫規則", fontsize=15, color=GREY, fontweight="bold")
    box(ax, 0.2, 3.6, 3.2, 1.4, "資料\n(病人檢驗值)", GREY)
    box(ax, 0.2, 1.0, 3.2, 1.4, "規則\n(人寫的 if-else)", ORANGE)
    box(ax, 4.6, 2.3, 2.2, 1.4, "程式", "#455A64")
    box(ax, 7.8, 2.3, 2.0, 1.4, "答案\n(診斷)", TEAL)
    arrow(ax, 3.4, 4.3, 4.6, 3.3)
    arrow(ax, 3.4, 1.7, 4.6, 2.7)
    arrow(ax, 6.8, 3.0, 7.8, 3.0)

    ax = axes[1]
    ax.set_title("機器學習：從資料學出規則", fontsize=15, color=TEAL, fontweight="bold")
    box(ax, 0.2, 3.6, 3.2, 1.4, "資料\n(病人檢驗值)", GREY)
    box(ax, 0.2, 1.0, 3.2, 1.4, "答案\n(已知診斷)", TEAL)
    box(ax, 4.6, 2.3, 2.2, 1.4, "學習\n演算法", "#455A64")
    box(ax, 7.8, 2.3, 2.0, 1.4, "規則\n(模型)", ORANGE)
    arrow(ax, 3.4, 4.3, 4.6, 3.3)
    arrow(ax, 3.4, 1.7, 4.6, 2.7)
    arrow(ax, 6.8, 3.0, 7.8, 3.0)

    fig.tight_layout()
    save(fig, "traditional_vs_ml.png")


# ---------------------------------------------------------------- fig 2
def fig_ai_ml_dl():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.add_patch(Ellipse((5, 3.6), 9.6, 6.9, facecolor="#ECEFF1", edgecolor=GREY, lw=2))
    ax.add_patch(Ellipse((5.4, 3.0), 6.8, 4.8, facecolor="#B2DFDB", edgecolor=TEAL, lw=2))
    ax.add_patch(Ellipse((5.9, 2.4), 3.6, 2.4, facecolor="#FFCCBC", edgecolor=ORANGE, lw=2))
    ax.text(5, 6.35, "人工智慧 (AI)", ha="center", fontsize=16, fontweight="bold", color="#37474F")
    ax.text(5, 5.85, "讓電腦做出「看起來聰明」的事，含人寫規則的專家系統", ha="center", fontsize=10, color="#37474F")
    ax.text(5.4, 4.75, "機器學習 (ML)", ha="center", fontsize=15, fontweight="bold", color="#00695C")
    ax.text(5.4, 4.3, "從資料自動學出規則：迴歸、樹、SVM…", ha="center", fontsize=10, color="#00695C")
    ax.text(5.9, 2.85, "深度學習 (DL)", ha="center", fontsize=14, fontweight="bold", color="#BF360C")
    ax.text(5.9, 1.75, "多層神經網路\n影像、語音、文字", ha="center", fontsize=10, color="#BF360C")
    save(fig, "ai_ml_dl.png")


# ---------------------------------------------------------------- fig 3
def fig_supervised_vs_unsupervised():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    cols = ["mean radius", "mean concave points"]
    X = df[cols].to_numpy()
    y = df["target"].to_numpy()  # 0 = malignant, 1 = benign

    km = KMeans(n_clusters=2, random_state=42).fit(StandardScaler().fit_transform(X))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharex=True, sharey=True)
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=14, color=GREY, alpha=0.6)
    ax.set_title("原始資料：只有量測值", fontsize=11)

    ax = axes[1]
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=14, color=TEAL, alpha=0.7, label="良性 (benign)")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=14, color=ORANGE, alpha=0.7, label="惡性 (malignant)")
    ax.set_title("監督式：每點都有病理答案", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2]
    cmap = plt.get_cmap("tab10")
    for k in range(2):
        m = km.labels_ == k
        ax.scatter(X[m, 0], X[m, 1], s=14, color=cmap(k + 2), alpha=0.7, label=f"第 {k + 1} 群")
    ax.set_title("非監督式：沒有答案，自己分群", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

    for ax in axes:
        ax.set_xlabel("細胞核平均半徑 (mean radius)", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("凹點平均 (mean concave points)", fontsize=9)
    fig.tight_layout()
    save(fig, "supervised_vs_unsupervised.png")


# ---------------------------------------------------------------- fig 4
def fig_overfitting_depth():
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    depths = list(range(1, 16))
    tr_acc, te_acc = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_tr, y_tr)
        tr_acc.append(m.score(X_tr, y_tr))
        te_acc.append(m.score(X_te, y_te))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(depths, tr_acc, "o-", color=TEAL, lw=2, label="訓練集準確率")
    ax.plot(depths, te_acc, "s-", color=ORANGE, lw=2, label="測試集準確率")
    best = int(np.argmax(te_acc))
    ax.annotate(f"測試集最高\n深度 {depths[best]}：{te_acc[best]:.3f}",
                xy=(depths[best], te_acc[best]), xytext=(depths[best] + 2.5, te_acc[best] - 0.05),
                fontsize=10, arrowprops=dict(arrowstyle="->", color=GREY))
    ax.axvspan(6.5, 15.5, color=GREY, alpha=0.08)
    ax.text(11, 0.9, "訓練集已背到滿分，\n測試集卻沒有跟著進步", ha="center", fontsize=10, color=GREY)
    ax.set_xlabel("決策樹最大深度 (max_depth)：愈大模型愈複雜", fontsize=11)
    ax.set_ylabel("準確率", fontsize=11)
    ax.set_ylim(0.88, 1.01)
    ax.set_xticks(depths)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_title("WDBC 乳癌資料：模型愈複雜，訓練分數愈好看", fontsize=13)
    fig.tight_layout()
    save(fig, "overfitting_depth.png")
    for d, a, b in zip(depths, tr_acc, te_acc):
        print(f"depth={d:2d} train={a:.3f} test={b:.3f}")


if __name__ == "__main__":
    fig_traditional_vs_ml()
    fig_ai_ml_dl()
    fig_supervised_vs_unsupervised()
    fig_overfitting_depth()
