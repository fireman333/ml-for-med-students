"""Figures for chapter 10 (dimensionality reduction).

Run from ml-site/:  .venv/bin/python scripts/figs_ch10.py
Outputs: docs/assets/img/ch10/{be_stability,scree,pca_2d}.png
"""
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "img" / "ch10"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, facecolor="white")
    plt.close(fig)
    print("saved", OUT / name)


def backward_eliminate(X, y, alpha=0.05):
    cols = list(X.columns)
    while cols:
        model = sm.OLS(y, sm.add_constant(X[cols])).fit()
        pvals = model.pvalues.drop("const")
        worst = pvals.idxmax()
        if pvals[worst] <= alpha:
            break
        cols.remove(worst)
    return cols


# 1. Bootstrap stability of backward elimination (diabetes)
X, y = load_diabetes(return_X_y=True, as_frame=True)
kept_full = set(backward_eliminate(X, y))
rng = np.random.default_rng(42)
n_boot = 200
freq = Counter()
for _ in range(n_boot):
    idx = rng.integers(0, len(X), len(X))
    freq.update(backward_eliminate(X.iloc[idx].reset_index(drop=True),
                                   y.iloc[idx].reset_index(drop=True)))
inc = pd.Series({c: freq[c] / n_boot for c in X.columns}).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 4))
colors = [TEAL if c in kept_full else GREY for c in inc.index]
bars = ax.bar(inc.index, inc.values, color=colors)
for b, v in zip(bars, inc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("被留下的比例")
ax.set_title("反向淘汰法在 200 次自助重抽中，各變數被留下的比例")
ax.bar([0], [0], color=TEAL, label="原始資料中被留下")
ax.bar([0], [0], color=GREY, label="原始資料中被刪除")
ax.legend(loc="upper right", fontsize=9)
save(fig, "be_stability.png")

# 2. Scree plot (WDBC, standardized)
data = load_breast_cancer()
Xw, yw = data.data, data.target
Xw_std = StandardScaler().fit_transform(Xw)
ratio = PCA().fit(Xw_std).explained_variance_ratio_
cum = np.cumsum(ratio)
k = np.arange(1, 16)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(k, ratio[:15], color=TEAL, label="單一主成分解釋的變異比例")
ax.plot(k, cum[:15], "o-", color=ORANGE, label="累積解釋比例")
ax.axhline(0.95, color=GREY, ls="--", lw=1)
ax.text(0.6, 0.965, "95%", va="bottom", color=GREY, fontsize=9)
n95 = int(np.argmax(cum >= 0.95) + 1)
ax.annotate(f"第 {n95} 個主成分累積達 95%", xy=(n95, cum[n95 - 1]), xytext=(n95 + 1, 0.72),
            arrowprops=dict(arrowstyle="->", color=GREY), fontsize=9)
ax.set_xticks(k)
ax.set_xlabel("主成分編號")
ax.set_ylabel("解釋變異比例")
ax.set_ylim(0, 1.05)
ax.set_title("陡坡圖（scree plot）：WDBC 30 個特徵標準化後做 PCA")
ax.legend(loc="center right", fontsize=9)
save(fig, "scree.png")

# 3. PCA 2D scatter: standardized vs. raw
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, (Xin, title) in zip(axes, [(Xw_std, "先標準化再做 PCA"), (Xw, "沒有標準化就做 PCA")]):
    p = PCA(n_components=2).fit(Xin)
    Z = p.transform(Xin)
    for label, color, name in [(0, ORANGE, "惡性"), (1, TEAL, "良性")]:
        m = yw == label
        ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.55, color=color, label=name)
    r = p.explained_variance_ratio_
    ax.set_xlabel(f"PC1（{r[0]:.0%}）")
    ax.set_ylabel(f"PC2（{r[1]:.0%}）")
    ax.set_title(title)
    ax.legend(fontsize=9)
axes[1].text(0.03, 0.97, "PC1 幾乎只反映 area（面積）\n因為它的數值範圍最大",
             transform=axes[1].transAxes, ha="left", va="top", fontsize=9, color=GREY)
save(fig, "pca_2d.png")
