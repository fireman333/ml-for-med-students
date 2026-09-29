"""Generate figures for chapter 05 (naive Bayes).

Run from ml-site/:  .venv/bin/python scripts/figs_ch05.py
Outputs: docs/assets/img/ch05/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "img" / "ch05"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def ppv(prev, sens, spec):
    return sens * prev / (sens * prev + (1 - spec) * (1 - prev))


# ---- Fig 1: PPV vs prevalence --------------------------------------------
def fig_ppv_prevalence():
    prev = np.logspace(-3, np.log10(0.5), 300)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for sens, spec, c in [(0.90, 0.95, TEAL), (0.90, 0.99, ORANGE), (0.99, 0.90, GREY)]:
        ax.plot(prev * 100, ppv(prev, sens, spec) * 100, color=c, lw=2.2,
                label=f"敏感度 {sens:.0%}、特異度 {spec:.0%}")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 50])
    ax.set_xticklabels(["0.1%", "0.3%", "1%", "3%", "10%", "30%", "50%"])
    ax.axvline(1, color="black", ls=":", lw=1)
    y1 = ppv(0.01, 0.90, 0.95) * 100
    ax.scatter([1], [y1], color=TEAL, zorder=5)
    ax.annotate(f"盛行率 1% 時 PPV 只有 {y1:.1f}%", xy=(1, y1), xytext=(1.6, y1 + 22),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)
    ax.set_xlabel("盛行率（檢驗前機率，對數刻度）")
    ax.set_ylabel("陽性預測值 PPV（%）")
    ax.set_ylim(0, 100)
    ax.set_title("同一支檢驗，盛行率越低，陽性結果越不可信")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "ppv_vs_prevalence.png")


# ---- Fig 2: Gaussian likelihoods per class (WDBC) -------------------------
def fig_gaussian_likelihood():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    y = (y == 0).astype(int)  # 1 = malignant
    feats = ["mean radius", "mean concave points"]
    names = ["平均半徑（mean radius）", "平均凹點（mean concave points）"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, f, nm in zip(axes, feats, names):
        for cls, c, lab in [(0, TEAL, "良性"), (1, ORANGE, "惡性")]:
            v = X.loc[y == cls, f]
            ax.hist(v, bins=30, density=True, alpha=0.35, color=c)
            grid = np.linspace(X[f].min(), X[f].max(), 300)
            ax.plot(grid, norm.pdf(grid, v.mean(), v.std()), color=c, lw=2.2,
                    label=f"{lab}：平均 {v.mean():.3g}、標準差 {v.std():.2g}")
        ax.set_xlabel(nm)
        ax.set_ylabel("機率密度")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("GaussianNB 的做法：每個類別、每個特徵各配一條常態曲線（直方圖為實際資料）")
    fig.tight_layout()
    save(fig, "gaussian_likelihood.png")


# ---- Fig 3: correlated features + probability calibration -----------------
def fig_calibration():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    y = (y == 0).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    p = GaussianNB().fit(Xtr, ytr).predict_proba(Xte)[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    ax.scatter(X["mean radius"], X["mean perimeter"], s=8, color=GREY, alpha=0.6)
    r = X["mean radius"].corr(X["mean perimeter"])
    ax.set_xlabel("平均半徑")
    ax.set_ylabel("平均周長")
    ax.set_title(f"特徵高度相關（r = {r:.3f}）\n違反「條件獨立」假設")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(p, bins=20, range=(0, 1), color=TEAL, edgecolor="white")
    extreme = np.mean((p < 0.01) | (p > 0.99))
    ax.set_xlabel("predict_proba 給出的惡性機率")
    ax.set_ylabel("測試集腫瘤數")
    ax.set_title(f"機率幾乎都擠在 0 或 1\n（{extreme:.0%} 落在 <1% 或 >99%）")
    ax.grid(alpha=0.3)

    ax = axes[2]
    wrong = (p > 0.5) != yte.to_numpy()
    pw = np.sort(p[wrong])
    n_conf = int(np.sum(pw < 0.01))
    ax.scatter(pw, np.arange(len(pw)), color=ORANGE, s=50, zorder=3)
    ax.set_xscale("log")
    ax.axvline(0.01, color=GREY, ls="--", lw=1)
    ax.axvline(0.5, color="black", ls=":", lw=1)
    ax.set_ylim(-1, len(pw))
    ax.text(0.011, -0.8, "1%", color=GREY, fontsize=9)
    ax.text(0.45, -0.8, "閾值 50%", fontsize=9, ha="right")
    ax.set_yticks(range(len(pw)))
    ax.set_yticklabels([f"誤判 #{i + 1}" for i in range(len(pw))], fontsize=8)
    ax.set_xlabel("模型給這些「實為惡性」腫瘤的惡性機率（對數刻度）")
    ax.set_title(f"被判錯的 {len(pw)} 例中，\n{n_conf} 例模型給的惡性機率 <1%")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "naive_assumption_calibration.png")


# ---- Fig 4: text classification confusion matrix --------------------------
def fig_text_confusion():
    base = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/"
    try:
        tr = pd.read_csv(base + "medical_tc_train.csv")
        te = pd.read_csv(base + "medical_tc_test.csv")
    except Exception as e:  # network failure: skip this figure, keep the others
        print("skip text figure, download failed:", e)
        return
    model = make_pipeline(CountVectorizer(stop_words="english", min_df=2), MultinomialNB())
    model.fit(tr["medical_abstract"], tr["condition_label"])
    labels = ["腫瘤", "消化系統", "神經系統", "心血管", "一般病理"]
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ConfusionMatrixDisplay.from_estimator(
        model, te["medical_abstract"], te["condition_label"],
        display_labels=labels, cmap="Greens", colorbar=False, ax=ax)
    ax.set_xlabel("模型預測類別")
    ax.set_ylabel("真實類別")
    ax.set_title("MultinomialNB 分類醫學摘要（測試集 2,888 篇）")
    fig.tight_layout()
    save(fig, "text_confusion.png")


if __name__ == "__main__":
    fig_ppv_prevalence()
    fig_gaussian_likelihood()
    fig_calibration()
    fig_text_confusion()
