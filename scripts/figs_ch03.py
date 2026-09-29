"""Figures for chapter 03 (data preprocessing).

Run from ml-site/:  .venv/bin/python scripts/figs_ch03.py
Output: docs/assets/img/ch03/*.png
Needs network once to download UCI CKD (id 336, CC BY 4.0).
"""
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
OUT = Path("docs/assets/img/ch03")
OUT.mkdir(parents=True, exist_ok=True)
CKD_URL = "https://archive.ics.uci.edu/static/public/336/data.csv"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def load_ckd():
    try:
        r = requests.get(CKD_URL, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Cannot download UCI CKD ({e}); figures need the real data.")
    df = pd.read_csv(io.StringIO(r.text))
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].str.strip()
    return df


def fig_missing(ckd):
    """Missing rate per column, split by class -> informative missingness."""
    rate = ckd.drop(columns="class").isna().groupby(ckd["class"]).mean().T
    rate = rate.sort_values("ckd")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y = np.arange(len(rate))
    ax.barh(y + 0.2, rate["ckd"] * 100, height=0.4, color=ORANGE, label="CKD 組（n=250）")
    ax.barh(y - 0.2, rate["notckd"] * 100, height=0.4, color=TEAL, label="非 CKD 組（n=150）")
    ax.set_yticks(y, rate.index)
    ax.set_xlabel("遺漏比例（%）")
    ax.set_title("UCI 慢性腎臟病資料：各欄遺漏比例（兩組差很多）")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "missing_by_class.png")


def fig_unit_mix(ckd):
    """Inject umol/L creatinine for 20% of rows, then show detection and conversion."""
    sc = ckd["sc"].copy()
    site_b = ckd.sample(frac=0.2, random_state=42).index
    mixed = sc.copy()
    mixed.loc[site_b] = mixed.loc[site_b] * 88.4
    is_b = ckd.index.isin(site_b)
    bins = np.logspace(-1, 4, 50)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    ax = axes[0]
    ax.hist(mixed[~is_b].dropna(), bins=bins, color=TEAL, alpha=0.8, label="醫院 A（mg/dL）")
    ax.hist(mixed[is_b].dropna(), bins=bins, color=ORANGE, alpha=0.8, label="醫院 B（µmol/L）")
    ax.set_xscale("log")
    ax.set_title("合併後：兩座分開的山")
    ax.set_xlabel("肌酸酐紀錄值（對數刻度）")
    ax.set_ylabel("病人數")
    ax.legend()
    ax = axes[1]
    fixed = mixed.copy()
    fixed.loc[site_b] = fixed.loc[site_b] / 88.4
    ax.hist(fixed[~is_b].dropna(), bins=bins, color=TEAL, alpha=0.8, label="醫院 A")
    ax.hist(fixed[is_b].dropna(), bins=bins, color=ORANGE, alpha=0.8, label="醫院 B（已換算）")
    ax.set_xscale("log")
    ax.set_title("依來源換算成 mg/dL 之後")
    ax.set_xlabel("肌酸酐（mg/dL，對數刻度）")
    ax.legend()
    fig.suptitle("人工注入的單位不一（教學改造，原始資料沒有這個問題）", y=1.02)
    save(fig, "unit_mix.png")


def fig_scaling(ckd):
    """Raw vs standardized vs min-max on hemoglobin and WBC."""
    d = ckd[["hemo", "wbcc"]].dropna()
    versions = [
        ("原始值", d.to_numpy()),
        ("標準化（z 分數）", StandardScaler().fit_transform(d)),
        ("正規化（Min-Max 0–1）", MinMaxScaler().fit_transform(d)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, arr) in zip(axes, versions):
        ax.scatter(arr[:, 0], arr[:, 1], s=10, color=TEAL, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("血紅素 hemo")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("白血球 wbcc")
    fig.suptitle("同一群病人，三種尺度：形狀不變，只有刻度改變", y=1.03)
    save(fig, "scaling.png")


def fig_leakage():
    """Feature selection outside vs inside CV, on pure noise, 20 repetitions."""
    wrong, right = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(200, 5000))
        y = rng.integers(0, 2, size=200)
        X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
        wrong.append(cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=5).mean())
        pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
        right.append(cross_val_score(pipe, X, y, cv=5).mean())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, 20)
    ax.scatter(np.zeros(20) + jitter, wrong, color=ORANGE, alpha=0.8)
    ax.scatter(np.ones(20) + jitter, right, color=TEAL, alpha=0.8)
    ax.hlines([np.mean(wrong), np.mean(right)], [-0.25, 0.75], [0.25, 1.25], colors="black")
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.text(1.42, 0.5, "猜硬幣 0.5", va="center", color=GREY,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_xticks([0, 1], ["錯：先用全部資料挑特徵\n再交叉驗證", "對：挑特徵放進 Pipeline\n每折各自挑"])
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("交叉驗證準確率")
    ax.set_title("純隨機雜訊上的資料洩漏（重複 20 次）")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "leakage.png")
    print(f"leakage means: wrong={np.mean(wrong):.2f}, right={np.mean(right):.2f}")


if __name__ == "__main__":
    ckd = load_ckd()
    fig_missing(ckd)
    fig_unit_mix(ckd)
    fig_scaling(ckd)
    fig_leakage()
