"""Generate figures and demo data for Chapter 11 (model selection).

Run from ml-site/:  .venv/bin/python scripts/figs_ch11.py
Outputs:
  docs/assets/img/ch11/*.png
  docs/assets/js/demos/ch11-roc-data.js   (test-set scores for the ROC demo)
"""
import json
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split, validation_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False
# numpy 2.1 + macOS Accelerate emits spurious matmul RuntimeWarnings; results are correct
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs/assets/img/ch11"
DEMO = ROOT / "docs/assets/js/demos"
IMG.mkdir(parents=True, exist_ok=True)
DEMO.mkdir(parents=True, exist_ok=True)
SEED = 42
CDC_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
CACHE = Path(tempfile.gettempdir()) / "ml_site_cdc_diabetes_891.csv"


def save(fig, name):
    fig.savefig(IMG / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load_cdc(n=20000):
    """Return a stratified 20k subsample of CDC Diabetes Health Indicators (UCI 891, CC0)."""
    if CACHE.exists():
        df = pd.read_csv(CACHE)
    else:
        t0 = time.time()
        df = pd.read_csv(CDC_URL)
        df.to_csv(CACHE, index=False)
        print(f"downloaded CDC data in {time.time() - t0:.0f}s")
    df = df.drop(columns=["ID"], errors="ignore")
    sub, _ = train_test_split(df, train_size=n, stratify=df["Diabetes_binary"], random_state=SEED)
    return sub.drop(columns="Diabetes_binary"), sub["Diabetes_binary"]


def models():
    return {
        "邏輯迴歸": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
        "決策樹": DecisionTreeClassifier(random_state=SEED),
        "隨機森林": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "神經網路 (MLP)": make_pipeline(StandardScaler(),
                                    MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=SEED)),
        "HistGB": HistGradientBoostingClassifier(random_state=SEED),
    }


# ---------- Fig 1: CV comparison on WDBC ----------
def fig_cv_compare():
    X, y = load_breast_cancer(return_X_y=True)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    res = {name: cross_val_score(m, X, y, cv=cv, scoring="roc_auc") for name, m in models().items()}
    order = sorted(res, key=lambda k: np.mean(res[k]))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, name in enumerate(order):
        s = res[name]
        ax.scatter(s, np.full_like(s, i) + np.random.default_rng(i).uniform(-.12, .12, len(s)),
                   color=GREY, alpha=.5, s=18)
        ax.errorbar(s.mean(), i, xerr=s.std(), fmt="o", color=TEAL, ms=8, capsize=4, lw=2)
        ax.text(1.003, i, f"{s.mean():.3f} ± {s.std():.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(right=1.0)
    ax.set_xlabel("AUC（10 折分層交叉驗證；灰點＝每一折，綠點與誤差線＝平均 ± 標準差）")
    ax.set_title("WDBC：7 種模型的交叉驗證 AUC")
    ax.grid(axis="x", alpha=.3)
    save(fig, "cv_model_compare.png")
    print({k: (round(v.mean(), 4), round(v.std(), 4)) for k, v in res.items()})


# ---------- Figs 2-4 + demo data: CDC diabetes ----------
def fig_cdc():
    X, y = load_cdc()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    print("CDC subsample", X.shape, "prevalence", round(y.mean(), 4))

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    lr_bal = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X_tr, y_tr)
    hgb = HistGradientBoostingClassifier(random_state=SEED).fit(X_tr, y_tr)
    p_lr = lr.predict_proba(X_te)[:, 1]
    p_bal = lr_bal.predict_proba(X_te)[:, 1]
    p_hgb = hgb.predict_proba(X_te)[:, 1]

    # Fig 2: ROC + PR
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
    for p, name, c in [(p_lr, "邏輯迴歸", TEAL), (p_hgb, "HistGB", ORANGE)]:
        fpr, tpr, _ = roc_curve(y_te, p)
        a1.plot(fpr, tpr, color=c, lw=2, label=f"{name}  AUC = {roc_auc_score(y_te, p):.3f}")
        prec, rec, _ = precision_recall_curve(y_te, p)
        a2.plot(rec, prec, color=c, lw=2, label=f"{name}  AP = {average_precision_score(y_te, p):.3f}")
    a1.plot([0, 1], [0, 1], "--", color=GREY, label="亂猜（AUC 0.5）")
    a1.set_xlabel("1 − 特異度（偽陽性率）")
    a1.set_ylabel("敏感度（真陽性率）")
    a1.set_title("ROC 曲線")
    a2.axhline(y_te.mean(), ls="--", color=GREY, label=f"盛行率 {y_te.mean():.1%}（亂猜基準）")
    a2.set_xlabel("敏感度（召回率）")
    a2.set_ylabel("陽性預測值（精確率）")
    a2.set_title("PR 曲線")
    for a in (a1, a2):
        a.legend(fontsize=8.5, loc="lower right" if a is a1 else "upper right")
        a.grid(alpha=.3)
    fig.suptitle("CDC 糖尿病指標（抽樣 2 萬人，測試集 5,000 人）", y=1.02)
    save(fig, "roc_pr_cdc.png")

    # Fig 3: metrics vs threshold
    ts = np.linspace(0.02, 0.8, 79)
    yv = y_te.to_numpy()
    sens, spec, ppv = [], [], []
    for t in ts:
        pred = p_lr >= t
        tp = np.sum(pred & (yv == 1)); fp = np.sum(pred & (yv == 0))
        fn = np.sum(~pred & (yv == 1)); tn = np.sum(~pred & (yv == 0))
        sens.append(tp / (tp + fn)); spec.append(tn / (tn + fp))
        ppv.append(tp / (tp + fp) if tp + fp else np.nan)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ts, sens, color=TEAL, lw=2, label="敏感度")
    ax.plot(ts, spec, color=ORANGE, lw=2, label="特異度")
    ax.plot(ts, ppv, color=GREY, lw=2, ls="--", label="陽性預測值 PPV")
    ax.axvline(0.5, color="k", lw=1, ls=":")
    ax.text(0.505, 0.05, "sklearn 預設\n閾值 0.5", fontsize=9)
    ax.axvline(y_tr.mean(), color="k", lw=1, ls=":")
    ax.text(y_tr.mean() + 0.005, 0.05, f"閾值＝盛行率\n{y_tr.mean():.2f}", fontsize=9)
    ax.set_xlabel("分類閾值（預測機率 ≥ 閾值 就判為陽性）")
    ax.set_ylabel("比例")
    ax.set_title("同一個邏輯迴歸模型，換閾值就換一組敏感度／特異度")
    ax.legend(); ax.grid(alpha=.3)
    save(fig, "threshold_tradeoff.png")

    # Fig 4: calibration, balanced vs not
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="完美校準")
    for p, name, c in [(p_lr, "邏輯迴歸（未加權）", TEAL), (p_bal, "邏輯迴歸 class_weight='balanced'", ORANGE)]:
        fp_, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax.plot(mp, fp_, "o-", color=c, lw=2, label=f"{name}\nBrier = {brier_score_loss(y_te, p):.3f}")
    ax.set_xlabel("模型預測機率（每組平均）")
    ax.set_ylabel("實際陽性比例")
    ax.set_title("校準曲線：加權後排序能力相近，機率卻被整體高估")
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=.3)
    save(fig, "calibration.png")
    print("AUC lr / balanced / hgb:", round(roc_auc_score(y_te, p_lr), 4),
          round(roc_auc_score(y_te, p_bal), 4), round(roc_auc_score(y_te, p_hgb), 4))

    # Fig 5: validation curve (bias-variance) for tree depth
    depths = np.arange(1, 16)
    tr_s, va_s = validation_curve(DecisionTreeClassifier(random_state=SEED), X_tr, y_tr,
                                  param_name="max_depth", param_range=depths,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for s, name, c in [(tr_s, "訓練集", GREY), (va_s, "交叉驗證", TEAL)]:
        ax.plot(depths, s.mean(1), "o-", color=c, lw=2, label=name)
        ax.fill_between(depths, s.mean(1) - s.std(1), s.mean(1) + s.std(1), color=c, alpha=.2)
    best = depths[va_s.mean(1).argmax()]
    ax.axvline(best, color=ORANGE, ls=":", lw=1.5)
    ax.text(best + .2, ax.get_ylim()[0] + .02, f"交叉驗證最佳深度 ≈ {best}", color=ORANGE)
    ax.text(1.2, 0.93, "偏差大\n（配適不足）", fontsize=9)
    ax.text(11.3, 0.80, "變異大\n（過度配適）", fontsize=9)
    ax.set_xlabel("決策樹最大深度 max_depth")
    ax.set_ylabel("AUC")
    ax.set_title("驗證曲線：模型愈複雜，訓練分數一路升，驗證分數先升後降")
    ax.legend(loc="lower left"); ax.grid(alpha=.3)
    save(fig, "validation_curve_depth.png")

    # Demo data: rounded scores so JS and Python compute identical metrics
    scores = np.round(p_lr, 3)
    data = {"score": scores.tolist(), "label": yv.astype(int).tolist()}
    js = ("// Auto-generated by scripts/figs_ch11.py. Logistic regression test-set scores\n"
          "// on a 20k subsample of CDC Diabetes Health Indicators (UCI 891, CC0).\n"
          "window.CH11_ROC_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    (DEMO / "ch11-roc-data.js").write_text(js, encoding="utf-8")
    print("wrote ch11-roc-data.js", len(js) // 1024, "KB")
    for t in (0.14, 0.3, 0.5):
        pred = scores >= t
        tp = np.sum(pred & (yv == 1)); fn = np.sum(~pred & (yv == 1))
        tn = np.sum(~pred & (yv == 0)); fp = np.sum(pred & (yv == 0))
        print(f"check t={t}: TP={tp} FP={fp} FN={fn} TN={tn} sens={tp/(tp+fn):.3f} spec={tn/(tn+fp):.3f}")


if __name__ == "__main__":
    fig_cv_compare()
    fig_cdc()
