"""Generate figures for chapter 04 (regression).

Run from ml-site/:  .venv/bin/python scripts/figs_ch04.py
Outputs PNGs to docs/assets/img/ch04/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "img" / "ch04"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


# ---------- Figure 1: simple linear regression with residuals ----------
df = load_diabetes(as_frame=True, scaled=False).frame
X, y = df[["bmi"]], df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
lin = LinearRegression().fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(X_train["bmi"], y_train, s=14, color=GREY, alpha=0.55, label="訓練集病人")
xs = pd.DataFrame({"bmi": np.linspace(X["bmi"].min(), X["bmi"].max(), 100)})
ax.plot(xs["bmi"], lin.predict(xs), color=TEAL, lw=2.5,
        label=f"最小平方直線：y = {lin.coef_[0]:.1f} × BMI {lin.intercept_:+.0f}")
rng = np.random.default_rng(42)
idx = rng.choice(len(X_train), 12, replace=False)
for i in idx:
    xi, yi = X_train["bmi"].iloc[i], y_train.iloc[i]
    ax.plot([xi, xi], [yi, lin.predict(pd.DataFrame({"bmi": [xi]}))[0]], color=ORANGE, lw=1)
ax.plot([], [], color=ORANGE, lw=1, label="殘差（實際值 − 預測值）")
ax.set_xlabel("BMI（kg/m²）")
ax.set_ylabel("一年後疾病進展指標")
ax.set_title("sklearn diabetes 資料：用 BMI 預測疾病進展")
ax.legend(fontsize=8, loc="upper left")
save(fig, "linear_fit.png")

# ---------- Figure 2: gradient descent intuition ----------
x_std = ((X_train["bmi"] - X_train["bmi"].mean()) / X_train["bmi"].std(ddof=0)).to_numpy()
yv = y_train.to_numpy()
b_opt = yv.mean()


def mse_w(w):
    return np.mean((w * x_std + b_opt - yv) ** 2)


def run_gd(lr, steps):
    w, b, hist, loss = 0.0, 0.0, [], []
    for _ in range(steps):
        err = w * x_std + b - yv
        hist.append(w)
        loss.append(np.mean(err ** 2))
        w -= lr * 2 * np.mean(err * x_std)
        b -= lr * 2 * np.mean(err)
    return np.array(hist), np.array(loss)


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ws = np.linspace(-10, 100, 200)
axes[0].plot(ws, [mse_w(w) for w in ws], color=GREY)
hist, _ = run_gd(0.1, 15)
axes[0].plot(hist, [mse_w(w) for w in hist], "o-", color=TEAL, ms=4, label="每一步的位置（學習率 0.1）")
axes[0].set_xlabel("斜率 w（標準化 BMI）")
axes[0].set_ylabel("MSE")
axes[0].set_title("沿著坡度往下走，找 MSE 最低點")
axes[0].legend(fontsize=8)
for lr, c, lab in [(0.01, GREY, "學習率 0.01（太小）"), (0.1, TEAL, "學習率 0.1（剛好）"), (1.02, ORANGE, "學習率 1.02（太大，發散）")]:
    _, loss = run_gd(lr, 40)
    axes[1].plot(loss, color=c, lw=2, label=lab)
axes[1].set_xlabel("迭代次數")
axes[1].set_ylabel("MSE")
axes[1].set_yscale("log")
axes[1].set_title("學習率太小走很慢，太大會越跳越遠")
axes[1].legend(fontsize=8)
fig.tight_layout()
save(fig, "gradient_descent.png")

# ---------- Figure 3: polynomial degree 1 / 3 / 15 ----------
def curve(x):
    return 100 * x / (2 + x)


rng = np.random.default_rng(42)
x_tr = np.sort(rng.uniform(0, 10, 25))
y_tr = curve(x_tr) + rng.normal(0, 8, 25)
x_te = rng.uniform(x_tr.min(), x_tr.max(), 200)
y_te = curve(x_te) + rng.normal(0, 8, 200)
grid = np.linspace(x_tr.min(), x_tr.max(), 400).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, d in zip(axes, [1, 3, 15]):
    mdl = make_pipeline(MinMaxScaler(feature_range=(-1, 1)), PolynomialFeatures(d), LinearRegression())
    mdl.fit(x_tr.reshape(-1, 1), y_tr)
    tr = mean_squared_error(y_tr, mdl.predict(x_tr.reshape(-1, 1)))
    te = mean_squared_error(y_te, mdl.predict(x_te.reshape(-1, 1)))
    print(f"degree {d}: train MSE {tr:.1f}, test MSE {te:.1f}")
    ax.scatter(x_te, y_te, s=8, color=GREY, alpha=0.3, label="測試集")
    ax.scatter(x_tr, y_tr, s=22, color="black", label="訓練集")
    ax.plot(grid, mdl.predict(grid), color=ORANGE if d == 15 else TEAL, lw=2)
    ax.set_ylim(-20, 120)
    ax.set_title(f"{d} 次方\n訓練 MSE {tr:.0f}／測試 MSE {te:.0f}")
    ax.set_xlabel("劑量（任意單位）")
axes[0].set_ylabel("反應")
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("模擬劑量反應資料：次方愈高，訓練誤差愈小，測試誤差卻可能暴增", y=1.04)
save(fig, "poly_overfit.png")

# ---------- Figure 4: sigmoid ----------
z = np.linspace(-8, 8, 300)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(z, 1 / (1 + np.exp(-z)), color=TEAL, lw=2.5)
ax.axhline(0.5, color=GREY, ls="--", lw=1)
ax.axhline(0.2, color=ORANGE, ls=":", lw=1.5)
ax.text(-7.8, 0.53, "閾值 0.5", color=GREY, fontsize=9)
ax.text(-7.8, 0.23, "閾值 0.2（寧可多抓）", color=ORANGE, fontsize=9)
ax.set_xlabel("線性分數 z = b₀ + b₁x₁ + …（即 log-odds）")
ax.set_ylabel("預測機率 p")
ax.set_title("sigmoid 把任意大小的分數壓進 0 到 1 之間")
save(fig, "sigmoid.png")

# ---------- Figure 5: confusion matrices at two thresholds (WDBC) ----------
bc = load_breast_cancer(as_frame=True)
Xb, yb = bc.data, 1 - bc.target  # 1 = malignant
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.25, stratify=yb, random_state=42)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xb_tr, yb_tr)
proba = clf.predict_proba(Xb_te)[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, th in zip(axes, [0.5, 0.2]):
    cm = confusion_matrix(yb_te, (proba >= th).astype(int))
    tn, fp, fn, tp = cm.ravel()
    ax.imshow(cm, cmap="Greens")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["預測良性", "預測惡性"])
    ax.set_yticks([0, 1], ["實際良性", "實際惡性"])
    ax.set_title(f"閾值 {th}\n敏感度 {tp / (tp + fn):.1%}，特異度 {tn / (tn + fp):.1%}")
fig.suptitle("WDBC 測試集（143 例）：閾值下修，漏診變少、誤報變多", y=1.03)
fig.tight_layout()
save(fig, "threshold_confusion.png")
