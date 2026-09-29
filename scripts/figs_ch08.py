"""Figures for chapter 08 (K-Means clustering).

Run from ml-site/:  .venv/bin/python scripts/figs_ch08.py
Outputs PNGs to docs/assets/img/ch08/ and prints the numbers quoted in the chapter.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

TEAL, ORANGE, GRAY = "#00897B", "#F4511E", "#607D8B"
OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "img" / "ch08"
OUT.mkdir(parents=True, exist_ok=True)
CMAP = plt.get_cmap("tab10")

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/"
       "heart_failure_clinical_records_dataset.csv")
COLS = ["age", "creatinine_phosphokinase", "ejection_fraction",
        "platelets", "serum_creatinine", "serum_sodium"]
COL_ZH = ["年齡", "CPK（log）", "射出分率", "血小板", "肌酸酐（log）", "血鈉"]


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def assign(X, centers):
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    return d.argmin(axis=1)


def update(X, labels, k):
    return np.array([X[labels == j].mean(axis=0) for j in range(k)])


def run_manual(X, k, seed, max_iter=20):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)]
    snaps = []
    for it in range(1, max_iter + 1):
        labels = assign(X, centers)
        snaps.append((it, labels, centers))
        new = update(X, labels, k)
        shift = np.linalg.norm(new - centers)
        centers = new
        if shift < 1e-6:
            break
    return snaps


# ---------- 1. step-by-step on toy data ----------
X_toy, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.9, random_state=42)
snaps = run_manual(X_toy, 3, seed=6)
print("toy: converged after", len(snaps), "iterations; sizes", np.bincount(snaps[-1][1]))

fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharex=True, sharey=True)
it0, lab0, c0 = snaps[0]
axes[0].scatter(X_toy[:, 0], X_toy[:, 1], s=9, color=GRAY, alpha=.6)
axes[0].scatter(c0[:, 0], c0[:, 1], c=[CMAP(i) for i in range(3)], marker="X", s=180, edgecolor="black")
axes[0].set_title("① 隨機放 K=3 個群中心")
for ax, (it, lab, c), title in zip(axes[1:], [snaps[0], snaps[1], snaps[-1]],
                                  ["② 每個點歸給最近的中心", "③ 中心搬到成員平均，再分配",
                                   f"④ 重複到不再變動（第 {len(snaps)} 輪）"]):
    ax.scatter(X_toy[:, 0], X_toy[:, 1], c=[CMAP(i) for i in lab], s=9, alpha=.7)
    ax.scatter(c[:, 0], c[:, 1], c=[CMAP(i) for i in range(3)], marker="X", s=180, edgecolor="black")
    ax.set_title(title)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("K-Means 的四個步驟（玩具資料，X = 群中心）", y=1.03)
save(fig, "kmeans_steps.png")

# ---------- load heart failure ----------
df = pd.read_csv(URL)
df.columns = df.columns.str.lower()
X = df[COLS].copy()
X["creatinine_phosphokinase"] = np.log(X["creatinine_phosphokinase"])
X["serum_creatinine"] = np.log(X["serum_creatinine"])
X_scaled = StandardScaler().fit_transform(X)

# ---------- 2. scaling effect ----------
km_raw = KMeans(n_clusters=3, n_init=10, random_state=42).fit(df[COLS])
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_scaled)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, labels, title in [(axes[0], km_raw.labels_, "沒有標準化：只按血小板切三段"),
                          (axes[1], km.labels_, "取 log + 標準化後：多個變數共同決定")]:
    ax.scatter(df["serum_creatinine"], df["platelets"] / 1000, c=[CMAP(i) for i in labels], s=14, alpha=.75)
    ax.set_xscale("log")
    ax.set_xlabel("血清肌酸酐（mg/dL，對數刻度）")
    ax.set_title(title)
axes[0].set_ylabel("血小板（×1000/µL）")
save(fig, "scaling_effect.png")
print("raw clusters platelets range:\n", df.assign(c=km_raw.labels_).groupby("c")["platelets"].agg(["size", "min", "max"]))

# ---------- 3. elbow + silhouette ----------
ks = list(range(2, 9))
inert, sils = [], []
for k in ks:
    m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inert.append(m.inertia_)
    sils.append(silhouette_score(X_scaled, m.labels_))
toy_sil = silhouette_score(X_toy, KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_toy).labels_)
print("k, inertia, silhouette:", [(k, round(i, 1), round(s, 3)) for k, i, s in zip(ks, inert, sils)])
print("toy silhouette:", round(toy_sil, 3))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(ks, inert, "o-", color=TEAL)
axes[0].set_xlabel("群數 K"); axes[0].set_ylabel("群內平方和（inertia）")
axes[0].set_title("手肘法：曲線平滑，找不到明顯的「肘」")
axes[1].plot(ks, sils, "o-", color=ORANGE, label="心衰竭資料")
axes[1].axhline(toy_sil, color=GRAY, ls="--", label=f"玩具資料 K=3（{toy_sil:.2f}）")
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("群數 K"); axes[1].set_ylabel("平均輪廓係數")
axes[1].set_title("輪廓係數：每個 K 都低於 0.2")
axes[1].legend(loc="upper right", fontsize=9)
save(fig, "elbow_silhouette.png")

# ---------- 4. cluster profile + post-hoc death rate ----------
centers = pd.DataFrame(km.cluster_centers_, columns=COL_ZH)
out = df.assign(cluster=km.labels_).groupby("cluster")["death_event"].agg(["size", "sum", "mean"])
print("cluster outcome:\n", out)
print("cluster medians:\n", df.assign(cluster=km.labels_).groupby("cluster")[COLS].median().round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
im = axes[0].imshow(centers.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
axes[0].set_xticks(range(len(COL_ZH))); axes[0].set_xticklabels(COL_ZH, rotation=20)
axes[0].set_yticks(range(3)); axes[0].set_yticklabels([f"群 {i}（n={n}）" for i, n in enumerate(out["size"])])
for i in range(3):
    for j in range(len(COL_ZH)):
        v = centers.values[i, j]
        axes[0].text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9,
                     color="white" if abs(v) > 0.9 else "black")
axes[0].set_title("各群中心（標準化後，0 = 全體平均）")
fig.colorbar(im, ax=axes[0], fraction=0.03)
axes[1].barh([f"群 {i}" for i in range(3)], out["mean"] * 100, color=[CMAP(i) for i in range(3)])
for i, (m, s, n) in enumerate(zip(out["mean"], out["sum"], out["size"])):
    axes[1].text(m * 100 + 1, i, f"{m*100:.1f}%（{s}/{n}）", va="center", fontsize=9)
axes[1].set_xlim(0, 100); axes[1].invert_yaxis()
axes[1].set_xlabel("追蹤期間死亡比例（%）")
axes[1].set_title("事後對照（分群時沒用到）")
save(fig, "cluster_profile.png")

# ---------- 5. stability ----------
for n_init in [1, 10]:
    runs = [KMeans(n_clusters=3, n_init=n_init, random_state=s).fit(X_scaled) for s in range(10)]
    ari = [adjusted_rand_score(runs[0].labels_, r.labels_) for r in runs[1:]]
    print(f"n_init={n_init}: ARI min={min(ari):.2f} median={np.median(ari):.2f}")

# ---------- 6. limitations: local optimum + moons ----------
bad = run_manual(X_toy, 3, seed=0)
X_moon, y_moon = make_moons(n_samples=300, noise=0.06, random_state=42)
km_moon = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X_moon)
print("bad init sizes:", np.bincount(bad[-1][1]), "moons ARI:", round(adjusted_rand_score(y_moon, km_moon.labels_), 2))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
_, lab, c = bad[-1]
axes[0].scatter(X_toy[:, 0], X_toy[:, 1], c=[CMAP(i) for i in lab], s=10, alpha=.7)
axes[0].scatter(c[:, 0], c[:, 1], c=[CMAP(i) for i in range(3)], marker="X", s=180, edgecolor="black")
axes[0].set_title("初始值不好：兩團被併成一群、一團被切兩半")
axes[1].scatter(X_moon[:, 0], X_moon[:, 1], c=[CMAP(i) for i in km_moon.labels_], s=10, alpha=.7)
axes[1].scatter(*km_moon.cluster_centers_.T, c=[CMAP(i) for i in range(2)], marker="X", s=180, edgecolor="black")
axes[1].set_title("非球形的群：K-Means 從中間直直切一刀")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
save(fig, "limitations.png")
