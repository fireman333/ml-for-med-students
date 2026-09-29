"""Figures for chapter 02 (python-toolkit).

Run from ml-site/:  .venv/bin/python scripts/figs_ch02.py
Outputs: docs/assets/img/ch02/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "img" / "ch02"
OUT.mkdir(parents=True, exist_ok=True)
URL = "https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def draw_table(ax, x0, y0, values, fmt, cell_w=1.0, cell_h=0.6, fc="white", ec=GREY, tc="black", ls="-"):
    """Draw a 2D array as a grid of cells; (x0, y0) is the top-left corner."""
    values = np.atleast_2d(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            x, y = x0 + j * cell_w, y0 - (i + 1) * cell_h
            ax.add_patch(Rectangle((x, y), cell_w, cell_h, fc=fc, ec=ec, lw=1.2, ls=ls))
            ax.text(x + cell_w / 2, y + cell_h / 2, fmt(values[i, j]), ha="center", va="center", fontsize=11, color=tc)


# ---------------------------------------------------------------- fig 1
def fig_axis_broadcasting():
    labs = np.array([[1.9, 130], [1.1, 137], [0.9, 140], [2.7, 134]])
    mu = labs.mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # (a) axis
    ax = axes[0]
    ax.set_xlim(-2.2, 4.4)
    ax.set_ylim(-3.6, 1.2)
    ax.axis("off")
    ax.set_title("(a) axis 的方向：axis=0 往下壓、axis=1 往右壓", fontsize=12)
    for j, h in enumerate(["肌酸酐", "血清鈉"]):
        ax.text(j + 0.5, 0.2, h, ha="center", fontsize=11, color=GREY)
    for i in range(4):
        ax.text(-0.15, -(i + 0.5) * 0.6, f"病人 {i + 1}", ha="right", va="center", fontsize=10, color=GREY)
    draw_table(ax, 0, 0, labs, lambda v: f"{v:g}")
    for xa in (0.13, 1.13):
        ax.add_patch(FancyArrowPatch((xa, -0.05), (xa, -2.5), arrowstyle="-|>", mutation_scale=14, color=TEAL, lw=1.6))
    draw_table(ax, 0, -2.6, mu[None, :], lambda v: f"{v:g}", fc="#E0F2F1", ec=TEAL)
    ax.text(1.0, -3.45, "mean(axis=0) → 每一欄一個值，shape (2,)", ha="center", fontsize=10, color=TEAL)
    ax.add_patch(FancyArrowPatch((2.1, -1.2), (2.9, -1.2), arrowstyle="-|>", mutation_scale=18, color=ORANGE, lw=2))
    row_mean = labs.mean(axis=1)
    draw_table(ax, 3.0, 0, row_mean[:, None], lambda v: f"{v:.2f}", fc="#FBE9E7", ec=ORANGE, cell_w=1.3)
    ax.text(3.65, 0.2, "mean(axis=1)", ha="center", fontsize=10, color=ORANGE)

    # (b) broadcasting
    ax = axes[1]
    ax.set_xlim(-0.3, 10.0)
    ax.set_ylim(-3.6, 1.2)
    ax.axis("off")
    ax.set_title("(b) 廣播：(4, 2) 減 (2,)，平均值自動複製到每一列", fontsize=12)
    draw_table(ax, 0, 0, labs, lambda v: f"{v:g}")
    ax.text(2.35, -1.2, "−", fontsize=20, ha="center", va="center")
    draw_table(ax, 2.7, 0, mu[None, :], lambda v: f"{v:g}", fc="#E0F2F1", ec=TEAL, cell_w=1.45)
    ghost = np.repeat(mu[None, :], 3, axis=0)
    draw_table(ax, 2.7, -0.6, ghost, lambda v: f"{v:g}", fc="white", ec="#80CBC4", tc="#80CBC4", ls="--", cell_w=1.45)
    ax.text(4.15, -3.1, "虛線部分 NumPy 不會真的複製，\n只是「當作」每列都有", ha="center", fontsize=9, color=TEAL)
    ax.text(5.95, -1.2, "=", fontsize=20, ha="center", va="center")
    draw_table(ax, 6.3, 0, labs - mu, lambda v: f"{v:+.2f}", fc="#FFF3E0", ec=ORANGE, cell_w=1.6)
    ax.text(7.9, 0.2, "每位病人與平均值的差", ha="center", fontsize=10, color=ORANGE)
    save(fig, "axis_broadcasting.png")


# ---------------------------------------------------------------- data
def load():
    df = pd.read_csv(URL).rename(columns={"DEATH_EVENT": "death_event"})
    assert df.shape == (299, 13), df.shape
    return df


# ---------------------------------------------------------------- fig 2
def fig_eda(df):
    dead, alive = df[df["death_event"] == 1], df[df["death_event"] == 0]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.9))
    bins = np.arange(10, 85, 5)
    axes[0].hist(alive["ejection_fraction"], bins=bins, alpha=0.65, color=TEAL, label=f"存活（n={len(alive)}）")
    axes[0].hist(dead["ejection_fraction"], bins=bins, alpha=0.65, color=ORANGE, label=f"死亡（n={len(dead)}）")
    axes[0].set_title("(a) 直方圖：EF 分布")
    axes[0].set_xlabel("左心室射出分率 EF（%）")
    axes[0].set_ylabel("病人數")
    axes[0].legend()

    bp = axes[1].boxplot([alive["serum_creatinine"], dead["serum_creatinine"]],
                         tick_labels=["存活", "死亡"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [TEAL, ORANGE]):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    axes[1].set_title("(b) 盒鬚圖：血清肌酸酐")
    axes[1].set_ylabel("血清肌酸酐（mg/dL）")

    colors = np.where(df["death_event"] == 1, ORANGE, TEAL)
    axes[2].scatter(df["serum_creatinine"], df["serum_sodium"], c=colors, alpha=0.6, s=22, edgecolors="none")
    r = stats.pearsonr(df["serum_creatinine"], df["serum_sodium"]).statistic
    axes[2].set_title(f"(c) 散佈圖：r = {r:.2f}")
    axes[2].set_xlabel("血清肌酸酐（mg/dL）")
    axes[2].set_ylabel("血清鈉（mEq/L）")
    axes[2].scatter([], [], c=TEAL, label="存活")
    axes[2].scatter([], [], c=ORANGE, label="死亡")
    axes[2].legend(loc="lower right")
    fig.suptitle("Heart Failure Clinical Records（UCI 519，n = 299）", fontsize=12, color=GREY)
    fig.tight_layout()
    save(fig, "eda_panels.png")


# ---------------------------------------------------------------- fig 3
def fig_effect_ci(df):
    dead, alive = df[df["death_event"] == 1], df[df["death_event"] == 0]
    t = stats.ttest_ind(dead["ejection_fraction"], alive["ejection_fraction"], equal_var=False)
    ci = t.confidence_interval(confidence_level=0.95)
    ef_diff = dead["ejection_fraction"].mean() - alive["ejection_fraction"].mean()

    tab = pd.crosstab(df["high_blood_pressure"], df["death_event"])
    chi = stats.chi2_contingency(tab)
    n1, n0 = tab.loc[1].sum(), tab.loc[0].sum()
    p1, p0 = tab.loc[1, 1] / n1, tab.loc[0, 1] / n0
    rd = (p1 - p0) * 100
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0) * 100
    rd_lo, rd_hi = rd - 1.96 * se, rd + 1.96 * se  # Wald CI, for illustration

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2))
    rows = [
        (axes[0], ef_diff, ci.low, ci.high, t.pvalue, "EF 平均差（死亡組 − 存活組，百分點）", TEAL,
         f"差 {ef_diff:.1f}（95% CI {ci.low:.1f} 至 {ci.high:.1f}），p = {t.pvalue:.2g}\n信賴區間不含 0：達統計顯著", (-14, 14)),
        (axes[1], rd, rd_lo, rd_hi, chi.pvalue, "死亡比例差（高血壓 − 無高血壓，百分點）", ORANGE,
         f"差 {rd:.1f}（95% CI {rd_lo:.1f} 至 {rd_hi:.1f}），p = {chi.pvalue:.2f}\n信賴區間跨過 0：未達統計顯著（不等於「沒有關聯」）", (-14, 24)),
    ]
    for ax, est, lo, hi, p, label, c, note, xlim in rows:
        ax.axvline(0, color=GREY, ls="--", lw=1)
        ax.errorbar([est], [0], xerr=[[est - lo], [hi - est]], fmt="o", color=c, ms=9, capsize=6, lw=2.2)
        ax.set_xlim(*xlim)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_xlabel(label, fontsize=10)
        ax.text(xlim[1] - 0.3, 0.45, note, ha="right", va="center", fontsize=9.5, color="#263238")
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
    fig.suptitle("看 p 值之前，先看效果大小與 95% 信賴區間", fontsize=12)
    fig.tight_layout()
    save(fig, "effect_ci.png")


if __name__ == "__main__":
    fig_axis_broadcasting()
    df = load()
    fig_eda(df)
    fig_effect_ci(df)
