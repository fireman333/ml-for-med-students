"""Figures for chapter 15 (attention and Transformer).

Run from ml-site/:
    .venv-colab/bin/python scripts/figs_ch15.py     # all figures (needs keras + tensorflow)
    .venv/bin/python scripts/figs_ch15.py           # concept figure only (no keras in .venv)

Outputs PNGs to docs/assets/img/ch15/ and the attention data used by the
interactive demo to docs/assets/js/demos/ch15-attention-data.js.
The model code mirrors docs/notebooks/ch15_transformer.ipynb (same seed, same layers),
so the numbers match the notebook on the same machine; other versions/hardware may differ slightly.
"""
import json
import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"   # must be set before importing keras

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
plt.rcParams["axes.unicode_minus"] = False

TEAL, ORANGE, GREY = "#00897B", "#F4511E", "#607D8B"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "img" / "ch15"
OUT.mkdir(parents=True, exist_ok=True)
DEMO_DATA = ROOT / "docs" / "assets" / "js" / "demos" / "ch15-attention-data.js"
RS = 42
EXAMPLE = 12   # test-set index used for the attention figure and the demo (a malaria description)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def arrow(ax, p, q, color=GREY, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))


# ---------------------------------------------------------------- 1. concept pipeline
def fig_pipeline():
    """Text -> tokens -> ids -> embedding vectors -> self-attention -> [CLS] -> diagnosis."""
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.45, 5.6)
    ax.axis("off")
    tokens = ["[CLS]", "fever", "chills", "sweating", "muscles", "ache"]
    ids = ["1039", "17", "86", "60", "95", "131"]
    xs = np.linspace(2.4, 9.7, len(tokens))
    rng = np.random.default_rng(RS)

    ax.text(0.1, 5.25, "① 原始文字", fontsize=11, color=GREY)
    ax.text(6.0, 5.25, "\"Fever, chills, sweating, my muscles ache.\"", fontsize=11, ha="center", style="italic")

    ax.text(0.1, 4.45, "② 斷詞", fontsize=11, color=GREY)
    ax.text(0.1, 3.75, "③ 查編號（示意）", fontsize=11, color=GREY)
    ax.text(0.1, 2.75, "④ 嵌入向量", fontsize=11, color=GREY)
    ax.text(0.1, 2.48, "（字＋位置）", fontsize=9, color=GREY)
    ax.text(0.1, 1.0, "⑤ 自注意力", fontsize=11, color=GREY)

    for k, (x, tok, i) in enumerate(zip(xs, tokens, ids)):
        is_cls = k == 0
        ec = ORANGE if is_cls else TEAL
        ax.add_patch(FancyBboxPatch((x - 0.55, 4.25), 1.1, 0.42, boxstyle="round,pad=0.03",
                                    fc="white", ec=ec, lw=1.6))
        ax.text(x, 4.46, tok, ha="center", va="center", fontsize=10)
        ax.text(x, 3.8, i, ha="center", va="center", fontsize=10, color=GREY, family="monospace")
        arrow(ax, (x, 4.22), (x, 3.98))
        arrow(ax, (x, 3.62), (x, 3.32))
        vals = rng.uniform(0.15, 0.95, size=6)
        for j, v in enumerate(vals):
            ax.add_patch(Rectangle((x - 0.18, 2.1 + j * 0.19), 0.36, 0.17,
                                   fc=plt.cm.Greens(v), ec="white", lw=0.5))

    # attention arcs from [CLS] to every word (thickness = toy weight)
    weights = [0.08, 0.20, 0.22, 0.30, 0.10, 0.10]
    for x, w in zip(xs[1:], weights[1:]):
        ax.annotate("", xy=(x, 1.62), xytext=(xs[0], 1.62),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.6 + 9 * w, alpha=0.75,
                                    connectionstyle="arc3,rad=0.35"))
        ax.text(x, 1.82, f"{w:.2f}", ha="center", fontsize=9, color=ORANGE)
    ax.text(xs[0], 1.82, f"{weights[0]:.2f}", ha="center", fontsize=9, color=ORANGE)
    ax.text(6.0, -0.3, "[CLS] 分給每個字的注意力權重（示意數字，加總 = 1）", ha="center", fontsize=9.5, color=ORANGE)

    ax.add_patch(FancyBboxPatch((8.85, 4.95), 1.6, 0.5, boxstyle="round,pad=0.04", fc=TEAL, ec=TEAL, alpha=0.12))
    ax.text(9.65, 5.2, "⑥ [CLS] 輸出 → softmax\n→ 22 個診斷的機率", ha="center", va="center", fontsize=9)
    save(fig, "pipeline.png")


# ---------------------------------------------------------------- model (needs keras)
def train_model():
    import keras
    from keras import layers, ops
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    keras.utils.set_random_seed(RS)
    base = "https://huggingface.co/datasets/gretelai/symptom_to_diagnosis/resolve/main/"
    try:
        train_df = pd.read_json(base + "train.jsonl", lines=True)
        test_df = pd.read_json(base + "test.jsonl", lines=True)
    except Exception as e:
        raise RuntimeError("Could not download symptom_to_diagnosis from Hugging Face: " + repr(e)) from e
    Xtr_text, ytr_text = train_df["input_text"], train_df["output_text"]
    Xte_text, yte_text = test_df["input_text"], test_df["output_text"]

    baseline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=2000))
    baseline.fit(Xtr_text, ytr_text)
    pred_base = baseline.predict(Xte_text)

    maxlen = 64
    vec = layers.TextVectorization(max_tokens=5000, output_sequence_length=maxlen)
    vec.adapt(Xtr_text.to_numpy())
    vocab = [str(t) for t in vec.get_vocabulary()] + ["[CLS]"]
    cls_id, vocab_size = len(vocab) - 1, len(vocab)

    def encode(texts):
        ids = vec(np.asarray(texts)).numpy()
        return np.concatenate([np.full((len(ids), 1), cls_id), ids], axis=1).astype("int32")

    X_tr, X_te = encode(Xtr_text), encode(Xte_text)
    classes = sorted(ytr_text.unique())
    y_tr = ytr_text.map({c: i for i, c in enumerate(classes)}).to_numpy()

    class TokenAndPositionEmbedding(layers.Layer):
        def __init__(self, maxlen, vocab_size, dim, **kwargs):
            super().__init__(**kwargs)
            self.token_emb = layers.Embedding(vocab_size, dim)
            self.pos_emb = layers.Embedding(maxlen, dim)

        def call(self, x):
            return self.token_emb(x) + self.pos_emb(ops.arange(ops.shape(x)[-1]))

    def build_model(num_heads=2, dim=64):
        seq_len = maxlen + 1
        inputs = keras.Input(shape=(seq_len,), dtype="int32")
        pad_mask = ops.expand_dims(ops.not_equal(inputs, 0), 1)
        x = TokenAndPositionEmbedding(seq_len, vocab_size, dim)(inputs)
        attn_out, attn_scores = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim // num_heads, name="self_attention")(
            x, x, attention_mask=pad_mask, return_attention_scores=True)
        x = layers.LayerNormalization()(x + layers.Dropout(0.1)(attn_out))
        ff = layers.Dense(2 * dim, activation="relu")(x)
        ff = layers.Dense(dim)(ff)
        x = layers.LayerNormalization()(x + layers.Dropout(0.1)(ff))
        cls_vec = layers.Dropout(0.2)(x[:, 0, :])
        outputs = layers.Dense(len(classes), activation="softmax")(cls_vec)
        model = keras.Model(inputs, outputs)
        attn_model = keras.Model(inputs, attn_scores)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-3),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model, attn_model

    build_model()                       # same call order as the notebook (section 4)
    keras.utils.set_random_seed(RS)
    model, attn_model = build_model()
    model.fit(X_tr, y_tr, validation_split=0.15, epochs=60, batch_size=32, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                                       restore_best_weights=True)])
    pred_tf = np.array(classes)[model.predict(X_te, verbose=0).argmax(axis=1)]
    return dict(model=model, attn_model=attn_model, X_te=X_te, vocab=vocab, classes=classes,
                y_true=yte_text.to_numpy(), pred_base=pred_base, pred_tf=pred_tf,
                text=Xte_text.to_numpy())


# ---------------------------------------------------------------- 2. attention heatmap
def fig_attention(r):
    i = EXAMPLE
    X_te, vocab = r["X_te"], r["vocab"]
    n = int((X_te[i] != 0).sum())
    tokens = [vocab[t] for t in X_te[i, :n]]
    scores = r["attn_model"].predict(X_te[i:i + 1], verbose=0)[0][:, :n, :n]   # (heads, n, n)

    fig, ax = plt.subplots(2, 1, figsize=(10.5, 9.6), gridspec_kw={"height_ratios": [1, 4]})
    im0 = ax[0].imshow(scores[:, 0, :], cmap="Greens", aspect="auto")
    ax[0].set_yticks(range(scores.shape[0]), [f"第 {h + 1} 頭" for h in range(scores.shape[0])])
    ax[0].set_xticks(range(n), tokens, rotation=90, fontsize=9)
    ax[0].set_title(f"[CLS] 分給每個字的注意力（平均分配時每格 = 1/{n} ≈ {1 / n:.3f}）", fontsize=11)
    fig.colorbar(im0, ax=ax[0], fraction=0.025)
    im1 = ax[1].imshow(scores[0], cmap="Greens")
    ax[1].set_xticks(range(n), tokens, rotation=90, fontsize=9)
    ax[1].set_yticks(range(n), tokens, fontsize=9)
    ax[1].set_xlabel("被看的字（key）")
    ax[1].set_ylabel("發問的字（query）")
    ax[1].set_title("第 1 頭的完整自注意力矩陣（每一列加總 = 1）", fontsize=11)
    fig.colorbar(im1, ax=ax[1], fraction=0.035)
    fig.suptitle(f"測試集第 {i} 句｜真實診斷：{r['y_true'][i]}", fontsize=12)
    fig.tight_layout()
    save(fig, "attention_heatmap.png")

    # export for the interactive demo (rounded to 3 decimals, both heads)
    payload = {
        "text": str(r["text"][i]),
        "label": str(r["y_true"][i]),
        "pred": str(r["pred_tf"][i]),
        "tokens": tokens,
        "heads": np.round(scores, 3).tolist(),
    }
    DEMO_DATA.write_text(
        "// Auto-generated by scripts/figs_ch15.py -- do not edit by hand.\n"
        "// Self-attention weights of the chapter-15 mini Transformer for one test sentence\n"
        "// (gretelai/symptom_to_diagnosis, Apache-2.0).\n"
        "window.CH15_ATTENTION = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8")
    print("saved", DEMO_DATA)
    top = np.argsort(-scores[:, 0, :].mean(0))[:5]
    print("top [CLS] tokens:", [(tokens[j], round(float(scores[:, 0, j].mean()), 3)) for j in top])


# ---------------------------------------------------------------- 3. accuracy + bootstrap CI
def fig_accuracy(r):
    y = r["y_true"]
    c_base = (r["pred_base"] == y).astype(float)
    c_tf = (r["pred_tf"] == y).astype(float)
    rng = np.random.default_rng(RS)
    idx = rng.integers(0, len(y), size=(2000, len(y)))
    boot_base, boot_tf = c_base[idx].mean(1), c_tf[idx].mean(1)
    diff = boot_tf - boot_base
    acc = [c_base.mean(), c_tf.mean()]
    ci = [np.percentile(boot_base, [2.5, 97.5]), np.percentile(boot_tf, [2.5, 97.5])]
    dlo, dhi = np.percentile(diff, [2.5, 97.5])
    print(f"accuracy TF-IDF {acc[0]:.3f} CI {ci[0].round(3)} | Transformer {acc[1]:.3f} CI {ci[1].round(3)}")
    print(f"difference {acc[1] - acc[0]:+.3f}, 95% CI [{dlo:+.3f}, {dhi:+.3f}]")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    names = ["TF-IDF＋邏輯迴歸", "迷你 Transformer"]
    for k, (a, (lo, hi), col) in enumerate(zip(acc, ci, [GREY, TEAL])):
        ax[0].errorbar(a, k, xerr=[[a - lo], [hi - a]], fmt="o", color=col, capsize=6, ms=9, lw=2)
        ax[0].text(hi + 0.006, k, f"{a:.3f}\n[{lo:.3f}, {hi:.3f}]", va="center", fontsize=9)
    ax[0].set_yticks([0, 1], names)
    ax[0].set_ylim(-0.6, 1.6)
    ax[0].set_xlim(0.84, 1.0)
    ax[0].set_xlabel("測試集準確率（n = 212，點 = 實際值，線 = bootstrap 95% 信賴區間）")
    ax[0].set_title("兩個模型的準確率", fontsize=11)
    ax[1].hist(diff, bins=np.arange(-0.06, 0.11, 0.0047), color=TEAL, alpha=0.7)
    ax[1].axvline(0, color=ORANGE, lw=2, ls="--")
    ax[1].axvspan(dlo, dhi, color=GREY, alpha=0.15)
    ax[1].set_xlabel("準確率差距（Transformer − TF-IDF）")
    ax[1].set_ylabel("bootstrap 次數")
    crosses = "跨過 0" if dlo < 0 < dhi else "未跨過 0"
    ax[1].set_title(f"差距 {acc[1] - acc[0]:+.3f}，95% CI [{dlo:+.3f}, {dhi:+.3f}] {crosses}", fontsize=11)
    fig.tight_layout()
    save(fig, "accuracy_ci.png")


if __name__ == "__main__":
    fig_pipeline()
    try:
        import keras  # noqa: F401
    except ImportError:
        print("keras not installed -> skipped attention_heatmap.png / accuracy_ci.png "
              "(run with .venv-colab/bin/python)")
    else:
        result = train_model()
        fig_attention(result)
        fig_accuracy(result)
