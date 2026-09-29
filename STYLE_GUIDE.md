# 寫作規範（所有章節 agent 必讀）

網站：「醫學生的機器學習入門」— MkDocs Material 靜態站，部署於 Cloudflare Pages，公開科普／作品集用途。
讀者：醫學／生醫相關科系學生（非資工背景），數學到高中程度，懂基本統計概念（平均、標準差、p 值、敏感度／特異度）。

## 1. 參考資料（寫之前讀）

- `research/01_curricula.md` — 每章教學順序、比喻、常見誤解（萃取教法，**不可抄原文**）
- `research/02_datasets.md` — 每章主推資料集與已實測的載入程式碼
- `research/03_interactive.md` — 互動元件與授權
- `research/04_api_terms.md` — **API 陷阱（必遵守）** 與 **台灣術語對照表（必遵守）**

## 2. 語言與用語

- 繁體中文、台灣用語。術語一律照 `04_api_terms.md` 對照表：資料（非數據）、程式（非程序）、函式、套件、變數、預設、演算法、最佳化、迴歸、支援向量機、過擬合……
- ML 術語首次出現寫「中文（English）」，之後只用中文；醫學名詞首次出現「English（中文）」。
- 語氣：像學長姐帶學弟妹，親切但不油。不用 emoji 當裝飾（admonition 圖示除外）。不用「讓我們一起…吧！」這類空話。
- 禁用詞：caveat、per se、a priori、ad hoc（改中文白話）。
- **統計措辭嚴謹**：未達顯著寫「未達統計顯著」，不可寫「沒差」「無效」；觀察性關聯不寫成因果。
- **身分**：全站不得以「醫師」「醫生」自稱作者；醫學案例一律是公開資料集的教學示範，加一句「本例僅供學習，不構成臨床建議」即可（一次，放在醫學案例段落）。
- ML 模型的醫學應用要有分寸：提到「模型預測準確率 95%」時，一併提醒 資料集小、單一來源、未經外部驗證。

## 3. 每章檔案與結構

你負責的檔案（只能動這些，不准改 `mkdocs.yml`、`extra.css` 或其他章節）：

| 產物 | 路徑 |
|---|---|
| 章節頁 | `docs/chapters/NN-slug.md` |
| Notebook | `docs/notebooks/chNN_slug.ipynb` |
| 產圖腳本 | `scripts/figs_chNN.py` → 輸出 `docs/assets/img/chNN/*.png` |
| 互動 demo（選配） | `docs/assets/js/demos/chNN-*.js` |
| 本章參考文獻 | `research/refs/chNN.md`（附錄彙整用：標題、URL、授權、用途） |

章節頁結構（標題文字可依內容調整，順序固定）：

```markdown
# 第 N 章　標題

<p class="chapter-meta">預計閱讀 25 分鐘 ・ 先備：第 X 章</p>

!!! abstract "本章你會學到"
    - 3–5 點，動詞開頭

<div class="nb-links" markdown>
[:simple-googlecolab: 在 Colab 開啟](https://colab.research.google.com/github/GITHUB_REPO_PLACEHOLDER/blob/main/docs/notebooks/chNN_slug.ipynb){ .md-button .md-button--primary }
[:material-download: 下載 Notebook](../notebooks/chNN_slug.ipynb){ .md-button }
</div>

## 1. 直覺：從一個臨床情境開始
（生活／醫學比喻，不出現公式）

## 2. 核心概念
（圖解為主；公式放在可折疊區塊）
??? note "數學補充（可跳過）"
    $$ ... $$

## 3. 動手做
（程式碼區塊與 notebook 內容一致；每段程式碼前一句話說明要做什麼，後一句話說明輸出代表什麼）

## 4. 醫學案例
!!! info "僅供學習"
    ...

## 5. 互動體驗（有才放）

## 6. 常見陷阱
!!! warning "..."  （2–4 個）

## 7. 小測驗
??? question "Q1. ..."
    **答案：** ...（附一句解釋）
（3 題）

## 8. 重點整理
- ...

## 延伸閱讀
- [標題](URL) — 一句話說明
```

- 篇幅：3,000–5,000 中文字（不含程式碼）。
- 每章至少 2 張自產圖（`scripts/figs_chNN.py` 產生），1 個 mermaid 流程圖或表格。
- 圖片引用：`![說明](../assets/img/chNN/xxx.png)`，並加 `{ loading=lazy }`。
- 內部連結用相對路徑，如 `[第 4 章](04-regression.md)`。
- 公式用 `$...$` / `$$...$$`（KaTeX）。

## 4. 產圖腳本

- matplotlib，檔頭設定中文字型：
  ```python
  import matplotlib; matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]
  plt.rcParams["axes.unicode_minus"] = False
  ```
- 輸出 PNG，`dpi=150`，寬度 ≤ 1600 px，透明背景用 `facecolor="white"`（深色模式下仍可讀）。
- 配色：主色 teal `#00897B`、強調 `#F4511E`、中性灰 `#607D8B`；分類色用 `tab10`。
- 圖上文字用中文；固定 `random_state=42` 讓圖可重現。
- 執行：`.venv/bin/python scripts/figs_chNN.py`（在 `ml-site/` 目錄下）。

## 5. Notebook

- 用 `nbformat` 以 Python 產生或直接寫 JSON；第一格 markdown：章名、Colab 說明、網站連結。
- **圖上標籤用英文**（Colab 無中文字型，會變方塊）；markdown 說明用中文。
- Colab 未預裝的套件（依 04 研究結果）才在第一個 code cell `%pip install -q ...`；本機執行時該 cell 也要能跑（或用 `try/except ImportError`）。
- 資料載入用 `02_datasets.md` 實測過的方式；網路下載失敗要有清楚錯誤訊息。
- 每個 code cell 前有 markdown 說明；末尾有 2–3 個「動手試試」練習（改參數觀察）。
- **版本相容底線 = Colab 現況**：scikit-learn 1.6.1 / pandas 2.2.3 / numpy 2.1.3 / keras 3.13.2 + TF 2.20；同時必須在 sklearn 1.9.1 / pandas 3.0.6 / numpy 2.5 跑得過且**無 FutureWarning/DeprecationWarning**。不要 `pip install -U scikit-learn`。第一個 code cell 印出版本。
- 必避（詳見 `research/04_api_terms.md` A 節）：`inplace=True` 與連鎖賦值（pandas 3 靜默失效，一律 `df["a"] = ...` / `.loc`）；`OneHotEncoder(sparse=)`→`sparse_output`；`LogisticRegression` 不寫 `penalty`/`l1_ratio`；`MLPClassifier(early_stopping=True)` 不用；`SVC(probability=True)` 不用（要機率用 `CalibratedClassifierCV(SVC(), ensemble=False)` 或 `decision_function`）；`np.float_`/`np.NaN`/`np.trapz`/`np.in1d` 不用；`chi2` 前不可標準化出負值；Keras 用 `learning_rate=`。
- **必須在兩個 kernel 實測執行**，零錯誤才算完成（最後保留 ml-colab 的輸出）：
  ```bash
  .venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=ml-latest --ExecutePreprocessor.timeout=900 --output /tmp/chNN_latest.ipynb docs/notebooks/chNN_slug.ipynb
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=ml-colab --ExecutePreprocessor.timeout=900 docs/notebooks/chNN_slug.ipynb
  ```
  兩次的 stderr 都要檢查有無 Warning。產生 notebook 時 metadata kernelspec 設為 `{"name": "python3", "display_name": "Python 3", "language": "python"}`（執行完再改回，讓 Colab 能開）。
- 已裝套件：兩個 venv 都有 numpy/pandas/matplotlib/scipy/sklearn/statsmodels/seaborn/ucimlrepo/requests；只有 ml-colab 有 tensorflow/keras。需要其他套件先 `pip install` 到兩個 venv 並在 `research/refs/chNN.md` 註記。
- 網路資料：`ucimlrepo` 可能要 50–60 秒且偶爾失敗；能用 sklearn 內建或直接 CSV URL 就優先用，並包 `try/except` 給清楚錯誤訊息。
- 執行時間目標 < 3 分鐘（Colab 免費版 CPU）。

## 6. 互動 demo（自寫）

- 放 `docs/assets/js/demos/chNN-name.js`，只用原生 JS 或 Plotly（已由 mkdocs.yml 全站載入）。
- 全站已關閉 navigation.instant（每頁完整載入）。頁內 script 早於 Material 主程式執行，**不可直接呼叫 `document$`**。寫法：`function init(){ const el = document.getElementById("chNN-demo"); if (!el) return; ... } if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();`
- 在章節頁用 `<script src="../../assets/js/demos/chNN-name.js"></script>` 引入（放在頁面最後）。容器：`<div class="demo-box"><div id="chNN-demo"></div><div class="controls">…</div></div>`。
- 深淺色模式都要看得清楚（用 CSS 變數或讀 `document.body.getAttribute("data-md-color-scheme")`）。
- 外部 demo：TensorFlow Playground（Apache-2.0）可 iframe；MLU-Explain（CC BY-SA）、Naftali Harris、setosa.io **只放連結不 iframe**。

## 7. 醫學資料與倫理

- 只用公開、授權允許的資料集；在章節中寫出資料集來源與授權。
- 不放任何真實可識別病人資料。

## 8. 完成定義（交回前自查）

1. `.venv/bin/mkdocs build --strict -d /tmp/mlsite_chNN` 無 warning（在 `ml-site/` 下執行；**一定要加 `-d` 到自己的暫存目錄**，因為其他章節 agent 同時在 build；若失敗原因在別章，註明即可）。不要執行任何 git 指令。
2. Notebook `nbconvert --execute` 零錯誤
3. 產圖腳本可重跑、圖檔存在且被章節引用
4. 術語 grep：章節中不出現「數據」「程序」「函數」（統計「函數」語境例外需合理）「默認」「算法」「優化」「回歸」
5. 字數 3,000–5,000
