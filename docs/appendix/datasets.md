# 資料集一覽

本站所有範例都使用公開資料集。下表依「實際被各章 notebook 載入」整理，包含使用章節、載入方式、樣本數、授權、引用與已知問題。

!!! warning "使用前請注意"
    - 授權欄標「**未驗證**」或「**查不到**」者，代表撰寫時未能確認授權條款，重新散布或商業使用前請自行到原始頁面查證。
    - UCI Machine Learning Repository 的各資料集，其網頁授權區塊實查皆為 CC BY 4.0（使用時需標註來源）。
    - 這些資料集多為單一來源、樣本數小，模型在上面的表現**不能推論到真實臨床情境**，僅供教學。
    - 本站不重新散布這些資料：notebook 於執行時才從原站下載，或使用 scikit-learn 內建版本。

## 主要資料集（notebook 實際載入）

| 名稱 | 使用章節 | 載入方式（notebook 程式碼） | 樣本數 × 特徵 | Outcome | 授權 | 引用 | 已知問題 |
|---|---|---|---|---|---|---|---|
| Breast Cancer Wisconsin (Diagnostic)，WDBC | 01、04、05、06、09、10、11 | `sklearn.datasets.load_breast_cancer`（離線可用）；[UCI 17](https://archive.ics.uci.edu/dataset/17) | 569 × 30 | 良性（357）／惡性（212） | CC BY 4.0（UCI 頁面實查） | Street WN, Wolberg WH, Mangasarian OL. SPIE 1905, 1993；DOI 10.24432/C5DW2B | 30 個特徵是 10 種細胞核影像測量值 × (mean, SE, worst)，高度共線；sklearn 版 target 0=惡性、1=良性，與直覺相反；準確率偏高，易讓人低估難度 |
| Heart Failure Clinical Records | 02、07、08 | 第 2 章：`pd.read_csv("https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip")`，`ucimlrepo.fetch_ucirepo(id=519)` 為備援；第 7、8 章：`pd.read_csv(".../machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")`；[UCI 519](https://archive.ics.uci.edu/dataset/519) | 299 × 12（加 death_event 共 13 欄） | death_event（死亡 32.1%） | CC BY 4.0（UCI 頁面實查） | Chicco D, Jurman G. BMC Med Inform Decis Mak 2020;20:16；原始資料 Ahmad T et al. PLoS One 2017;12:e0181001；DOI 10.24432/C5Z89R | 單一醫院（巴基斯坦 Faisalabad）、樣本小；`time`（追蹤天數）是 outcome 的結果性變數，用於預測會有資料洩漏（第 7 章 §4.2 示範），須移除或明講；`platelets` 欄疑似平均值填補（本站作者推測） |
| Chronic Kidney Disease（CKD） | 03 | `pd.read_csv` 讀 `https://archive.ics.uci.edu/static/public/336/data.csv`（經 `requests` 下載）；下載失敗時退回模擬資料 `make_fake_ckd()`，輸出會標示 SYNTHETIC；[UCI 336](https://archive.ics.uci.edu/dataset/336) | 400 × 24 | ckd／notckd | CC BY 4.0（UCI 頁面實查） | Rubini L, Soundarapandian P, Eswaran P（Apollo Hospitals, 印度）, 2015；DOI 10.24432/C5G020 | 約 1,012 格遺漏值；類別欄字串不一致；標籤含髒值（`'ckd\t'` 與 `'ckd'` 並存）；原始資料沒有混單位，第 3 章的肌酸酐 µmol/L 混雜（20% 列 ×88.4）是教學用人工注入 |
| Diabetes（sklearn 內建） | 04、10 | `sklearn.datasets.load_diabetes`（第 4 章 `scaled=False` 保留原始單位；第 10 章 `return_X_y=True, as_frame=True`） | 442 × 10 | 一年後疾病進展指標（連續） | 查不到（隨 scikit-learn BSD-3 散布；原始資料授權未驗證） | Efron B, Hastie T, Johnstone I, Tibshirani R. Least angle regression. Ann Stat 2004;32:407–499 | 預設 `scaled=True` 時特徵已置中縮放、失去原始單位；`sex` 為連續數值編碼；s1、s2 高度共線 |
| Medical Abstracts TC Corpus | 05 | `pd.read_csv` 讀 GitHub raw：`https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv`、`medical_tc_test.csv`；[GitHub（sebischair）](https://github.com/sebischair/Medical-Abstracts-TC-Corpus) | 訓練 11,550、測試 2,888（文字） | 5 類疾病大類（腫瘤、消化、神經、心血管、一般病理） | Hugging Face 卡片（TimSchopf/medical_abstracts）標 CC BY-SA 3.0；GitHub repo 本身未標授權；未驗證 | 請以 repo README 為準（撰寫時未核對論文引用細節） | 單標籤但摘要可能屬多類；「一般病理」為雜項類別、界線模糊；MultinomialNB 準確率約 0.53（實測），屬預期 |
| Heart Disease（Cleveland 子集） | 07、10 | `pd.read_csv(".../machine-learning-databases/heart-disease/processed.cleveland.data", names=cols, na_values="?")`，第 7 章以 `ucimlrepo.fetch_ucirepo(id=45)` 為備援；[UCI 45](https://archive.ics.uci.edu/dataset/45) | 303 × 13（加 num 共 14 欄） | 是否有心臟病（num > 0 二元化） | CC BY 4.0（UCI 頁面實查） | Detrano R et al. Am J Cardiol 1989;64:304–310；DOI 10.24432/C52P4X | 1988 年單一中心；遺漏值 6 格（ca、thal）；`cp`、`thal`、`slope` 為類別卻以整數呈現，線性模型須 one-hot；無表頭、`?` 為遺漏值；網路流傳的 heart.csv 來源不明，請固定引用 UCI 45 |
| PneumoniaMNIST（MedMNIST v2） | 09 | NumPy 讀取 Zenodo 的 `pneumoniamnist.npz`：`https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1`（不需 medmnist 套件）；[MedMNIST](https://medmnist.com/) | 訓練集 4,708，28×28 灰階影像 | 正常（1,214）／肺炎（3,494） | CC BY 4.0（`medmnist.INFO` 實測） | Yang J et al. Sci Data 2023;10:41；Yang J, Shi R, Ni B. IEEE ISBI 2021；原始影像 Kermany DS et al. Cell 2018;172:1122–1131 | 肺炎佔約 74%，準確率會虛高，須搭配 AUC 與召回率；28×28 解析度遠低於臨床可用；原始影像為單一醫學中心的小兒胸部 X 光；MedMNIST 各子集授權不同（如 DermaMNIST 為 CC BY-NC 4.0），勿假設整包同授權 |
| CDC Diabetes Health Indicators | 11 | `pd.read_csv("https://archive.ics.uci.edu/static/public/891/data.csv")`，備援 `ucimlrepo.fetch_ucirepo(id=891)`；[UCI 891](https://archive.ics.uci.edu/dataset/891) | 253,680 × 21 | Diabetes_binary（陽性 13.9%） | CC0 1.0（UCI 頁面指向 Kaggle 原始資料集；經 Kaggle API 於 2026-09-30 查詢，`licenseName` 為 CC0: Public Domain） | DOI 10.24432/C53919；源自 CDC BRFSS 2015 | BRFSS 自陳問卷，非診斷資料；Age 為 13 級分組編碼；資料量大，SVM 等須抽樣（如分層抽樣 20,000 列） |

## 補充示範資料集（僅在小節內使用）

| 名稱 | 使用章節 | 載入方式（notebook 程式碼） | 樣本數 × 特徵 | Outcome | 授權 | 引用 | 已知問題 |
|---|---|---|---|---|---|---|---|
| Pima Indians Diabetes | 03（§「0 當遺漏值」補充示範；失敗會跳過） | `sklearn.datasets.fetch_openml(data_id=37, as_frame=True)`；[OpenML 37](https://www.openml.org/d/37) | 768 × 8 | 糖尿病陽性（34.9%） | 查不到；未驗證 | Smith JW et al. Proc Symp Comput Appl Med Care 1988:261–265 | 0 代表遺漏值（Glucose、BloodPressure、SkinThickness、Insulin、BMI 皆有）；全為 21 歲以上 Pima 印地安女性 |

## 自產／模擬資料（不涉及第三方授權）

| 名稱 | 使用章節 | 產生方式 | 用途 |
|---|---|---|---|
| 模擬劑量反應資料 | 04 | Emax 型曲線 100x/(2+x) 加常態雜訊，seed 42（本站自產） | 多項式過擬合示範、互動 demo |
| `make_moons` | 06、08 | `sklearn.datasets.make_moons`（第 6 章 n=150、noise=0.3；第 8 章 n=300、noise=0.06） | 非線性邊界（SVM 核函數）、K-Means 的限制 |
| `make_blobs` | 08 | `sklearn.datasets.make_blobs(n_samples=300, centers=3)` | K-Means 迭代示範 |
| 模擬身高／體重 | 10（互動 demo） | 固定亂數種子的標準化模擬資料（r ≈ 0.8），見 `ch10-pca-projection.js` | PCA 投影互動示範 |
| 模擬 CKD（備援） | 03 | notebook 內 `make_fake_ckd()`，僅在 UCI 下載失敗時使用 | 離線備用，輸出標示 SYNTHETIC |

## 備選資料集（目前沒有任何章節使用）

原計畫列入、但寫作過程中被其他資料集取代的資料集，保留供日後練習或延伸參考。

| 名稱 | 來源 | 樣本數 × 特徵 | 授權 | 備註 |
|---|---|---|---|---|
| Insurance（insurance.csv） | [GitHub（stedy）](https://github.com/stedy/Machine-Learning-with-R-datasets) | 1,338 × 7 | 查不到（repo 未見授權檔）；未驗證 | 第 4 章改用 sklearn diabetes；此資料為教學造出的資料，非真實理賠紀錄 |
| Breast Cancer Coimbra | [UCI 451](https://archive.ics.uci.edu/dataset/451) | 116 × 9 | CC BY 4.0（UCI 頁面實查） | 樣本極小，適合討論小樣本的過度自信 |
| Mammography | [OpenML](https://www.openml.org/search?type=data&q=mammography) | 11,183 × 6 | 查不到；未驗證 | 極端不平衡（陽性 2.3%） |
| Digits（sklearn 內建） | `sklearn.datasets.load_digits` | 1,797 × 64 | 查不到；未驗證 | 非醫學資料，僅作離線備用 |

## 開放資料 API（第 3 章示範）

| 來源 | 實際使用 | 授權 | 備註 |
|---|---|---|---|
| WHO Global Health Observatory（GHO）OData API | notebook 以 `https://ghoapi.azureedge.net/api/` 查詢出生時平均餘命（指標 WHOSIS_000001）；另以 `robotparser` 讀 `https://www.who.int/robots.txt` 示範爬蟲規範 | 未驗證（依 WHO Terms of use，API 說明頁未載明具體授權） | 撰寫時 `ghoapi.who.int` 在部分網路解析失敗、`azureedge.net` 網域可用；網域可能變動，程式有離線備用值 |
| data.gov.tw 政府資料開放平臺 | 僅在正文介紹，notebook 未載入任何資料集 | 政府資料開放授權條款－第 1 版（允許任何目的利用，需註明資料提供機關） | 實際使用的衛福部／健保署資料集尚未挑選與實測 |

## 使用這些資料的提醒

- 本站不放任何可識別病人的資料。
- 資料集都很小、來源單一、未經外部驗證；教材中出現的高準確率不代表模型可用於臨床。
- 各章 notebook 的資料載入方式以實測為準；網路下載可能失敗或變慢，notebook 內有錯誤訊息提示。
